"""``Analyzer``: live visual data for the UI (a ``SideProcessor``).

One instance observes the **input** chunks (``source="input"``); since step 06 a second one
observes the chain's **output** (``source="output"``), so the page can show what was removed.

Emits three kinds of events at configured rates that do not depend on the chunk size (an event
is due every ``sample_rate / rate`` frames, counted sample-accurately across chunk boundaries;
one chunk may produce several events of a kind, or none):

- ``LevelFrame`` (``levels_hz``): per channel RMS dBFS and peak dBFS over the interval, and a
  clipping flag (any ``|x| >= 0.999``). Computed with ``dsp/levels.py``.
- ``WaveformFrame`` (``waveform_hz``): ``waveform_points`` min/max pairs over the interval, of the
  mono mix. The mix comes from ``ChannelReducer("mean")``, the only allowed way to downmix.
- ``SpectrogramFrame`` (``spectrogram_hz``): Hann-windowed FFT of the most recent
  ``spectrogram_fft`` samples of the mono mix, power summed into ``spectrogram_bands`` mel bands
  from ``spectrogram_fmin_hz`` to Nyquist, in dB, floored at ``spectrogram_db_floor``. The FFT is
  scaled so a full-scale sine reads about 0 dB in its band.

All dB values are rounded to 0.1 dB (smaller messages; the UI cannot show finer steps anyway).
"""

from __future__ import annotations

import math

import numpy as np

from soundspy.config import AnalysisConfig
from soundspy.core.audio import AudioBuffer
from soundspy.core.events import Event, LevelFrame, SpectrogramFrame, WaveformFrame
from soundspy.dsp.channels import ChannelReducer
from soundspy.dsp.levels import clipping_ratio, peak_dbfs, rms_dbfs
from soundspy.dsp.mel import mel_filterbank
from soundspy.engine.processor import SideProcessor


def _round(values: np.ndarray) -> list[float]:
    return [round(float(v), 1) for v in values]


class _Schedule:
    """Sample-accurate emission times for one event rate."""

    def __init__(self, sample_rate: int, rate_hz: float) -> None:
        self.period = sample_rate / rate_hz
        self.next_due = self.period  # absolute frame index of the next emission

    def due_in(self, start: int, n: int) -> list[int]:
        """Offsets within the block ``[start, start + n)`` at which events are due."""
        offsets = []
        while self.next_due <= start + n:
            offsets.append(max(0, math.ceil(self.next_due) - start))
            self.next_due += self.period
        return offsets


class Analyzer(SideProcessor):
    def __init__(self, config: AnalysisConfig, source: str = "input") -> None:
        if source not in ("input", "output"):
            raise ValueError(f"source must be 'input' or 'output', got {source!r}")
        self.config = config
        self.source = source  # stamped on every event: which side of the chain it observes
        self._reducer = ChannelReducer("mean")

    def configure(self, sample_rate: int, num_channels: int) -> None:
        super().configure(sample_rate, num_channels)
        cfg = self.config
        nyquist = sample_rate / 2
        if cfg.spectrogram_fmin_hz >= nyquist:
            raise ValueError(f"spectrogram_fmin_hz must be below Nyquist ({nyquist} Hz)")
        self._reducer.configure(sample_rate, num_channels)
        self._levels = _Schedule(sample_rate, cfg.levels_hz)
        self._wave = _Schedule(sample_rate, cfg.waveform_hz)
        self._spec = _Schedule(sample_rate, cfg.spectrogram_hz)
        self._frame = 0
        self._level_parts: list[np.ndarray] = []
        self._wave_parts: list[np.ndarray] = []
        self._history = np.zeros(cfg.spectrogram_fft, dtype=np.float32)  # recent mono samples
        self._window = np.hanning(cfg.spectrogram_fft).astype(np.float64)
        self._fft_scale = 2.0 / self._window.sum()
        self._bank = mel_filterbank(
            sample_rate, cfg.spectrogram_fft, cfg.spectrogram_bands, cfg.spectrogram_fmin_hz
        )

    def reset(self) -> None:
        if self.sample_rate is not None and self.num_channels is not None:
            self.configure(self.sample_rate, self.num_channels)

    # --- per-kind computations -----------------------------------------------------------------

    def _level_event(self, t: float) -> LevelFrame:
        buf = AudioBuffer(np.concatenate(self._level_parts), self.sample_rate)
        self._level_parts = []
        return LevelFrame(
            t=t,
            rms_dbfs=_round(rms_dbfs(buf)),
            peak_dbfs=_round(peak_dbfs(buf)),
            clipping=[bool(c > 0) for c in clipping_ratio(buf)],
            source=self.source,
        )

    def _wave_event(self, t: float) -> WaveformFrame:
        mono = np.concatenate(self._wave_parts).ravel()  # already (n, 1) from ChannelReducer
        self._wave_parts = []
        bins = np.array_split(mono, min(self.config.waveform_points, max(1, mono.size)))
        mins = [round(float(b.min()), 4) if b.size else 0.0 for b in bins]
        maxs = [round(float(b.max()), 4) if b.size else 0.0 for b in bins]
        return WaveformFrame(t=t, mins=mins, maxs=maxs, source=self.source)

    def _spec_event(self, t: float) -> SpectrogramFrame:
        spectrum = np.fft.rfft(self._history.astype(np.float64) * self._window)
        power = (np.abs(spectrum) * self._fft_scale) ** 2
        bands = self._bank @ power
        floor = self.config.spectrogram_db_floor
        db = np.maximum(10.0 * np.log10(np.maximum(bands, 1e-20)), floor)
        return SpectrogramFrame(
            t=t,
            magnitudes_db=_round(db),
            freq_max_hz=self.sample_rate / 2,
            freq_min_hz=self.config.spectrogram_fmin_hz,
            source=self.source,
        )

    def _push_history(self, mono: np.ndarray) -> None:
        n = self._history.size
        if mono.size >= n:
            self._history = mono[-n:].copy()
        else:
            self._history = np.concatenate((self._history[mono.size :], mono))

    # --- SideProcessor -------------------------------------------------------------------------

    def observe(self, chunk: AudioBuffer) -> list[Event]:
        if self.sample_rate is None:
            raise RuntimeError("Analyzer.configure() must be called before observe()")
        n, start, sr = chunk.num_frames, self._frame, self.sample_rate
        mono = self._reducer.process(chunk).samples  # (n, 1), via the sanctioned downmix
        timed: list[tuple[int, int, Event]] = []  # (frame, order, event) for time ordering

        cut = 0
        for off in self._levels.due_in(start, n):
            self._level_parts.append(chunk.samples[cut:off])
            cut = off
            timed.append((start + off, 0, self._level_event((start + off) / sr)))
        self._level_parts.append(chunk.samples[cut:])

        cut = 0
        for off in self._wave.due_in(start, n):
            self._wave_parts.append(mono[cut:off])
            cut = off
            timed.append((start + off, 1, self._wave_event((start + off) / sr)))
        self._wave_parts.append(mono[cut:])

        cut = 0
        for off in self._spec.due_in(start, n):
            self._push_history(mono[cut:off, 0])
            cut = off
            timed.append((start + off, 2, self._spec_event((start + off) / sr)))
        self._push_history(mono[cut:, 0])

        self._frame += n
        timed.sort(key=lambda item: (item[0], item[1]))
        return [event for _, _, event in timed]

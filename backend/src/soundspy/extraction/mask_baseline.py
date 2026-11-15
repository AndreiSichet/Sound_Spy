"""Baseline extractor: a detector-gated Wiener filter (no training).

The signal goes through a streaming STFT (``dsp/stft.py``); every frame gets one gain per
frequency bin, built from three parts:

1. **Target activity.** An ``ActivitySource`` says whether any selected class is sounding now
   (live: the detector's shared state; offline: a timeline; evaluation: ground truth).
2. **Gate.** While the target is inactive, every bin gets the floor gain
   ``g_min = 10^(-attenuation_db / 20)``. The gate opens over ``open_ms``, stays open for
   ``hold_ms`` after the target stops, then closes over ``close_ms``. Its position ``o`` (0 =
   closed, 1 = open) moves linearly in dB, so there are no clicks and no chopped word endings.
3. **Spectral suppression while the gate is open.** A noise power spectrum ``N`` is estimated

   * by recursive averaging (``noise_alpha``) over the frames where the target is **inactive**
     (everything that is not the target is "noise" here), and
   * by **minimum statistics** as a fallback: the minimum of the smoothed power over the last
     ``minstat_window_s`` (times ``minstat_bias``). It follows whatever stays steady, without
     needing the detector, but for the same reason it also removes a *steady target* (a held
     note, a continuous alarm, road noise). It is therefore used only when no inactive frame was
     seen for ``noise_max_age_s``, or when there is no activity source at all.

   From ``N`` come the decision-directed a-priori SNR ``xi`` (Ephraim-Malah, ``dd_alpha``) and the
   Wiener gain ``xi / (1 + xi)``, floored at ``g_min`` and smoothed over time (``gain_smoothing``).

The gain of a frame is ``G^o * g_min^(1 - o)`` per bin: the Wiener gain when the gate is open, the
floor when it is closed, a smooth blend in between.

**A late detector must not teach the filter that the target is noise.** The live detector
switches on ~0.3 s after a sound starts. A frame is therefore used for the noise average only
after the target has stayed inactive for ``ActivitySource.lag_s`` *after* it.

**Empty target** (``TargetSpec.labels`` empty) = pass-through: unity gain, but still through the
STFT, so the latency stays the same and selecting a class does not make the sound jump. The
change is ramped over ``open_ms``.

**No activity source** (no detection model): no gating. The gate stays open and only the
minimum-statistics estimate is used, i.e. plain noise reduction.

Works on one channel (the chain reduces first). Latency: ``n_fft`` samples.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from soundspy.config import MaskBaselineConfig
from soundspy.core.audio import AudioBuffer
from soundspy.core.target import TargetSpec
from soundspy.dsp.stft import StftSpec, StreamingISTFT, StreamingSTFT
from soundspy.extraction.activity import ActivitySource
from soundspy.extraction.base import Extractor

_EPS = 1e-12
MINSTAT_SUBWINDOWS = 8


class MaskBaselineExtractor(Extractor):
    method = "mask_baseline"

    def __init__(self, config: MaskBaselineConfig, activity: ActivitySource | None = None) -> None:
        self.config = config
        self.activity = activity
        self.spec = StftSpec(config.n_fft, config.hop)
        self.target_active = False
        self.gate = 0.0

    # --- Processor -----------------------------------------------------------------------------

    @property
    def latency_samples(self) -> int:
        return self.spec.latency

    def configure(self, sample_rate: int, num_channels: int) -> int:
        if num_channels != 1:
            raise ValueError(
                f"MaskBaselineExtractor works on one channel, got {num_channels} "
                "(put a ChannelReducer or a spatial filter before it)")
        super().configure(sample_rate, num_channels)
        cfg, hop_s = self.config, self.spec.hop / sample_rate
        self._stft = StreamingSTFT(self.spec, 1)
        self._istft = StreamingISTFT(self.spec, 1)
        frames = lambda seconds: max(0, round(seconds / hop_s))  # noqa: E731
        self._open_step = 1.0 if cfg.open_ms <= 0 else min(1.0, 1000 * hop_s / cfg.open_ms)
        self._close_step = 1.0 if cfg.close_ms <= 0 else min(1.0, 1000 * hop_s / cfg.close_ms)
        self._hold_frames = frames(cfg.hold_ms / 1000)
        self._max_age = frames(cfg.noise_max_age_s)
        self._guard = frames(self.activity.lag_s) if self.activity is not None else 0
        self._sub_frames = max(1, math.ceil(frames(cfg.minstat_window_s) / MINSTAT_SUBWINDOWS))
        self.reset()
        return 1

    def reset(self) -> None:
        if self.sample_rate is None:
            return
        bins = self.spec.bins
        self._stft.reset()
        self._istft.reset()
        if self.activity is not None:
            self.activity.reset()
        self._frame = 0
        self._smoothed: np.ndarray | None = None
        self._mins = np.full((MINSTAT_SUBWINDOWS, bins), np.inf, dtype=np.float32)
        self._min_done = np.full(bins, np.inf, dtype=np.float32)
        self._sub_min = np.full(bins, np.inf, dtype=np.float32)
        self._sub_count = self._sub_index = 0
        self._clean_prev = np.zeros(bins, dtype=np.float32)
        self._gain = np.ones(bins, dtype=np.float32)
        self._mix = 1.0 if self._processing() else 0.0
        self.gate, self._hold = 0.0, 0
        self.target_active = False
        self._forget_noise()

    def _forget_noise(self) -> None:
        self._noise: np.ndarray | None = None
        self._noise_frame = -(10 ** 12)
        self._inactive_run = 0
        self._history: deque[np.ndarray] = deque(maxlen=getattr(self, "_guard", 0) + 1)

    def set_target(self, target: TargetSpec) -> None:
        previous = self.target
        super().set_target(target)
        if previous is None or previous.labels != target.labels:
            if self.sample_rate is not None:
                self._forget_noise()  # "everything else" means something else now

    def _processing(self) -> bool:
        return self.target is not None and bool(self.target.labels)

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        if self.sample_rate is None:
            raise RuntimeError("MaskBaselineExtractor.configure() must be called first")
        if chunk.num_channels != 1:
            raise ValueError(f"expected 1 channel, got {chunk.num_channels}")
        spectra = self._stft.push(chunk.samples)
        k, bins = spectra.shape[0], self.spec.bins
        if k:
            flat = spectra.reshape(k, bins)
            out = np.empty_like(flat)
            for i in range(k):
                gain = self._frame_gain(flat[i])
                out[i] = flat[i] if gain is None else flat[i] * gain
            self._istft.push(out.reshape(k, 1, bins))
        samples = self._istft.read(chunk.num_frames)
        return AudioBuffer(samples, chunk.sample_rate,
                           {**chunk.metadata, "extraction": self.method})

    def status_info(self) -> dict[str, Any]:
        info = super().status_info()
        info.update(gating=self.activity is not None, target_active=bool(self.target_active))
        return info

    # --- one frame -----------------------------------------------------------------------------

    def _frame_gain(self, spectrum: np.ndarray) -> np.ndarray | float | None:
        """Gain for one frame (per bin, or a scalar; ``None`` = unity)."""
        cfg, k = self.config, self._frame
        self._frame += 1
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32, copy=False)

        # minimum statistics (always running, so it is ready when it is needed)
        if self._smoothed is None:
            self._smoothed = power.copy()
        else:
            a = cfg.minstat_smoothing
            self._smoothed = a * self._smoothed + (1.0 - a) * power
        np.minimum(self._sub_min, self._smoothed, out=self._sub_min)
        self._sub_count += 1
        if self._sub_count >= self._sub_frames:
            self._mins[self._sub_index] = self._sub_min
            self._sub_index = (self._sub_index + 1) % MINSTAT_SUBWINDOWS
            self._min_done = self._mins.min(axis=0)
            self._sub_min = self._smoothed.copy()
            self._sub_count = 0

        processing = self._processing()
        target = 1.0 if processing else 0.0
        if self._mix != target:  # ramp into / out of pass-through
            step = self._open_step if target > self._mix else -self._open_step
            self._mix = min(1.0, max(0.0, self._mix + step))
        if not processing:
            self.target_active = False
            if self._mix <= 0.0:
                return None

        # target activity
        gated = self.activity is not None and processing
        if gated:
            sr = self.sample_rate
            active = self.activity.active(self.target.labels, self.spec.frame_centre(k) / sr,
                                          (k + 1) * self.spec.hop / sr)
        else:
            active = True
        self.target_active = bool(active) and processing

        # detector-gated noise average (with the lag guard)
        if gated:
            self._inactive_run = 0 if active else self._inactive_run + 1
            if self._guard:
                self._history.append(power)
            if not active and self._inactive_run > self._guard:
                confirmed = self._history[0] if self._guard else power
                if self._noise is None:
                    self._noise = confirmed.astype(np.float32, copy=True)
                else:
                    a = cfg.noise_alpha
                    self._noise = a * self._noise + (1.0 - a) * confirmed
                self._noise_frame = k

        # gate position
        if active:
            self._hold = self._hold_frames
            self.gate = min(1.0, self.gate + self._open_step)
        elif self._hold > 0:
            self._hold -= 1
        else:
            self.gate = max(0.0, self.gate - self._close_step)

        attenuation_db = self.target.attenuation_db if self.target is not None else 0.0
        log_floor = -attenuation_db * math.log(10.0) / 20.0
        g_min = math.exp(log_floor)

        # Wiener gain (kept up to date also while the gate is closed)
        if self._noise is not None and k - self._noise_frame <= self._max_age:
            noise = self._noise  # what "not the target" sounded like, recently
        else:  # fallback: whatever has been steady lately (also a steady target, see docstring)
            noise = cfg.minstat_bias * np.minimum(self._min_done, self._sub_min)
        noise = np.maximum(noise, _EPS)
        xi = (cfg.dd_alpha * self._clean_prev + (1.0 - cfg.dd_alpha)
              * np.maximum(power - noise, 0.0)) / noise
        wiener = np.maximum(xi / (1.0 + xi), g_min)
        self._clean_prev = wiener * wiener * power
        b = cfg.gain_smoothing
        self._gain = b * self._gain + (1.0 - b) * wiener

        if self.gate <= 0.0:
            return math.exp(self._mix * log_floor)
        log_gain = self.gate * np.log(np.maximum(self._gain, g_min)) + (1.0 - self.gate) * log_floor
        return np.exp(self._mix * log_gain).astype(np.float32, copy=False)

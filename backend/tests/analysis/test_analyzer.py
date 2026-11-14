"""Analyzer: rates, spectrogram content, levels, waveform; and the mel filterbank."""

from __future__ import annotations

import numpy as np
import pytest

from soundspy.analysis.analyzer import Analyzer
from soundspy.core.audio import AudioBuffer
from soundspy.core.events import LevelFrame, SpectrogramFrame, WaveformFrame
from soundspy.dsp.levels import peak_dbfs, rms_dbfs
from soundspy.dsp.mel import hz_to_mel, mel_band_edges_hz, mel_filterbank, mel_to_hz

SR = 48000


def _run(analyzer: Analyzer, samples: np.ndarray, chunk: int) -> list:
    events = []
    for i in range(0, samples.shape[0], chunk):
        events += analyzer.observe(AudioBuffer(samples[i : i + chunk], SR))
    return events


def _analyzer(config, channels: int = 2) -> Analyzer:
    analyzer = Analyzer(config.analysis)
    analyzer.configure(SR, channels)
    return analyzer


def _sine(freq: float, seconds: float, amplitude: float = 0.5, channels: int = 2) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.repeat(tone[:, None], channels, axis=1)


@pytest.mark.parametrize("chunk", [37, 480, 960, 4800, 48000])
def test_rates_do_not_depend_on_chunk_size(config, chunk) -> None:
    events = _run(_analyzer(config), _sine(440, 1.0), chunk)
    kinds = [type(e) for e in events]
    assert kinds.count(LevelFrame) == 20  # levels_hz
    assert kinds.count(WaveformFrame) == 25  # waveform_hz
    assert kinds.count(SpectrogramFrame) == 25  # spectrogram_hz
    times = [e.t for e in events]
    assert times == sorted(times) and times[-1] == pytest.approx(1.0)
    levels_t = [e.t for e in events if isinstance(e, LevelFrame)]
    np.testing.assert_allclose(levels_t, np.arange(1, 21) / 20)


def test_sine_peaks_in_the_right_mel_band(config) -> None:
    events = _run(_analyzer(config), _sine(1000, 0.5), 960)
    frame = [e for e in events if isinstance(e, SpectrogramFrame)][-1]
    cfg = config.analysis
    centres = mel_band_edges_hz(cfg.spectrogram_bands, cfg.spectrogram_fmin_hz, SR / 2)[1:-1]
    peak_band = int(np.argmax(frame.magnitudes_db))
    assert abs(peak_band - int(np.argmin(np.abs(centres - 1000)))) <= 1
    assert -12 < max(frame.magnitudes_db) < 3  # a -6 dBFS sine, Hann window spread
    assert frame.freq_min_hz == cfg.spectrogram_fmin_hz and frame.freq_max_hz == SR / 2
    assert len(frame.magnitudes_db) == cfg.spectrogram_bands


def test_silence_is_the_floor(config) -> None:
    events = _run(_analyzer(config), np.zeros((SR // 2, 2), dtype=np.float32), 960)
    floor = config.analysis.spectrogram_db_floor
    for e in events:
        if isinstance(e, SpectrogramFrame):
            assert set(e.magnitudes_db) == {floor}
        if isinstance(e, LevelFrame):
            assert e.rms_dbfs == [-120.0, -120.0] and e.clipping == [False, False]


def test_levels_match_dsp_levels(config) -> None:
    rng = np.random.default_rng(0)
    samples = (0.1 * rng.standard_normal((2400, 3))).astype(np.float32)
    samples[100, 2] = 1.0  # clipping on channel 2 only
    events = _run(_analyzer(config, channels=3), samples, 2400)
    level = next(e for e in events if isinstance(e, LevelFrame))
    buf = AudioBuffer(samples, SR)
    np.testing.assert_allclose(level.rms_dbfs, np.round(rms_dbfs(buf), 1))
    np.testing.assert_allclose(level.peak_dbfs, np.round(peak_dbfs(buf), 1))
    assert level.clipping == [False, False, True]


def test_waveform_min_max_of_the_mono_mix(config) -> None:
    n = 1920  # one waveform interval at 25 Hz
    ramp = np.linspace(-1, 1, n, dtype=np.float32)
    samples = np.stack([ramp, -ramp], axis=1)  # mean of the two channels is 0
    wave = next(e for e in _run(_analyzer(config), samples, n) if isinstance(e, WaveformFrame))
    assert len(wave.mins) == len(wave.maxs) == config.analysis.waveform_points
    assert max(map(abs, wave.mins + wave.maxs)) < 1e-6
    wave = next(
        e for e in _run(_analyzer(config), np.stack([ramp, ramp], 1), n)
        if isinstance(e, WaveformFrame)
    )
    bins = np.array_split(ramp, config.analysis.waveform_points)
    np.testing.assert_allclose(wave.mins, [round(float(b.min()), 4) for b in bins], atol=1e-4)
    np.testing.assert_allclose(wave.maxs, [round(float(b.max()), 4) for b in bins], atol=1e-4)


def test_values_are_rounded_to_a_tenth_of_a_db(config) -> None:
    events = _run(_analyzer(config), _sine(300, 0.2, amplitude=0.3), 960)
    for e in events:
        values = e.magnitudes_db if isinstance(e, SpectrogramFrame) else (
            e.rms_dbfs + e.peak_dbfs if isinstance(e, LevelFrame) else []
        )
        assert all(round(v, 1) == v for v in values)


def test_observe_requires_configure(config) -> None:
    with pytest.raises(RuntimeError):
        Analyzer(config.analysis).observe(AudioBuffer(np.zeros((10, 1), dtype=np.float32), SR))


# --- mel filterbank ----------------------------------------------------------------------------


def test_mel_round_trip_and_known_value() -> None:
    assert float(hz_to_mel(1000)) == pytest.approx(1000, abs=0.5)  # HTK: 1000 Hz ~ 1000 mel
    np.testing.assert_allclose(mel_to_hz(hz_to_mel([50, 440, 8000])), [50, 440, 8000])


def test_filterbank_shape_and_coverage() -> None:
    bank = mel_filterbank(48000, 2048, 128, 50)
    assert bank.shape == (128, 1025)
    assert np.all(bank.max(axis=1) > 0)  # no empty band, even the narrow low ones
    assert bank.min() >= 0 and bank.max() <= 1
    with pytest.raises(ValueError):
        mel_filterbank(16000, 512, 40, 9000)  # fmin above Nyquist

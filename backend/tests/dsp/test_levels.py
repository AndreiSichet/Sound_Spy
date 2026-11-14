"""Level measurements and channel comparison on signals with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from soundspy.core.audio import AudioBuffer
from soundspy.dsp.levels import (
    DB_FLOOR,
    channel_similarity,
    clipping_ratio,
    compare_to_reference,
    digital_silence_ratio,
    noise_floor_dbfs,
    peak_dbfs,
    rms_dbfs,
)

SR = 48000


def _sine(amplitude: float = 1.0, seconds: float = 1.0, freq: float = 1000.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _buf(*channels: np.ndarray) -> AudioBuffer:
    return AudioBuffer(np.stack(channels, axis=1).astype(np.float32), SR)


def test_full_scale_sine() -> None:
    buf = _buf(_sine(1.0), _sine(0.5))
    np.testing.assert_allclose(rms_dbfs(buf), [-3.0103, -9.0309], atol=0.01)
    np.testing.assert_allclose(peak_dbfs(buf), [0.0, -6.0206], atol=0.01)


def test_silence_is_the_floor_not_minus_infinity() -> None:
    buf = _buf(np.zeros(100, dtype=np.float32))
    assert rms_dbfs(buf)[0] == DB_FLOOR and peak_dbfs(buf)[0] == DB_FLOOR


def test_clipping_ratio() -> None:
    clipped = np.clip(_sine(2.0), -1.0, 1.0)
    buf = _buf(clipped, _sine(0.5))
    ratio = clipping_ratio(buf)
    assert 0.5 < ratio[0] < 0.8 and ratio[1] == 0.0


def test_digital_silence_ratio_counts_only_stretches() -> None:
    signal = _sine(0.5, seconds=0.1)
    signal[:1200] = 0.0  # 25 ms of silence: a quarter of the samples
    ratio = digital_silence_ratio(_buf(signal, _sine(0.5, seconds=0.1)))
    assert ratio[0] == pytest.approx(0.25) and ratio[1] == 0.0


def test_isolated_zeros_from_a_16_bit_converter_are_not_silence() -> None:
    # Like the dev laptop: quiet noise on a 16-bit grid hits the "0" code ~1-2 % of the time.
    rng = np.random.default_rng(5)
    codes = np.round(rng.normal(0.0, 20.0, SR)).astype(np.float32)
    signal = codes * np.float32(2.3542 / 32768)
    assert 0.01 < np.mean(signal == 0) < 0.03
    assert digital_silence_ratio(_buf(signal))[0] == 0.0


def test_noise_floor_finds_the_quiet_part() -> None:
    rng = np.random.default_rng(0)
    quiet = 0.001 * rng.standard_normal(SR)  # about -60 dBFS RMS
    loud = 0.3 * rng.standard_normal(SR // 4)
    signal = np.concatenate([quiet, loud]).astype(np.float32)
    floor = noise_floor_dbfs(_buf(signal))[0]
    assert -63 < floor < -57
    assert rms_dbfs(_buf(signal))[0] > -20  # the overall level is dominated by the loud part


def test_channel_similarity_detects_a_delay() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(SR // 2)
    b = np.concatenate([np.zeros(7), a[:-7]])  # b hears it 7 samples later
    sim = channel_similarity(a, b, SR)
    assert sim.best_lag_samples == 7
    assert sim.best_lag_s == pytest.approx(7 / SR)
    assert sim.peak_corr > 0.99 and abs(sim.corr_zero_lag) < 0.1
    assert not sim.identical
    assert channel_similarity(b, a, SR).best_lag_samples == -7  # sign flips with the order


def test_channel_similarity_identical_channels() -> None:
    a = np.random.default_rng(2).standard_normal(4800)
    sim = channel_similarity(a, a.copy(), SR)
    assert sim.identical and sim.best_lag_samples == 0
    assert sim.corr_zero_lag == pytest.approx(1.0)


def test_channel_similarity_respects_max_lag() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal(SR // 2)
    b = np.concatenate([np.zeros(100), a[:-100]])  # 100 samples > 1 ms at 48 kHz
    assert abs(channel_similarity(a, b, SR, max_lag_s=0.001).best_lag_samples) <= 48
    assert channel_similarity(a, b, SR, max_lag_s=0.003).best_lag_samples == 100


def test_silent_channel_has_zero_correlation() -> None:
    sim = channel_similarity(np.zeros(100), np.ones(100), SR)
    assert sim.corr_zero_lag == 0.0 and not sim.identical


def test_compare_to_reference() -> None:
    a = np.random.default_rng(4).standard_normal(4800).astype(np.float32)
    buf = _buf(a, a, np.roll(a, 3))
    result = compare_to_reference(buf)
    assert set(result) == {1, 2}
    assert result[1].identical and result[2].best_lag_samples == 3

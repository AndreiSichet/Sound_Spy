"""Polyphase resampling: exact durations, frequency preserved, all channels kept."""

from __future__ import annotations

import numpy as np
import pytest

from soundspy.core.audio import AudioBuffer
from soundspy.dsp.resample import ratio, resample, resample_array


def tone(seconds: float, sr: int, freq: float, channels: int = 1) -> AudioBuffer:
    t = np.arange(round(seconds * sr)) / sr
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    return AudioBuffer(np.repeat(x.reshape(-1, 1), channels, axis=1).astype(np.float32), sr)


def test_ratio_in_lowest_terms() -> None:
    assert ratio(44_100, 16_000) == (160, 441)
    assert ratio(44_100, 32_000) == (320, 441)
    assert ratio(48_000, 16_000) == (1, 3)
    with pytest.raises(ValueError):
        ratio(0, 16_000)


@pytest.mark.parametrize(("target", "frames"), [(16_000, 96_000), (32_000, 192_000)])
def test_44k1_keeps_durations_exact(target: int, frames: int) -> None:
    out = resample(tone(6.0, 44_100, 440.0), target)
    assert out.sample_rate == target
    assert out.num_frames == frames
    assert out.duration_s == 6.0


def test_frequency_is_preserved() -> None:
    out = resample(tone(1.0, 44_100, 1000.0), 16_000)
    spectrum = np.abs(np.fft.rfft(out.samples.ravel()))
    peak_hz = np.argmax(spectrum) * 16_000 / out.num_frames
    assert peak_hz == pytest.approx(1000.0, abs=2.0)


def test_content_above_the_new_nyquist_is_removed() -> None:
    out = resample(tone(1.0, 44_100, 12_000.0), 16_000)  # 12 kHz > 8 kHz Nyquist
    assert np.max(np.abs(out.samples[1000:-1000])) < 0.01


def test_all_channels_are_kept_and_same_rate_is_a_no_op() -> None:
    stereo = tone(0.5, 48_000, 300.0, channels=2)
    out = resample(stereo, 16_000)
    assert out.samples.shape == (8_000, 2)
    assert resample(stereo, 48_000) is stereo
    assert resample_array(np.zeros((0, 2), np.float32), 48_000, 16_000).shape == (0, 2)

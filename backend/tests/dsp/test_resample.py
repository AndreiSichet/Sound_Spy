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


# --- step 07: StreamingResampler (stateful, chunk by chunk) ------------------------------------


def _noise(n: int, channels: int = 1, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, channels)).astype(np.float32) * 0.3


@pytest.mark.parametrize(("source", "target"), [(48000, 44100), (44100, 48000), (48000, 16000),
                                                (16000, 48000), (44100, 16000)])
def test_streaming_resampler_chunked_equals_one_shot(source: int, target: int) -> None:
    from soundspy.dsp.resample import StreamingResampler

    x = _noise(source // 4, channels=2)
    whole = StreamingResampler(source, target, channels=2).push(x)
    chunked = StreamingResampler(source, target, channels=2)
    pieces, start = [], 0
    sizes = [1, 7, 960, 3, 441, 0, 960, 13, 2048, 5]
    i = 0
    while start < len(x):
        size = sizes[i % len(sizes)]
        pieces.append(chunked.push(x[start:start + size]))
        start, i = start + size, i + 1
    out = np.concatenate(pieces)
    assert out.shape == whole.shape == (chunked.output_count(len(x)), 2)
    np.testing.assert_allclose(out, whole, atol=1e-5)  # no clicks at chunk borders
    assert whole.dtype == np.float32


def test_streaming_resampler_matches_scipy_after_its_delay() -> None:
    """The streaming output is scipy's zero-phase polyphase result, delayed by ``delay``."""
    from scipy.signal import firwin, resample_poly

    from soundspy.dsp.resample import KAISER_BETA, StreamingResampler

    r = StreamingResampler(48000, 16000)
    assert (r.up, r.down) == (1, 3) and float(r.delay).is_integer()
    x = _noise(9600)
    out = r.push(x).ravel()
    h = firwin(r.taps, 1.0 / 3, window=("kaiser", KAISER_BETA))
    reference = resample_poly(x.ravel().astype(np.float64), 1, 3, window=h)
    d = int(r.delay)
    np.testing.assert_allclose(out[d:], reference[: len(out) - d], atol=1e-5)
    assert float(np.max(np.abs(out[: d // 2]))) < 0.01  # only filter ripple before the signal


def test_streaming_resampler_reset_and_shapes() -> None:
    from soundspy.dsp.resample import StreamingResampler

    r = StreamingResampler(48000, 44100)
    x = _noise(2000)
    first = r.push(x)
    r.push(_noise(500, seed=1))
    r.reset()
    np.testing.assert_array_equal(r.push(x), first)  # same state as a new one
    assert r.push(np.zeros((0, 1), dtype=np.float32)).shape == (0, 1)
    with pytest.raises(ValueError):
        r.push(np.zeros((10, 2), dtype=np.float32))  # channel count is fixed
    with pytest.raises(ValueError):
        StreamingResampler(48000, 44100, taps=100)  # even length: no whole-sample centre
    same = StreamingResampler(48000, 48000, channels=3)  # same rate: identity, no delay
    y = _noise(100, channels=3)
    np.testing.assert_array_equal(same.push(y), y)
    assert same.delay == 0.0


def test_tone_survives_the_round_trip_with_a_whole_sample_delay() -> None:
    """48 -> 44.1 -> 48 kHz as the neural extractor does it: the second filter's length makes
    the total delay a whole number of samples, so the result lines up with the input."""
    from soundspy.dsp.resample import StreamingResampler, taps_for_whole_delay

    down = StreamingResampler(48000, 44100)
    up = StreamingResampler(44100, 48000, taps=taps_for_whole_delay(down, 48000, 44100))
    total = down.delay * 48000 / 44100 + up.delay
    assert abs(total - round(total)) < 1e-9 and round(total) == 22
    t = np.arange(48000) / 48000
    x = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32).reshape(-1, 1)
    y = up.push(down.push(x))
    assert len(y) == len(x)
    delay = round(total)
    error = y[delay + 2000:].ravel() - x[2000: len(x) - delay].ravel()
    assert float(np.max(np.abs(error))) < 2e-3

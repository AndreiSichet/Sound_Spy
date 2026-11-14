"""SpeakerSink against a fake output stream driven one callback at a time (deterministic)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from soundspy.core.audio import AudioBuffer
from soundspy.io.devices import resolve_output
from soundspy.io.speaker_sink import PeakLimiter, SpeakerSink, db_to_gain

SR = 48000
CHUNK = 960  # 20 ms


@pytest.fixture
def manual_output(fake_sounddevice):
    fake_sounddevice.output_autorun = False
    return fake_sounddevice


def _sink(config, channels: int = 1, **overrides) -> tuple[SpeakerSink, object]:
    output = dataclasses.replace(config.output, **overrides)
    sink = SpeakerSink(resolve_output(), output)
    sink.open(SR, channels)
    import sys

    stream = sys.modules["sounddevice"].output_streams()[-1]
    return sink, stream


def _chunk(value: float, channels: int = 1, n: int = CHUNK) -> AudioBuffer:
    return AudioBuffer(np.full((n, channels), value, dtype=np.float32), SR)


def test_prebuffer_then_play(config, manual_output) -> None:
    sink, stream = _sink(config, prebuffer_chunks=2)
    sink.write(_chunk(0.1))
    assert not stream.pull(CHUNK).any()  # still prebuffering: silence, no underrun
    assert sink.underruns == 0
    sink.write(_chunk(0.2))
    np.testing.assert_allclose(stream.pull(CHUNK), 0.1)
    np.testing.assert_allclose(stream.pull(CHUNK), 0.2)
    assert sink.frames_played == 2 * CHUNK
    sink.close()


def test_underrun_gives_silence_and_counts(config, manual_output) -> None:
    sink, stream = _sink(config, prebuffer_chunks=1)
    sink.write(_chunk(0.3))
    out = stream.pull(CHUNK + 100)  # asks for more than is queued
    np.testing.assert_allclose(out[:CHUNK], 0.3)
    assert not out[CHUNK:].any()
    assert sink.underruns == 1
    assert not stream.pull(CHUNK).any() and sink.underruns == 1  # waits for the prebuffer again
    sink.close()


def test_drop_oldest_when_the_queue_is_full(config, manual_output) -> None:
    sink, stream = _sink(config, prebuffer_chunks=2, max_queue_chunks=3)
    for i in range(5):
        sink.write(_chunk(0.01 * (i + 1)))
    assert sink.dropped_chunks == 2
    np.testing.assert_allclose(stream.pull(CHUNK), 0.03)  # chunks 1 and 2 were dropped
    sink.close()


def test_mono_is_upmixed_and_gain_applied(config, manual_output) -> None:
    sink, stream = _sink(config, prebuffer_chunks=1)
    assert stream.channels == 2
    sink.set_gain(6.0)
    sink.write(_chunk(0.1))
    out = stream.pull(CHUNK)
    np.testing.assert_allclose(out[:, 0], out[:, 1])
    np.testing.assert_allclose(out[:, 0], 0.1 * db_to_gain(6.0), rtol=1e-5)
    sink.close()


def test_mute_plays_silence_but_keeps_running(config, manual_output) -> None:
    sink, stream = _sink(config, prebuffer_chunks=1)
    sink.set_gain(0.0, muted=True)
    sink.write(_chunk(0.5))
    assert not stream.pull(CHUNK).any() and sink.underruns == 0
    sink.close()


def test_limiter_keeps_hot_input_below_the_ceiling(config, manual_output) -> None:
    sink, stream = _sink(config, channels=2, prebuffer_chunks=1)
    sink.set_gain(12.0)  # +7 dBFS input plus the maximum gain
    t = np.arange(CHUNK * 10) / SR
    hot = (db_to_gain(7.0) * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    for i in range(10):
        part = hot[i * CHUNK : (i + 1) * CHUNK]
        sink.write(AudioBuffer(np.stack([part, part], 1), SR))
    played = np.concatenate([stream.pull(CHUNK) for _ in range(10)])
    ceiling = db_to_gain(config.output.limiter_ceiling_dbfs)
    assert np.abs(played).max() <= ceiling + 1e-6
    assert np.abs(played).max() > 0.5 * ceiling  # limited, not silenced
    sink.close()


def test_limiter_releases_after_a_peak() -> None:
    limiter = PeakLimiter(SR, -1.0, 50)
    loud = np.full((10, 1), 4.0, dtype=np.float32)
    limiter.process(loud)
    assert limiter.env < 0.3
    quiet = np.full((SR // 2, 1), 0.1, dtype=np.float32)
    out = limiter.process(quiet)
    assert out[-1, 0] == pytest.approx(0.1, rel=1e-3)  # fully recovered after 0.5 s
    assert out[0, 0] < 0.05  # but still reduced right after the peak


def test_too_many_channels_is_an_error(config, manual_output) -> None:
    sink = SpeakerSink(resolve_output(), config.output)
    with pytest.raises(ValueError, match="reduce channels"):
        sink.open(SR, 3)


def test_latency_counts_the_queue(config, manual_output) -> None:
    sink, stream = _sink(config, prebuffer_chunks=2)
    sink.write(_chunk(0.1))
    sink.write(_chunk(0.1))
    assert sink.latency_s == pytest.approx(stream.latency + 2 * CHUNK / SR)
    sink.flush()
    assert sink.queued_frames == 0
    sink.close()


def test_write_requires_open(config) -> None:
    with pytest.raises(RuntimeError):
        SpeakerSink(resolve_output(), config.output).write(_chunk(0.1))

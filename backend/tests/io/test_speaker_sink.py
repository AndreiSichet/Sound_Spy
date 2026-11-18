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


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _clocked_sink(config, **overrides) -> tuple[SpeakerSink, object, _Clock]:
    import sys

    clock = _Clock()
    output = dataclasses.replace(config.output, **overrides)
    sink = SpeakerSink(resolve_output(), output, clock=clock)
    sink.open(SR, 1)
    return sink, sys.modules["sounddevice"].output_streams()[-1], clock


def test_latency_returns_to_normal_after_a_burst(config, manual_output) -> None:
    """A stall then a burst fills the queue; one second later it is back at its cushion."""
    sink, stream, clock = _clocked_sink(config, prebuffer_chunks=2, max_queue_chunks=8,
                                        drain_after_s=1.0, drain_fade_ms=5.0)

    def tick(value: float) -> None:  # 20 ms of steady state: one chunk in, one chunk out
        sink.write(_chunk(value))
        stream.pull(CHUNK)
        clock.now += CHUNK / SR

    sink.write(_chunk(0.1))
    sink.write(_chunk(0.1))
    for _ in range(10):
        tick(0.1)
    normal = sink.latency_s
    assert sink.queued_frames == 2 * CHUNK

    for _ in range(5):  # the burst: five chunks at once
        sink.write(_chunk(0.1))
    stuck = sink.latency_s
    assert stuck == pytest.approx(normal + 5 * CHUNK / SR)

    for _ in range(40):  # 0.8 s later nothing has changed: the queue never empties by itself
        tick(0.1)
    assert sink.latency_s == pytest.approx(stuck) and sink.drained == 0
    for _ in range(15):  # past drain_after_s
        tick(0.1)
    assert sink.drained == 1 and sink.drained_chunks == 6
    assert normal - CHUNK / SR - 1e-9 <= sink.latency_s <= normal + 1e-9
    for _ in range(100):  # and it does not drain again while things are normal
        tick(0.1)
    assert sink.drained == 1 and sink.underruns == 0 and sink.dropped_chunks == 0
    sink.close()


def test_drain_crossfades_instead_of_clicking(config, manual_output) -> None:
    sink, stream, clock = _clocked_sink(config, prebuffer_chunks=2, max_queue_chunks=8,
                                        drain_after_s=0.5, drain_fade_ms=5.0)
    for _ in range(5):
        sink.write(_chunk(0.4))  # old audio
    sink.write(_chunk(-0.4))  # new audio, as different as possible
    sink.write(_chunk(-0.4))
    clock.now += 1.0
    sink.write(_chunk(-0.4))
    assert sink.drained == 1 and sink.queued_frames == 2 * CHUNK
    out = stream.pull(2 * CHUNK).T[0]
    fade = round(0.005 * SR)
    assert out[0] == pytest.approx(0.4)  # starts where the old audio would have continued
    assert np.abs(np.diff(out)).max() <= 0.8 / (fade - 1) + 1e-6  # no jump, a ramp
    np.testing.assert_allclose(out[fade:], -0.4)
    sink.close()


def test_a_short_excursion_does_not_drain(config, manual_output) -> None:
    sink, stream, clock = _clocked_sink(config, prebuffer_chunks=2, max_queue_chunks=8,
                                        drain_after_s=1.0)
    for _ in range(5):
        sink.write(_chunk(0.1))
    clock.now += 0.6
    stream.pull(3 * CHUNK)  # the output caught up by itself
    sink.write(_chunk(0.1))  # back at 3 queued: at the cushion, the timer restarts
    clock.now += 0.6
    for _ in range(2):
        sink.write(_chunk(0.1))  # above again, but not for a whole second
    assert sink.drained == 0
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

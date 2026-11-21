"""MicSource against the fake sounddevice stream (a thread calling the callback).

Fake device signals (see conftest): ``delayed_noise`` (channel c = channel 0 delayed by 3c
samples), ``channel_constant`` (device channel c holds c / 10), ``frame_ramp`` (each sample holds
its absolute frame number).
"""

from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np
import pytest

from soundspy.config import CaptureConfig, DeviceProfile
from soundspy.io.devices import resolve_input
from soundspy.io.mic_source import MicSource, MicStallError

LAPTOP = DeviceProfile(name="laptop")
RESPEAKER = DeviceProfile(name="respeaker", device="reSpeaker", raw_channels=(2, 3, 4, 5))
CAPTURE = CaptureConfig(stall_timeout_s=2.0, queue_chunks=32, startup_silence_max_s=1.0)


def capture(**overrides) -> CaptureConfig:
    return dataclasses.replace(CAPTURE, **overrides)


def test_read_returns_exact_length_dtype_and_metadata() -> None:
    with MicSource(resolve_input(LAPTOP), CAPTURE) as mic:
        buf = mic.read(0.25)
    assert buf.samples.shape == (12000, 2)  # round(0.25 * 48000)
    assert buf.samples.dtype == np.float32
    assert buf.sample_rate == 48000
    assert buf.metadata["device"].startswith("Microphone Array (Intel")
    assert buf.metadata["hostapi"] == "Windows WASAPI"
    assert buf.metadata["profile"] == "laptop"
    assert buf.metadata["channel_indices"] == [0, 1]
    assert buf.metadata["overflows"] == 0
    assert np.abs(buf.samples).max() > 0
    assert mic.stats.frames_delivered == 12000


def test_wasapi_stream_is_opened_with_auto_convert(fake_sounddevice) -> None:
    resolved = resolve_input(DeviceProfile(name="p", sample_rate=16000))  # not the native 48 kHz
    with MicSource(resolved, CAPTURE) as mic:
        assert mic.read(0.05).sample_rate == 16000
    stream = fake_sounddevice.streams[-1]
    assert stream.extra_settings.auto_convert is True
    assert (stream.device, stream.channels, stream.samplerate) == (5, 2, 16000)


def test_respeaker_keeps_channels_2_to_5_in_order(fake_sounddevice) -> None:
    fake_sounddevice.use_signal("channel_constant")
    with MicSource(resolve_input(RESPEAKER), CAPTURE) as mic:
        buf = mic.read(0.05)
    assert fake_sounddevice.streams[-1].channels == 6  # opened max(raw_channels) + 1
    assert buf.num_channels == 4
    np.testing.assert_allclose(buf.samples[0], [0.2, 0.3, 0.4, 0.5])
    assert buf.metadata["channel_indices"] == [2, 3, 4, 5]


def test_chunks_are_exact_and_contiguous(fake_sounddevice) -> None:
    fake_sounddevice.use_signal("frame_ramp")
    received = []
    with MicSource(resolve_input(LAPTOP), CAPTURE) as mic:
        for chunk in mic.chunks(1000):
            received.append(chunk)
            if len(received) == 5:
                break
    assert all(c.samples.shape == (1000, 2) for c in received)
    starts = [int(c.samples[0, 0]) for c in received]
    assert np.diff(starts).tolist() == [1000] * 4  # nothing lost with a fast consumer
    assert [c.metadata["chunk_index"] for c in received] == [0, 1, 2, 3, 4]


def test_breaking_out_of_chunks_stops_the_stream(fake_sounddevice) -> None:
    mic = MicSource(resolve_input(LAPTOP), CAPTURE)
    mic.open()
    for _ in mic.chunks(512):
        break
    stream = fake_sounddevice.streams[-1]
    assert not mic.is_open and stream.closed and not stream.active


def test_close_from_another_thread_ends_chunks(fake_sounddevice) -> None:
    mic = MicSource(resolve_input(LAPTOP), CAPTURE)
    mic.open()
    timer = threading.Timer(0.2, mic.close)
    timer.start()
    count = sum(1 for _ in mic.chunks(256))
    timer.join()
    assert count > 0
    assert not fake_sounddevice.streams[-1].active


def test_slow_consumer_drops_oldest_chunks(fake_sounddevice) -> None:
    fake_sounddevice.use_signal("frame_ramp")
    fake_sounddevice.speed = 200.0
    starts = []
    with MicSource(resolve_input(LAPTOP), capture(queue_chunks=2)) as mic:
        for chunk in mic.chunks(256):
            starts.append(int(chunk.samples[0, 0]))
            time.sleep(0.05)  # far slower than the stream
            if len(starts) == 4:
                break
        dropped = mic.stats.dropped_chunks
    assert dropped > 0
    # After a stall the consumer gets recent audio, not the backlog: start frames jump ahead.
    assert max(np.diff(starts)) > 256


def test_stall_is_detected(fake_sounddevice) -> None:
    fake_sounddevice.stall_after_frames = 0
    with MicSource(resolve_input(LAPTOP), capture(stall_timeout_s=0.3)) as mic:
        started = time.monotonic()
        with pytest.raises(MicStallError, match="unplugged, is another app using it"):
            mic.read(1.0)
        assert time.monotonic() - started < 2.0


def test_stall_is_detected_while_chunking(fake_sounddevice) -> None:
    fake_sounddevice.stall_after_frames = 1024
    with MicSource(resolve_input(LAPTOP), capture(stall_timeout_s=0.3)) as mic:
        with pytest.raises(MicStallError):
            for _ in mic.chunks(256):
                pass


def test_overflows_are_counted(fake_sounddevice) -> None:
    fake_sounddevice.overflow_every = 2
    with MicSource(resolve_input(LAPTOP), CAPTURE) as mic:
        buf = mic.read(0.1)
    assert buf.metadata["overflows"] > 0 and mic.stats.overflows >= buf.metadata["overflows"]


def test_startup_silence_is_discarded(fake_sounddevice) -> None:
    # The development laptop's WASAPI stream starts with exactly 500 ms of digital zeros.
    fake_sounddevice.leading_zero_frames = 24000
    with MicSource(resolve_input(LAPTOP), CAPTURE) as mic:
        buf = mic.read(0.2)
    assert mic.stats.warmup_frames_discarded == 24000
    assert np.all(buf.samples[0] != 0)  # starts with real audio


def test_muted_mic_still_delivers_silence_after_the_warmup_limit(fake_sounddevice) -> None:
    fake_sounddevice.leading_zero_frames = 10**9  # all zeros, like a muted microphone
    with MicSource(resolve_input(LAPTOP), capture(startup_silence_max_s=0.1)) as mic:
        buf = mic.read(0.2)
    assert mic.stats.warmup_frames_discarded == 4800  # stops discarding at 0.1 s
    assert not buf.samples.any()


def test_read_requires_open_stream() -> None:
    with pytest.raises(RuntimeError, match="open stream"):
        MicSource(resolve_input(LAPTOP), CAPTURE).read(0.1)

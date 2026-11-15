"""Session: worker thread, events, target changes, recording, errors, timing, monitoring."""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundspy.core.audio import AudioBuffer
from soundspy.core.events import (
    EngineStatus,
    LevelFrame,
    SpectrogramFrame,
    WarningEvent,
    WaveformFrame,
)
from soundspy.core.target import TargetSpec
from soundspy.engine.chain import ProcessorChain
from soundspy.engine.factory import build_chain
from soundspy.engine.processor import Passthrough, Processor
from soundspy.engine.session import Session, SessionError
from soundspy.io.devices import resolve_output
from soundspy.io.file_source import FileSource
from soundspy.io.speaker_sink import SpeakerSink
from soundspy.io.storage import read_sidecar

SR = 48000
N = SR  # one second of audio in the test file


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    """A 1 s stereo file whose samples identify their frame: value = frame / N (ch1 negated)."""
    ramp = (np.arange(N, dtype=np.float32) / N)[:, None]
    path = tmp_path / "ramp.wav"
    sf.write(str(path), np.hstack([ramp, -ramp]), SR, subtype="FLOAT")
    return path


class Collector:
    def __init__(self) -> None:
        self.events: list = []
        self._lock = threading.Lock()

    def __call__(self, event) -> None:
        with self._lock:
            self.events.append(event)

    def of(self, kind) -> list:
        with self._lock:
            return [e for e in self.events if isinstance(e, kind)]

    def wait_for(self, predicate, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("condition not reached in time")


def _session(config, wav, chain=None, output=None, loop=True, realtime=True) -> tuple:
    events = Collector()
    session = Session(
        FileSource(wav, realtime=realtime, loop=loop),
        chain or build_chain("semantic", config, config.labels),
        output=output,
        on_event=events,
        config=config,
    )
    return session, events


def test_start_stop_emits_every_event_type(config, wav, no_leaked_threads) -> None:
    session, events = _session(config, wav)
    session.start()
    events.wait_for(lambda: events.of(SpectrogramFrame) and events.of(WaveformFrame)
                    and events.of(LevelFrame))
    session.stop()
    session.stop()  # idempotent
    statuses = events.of(EngineStatus)
    assert statuses[0].state == "running" and statuses[-1].state == "idle"
    assert not session.thread_alive
    status = session.status
    assert status.chunks_processed > 0 and status.rtf_p95 is not None
    assert status.proc_ms_mean is not None and status.chunk_ms == 20.0
    assert status.source == "file" and status.channels == 2 and status.sample_rate == SR
    # step 06: the output is observed too, and the extractor is reported
    for kind in (SpectrogramFrame, WaveformFrame, LevelFrame):
        assert {e.source for e in events.of(kind)} == {"input", "output"}
    assert status.extraction == {"method": "mask_baseline", "latency_ms": 21.33,
                                 "gating": False, "target_active": False}
    assert status.bypass is False
    codes = [w.code for w in events.of(WarningEvent)]
    assert "detector_unavailable" in codes and "gating_unavailable" in codes


class TargetProbe(Processor):
    def __init__(self) -> None:
        self.calls: list[tuple[int, TargetSpec]] = []

    def set_target(self, target: TargetSpec) -> None:
        super().set_target(target)
        self.calls.append((threading.get_ident(), target))

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        return chunk


def test_set_target_is_applied_by_the_worker_between_chunks(config, wav) -> None:
    probe = TargetProbe()
    session, _ = _session(config, wav, chain=ProcessorChain([probe]))
    session.start()
    target = TargetSpec(labels={"speech"})
    session.set_target(target)
    deadline = time.monotonic() + 3
    while not probe.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    worker = session._thread.ident
    session.stop()
    assert probe.calls and probe.calls[0] == (worker, target)  # applied on the worker thread
    assert session.target is target


def test_recording_writes_the_raw_input(config, wav, tmp_path) -> None:
    session, _ = _session(config, wav)
    session.start()
    session.start_recording(tmp_path / "rec", label="live", notes="test")
    with pytest.raises(SessionError, match="already recording"):
        session.start_recording(tmp_path / "rec")
    time.sleep(0.4)
    path = session.stop_recording()
    session.stop()
    recorded, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR and recorded.shape[1] == 2  # every input channel, before the reducer
    assert recorded.shape[0] >= 0.2 * SR
    # The file is a ramp: the recording must be one contiguous (looping) stretch of it.
    offset = int(round(recorded[0, 0] * N))
    expected = (np.arange(offset, offset + recorded.shape[0]) % N).astype(np.float32) / N
    np.testing.assert_allclose(recorded[:, 0], expected, atol=1e-6)
    np.testing.assert_allclose(recorded[:, 1], -expected, atol=1e-6)
    sidecar = read_sidecar(path)
    assert sidecar["label"] == "live" and sidecar["notes"] == "test"
    assert sidecar["device"] == "file: ramp.wav" and sidecar["channels"] == 2
    assert sidecar["duration_s"] == pytest.approx(recorded.shape[0] / SR)


def test_stop_finishes_an_active_recording(config, wav, tmp_path) -> None:
    session, _ = _session(config, wav)
    session.start()
    session.start_recording(tmp_path / "rec")
    time.sleep(0.1)
    session.stop()
    assert session.last_recording is not None and session.last_recording.is_file()
    assert session.last_recording.with_suffix(".json").is_file()
    with pytest.raises(SessionError):
        session.stop_recording()


class Exploding(Processor):
    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        raise RuntimeError("model crashed")


def test_worker_error_sets_error_state(config, wav, no_leaked_threads) -> None:
    session, events = _session(config, wav, chain=ProcessorChain([Exploding()]))
    session.start()
    events.wait_for(lambda: any(s.state == "error" for s in events.of(EngineStatus)))
    assert session.state == "error" and "model crashed" in session.status.message
    session.stop()  # still safe after an error
    assert session.state == "error"


class Slow(Processor):
    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        time.sleep(0.03)  # 30 ms of work per 20 ms chunk: RTF 1.5
        return chunk


def test_slow_processing_raises_a_warning(config, wav) -> None:
    engine = dataclasses.replace(config.engine, rtf_window_s=0.2)
    session, events = _session(
        dataclasses.replace(config, engine=engine), wav, chain=ProcessorChain([Slow()])
    )
    session.start()
    events.wait_for(lambda: events.of(WarningEvent))
    session.stop()
    warning = events.of(WarningEvent)[0]
    assert warning.code == "rtf_high" and "real-time factor" in warning.message
    assert session.status.rtf_p95 > 1.0


def test_file_without_loop_finishes_by_itself(config, wav, no_leaked_threads) -> None:
    session, events = _session(config, wav, loop=False, realtime=False)
    session.start()
    events.wait_for(lambda: session.state == "idle" and not session.thread_alive)
    assert session.status.message == "source finished"
    session.stop()


def test_monitoring_plays_the_chain_output(
    config, tmp_path, fake_sounddevice, no_leaked_threads
) -> None:
    # Not the `wav` fixture: its channels cancel out in the mono mix the chain plays.
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(N) / SR)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), np.stack([tone, tone], 1), SR, subtype="FLOAT")
    sink = SpeakerSink(resolve_output(), config.output)
    session, _ = _session(config, wav, output=sink)
    session.start()
    assert session.status.output_device == "Speakers (Realtek(R) Audio)"
    with pytest.raises(SessionError):
        Session(
            FileSource(wav), ProcessorChain([Passthrough()]), output=None,
            on_event=lambda e: None, config=config,
        ).set_monitor(True, 0.0)
    session.set_monitor(True, -6.0)
    deadline = time.monotonic() + 3
    stream = fake_sounddevice.output_streams()[-1]
    while not np.nan_to_num(stream.played_audio()).any() and time.monotonic() < deadline:
        time.sleep(0.02)
    status = session.status
    session.stop()
    played = stream.played_audio()
    assert np.abs(played).max() > 0  # the (mono-mixed) chain output reached the speakers
    assert played.shape[1] == 2  # mono upmixed to the stereo output
    assert status.monitoring and status.gain_db == -6.0
    assert status.latency_ms > 20.0 + 21.3  # chunk + the extractor's STFT + output stream
    assert stream.closed


class Halver(Processor):
    """The "processing" of the bypass test: half the level, 480 samples late."""

    latency_samples = 480

    def configure(self, sample_rate: int, num_channels: int) -> int:
        super().configure(sample_rate, num_channels)
        self._buffer = np.zeros((480, num_channels), dtype=np.float32)
        return num_channels

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        joined = np.concatenate((self._buffer, chunk.samples * np.float32(0.5)))
        self._buffer = joined[chunk.num_frames:]
        return AudioBuffer(np.ascontiguousarray(joined[: chunk.num_frames]), chunk.sample_rate)


def test_bypass_sends_the_delay_matched_input_to_the_speakers(
    config, tmp_path, fake_sounddevice, no_leaked_threads
) -> None:
    wav = tmp_path / "level.wav"
    sf.write(str(wav), np.full((N, 1), 0.4, dtype=np.float32), SR, subtype="FLOAT")
    tap = Passthrough()
    chain = ProcessorChain([tap, Halver()], names={"reducer": tap, "extractor": tap},
                           dry_after="reducer")
    sink = SpeakerSink(resolve_output(), config.output)
    session, events = _session(config, wav, chain=chain, output=sink)
    session.start()
    session.set_monitor(True, 0.0)
    stream = fake_sounddevice.output_streams()[-1]

    def level() -> float:
        played = np.nan_to_num(stream.played_audio())
        return float(np.abs(played[-2000:]).max()) if played.shape[0] > 4000 else 0.0

    events.wait_for(lambda: abs(level() - 0.2) < 1e-3)  # the processed signal (halved)
    assert session.status.bypass is False
    session.set_bypass(True)
    assert session.status.bypass is True  # reported at once, applied at the next chunk
    events.wait_for(lambda: abs(level() - 0.4) < 1e-3)  # the unprocessed input
    assert session.chunks_processed > 0  # processing kept running
    session.set_bypass(False)
    events.wait_for(lambda: abs(level() - 0.2) < 1e-3)
    status = session.status
    session.stop()
    assert status.extraction == {"method": "passthrough", "latency_ms": 0.0, "gating": False,
                                 "target_active": False}
    assert status.latency_ms > 20.0 + 10.0  # includes the chain's 480 samples (10 ms)


def test_long_lived_objects_are_frozen_while_a_session_runs(config, wav) -> None:
    """A full garbage collection pauses every thread; frozen objects make it short."""
    import gc

    assert gc.get_freeze_count() == 0
    session, events = _session(config, wav)
    session.start()
    try:
        assert gc.get_freeze_count() > 10_000  # the interpreter, numpy, the config, ...
        events.wait_for(lambda: session.chunks_processed > 5)
        started = time.perf_counter()
        gc.collect()  # what would otherwise stall the audio thread for tens of milliseconds
        assert time.perf_counter() - started < 0.02
    finally:
        session.stop()
    assert gc.get_freeze_count() == 0  # normal collection again between sessions


def test_bypass_before_start_is_kept(config, wav) -> None:
    session, _ = _session(config, wav)
    session.set_bypass(True)
    assert session.bypass is True and session.status.bypass is True


def test_a_session_starts_only_once(config, wav) -> None:
    session, _ = _session(config, wav)
    session.start()
    with pytest.raises(SessionError):
        session.start()
    session.stop()

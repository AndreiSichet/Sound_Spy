"""Step 08 sessions: warming holds the audio back until the models are warm (progress in the
status), a stop while warming leaks nothing, a lost speaker or microphone ends in a clear error,
the model preloader, and ``soundspy soak`` end to end."""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundspy.core.audio import AudioBuffer
from soundspy.core.events import EngineStatus
from soundspy.engine.chain import ProcessorChain
from soundspy.engine.processor import Processor
from soundspy.engine.session import Session
from soundspy.io.devices import resolve_input, resolve_output
from soundspy.io.file_source import FileSource
from soundspy.io.mic_source import MicSource
from soundspy.io.speaker_sink import SpeakerSink

SR = 48000


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "tone.wav"
    t = np.arange(SR) / SR
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)
    return path


class SlowStart(Processor):
    """A stage whose warm-up takes a while; counts the chunks it saw."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.chunks = 0
        self.started = threading.Event()

    def start(self) -> None:
        time.sleep(self.seconds)
        self.started.set()

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        assert self.started.is_set(), "a chunk met a cold stage"
        self.chunks += 1
        return chunk


def _events():
    seen: list = []
    lock = threading.Lock()

    def on_event(e) -> None:
        with lock:
            seen.append(e)

    return seen, on_event


def test_warming_holds_audio_until_the_models_are_warm(config, wav, fake_sounddevice,
                                                       no_leaked_threads) -> None:
    stage = SlowStart(0.4)
    seen, on_event = _events()
    progress_seen = threading.Event()

    def prepare(progress, stopping) -> None:  # e.g. waiting for the preloaded models
        progress("Preparing models: loading the test model (1/1)")
        progress_seen.set()

    output = SpeakerSink(resolve_output(), config.output)
    session = Session(FileSource(wav, realtime=True, loop=True), None,
                      chain_factory=lambda: ProcessorChain([stage]), prepare=prepare,
                      output=output, on_event=on_event, config=config)
    session.set_monitor(True, 0.0)
    session.start()
    assert session.state == "warming" and session.active and not session.running
    assert progress_seen.wait(2)
    time.sleep(0.1)
    assert stage.chunks == 0  # nothing processed (nor played) while warming
    assert session.wait_until_running(5)
    deadline = time.monotonic() + 3
    while stage.chunks < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    session.stop()
    states = [e.state for e in seen if isinstance(e, EngineStatus)]
    messages = [e.message for e in seen if isinstance(e, EngineStatus) and e.state == "warming"]
    assert states[0] == "warming" and "running" in states
    assert any("loading the test model" in m for m in messages)
    assert session.warmup_s is not None and session.warmup_s >= 0.4
    assert stage.chunks >= 5


def test_stop_while_warming_leaks_nothing(config, wav, no_leaked_threads) -> None:
    stage = SlowStart(0.5)
    session = Session(FileSource(wav, realtime=True, loop=True), None,
                      chain_factory=lambda: ProcessorChain([stage]),
                      output=SpeakerSink(resolve_output(), config.output),
                      on_event=lambda e: None, config=config)
    session.start()
    time.sleep(0.05)
    session.stop()
    assert session.state == "idle" and not session.thread_alive


def test_a_lost_speaker_ends_the_session_with_a_clear_error(config, wav, fake_sounddevice,
                                                            no_leaked_threads) -> None:
    config = dataclasses.replace(config, output=dataclasses.replace(config.output,
                                                                    stall_timeout_s=0.2))
    seen, on_event = _events()
    output = SpeakerSink(resolve_output(), config.output)
    session = Session(FileSource(wav, realtime=True, loop=True), ProcessorChain([]),
                      output=output, on_event=on_event, config=config)
    session.set_monitor(True, 0.0)
    session.start()
    assert session.wait_until_running(5)
    stream = fake_sounddevice.output_streams()[-1]
    deadline = time.monotonic() + 2
    while not stream.played and time.monotonic() < deadline:
        time.sleep(0.01)
    stream.stop()  # unplugged: the device stops calling back
    deadline = time.monotonic() + 5
    while session.state != "error" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert session.state == "error"
    assert "stopped playing" in session.status.message and "Restart" in session.status.message
    session.stop()


def test_a_lost_microphone_ends_the_session_with_a_clear_error(config, fake_sounddevice,
                                                               no_leaked_threads) -> None:
    config = dataclasses.replace(config, capture=dataclasses.replace(config.capture,
                                                                     stall_timeout_s=0.3))
    fake_sounddevice.stall_after_frames = 4800  # unplugged after 0.1 s of audio
    session = Session(MicSource(resolve_input(config.device_profile), config.capture),
                      ProcessorChain([]), output=None, on_event=lambda e: None, config=config)
    session.start()
    deadline = time.monotonic() + 5
    while session.state != "error" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert session.state == "error" and "unplugged" in session.status.message
    session.stop()


def test_preloader_loads_the_tiny_models(tiny_neural) -> None:
    from soundspy.engine.preload import ModelPreloader

    preloader = ModelPreloader(tiny_neural, method="neural_tse")
    assert [s.name for s in preloader.steps] == ["the Waveformer"]  # no detector in tests
    messages: list[str] = []
    preloader.start()
    assert preloader.wait(messages.append, timeout=30)
    status = preloader.status()
    assert status["state"] == "ready" and status["done"] == status["total"] == 1
    assert status["steps"][0]["error"] is None and status["memory"]["ram_mb"] is not None
    assert all("Preparing models" in m for m in messages)


def test_soak_runs_and_summarises(config, wav, fake_sounddevice, no_leaked_threads) -> None:
    from soundspy.cli.soak_cmd import soak

    result = soak(config, minutes=0.02, every=0.4, method="mask_baseline", labels=["speech"],
                  attenuation=40.0, source=str(wav), progress=lambda m: None)
    summary = result["summary"]
    assert summary["state_end"] == "running" and summary["underruns"] >= 0
    assert result["warmup_s"] is not None and len(result["samples"]) >= 2
    assert result["settings"]["chunk_ms"] == config.engine.chunk_ms

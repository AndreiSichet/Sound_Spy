"""Shared pytest fixtures and configuration.

- Replaces ``sounddevice`` with ``FakeSoundDevice`` before any test module is imported, so tests
  never touch audio hardware (CLAUDE.md rule 11). The fake models a Windows laptop: the same
  built-in microphone under MME (name truncated to 31 characters, 44.1 kHz) and under WASAPI
  (full name, 48 kHz), a 6-channel reSpeaker array, and output devices. Its ``InputStream`` calls
  the callback from a thread with deterministic per-channel delayed noise.
- Skips tests marked ``gpu`` / ``data`` unless ``--run-gpu`` / ``--run-data`` is given.
- Config fixtures; ``app_config`` points the database and ``data/raw`` at a temporary folder.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

LAPTOP_MIC = "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)"
RESPEAKER = "reSpeaker 4 Mic Array (UAC1.0)"
FAKE_CHANNEL_DELAY = 3  # samples between consecutive channels of the default signal
FAKE_NOISE_AMPLITUDE = 0.1
STALE_HANDLE_ERROR = "Error starting stream: Unanticipated host error [PaErrorCode -9999]"

SignalFn = Callable[[int, int, int, int], np.ndarray]

_NOISE = np.random.default_rng(1234).standard_normal(1 << 18).astype(np.float32)


def delayed_noise(start: int, frames: int, channels: int, samplerate: int) -> np.ndarray:
    """Noise where channel c is channel 0 delayed by ``c * FAKE_CHANNEL_DELAY`` samples."""
    n = np.arange(start, start + frames)[:, None]
    delays = np.arange(channels)[None, :] * FAKE_CHANNEL_DELAY
    return FAKE_NOISE_AMPLITUDE * _NOISE[(n - delays) % _NOISE.size]


def channel_constant(start: int, frames: int, channels: int, samplerate: int) -> np.ndarray:
    """Every sample of device channel c equals c / 10 (shows which channels were kept)."""
    return np.tile(np.arange(channels, dtype=np.float32) / 10, (frames, 1))


def frame_ramp(start: int, frames: int, channels: int, samplerate: int) -> np.ndarray:
    """Every sample equals its absolute frame number (shows which frames were delivered)."""
    column = np.arange(start, start + frames, dtype=np.float32)[:, None]
    return np.repeat(column, channels, axis=1)


class PortAudioError(Exception):
    pass


class CallbackStop(Exception):  # noqa: N818 - same name as in sounddevice
    pass


class CallbackAbort(Exception):  # noqa: N818
    pass


class WasapiSettings:
    def __init__(self, exclusive: bool = False, auto_convert: bool = False) -> None:
        self.exclusive = exclusive
        self.auto_convert = auto_convert


class FakeInputStream:
    latency = 0.01

    def __init__(
        self,
        fake: FakeSoundDevice,
        *,
        device: int | None = None,
        channels: int | None = None,
        samplerate: float | None = None,
        dtype: str = "float32",
        callback=None,
        blocksize: int = 0,
        extra_settings=None,
        **_: object,
    ) -> None:
        info = fake.devices[device]
        if fake.fail_input_opens > 0:  # like stale WASAPI handles after a jack was plugged
            fake.fail_input_opens -= 1
            raise PortAudioError(STALE_HANDLE_ERROR)
        if channels is None or channels > info["max_input_channels"]:
            raise PortAudioError(f"Invalid number of channels [PaErrorCode -9998]: {channels}")
        hostapi = fake.hostapis[info["hostapi"]]["name"]
        native = int(info["default_samplerate"])
        auto_convert = bool(getattr(extra_settings, "auto_convert", False))
        if hostapi == "Windows WASAPI" and int(samplerate) != native and not auto_convert:
            # Real WASAPI shared mode only accepts the device's own rate.
            raise PortAudioError("Invalid sample rate [PaErrorCode -9997]")
        self.fake = fake
        self.device = device
        self.channels = channels
        self.samplerate = int(samplerate)
        self.dtype = dtype
        self.callback = callback
        self.blocksize = blocksize or 256
        self.extra_settings = extra_settings
        self.closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        fake.streams.append(self)

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="fake-input-stream", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        fake, frame, block = self.fake, 0, 0
        period = self.blocksize / self.samplerate / fake.speed
        while not self._stop.is_set():
            if fake.stall_after_frames is not None and frame >= fake.stall_after_frames:
                time.sleep(0.005)
                continue
            data = np.array(
                fake.signal(frame, self.blocksize, self.channels, self.samplerate),
                dtype=np.float32,
            )
            if frame < fake.leading_zero_frames:  # like WASAPI's start-up silence
                data[: fake.leading_zero_frames - frame] = 0.0
            overflow = bool(fake.overflow_every) and block % fake.overflow_every == 0
            status = types.SimpleNamespace(input_overflow=overflow)
            try:
                self.callback(data, self.blocksize, None, status)
            except (CallbackStop, CallbackAbort):
                break
            frame += self.blocksize
            block += 1
            time.sleep(period)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def close(self) -> None:
        self.stop()
        self.closed = True


class FakeOutputStream:
    """Pulls frames through the callback like a sound card and keeps everything it 'played'.

    With ``fake.output_autorun`` False no thread is started; tests call ``pull(frames)`` to run
    exactly one callback, which makes sink tests deterministic.
    """

    latency = 0.012

    def __init__(
        self,
        fake: FakeSoundDevice,
        *,
        device: int | None = None,
        channels: int | None = None,
        samplerate: float | None = None,
        dtype: str = "float32",
        callback=None,
        blocksize: int = 0,
        extra_settings=None,
        **_: object,
    ) -> None:
        info = fake.devices[device]
        if fake.fail_output_opens > 0:
            fake.fail_output_opens -= 1
            raise PortAudioError(STALE_HANDLE_ERROR)
        if channels is None or channels > info["max_output_channels"]:
            raise PortAudioError(f"Invalid number of channels [PaErrorCode -9998]: {channels}")
        hostapi = fake.hostapis[info["hostapi"]]["name"]
        auto_convert = bool(getattr(extra_settings, "auto_convert", False))
        if hostapi == "Windows WASAPI" and int(samplerate) != int(info["default_samplerate"]) \
                and not auto_convert:
            raise PortAudioError("Invalid sample rate [PaErrorCode -9997]")
        self.fake = fake
        self.device, self.channels, self.samplerate = device, channels, int(samplerate)
        self.callback = callback
        self.blocksize = blocksize or 256
        self.extra_settings = extra_settings
        self.played: list[np.ndarray] = []
        self.closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        fake.streams.append(self)

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def pull(self, frames: int | None = None) -> np.ndarray:
        frames = frames or self.blocksize
        out = np.full((frames, self.channels), np.nan, dtype=np.float32)
        self.callback(out, frames, None, types.SimpleNamespace(output_underflow=False))
        self.played.append(out.copy())
        return out

    def played_audio(self) -> np.ndarray:
        if not self.played:
            return np.empty((0, self.channels), dtype=np.float32)
        return np.concatenate(self.played)

    def start(self) -> None:
        if self.fake.output_autorun:
            self._thread = threading.Thread(
                target=self._run, name="fake-output-stream", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        period = self.blocksize / self.samplerate / self.fake.speed
        while not self._stop.is_set():
            try:
                self.pull()
            except (CallbackStop, CallbackAbort):
                break
            time.sleep(period)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def close(self) -> None:
        self.stop()
        self.closed = True


class FakeSoundDevice(types.ModuleType):
    """Stand-in for the ``sounddevice`` module (see module docstring)."""

    PortAudioError = PortAudioError
    CallbackStop = CallbackStop
    CallbackAbort = CallbackAbort
    WasapiSettings = WasapiSettings

    def __init__(self) -> None:
        super().__init__("sounddevice")
        self.reset()

    def reset(self) -> None:
        def dev(name, hostapi, ins, outs, rate):
            return {
                "name": name,
                "hostapi": hostapi,
                "max_input_channels": ins,
                "max_output_channels": outs,
                "default_samplerate": float(rate),
            }

        self.devices = [
            dev("Microsoft Sound Mapper - Input", 0, 2, 0, 44100),
            dev(LAPTOP_MIC[:31], 0, 2, 0, 44100),  # MME truncates names to 31 characters
            dev(RESPEAKER[:31], 0, 6, 0, 16000),
            dev("Microsoft Sound Mapper - Output", 0, 0, 2, 44100),
            dev("Speakers (Realtek(R) Audio)", 0, 0, 2, 44100),
            dev(LAPTOP_MIC, 1, 2, 0, 48000),
            dev(RESPEAKER, 1, 6, 0, 16000),
            dev("Speakers (Realtek(R) Audio)", 1, 0, 2, 48000),
            dev("Headphones (USB Audio)", 1, 0, 2, 48000),
        ]
        for i, d in enumerate(self.devices):
            d["index"] = i
        self.hostapis = [
            {"name": "MME", "devices": [0, 1, 2, 3, 4], "default_input_device": 1,
             "default_output_device": 4},
            {"name": "Windows WASAPI", "devices": [5, 6, 7, 8], "default_input_device": 5,
             "default_output_device": 7},
        ]
        # PortAudio's defaults come from the default host API, which is MME on Windows.
        self.default = types.SimpleNamespace(device=[1, 4], samplerate=None, channels=None)
        self.signal: SignalFn = delayed_noise
        self.speed = 20.0  # stream runs this many times faster than real time
        self.stall_after_frames: int | None = None
        self.overflow_every = 0
        self.leading_zero_frames = 0  # digital silence at the start of every stream
        self.output_autorun = True  # False: output callbacks only via stream.pull()
        self.fail_input_opens = 0  # the next N InputStream(...) calls raise PortAudioError
        self.fail_output_opens = 0
        self.refreshes = 0  # how often PortAudio was re-initialised (_terminate + _initialize)
        self.streams: list[FakeInputStream | FakeOutputStream] = []

    SIGNALS: dict[str, SignalFn] = {
        "delayed_noise": delayed_noise,
        "channel_constant": channel_constant,
        "frame_ramp": frame_ramp,
    }

    def use_signal(self, name: str) -> None:
        """Select the test signal by name (tests cannot import conftest directly)."""
        self.signal = self.SIGNALS[name]

    def query_devices(self, device=None, kind=None):
        if device is not None:
            return dict(self.devices[device])
        if kind == "input":
            return dict(self.devices[self.default.device[0]])
        if kind == "output":
            return dict(self.devices[self.default.device[1]])
        return [dict(d) for d in self.devices]

    def query_hostapis(self, index=None):
        return [dict(h) for h in self.hostapis] if index is None else dict(self.hostapis[index])

    def _terminate(self) -> None:
        pass

    def _initialize(self) -> None:
        self.refreshes += 1

    def InputStream(self, **kwargs):  # noqa: N802 - mirrors the sounddevice class name
        return FakeInputStream(self, **kwargs)

    def OutputStream(self, **kwargs):  # noqa: N802
        return FakeOutputStream(self, **kwargs)

    def output_streams(self) -> list[FakeOutputStream]:
        return [s for s in self.streams if isinstance(s, FakeOutputStream)]

    def _no_hardware(self, *args, **kwargs):
        raise RuntimeError("fake sounddevice: blocking capture/playback is not available in tests")

    rec = playrec = play = wait = _no_hardware
    Stream = _no_hardware


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-gpu", action="store_true", help="run tests marked 'gpu'")
    parser.addoption("--run-data", action="store_true", help="run tests marked 'data'")


def pytest_configure(config: pytest.Config) -> None:
    sys.modules["sounddevice"] = FakeSoundDevice()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, option in (("gpu", "--run-gpu"), ("data", "--run-data")):
        if config.getoption(option):
            continue
        skip = pytest.mark.skip(reason=f"needs {option}")
        for item in items:
            # Real markers only: ``item.keywords`` also contains folder names (tests/data).
            if item.get_closest_marker(marker) is not None:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def _reset_fake_sounddevice():
    """Each test starts with the default fake devices and no running streams."""
    fake = sys.modules["sounddevice"]
    fake.reset()
    yield
    for stream in fake.streams:
        stream.close()


@pytest.fixture
def fake_sounddevice() -> FakeSoundDevice:
    """The fake ``sounddevice`` module installed for the test session."""
    return sys.modules["sounddevice"]


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def configs_dir() -> Path:
    return BACKEND_DIR / "configs"


@pytest.fixture
def config():
    """The real project configuration (laptop profile), without the trained detection model.

    ``detection.checkpoint = None``: once ``best.pt`` exists, chains built from the real config
    would otherwise load it (on the GPU if there is one, against rule 11). Tests that need a
    model use ``stub_experiment``.
    """
    from soundspy.config import load_config

    config = load_config()
    return dataclasses.replace(config, detection=dataclasses.replace(config.detection,
                                                                     checkpoint=None))


AUDIO_THREAD_NAMES = ("soundspy-session", "fake-input-stream", "fake-output-stream",
                      "soundspy-detector", "soundspy-runs")


@pytest.fixture
def no_leaked_threads():
    """Fail a test that leaves a session worker or a (fake) audio stream thread running.

    Runs its check before the autouse fixture closes leftover fake streams, so the code under
    test must have stopped them itself.
    """
    before = {t.ident for t in threading.enumerate()}
    yield
    deadline = time.monotonic() + 3.0
    while True:
        leaked = [
            t.name
            for t in threading.enumerate()
            if t.ident not in before and t.is_alive() and t.name in AUDIO_THREAD_NAMES
        ]
        if not leaked or time.monotonic() > deadline:
            break
        time.sleep(0.02)
    assert not leaked, f"threads still running after the test: {leaked}"


@pytest.fixture
def app_config(config, tmp_path: Path):
    """Project configuration with the database, ``data/raw`` and ``outputs`` in a temporary
    folder (processing runs are written to ``outputs/processed``).

    No detection model (``detection.checkpoint = None``): sessions start without the detector
    and warn, so tests never load a real trained model (use ``stub_experiment`` for one).
    """
    db_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    paths = {**config.paths, "raw": tmp_path / "raw", "outputs": tmp_path / "outputs"}
    detection = dataclasses.replace(config.detection, checkpoint=None)
    return dataclasses.replace(config, database_url=db_url, paths=paths, detection=detection)


@pytest.fixture
def stub_experiment(config, tmp_path: Path):
    """A trained-looking detection experiment with the tiny ``stub`` backbone (CPU, no data).

    Returns ``(config, experiment_dir)``; ``config.detection.checkpoint`` points at its
    ``best.pt`` (16 kHz, 1 s window, 1 s context, threshold 0.5 for every class).
    """
    import json

    from soundspy.detection.model import AudioSetRows, DetectorNet, ModelSpec, save_checkpoint

    labels = list(config.labels)
    rows = AudioSetRows({lab: [i] for i, lab in enumerate(labels)},
                        {lab: [i, i + 10] for i, lab in enumerate(labels)})
    spec = ModelSpec("stub", labels, 16_000, 1.0, rows, {}, 1.0)
    model = DetectorNet(spec).eval()
    exp = tmp_path / "experiments" / "stub_v1"
    save_checkpoint(exp / "checkpoints" / "best.pt", model, {"epoch": 1})
    (exp / "thresholds.json").write_text(
        json.dumps({"thresholds": {lab: 0.5 for lab in labels}}), encoding="utf-8")
    detection = dataclasses.replace(config.detection, checkpoint=exp / "checkpoints" / "best.pt",
                                    live=dataclasses.replace(config.detection.live, device="cpu"))
    paths = {**config.paths, "experiments": tmp_path / "experiments"}
    return dataclasses.replace(config, detection=detection, paths=paths), exp

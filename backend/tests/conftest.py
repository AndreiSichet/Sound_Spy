"""Shared pytest fixtures and configuration.

- Replaces ``sounddevice`` with a fake module before any test module is imported, so
  tests never touch audio hardware (CLAUDE.md rule 11).
- Skips tests marked ``gpu`` / ``data`` unless ``--run-gpu`` / ``--run-data`` is given.
- Config fixtures; ``app_config`` points the database at a temporary SQLite file.

TODO(step-02): extend the fake backend (capture/streams returning synthetic audio) once
MicSource exists.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


class FakeSoundDevice(types.ModuleType):
    """Minimal stand-in for the ``sounddevice`` module: one fake WASAPI input device."""

    def __init__(self) -> None:
        super().__init__("sounddevice")
        self.devices = [
            {
                "name": "Fake Microphone",
                "index": 0,
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            }
        ]
        self.hostapis = [{"name": "Windows WASAPI", "devices": [0], "default_input_device": 0}]
        self.default = types.SimpleNamespace(device=[0, -1], samplerate=None, channels=None)

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            return list(self.devices)
        return self.devices[0]

    def query_hostapis(self, index=None):
        return list(self.hostapis) if index is None else self.hostapis[index]

    def _no_hardware(self, *args, **kwargs):
        raise RuntimeError("fake sounddevice: audio capture is not available in tests")

    rec = playrec = play = wait = _no_hardware
    InputStream = Stream = OutputStream = _no_hardware


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
            if marker in item.keywords:
                item.add_marker(skip)


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
    """The real project configuration (laptop profile)."""
    from soundspy.config import load_config

    return load_config()


@pytest.fixture
def app_config(config, tmp_path: Path):
    """Project configuration with the database in a temporary SQLite file."""
    db_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    return dataclasses.replace(config, database_url=db_url)

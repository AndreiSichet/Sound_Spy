"""Recording storage: naming, sidecar, immutability, listing."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundspy import __version__
from soundspy.core.audio import AudioBuffer
from soundspy.io import storage
from soundspy.io.storage import iter_recordings, load_recording, save_recording

FIXED = datetime(2026, 11, 15, 9, 30, 5, 123456, tzinfo=timezone(timedelta(hours=2)))


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED


def _buffer() -> AudioBuffer:
    samples = np.linspace(-0.5, 0.5, 4800 * 2, dtype=np.float32).reshape(4800, 2)
    meta = {
        "device": "Microphone Array",
        "hostapi": "Windows WASAPI",
        "profile": "laptop",
        "channel_indices": [0, 1],
        "overflows": 1,
    }
    return AudioBuffer(samples, 48000, meta)


def test_file_name_and_sidecar(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "datetime", _FixedDatetime)
    wav = save_recording(_buffer(), tmp_path, label="dish test!", notes="near the sink")
    assert wav.name == "dish-test_20261115-093005-123.wav"
    sidecar = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar == {
        "file": wav.name,
        "sample_rate": 48000,
        "channels": 2,
        "duration_s": 0.1,
        "created": "2026-11-15T09:30:05.123+02:00",
        "device": "Microphone Array",
        "hostapi": "Windows WASAPI",
        "profile": "laptop",
        "channel_indices": [0, 1],
        "label": "dish test!",
        "notes": "near the sink",
        "overflows": 1,
        "app_version": __version__,
    }
    info = sf.info(str(wav))
    assert (info.samplerate, info.channels, info.subtype) == (48000, 2, "FLOAT")


def test_default_name_without_label(tmp_path: Path) -> None:
    wav = save_recording(_buffer(), tmp_path)
    assert re.fullmatch(r"rec_\d{8}-\d{6}-\d{3}\.wav", wav.name)


def test_never_overwrites(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "datetime", _FixedDatetime)
    first = save_recording(_buffer(), tmp_path, label="x")
    before = first.read_bytes()
    with pytest.raises(FileExistsError, match="never overwritten"):
        save_recording(_buffer(), tmp_path, label="x")
    assert first.read_bytes() == before


def test_load_round_trip(tmp_path: Path) -> None:
    original = _buffer()
    wav = save_recording(original, tmp_path, label="speech", profile="laptop")
    buffer, metadata = load_recording(wav)
    np.testing.assert_array_equal(buffer.samples, original.samples)
    assert buffer.sample_rate == 48000 and metadata["label"] == "speech"


def test_iter_recordings_needs_a_sidecar(tmp_path: Path) -> None:
    a = save_recording(_buffer(), tmp_path, label="a")
    sf.write(str(tmp_path / "orphan.wav"), np.zeros((10, 1), dtype=np.float32), 16000)
    (tmp_path / "broken.wav").write_bytes(b"")
    (tmp_path / "broken.json").write_text("[1, 2]", encoding="utf-8")
    found = list(iter_recordings(tmp_path))
    assert [p for p, _ in found] == [a]
    assert list(iter_recordings(tmp_path / "missing")) == []

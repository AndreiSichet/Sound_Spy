"""GET /api/devices and GET /api/recordings (+ the recordings index service)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from soundspy.api.app import create_app
from soundspy.api.db import Recording, init_db, make_engine, make_session_factory
from soundspy.api.services.recordings import stored_path, sync_recordings
from soundspy.core.audio import AudioBuffer
from soundspy.io import devices as devices_module
from soundspy.io.devices import AudioBackendUnavailable
from soundspy.io.storage import save_recording


def _client(config) -> TestClient:
    return TestClient(create_app(config))


def _record(directory: Path, label: str) -> Path:
    buf = AudioBuffer(
        np.zeros((4800, 2), dtype=np.float32),
        48000,
        {"device": "Mic", "hostapi": "Windows WASAPI", "channel_indices": [0, 1]},
    )
    return save_recording(buf, directory, label=label, profile="laptop")


# --- /api/devices -----------------------------------------------------------------------------


def test_devices(app_config) -> None:
    with _client(app_config) as client:
        response = client.get("/api/devices")
    assert response.status_code == 200
    body = response.json()
    assert body["active_profile"] == "laptop" and body["error"] is None
    assert body["resolved_input"] == {
        "index": 5,
        "name": "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)",
        "hostapi": "Windows WASAPI",
        "channels_used": [0, 1],
        "sample_rate": 48000,
    }
    assert {d["hostapi"] for d in body["inputs"]} == {"MME", "Windows WASAPI"}
    default = next(d for d in body["inputs"] if d["is_default"])
    assert default == {
        "index": 1,
        "name": "Microphone Array (Intel® Smart ",
        "hostapi": "MME",
        "max_channels": 2,
        "default_samplerate": 44100,
        "is_default": True,
    }
    assert len(body["outputs"]) == 4


def test_devices_with_unresolvable_profile_is_200_with_error(app_config) -> None:
    profile = dataclasses.replace(app_config.device_profile, device="Zoom H6")
    config = dataclasses.replace(app_config, device_profile=profile)
    with _client(config) as client:
        response = client.get("/api/devices")
    assert response.status_code == 200
    body = response.json()
    assert body["resolved_input"] is None and "Zoom H6" in body["error"]
    assert len(body["inputs"]) == 5


def test_devices_without_portaudio_is_503(app_config, monkeypatch) -> None:
    def unavailable():
        raise AudioBackendUnavailable("PortAudio library not found")

    monkeypatch.setattr(devices_module, "_sd", unavailable)
    with _client(app_config) as client:
        response = client.get("/api/devices")
    assert response.status_code == 503
    assert "PortAudio library not found" in response.json()["detail"]


# --- /api/recordings --------------------------------------------------------------------------


def test_recordings_are_indexed_and_listed_newest_first(app_config) -> None:
    raw = app_config.paths["raw"]
    first = _record(raw, "first")
    with _client(app_config) as client:  # startup sync indexes `first`
        _record(raw, "second")  # written while the API runs, like `soundspy record`
        body = client.get("/api/recordings").json()
    assert [r["label"] for r in body] == ["second", "first"]
    row = body[1]
    assert row["path"].endswith(first.name)
    assert (row["sample_rate"], row["channels"], row["duration_s"]) == (48000, 2, 0.1)
    assert (row["device"], row["profile"], row["notes"]) == ("Mic", "laptop", "")
    assert row["created_at"].endswith("Z") or row["created_at"].endswith("+00:00")


def test_deleted_file_disappears_from_the_list(app_config) -> None:
    raw = app_config.paths["raw"]
    wav = _record(raw, "gone")
    with _client(app_config) as client:
        assert len(client.get("/api/recordings").json()) == 1
        wav.unlink()
        assert client.get("/api/recordings").json() == []


def test_sync_service(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{(tmp_path / 'i.db').as_posix()}", tmp_path)
    init_db(engine)
    factory = make_session_factory(engine)
    raw = tmp_path / "data" / "raw"
    wav = _record(raw, "a")
    _record(raw, "b").with_suffix(".json").unlink()  # no sidecar -> not indexed
    with factory() as db:
        first = sync_recordings(db, raw, tmp_path)
        again = sync_recordings(db, raw, tmp_path)
        paths = list(db.scalars(select(Recording.path)))
    assert (first.added, first.removed, again.added, again.removed) == (1, 0, 0, 0)
    assert paths == [f"data/raw/{wav.name}"]  # relative to the project root, forward slashes
    engine.dispose()


def test_stored_path_outside_project_is_absolute(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "x.wav"
    assert stored_path(outside, tmp_path / "project") == outside.resolve().as_posix()


@pytest.mark.parametrize("label", ["speech", None])
def test_cli_style_files_need_no_api(tmp_path: Path, label) -> None:
    # `soundspy record` writes only files; indexing happens later. Nothing here touches the DB.
    wav = _record(tmp_path, label)
    assert wav.is_file() and wav.with_suffix(".json").is_file()

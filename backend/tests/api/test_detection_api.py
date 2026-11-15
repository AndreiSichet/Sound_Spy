"""Detection through the API: live ``detections`` messages, status, offline /detect route."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from soundspy.api.app import create_app
from soundspy.core.audio import AudioBuffer
from soundspy.io.storage import save_recording

SR = 48_000
FILE_SOURCE = {"mode": "SEMANTIC",
               "source": {"type": "file", "path": "audio/tone.wav", "loop": True}}


def write_tone(root: Path, seconds: float = 2.0) -> None:
    t = np.arange(int(SR * seconds)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    sf.write(str(root / "audio" / "tone.wav"), np.stack([tone, tone], 1), SR, subtype="FLOAT")


@pytest.fixture
def det_config(app_config, stub_experiment, tmp_path: Path):
    """App config (tmp DB, project root = tmp) with the stub detection model."""
    stub_cfg, _exp = stub_experiment
    write_tone(tmp_path)
    paths = {**app_config.paths, "raw": tmp_path / "data" / "raw"}
    return dataclasses.replace(app_config, project_root=tmp_path, paths=paths,
                               detection=stub_cfg.detection)


@pytest.fixture
def no_model_config(app_config, tmp_path: Path):
    write_tone(tmp_path)
    paths = {**app_config.paths, "raw": tmp_path / "data" / "raw"}
    return dataclasses.replace(app_config, project_root=tmp_path, paths=paths)


def collect(ws, wanted: set[str], limit: int = 2000) -> dict[str, list[dict]]:
    seen: dict[str, list[dict]] = {}
    for _ in range(limit):
        message = ws.receive_json()
        seen.setdefault(message["type"], []).append(message)
        if wanted <= set(seen) and len(seen.get("detections", [])) >= 3:
            break
    return seen


def test_live_detections_over_the_websocket(det_config, no_leaked_threads) -> None:
    labels = list(det_config.labels)
    with TestClient(create_app(det_config)) as client, \
            client.websocket_connect("/ws/session") as ws:
        ws.receive_json()  # idle status
        assert client.post("/api/session/start", json=FILE_SOURCE).status_code == 200
        seen = collect(ws, {"detections", "status"})
        status = client.get("/api/session").json()
        client.post("/api/session/stop")
    frames = seen["detections"]
    assert len(frames) >= 3
    first = frames[0]
    assert first["t"] == pytest.approx(1.0)  # first full 1 s window
    assert set(first["payload"]["probabilities"]) == set(labels)
    assert set(first["payload"]["active"]) <= set(labels)
    hops = np.diff([f["t"] for f in frames])
    assert np.all(hops >= 0.25 - 1e-6)  # one per 0.25 s hop (skipped windows leave gaps)
    det = status["detector"]
    assert det["model"] == "stub_v1" and det["device"] == "cpu" and det["hop_s"] == 0.25
    assert "warning" not in seen or all(
        m["payload"]["code"] != "detector_unavailable" for m in seen["warning"])


def test_session_without_model_starts_and_warns(no_model_config, no_leaked_threads) -> None:
    with TestClient(create_app(no_model_config)) as client, \
            client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        started = client.post("/api/session/start", json=FILE_SOURCE)
        assert started.status_code == 200 and started.json()["detector"] is None
        warning = None
        for _ in range(500):
            message = ws.receive_json()
            if message["type"] == "warning":
                warning = message
                break
        client.post("/api/session/stop")
    assert warning is not None
    assert warning["payload"]["code"] == "detector_unavailable"
    assert "detection" in warning["payload"]["message"].lower()


def make_recording(config, seconds: float) -> None:
    rng = np.random.default_rng(1)
    buf = AudioBuffer(rng.normal(0, 0.05, (int(SR * seconds), 2)).astype(np.float32), SR)
    save_recording(buf, config.paths["raw"], label="det-test")


def test_offline_detect_route(det_config) -> None:
    make_recording(det_config, 3.0)
    with TestClient(create_app(det_config)) as client:
        rec = client.get("/api/recordings").json()[0]
        res = client.post(f"/api/recordings/{rec['id']}/detect")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["duration_s"] == pytest.approx(3.0)
        assert body["times"][0] == 1.0 and body["times"][-1] == 3.0
        assert set(body["probabilities"]) == set(det_config.labels)
        assert client.post("/api/recordings/999/detect").status_code == 404


def test_long_recordings_are_413(det_config) -> None:
    cfg = dataclasses.replace(det_config, detection=dataclasses.replace(
        det_config.detection, offline_max_seconds=2.0))
    make_recording(cfg, 3.0)
    with TestClient(create_app(cfg)) as client:
        rec = client.get("/api/recordings").json()[0]
        res = client.post(f"/api/recordings/{rec['id']}/detect")
    assert res.status_code == 413 and "soundspy detect" in res.json()["detail"]


def test_detect_without_model_is_503(no_model_config) -> None:
    make_recording(no_model_config, 1.5)
    with TestClient(create_app(no_model_config)) as client:
        rec = client.get("/api/recordings").json()[0]
        res = client.post(f"/api/recordings/{rec['id']}/detect")
    assert res.status_code == 503

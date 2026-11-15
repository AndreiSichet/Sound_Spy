"""Session routes, recording through the API, the WebSocket during a session, slow clients."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from soundspy.api.app import create_app
from soundspy.api.schemas import WsMessage
from soundspy.api.ws import ClientQueue
from soundspy.io import devices as devices_module
from soundspy.io.devices import AudioBackendUnavailable

SR = 48000


@pytest.fixture
def root_config(app_config, tmp_path: Path):
    """Project root = tmp folder, with a short tone file to use as a session source."""
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    (tmp_path / "audio").mkdir()
    sf.write(str(tmp_path / "audio" / "tone.wav"), np.stack([tone, tone], 1), SR, subtype="FLOAT")
    paths = {**app_config.paths, "raw": tmp_path / "data" / "raw"}
    return dataclasses.replace(app_config, project_root=tmp_path, paths=paths)


@pytest.fixture
def client(root_config):
    with TestClient(create_app(root_config)) as test_client:
        yield test_client


FILE_SOURCE = {"mode": "SEMANTIC", "source": {"type": "file", "path": "audio/tone.wav"}}


def test_idle_status(client: TestClient) -> None:
    body = client.get("/api/session").json()
    assert body["state"] == "idle" and body["monitoring"] is False


def test_start_conflict_and_idempotent_stop(client: TestClient, no_leaked_threads) -> None:
    started = client.post("/api/session/start", json=FILE_SOURCE)
    assert started.status_code == 200, started.text
    assert started.json()["state"] == "running" and started.json()["source"] == "file"
    assert client.post("/api/session/start", json=FILE_SOURCE).status_code == 409
    assert client.get("/api/session").json()["state"] == "running"
    stopped = client.post("/api/session/stop")
    assert stopped.status_code == 200 and stopped.json()["state"] == "idle"
    assert client.post("/api/session/stop").status_code == 200  # idempotent
    assert client.post("/api/session/start", json=FILE_SOURCE).status_code == 200  # restartable
    client.post("/api/session/stop")


def test_microphone_session(client: TestClient, no_leaked_threads) -> None:
    body = client.post("/api/session/start", json={"mode": "semantic"}).json()
    assert body["source"] == "mic" and body["channels"] == 2 and body["sample_rate"] == 48000
    assert body["output_device"] == "Speakers (Realtek(R) Audio)"
    client.post("/api/session/stop")


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"mode": "spatial"}, "module 3"),
        ({"mode": "nonsense"}, "unknown mode"),
        ({"source": {"type": "file"}}, "needs 'path'"),
        ({"source": {"type": "file", "path": "audio/missing.wav"}}, "no such file"),
        ({"source": {"type": "file", "path": "../outside.wav"}}, "inside the project"),
    ],
)
def test_bad_start_requests_are_400(client: TestClient, body, fragment) -> None:
    response = client.post("/api/session/start", json=body)
    assert response.status_code == 400 and fragment in response.json()["detail"]


def test_no_audio_backend_is_503(client: TestClient, monkeypatch) -> None:
    def unavailable():
        raise AudioBackendUnavailable("PortAudio library not found")

    monkeypatch.setattr(devices_module, "_sd", unavailable)
    response = client.post("/api/session/start", json={"mode": "semantic"})
    assert response.status_code == 503 and "PortAudio" in response.json()["detail"]


def test_target_validation(client: TestClient) -> None:
    assert client.put("/api/session/target", json={"labels": ["laughter"]}).status_code == 422
    ok = client.put("/api/session/target", json={"labels": ["speech", "music"]})
    assert ok.status_code == 200 and sorted(ok.json()["labels"]) == ["music", "speech"]
    assert client.app.state.session_manager.target.labels == {"speech", "music"}


@pytest.mark.parametrize(("gain", "code"), [(13, 422), (-41, 422), (12, 200), (-40, 200)])
def test_monitor_gain_limits(client: TestClient, gain, code) -> None:
    response = client.put("/api/session/monitor", json={"enabled": False, "gain_db": gain})
    assert response.status_code == code


def test_monitor_during_a_session(client: TestClient) -> None:
    client.post("/api/session/start", json=FILE_SOURCE)
    body = client.put("/api/session/monitor", json={"enabled": True, "gain_db": -6}).json()
    time.sleep(0.1)
    status = client.get("/api/session").json()
    client.post("/api/session/stop")
    assert body["gain_db"] == -6.0
    assert status["monitoring"] is True and status["latency_ms"] > 20


def test_listen_setting_is_shown_while_idle(client: TestClient) -> None:
    # Owner's report: untick Listen during a session, stop, tick it again -> the box stayed
    # unticked until the next Start, because the reply carried the old session's status.
    off = {"enabled": False, "gain_db": 1, "muted": False}
    on = {"enabled": True, "gain_db": 1, "muted": False}
    assert client.put("/api/session/monitor", json=on).json()["monitoring"] is True  # no session
    client.post("/api/session/start", json=FILE_SOURCE)
    assert client.put("/api/session/monitor", json=off).json()["monitoring"] is False
    stopped = client.post("/api/session/stop").json()
    assert stopped["state"] == "idle" and stopped["monitoring"] is False
    ticked = client.put("/api/session/monitor", json=on).json()
    assert ticked["state"] == "idle" and ticked["monitoring"] is True and ticked["gain_db"] == 1
    assert client.get("/api/session").json()["monitoring"] is True  # heartbeat shows the same
    restarted = client.post("/api/session/start", json=FILE_SOURCE).json()
    assert restarted["monitoring"] is True  # the stored choice is applied to the new session
    client.post("/api/session/stop")


def test_record_through_the_api(client: TestClient, no_leaked_threads) -> None:
    assert client.post("/api/session/record/start", json={}).status_code == 409  # no session
    client.post("/api/session/start", json=FILE_SOURCE)
    started = client.post("/api/session/record/start", json={"label": "ui", "notes": "n"})
    assert started.status_code == 200 and started.json()["recording"] is True
    assert client.post("/api/session/record/start", json={}).status_code == 409  # twice
    time.sleep(0.3)
    stopped = client.post("/api/session/record/stop")
    client.post("/api/session/stop")
    assert stopped.status_code == 200, stopped.text
    row = stopped.json()
    assert row["label"] == "ui" and row["notes"] == "n" and row["channels"] == 2
    assert row["path"].startswith("data/raw/ui_") and row["duration_s"] > 0.1
    audio = client.get(f"/api/recordings/{row['id']}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert audio.content[:4] == b"RIFF"
    assert client.get(f"/api/recordings/{row['id']}").json()["id"] == row["id"]
    assert [r["id"] for r in client.get("/api/recordings").json()] == [row["id"]]
    assert client.post("/api/session/record/stop").status_code == 409


def test_missing_recordings_are_404(client: TestClient) -> None:
    assert client.get("/api/recordings/999").status_code == 404
    assert client.get("/api/recordings/999/audio").status_code == 404


def test_websocket_streams_live_data(client: TestClient) -> None:
    seen: set[str] = set()
    with client.websocket_connect("/ws/session") as ws:
        assert ws.receive_json()["payload"]["state"] == "idle"
        client.post("/api/session/start", json=FILE_SOURCE)
        for _ in range(400):
            message = ws.receive_json()
            seen.add(message["type"])
            if message["type"] == "spectrogram":
                assert len(message["payload"]["magnitudes_db"]) == 128
            if message["type"] in ("levels", "waveform", "spectrogram"):
                seen.add(f"{message['type']}:{message['payload']['source']}")
            if {"status", "spectrogram:input", "spectrogram:output", "levels:output",
                    "waveform:output", "levels:input", "waveform:input"} <= seen:
                break
        client.post("/api/session/stop")
    assert {"status", "levels", "waveform", "spectrogram"} <= seen
    assert {"spectrogram:input", "spectrogram:output", "levels:output", "waveform:output"} <= seen


def test_bypass_over_rest_and_websocket(client: TestClient, no_leaked_threads) -> None:
    assert client.get("/api/session").json()["bypass"] is False
    body = client.put("/api/session/bypass", json={"enabled": True}).json()
    assert body["bypass"] is True and body["state"] == "idle"  # remembered while idle
    started = client.post("/api/session/start", json=FILE_SOURCE).json()
    assert started["bypass"] is True  # the new session starts bypassed
    assert started["extraction"]["method"] == "mask_baseline"
    assert started["extraction"]["latency_ms"] == pytest.approx(21.33)
    assert started["extraction"]["gating"] is False  # no detection model in tests
    assert started["latency_ms"] >= 20.0 + 21.3  # chunk + STFT
    with client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_bypass", "t": 0, "payload": {"enabled": False}})
        for _ in range(500):
            message = ws.receive_json()
            if message["type"] == "status" and message["payload"]["bypass"] is False:
                break
        else:
            raise AssertionError("no status with bypass off")
        ws.send_json({"type": "set_bypass", "t": 0, "payload": {"enabled": "maybe"}})
        for _ in range(500):
            message = ws.receive_json()
            if message["type"] == "error":
                assert "set_bypass" in message["payload"]["message"]
                break
        else:
            raise AssertionError("no error for a malformed set_bypass")
    assert client.get("/api/session").json()["bypass"] is False
    assert client.put("/api/session/bypass", json={}).status_code == 422
    client.post("/api/session/stop")


def test_selected_target_reaches_the_extractor(client: TestClient, no_leaked_threads) -> None:
    client.put("/api/session/target", json={"labels": ["speech"], "attenuation_db": 12})
    client.post("/api/session/start", json=FILE_SOURCE)
    session = client.app.state.session_manager.session
    extractor = session._chain.stage("extractor")
    deadline = time.monotonic() + 3
    while extractor.target is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert extractor.target.labels == {"speech"} and extractor.target.attenuation_db == 12.0
    client.post("/api/session/stop")


def test_app_shutdown_stops_the_session(root_config, no_leaked_threads) -> None:
    with TestClient(create_app(root_config)) as test_client:
        assert test_client.post("/api/session/start", json=FILE_SOURCE).status_code == 200
    # leaving the context runs the lifespan shutdown; no_leaked_threads checks the worker is gone


# --- slow clients ------------------------------------------------------------------------------


def _msg(kind: str) -> WsMessage:
    return WsMessage(type=kind, payload={})


def test_slow_client_drops_visual_messages_but_keeps_status() -> None:
    queue = ClientQueue(maxsize=3)
    queue.put(_msg("status"))
    for _ in range(5):
        queue.put(_msg("levels"))
    queue.put(_msg("warning"))
    types_ = [m.type for m in queue.items]
    assert types_ == ["status", "levels", "warning"]
    assert queue.dropped == 4


def test_queue_never_drops_status_even_when_over_full() -> None:
    queue = ClientQueue(maxsize=2)
    for kind in ["status", "error", "warning", "status"]:
        queue.put(_msg(kind))
    assert [m.type for m in queue.items] == ["status", "error", "warning", "status"]
    queue.put(_msg("spectrogram"))  # nothing droppable but the newcomer itself
    assert [m.type for m in queue.items][-1] != "spectrogram" and queue.dropped == 1

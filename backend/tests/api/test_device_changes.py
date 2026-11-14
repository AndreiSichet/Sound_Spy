"""Device changes while the backend runs (e.g. headphones plugged into a Realtek jack).

PortAudio keeps the device handles it found at start-up; after Windows reconfigures an endpoint,
opening a stream through them can fail. The session manager re-scans the devices before every
start, retries once after another re-scan, and explains a failure that persists.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from soundspy.api.app import create_app
from soundspy.cli.serve_cmd import log_config


@pytest.fixture
def client(app_config, tmp_path: Path):
    config = dataclasses.replace(app_config, project_root=tmp_path)
    with TestClient(create_app(config)) as test_client:
        yield test_client


def test_start_rescans_devices(client: TestClient, fake_sounddevice) -> None:
    before = fake_sounddevice.refreshes
    assert client.post("/api/session/start", json={"mode": "semantic"}).status_code == 200
    client.post("/api/session/stop")
    assert fake_sounddevice.refreshes == before + 1


def test_a_failed_open_is_retried_after_a_rescan(client: TestClient, fake_sounddevice) -> None:
    fake_sounddevice.fail_input_opens = 1  # stale handle: the first open fails
    before = fake_sounddevice.refreshes
    response = client.post("/api/session/start", json={"mode": "semantic"})
    client.post("/api/session/stop")
    assert response.status_code == 200 and response.json()["state"] == "running"
    assert fake_sounddevice.refreshes == before + 2  # before the start, and before the retry


def test_a_persistent_open_failure_is_explained(client: TestClient, fake_sounddevice) -> None:
    fake_sounddevice.fail_input_opens = 2
    response = client.post("/api/session/start", json={"mode": "semantic"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "could not open the audio device" in detail and "plugged in or out" in detail
    assert client.get("/api/session").json()["state"] == "idle"
    assert client.post("/api/session/start", json={"mode": "semantic"}).status_code == 200
    client.post("/api/session/stop")


def test_session_without_output_says_why(client: TestClient, fake_sounddevice) -> None:
    fake_sounddevice.fail_output_opens = 1
    with client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        started = client.post("/api/session/start", json={"mode": "semantic"}).json()
        warning = None
        for _ in range(50):
            message = ws.receive_json()
            if message["type"] == "warning":
                warning = message["payload"]
                break
    assert started["state"] == "running" and started["output_device"] is None
    assert "Listening is unavailable" in started["message"]
    assert warning is not None and warning["code"] == "output_unavailable"
    response = client.put("/api/session/monitor", json={"enabled": True, "gain_db": 0})
    assert response.status_code == 409 and "no output device" in response.json()["detail"]
    client.post("/api/session/stop")


def test_devices_route_rescans_only_when_idle(client: TestClient, fake_sounddevice) -> None:
    before = fake_sounddevice.refreshes
    client.get("/api/devices")
    assert fake_sounddevice.refreshes == before + 1
    client.post("/api/session/start", json={"mode": "semantic"})
    during = fake_sounddevice.refreshes
    client.get("/api/devices")  # must not re-initialise PortAudio under a running stream
    assert fake_sounddevice.refreshes == during
    client.post("/api/session/stop")


def test_serve_logs_soundspy_messages() -> None:
    config = log_config()
    assert config["loggers"]["soundspy"]["level"] == "INFO"
    assert "default" in config["loggers"]["soundspy"]["handlers"]

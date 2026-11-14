"""API tests with FastAPI's TestClient (no hardware, temporary SQLite database)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from soundspy import __version__
from soundspy.api.app import create_app


@pytest.fixture
def client(app_config):
    with TestClient(create_app(app_config)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_config(client: TestClient) -> None:
    body = client.get("/api/config").json()
    assert [c["name"] for c in body["classes"]] == [
        "speech", "music", "footsteps", "dishes", "chair_scrape", "alarm", "traffic"
    ]
    dishes = body["classes"][3]
    assert dishes == {
        "name": "dishes",
        "display_name": "Dishes & cutlery",
        "description": "Plates, cups, pots and cutlery clinking",
    }  # AudioSet names are dataset-only and must not reach the UI
    modes = {m["mode"]: m for m in body["modes"]}
    assert modes["semantic"]["available"] is True
    assert modes["spatial"]["available"] is False
    assert modes["sources"]["available"] is False
    assert body["device_profile"] == "laptop"


@pytest.mark.parametrize(
    ("method", "path", "step"),
    [
        ("post", "/api/session/start", "step-03"),
        ("post", "/api/session/stop", "step-03"),
        ("get", "/api/recordings/1", "step-03"),
        ("get", "/api/recordings/1/audio", "step-03"),
        ("post", "/api/recordings/1/process", "step-06"),
    ],
)
def test_stub_routes_return_501(client: TestClient, method: str, path: str, step: str) -> None:
    response = getattr(client, method)(path)
    assert response.status_code == 501
    assert response.json() == {"detail": f"not implemented ({step})"}


def test_put_target_returns_501(client: TestClient) -> None:
    response = client.put("/api/session/target", json={"labels": ["speech"]})
    assert response.status_code == 501


def test_openapi_lists_all_routes(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {
        "/api/health",
        "/api/config",
        "/api/devices",
        "/api/session/start",
        "/api/session/stop",
        "/api/session/target",
        "/api/recordings",
        "/api/recordings/{recording_id}",
        "/api/recordings/{recording_id}/audio",
        "/api/recordings/{recording_id}/process",
    } <= paths


def test_cors_allows_only_vite_dev_origin(client: TestClient) -> None:
    headers = {"Access-Control-Request-Method": "GET"}
    ok = client.options("/api/health", headers={**headers, "Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/api/health", headers={**headers, "Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


# --- WebSocket ---------------------------------------------------------------------------------


def test_ws_sends_initial_status(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        message = ws.receive_json()
    assert message["type"] == "status"
    assert message["t"] == 0.0
    assert message["payload"]["state"] == "idle"


def test_ws_heartbeat(app_config) -> None:
    import dataclasses

    fast_api = dataclasses.replace(app_config.api, ws_heartbeat_s=0.05)
    fast = dataclasses.replace(app_config, api=fast_api)
    with TestClient(create_app(fast)) as client, client.websocket_connect("/ws/session") as ws:
        assert ws.receive_json()["type"] == "status"
        assert ws.receive_json()["type"] == "status"  # heartbeat


@pytest.mark.parametrize(
    ("sent", "fragment"),
    [
        ('{"type": "bogus", "payload": {}}', "unknown message type"),
        ("not json", "invalid JSON"),
        ('{"type": "status", "payload": {}}', "unknown message type"),
        ('{"type": "set_target", "payload": {}}', "not implemented (step-03)"),
    ],
)
def test_ws_answers_client_messages_with_error(
    client: TestClient, sent: str, fragment: str
) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ws.receive_json()  # initial status
        ws.send_text(sent)
        reply = ws.receive_json()
    assert reply["type"] == "error"
    assert fragment in reply["payload"]["message"]

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


def test_no_route_is_a_stub_any_more(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    stubs = [f"{method} {path}" for path, methods in spec["paths"].items()
             for method, op in methods.items() if "501" in op.get("responses", {})]
    assert stubs == []


def test_openapi_lists_all_routes(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {
        "/api/health",
        "/api/config",
        "/api/devices",
        "/api/session",
        "/api/session/start",
        "/api/session/stop",
        "/api/session/target",
        "/api/session/monitor",
        "/api/session/record/start",
        "/api/session/record/stop",
        "/api/recordings",
        "/api/recordings/{recording_id}",
        "/api/recordings/{recording_id}/audio",
        "/api/recordings/{recording_id}/process",
        "/api/recordings/{recording_id}/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/audio",
        "/api/session/bypass",
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
        ('{"type": "set_target", "payload": {"labels": ["laughter"]}}', "unknown labels"),
        ('{"type": "set_target", "payload": {"attenuation_db": -3}}', "invalid set_target"),
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


def test_ws_set_target_is_stored(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        ws.send_text('{"type": "set_target", "payload": {"labels": ["speech"]}}')
        reply = ws.receive_json()
    assert reply["type"] == "status"
    sessions = client.app.state.session_manager
    assert sessions.target is not None and sessions.target.labels == {"speech"}

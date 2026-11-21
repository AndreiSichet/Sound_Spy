"""Step 08 API: the built page at ``/`` (SPA fallback), rescanning devices, the output device in
``data/user_settings.json``, the preloaded-models route, the rotating log file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from soundspy.api.app import create_app
from soundspy.api.static import mount_frontend
from soundspy.user_settings import (
    apply_user_settings,
    load_user_settings,
    save_user_settings,
    settings_path,
)

FILE_SOURCE = {"mode": "semantic", "source": {"type": "file", "path": "audio/tone.wav"}}


@pytest.fixture
def client(app_config):
    with TestClient(create_app(app_config)) as c:
        yield c


@pytest.fixture
def page(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Sound Spy</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('hi')", encoding="utf-8")
    return dist


def test_the_built_page_is_served_with_an_spa_fallback(app_config, page) -> None:
    with TestClient(create_app(app_config, frontend=page)) as client:
        assert "Sound Spy" in client.get("/").text
        assert client.get("/assets/app.js").text == "console.log('hi')"
        assert "Sound Spy" in client.get("/semantic/anything").text  # the page decides
        assert client.get("/api/health").json()["status"] == "ok"  # the API keeps precedence
        assert client.get("/api/no-such-route").status_code == 404


def test_a_missing_build_says_how_to_build(tmp_path: Path) -> None:
    from fastapi import FastAPI

    with pytest.raises(FileNotFoundError, match="npm run build"):
        mount_frontend(FastAPI(), tmp_path / "nothing")


def test_rescan_devices_only_when_idle(client: TestClient, fake_sounddevice) -> None:
    before = fake_sounddevice.refreshes
    body = client.post("/api/devices/rescan").json()
    assert fake_sounddevice.refreshes == before + 1
    assert body["selected_output"]["name"] == "Speakers (Realtek(R) Audio)"
    client.post("/api/session/start", json={"mode": "semantic"})
    response = client.post("/api/devices/rescan")
    assert response.status_code == 409 and "stop the session" in response.json()["detail"]
    client.post("/api/session/stop")


def test_the_output_device_is_stored_and_used(app_config, tmp_path: Path, start_running,
                                              no_leaked_threads) -> None:
    import dataclasses

    import numpy as np
    import soundfile as sf

    (tmp_path / "audio").mkdir()
    sf.write(str(tmp_path / "audio" / "tone.wav"), np.zeros((48000, 2), np.float32), 48000)
    config = dataclasses.replace(app_config, project_root=tmp_path)
    with TestClient(create_app(config)) as client:
        assert client.get("/api/settings").json() == {"output_device": None}
        bad = client.put("/api/settings/output-device", json={"device": "No Such Thing"})
        assert bad.status_code == 400
        saved = client.put("/api/settings/output-device", json={"device": "Headphones"}).json()
        assert saved == {"output_device": "Headphones"}
        devices = client.get("/api/devices").json()
        assert devices["selected_output"]["name"] == "Headphones (USB Audio)"
        status = start_running(client, FILE_SOURCE)
        assert status["output_device"] == "Headphones (USB Audio)"
        assert client.put("/api/settings/output-device", json={"device": None}).status_code == 409
        client.post("/api/session/stop")
    stored = json.loads(settings_path(config).read_text(encoding="utf-8"))
    assert stored == {"output_device": "Headphones"}
    with TestClient(create_app(config)) as client:  # a restart keeps the choice
        assert client.get("/api/devices").json()["selected_output"]["name"].startswith("Headph")


def test_user_settings_overlay(app_config) -> None:
    assert load_user_settings(app_config) == {}
    save_user_settings(app_config, output_device=8)
    assert apply_user_settings(app_config).output.device == 8
    settings_path(app_config).write_text('{"output_device": [1, 2], "other": 1}', "utf-8")
    assert apply_user_settings(app_config).output.device == app_config.output.device  # ignored
    settings_path(app_config).write_text("not json", encoding="utf-8")
    assert load_user_settings(app_config) == {}
    with pytest.raises(ValueError, match="unknown"):
        save_user_settings(app_config, colour="red")


def test_models_route_with_and_without_preloading(app_config) -> None:
    import dataclasses

    with TestClient(create_app(app_config)) as client:
        body = client.get("/api/models").json()
        assert body["state"] == "off" and "ram_mb" in body["resources"]
    preloading = dataclasses.replace(app_config, api=dataclasses.replace(app_config.api,
                                                                        preload_models=True))
    with TestClient(create_app(preloading)) as client:
        app = client.app
        assert app.state.preloader.wait(timeout=10)
        body = client.get("/api/models").json()
        assert body["state"] == "ready" and body["total"] == 0  # nothing to load in tests


def test_log_config_adds_a_rotating_file(app_config) -> None:
    from soundspy.cli.serve_cmd import log_config, log_path

    cfg = log_config(app_config)
    assert cfg["handlers"]["file"]["filename"] == str(log_path(app_config))
    assert log_path(app_config).parent.name == "logs"
    assert "file" in cfg["loggers"]["soundspy"]["handlers"]
    assert "file" in cfg["loggers"]["uvicorn.error"]["handlers"]
    assert log_config()["loggers"]["soundspy"]["handlers"] == ["default"]  # no config: console

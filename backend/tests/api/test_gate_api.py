"""Step 07b: the neural output gate as a session / run setting (tiny random model, CPU)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from soundspy.api.app import create_app
from soundspy.cli.main import main
from soundspy.core.audio import AudioBuffer
from soundspy.engine.factory import build_chain
from soundspy.extraction.offline import runs_dir
from soundspy.io.storage import save_recording

SR = 48000
FILE_SOURCE = {"mode": "SEMANTIC", "source": {"type": "file", "path": "audio/tone.wav"}}


def _config(app_config, neural_kit, tmp_path: Path, *, gate: bool = True):
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    (tmp_path / "audio").mkdir(exist_ok=True)
    sf.write(str(tmp_path / "audio" / "tone.wav"), np.stack([tone, tone], 1), SR, subtype="FLOAT")
    paths = {**app_config.paths, "raw": tmp_path / "data" / "raw"}
    config = neural_kit.make(dataclasses.replace(app_config, project_root=tmp_path, paths=paths),
                             tmp_path)
    neural = config.extraction.neural_tse
    return dataclasses.replace(config, extraction=dataclasses.replace(
        config.extraction, neural_tse=dataclasses.replace(
            neural, gate=dataclasses.replace(neural.gate, enabled=gate))))


def test_chain_follows_the_gate_switch(app_config, neural_kit, tmp_path) -> None:
    config = _config(app_config, neural_kit, tmp_path)
    assert build_chain("semantic", config, config.labels).stage("extractor") \
        .status_info()["gate"] is True
    off = build_chain("semantic", config, config.labels, gate=False).stage("extractor")
    assert off.status_info()["gate"] is False
    baseline = build_chain("semantic", config, config.labels, method="mask_baseline", gate=True)
    assert "gate" not in baseline.stage("extractor").status_info()  # the baseline ignores it


def test_session_gate_switch(app_config, neural_kit, tmp_path, no_leaked_threads) -> None:
    config = _config(app_config, neural_kit, tmp_path)
    with TestClient(create_app(config)) as client:
        assert client.get("/api/config").json()["default_gate"] is True
        assert client.get("/api/session").json()["extraction_gate"] is True
        body = client.put("/api/session/gate", json={"enabled": False}).json()
        assert body["extraction_gate"] is False and body["state"] == "idle"
        started = client.post("/api/session/start", json=FILE_SOURCE).json()
        assert started["extraction"]["gate"] is False and started["extraction_gate"] is False
        res = client.put("/api/session/gate", json={"enabled": True})  # not while running
        assert res.status_code == 409 and "stop the session" in res.json()["detail"]
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "set_gate", "t": 0, "payload": {"enabled": True}})
            for _ in range(500):
                message = ws.receive_json()
                if message["type"] == "error":
                    assert "set_gate: stop the session" in message["payload"]["message"]
                    break
            else:
                raise AssertionError("no error for set_gate during a session")
        client.post("/api/session/stop")
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "set_gate", "t": 0, "payload": {"enabled": True}})
            seen = []
            for _ in range(50):
                message = ws.receive_json()
                seen.append((message["type"], message["payload"].get("extraction_gate"),
                             message["payload"].get("message")))
                if message["type"] == "status" and message["payload"]["extraction_gate"]:
                    break
            else:
                raise AssertionError(f"set_gate was not answered with the new status: {seen}")
        started = client.post("/api/session/start", json={**FILE_SOURCE, "gate": False}).json()
        assert started["extraction"]["gate"] is False  # the start request overrides the choice
        client.post("/api/session/stop")


def test_runs_record_the_model_and_the_gate(app_config, neural_kit, tmp_path, monkeypatch
                                            ) -> None:
    config = _config(app_config, neural_kit, tmp_path)
    rng = np.random.default_rng(1)
    path = save_recording(AudioBuffer(rng.normal(0, 0.05, (SR, 2)).astype(np.float32), SR,
                                      {"device": "Mic"}), config.paths["raw"], label="g",
                          profile="laptop")
    with TestClient(create_app(config)) as client:
        rec = client.get("/api/recordings").json()[0]
        run = client.post(f"/api/recordings/{rec['id']}/process",
                          json={"labels": ["speech"], "method": "neural_tse", "gate": False})
        assert run.status_code == 202
        run_id = run.json()["run_id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            done = client.get(f"/api/runs/{run_id}").json()
            if done["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
    assert done["status"] == "done", done
    assert done["info"]["gate"] is False
    sidecar = json.loads((runs_dir(config) / f"{run_id}.json").read_text("utf-8"))
    params = sidecar["parameters"]
    assert params["model_name"] == "tse_tiny" and params["gate"]["enabled"] is False
    assert params["gate"]["depth_db"] == config.extraction.neural_tse.gate.depth_db

    # the CLI switch
    import soundspy.config

    monkeypatch.setattr(soundspy.config, "load_config", lambda *a, **k: config)
    assert main(["extract", str(path), "--labels", "speech", "--method", "neural_tse",
                 "--gate"]) == 0
    newest = max(runs_dir(config).glob("*.json"), key=lambda p: p.stat().st_mtime)
    assert json.loads(newest.read_text("utf-8"))["parameters"]["gate"]["enabled"] is True

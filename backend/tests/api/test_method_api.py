"""Step 07: choosing the extraction method (per session, per offline run), the fallback without
a neural model, the 15 dB default. The neural model here is the tiny random one (CPU)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from soundspy.api.app import create_app
from soundspy.cli.main import main
from soundspy.core.audio import AudioBuffer
from soundspy.engine.factory import build_chain
from soundspy.engine.processor import Passthrough
from soundspy.extraction.mask_baseline import MaskBaselineExtractor
from soundspy.extraction.neural_tse import NeuralTSEExtractor
from soundspy.extraction.offline import (
    extract_file,
    resolve_method,
    run_method,
    runs_dir,
)
from soundspy.io.storage import save_recording

SR = 48000
FILE_SOURCE = {"mode": "SEMANTIC", "source": {"type": "file", "path": "audio/tone.wav"}}


@pytest.fixture
def root_config(app_config, tmp_path: Path):
    """Project root = tmp folder, a tone file as session source, no models at all."""
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    (tmp_path / "audio").mkdir()
    sf.write(str(tmp_path / "audio" / "tone.wav"), np.stack([tone, tone], 1), SR, subtype="FLOAT")
    paths = {**app_config.paths, "raw": tmp_path / "data" / "raw"}
    return dataclasses.replace(app_config, project_root=tmp_path, paths=paths)


@pytest.fixture
def neural_config(root_config, neural_kit, tmp_path: Path):
    """``root_config`` + the tiny neural extractor as the configured method."""
    return neural_kit.make(root_config, tmp_path)


def no_checkpoint(config):
    """Method ``neural_tse`` (the shipped default) but its model file does not exist."""
    neural = dataclasses.replace(config.extraction.neural_tse,
                                 checkpoint=config.project_root / "experiments" / "x" / "best.pt")
    return dataclasses.replace(config, extraction=dataclasses.replace(
        config.extraction, method="neural_tse", neural_tse=neural))


def make_recording(config, seconds: float = 1.0) -> Path:
    rng = np.random.default_rng(1)
    buf = AudioBuffer(rng.normal(0, 0.05, (int(SR * seconds), 2)).astype(np.float32), SR,
                      {"device": "Mic"})
    return save_recording(buf, config.paths["raw"], label="method-test", profile="laptop")


def wait_for_run(client: TestClient, run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in ("done", "failed"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish")


# --- resolving the method ---------------------------------------------------------------------


def test_method_resolution_and_fallback(config, tiny_neural) -> None:
    assert resolve_method(tiny_neural) == ("neural_tse", None)
    assert resolve_method(tiny_neural, "mask_baseline") == ("mask_baseline", None)
    assert resolve_method(tiny_neural, "passthrough") == ("passthrough", None)
    method, warning = resolve_method(no_checkpoint(config))
    assert method == "mask_baseline"
    assert "Neural extraction is not available" in warning and "soundspy train extraction" in \
        warning and "baseline" in warning
    method, warning = resolve_method(config, "neural_tse")  # the fixture: no checkpoint set
    assert method == "mask_baseline" and "no checkpoint is configured" in warning
    with pytest.raises(ValueError, match="unknown extraction method"):
        resolve_method(config, "magic")
    assert run_method(tiny_neural) == "neural_tse"
    assert run_method(no_checkpoint(config)) == "mask_baseline"  # the default that can run
    assert run_method(no_checkpoint(config), "neural_tse") == "neural_tse"  # asked for: kept
    passthrough = dataclasses.replace(config, extraction=dataclasses.replace(
        config.extraction, method="passthrough"))
    assert run_method(passthrough) == "mask_baseline"  # a run always extracts something
    with pytest.raises(ValueError, match="unknown method"):
        run_method(config, "passthrough")


def test_chain_is_built_with_the_chosen_method(config, tiny_neural, tmp_path) -> None:
    chain = build_chain("semantic", tiny_neural, tiny_neural.labels)
    extractor = chain.stage("extractor")
    assert isinstance(extractor, NeuralTSEExtractor)
    # the neural method does not use the detector: no "gating unavailable" warning
    assert [code for code, _ in chain.warnings] == ["detector_unavailable"]
    assert chain.configure(SR, 2) == 1
    assert chain.latency_samples == extractor.latency_samples > 0
    chain.start()  # the model's warm-up, on the calling thread
    out, _events = chain.process(AudioBuffer(np.zeros((960, 2), dtype=np.float32), SR))
    assert out.samples.shape == (960, 1) and chain.dry.samples.shape == (960, 1)

    for method, kind in (("mask_baseline", MaskBaselineExtractor), ("passthrough", Passthrough)):
        other = build_chain("semantic", tiny_neural, tiny_neural.labels, method=method)
        assert isinstance(other.stage("extractor"), kind)

    # no model file -> the baseline, and the session is told why
    chain = build_chain("semantic", no_checkpoint(config), config.labels)
    assert isinstance(chain.stage("extractor"), MaskBaselineExtractor)
    codes = [code for code, _ in chain.warnings]
    assert codes == ["detector_unavailable", "extraction_fallback", "gating_unavailable"]
    message = dict(chain.warnings)["extraction_fallback"]
    assert "Neural extraction is not available" in message and "baseline" in message

    # a file that is not a checkpoint -> the same fallback instead of a failed session
    broken = no_checkpoint(config)
    path = tmp_path / "broken.pt"
    path.write_bytes(b"junk")
    broken = dataclasses.replace(broken, extraction=dataclasses.replace(
        broken.extraction, neural_tse=dataclasses.replace(broken.extraction.neural_tse,
                                                          checkpoint=path)))
    chain = build_chain("semantic", broken, config.labels)
    assert isinstance(chain.stage("extractor"), MaskBaselineExtractor)
    assert "extraction_fallback" in [code for code, _ in chain.warnings]


# --- the API ----------------------------------------------------------------------------------


def test_config_lists_the_methods(root_config, neural_config) -> None:
    with TestClient(create_app(neural_config)) as client:
        body = client.get("/api/config").json()
    assert body["default_method"] == "neural_tse" and body["default_attenuation_db"] == 15.0
    assert body["extraction_methods"] == [
        {"name": "neural_tse", "display_name": "Neural", "available": True, "reason": None},
        {"name": "mask_baseline", "display_name": "Baseline", "available": True,
         "reason": None}]
    with TestClient(create_app(no_checkpoint(root_config))) as client:
        body = client.get("/api/config").json()
        status = client.get("/api/session").json()
    assert body["default_method"] == "mask_baseline"  # what a session would really use
    neural = body["extraction_methods"][0]
    assert neural["available"] is False and "soundspy train extraction" in neural["reason"]
    assert status["extraction_method"] == "neural_tse"  # the configured choice


def test_neural_session_and_switching_between_sessions(neural_config, no_leaked_threads
                                                       ) -> None:
    with TestClient(create_app(neural_config)) as client:
        assert client.get("/api/session").json()["extraction_method"] == "neural_tse"
        # no attenuation given: the live default (15 dB), echoed back
        body = client.put("/api/session/target", json={"labels": ["speech"]}).json()
        assert body["attenuation_db"] == 15.0

        started = client.post("/api/session/start", json=FILE_SOURCE).json()
        info = started["extraction"]
        assert started["extraction_method"] == "neural_tse" and info["method"] == "neural_tse"
        assert info["device"] == "cpu" and info["model"] == "tse_tiny"
        assert info["gating"] is False and info["latency_ms"] > 0
        assert started["latency_ms"] >= 20.0 + info["latency_ms"]
        extractor = client.app.state.session_manager.session._chain.stage("extractor")
        assert isinstance(extractor, NeuralTSEExtractor)
        deadline = time.monotonic() + 5
        status = started
        while time.monotonic() < deadline and not (status["extraction"] or {}).get("model_ms"):
            time.sleep(0.05)
            status = client.get("/api/session").json()
        assert status["extraction"]["model_ms"] > 0  # the model's time per call is reported
        assert status["extraction"]["model_ms_p95"] >= status["extraction"]["model_ms"] * 0.5
        assert status["extraction"]["target_active"] is True
        assert extractor.target.attenuation_db == 15.0

        # not while a session runs: the methods have different delays
        res = client.put("/api/session/method", json={"method": "mask_baseline"})
        assert res.status_code == 409 and "stop the session" in res.json()["detail"]
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "set_method", "t": 0, "payload": {"method": "mask_baseline"}})
            for _ in range(500):
                message = ws.receive_json()
                if message["type"] == "error":
                    assert "set_method" in message["payload"]["message"]
                    break
            else:
                raise AssertionError("no error for set_method during a session")
        client.post("/api/session/stop")

        body = client.put("/api/session/method", json={"method": "mask_baseline"}).json()
        assert body["extraction_method"] == "mask_baseline" and body["state"] == "idle"
        started = client.post("/api/session/start", json=FILE_SOURCE).json()
        assert started["extraction"]["method"] == "mask_baseline"
        assert started["extraction"]["latency_ms"] == pytest.approx(21.33)
        client.post("/api/session/stop")

        # over the WebSocket while idle, and per start request
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "set_method", "t": 0, "payload": {"method": "neural_tse"}})
            for _ in range(100):
                message = ws.receive_json()
                if message["type"] == "status" and \
                        message["payload"]["extraction_method"] == "neural_tse":
                    break
            else:
                raise AssertionError("no status with the new method")
            ws.send_json({"type": "set_method", "t": 0, "payload": {"method": "magic"}})
            for _ in range(100):
                message = ws.receive_json()
                if message["type"] == "error":
                    break
            else:
                raise AssertionError("no error for an unknown method")
        started = client.post("/api/session/start",
                              json={**FILE_SOURCE, "method": "mask_baseline"}).json()
        assert started["extraction"]["method"] == "mask_baseline"  # this session only
        client.post("/api/session/stop")
        assert client.get("/api/session").json()["extraction_method"] == "neural_tse"
        assert client.put("/api/session/method", json={"method": "magic"}).status_code == 422
        assert client.put("/api/session/method", json={}).status_code == 422
        assert client.post("/api/session/start",
                           json={**FILE_SOURCE, "method": "magic"}).status_code == 422


def test_session_falls_back_to_the_baseline_with_a_warning(root_config, no_leaked_threads
                                                           ) -> None:
    with TestClient(create_app(no_checkpoint(root_config))) as client:
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            started = client.post("/api/session/start", json=FILE_SOURCE).json()
            assert started["extraction"]["method"] == "mask_baseline"  # what really runs
            assert started["extraction_method"] == "neural_tse"  # what was chosen
            warnings: dict[str, str] = {}
            for _ in range(500):
                message = ws.receive_json()
                if message["type"] == "warning":
                    warnings[message["payload"]["code"]] = message["payload"]["message"]
                if "extraction_fallback" in warnings:
                    break
        client.post("/api/session/stop")
    assert "Neural extraction is not available" in warnings["extraction_fallback"]
    assert "baseline" in warnings["extraction_fallback"]


# --- offline runs -----------------------------------------------------------------------------


def test_process_with_the_neural_method(neural_config, no_leaked_threads) -> None:
    make_recording(neural_config)
    with TestClient(create_app(neural_config)) as client:
        rec = client.get("/api/recordings").json()[0]
        url = f"/api/recordings/{rec['id']}/process"
        res = client.post(url, json={"labels": ["speech"]})  # defaults: neural, 15 dB
        assert res.status_code == 202, res.text
        run = res.json()
        assert run["method"] == "neural_tse" and run["attenuation_db"] == 15.0
        done = wait_for_run(client, run["run_id"])
        assert done["status"] == "done", done["error"]
        assert done["method"] == "neural_tse" and done["activity_mode"] == "none"
        sidecar = json.loads((runs_dir(neural_config) / f"{run['run_id']}.json")
                             .read_text("utf-8"))
        parameters = sidecar["parameters"]
        assert parameters["checkpoint"].replace("\\", "/").endswith(
            "experiments/tse_tiny/checkpoints/best.pt")
        assert parameters["model"]["sample_rate"] == 16000 and parameters["block_chunks"] == 1
        assert parameters["checkpoint_modified"]
        data, sr = sf.read(str(runs_dir(neural_config) / sidecar["file"]), always_2d=True)
        assert sr == SR and data.shape == (SR, 1)  # mono, aligned, same length

        # the baseline, asked for explicitly, needs the detection model (none in this config)
        res = client.post(url, json={"labels": ["speech"], "method": "mask_baseline"})
        assert res.status_code == 503 and "detection model" in res.json()["detail"]
        assert client.post(url, json={"labels": ["speech"],
                                      "method": "passthrough"}).status_code == 422
        listed = client.get(f"/api/recordings/{rec['id']}/runs").json()
        assert [r["method"] for r in listed] == ["neural_tse"]


def test_process_without_the_neural_model_is_503(root_config) -> None:
    config = no_checkpoint(root_config)
    make_recording(config)
    with TestClient(create_app(config)) as client:
        rec = client.get("/api/recordings").json()[0]
        res = client.post(f"/api/recordings/{rec['id']}/process",
                          json={"labels": ["speech"], "method": "neural_tse"})
        assert res.status_code == 503
        assert "neural extraction is not available" in res.json()["detail"]
        # no method given: the default that can run is the baseline, which needs the detector
        res = client.post(f"/api/recordings/{rec['id']}/process", json={"labels": ["speech"]})
        assert res.status_code == 503 and "detection model" in res.json()["detail"]


def test_extract_file_and_cli_with_the_neural_method(neural_config, monkeypatch, capsys,
                                                     tmp_path: Path) -> None:
    wav = make_recording(neural_config, 0.5)
    result = extract_file(neural_config, wav, ["speech"], method="neural_tse")
    assert result.audio.samples.shape == (SR // 2, 1) and result.activity is None
    assert result.info["method"] == "neural_tse" and result.info["model"] == "tse_tiny"
    assert result.info["device"] == "cpu" and result.info["activity_mode"] == "none"
    untouched = extract_file(neural_config, wav, ["speech"], method="neural_tse",
                             attenuation_db=0.0)
    mono = sf.read(str(wav), always_2d=True)[0].mean(axis=1)
    np.testing.assert_allclose(untouched.audio.samples.ravel(), mono, atol=1e-5)  # 0 dB = input
    assert not np.allclose(result.audio.samples.ravel(), mono, atol=1e-3)

    import soundspy.config

    monkeypatch.setattr(soundspy.config, "load_config", lambda *a, **k: neural_config)
    out = tmp_path / "neural.wav"
    assert main(["extract", str(wav), "--labels", "speech", "--method", "neural_tse",
                 "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "neural_tse" in printed and "-15 dB" in printed
    (sidecar,) = runs_dir(neural_config).glob("*.json")
    data = json.loads(sidecar.read_text("utf-8"))
    assert data["method"] == "neural_tse" and data["attenuation_db"] == 15.0
    assert sf.info(str(out)).frames == SR // 2
    with pytest.raises(SystemExit):  # argparse: not one of the offered methods
        main(["extract", str(wav), "--labels", "speech", "--method", "magic"])

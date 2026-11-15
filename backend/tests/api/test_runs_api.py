"""Offline processing runs: sidecars, the rebuildable index, the one-job worker, routes, CLI."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from soundspy.api.app import create_app
from soundspy.api.db import ProcessingRun, Recording, make_engine, make_session_factory
from soundspy.api.services.index import rebuild
from soundspy.cli.main import main
from soundspy.core.audio import AudioBuffer
from soundspy.extraction.offline import execute_run, runs_dir, start_run
from soundspy.io import runs as run_store
from soundspy.io.storage import save_recording

SR = 48000


@pytest.fixture
def run_config(app_config, stub_experiment, tmp_path: Path):
    """App config (tmp DB, project root = tmp) with the stub detection model."""
    stub_cfg, _exp = stub_experiment
    paths = {**app_config.paths, "raw": tmp_path / "data" / "raw",
             "outputs": tmp_path / "outputs"}
    return dataclasses.replace(app_config, project_root=tmp_path, paths=paths,
                               detection=stub_cfg.detection)


def make_recording(config, seconds: float = 2.0, label: str = "run-test") -> Path:
    rng = np.random.default_rng(1)
    buf = AudioBuffer(rng.normal(0, 0.05, (int(SR * seconds), 2)).astype(np.float32), SR,
                      {"device": "Mic"})
    return save_recording(buf, config.paths["raw"], label=label, profile="laptop")


def wait_for_run(client: TestClient, run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in ("done", "failed"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish")


def test_process_writes_a_run_with_sidecar_audio_and_ws_messages(run_config, no_leaked_threads
                                                                  ) -> None:
    wav = make_recording(run_config)
    with TestClient(create_app(run_config)) as client:
        rec = client.get("/api/recordings").json()[0]
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()  # the initial status
            res = client.post(f"/api/recordings/{rec['id']}/process",
                              json={"labels": ["speech", "music"], "attenuation_db": 20})
            assert res.status_code == 202, res.text
            run = res.json()
            assert run["status"] == "pending" and run["labels"] == ["music", "speech"]
            assert run["recording_id"] == rec["id"] and run["attenuation_db"] == 20.0
            statuses = []
            while "done" not in statuses and "failed" not in statuses:
                message = ws.receive_json()
                if message["type"] == "run":
                    assert message["payload"]["run_id"] == run["run_id"]
                    statuses.append(message["payload"]["status"])
        assert statuses[0] == "pending" and "running" in statuses and statuses[-1] == "done"

        done = wait_for_run(client, run["run_id"])
        assert done["status"] == "done" and done["error"] is None
        assert done["duration_s"] == pytest.approx(2.0) and done["processing_seconds"] > 0
        assert done["method"] == "mask_baseline" and done["activity_mode"] == "timeline"
        assert set(done["info"]["target_segments"]) == {"music", "speech"}

        # the sidecar is the source of truth, next to the processed audio
        directory = runs_dir(run_config)
        sidecar = json.loads((directory / f"{run['run_id']}.json").read_text("utf-8"))
        assert sidecar["status"] == "done" and sidecar["labels"] == ["music", "speech"]
        assert sidecar["recording"] == f"data/raw/{wav.name}"
        assert sidecar["parameters"]["n_fft"] == 1024 and sidecar["app_version"]
        assert sidecar["file"] == f"{run['run_id']}.wav"
        data, sr = sf.read(str(directory / sidecar["file"]), always_2d=True)
        assert sr == SR and data.shape == (2 * SR, 1)  # mono, same length = aligned for A/B
        assert done["output_path"] == f"outputs/processed/{run['run_id']}.wav"

        audio = client.get(f"/api/runs/{run['run_id']}/audio")
        assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
        listed = client.get(f"/api/recordings/{rec['id']}/runs").json()
        assert [r["run_id"] for r in listed] == [run["run_id"]]
        assert client.get("/api/runs/nope").status_code == 404
        assert client.get("/api/runs/nope/audio").status_code == 404
        assert client.get("/api/recordings/999/runs").status_code == 404


def test_process_request_validation(run_config) -> None:
    make_recording(run_config)
    with TestClient(create_app(run_config)) as client:
        rec = client.get("/api/recordings").json()[0]
        url = f"/api/recordings/{rec['id']}/process"
        assert client.post(url, json={"labels": []}).status_code == 422
        assert client.post(url, json={}).status_code == 422
        unknown = client.post(url, json={"labels": ["dragon"]})
        assert unknown.status_code == 422 and "dragon" in unknown.json()["detail"]
        assert client.post(url, json={"labels": ["speech"], "method": "magic"}).status_code == 422
        assert client.post(url, json={"labels": ["speech"],
                                      "attenuation_db": -3}).status_code == 422
        assert client.post("/api/recordings/999/process",
                           json={"labels": ["speech"]}).status_code == 404
        assert client.get(f"/api/recordings/{rec['id']}/runs").json() == []  # nothing was made
        assert list(runs_dir(run_config).glob("*.json")) == []


def test_long_recordings_are_413_and_missing_files_404(run_config) -> None:
    cfg = dataclasses.replace(run_config, extraction=dataclasses.replace(
        run_config.extraction, offline_max_seconds=1.0))
    wav = make_recording(cfg, 2.0)
    with TestClient(create_app(cfg)) as client:
        rec = client.get("/api/recordings").json()[0]
        url = f"/api/recordings/{rec['id']}/process"
        res = client.post(url, json={"labels": ["speech"]})
        assert res.status_code == 413 and "soundspy extract" in res.json()["detail"]
        wav.unlink()
        client.get("/api/recordings")  # re-index: now marked missing
        assert client.post(url, json={"labels": ["speech"]}).status_code == 404


def test_process_without_a_detection_model_is_503(app_config, tmp_path: Path) -> None:
    make_recording(app_config)
    with TestClient(create_app(app_config)) as client:
        rec = client.get("/api/recordings").json()[0]
        res = client.post(f"/api/recordings/{rec['id']}/process", json={"labels": ["speech"]})
    assert res.status_code == 503 and "detection model" in res.json()["detail"]


def test_only_one_job_runs_at_a_time(run_config, no_leaked_threads) -> None:
    make_recording(run_config)
    running: list[str] = []
    most = 0
    order: list[str] = []
    release = threading.Event()
    lock = threading.Lock()

    def slow_execute(config, run_id: str) -> dict:
        nonlocal most
        with lock:
            running.append(run_id)
            most = max(most, len(running))
            order.append(run_id)
        release.wait(5.0)
        with lock:
            running.remove(run_id)
        return run_store.update_run(runs_dir(config), run_id, status="done")

    with TestClient(create_app(run_config)) as client:
        client.app.state.run_manager._execute = slow_execute
        rec = client.get("/api/recordings").json()[0]
        url = f"/api/recordings/{rec['id']}/process"
        ids = [client.post(url, json={"labels": [lab]}).json()["run_id"]
               for lab in ("speech", "music", "alarm")]
        assert len(set(ids)) == 3
        deadline = time.monotonic() + 5
        while not order and time.monotonic() < deadline:
            time.sleep(0.01)
        states = {r["run_id"]: r["status"]
                  for r in client.get(f"/api/recordings/{rec['id']}/runs").json()}
        assert states[ids[0]] == "running"  # the first is being processed ...
        assert states[ids[1]] == states[ids[2]] == "pending"  # ... the others wait their turn
        assert client.get(f"/api/runs/{ids[1]}/audio").status_code == 409  # nothing to play yet
        release.set()
        for run_id in ids:
            assert wait_for_run(client, run_id)["status"] == "done"
    assert most == 1 and order == ids  # never two at once, in submission order


def test_a_failing_job_is_recorded_and_the_worker_goes_on(run_config, no_leaked_threads) -> None:
    make_recording(run_config)

    def execute(config, run_id: str) -> dict:
        if run_store.load_run(runs_dir(config), run_id)["labels"] == ["music"]:
            raise RuntimeError("boom")
        return execute_run(config, run_id)

    with TestClient(create_app(run_config)) as client:
        client.app.state.run_manager._execute = execute
        rec = client.get("/api/recordings").json()[0]
        url = f"/api/recordings/{rec['id']}/process"
        bad = client.post(url, json={"labels": ["music"]}).json()["run_id"]
        good = client.post(url, json={"labels": ["speech"]}).json()["run_id"]
        failed = wait_for_run(client, bad)
        assert failed["status"] == "failed" and failed["error"] == "RuntimeError: boom"
        assert client.get(f"/api/runs/{bad}/audio").status_code == 409
        assert wait_for_run(client, good)["status"] == "done"
    assert run_store.load_run(runs_dir(run_config), bad)["status"] == "failed"


def test_indexing_from_two_threads_at_once_is_safe(run_config) -> None:
    """The worker and request threads both re-index; neither may insert a row twice."""
    from soundspy.api.db import init_db
    from soundspy.api.services.index import sync_all

    wav = make_recording(run_config)
    for label in ("speech", "music", "alarm", "traffic"):
        start_run(run_config, wav, [label])
    engine = make_engine(run_config.database_url, run_config.project_root)
    init_db(engine)
    factory = make_session_factory(engine)
    errors: list[Exception] = []
    barrier = threading.Barrier(6)

    def index() -> None:
        try:
            barrier.wait(5.0)
            for _ in range(5):
                with factory() as db:
                    sync_all(db, run_config)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=index) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30.0)
    assert errors == []
    with factory() as db:
        assert len(db.scalars(select(ProcessingRun)).all()) == 4
        assert len(db.scalars(select(Recording)).all()) == 1
    engine.dispose()


def snapshot(config) -> tuple[list, list]:
    engine = make_engine(config.database_url, config.project_root)
    with make_session_factory(engine)() as db:
        recordings = [(r.path, r.label, r.duration_s, r.missing)
                      for r in db.scalars(select(Recording).order_by(Recording.path))]
        runs = [(r.run_id, r.recording_path, tuple(r.labels), r.status, r.output_path,
                 r.recording.path if r.recording else None)
                for r in db.scalars(select(ProcessingRun).order_by(ProcessingRun.run_id))]
    engine.dispose()
    return recordings, runs


def test_the_database_is_rebuilt_from_the_sidecars(run_config, no_leaked_threads) -> None:
    make_recording(run_config, label="a")
    make_recording(run_config, 1.5, label="b")
    with TestClient(create_app(run_config)) as client:
        rows = {r["label"]: r["id"] for r in client.get("/api/recordings").json()}
        run = client.post(f"/api/recordings/{rows['a']}/process",
                          json={"labels": ["speech"]}).json()
        wait_for_run(client, run["run_id"])
    before = snapshot(run_config)
    assert len(before[0]) == 2 and len(before[1]) == 1 and before[1][0][3] == "done"

    engine = make_engine(run_config.database_url, run_config.project_root)
    result = rebuild(engine, run_config)  # what `soundspy db rebuild` does
    engine.dispose()
    assert (result.recordings.added, result.runs.added) == (2, 1)
    assert snapshot(run_config) == before  # nothing was only in the database

    # a schema-version mismatch triggers the same rebuild when the app starts
    engine = make_engine(run_config.database_url, run_config.project_root)
    with engine.begin() as connection:
        connection.execute(text("UPDATE meta SET value = '1'"))
        connection.execute(text("DELETE FROM processing_runs"))
        connection.execute(text("DELETE FROM recordings"))
    engine.dispose()
    with TestClient(create_app(run_config)) as client:
        assert len(client.get("/api/recordings").json()) == 2
        assert client.get(f"/api/runs/{run['run_id']}").json()["status"] == "done"
    assert snapshot(run_config) == before


def test_runs_survive_their_recording(run_config, no_leaked_threads) -> None:
    wav = make_recording(run_config)
    with TestClient(create_app(run_config)) as client:
        rec = client.get("/api/recordings").json()[0]
        run = client.post(f"/api/recordings/{rec['id']}/process",
                          json={"labels": ["speech"]}).json()
        wait_for_run(client, run["run_id"])
        wav.unlink()  # WAV gone, sidecar kept: the recording is "missing", its runs stay listed
        assert client.get("/api/recordings").json()[0]["missing"] is True
        assert len(client.get(f"/api/recordings/{rec['id']}/runs").json()) == 1
        wav.with_suffix(".json").unlink()  # nothing describes the recording any more
        assert client.get("/api/recordings").json() == []
        orphan = client.get(f"/api/runs/{run['run_id']}").json()
        assert orphan["recording_id"] is None and orphan["recording_path"].endswith(wav.name)
        assert client.get(f"/api/runs/{run['run_id']}/audio").status_code == 200
        # deleting the run's sidecar removes it from the index too
        (runs_dir(run_config) / f"{run['run_id']}.json").unlink()
        client.get("/api/recordings")
        assert client.get(f"/api/runs/{run['run_id']}").status_code == 404


def test_runs_left_unfinished_are_marked_interrupted_at_start(run_config) -> None:
    wav = make_recording(run_config)
    left = start_run(run_config, wav, ["speech"])  # e.g. queued when the app was closed
    assert left["status"] == "pending"
    with TestClient(create_app(run_config)) as client:
        run = client.get(f"/api/runs/{left['run_id']}").json()
    assert run["status"] == "failed" and "interrupted" in run["error"]
    assert run_store.load_run(runs_dir(run_config), left["run_id"])["status"] == "failed"


def test_run_store_basics(tmp_path: Path) -> None:
    kwargs = dict(recording="data/raw/a.wav", labels=["speech"], attenuation_db=30,
                  method="mask_baseline", parameters={}, activity_mode="timeline")
    first = run_store.create_run(tmp_path, **kwargs)
    second = run_store.create_run(tmp_path, **kwargs)
    assert first["run_id"] != second["run_id"] and first["run_id"].startswith("run_")
    with pytest.raises(ValueError):
        run_store.update_run(tmp_path, first["run_id"], status="maybe")
    with pytest.raises(run_store.RunNotFoundError):
        run_store.load_run(tmp_path, "run_nope")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "other.json").write_text('{"hello": 1}', encoding="utf-8")
    assert {d["run_id"] for _, d in run_store.iter_runs(tmp_path)} == {
        first["run_id"], second["run_id"]}  # the broken and the foreign JSON are skipped
    assert run_store.interrupt_unfinished(tmp_path) == 2
    assert run_store.interrupt_unfinished(tmp_path) == 0
    assert not list(tmp_path.glob("*.tmp"))  # sidecars are replaced atomically


def test_sidecars_can_be_read_while_they_are_rewritten(tmp_path: Path) -> None:
    """The worker rewrites a run's sidecar while requests read it (Windows locks open files)."""
    run = run_store.create_run(tmp_path, recording="data/raw/a.wav", labels=["speech"],
                               attenuation_db=30, method="mask_baseline", parameters={},
                               activity_mode="timeline")
    errors: list[Exception] = []
    stop = threading.Event()

    def write() -> None:
        try:
            for i in range(300):
                run_store.update_run(tmp_path, run["run_id"], info={"i": i})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    def read() -> None:
        try:
            while not stop.is_set():
                assert [d["run_id"] for _, d in run_store.iter_runs(tmp_path)] == [run["run_id"]]
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write), *(threading.Thread(target=read) for _ in range(3))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30.0)
    assert errors == []
    assert run_store.load_run(tmp_path, run["run_id"])["info"] == {"i": 299}


def test_an_unreadable_sidecar_keeps_its_index_row(run_config, monkeypatch) -> None:
    from soundspy.api.db import init_db
    from soundspy.api.services.index import sync_all

    wav = make_recording(run_config)
    run = start_run(run_config, wav, ["speech"])
    engine = make_engine(run_config.database_url, run_config.project_root)
    init_db(engine)
    with make_session_factory(engine)() as db:
        assert sync_all(db, run_config).runs.added == 1
        monkeypatch.setattr(run_store, "iter_runs", lambda directory: iter(()))  # "locked"
        result = sync_all(db, run_config)
        assert result.runs.removed == 0  # the file is still there: not forgotten
        assert db.scalars(select(ProcessingRun)).one().run_id == run["run_id"]
    engine.dispose()


def test_execute_run_records_failures_instead_of_raising(run_config) -> None:
    wav = make_recording(run_config)
    run = start_run(run_config, wav, ["speech"])
    wav.unlink()
    done = execute_run(run_config, run["run_id"])
    assert done["status"] == "failed" and "recording file is gone" in done["error"]
    with pytest.raises(ValueError, match="at least one sound class"):
        start_run(run_config, wav, [])
    with pytest.raises(ValueError, match="unknown method"):
        start_run(run_config, wav, ["speech"], method="passthrough")


def test_cli_extract_and_db_rebuild(run_config, monkeypatch, capsys, tmp_path: Path) -> None:
    import soundspy.config

    monkeypatch.setattr(soundspy.config, "load_config", lambda *a, **k: run_config)
    wav = make_recording(run_config)
    out = tmp_path / "copies" / "speech_only.wav"
    assert main(["extract", str(wav), "--labels", "speech", "--attenuation-db", "24",
                 "--out", str(out), "--device", "cpu"]) == 0
    printed = capsys.readouterr().out
    assert "speech from" in printed and "-24 dB" in printed and str(out) in printed
    (sidecar,) = runs_dir(run_config).glob("*.json")
    data = json.loads(sidecar.read_text("utf-8"))
    assert data["status"] == "done" and data["attenuation_db"] == 24.0
    assert sf.info(str(out)).frames == sf.info(str(wav)).frames
    assert sf.info(str(sidecar.with_suffix(".wav"))).channels == 1

    assert main(["extract", str(wav), "--labels", "dragon"]) == 1
    assert "dragon" in capsys.readouterr().out
    assert main(["extract", str(tmp_path / "nope.wav"), "--labels", "speech"]) == 1

    assert main(["db", "rebuild"]) == 0  # the CLI run is indexed from its sidecar
    assert "1 recording(s) (0 with a missing file), 1 processing run(s)" in capsys.readouterr().out
    recordings, runs = snapshot(run_config)
    assert len(recordings) == 1 and runs[0][0] == data["run_id"] and runs[0][5] is not None

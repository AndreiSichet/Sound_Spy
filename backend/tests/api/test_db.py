"""Database tests against a temporary SQLite file."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from soundspy.api.db import (
    ProcessingRun,
    Recording,
    init_db,
    make_engine,
    make_session_factory,
)
from soundspy.api.db.base import resolve_database_url


@pytest.fixture
def session_factory(tmp_path: Path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}", tmp_path)
    init_db(engine)
    yield make_session_factory(engine)
    engine.dispose()


def _recording(**overrides) -> Recording:
    values = dict(
        path="data/raw/2026-11-14_120000.wav",
        duration_s=5.0,
        sample_rate=48000,
        channels=2,
        device="Microphone Array (Realtek)",
        profile="laptop",
    )
    values.update(overrides)
    return Recording(**values)


def test_tables_created(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{(tmp_path / 'x.db').as_posix()}", tmp_path)
    init_db(engine)
    assert set(inspect(engine).get_table_names()) == {"recordings", "processing_runs"}
    engine.dispose()


def test_insert_and_read_recording(session_factory) -> None:
    with session_factory() as session:
        session.add(_recording(label="speech"))
        session.commit()
    with session_factory() as session:
        rec = session.scalars(select(Recording)).one()
    assert rec.id == 1 and rec.label == "speech" and rec.notes == ""
    assert rec.created_at is not None and rec.channels == 2


def test_recording_path_unique(session_factory) -> None:
    with session_factory() as session:
        session.add_all([_recording(), _recording()])
        with pytest.raises(IntegrityError):
            session.commit()


def test_processing_run_links_to_recording(session_factory) -> None:
    with session_factory() as session:
        rec = _recording()
        target = {"labels": ["speech"]}
        rec.runs.append(ProcessingRun(mode="semantic", target_json=target, method="mask_baseline"))
        session.add(rec)
        session.commit()
    with session_factory() as session:
        run = session.scalars(select(ProcessingRun)).one()
        assert run.status == "pending" and run.recording.profile == "laptop"
        assert run.target_json == {"labels": ["speech"]} and run.metrics_json is None


def test_processing_run_rejects_unknown_method(session_factory) -> None:
    with session_factory() as session:
        rec = _recording()
        session.add(rec)
        session.flush()
        session.add(ProcessingRun(recording_id=rec.id, mode="semantic", target_json={}, method="x"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_foreign_keys_enforced(session_factory) -> None:
    with session_factory() as session:
        session.add(
            ProcessingRun(recording_id=999, mode="semantic", target_json={}, method="neural_tse")
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_relative_sqlite_url_resolves_to_project_root(tmp_path: Path) -> None:
    url = resolve_database_url("sqlite:///data/soundspy.db", tmp_path)
    assert Path(url.database) == (tmp_path / "data" / "soundspy.db").resolve()
    assert resolve_database_url("sqlite://", tmp_path).database is None


def test_engine_creates_db_file_at_resolved_path(tmp_path: Path) -> None:
    engine = make_engine("sqlite:///nested/dir/app.db", tmp_path)
    init_db(engine)
    engine.dispose()
    assert (tmp_path / "nested" / "dir" / "app.db").is_file()

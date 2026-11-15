"""Database tests against a temporary SQLite file."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from soundspy.api.db import (
    SCHEMA_VERSION,
    ProcessingRun,
    Recording,
    init_db,
    make_engine,
    make_session_factory,
)
from soundspy.api.db.base import resolve_database_url, stored_schema_version


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
    assert set(inspect(engine).get_table_names()) == {"recordings", "processing_runs", "meta"}
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


def _run(**overrides) -> ProcessingRun:
    values = dict(run_id="run_20261116-100000-000", recording_path="data/raw/x.wav",
                  labels=["speech"], attenuation_db=30.0, method="mask_baseline")
    values.update(overrides)
    return ProcessingRun(**values)


def test_processing_run_links_to_recording(session_factory) -> None:
    with session_factory() as session:
        rec = _recording()
        rec.runs.append(_run())
        session.add(rec)
        session.commit()
    with session_factory() as session:
        run = session.scalars(select(ProcessingRun)).one()
        assert run.status == "pending" and run.recording.profile == "laptop"
        assert run.labels == ["speech"] and run.info == {} and run.output_path is None
        assert run.recording.missing is False


def test_run_ids_are_unique_and_recordings_may_be_absent(session_factory) -> None:
    with session_factory() as session:
        session.add(_run())  # no recording row: allowed (its files may be gone)
        session.commit()
        session.add(_run())
        with pytest.raises(IntegrityError):
            session.commit()


def test_foreign_keys_enforced(session_factory) -> None:
    with session_factory() as session:
        session.add(_run(recording_id=999))
        with pytest.raises(IntegrityError):
            session.commit()


def test_schema_version_is_stored(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{(tmp_path / 'v.db').as_posix()}", tmp_path)
    assert stored_schema_version(engine) is None
    assert init_db(engine) is True  # a new database is built ...
    assert stored_schema_version(engine) == SCHEMA_VERSION
    assert init_db(engine) is False  # ... and left alone when the version matches
    engine.dispose()


@pytest.mark.parametrize("stored", ["1", "999", "not-a-number", None])
def test_schema_version_mismatch_triggers_a_rebuild(tmp_path: Path, stored) -> None:
    engine = make_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}", tmp_path)
    init_db(engine)
    factory = make_session_factory(engine)
    with factory() as session:
        session.add(_recording())
        session.execute(text("CREATE TABLE left_over_from_an_old_schema (x INTEGER)"))
        if stored is None:
            session.execute(text("DELETE FROM meta"))
        else:
            session.execute(text("UPDATE meta SET value = :v"), {"v": stored})
        session.commit()
    assert init_db(engine) is True
    assert stored_schema_version(engine) == SCHEMA_VERSION
    assert set(inspect(engine).get_table_names()) == {"recordings", "processing_runs", "meta"}
    with factory() as session:  # the index is empty again; the caller re-reads the sidecars
        assert session.scalars(select(Recording)).all() == []
    engine.dispose()


def test_pre_versioning_database_is_rebuilt(tmp_path: Path) -> None:
    """The step 01-05 database: tables but no ``meta``, and an older ``recordings`` layout."""
    engine = make_engine(f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}", tmp_path)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE recordings (id INTEGER PRIMARY KEY, path TEXT)"))
        connection.execute(text("INSERT INTO recordings (path) VALUES ('data/raw/a.wav')"))
    assert init_db(engine) is True
    with make_session_factory(engine)() as session:
        session.add(_recording(missing=True))  # the new column exists
        session.commit()
    engine.dispose()


def test_relative_sqlite_url_resolves_to_project_root(tmp_path: Path) -> None:
    url = resolve_database_url("sqlite:///data/soundspy.db", tmp_path)
    assert Path(url.database) == (tmp_path / "data" / "soundspy.db").resolve()
    assert resolve_database_url("sqlite://", tmp_path).database is None


def test_engine_creates_db_file_at_resolved_path(tmp_path: Path) -> None:
    engine = make_engine("sqlite:///nested/dir/app.db", tmp_path)
    init_db(engine)
    engine.dispose()
    assert (tmp_path / "nested" / "dir" / "app.db").is_file()

"""Database engine, session factory, declarative ``Base`` and ``init_db()``.

There is no module-level global engine: ``create_app`` builds one per app instance from the
configured URL, so tests can point each app at its own temporary SQLite file.
Relative SQLite paths (``sqlite:///data/soundspy.db``) are resolved against the project root.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import URL, Engine, MetaData, create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def _is_sqlite_file(url: URL) -> bool:
    return url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:")


def resolve_database_url(url: str, project_root: Path) -> URL:
    """Parse ``url``; a relative SQLite file path is made absolute against ``project_root``.

    Returns a ``URL`` object (not a string): rendering a Windows path back to a string
    percent-encodes it.
    """
    parsed = make_url(url)
    if not _is_sqlite_file(parsed):
        return parsed
    db_path = Path(parsed.database)
    if not db_path.is_absolute():
        db_path = (project_root / db_path).resolve()
    return parsed.set(database=str(db_path))


def make_engine(url: str, project_root: Path) -> Engine:
    resolved = resolve_database_url(url, project_root)
    is_sqlite = resolved.get_backend_name() == "sqlite"
    if _is_sqlite_file(resolved):
        Path(resolved.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        resolved,
        connect_args={"check_same_thread": False} if is_sqlite else {},
    )
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record) -> None:  # pragma: no cover
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def stored_schema_version(engine: Engine) -> int | None:
    """The version recorded in the database; ``None`` if it has none (new, or pre-versioning)."""
    if "meta" not in inspect(engine).get_table_names():
        return None
    with engine.connect() as connection:
        value = connection.execute(
            text("SELECT value FROM meta WHERE key = 'schema_version'")).scalar()
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def rebuild_db(engine: Engine) -> None:
    """Drop **every** table and create the current schema (empty; re-sync from the sidecars).

    Safe because the database only indexes sidecar files. Tables are found by reflection, so
    tables of an older schema that the code no longer knows are dropped too.
    """
    from soundspy.api.db import models

    old = MetaData()
    old.reflect(engine)
    old.drop_all(engine)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(models.Meta(key="schema_version", value=str(models.SCHEMA_VERSION)))
        session.commit()


def init_db(engine: Engine) -> bool:
    """Make the database match the code's schema. Returns True if it was (re)built.

    No migrations: an empty database gets the tables; a database whose stored schema version
    differs from ``SCHEMA_VERSION`` (or that has tables but no version) is dropped and recreated.
    The caller then re-indexes the sidecars (``services/index.py``).
    """
    from soundspy.api.db import models

    if stored_schema_version(engine) == models.SCHEMA_VERSION:
        Base.metadata.create_all(engine)  # no-op unless a table was removed by hand
        return False
    rebuild_db(engine)
    return True

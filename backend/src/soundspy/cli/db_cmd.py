"""``soundspy db``: the app database.

    soundspy db rebuild

The database (``data/soundspy.db``) is a **disposable index** of sidecar files: every recording
has a ``.json`` next to its WAV in ``data/raw``, every processing run one in
``outputs/processed``. ``rebuild`` drops all tables, creates the current schema and indexes the
sidecars again. Nothing is lost, because nothing is stored only in the database. The API does
the same by itself at start-up when the stored schema version is not the code's, so this command
is only needed to force it (stop ``soundspy serve`` first: SQLite allows one writer).

This module and ``serve_cmd`` are the CLI's two entry points into ``soundspy.api``; the import
happens inside ``run`` so the rest of the CLI works without the ``api`` extra.
"""

from __future__ import annotations

import argparse

NAME = "db"
HELP = "the app database (an index of the sidecar files): rebuild"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("action", choices=("rebuild",))
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from sqlalchemy.exc import SQLAlchemyError

    from soundspy.api.db.base import make_engine
    from soundspy.api.db.models import SCHEMA_VERSION
    from soundspy.api.services.index import rebuild
    from soundspy.config import ConfigError, load_config

    try:
        config = load_config()
        engine = make_engine(config.database_url, config.project_root)
        try:
            result = rebuild(engine, config)
        finally:
            engine.dispose()
    except (ConfigError, SQLAlchemyError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    rec, runs = result.recordings, result.runs
    print(f"database rebuilt (schema version {SCHEMA_VERSION}): {rec.added} recording(s) "
          f"({rec.missing} with a missing file), {runs.added} processing run(s)")
    return 0

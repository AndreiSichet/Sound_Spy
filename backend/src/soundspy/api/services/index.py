"""The whole index: recordings and processing runs, from their sidecar files.

The database stores nothing that is not in a sidecar, so it can be thrown away at any time:

* ``sync_all`` brings both tables in step with the files (API start-up, listing routes);
* ``rebuild`` drops every table, recreates the schema and indexes everything again
  (``soundspy db rebuild``; also what happens automatically at start-up when the schema version
  stored in the database differs from the code's, see ``db/base.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from soundspy.api.db.base import make_session_factory, rebuild_db
from soundspy.api.services.recordings import SyncResult, sync_recordings
from soundspy.api.services.runs import RunSyncResult, sync_runs
from soundspy.config import AppConfig
from soundspy.extraction.offline import runs_dir


@dataclass(frozen=True)
class IndexResult:
    recordings: SyncResult
    runs: RunSyncResult


def sync_all(db: Session, config: AppConfig) -> IndexResult:
    """Recordings first: runs are linked to their recording's row."""
    recordings = sync_recordings(db, config.paths["raw"], config.project_root)
    runs = sync_runs(db, runs_dir(config), config.project_root)
    return IndexResult(recordings, runs)


def rebuild(engine: Engine, config: AppConfig) -> IndexResult:
    rebuild_db(engine)
    with make_session_factory(engine)() as db:
        return sync_all(db, config)

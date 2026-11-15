"""App database (SQLite via SQLAlchemy 2.x): an index of recordings and processing runs.

Audio files and their JSON sidecars stay on disk and are the source of truth; the database only
indexes them and is rebuilt from them whenever its schema version differs from the code's.
"""

from soundspy.api.db.base import Base, init_db, make_engine, make_session_factory, rebuild_db
from soundspy.api.db.models import SCHEMA_VERSION, Meta, ProcessingRun, Recording

__all__ = ["SCHEMA_VERSION", "Base", "Meta", "ProcessingRun", "Recording", "init_db",
           "make_engine", "make_session_factory", "rebuild_db"]

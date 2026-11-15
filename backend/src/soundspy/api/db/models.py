"""Database tables: ``Recording``, ``ProcessingRun`` and ``Meta``.

The database is a **disposable index** of the sidecar files on disk (``data/raw/*.json`` for
recordings, ``outputs/processed/*.json`` for processing runs). Nothing is stored only here, so a
schema change needs no migration: ``SCHEMA_VERSION`` is bumped, and the next start drops the
tables, recreates them and re-reads the sidecars (``db/base.py``, ``services/index.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from soundspy.api.db.base import Base

#: Bump whenever a table changes. A database with another version is rebuilt from the sidecars.
SCHEMA_VERSION = 2


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Meta(Base):
    """Key/value facts about the database itself (``schema_version``)."""

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[str] = mapped_column(String(255))


class Recording(Base):
    """A raw recording on disk (WAV + JSON sidecar). Immutable once written.

    ``missing`` is true while the sidecar exists but the WAV does not (moved or deleted): the row
    and its runs stay visible, but it cannot be played or processed.
    """

    __tablename__ = "recordings"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    path: Mapped[str] = mapped_column(String(1024), unique=True)
    duration_s: Mapped[float]
    sample_rate: Mapped[int]
    channels: Mapped[int]
    device: Mapped[str] = mapped_column(String(255))
    profile: Mapped[str] = mapped_column(String(100))
    label: Mapped[str | None] = mapped_column(String(100), default=None)
    notes: Mapped[str] = mapped_column(Text, default="")
    missing: Mapped[bool] = mapped_column(default=False)

    runs: Mapped[list[ProcessingRun]] = relationship(back_populates="recording")


class ProcessingRun(Base):
    """Index entry of one processing run (``outputs/processed/<run_id>.json``)."""

    __tablename__ = "processing_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # NULL when the recording is not indexed (its files are gone); the path is kept regardless
    recording_id: Mapped[int | None] = mapped_column(
        ForeignKey("recordings.id", ondelete="SET NULL"), default=None)
    recording_path: Mapped[str] = mapped_column(String(1024))
    labels: Mapped[list[str]] = mapped_column(JSON)
    attenuation_db: Mapped[float]
    method: Mapped[str] = mapped_column(String(40))
    activity_mode: Mapped[str] = mapped_column(String(20), default="")
    status: Mapped[str] = mapped_column(String(10), default="pending")
    error: Mapped[str | None] = mapped_column(Text, default=None)
    output_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    duration_s: Mapped[float | None] = mapped_column(default=None)
    processing_seconds: Mapped[float | None] = mapped_column(default=None)
    info: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    recording: Mapped[Recording | None] = relationship(back_populates="runs")

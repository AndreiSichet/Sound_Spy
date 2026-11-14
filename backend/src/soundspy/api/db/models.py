"""Database tables: ``Recording`` and ``ProcessingRun``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from soundspy.api.db.base import Base

METHODS = ("mask_baseline", "neural_tse")
STATUSES = ("pending", "done", "failed")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Recording(Base):
    """A raw recording on disk (WAV + JSON sidecar). Immutable once written."""

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

    runs: Mapped[list[ProcessingRun]] = relationship(
        back_populates="recording", cascade="all, delete-orphan"
    )


class ProcessingRun(Base):
    """One offline processing of a recording (target, method, output, metrics)."""

    __tablename__ = "processing_runs"
    __table_args__ = (
        CheckConstraint(_in("method", METHODS), name="ck_processing_runs_method"),
        CheckConstraint(_in("status", STATUSES), name="ck_processing_runs_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    recording_id: Mapped[int] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"))
    mode: Mapped[str] = mapped_column(String(20))
    target_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    method: Mapped[str] = mapped_column(String(20))
    output_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    status: Mapped[str] = mapped_column(String(10), default="pending")
    error: Mapped[str | None] = mapped_column(Text, default=None)

    recording: Mapped[Recording] = relationship(back_populates="runs")

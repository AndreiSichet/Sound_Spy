"""Processing runs in the API: the index of run sidecars and the background worker.

**Index.** ``sync_runs`` makes the ``processing_runs`` table equal to the run sidecars in
``outputs/processed`` (the source of truth, ``io/runs.py``): new sidecars are inserted, changed
ones updated, rows without a sidecar deleted. A run is linked to its recording by path.

**Worker.** ``RunManager`` owns one thread (``soundspy-runs``) and a queue: only one offline job
runs at a time, the rest wait in order. ``submit`` writes the ``pending`` sidecar, indexes it and
queues it. The worker executes it with ``extraction.offline.execute_run`` (which rewrites the
sidecar: ``running``, then ``done`` or ``failed``), re-indexes it, and publishes every state
change to the WebSocket clients as a ``run`` message, so the page updates without polling.
The thread -> event-loop hand-over is the same as for session events
(``loop.call_soon_threadsafe``).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from soundspy.api.db.models import ProcessingRun, Recording
from soundspy.api.schemas import RunModel, WsMessage
from soundspy.api.services.recordings import (
    INDEX_LOCK,
    absolute_path,
    created_at,
    stored_path,
)
from soundspy.api.ws import ConnectionManager
from soundspy.config import AppConfig
from soundspy.extraction import offline
from soundspy.io import runs as run_store

logger = logging.getLogger("soundspy.api")

THREAD_NAME = "soundspy-runs"


@dataclass(frozen=True)
class RunSyncResult:
    added: int
    updated: int
    removed: int


def _apply(row: ProcessingRun, data: dict[str, Any], sidecar: Path, project_root: Path,
           recording_ids: dict[str, int]) -> bool:
    """Copy a sidecar into its row; True if anything changed."""
    output = None
    if data.get("file"):
        output = stored_path(sidecar.with_name(str(data["file"])), project_root)
    recording = str(data.get("recording") or "")
    values = {
        "created_at": created_at(data, sidecar).replace(tzinfo=None),
        "recording_id": recording_ids.get(recording),
        "recording_path": recording,
        "labels": list(data.get("labels") or []),
        "attenuation_db": float(data.get("attenuation_db") or 0.0),
        "method": str(data.get("method") or ""),
        "activity_mode": str(data.get("activity_mode") or ""),
        "status": str(data.get("status") or "failed"),
        "error": data.get("error"),
        "output_path": output,
        "duration_s": data.get("duration_s"),
        "processing_seconds": data.get("processing_seconds"),
        "info": dict(data.get("info") or {}),
    }
    changed = False
    for key, value in values.items():
        current = getattr(row, key)
        if key == "created_at" and current is not None:
            current = current.replace(tzinfo=None)
        if current != value:
            setattr(row, key, value)
            changed = True
    return changed


def sync_runs(db: Session, processed_dir: Path, project_root: Path) -> RunSyncResult:
    with INDEX_LOCK:  # the worker and request threads must not index at the same time
        db.expire_all()  # see what the other thread committed, not this session's old copy
        return _sync_runs(db, processed_dir, project_root)


def _sync_runs(db: Session, processed_dir: Path, project_root: Path) -> RunSyncResult:
    existing = {row.run_id: row for row in db.scalars(select(ProcessingRun))}
    recording_ids = {row.path: row.id for row in db.scalars(select(Recording))}
    seen: set[str] = set()
    added = updated = 0
    for sidecar, data in run_store.iter_runs(processed_dir):
        run_id = data["run_id"]
        seen.add(run_id)
        row = existing.get(run_id)
        if row is None:
            row = ProcessingRun(run_id=run_id, recording_path="", labels=[], attenuation_db=0.0,
                                method="")
            _apply(row, data, sidecar, project_root, recording_ids)
            db.add(row)
            added += 1
        elif _apply(row, data, sidecar, project_root, recording_ids):
            updated += 1
    removed = 0
    for run_id, row in existing.items():
        # only when the sidecar is really gone: one that exists but could not be read just now
        # (being rewritten, locked by another process) keeps its row until the next look
        if run_id not in seen and not run_store.sidecar_path(processed_dir, run_id).exists():
            db.delete(row)
            removed += 1
    db.commit()
    return RunSyncResult(added, updated, removed)


def run_message(row: ProcessingRun) -> WsMessage:
    return WsMessage(type="run", t=0.0,
                     payload=RunModel.model_validate(row).model_dump(mode="json"))


class RunManager:
    """One offline job at a time, in submission order."""

    def __init__(
        self,
        config: AppConfig,
        session_factory: sessionmaker[Session],
        ws_manager: ConnectionManager,
        loop: asyncio.AbstractEventLoop,
        *,
        execute: Callable[[AppConfig, str], dict[str, Any]] = offline.execute_run,
    ) -> None:
        self.config = config
        self.session_factory = session_factory
        self.ws_manager = ws_manager
        self.loop = loop
        self._execute = execute
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.current: str | None = None  # run id being processed

    @property
    def directory(self) -> Path:
        return offline.runs_dir(self.config)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def submit(self, db: Session, recording: Recording, labels: list[str], *,
               attenuation_db: float, method: str | None) -> ProcessingRun:
        """Create, index and queue a run. Raises ``ValueError`` for bad labels or method."""
        path = absolute_path(recording.path, self.config.project_root)
        data = offline.start_run(self.config, path, labels, attenuation_db=attenuation_db,
                                 method=method)
        row = self._index(db, data["run_id"])
        self._ensure_thread()
        self._queue.put(data["run_id"])
        self._publish(row)
        return row

    def _index(self, db: Session, run_id: str) -> ProcessingRun:
        sync_runs(db, self.directory, self.config.project_root)
        return db.scalars(select(ProcessingRun).where(ProcessingRun.run_id == run_id)).one()

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._work, name=THREAD_NAME, daemon=True)
                self._thread.start()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop after the current job; queued jobs stay ``pending`` on disk and are marked
        interrupted at the next start."""
        with self._lock:
            thread = self._thread
        if thread is None:
            return
        self._queue.put(None)
        thread.join(timeout=timeout)

    # --- worker thread -------------------------------------------------------------------------

    def _work(self) -> None:
        while True:
            run_id = self._queue.get()
            if run_id is None:
                return
            self.current = run_id
            try:
                try:
                    run_store.update_run(self.directory, run_id, status="running")
                    self._reindex(run_id)
                    self._execute(self.config, run_id)
                except Exception as exc:  # noqa: BLE001 - never let one job kill the worker
                    logger.exception("processing run %s failed", run_id)
                    try:
                        run_store.fail_run(self.directory, run_id, f"{type(exc).__name__}: {exc}")
                    except Exception:  # noqa: BLE001
                        logger.exception("could not record the failure of run %s", run_id)
                self._reindex(run_id)
            finally:
                self.current = None

    def _reindex(self, run_id: str) -> None:
        try:
            with self.session_factory() as db:
                self._publish(self._index(db, run_id))
        except Exception:  # noqa: BLE001
            logger.exception("could not index run %s", run_id)

    def _publish(self, row: ProcessingRun) -> None:
        message = run_message(row)
        try:
            self.loop.call_soon_threadsafe(self.ws_manager.publish, message)
        except RuntimeError:  # the loop is closed (app shutting down)
            pass

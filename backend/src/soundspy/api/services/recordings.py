"""Keep the ``recordings`` table in step with the recordings on disk.

The JSON sidecar next to each WAV is the source of truth; the database is a disposable **index**
of it. ``sync_recordings`` therefore makes the table equal to what the sidecars say:

- a row is inserted for every sidecar that is not indexed yet;
- a row whose **WAV is gone but whose sidecar is still there** is marked ``missing`` (not
  deleted): the recording stays in the list with its processing runs, but cannot be played or
  processed. If the WAV comes back, the mark is cleared;
- a row whose **sidecar is gone** is deleted: nothing on disk describes that recording any more,
  and keeping it would be information that exists only in the database (a rebuild would lose
  it). Its processing runs are not lost: they have their own sidecars and stay indexed without a
  recording.

Paths are stored relative to the project root when the file lives inside it (portable across
machines), absolute otherwise, always with forward slashes.

Runs on API startup and on every ``GET /api/recordings``. The CLI never calls this: ``soundspy
record`` writes files only, and the API indexes them when it next looks.

**One indexer at a time.** Request threads and the processing worker all re-index; two of them
reading "this sidecar is new" at the same moment would both insert it. ``INDEX_LOCK`` serialises
every sync (recordings and runs).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from soundspy.api.db.models import Recording
from soundspy.io.storage import absolute_path, iter_sidecars, stored_path

__all__ = ["INDEX_LOCK", "SyncResult", "absolute_path", "created_at", "stored_path",
           "sync_recordings"]

INDEX_LOCK = threading.RLock()


@dataclass(frozen=True)
class SyncResult:
    added: int
    removed: int
    missing: int = 0  # rows currently marked missing


def created_at(sidecar: dict[str, Any], fallback: Path) -> datetime:
    """Creation time in UTC. SQLite keeps no time zone, so everything is stored as UTC to keep
    newest-first ordering correct across time zones and daylight-saving changes."""
    created = sidecar.get("created")
    if isinstance(created, str):
        try:
            parsed = datetime.fromisoformat(created)
            return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(fallback.stat().st_mtime, tz=UTC)
    except OSError:
        return datetime.now(UTC)


def _row_from_sidecar(wav_path: Path, sidecar: dict[str, Any], path: str,
                      missing: bool) -> Recording:
    fallback = wav_path if not missing else wav_path.with_suffix(".json")
    return Recording(
        created_at=created_at(sidecar, fallback),
        path=path,
        duration_s=float(sidecar.get("duration_s", 0.0)),
        sample_rate=int(sidecar.get("sample_rate", 0)),
        channels=int(sidecar.get("channels", 0)),
        device=str(sidecar.get("device") or "unknown"),
        profile=str(sidecar.get("profile") or "unknown"),
        label=sidecar.get("label"),
        notes=str(sidecar.get("notes") or ""),
        missing=missing,
    )


def sync_recordings(db: Session, raw_dir: Path, project_root: Path) -> SyncResult:
    with INDEX_LOCK:
        return _sync_recordings(db, raw_dir, project_root)


def _sync_recordings(db: Session, raw_dir: Path, project_root: Path) -> SyncResult:
    existing = {row.path: row for row in db.scalars(select(Recording))}
    on_disk: dict[str, tuple[Path, dict[str, Any], bool]] = {}
    for wav_path, sidecar, exists in iter_sidecars(raw_dir):
        on_disk[stored_path(wav_path, project_root)] = (wav_path, sidecar, exists)

    added = removed = missing = 0
    for path, row in existing.items():
        found = on_disk.get(path)
        if found is None:  # no sidecar: nothing describes this recording any more
            db.delete(row)
            removed += 1
            continue
        row.missing = not found[2]
        missing += row.missing
    for path, (wav_path, sidecar, exists) in on_disk.items():
        if path not in existing:
            db.add(_row_from_sidecar(wav_path, sidecar, path, missing=not exists))
            added += 1
            missing += not exists
    db.commit()
    return SyncResult(added=added, removed=removed, missing=missing)

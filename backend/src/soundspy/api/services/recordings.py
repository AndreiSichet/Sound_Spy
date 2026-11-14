"""Keep the ``recordings`` table in step with the recordings on disk.

The JSON sidecar next to each WAV is the source of truth; the database is an **index** of it.
``sync_recordings`` therefore:

- inserts a row for every sidecar whose WAV is not indexed yet;
- **deletes** rows whose WAV file no longer exists. An index entry for a file that is gone has
  nothing left to point at; if the file comes back, the next sync re-adds it from its sidecar.
  (Deleting a row also deletes its processing runs. Revisit with a ``missing_since`` column and
  a migration once processing runs exist and are worth keeping - step 06.)

Paths are stored relative to the project root when the file lives inside it (portable across
machines), absolute otherwise, always with forward slashes.

Runs on API startup and on every ``GET /api/recordings``. The CLI never calls this: ``soundspy
record`` writes files only, and the API indexes them when it next looks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from soundspy.api.db.models import Recording
from soundspy.io.storage import iter_recordings


@dataclass(frozen=True)
class SyncResult:
    added: int
    removed: int


def stored_path(path: Path, project_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def absolute_path(stored: str, project_root: Path) -> Path:
    path = Path(stored)
    return path if path.is_absolute() else project_root / path


def _created_at(sidecar: dict[str, Any], wav_path: Path) -> datetime:
    """Creation time in UTC. SQLite keeps no time zone, so everything is stored as UTC to keep
    newest-first ordering correct across time zones and daylight-saving changes."""
    created = sidecar.get("created")
    if isinstance(created, str):
        try:
            parsed = datetime.fromisoformat(created)
            return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(wav_path.stat().st_mtime, tz=UTC)


def _row_from_sidecar(wav_path: Path, sidecar: dict[str, Any], path: str) -> Recording:
    return Recording(
        created_at=_created_at(sidecar, wav_path),
        path=path,
        duration_s=float(sidecar.get("duration_s", 0.0)),
        sample_rate=int(sidecar.get("sample_rate", 0)),
        channels=int(sidecar.get("channels", 0)),
        device=str(sidecar.get("device") or "unknown"),
        profile=str(sidecar.get("profile") or "unknown"),
        label=sidecar.get("label"),
        notes=str(sidecar.get("notes") or ""),
    )


def sync_recordings(db: Session, raw_dir: Path, project_root: Path) -> SyncResult:
    existing = {row.path: row for row in db.scalars(select(Recording))}

    removed = 0
    for path, row in existing.items():
        if not absolute_path(path, project_root).is_file():
            db.delete(row)
            removed += 1

    added = 0
    for wav_path, sidecar in iter_recordings(raw_dir):
        path = stored_path(wav_path, project_root)
        if path in existing:
            continue
        db.add(_row_from_sidecar(wav_path, sidecar, path))
        added += 1

    db.commit()
    return SyncResult(added=added, removed=removed)

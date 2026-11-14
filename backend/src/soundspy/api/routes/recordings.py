"""Recording routes: list, details, audio file, offline processing.

``GET /api/recordings`` re-indexes ``data/raw`` (sidecars are the source of truth) and lists the
recordings, newest first.

TODO(step-03): get / audio.
TODO(step-06): process (offline extraction run -> ProcessingRun).
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from soundspy.api.db.models import Recording
from soundspy.api.deps import ConfigDep, DbDep
from soundspy.api.routes import NOT_IMPLEMENTED, not_implemented
from soundspy.api.schemas import RecordingModel
from soundspy.api.services.recordings import sync_recordings

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


@router.get("", response_model=list[RecordingModel])
def list_recordings(config: ConfigDep, db: DbDep) -> list[Recording]:
    sync_recordings(db, config.paths["raw"], config.project_root)
    query = select(Recording).order_by(Recording.created_at.desc(), Recording.id.desc())
    return list(db.scalars(query))


@router.get("/{recording_id}", responses=NOT_IMPLEMENTED)
def get_recording(recording_id: int) -> None:
    raise not_implemented("step-03")


@router.get("/{recording_id}/audio", responses=NOT_IMPLEMENTED)
def get_recording_audio(recording_id: int) -> None:
    raise not_implemented("step-03")


@router.post("/{recording_id}/process", responses=NOT_IMPLEMENTED)
def process_recording(recording_id: int) -> None:
    raise not_implemented("step-06")

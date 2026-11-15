"""Recording routes: list, details, audio file, offline processing.

``GET /api/recordings`` re-indexes ``data/raw`` (sidecars are the source of truth) and lists the
recordings, newest first. ``GET /api/recordings/{id}/audio`` streams the WAV for in-browser
playback (the browser plays *saved* files only; live audio goes to the speakers from the backend).
``POST /api/recordings/{id}/detect`` (step 05) runs offline detection and returns the timeline;
synchronous, so recordings longer than ``detection.offline.max_seconds`` are refused (413).

TODO(step-06): process (offline extraction run -> ProcessingRun).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select

from soundspy.api.db.models import Recording
from soundspy.api.deps import ConfigDep, DbDep
from soundspy.api.routes import NOT_IMPLEMENTED, not_implemented
from soundspy.api.schemas import DetectionTimelineModel, RecordingModel
from soundspy.api.services.recordings import absolute_path, sync_recordings

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

NOT_FOUND = {404: {"description": "No such recording, or its file is gone"}}


@router.get("", response_model=list[RecordingModel])
def list_recordings(config: ConfigDep, db: DbDep) -> list[Recording]:
    sync_recordings(db, config.paths["raw"], config.project_root)
    query = select(Recording).order_by(Recording.created_at.desc(), Recording.id.desc())
    return list(db.scalars(query))


def _recording(db: DbDep, recording_id: int) -> Recording:
    row = db.get(Recording, recording_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no recording with id {recording_id}")
    return row


@router.get("/{recording_id}", response_model=RecordingModel, responses=NOT_FOUND)
def get_recording(recording_id: int, db: DbDep) -> Recording:
    return _recording(db, recording_id)


@router.get(
    "/{recording_id}/audio",
    response_class=FileResponse,
    responses={**NOT_FOUND, 200: {"content": {"audio/wav": {}}}},
)
def get_recording_audio(recording_id: int, config: ConfigDep, db: DbDep) -> FileResponse:
    row = _recording(db, recording_id)
    path = absolute_path(row.path, config.project_root)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"the file of recording {recording_id} is gone")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@router.post(
    "/{recording_id}/detect",
    response_model=DetectionTimelineModel,
    responses={**NOT_FOUND, 413: {"description": "Recording too long for synchronous detection"},
               503: {"description": "No detection model available"}},
)
def detect_recording(recording_id: int, config: ConfigDep, db: DbDep) -> dict:
    import soundfile as sf

    from soundspy.detection.inference import DetectorUnavailable, detect_file

    row = _recording(db, recording_id)
    path = absolute_path(row.path, config.project_root)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"the file of recording {recording_id} is gone")
    limit = config.detection.offline_max_seconds
    duration = sf.info(str(path)).duration
    if duration > limit:
        raise HTTPException(status_code=413,
                            detail=f"recording is {duration:.0f} s long; detection on the page "
                                   f"is limited to {limit:.0f} s (use `soundspy detect`)")
    try:
        return detect_file(config, path).to_json()
    except DetectorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/{recording_id}/process", responses=NOT_IMPLEMENTED)
def process_recording(recording_id: int) -> None:
    raise not_implemented("step-06")

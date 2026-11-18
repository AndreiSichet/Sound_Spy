"""Recording routes: list, details, audio file, offline detection and processing.

``GET /api/recordings`` re-indexes the sidecars (the source of truth) and lists the recordings,
newest first; one whose WAV is gone is listed with ``missing: true``.
``GET /api/recordings/{id}/audio`` streams the WAV for in-browser playback (the browser plays
*saved* files only; live audio goes to the speakers from the backend).
``POST /api/recordings/{id}/detect`` (step 05) runs offline detection and returns the timeline;
synchronous, so recordings longer than ``detection.offline.max_seconds`` are refused (413).

Step 06, offline extraction:

- ``POST /api/recordings/{id}/process`` with ``{labels, attenuation_db?, method?}`` returns the
  new run (``pending``, HTTP 202); the background worker processes it, one job at a time.
  ``method`` is ``neural_tse`` or ``mask_baseline`` (step 07; default: the configured method,
  or the baseline while the neural model does not exist). 404 file gone, 413 longer than
  ``extraction.offline.max_seconds``, 422 no / unknown labels, 503 the method's model is missing
  (the neural checkpoint, or the detection model for the baseline).
- ``GET /api/recordings/{id}/runs``: the recording's runs, newest first.

Run status and audio: ``routes/runs.py``. State changes are pushed over the WebSocket (``run``).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import select

from soundspy.api.db.models import ProcessingRun, Recording
from soundspy.api.deps import ConfigDep, DbDep
from soundspy.api.schemas import DetectionTimelineModel, ProcessRequest, RecordingModel, RunModel
from soundspy.api.services.index import sync_all
from soundspy.api.services.recordings import absolute_path
from soundspy.extraction.offline import neural_available, run_method

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

NOT_FOUND = {404: {"description": "No such recording, or its file is gone"}}


@router.get("", response_model=list[RecordingModel])
def list_recordings(config: ConfigDep, db: DbDep) -> list[Recording]:
    sync_all(db, config)
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


@router.post(
    "/{recording_id}/process",
    response_model=RunModel,
    status_code=202,
    responses={**NOT_FOUND, 413: {"description": "Recording too long for offline processing"},
               422: {"description": "No labels, or an unknown label"},
               503: {"description": "No detection model available"}},
)
def process_recording(recording_id: int, body: ProcessRequest, request: Request,
                      config: ConfigDep, db: DbDep) -> ProcessingRun:
    row = _recording(db, recording_id)
    path = absolute_path(row.path, config.project_root)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"the file of recording {recording_id} is gone")
    limit = config.extraction.offline_max_seconds
    if row.duration_s > limit:
        raise HTTPException(
            status_code=413,
            detail=f"recording is {row.duration_s:.0f} s long; processing on the page is limited "
                   f"to {limit:.0f} s (use `soundspy extract`)")
    method = run_method(config, body.method)
    if method == "neural_tse":
        reason = neural_available(config)
        if reason is not None:
            raise HTTPException(status_code=503,
                                detail=f"neural extraction is not available: {reason}")
    else:
        detection = config.detection
        if detection is None or detection.checkpoint is None or \
                not detection.checkpoint.is_file():
            raise HTTPException(
                status_code=503,
                detail="no detection model is available (detection.checkpoint): the baseline "
                       "extractor needs it to find the selected sounds")
    try:
        return request.app.state.run_manager.submit(
            db, row, body.labels, attenuation_db=body.attenuation_db, method=method,
            gate=body.gate)
    except ValueError as exc:  # unknown label
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{recording_id}/runs", response_model=list[RunModel], responses=NOT_FOUND)
def list_runs(recording_id: int, config: ConfigDep, db: DbDep) -> list[ProcessingRun]:
    _recording(db, recording_id)
    sync_all(db, config)  # also picks up runs made with `soundspy extract`
    query = (select(ProcessingRun).where(ProcessingRun.recording_id == recording_id)
             .order_by(ProcessingRun.created_at.desc(), ProcessingRun.id.desc()))
    return list(db.scalars(query))

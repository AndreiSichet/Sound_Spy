"""Processing-run routes (step 06).

| Method | Path | Result |
|---|---|---|
| GET | /api/runs/{run_id} | the run (status, result); 404 unknown |
| GET | /api/runs/{run_id}/audio | the processed WAV; 404 unknown, 409 not finished |

Runs are created with ``POST /api/recordings/{id}/process`` and listed per recording with
``GET /api/recordings/{id}/runs`` (``routes/recordings.py``). ``run_id`` is the sidecar's name
(``run_<timestamp>``), stable across database rebuilds.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select

from soundspy.api.db.models import ProcessingRun
from soundspy.api.deps import ConfigDep, DbDep
from soundspy.api.schemas import RunModel
from soundspy.api.services.recordings import absolute_path
from soundspy.api.services.runs import sync_runs
from soundspy.extraction.offline import runs_dir
from soundspy.io.runs import FINISHED

router = APIRouter(prefix="/api/runs", tags=["runs"])

NOT_FOUND = {404: {"description": "No such run"}}


def find_run(db: DbDep, config: ConfigDep, run_id: str) -> ProcessingRun:
    query = select(ProcessingRun).where(ProcessingRun.run_id == run_id)
    row = db.scalars(query).one_or_none()
    # Unknown (maybe made by the CLI since the last look) or still in progress: read the
    # sidecars again, so a client that polls always sees the run's current state.
    if row is None or row.status not in FINISHED:
        sync_runs(db, runs_dir(config), config.project_root)
        row = db.scalars(query).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
    return row


@router.get("/{run_id}", response_model=RunModel, responses=NOT_FOUND)
def get_run(run_id: str, config: ConfigDep, db: DbDep) -> ProcessingRun:
    return find_run(db, config, run_id)


@router.get(
    "/{run_id}/audio",
    response_class=FileResponse,
    responses={**NOT_FOUND, 409: {"description": "The run has no audio (not finished or failed)"},
               200: {"content": {"audio/wav": {}}}},
)
def get_run_audio(run_id: str, config: ConfigDep, db: DbDep) -> FileResponse:
    row = find_run(db, config, run_id)
    if row.status != "done" or not row.output_path:
        raise HTTPException(status_code=409, detail=f"run {run_id} is {row.status}: no audio")
    path = absolute_path(row.output_path, config.project_root)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"the audio of run {run_id} is gone")
    return FileResponse(path, media_type="audio/wav", filename=path.name)

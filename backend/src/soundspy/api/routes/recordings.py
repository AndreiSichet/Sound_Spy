"""Recording routes: list, details, audio file, offline processing.

TODO(step-03): list / get / audio (recordings made from the live session).
TODO(step-06): process (offline extraction run -> ProcessingRun).
"""

from __future__ import annotations

from fastapi import APIRouter

from soundspy.api.routes import NOT_IMPLEMENTED, not_implemented

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


@router.get("", responses=NOT_IMPLEMENTED)
def list_recordings() -> None:
    raise not_implemented("step-03")


@router.get("/{recording_id}", responses=NOT_IMPLEMENTED)
def get_recording(recording_id: int) -> None:
    raise not_implemented("step-03")


@router.get("/{recording_id}/audio", responses=NOT_IMPLEMENTED)
def get_recording_audio(recording_id: int) -> None:
    raise not_implemented("step-03")


@router.post("/{recording_id}/process", responses=NOT_IMPLEMENTED)
def process_recording(recording_id: int) -> None:
    raise not_implemented("step-06")

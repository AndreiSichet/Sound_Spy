"""Live session routes: start/stop the engine and change the target.

TODO(step-03): implement on top of ``engine.session.Session``.
"""

from __future__ import annotations

from fastapi import APIRouter

from soundspy.api.routes import NOT_IMPLEMENTED, not_implemented
from soundspy.api.schemas import TargetSpecModel

router = APIRouter(prefix="/api/session", tags=["session"])


@router.post("/start", responses=NOT_IMPLEMENTED)
def start_session() -> None:
    raise not_implemented("step-03")


@router.post("/stop", responses=NOT_IMPLEMENTED)
def stop_session() -> None:
    raise not_implemented("step-03")


@router.put("/target", responses=NOT_IMPLEMENTED)
def set_target(target: TargetSpecModel) -> None:
    raise not_implemented("step-03")

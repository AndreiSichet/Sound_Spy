"""Audio device routes.

TODO(step-02): list input/output devices and the active device profile.
"""

from __future__ import annotations

from fastapi import APIRouter

from soundspy.api.routes import NOT_IMPLEMENTED, not_implemented

router = APIRouter(prefix="/api", tags=["devices"])


@router.get("/devices", responses=NOT_IMPLEMENTED)
def list_devices() -> None:
    raise not_implemented("step-02")

"""Audio device routes.

``GET /api/devices``: every input and output device plus what the active device profile resolves
to. An unresolvable profile is still HTTP 200 (``resolved_input: null`` + ``error``) so the UI can
show why; HTTP 503 means PortAudio itself is unavailable.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from soundspy.api.deps import ConfigDep
from soundspy.api.schemas import DevicesResponse
from soundspy.io.devices import AudioBackendUnavailable, devices_report, refresh_devices

router = APIRouter(prefix="/api", tags=["devices"])


@router.get(
    "/devices",
    response_model=DevicesResponse,
    responses={503: {"description": "PortAudio (the audio backend) is unavailable"}},
)
def list_devices(config: ConfigDep, request: Request) -> DevicesResponse:
    try:
        # Re-read the device list (headphones/USB devices may have changed), but never while a
        # session runs: re-initialising PortAudio would kill its open streams.
        sessions = getattr(request.app.state, "session_manager", None)
        if sessions is None or not sessions.running:
            refresh_devices()
        report = devices_report(config.device_profile)
    except AudioBackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"audio backend unavailable: {exc}") from exc
    return DevicesResponse.model_validate(report)

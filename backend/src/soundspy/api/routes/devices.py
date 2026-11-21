"""Audio device routes.

``GET /api/devices``: every input and output device plus what the active device profile resolves
to. An unresolvable profile is still HTTP 200 (``resolved_input: null`` + ``error``) so the UI can
show why; HTTP 503 means PortAudio itself is unavailable.

``POST /api/devices/rescan`` (step 08): re-initialise PortAudio, so a USB microphone or
headphones plugged in after the backend started can be used without restarting it; 409 while a
session runs (re-initialising would kill its streams). Same answer as ``GET``.
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
    return DevicesResponse.model_validate(_with_output(report, config))


def _with_output(report: dict, config) -> dict:
    """``selected_output``: the device sessions play on (the user's choice or the default)."""
    from soundspy.io.devices import DeviceResolutionError, device_to_dict, resolve_output

    try:
        report["selected_output"] = device_to_dict(resolve_output(config.output.device))
    except DeviceResolutionError as exc:
        report["selected_output"] = None
        report["output_error"] = str(exc)
    report["output_setting"] = config.output.device
    return report


@router.post(
    "/devices/rescan",
    response_model=DevicesResponse,
    responses={409: {"description": "A session is running"},
               503: {"description": "PortAudio (the audio backend) is unavailable"}},
)
def rescan_devices(config: ConfigDep, request: Request) -> DevicesResponse:
    sessions = getattr(request.app.state, "session_manager", None)
    if sessions is not None and sessions.running:
        raise HTTPException(status_code=409,
                            detail="stop the session to rescan the audio devices")
    try:
        refresh_devices()
        report = devices_report(config.device_profile)
    except AudioBackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"audio backend unavailable: {exc}") from exc
    return DevicesResponse.model_validate(_with_output(report, config))

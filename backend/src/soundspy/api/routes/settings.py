"""User settings routes (step 08): the choices the page stores in ``data/user_settings.json``.

| Method | Path | Result |
|---|---|---|
| GET | /api/settings | the stored settings |
| PUT | /api/settings/output-device | ``{"device": index, name or null}``: the output |

(``PUT``: 400 no such device, 409 while a session runs.)

The new output is checked against the devices (``resolve_output``), stored, and used from the next
session on (``SessionManager.config``).
"""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, HTTPException, Request

from soundspy.api.schemas import OutputDeviceRequest, UserSettingsModel
from soundspy.io.devices import AudioBackendUnavailable, DeviceResolutionError, resolve_output
from soundspy.user_settings import load_user_settings, save_user_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=UserSettingsModel)
def get_settings(request: Request) -> UserSettingsModel:
    return UserSettingsModel(**load_user_settings(request.app.state.config))


@router.put("/output-device", response_model=UserSettingsModel,
            responses={400: {"description": "No such output device"},
                       409: {"description": "A session is running"}})
def set_output_device(body: OutputDeviceRequest, request: Request) -> UserSettingsModel:
    sessions = request.app.state.session_manager
    if sessions.running:
        raise HTTPException(status_code=409, detail="stop the session to change the output")
    try:
        resolve_output(body.device)  # a device that exists now
    except DeviceResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AudioBackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"audio backend unavailable: {exc}") from exc
    config = request.app.state.config
    saved = save_user_settings(config, output_device=body.device)
    config = dataclasses.replace(config, output=dataclasses.replace(config.output,
                                                                    device=body.device))
    request.app.state.config = sessions.config = config
    return UserSettingsModel(**saved)

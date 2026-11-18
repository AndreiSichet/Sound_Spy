"""Live session routes.

| Method | Path | Result |
|---|---|---|
| GET  | /api/session | current status (idle if none) |
| POST | /api/session/start | status; 409 running, 400 bad mode/source, 503 no audio backend |
| POST | /api/session/stop | status (idempotent) |
| PUT  | /api/session/target | the stored target; 422 unknown label |
| PUT  | /api/session/monitor | status; gain_db in [-40, +12]; 409 no output device |
| PUT  | /api/session/bypass | status; ``{"enabled": bool}``: hear the unprocessed signal (A/B) |
| PUT  | /api/session/method | status; ``{"method"}``: hybrid / neural_tse / baseline; 409 running |
| PUT  | /api/session/gate | status; ``{"enabled": bool}``: the method's gate; 409 running |
| POST | /api/session/record/start | status; 409 no session / already recording |
| POST | /api/session/record/stop | the new Recording; 409 not recording |
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from soundspy.api.deps import DbDep
from soundspy.api.schemas import (
    BypassRequest,
    EngineStatusModel,
    GateRequest,
    MethodRequest,
    MonitorRequest,
    RecordingModel,
    RecordStartRequest,
    SessionStartRequest,
    TargetSpecModel,
)
from soundspy.api.services.session_manager import (
    AudioOpenError,
    BadRequestError,
    ConflictError,
    SessionManager,
)
from soundspy.io.devices import AudioBackendUnavailable

router = APIRouter(prefix="/api/session", tags=["session"])

ERRORS = {
    400: {"description": "Invalid mode or source"},
    409: {"description": "Conflicts with the current session state"},
    503: {"description": "Audio backend unavailable"},
}


def _sessions(request: Request) -> SessionManager:
    return request.app.state.session_manager


def _status(sessions: SessionManager) -> EngineStatusModel:
    return EngineStatusModel.model_validate(sessions.status())


@router.get("", response_model=EngineStatusModel)
def get_session(request: Request) -> EngineStatusModel:
    return _status(_sessions(request))


@router.post("/start", response_model=EngineStatusModel, responses=ERRORS)
def start_session(body: SessionStartRequest, request: Request) -> EngineStatusModel:
    sessions = _sessions(request)
    try:
        sessions.start(body)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AudioBackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"audio backend unavailable: {exc}") from exc
    except AudioOpenError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - anything else still becomes a clear 503
        raise HTTPException(status_code=503, detail=f"could not open audio: {exc}") from exc
    return _status(sessions)


@router.post("/stop", response_model=EngineStatusModel)
def stop_session(request: Request, db: DbDep) -> EngineStatusModel:
    sessions = _sessions(request)
    sessions.stop(db)
    return _status(sessions)


@router.put(
    "/target", response_model=TargetSpecModel, responses={422: {"description": "Unknown label"}}
)
def set_target(body: TargetSpecModel, request: Request) -> TargetSpecModel:
    sessions = _sessions(request)
    target = body.to_core(sessions.config.extraction.default_attenuation_db)
    try:
        target.validate(sessions.labels)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    sessions.set_target(target)
    return body.model_copy(update={"attenuation_db": target.attenuation_db})


@router.put("/monitor", response_model=EngineStatusModel, responses=ERRORS)
def set_monitor(body: MonitorRequest, request: Request) -> EngineStatusModel:
    sessions = _sessions(request)
    try:
        sessions.set_monitor(body.enabled, body.gain_db, body.muted)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status(sessions)


@router.put("/bypass", response_model=EngineStatusModel)
def set_bypass(body: BypassRequest, request: Request) -> EngineStatusModel:
    sessions = _sessions(request)
    sessions.set_bypass(body.enabled)
    return _status(sessions)


@router.put("/method", response_model=EngineStatusModel, responses=ERRORS)
def set_method(body: MethodRequest, request: Request) -> EngineStatusModel:
    sessions = _sessions(request)
    try:
        sessions.set_method(body.method)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status(sessions)


@router.put("/gate", response_model=EngineStatusModel, responses=ERRORS)
def set_gate(body: GateRequest, request: Request) -> EngineStatusModel:
    sessions = _sessions(request)
    try:
        sessions.set_gate(body.enabled)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status(sessions)


@router.post("/record/start", response_model=EngineStatusModel, responses=ERRORS)
def record_start(body: RecordStartRequest, request: Request) -> EngineStatusModel:
    sessions = _sessions(request)
    try:
        sessions.record_start(body.label or None, body.notes)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status(sessions)


@router.post("/record/stop", response_model=RecordingModel, responses=ERRORS)
def record_stop(request: Request, db: DbDep) -> RecordingModel:
    try:
        row = _sessions(request).record_stop(db)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RecordingModel.model_validate(row)

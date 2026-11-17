"""Pydantic models mirroring the core types, REST responses and the WebSocket envelope.

WebSocket message envelope (both directions)::

    {"type": "status" | "levels" | "waveform" | "spectrogram" | "detections" | "sources"
             | "run" | "warning" | "error" | "set_target" | "set_bypass" | "set_method",
     "t": float,          # seconds since session start (0.0 when idle)
     "payload": {...}}

Server -> client payloads are the event models below without their ``t`` (it is in the
envelope). ``levels`` / ``waveform`` / ``spectrogram`` carry ``source``: ``"input"`` or
``"output"`` (step 06). ``detections`` carries a ``DetectionFrameModel`` (step 05:
``probabilities`` per class and the ``active`` classes, at the detector's hop rate); ``sources``
carries ``{"items": [...]}``; ``run`` carries a ``RunModel`` whenever an offline processing run
changes state (step 06). Client -> server: ``set_target`` with a ``TargetSpecModel`` payload,
``set_bypass`` with ``{"enabled": bool}``, ``set_method`` with ``{"method": "neural_tse" |
"mask_baseline"}`` (step 07; only while no session runs).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from soundspy.core import events as core_events
from soundspy.core.target import Direction, TargetSpec
from soundspy.engine.modes import Mode

MessageType = Literal[
    "status",
    "levels",
    "waveform",
    "spectrogram",
    "detections",
    "sources",
    "run",
    "warning",
    "error",
    "set_target",
    "set_bypass",
    "set_method",
]
MESSAGE_TYPES: tuple[str, ...] = MessageType.__args__  # type: ignore[attr-defined]
CLIENT_MESSAGE_TYPES: tuple[str, ...] = ("set_target", "set_bypass", "set_method")
# High-rate visual messages a slow client may lose (oldest first); everything else is kept.
DROPPABLE_TYPES: frozenset[str] = frozenset({"levels", "waveform", "spectrogram"})


class _FromCore(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- core mirrors -----------------------------------------------------------------------------


class DirectionModel(_FromCore):
    azimuth_deg: float = Field(ge=0.0, lt=360.0)
    width_deg: float = Field(gt=0.0, le=360.0)

    def to_core(self) -> Direction:
        return Direction(self.azimuth_deg, self.width_deg)


class TargetSpecModel(_FromCore):
    labels: list[str] = Field(default_factory=list)
    direction: DirectionModel | None = None
    source_id: str | None = None
    # None = the configured default (extraction.default_attenuation_db, 15 dB)
    attenuation_db: float | None = Field(default=None, ge=0.0)

    def to_core(self, default_attenuation_db: float = 30.0) -> TargetSpec:
        attenuation = (default_attenuation_db if self.attenuation_db is None
                       else self.attenuation_db)
        return TargetSpec(
            labels=frozenset(self.labels),
            direction=self.direction.to_core() if self.direction else None,
            source_id=self.source_id,
            attenuation_db=attenuation,
        )


class DetectionModel(_FromCore):
    label: str
    probability: float
    t_start: float
    t_end: float


class DetectionFrameModel(_FromCore):
    probabilities: dict[str, float]
    active: list[str]


class DetectorStatusModel(BaseModel):
    model: str
    device: str
    hop_s: float
    latency_ms: float | None = None
    skipped: int = 0


class DetectionTimelineModel(BaseModel):
    """Offline detection of a recording (``POST /api/recordings/{id}/detect``)."""

    duration_s: float
    hop_s: float
    window_s: float
    labels: list[str]
    times: list[float]
    probabilities: dict[str, list[float]]
    segments: list[DetectionModel]


class SoundSourceModel(_FromCore):
    id: str
    label: str
    direction: DirectionModel
    activity: float
    last_seen: float


Source = Literal["input", "output"]  # which side of the chain a visual frame observes


class LevelFrameModel(_FromCore):
    rms_dbfs: list[float]
    peak_dbfs: list[float]
    clipping: list[bool]
    source: Source = "input"


class WaveformFrameModel(_FromCore):
    mins: list[float]
    maxs: list[float]
    source: Source = "input"


class SpectrogramFrameModel(_FromCore):
    magnitudes_db: list[float]
    freq_max_hz: float
    freq_min_hz: float = 0.0
    source: Source = "input"


class ExtractionStatusModel(BaseModel):
    """The session's extractor (step 06)."""

    method: str
    latency_ms: float = 0.0  # its algorithmic delay (part of ``latency_ms`` of the status)
    gating: bool = False  # True when a detector tells it whether the selected sounds are present
    target_active: bool = False  # the selected sounds are currently considered present
    # neural extractor only (step 07)
    device: str | None = None
    model: str | None = None
    model_ms: float | None = None  # mean time of one model call (one per chunk)
    model_ms_p95: float | None = None
    compiled: bool | None = None  # the streaming step runs as a traced (TorchScript) module


class WarningModel(_FromCore):
    code: str
    message: str


class EngineStatusModel(_FromCore):
    state: Literal["idle", "running", "error"]
    mode: str | None = None
    device: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    message: str = ""
    source: str | None = None
    output_device: str | None = None
    monitoring: bool = False
    muted: bool = False
    gain_db: float = 0.0
    recording: bool = False
    recording_seconds: float = 0.0
    latency_ms: float | None = None
    chunk_ms: float | None = None
    chunks_processed: int = 0
    proc_ms_mean: float | None = None
    proc_ms_p95: float | None = None
    rtf_p95: float | None = None
    input_overflows: int = 0
    output_underruns: int = 0
    dropped_chunks: int = 0
    output_dropped: int = 0
    detector: DetectorStatusModel | None = None
    bypass: bool = False
    extraction: ExtractionStatusModel | None = None
    extraction_method: str | None = None  # the method chosen for sessions


class ErrorPayload(BaseModel):
    message: str


# --- WebSocket envelope ------------------------------------------------------------------------


class WsMessage(BaseModel):
    type: MessageType
    t: float = 0.0
    payload: dict[str, Any] = Field(default_factory=dict)


def status_message(status: core_events.EngineStatus) -> WsMessage:
    payload = EngineStatusModel.model_validate(status).model_dump()
    return WsMessage(type="status", t=status.t, payload=payload)


def error_message(message: str, t: float = 0.0) -> WsMessage:
    return WsMessage(type="error", t=t, payload=ErrorPayload(message=message).model_dump())


_EVENT_MESSAGES: dict[type, tuple[str, type[BaseModel]]] = {
    core_events.LevelFrame: ("levels", LevelFrameModel),
    core_events.WaveformFrame: ("waveform", WaveformFrameModel),
    core_events.SpectrogramFrame: ("spectrogram", SpectrogramFrameModel),
    core_events.WarningEvent: ("warning", WarningModel),
    core_events.EngineStatus: ("status", EngineStatusModel),
    core_events.DetectionFrame: ("detections", DetectionFrameModel),
}


def event_message(event: core_events.Event) -> WsMessage:
    """Wrap a core event in the WebSocket envelope."""
    if isinstance(event, core_events.Detection):
        payload = {"items": [DetectionModel.model_validate(event).model_dump()]}
        return WsMessage(type="detections", t=event.t, payload=payload)
    try:
        msg_type, model = _EVENT_MESSAGES[type(event)]
    except KeyError:
        raise TypeError(f"no WebSocket message for {type(event).__name__}") from None
    return WsMessage(type=msg_type, t=event.t, payload=model.model_validate(event).model_dump())


# --- session requests --------------------------------------------------------------------------


class SourceRequest(BaseModel):
    type: Literal["mic", "file"] = "mic"
    path: str | None = None  # file: relative to the project root
    loop: bool = True


SessionMethod = Literal["neural_tse", "mask_baseline", "passthrough"]
RunMethod = Literal["neural_tse", "mask_baseline"]


class SessionStartRequest(BaseModel):
    mode: str = "semantic"
    source: SourceRequest = Field(default_factory=SourceRequest)
    monitor: bool = False
    method: SessionMethod | None = None  # None = the chosen one (PUT /api/session/method)


class MethodRequest(BaseModel):
    method: SessionMethod


class MonitorRequest(BaseModel):
    enabled: bool
    gain_db: float = Field(default=0.0, ge=-40.0, le=12.0)
    muted: bool = False


class RecordStartRequest(BaseModel):
    label: str | None = Field(default=None, max_length=60)
    notes: str = Field(default="", max_length=2000)


class BypassRequest(BaseModel):
    enabled: bool


class ProcessRequest(BaseModel):
    """``POST /api/recordings/{id}/process``: extract ``labels`` from the recording."""

    labels: list[str] = Field(min_length=1)
    # None = extraction.default_attenuation_db
    attenuation_db: float | None = Field(default=None, ge=0.0, le=60.0)
    method: RunMethod | None = None  # None = the default method that can run


# --- REST --------------------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ClassInfo(BaseModel):
    """A selectable class as the UI sees it. AudioSet names are dataset-only and not exposed."""

    name: str  # internal ID, used in TargetSpec.labels
    display_name: str
    description: str


class ModeInfo(BaseModel):
    mode: Mode
    available: bool
    reason: str | None = None


class MethodInfo(BaseModel):
    """An extraction method the page can offer."""

    name: str  # neural_tse | mask_baseline
    display_name: str
    available: bool
    reason: str | None = None  # why not, when unavailable


class ConfigResponse(BaseModel):
    classes: list[ClassInfo]
    modes: list[ModeInfo]
    device_profile: str
    extraction_methods: list[MethodInfo] = Field(default_factory=list)
    default_method: str = "mask_baseline"  # what a session uses when nothing was chosen
    default_attenuation_db: float = 15.0


class NotImplementedResponse(BaseModel):
    detail: str


class AudioDeviceModel(BaseModel):
    index: int
    name: str
    hostapi: str
    max_channels: int
    default_samplerate: int
    is_default: bool


class ResolvedInputModel(BaseModel):
    index: int
    name: str
    hostapi: str
    channels_used: list[int]
    sample_rate: int


class DevicesResponse(BaseModel):
    active_profile: str
    resolved_input: ResolvedInputModel | None
    error: str | None = None
    inputs: list[AudioDeviceModel]
    outputs: list[AudioDeviceModel]


class RecordingModel(_FromCore):
    id: int
    created_at: datetime
    path: str  # relative to the project root when inside it
    duration_s: float
    sample_rate: int
    channels: int
    device: str
    profile: str
    label: str | None
    notes: str
    missing: bool = False  # the sidecar is there but the WAV is gone

    @field_validator("created_at")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        # SQLite returns naive datetimes; the index stores UTC (see services/recordings.py).
        return value if value.tzinfo else value.replace(tzinfo=UTC)


class RunModel(_FromCore):
    """One offline processing run (index entry of ``outputs/processed/<run_id>.json``)."""

    run_id: str
    created_at: datetime
    recording_id: int | None
    recording_path: str
    labels: list[str]
    attenuation_db: float
    method: str
    activity_mode: str = ""
    status: Literal["pending", "running", "done", "failed"]
    error: str | None = None
    output_path: str | None = None  # relative to the project root; set when done
    duration_s: float | None = None
    processing_seconds: float | None = None
    info: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=UTC)

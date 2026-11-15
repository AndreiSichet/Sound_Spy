"""Pydantic models mirroring the core types, REST responses and the WebSocket envelope.

WebSocket message envelope (both directions)::

    {"type": "status" | "levels" | "waveform" | "spectrogram" | "detections" | "sources"
             | "warning" | "error" | "set_target",
     "t": float,          # seconds since session start (0.0 when idle)
     "payload": {...}}

Server -> client payloads are the event models below without their ``t`` (it is in the
envelope). ``detections`` carries a ``DetectionFrameModel`` (step 05: ``probabilities`` per
class and the ``active`` classes, at the detector's hop rate); ``sources`` carries
``{"items": [...]}``. Client -> server: ``set_target`` with a ``TargetSpecModel`` payload.
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
    "warning",
    "error",
    "set_target",
]
MESSAGE_TYPES: tuple[str, ...] = MessageType.__args__  # type: ignore[attr-defined]
CLIENT_MESSAGE_TYPES: tuple[str, ...] = ("set_target",)
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
    attenuation_db: float = Field(default=30.0, ge=0.0)

    def to_core(self) -> TargetSpec:
        return TargetSpec(
            labels=frozenset(self.labels),
            direction=self.direction.to_core() if self.direction else None,
            source_id=self.source_id,
            attenuation_db=self.attenuation_db,
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


class LevelFrameModel(_FromCore):
    rms_dbfs: list[float]
    peak_dbfs: list[float]
    clipping: list[bool]


class WaveformFrameModel(_FromCore):
    mins: list[float]
    maxs: list[float]


class SpectrogramFrameModel(_FromCore):
    magnitudes_db: list[float]
    freq_max_hz: float
    freq_min_hz: float = 0.0


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


class SessionStartRequest(BaseModel):
    mode: str = "semantic"
    source: SourceRequest = Field(default_factory=SourceRequest)
    monitor: bool = False


class MonitorRequest(BaseModel):
    enabled: bool
    gain_db: float = Field(default=0.0, ge=-40.0, le=12.0)
    muted: bool = False


class RecordStartRequest(BaseModel):
    label: str | None = Field(default=None, max_length=60)
    notes: str = Field(default="", max_length=2000)


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


class ConfigResponse(BaseModel):
    classes: list[ClassInfo]
    modes: list[ModeInfo]
    device_profile: str


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

    @field_validator("created_at")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        # SQLite returns naive datetimes; the index stores UTC (see services/recordings.py).
        return value if value.tzinfo else value.replace(tzinfo=UTC)

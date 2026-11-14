"""Events emitted by side processors and the engine.

Every event carries ``t``: seconds since the session started (0.0 when no session is running).
Events are plain dataclasses; the API layer converts them to Pydantic models and wraps them in
the WebSocket message envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from soundspy.core.target import Direction

EngineState = Literal["idle", "running", "error"]


@dataclass(frozen=True)
class Event:
    """Base class of all events."""

    t: float


@dataclass(frozen=True)
class Detection(Event):
    """A class detected in the time span [t_start, t_end] (seconds since session start)."""

    label: str
    probability: float
    t_start: float
    t_end: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")
        if self.t_end < self.t_start:
            raise ValueError("t_end must be >= t_start")


@dataclass(frozen=True)
class SoundSource(Event):
    """A tracked source for the Sources tab: class, direction and current activity (0..1)."""

    id: str
    label: str
    direction: Direction
    activity: float
    last_seen: float


@dataclass(frozen=True)
class LevelFrame(Event):
    """Per-channel levels of one chunk."""

    rms_dbfs: list[float]
    peak_dbfs: list[float]
    clipping: list[bool]


@dataclass(frozen=True)
class SpectrogramFrame(Event):
    """One spectrogram column (magnitudes in dB, linearly spaced from 0 to ``freq_max_hz``)."""

    magnitudes_db: list[float]
    freq_max_hz: float


@dataclass(frozen=True)
class EngineStatus(Event):
    """State of the engine/session."""

    state: EngineState
    mode: str | None = None
    device: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    message: str = ""

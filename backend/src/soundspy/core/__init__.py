"""Core contracts shared by every layer: audio container, target spec, events, label set."""

from soundspy.core.audio import AudioBuffer
from soundspy.core.events import (
    Detection,
    EngineState,
    EngineStatus,
    Event,
    LevelFrame,
    SoundSource,
    SpectrogramFrame,
    WarningEvent,
    WaveformFrame,
)
from soundspy.core.labels import ClassDef, LabelSet
from soundspy.core.target import Direction, TargetSpec

__all__ = [
    "AudioBuffer",
    "ClassDef",
    "Detection",
    "Direction",
    "EngineState",
    "EngineStatus",
    "Event",
    "LabelSet",
    "LevelFrame",
    "SoundSource",
    "SpectrogramFrame",
    "TargetSpec",
    "WarningEvent",
    "WaveformFrame",
]

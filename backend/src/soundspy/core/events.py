"""Events emitted by side processors and the engine.

The visual frames (``LevelFrame``, ``WaveformFrame``, ``SpectrogramFrame``) carry ``source``:
``"input"`` for what the microphone heard, ``"output"`` for the chain's output (step 06).

Every event carries ``t``: seconds since the session started (0.0 when no session is running).
Events are plain dataclasses; the API layer converts them to Pydantic models and wraps them in
the WebSocket message envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from soundspy.core.target import Direction

EngineState = Literal["idle", "warming", "running", "error"]  # warming: step 08


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
class DetectionFrame(Event):
    """Live detection at time ``t`` (end of the analysed window, seconds since session start).

    ``probabilities``: smoothed probability per class (every class of the ``LabelSet``);
    ``active``: the classes currently switched on (threshold + hysteresis), in class order.
    """

    probabilities: dict[str, float]
    active: list[str]

    def __post_init__(self) -> None:
        for label, p in self.probabilities.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"probability of {label!r} must be in [0, 1], got {p}")
        unknown = [a for a in self.active if a not in self.probabilities]
        if unknown:
            raise ValueError(f"active labels without a probability: {unknown}")


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
    """Per-channel levels over one analysis interval."""

    rms_dbfs: list[float]
    peak_dbfs: list[float]
    clipping: list[bool]
    source: str = "input"  # "input" (what the microphone heard) | "output" (after the chain)


@dataclass(frozen=True)
class WaveformFrame(Event):
    """Min/max envelope of the mono mix over one interval (``waveform_points`` pairs)."""

    mins: list[float]
    maxs: list[float]
    source: str = "input"


@dataclass(frozen=True)
class SpectrogramFrame(Event):
    """One spectrogram column: mel-band magnitudes in dB between ``freq_min_hz`` and
    ``freq_max_hz`` (bands equally spaced on the mel scale, lowest first)."""

    magnitudes_db: list[float]
    freq_max_hz: float
    freq_min_hz: float = 0.0
    source: str = "input"


@dataclass(frozen=True)
class WarningEvent(Event):
    """Something the user should know that is not a failure (e.g. processing too slow).

    Named ``WarningEvent`` rather than ``Warning`` so it does not shadow Python's built-in."""

    code: str
    message: str


@dataclass(frozen=True)
class EngineStatus(Event):
    """State of the engine/session."""

    state: EngineState
    mode: str | None = None
    device: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    message: str = ""
    source: str | None = None  # "mic" | "file"
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
    dropped_chunks: int = 0  # input queue (consumer too slow)
    output_dropped: int = 0  # output queue (clock drift / bursts)
    output_drained: int = 0  # times the output queue was emptied back to its cushion (07b)
    # {model, device, hop_s, latency_ms, skipped} while a detector runs, else None (step 05)
    detector: dict[str, Any] | None = None
    # step 06: the speaker gets the unprocessed (delay-matched) signal instead of the output
    bypass: bool = False
    # {method, latency_ms, gating, target_active} of the chain's extractor, else None (step 06);
    # the neural extractor adds {device, model, model_ms, model_ms_p95} (step 07)
    extraction: dict[str, Any] | None = None
    # the method chosen for sessions (the next one, when idle); `extraction.method` is what the
    # running session really uses, which differs after a fallback (step 07)
    extraction_method: str | None = None
    extraction_gate: bool | None = None  # the neural output gate chosen for sessions (07b)
    # step 08: seconds from Start to the first processed chunk (models loaded and warm); while
    # warming, `message` says what is being prepared
    warmup_s: float | None = None

"""Dataset collectors: one module per dataset, all producing :class:`ClipRow` objects.

A collector reads a dataset's own metadata (CSV files, sidecars) and returns one row per clip
with AudioSet mids, original labels, split and leakage group. Audio facts (duration, sample
rate, channels) and roles are filled in by ``soundspy.data.prepare``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class CollectorError(RuntimeError):
    pass


@dataclass
class ClipRow:
    clip_id: str
    dataset: str
    path: str  # aliased, e.g. "fsd50k://FSD50K.dev_audio/64760.wav"
    file: Path  # absolute path, only used while preparing (not stored)
    mids: list[str]
    raw_labels: list[str]
    split: str
    group: str
    license: str
    role: str | None = None  # preset by the custom collector; otherwise from the role rule
    extra: dict[str, str] = field(default_factory=dict)

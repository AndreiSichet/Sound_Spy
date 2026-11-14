"""What the user wants to hear: ``Direction`` and ``TargetSpec``.

One ``TargetSpec`` serves all three tabs: the Semantic tab sets ``labels``, the Spatial tab adds
``direction``, the Sources tab sets ``source_id``. Each processor reads only the fields it needs.
``TargetSpec`` is the only way to tell processors what the user wants.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soundspy.core.labels import LabelSet


@dataclass(frozen=True)
class Direction:
    """A look direction: azimuth in [0, 360) degrees and beam width in (0, 360] degrees."""

    azimuth_deg: float
    width_deg: float = 30.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.azimuth_deg):
            raise ValueError(f"azimuth_deg must be finite, got {self.azimuth_deg}")
        if not (math.isfinite(self.width_deg) and 0.0 < self.width_deg <= 360.0):
            raise ValueError(f"width_deg must be in (0, 360], got {self.width_deg}")
        object.__setattr__(self, "azimuth_deg", float(self.azimuth_deg) % 360.0)
        object.__setattr__(self, "width_deg", float(self.width_deg))


@dataclass(frozen=True)
class TargetSpec:
    """Target of the processing chain.

    ``labels`` empty means "pass everything". ``attenuation_db`` (>= 0) is how much
    non-target sound is suppressed.
    """

    labels: frozenset[str] = field(default_factory=frozenset)
    direction: Direction | None = None
    source_id: str | None = None
    attenuation_db: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.labels, str):
            raise TypeError("labels must be a collection of label names, not a single string")
        labels: Iterable[str] = self.labels
        object.__setattr__(self, "labels", frozenset(labels))
        if not (math.isfinite(self.attenuation_db) and self.attenuation_db >= 0.0):
            raise ValueError(f"attenuation_db must be >= 0, got {self.attenuation_db}")
        object.__setattr__(self, "attenuation_db", float(self.attenuation_db))

    @property
    def passes_everything(self) -> bool:
        return not self.labels

    def validate(self, label_set: LabelSet) -> None:
        """Raise ``ValueError`` if any label is not part of ``label_set``."""
        label_set.validate(self.labels)

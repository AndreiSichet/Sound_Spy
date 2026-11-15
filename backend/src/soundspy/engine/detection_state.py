"""``DetectionState``: the latest detection, shared between the detector and the extractor.

The live ``Detector`` (a side processor) writes its newest smoothed probabilities and active
classes here; the extractor reads them to decide whether the target is sounding. One instance is
built per chain by ``engine/factory.py`` and handed to both stages (no globals). All access is
behind a lock, so reading from another thread (status, tests) is safe.

``t`` is the session time (seconds) of the **end** of the analysed window, ``None`` until the
first detection has arrived.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectionSnapshot:
    t: float | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    active: frozenset[str] = frozenset()

    @property
    def available(self) -> bool:
        """False until the detector has reported at least once."""
        return self.t is not None


class DetectionState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = DetectionSnapshot()
        self.updates = 0

    def update(self, t: float, probabilities: Mapping[str, float], active: Iterable[str]) -> None:
        snapshot = DetectionSnapshot(float(t), dict(probabilities), frozenset(active))
        with self._lock:
            self._snapshot = snapshot
            self.updates += 1

    def snapshot(self) -> DetectionSnapshot:
        with self._lock:
            return self._snapshot

    def reset(self) -> None:
        with self._lock:
            self._snapshot = DetectionSnapshot()
            self.updates = 0

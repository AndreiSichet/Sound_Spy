"""App modes (one per UI tab) and the stage kinds each mode uses.

Only the mapping lives here; building concrete chains from it is done in later steps.
"""

from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    SEMANTIC = "semantic"  # tab 1: select sound types
    SPATIAL = "spatial"  # tab 2: sound types + direction
    SOURCES = "sources"  # tab 3: one detected source on a map


class StageKind(StrEnum):
    # main path (Processor), in this order
    SPATIAL_FILTER = "spatial_filter"
    EXTRACTOR = "extractor"
    # side branch (SideProcessor)
    DETECTOR = "detector"
    ANALYZER = "analyzer"
    LOCALIZER = "localizer"
    TRACKER = "tracker"


MAIN_STAGES: tuple[StageKind, ...] = (StageKind.SPATIAL_FILTER, StageKind.EXTRACTOR)
SIDE_STAGES: tuple[StageKind, ...] = (
    StageKind.DETECTOR,
    StageKind.ANALYZER,
    StageKind.LOCALIZER,
    StageKind.TRACKER,
)

_SEMANTIC = frozenset({StageKind.EXTRACTOR, StageKind.DETECTOR, StageKind.ANALYZER})
_SPATIAL = _SEMANTIC | {StageKind.SPATIAL_FILTER, StageKind.LOCALIZER}
_SOURCES = _SPATIAL | {StageKind.TRACKER}

MODE_STAGES: dict[Mode, frozenset[StageKind]] = {
    Mode.SEMANTIC: _SEMANTIC,
    Mode.SPATIAL: _SPATIAL,
    Mode.SOURCES: _SOURCES,
}


def stages_for(mode: Mode) -> frozenset[StageKind]:
    return MODE_STAGES[Mode(mode)]


def main_stages_for(mode: Mode) -> list[StageKind]:
    """Main-path stage kinds of ``mode`` in processing order."""
    stages = stages_for(mode)
    return [kind for kind in MAIN_STAGES if kind in stages]


def side_stages_for(mode: Mode) -> list[StageKind]:
    stages = stages_for(mode)
    return [kind for kind in SIDE_STAGES if kind in stages]


def requires_array(mode: Mode) -> bool:
    """True if the mode needs a microphone array (a device profile with geometry)."""
    return StageKind.SPATIAL_FILTER in stages_for(mode) or StageKind.LOCALIZER in stages_for(mode)

"""Collector for the owner's own labelled clips (``paths.custom``, ``data/custom`` on C:).

Recorded with ``soundspy record-dataset``: ``data/custom/<class>/<class>_0001.wav`` + a JSON
sidecar holding ``class``, ``session_id`` and ``notes``. The owner chose the label while
recording, so the role is taken from it directly: a class key -> ``fg:<class>``,
``background`` -> ``bg``, anything else (e.g. a class removed from ``classes.yaml``) -> excluded.
Split: whole sessions shuffled with a fixed seed (``splits.custom``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from soundspy.core.labels import LabelSet
from soundspy.data.aliases import to_alias
from soundspy.data.collectors import ClipRow
from soundspy.data.roles import BG, EXCLUDED, fg_role
from soundspy.data.splits import split_groups
from soundspy.io.storage import iter_recordings

BACKGROUND = "background"
LICENSE = "own recording"


def collect(
    paths: Mapping[str, Path],
    labels: LabelSet,
    class_mids: Mapping[str, tuple[str, ...]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[ClipRow]:
    root = Path(paths["custom"])
    if not root.is_dir():
        return []
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for wav, side in iter_recordings(folder):
            cls = str(side.get("class") or folder.name)
            if cls == BACKGROUND:
                role, mids = BG, []
            elif cls in labels:
                role, mids = fg_role(cls), list(class_mids.get(cls, ()))
            else:
                role, mids = EXCLUDED, []
            session = side.get("session_id") or f"file:{wav.stem}"
            rows.append(ClipRow(
                clip_id=f"custom:{wav.stem}",
                dataset="custom",
                path=to_alias("custom", f"{folder.name}/{wav.name}"),
                file=wav,
                mids=mids,
                raw_labels=[cls],
                split="",
                group=f"session:{session}",
                license=LICENSE,
                role=role,
            ))
    assignment = split_groups(
        (r.group for r in rows), val_fraction=val_fraction, test_fraction=test_fraction,
        seed=seed,
    )
    for row in rows:
        row.split = assignment[row.group]
    return rows

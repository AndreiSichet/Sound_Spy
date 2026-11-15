"""ESC-50 collector (``<external>/esc50/ESC-50-master``).

``meta/esc50.csv`` columns: ``filename, fold, target, category, esc10, src_file, take``.
ESC-50's own 50 categories are mapped to AudioSet mids by the hand-written
``configs/data/esc50_to_audioset.yaml`` (``null`` = no sensible match -> no mids -> excluded).
Split from the official fold; group = source recording (``1-100032`` of ``1-100032-A-0.wav``).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from soundspy.config import load_yaml
from soundspy.data.aliases import dataset_root, to_alias
from soundspy.data.collectors import ClipRow, CollectorError
from soundspy.data.ontology import Ontology
from soundspy.data.splits import esc50_group, esc50_split

LICENSE = "CC BY-NC 3.0"
CATEGORIES = 50


def load_mapping(path: Path, ontology: Ontology | None = None) -> dict[str, list[str]]:
    """``{category: [mid, ...]}`` (empty list for ``null``); every mid checked if ``ontology``."""
    raw = load_yaml(path)
    mapping: dict[str, list[str]] = {}
    for category, mids in raw.items():
        if mids is None:
            mapping[category] = []
            continue
        if not isinstance(mids, list) or not all(isinstance(m, str) for m in mids):
            raise CollectorError(f"{path.name}: {category!r} must be a list of mids or null")
        if ontology is not None:
            unknown = [m for m in mids if m not in ontology]
            if unknown:
                raise CollectorError(f"{path.name}: {category!r} has unknown mids {unknown}")
        mapping[category] = list(mids)
    return mapping


def collect(
    paths: Mapping[str, Path],
    mapping: Mapping[str, list[str]],
    folds: Mapping[str, tuple[int, ...]],
) -> list[ClipRow]:
    root = dataset_root("esc50", paths)
    csv_path = root / "meta" / "esc50.csv"
    if not csv_path.exists():
        raise CollectorError(f"ESC-50 metadata not found at {csv_path}")
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            category = rec["category"]
            if category not in mapping:
                raise CollectorError(
                    f"ESC-50 category {category!r} is missing in esc50_to_audioset.yaml"
                )
            filename = rec["filename"]
            stem = filename.rsplit(".", 1)[0]
            rows.append(ClipRow(
                clip_id=f"esc50:{stem}",
                dataset="esc50",
                path=to_alias("esc50", f"audio/{filename}"),
                file=root / "audio" / filename,
                mids=list(mapping[category]),
                raw_labels=[category],
                split=esc50_split(int(rec["fold"]), folds),
                group=esc50_group(filename),
                license=LICENSE,
            ))
    missing = sorted(set(mapping) - {r.raw_labels[0] for r in rows})
    if missing:
        raise CollectorError(f"esc50_to_audioset.yaml lists categories not in ESC-50: {missing}")
    return rows

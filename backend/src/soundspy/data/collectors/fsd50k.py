"""FSD50K collector (``<external>/fsd50k``).

* ``FSD50K.ground_truth/dev.csv``: ``fname, labels, mids, split`` (split ``train``/``val``);
  ``eval.csv``: ``fname, labels, mids``. Labels and mids are comma-separated, in the same order,
  and **smeared**: ancestors are included (``Speech`` also carries ``Human_voice``).
* ``FSD50K.ground_truth/vocabulary.csv``: ``index, label, mid`` (no header). Used to map labels to
  mids (``Walk_and_footsteps`` -> ``/m/07pbtc8``), never by spelling.
* ``FSD50K.metadata/{dev,eval}_clips_info_FSD50K.json``: per clip ``license`` and ``uploader``.

Split: eval -> test; dev -> train/val by FSD50K's proposed split, then made uploader-disjoint
(``splits.majority_split``; the proposed split is kept in ``extra["official_split"]``).
Group: the uploader.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from pathlib import Path

from soundspy.data.aliases import dataset_root, to_alias
from soundspy.data.collectors import ClipRow, CollectorError
from soundspy.data.ontology import Ontology, fsd50k_label_map
from soundspy.data.splits import fsd50k_split, majority_split

_LICENSES = (
    (r"publicdomain/zero", "CC0"),
    (r"/by-nc/", "CC-BY-NC"),
    (r"/by/", "CC-BY"),
    (r"sampling\+", "Sampling+"),
)


def short_license(url: str) -> str:
    """``http://creativecommons.org/licenses/by-nc/3.0/`` -> ``CC-BY-NC 3.0``."""
    for pattern, name in _LICENSES:
        if re.search(pattern, url or ""):
            version = re.search(r"/(\d\.\d)/?$", url.strip())
            return f"{name} {version.group(1)}" if version else name
    return url or "unknown"


def read_vocabulary(path: Path) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return [(row[1], row[2]) for row in csv.reader(fh) if len(row) >= 3]


def _split_list(value: str) -> list[str]:
    return [v for v in (value or "").split(",") if v]


def collect(paths: Mapping[str, Path], ontology: Ontology | None = None) -> list[ClipRow]:
    root = dataset_root("fsd50k", paths)
    truth, meta = root / "FSD50K.ground_truth", root / "FSD50K.metadata"
    if not (truth / "dev.csv").exists():
        raise CollectorError(f"FSD50K ground truth not found in {truth}")
    if ontology is not None and (truth / "vocabulary.csv").exists():
        # Every FSD50K label must map to a known mid (raises otherwise).
        fsd50k_label_map(read_vocabulary(truth / "vocabulary.csv"), ontology)
    rows = []
    for part in ("dev", "eval"):
        info_path = meta / f"{part}_clips_info_FSD50K.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        with open(truth / f"{part}.csv", encoding="utf-8", newline="") as fh:
            for rec in csv.DictReader(fh):
                fname = str(rec["fname"]).strip()
                labels, mids = _split_list(rec["labels"]), _split_list(rec["mids"])
                if len(labels) != len(mids):
                    raise CollectorError(f"FSD50K {part}.csv: {fname} labels/mids mismatch")
                clip_info = info.get(fname, {})
                uploader = clip_info.get("uploader")
                rows.append(ClipRow(
                    clip_id=f"fsd50k:{fname}",
                    dataset="fsd50k",
                    path=to_alias("fsd50k", f"FSD50K.{part}_audio/{fname}.wav"),
                    file=root / f"FSD50K.{part}_audio" / f"{fname}.wav",
                    mids=mids,
                    raw_labels=labels,
                    split=fsd50k_split(part, rec.get("split")),
                    group=f"uploader:{uploader}" if uploader else f"clip:{fname}",
                    license=short_license(str(clip_info.get("license", ""))),
                ))
    dev = [r for r in rows if r.split in ("train", "val")]
    sides = majority_split((r.group, r.split) for r in dev)
    for row in dev:
        row.extra["official_split"] = row.split
        row.split = sides[row.group]
    return rows

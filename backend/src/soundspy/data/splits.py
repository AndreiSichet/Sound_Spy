"""Train / val / test splits without leakage.

Clips recorded by the same person or cut from the same recording sound alike; if they ended up
in train *and* test, the test score would be too optimistic. Every clip carries a ``group`` and
a group never spans two splits:

* **FSD50K** (group = uploader): ``eval.csv`` -> test (official; dev and eval share no
  uploader). ``dev.csv`` ``train``/``val`` is FSD50K's *proposed* split, which does share
  uploaders (926 of them on the real data), so each uploader goes wholly to the side holding
  most of its dev clips (:func:`majority_split`). 2,010 of 40,966 dev clips change side.
* **ESC-50** (group = source recording, the file-name part before ``-A``/``-B``): the official
  folds from the data config (``splits.esc50``); a fold keeps all takes of a recording together.
* **custom** (group = recording session): sessions are shuffled with a fixed seed and assigned
  whole to test, val, train (``splits.custom``).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

SPLITS = ("train", "val", "test")


class SplitError(ValueError):
    pass


def fsd50k_split(part: str, dev_split: str | None) -> str:
    """``part`` is ``"dev"`` or ``"eval"``; ``dev_split`` is dev.csv's ``split`` column."""
    if part == "eval":
        return "test"
    if part == "dev" and dev_split in ("train", "val"):
        return dev_split
    raise SplitError(f"FSD50K: unexpected part/split {part!r}/{dev_split!r}")


def majority_split(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """``(group, split)`` pairs -> one split per group: the split holding most of its clips.

    Used for FSD50K's dev set: its proposed train/val split shares uploaders (only dev vs. eval
    are uploader-disjoint), so every uploader is moved wholly to the side where it already has
    more clips (ties -> train). This keeps as much of the official split as possible while no
    uploader is in both train and val.
    """
    counts: dict[str, dict[str, int]] = {}
    for group, split in pairs:
        per = counts.setdefault(group, {})
        per[split] = per.get(split, 0) + 1
    order = {s: i for i, s in enumerate(SPLITS)}
    return {
        group: max(per, key=lambda s: (per[s], -order.get(s, 99)))
        for group, per in counts.items()
    }


def esc50_split(fold: int, folds: Mapping[str, Iterable[int]]) -> str:
    for split, fold_list in folds.items():
        if fold in fold_list:
            return split
    raise SplitError(f"ESC-50 fold {fold} is not assigned to any split in splits.esc50")


def esc50_group(filename: str) -> str:
    """``1-100032-A-0.wav`` -> ``1-100032`` (fold + freesound source id)."""
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 4:
        raise SplitError(f"unexpected ESC-50 file name {filename!r}")
    return "-".join(parts[:2])


def split_groups(
    groups: Iterable[str], *, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    """Assign whole groups (sessions) to splits, deterministic for ``seed``.

    With few groups every non-zero fraction still gets at least one group when there are
    enough groups to leave one for training.
    """
    unique = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_test = round(n * test_fraction)
    n_val = round(n * val_fraction)
    if n >= 3:
        n_test = max(n_test, 1 if test_fraction > 0 else 0)
        n_val = max(n_val, 1 if val_fraction > 0 else 0)
    while n_test + n_val >= n and (n_test or n_val):
        if n_val >= n_test and n_val:
            n_val -= 1
        else:
            n_test -= 1
    out = {}
    for i, group in enumerate(unique):
        out[group] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def leaking_groups(rows: Iterable[tuple[str, str, str]]) -> dict[tuple[str, str], set[str]]:
    """``(dataset, group) -> {splits}`` for every group found in more than one split."""
    seen: dict[tuple[str, str], set[str]] = {}
    for dataset, group, split in rows:
        seen.setdefault((dataset, group), set()).add(split)
    return {key: splits for key, splits in seen.items() if len(splits) > 1}

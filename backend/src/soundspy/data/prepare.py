"""``soundspy data prepare``: build the unified clip index and the activity maps.

1. Load the AudioSet ontology and check every ``audioset`` name in ``classes.yaml``.
2. Run the collectors of the enabled, downloaded datasets (ESC-50, FSD50K, custom).
3. Read every new clip once, in parallel: header facts + RMS envelope (``activity.py``).
   Already analysed clips are skipped, so a second run takes seconds.
4. Decide each clip's role (``roles.py``) and check that no leakage group spans two splits.
5. Write ``<processed>/clip_index.parquet`` and a CSV copy for inspection.

Index columns: ``clip_id, dataset, path, duration_s, sample_rate, channels, mids, raw_labels,
role, split, group, license``. ``mids`` / ``raw_labels`` are lists (``;``-joined in the CSV);
``path`` is an alias resolved with ``soundspy.data.aliases.resolve_alias``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from soundspy.config import AppConfig
from soundspy.data.activity import ActivityStore, compute_missing
from soundspy.data.collectors import ClipRow, CollectorError
from soundspy.data.download import is_done
from soundspy.data.ontology import Ontology, OntologyError, class_mids
from soundspy.data.roles import BG, EXCLUDED, RoleRules
from soundspy.data.splits import leaking_groups

INDEX_FILE = "clip_index.parquet"
INDEX_CSV = "clip_index.csv"
ACTIVITY_DIR = "activity"
COLUMNS = ("clip_id", "dataset", "path", "duration_s", "sample_rate", "channels", "mids",
           "raw_labels", "role", "split", "group", "license")


class PrepareError(RuntimeError):
    pass


@dataclass
class PrepareSummary:
    index_path: Path
    clips: Counter = field(default_factory=Counter)  # per dataset
    roles: Counter = field(default_factory=Counter)  # per role
    skipped: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    overlaps: Counter = field(default_factory=Counter)  # (class, class, ...) -> clips
    mixed: int = 0
    moved_split: int = 0  # FSD50K dev clips moved to keep uploaders out of both train and val
    bg_with_ancestor_labels: Counter = field(default_factory=Counter)  # label -> bg clips

    def text(self) -> str:
        lines = [f"clip index: {self.index_path}"]
        for ds, n in sorted(self.clips.items()):
            lines.append(f"  {ds:<7} {n:>6} clips")
        roles = ", ".join(f"{r} {n}" for r, n in sorted(self.roles.items()))
        lines.append(f"roles: {roles}")
        for msg in self.skipped:
            lines.append(f"skipped: {msg}")
        if self.unreadable:
            lines.append(f"unreadable files (left out): {len(self.unreadable)}, e.g. "
                         f"{', '.join(self.unreadable[:5])}")
        if self.moved_split:
            lines.append(f"FSD50K dev clips moved between train and val (uploader-disjoint): "
                         f"{self.moved_split}")
        lines.append(f"excluded as class + other sounds: {self.mixed}")
        for classes, n in self.overlaps.most_common():
            lines.append(f"excluded as overlap {' + '.join(classes)}: {n} clips")
        if self.bg_with_ancestor_labels:
            top = ", ".join(f"{lab} ({n})" for lab, n in
                            self.bg_with_ancestor_labels.most_common(8))
            lines.append(f"background clips carrying a broad parent label of a class: {top}")
        return "\n".join(lines)


def ontology_path(config: AppConfig) -> Path:
    return config.paths["external"] / "ontology" / "ontology.json"


def load_ontology(config: AppConfig) -> Ontology:
    try:
        return Ontology.load(ontology_path(config))
    except OntologyError as exc:
        raise PrepareError(str(exc)) from exc


def collect_rows(
    config: AppConfig, ontology: Ontology, summary: PrepareSummary
) -> list[ClipRow]:
    from soundspy.data.collectors import custom, esc50, fsd50k

    data, paths = config.data, config.paths
    try:
        mids = class_mids(config.labels, ontology)
    except OntologyError as exc:
        raise PrepareError(str(exc)) from exc
    rows: list[ClipRow] = []
    try:
        if data.enabled["esc50"]:
            if is_done(paths["external"], "esc50"):
                mapping = esc50.load_mapping(
                    config.configs_dir / "data" / "esc50_to_audioset.yaml", ontology)
                rows += esc50.collect(paths, mapping, data.esc50_folds)
            else:
                summary.skipped.append("esc50 not downloaded (soundspy data download esc50)")
        if data.enabled["fsd50k"]:
            if is_done(paths["external"], "fsd50k"):
                rows += fsd50k.collect(paths, ontology)
            else:
                summary.skipped.append(
                    "fsd50k not downloaded/extracted (soundspy data download fsd50k)")
        if data.enabled["custom"]:
            rows += custom.collect(
                paths, config.labels, mids, val_fraction=data.custom_val_fraction,
                test_fraction=data.custom_test_fraction, seed=data.custom_seed)
    except (CollectorError, OntologyError) as exc:
        raise PrepareError(str(exc)) from exc
    return rows


def activity_path(processed: Path, dataset: str) -> Path:
    return Path(processed) / ACTIVITY_DIR / f"{dataset}.npz"


def prepare(
    config: AppConfig, *, workers: int | None = None, progress: bool = True
) -> PrepareSummary:
    import pandas as pd

    processed = config.paths["processed"]
    summary = PrepareSummary(index_path=processed / INDEX_FILE)
    ontology = load_ontology(config)
    rules = RoleRules.build(config.labels, ontology)
    rows = collect_rows(config, ontology, summary)

    by_dataset: dict[str, list[ClipRow]] = {}
    for row in rows:
        by_dataset.setdefault(row.dataset, []).append(row)
    stores: dict[str, ActivityStore] = {}
    unreadable: set[str] = set()
    for dataset, ds_rows in by_dataset.items():
        path = activity_path(processed, dataset)
        store = ActivityStore.load(path, config.data.activity_hop_s)
        store.keep_only(r.clip_id for r in ds_rows)
        failed = compute_missing(store, {r.clip_id: r.file for r in ds_rows}, workers=workers,
                                 progress=progress, desc=f"{dataset} activity")
        unreadable.update(failed)
        store.save(path)
        stores[dataset] = store
    summary.unreadable = sorted(unreadable)

    ancestors_of_classes = frozenset().union(*rules.allowed.values()) - frozenset().union(
        *rules.fg.values())
    records = []
    for row in rows:
        if row.clip_id in unreadable:
            continue
        facts = stores[row.dataset].facts(row.clip_id)
        if row.role is not None:
            role = row.role
        else:
            decision = rules.decide(row.mids)
            role = decision.role
            if decision.overlap:
                summary.overlaps[decision.overlap] += 1
            elif role == EXCLUDED and row.mids:
                summary.mixed += 1
            if role == BG:
                for label, mid in zip(row.raw_labels, row.mids, strict=False):
                    if mid in ancestors_of_classes:
                        summary.bg_with_ancestor_labels[label] += 1
        if row.extra.get("official_split", row.split) != row.split:
            summary.moved_split += 1
        summary.clips[row.dataset] += 1
        summary.roles[role] += 1
        records.append({
            "clip_id": row.clip_id,
            "dataset": row.dataset,
            "path": row.path,
            "duration_s": facts.frames / facts.sample_rate,
            "sample_rate": facts.sample_rate,
            "channels": facts.channels,
            "mids": list(row.mids),
            "raw_labels": list(row.raw_labels),
            "role": role,
            "split": row.split,
            "group": row.group,
            "license": row.license,
        })

    leaks = leaking_groups((r["dataset"], r["group"], r["split"]) for r in records)
    if leaks:
        sample = ", ".join(f"{ds}:{g} in {sorted(s)}" for (ds, g), s in list(leaks.items())[:5])
        raise PrepareError(f"{len(leaks)} leakage group(s) span several splits: {sample}")
    if not records:
        raise PrepareError("no clips found: download a dataset first "
                           "(soundspy data download ontology esc50)")

    frame = pd.DataFrame.from_records(records, columns=list(COLUMNS))
    processed.mkdir(parents=True, exist_ok=True)
    tmp = processed / (INDEX_FILE + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(processed / INDEX_FILE)
    csv_frame = frame.copy()
    for col in ("mids", "raw_labels"):
        csv_frame[col] = csv_frame[col].map(";".join)
    csv_frame.to_csv(processed / INDEX_CSV, index=False)
    return summary


def load_clip_index(processed: Path):
    """The clip index as a pandas DataFrame (``mids``/``raw_labels`` as lists)."""
    import pandas as pd

    path = Path(processed) / INDEX_FILE
    if not path.exists():
        raise PrepareError(f"{path} not found: run `soundspy data prepare` first")
    frame = pd.read_parquet(path)
    for col in ("mids", "raw_labels"):
        frame[col] = frame[col].map(list)
    return frame


def load_activity(processed: Path, datasets=("esc50", "fsd50k", "custom")
                  ) -> dict[str, ActivityStore]:
    """Activity stores per dataset (only the ones that exist)."""
    out = {}
    for ds in datasets:
        path = activity_path(processed, ds)
        if path.exists():
            out[ds] = ActivityStore.load(path)
    return out



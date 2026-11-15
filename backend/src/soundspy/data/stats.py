"""``soundspy data stats``: coverage of the clip index (class x split x dataset)."""

from __future__ import annotations

from soundspy.config import AppConfig
from soundspy.data.roles import BG, EXCLUDED, role_class
from soundspy.data.splits import SPLITS


def coverage_table(index) -> list[dict]:
    """One dict per (row label, dataset) with ``{split: (clips, minutes)}``."""
    out = []
    frame = index.assign(label=index["role"].map(lambda r: role_class(r) or r))
    grouped = frame.groupby(["label", "dataset", "split"])["duration_s"].agg(["count", "sum"])
    for (label, dataset), sub in grouped.groupby(level=[0, 1]):
        cells = {split: (0, 0.0) for split in SPLITS}
        for (_l, _d, split), row in sub.iterrows():
            cells[split] = (int(row["count"]), float(row["sum"]) / 60.0)
        out.append({"label": label, "dataset": dataset, **cells})
    return out


def coverage_report(index, config: AppConfig) -> str:
    table = coverage_table(index)
    order = [*config.labels, BG, EXCLUDED]
    labels = sorted({r["label"] for r in table},
                    key=lambda lab: (order.index(lab) if lab in order else len(order), lab))
    header = f"{'class':<13}{'dataset':<9}" + "".join(
        f"{s + ' clips':>12}{'min':>8}" for s in SPLITS)
    lines = [header, "-" * len(header)]
    train_fg: dict[str, int] = {key: 0 for key in config.labels}
    for label in labels:
        rows = [r for r in table if r["label"] == label]
        totals = {s: [0, 0.0] for s in SPLITS}
        for r in sorted(rows, key=lambda r: r["dataset"]):
            cells = ""
            for s in SPLITS:
                n, minutes = r[s]
                totals[s][0] += n
                totals[s][1] += minutes
                cells += f"{n:>12}{minutes:>8.1f}"
            lines.append(f"{label:<13}{r['dataset']:<9}{cells}")
        if len(rows) > 1:
            cells = "".join(f"{totals[s][0]:>12}{totals[s][1]:>8.1f}" for s in SPLITS)
            lines.append(f"{label:<13}{'all':<9}{cells}")
        if label in train_fg:
            train_fg[label] = totals["train"][0]
        lines.append("")
    threshold = config.data.min_fg_clips_warning
    thin = [(k, n) for k, n in train_fg.items() if n < threshold]
    if thin:
        lines.append(f"WARNING: classes with fewer than {threshold} train foreground clips:")
        for key, n in thin:
            lines.append(f"  {key}: {n}")
    else:
        lines.append(f"Every class has at least {threshold} train foreground clips.")
    return "\n".join(lines)

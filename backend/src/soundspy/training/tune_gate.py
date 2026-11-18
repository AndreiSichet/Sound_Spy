"""Choosing the output gate's thresholds on the validation set (``soundspy evaluate gate``).

The model runs once over every validation example (the cached 44.1 kHz set the training
validates on: the first ``val_examples`` plus the target-absent ones after them); its estimates
are kept, and every gate setting of the grid is applied to them afterwards (the gate only
depends on the estimate and the input, so this gives exactly what a session would give). Per
setting, at the evaluation's attenuation (30 dB) and at ``high_attenuation_db`` (40 dB):

* **SI-SDRi** over the examples with a target (what the gate may cost), and the number of
  examples made more than 5 dB worse;
* **suppression** over the target-absent examples (what the gate is for), as the system output
  against the mixture, and the bare estimate against the mixture.

The test set is never used for this (only the final evaluation runs on it).
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.config import AppConfig, NeuralGateConfig
from soundspy.data.cache import ExtractionCache, split_dir
from soundspy.extraction import metrics as M
from soundspy.extraction.gate import OutputGate
from soundspy.extraction.neural_tse import NeuralCore, residual_mix

DEFAULT_GRID = {"open_db": (-8.0, -12.0, -16.0), "gap_db": (6.0, 10.0),
                "depth_db": (15.0, 25.0, 35.0)}


def grid_configs(base: NeuralGateConfig, grid: dict[str, Sequence[float]] | None = None
                 ) -> list[NeuralGateConfig]:
    """Every combination of ``open_db``, ``gap_db`` (close = open - gap) and ``depth_db``."""
    grid = grid or DEFAULT_GRID
    out = []
    for open_db, gap, depth in itertools.product(grid["open_db"], grid["gap_db"],
                                                 grid["depth_db"]):
        out.append(dataclasses.replace(base, enabled=True, open_db=float(open_db),
                                       close_db=float(open_db - gap), depth_db=float(depth)))
    return out


def tune_gate(config: AppConfig, core: NeuralCore, *, n: int | None = None,
              grid: dict[str, Sequence[float]] | None = None,
              progress: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = config.extraction.neural_tse
    ev = config.extraction.evaluation
    cache = ExtractionCache(split_dir(config, core.sample_rate, "val"))
    n = cfg.training.val_examples if n is None else n
    extra = [int(i) for i in np.flatnonzero(cache.empty)
             if i >= n][:cfg.training.val_empty_examples]
    indices = [*range(min(n, len(cache))), *extra]
    started = time.perf_counter()
    examples = []
    for count, i in enumerate(indices, 1):
        pair = np.asarray(cache.pair(i), dtype=np.float32)
        mixture, target = pair[0], pair[1]
        label = cache.target_hot[i].astype(np.float32)
        estimate = core.whole(mixture, label)[:mixture.shape[0]]
        examples.append((mixture, target, estimate, bool(M.is_silent(target))))
        if count % 100 == 0:
            progress(f"  model over {count}/{len(indices)} validation examples "
                     f"({time.perf_counter() - started:.0f} s)")
    settings = [None, *grid_configs(cfg.gate, grid)]
    rows = []
    for gate in settings:
        row: dict[str, Any] = {"gate": None if gate is None else
                               {k: getattr(gate, k) for k in ("open_db", "close_db",
                                                              "depth_db")}}
        for db in (ev.attenuation_db, ev.high_attenuation_db):
            si, worse, suppression, bare = [], 0, [], []
            for mixture, target, estimate, empty in examples:
                est = estimate if gate is None else OutputGate(
                    gate, core.sample_rate, core.hop).process(estimate, mixture)
                out = residual_mix(est, mixture, db)
                if empty:
                    suppression.append(M.suppression_db(out, mixture))
                    bare.append(M.suppression_db(est, mixture))
                else:
                    value = M.si_sdr_improvement(out, mixture, target)
                    si.append(value)
                    worse += value < -5.0
            row[f"{db:g}"] = {"si_sdr_i": float(np.mean(si)), "worse": int(worse),
                              "suppression_db": float(np.mean(suppression)),
                              "estimate_db": float(np.mean(bare)),
                              "with_target": len(si), "without_target": len(suppression)}
        rows.append(row)
    return {"model": core.name, "checkpoint": str(core.checkpoint), "examples": len(indices),
            "attenuations_db": [ev.attenuation_db, ev.high_attenuation_db],
            "rows": rows, "seconds": round(time.perf_counter() - started, 1)}


def choose(result: dict[str, Any], max_cost_db: float = 0.3) -> dict[str, Any] | None:
    """The setting with the strongest suppression at 30 dB among those that cost at most
    ``max_cost_db`` of SI-SDRi against no gate (``None`` if none qualifies)."""
    key = f"{result['attenuations_db'][0]:g}"
    rows = result["rows"]
    off = next(r for r in rows if r["gate"] is None)[key]["si_sdr_i"]
    ok = [r for r in rows if r["gate"] is not None and off - r[key]["si_sdr_i"] <= max_cost_db]
    return min(ok, key=lambda r: r[key]["suppression_db"]) if ok else None


def format_table(result: dict[str, Any]) -> str:
    low, high = (f"{a:g}" for a in result["attenuations_db"])
    lines = [f"| Gate (open / close / depth dB) | SI-SDRi at {low} dB | worse by > 5 dB | "
             f"suppression at {low} dB | at {high} dB | estimate alone |",
             "|---|---|---|---|---|---|"]
    for row in result["rows"]:
        g = row["gate"]
        name = "off" if g is None else f"{g['open_db']:g} / {g['close_db']:g} / {g['depth_db']:g}"
        a, b = row[low], row[high]
        lines.append(f"| {name} | {a['si_sdr_i']:+.2f} | {a['worse']} | "
                     f"{a['suppression_db']:.1f} | {b['suppression_db']:.1f} | "
                     f"{a['estimate_db']:.1f} |")
    return "\n".join(lines)


def run(config: AppConfig, checkpoint: Path, device: str = "auto", *, n: int | None = None,
        progress: Callable[[str], None] = print) -> dict[str, Any]:
    from soundspy.training.experiment import write_json

    core = NeuralCore.from_checkpoint(Path(checkpoint), device)
    result = tune_gate(config, core, n=n, progress=progress)
    result["chosen"] = choose(result)
    folder = Path(checkpoint).parent.parent
    write_json(folder / "gate_tuning.json", result)
    (folder / "gate_tuning.md").write_text(format_table(result) + "\n", encoding="utf-8")
    result["path"] = str(folder / "gate_tuning.json")
    return result

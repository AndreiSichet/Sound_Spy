"""Evaluation of extraction (``soundspy evaluate extraction``).

A fixed, seeded subset of a split's recipes is rendered at the processing rate; every recipe
gives ``mixture == target + rest`` exactly, so the correct output (the target alone) is known.
Five systems process the same mixtures:

==========  ====================================================================================
mixture     no processing: the reference every improvement is measured against
timeline    the baseline with the offline detector (windows centred on each time, no lag)
live        the baseline with the live detector's lag reproduced (causal windows, EMA, hysteresis)
oracle      the baseline with the **true** target activity: what a perfect detector would give
irm         the ideal ratio mask from the true target and rest, in the same STFT: the upper bound
            of any single-channel masking method
==========  ====================================================================================

So ``oracle - live`` is what detector errors and lag cost, and ``irm - oracle`` is what the
method itself leaves on the table.

Metrics (``extraction/metrics.py``), all on delay-compensated outputs: SI-SDR, SI-SDRi and SNRi
for recipes with a target; **suppression** (output energy relative to the mixture) for recipes
whose target class is absent, where the correct output is silence. Two diagnostics explain the
means: the **median** SI-SDRi (a few disastrous recipes dominate a mean), and **missed**: the
share of recipes with a target whose whole output sits at the attenuation floor, i.e. the gate
never opened. SI-SDR cannot see that case (a uniformly quieter mixture scores exactly 0 dB
improvement), so it is counted separately.

Outputs: ``experiments/ext_<method>/`` with ``metrics.json`` (overall and broken down by target
class, target-to-rest level, number of foreground sources), ``metrics.md`` (the same as tables),
``per_recipe.csv`` and ``config.json``; and ``outputs/eval/ext_<method>/`` with a few rendered
examples (mixture, target, every system's output) and an index describing them.
"""

from __future__ import annotations

import csv
import dataclasses
import random
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from soundspy import __version__
from soundspy.config import AppConfig
from soundspy.core.target import TargetSpec
from soundspy.data.mixtures import (
    ClipLibrary,
    Recipe,
    iter_recipes,
    recipes_path,
    render_with_config,
)
from soundspy.extraction import metrics as M
from soundspy.extraction.activity import (
    ActivitySource,
    OracleActivity,
    causal_timeline,
    centred_timeline,
    timeline_probabilities,
)
from soundspy.extraction.mask_baseline import MaskBaselineExtractor
from soundspy.extraction.offline import irm_oracle, run_offline
from soundspy.training.experiment import git_commit, read_json, write_json

SYSTEMS = ("mixture", "timeline", "live", "oracle", "irm")
SYSTEM_NAMES = {
    "mixture": "mixture (no processing)",
    "timeline": "mask_baseline + timeline",
    "live": "mask_baseline + live",
    "oracle": "mask_baseline + oracle",
    "irm": "IRM oracle",
}
EVALUATED_METHODS = ("mask_baseline",)
T2R_BINS = ("<= -5 dB", "-5 .. 5 dB", ">= 5 dB")
DURATION_BINS = ("< 1 s", "1 .. 3 s", ">= 3 s")
MISSED_MARGIN_DB = 1.0  # output within this of the floor everywhere = the gate never opened


class ExtractionEvalError(RuntimeError):
    pass


def experiment_name(method: str) -> str:
    return f"ext_{method}"


def select_recipes(recipes: list[Recipe], n: int, seed: int) -> list[Recipe]:
    """A fixed subset: ``n`` recipes drawn with ``seed``, kept in file order."""
    if n >= len(recipes):
        return list(recipes)
    chosen = sorted(random.Random(seed).sample(range(len(recipes)), n))
    return [recipes[i] for i in chosen]


def t2r_bin(target_to_rest_db: float) -> str:
    if target_to_rest_db <= -5.0:
        return T2R_BINS[0]
    return T2R_BINS[2] if target_to_rest_db >= 5.0 else T2R_BINS[1]


def duration_bin(active_s: float) -> str:
    if active_s < 1.0:
        return DURATION_BINS[0]
    return DURATION_BINS[2] if active_s >= 3.0 else DURATION_BINS[1]


def target_active_seconds(rendered) -> float:
    """How long any target class really sounds in the rendered recipe."""
    labels = list(rendered.recipe.target)
    columns = np.flatnonzero(rendered.target_hot > 0.5)
    if not labels or not columns.size:
        return 0.0
    active = rendered.class_activity.T[columns].max(axis=0) > 0.5
    return float(active.sum() / rendered.activity_rate_hz)


def foreground_sources(recipe: Recipe) -> int:
    return sum(1 for e in recipe.events if e.role == "fg")


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(float(np.mean(values)), 3) if values else None


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(float(np.median(values)), 3) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-system means over ``records`` (targets and empty targets separately)."""
    with_target = [r for r in records if not r["empty"]]
    empty = [r for r in records if r["empty"]]
    out: dict[str, Any] = {"n": len(with_target), "n_empty": len(empty)}
    for system in SYSTEMS:
        out[system] = {
            "si_sdr": _mean(r[system]["si_sdr"] for r in with_target),
            "si_sdr_i": _mean(r[system]["si_sdr_i"] for r in with_target),
            "si_sdr_i_median": _median(r[system]["si_sdr_i"] for r in with_target),
            "snr_i": _mean(r[system]["snr_i"] for r in with_target),
            "missed": _mean(float(r[system]["missed"]) for r in with_target),
            "suppression_db": _mean(r[system]["suppression_db"] for r in empty),
        }
    return out


def breakdowns(records: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    with_target = [r for r in records if not r["empty"]]
    by_class = {lab: summarize([r for r in with_target if lab in r["target"]]) for lab in labels}
    by_t2r = {b: summarize([r for r in with_target if r["t2r_bin"] == b]) for b in T2R_BINS}
    counts = sorted({r["foreground"] for r in with_target})
    by_fg = {str(c): summarize([r for r in with_target if r["foreground"] == c]) for c in counts}
    empty_by_class = {lab: summarize([r for r in records if r["empty"] and lab in r["target"]])
                      for lab in labels}
    by_duration = {b: summarize([r for r in with_target if r["duration_bin"] == b])
                   for b in DURATION_BINS}
    return {"by_target_class": by_class, "by_target_to_rest_db": by_t2r,
            "by_foreground_sources": by_fg, "by_target_duration": by_duration,
            "empty_by_target_class": empty_by_class}


# --------------------------------------------------------------------------------------------
# One recipe


def process_recipe(config: AppConfig, rendered, core, *, attenuation_db: float
                   ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Every system's output for one rendered recipe (1-D arrays) and the time each part took."""
    det, ext, cfg = config.detection, config.extraction, config.extraction.mask_baseline
    recipe = rendered.recipe
    mixture = rendered.mixture.samples.ravel()
    target = TargetSpec(labels=frozenset(recipe.target), attenuation_db=attenuation_db)
    live = det.live
    times: dict[str, float] = {}

    started = time.perf_counter()
    raw_centred, raw_causal = timeline_probabilities(core, rendered.mixture, live.hop_s)
    settings = dict(hop_s=live.hop_s, ema_alpha=live.ema_alpha, hysteresis=live.hysteresis)
    sources: dict[str, ActivitySource] = {
        "timeline": centred_timeline(core, rendered.mixture, raw=raw_centred, **settings),
        "live": causal_timeline(core, rendered.mixture, raw=raw_causal,
                                delay_s=ext.evaluation.live_delay_s, lag_s=cfg.noise_guard_s,
                                **settings),
        "oracle": OracleActivity(list(config.labels), rendered.class_activity,
                                 rendered.activity_rate_hz),
    }
    times["detector"] = time.perf_counter() - started

    started = time.perf_counter()
    outputs = {"mixture": mixture}
    for system, source in sources.items():
        extractor = MaskBaselineExtractor(cfg, source)
        outputs[system] = run_offline(extractor, rendered.mixture, target,
                                      chunk_frames=ext.offline_chunk_frames).samples.ravel()
    times["extractor"] = time.perf_counter() - started
    outputs["irm"] = irm_oracle(mixture, rendered.target.samples.ravel(),
                                rendered.rest.samples.ravel(), cfg)
    return outputs, times


def score(outputs: dict[str, np.ndarray], mixture: np.ndarray, reference: np.ndarray,
          attenuation_db: float) -> dict[str, dict[str, Any]]:
    empty = M.is_silent(reference)
    scores: dict[str, dict[str, Any]] = {}
    for system, out in outputs.items():
        level = M.suppression_db(out, mixture)
        if empty:
            scores[system] = {"si_sdr": None, "si_sdr_i": None, "snr_i": None, "missed": None,
                              "suppression_db": round(level, 3)}
        else:
            scores[system] = {
                "si_sdr": round(M.si_sdr(out, reference), 3),
                "si_sdr_i": round(M.si_sdr_improvement(out, mixture, reference), 3),
                "snr_i": round(M.snr_improvement(out, mixture, reference), 3),
                # the whole output at the floor: the target was there but the gate never opened
                "missed": bool(attenuation_db > MISSED_MARGIN_DB
                               and level <= -(attenuation_db - MISSED_MARGIN_DB)),
                "suppression_db": None,
            }
    return scores


# --------------------------------------------------------------------------------------------
# The whole evaluation


def evaluate_extraction(config: AppConfig, method: str = "mask_baseline", split: str = "test", *,
                        n: int | None = None, sample_rate: int | None = None,
                        device: str = "auto", examples: int | None = None,
                        progress: Callable[[str], None] = print) -> dict[str, Any]:
    if method not in EVALUATED_METHODS:
        raise ExtractionEvalError(f"unknown method {method!r}; methods: {list(EVALUATED_METHODS)}")
    ev, det = config.extraction.evaluation, config.detection
    n = ev.n if n is None else n
    sample_rate = ev.sample_rate if sample_rate is None else sample_rate
    examples = ev.examples if examples is None else examples
    if det is None or det.checkpoint is None:
        raise ExtractionEvalError("no detection model is configured (detection.checkpoint): "
                                  "the timeline and live systems need it")
    from soundspy.detection.inference import DetectorUnavailable, load_core

    try:
        core = load_core(Path(det.checkpoint), device if device != "auto" else det.live.device)
    except DetectorUnavailable as exc:
        raise ExtractionEvalError(str(exc)) from exc

    path = recipes_path(config.paths["mixtures"], split)
    recipes = list(iter_recipes(path)) if path.is_file() else []
    if not recipes:
        raise ExtractionEvalError(f"no {split} recipes: run `soundspy data mix` first")
    chosen = select_recipes(recipes, n, ev.seed)
    progress(f"{len(chosen)} of {len(recipes)} {split} recipes at {sample_rate} Hz; "
             f"detector {core.name} on {core.device}")

    library = ClipLibrary.from_config(config)
    out_dir = config.paths["outputs"] / "eval" / experiment_name(method)
    want = _example_ids(chosen, examples)
    if want:
        _clear_examples(out_dir)  # the folder then holds exactly what its index describes
    records: list[dict[str, Any]] = []
    listening: list[dict[str, Any]] = []
    seconds = {"render": 0.0, "detector": 0.0, "extractor": 0.0}
    started = time.perf_counter()
    for i, recipe in enumerate(chosen):
        t0 = time.perf_counter()
        rendered = render_with_config(recipe, sample_rate, library, config,
                                      activity_rate_hz=det.activity_rate_hz)
        seconds["render"] += time.perf_counter() - t0
        outputs, times = process_recipe(config, rendered, core,
                                        attenuation_db=ev.attenuation_db)
        for key, value in times.items():
            seconds[key] += value
        mixture = rendered.mixture.samples.ravel()
        reference = rendered.target.samples.ravel()
        scores = score(outputs, mixture, reference, ev.attenuation_db)
        active_s = target_active_seconds(rendered)
        record = {
            "id": recipe.id, "empty": M.is_silent(reference), "target": list(recipe.target),
            "present": list(recipe.present), "target_to_rest_db": recipe.target_to_rest_db,
            "t2r_bin": t2r_bin(recipe.target_to_rest_db), "foreground": foreground_sources(recipe),
            "target_active_s": round(active_s, 2), "duration_bin": duration_bin(active_s),
            **scores,
        }
        records.append(record)
        if recipe.id in want:
            listening.append(_write_example(out_dir, record, outputs, reference, sample_rate))
        if (i + 1) % 50 == 0 or i + 1 == len(chosen):
            progress(f"  {i + 1}/{len(chosen)} recipes ({time.perf_counter() - started:.0f} s)")

    total = time.perf_counter() - started
    result = {
        "method": method,
        "split": split,
        "recipes": len(records),
        "sample_rate": sample_rate,
        "attenuation_db": ev.attenuation_db,
        "seed": ev.seed,
        "detector": {"model": core.name, "device": str(core.device)},
        "seconds": {"total": round(total, 1), **{k: round(v, 1) for k, v in seconds.items()}},
        "audio_seconds": round(sum(r.duration_s for r in chosen), 1),
        "overall": summarize(records),
        **breakdowns(records, list(config.labels)),
    }
    exp = config.paths["experiments"] / experiment_name(method)
    exp.mkdir(parents=True, exist_ok=True)
    metrics_path = exp / "metrics.json"
    previous = read_json(metrics_path) if metrics_path.is_file() else {}
    write_json(metrics_path, {**previous, split: result})  # one entry per split
    (exp / "metrics.md").write_text(format_markdown(result, list(config.labels)), encoding="utf-8")
    write_per_recipe(exp / "per_recipe.csv", records)
    write_json(exp / "config.json", {
        "method": method,
        "extraction": dataclasses.asdict(config.extraction.mask_baseline),
        "evaluation": dataclasses.asdict(ev) | {"n": n, "sample_rate": sample_rate},
        "detection": {"checkpoint": str(det.checkpoint), "hop_s": det.live.hop_s,
                      "ema_alpha": det.live.ema_alpha, "hysteresis": det.live.hysteresis},
        "labels": list(config.labels),
        "app_version": __version__,
        "git": git_commit(config.project_root),
    })
    if listening:
        write_json(out_dir / "index.json", {"sample_rate": sample_rate, "systems": SYSTEM_NAMES,
                                            "examples": listening})
        (out_dir / "index.md").write_text(_examples_markdown(listening), encoding="utf-8")
    result["experiment_dir"] = str(exp)
    result["examples_dir"] = str(out_dir) if listening else None
    return result


def _example_ids(recipes: list[Recipe], examples: int) -> set[str]:
    """The first recipes of the subset, with a couple of empty-target ones among them."""
    if examples <= 0:
        return set()
    empties = [r.id for r in recipes if r.empty_target][: min(2, examples // 3)]
    others = [r.id for r in recipes if not r.empty_target][: examples - len(empties)]
    return set(empties) | set(others)


def _clear_examples(out_dir: Path) -> None:
    """Remove the example files a previous evaluation wrote here (another subset, other ids)."""
    if not out_dir.is_dir():
        return
    suffixes = tuple(f"_{name}.wav" for name in ("target", *SYSTEMS))
    for path in out_dir.iterdir():
        if path.name.endswith(suffixes) or path.name in ("index.json", "index.md"):
            path.unlink()


def _write_example(out_dir: Path, record: dict[str, Any], outputs: dict[str, np.ndarray],
                   reference: np.ndarray, sample_rate: int) -> dict[str, Any]:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"target": f"{record['id']}_target.wav"}
    sf.write(str(out_dir / files["target"]), reference, sample_rate, subtype="FLOAT")
    for system, samples in outputs.items():
        files[system] = f"{record['id']}_{system}.wav"
        sf.write(str(out_dir / files[system]), samples, sample_rate, subtype="FLOAT")
    return {**{k: record[k] for k in ("id", "empty", "target", "present", "target_to_rest_db",
                                      "foreground")},
            "files": files, "scores": {s: record[s] for s in SYSTEMS}}


def _examples_markdown(examples: list[dict[str, Any]]) -> str:
    lines = ["# Listening examples", "",
             "Each recipe: `_mixture` (input), `_target` (the correct output) and one file per "
             "system.", "",
             "| Recipe | Target | Also present | Target vs rest | SI-SDRi timeline / live / "
             "oracle / IRM (dB) |", "|---|---|---|---|---|"]
    for ex in examples:
        other = [p for p in ex["present"] if p not in ex["target"]] or ["(background only)"]
        if ex["empty"]:
            scores = "target absent; suppression " + " / ".join(
                f"{ex['scores'][s]['suppression_db']:.1f}"
                for s in ("timeline", "live", "oracle", "irm"))
        else:
            scores = " / ".join(f"{ex['scores'][s]['si_sdr_i']:+.1f}"
                                for s in ("timeline", "live", "oracle", "irm"))
        lines.append(f"| {ex['id']} | {', '.join(ex['target'])} | {', '.join(other)} | "
                     f"{ex['target_to_rest_db']:+.1f} dB | {scores} |")
    return "\n".join(lines) + "\n"


def write_per_recipe(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["id", "empty", "target", "present", "target_to_rest_db", "foreground",
              "target_active_s"]
    metrics = ("si_sdr", "si_sdr_i", "snr_i", "suppression_db")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields + [f"{s}_{m}" for s in SYSTEMS for m in metrics])
        for r in records:
            row = [r["id"], int(r["empty"]), "+".join(r["target"]), "+".join(r["present"]),
                   r["target_to_rest_db"], r["foreground"], r["target_active_s"]]
            row += ["" if r[s][m] is None else r[s][m] for s in SYSTEMS for m in metrics]
            writer.writerow(row)


# --------------------------------------------------------------------------------------------
# Tables


def _f(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _overall_table(summary: dict[str, Any]) -> list[str]:
    lines = ["| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | "
             "Target missed | Suppression, target absent (dB) |", "|---|---|---|---|---|---|---|"]
    for system in SYSTEMS:
        s = summary[system]
        missed = "-" if s["missed"] is None else f"{100 * s['missed']:.0f} %"
        lines.append(f"| {SYSTEM_NAMES[system]} | {_f(s['si_sdr'])} | {_f(s['si_sdr_i'])} | "
                     f"{_f(s['si_sdr_i_median'])} | {_f(s['snr_i'])} | {missed} | "
                     f"{_f(s['suppression_db'])} |")
    return lines


def _breakdown_table(title: str, groups: dict[str, Any], key: str = "si_sdr_i",
                     count: str = "n") -> list[str]:
    systems = [s for s in SYSTEMS if s != "mixture"]
    lines = [f"| {title} | recipes | " + " | ".join(SYSTEM_NAMES[s] for s in systems) + " |",
             "|---|---|" + "---|" * len(systems)]
    for name, summary in groups.items():
        if not summary[count]:
            continue
        cells = " | ".join(_f(summary[s][key]) for s in systems)
        lines.append(f"| {name} | {summary[count]} | {cells} |")
    return lines


def format_markdown(result: dict[str, Any], labels: list[str]) -> str:
    overall = result["overall"]
    sec = result["seconds"]
    lines = [
        f"# Extraction evaluation: {result['method']} ({result['split']})", "",
        f"{result['recipes']} recipes ({overall['n']} with a target, {overall['n_empty']} with "
        f"the target class absent), rendered at {result['sample_rate']} Hz, attenuation "
        f"{result['attenuation_db']:g} dB, detector {result['detector']['model']} on "
        f"{result['detector']['device']}. {result['audio_seconds']:.0f} s of audio in "
        f"{sec['total']:.0f} s (render {sec['render']:.0f} s, detector {sec['detector']:.0f} s, "
        f"extractor x3 {sec['extractor']:.0f} s).", "",
        "## Overall", "", *_overall_table(overall), "",
        "## SI-SDRi (dB) by target class", "",
        *_breakdown_table("Target class", result["by_target_class"]), "",
        "## SI-SDRi (dB) by target level relative to the rest", "",
        *_breakdown_table("Target vs rest", result["by_target_to_rest_db"]), "",
        "## SI-SDRi (dB) by number of foreground sources", "",
        *_breakdown_table("Foreground sources", result["by_foreground_sources"]), "",
        "## SI-SDRi (dB) by how long the target sounds", "",
        *_breakdown_table("Target active", result["by_target_duration"]), "",
        "## SNRi (dB) by target class", "",
        *_breakdown_table("Target class", result["by_target_class"], key="snr_i"), "",
        "## Suppression (dB) when the target class is absent, by selected class", "",
        *_breakdown_table("Selected class", result["empty_by_target_class"],
                          key="suppression_db", count="n_empty"), "",
    ]
    return "\n".join(lines)

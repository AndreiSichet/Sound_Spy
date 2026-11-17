"""Evaluation of extraction (``soundspy evaluate extraction``).

A fixed, seeded subset of a split's recipes is rendered at the processing rate; every recipe
gives ``mixture == target + rest`` exactly, so the correct output (the target alone) is known.
The same mixtures go through every system:

==============  ================================================================================
mixture         no processing: the reference every improvement is measured against
timeline        the baseline with the offline detector (windows centred on each time, no lag)
live            the baseline with the live detector's lag reproduced (causal windows, EMA,
                hysteresis)
oracle          the baseline with the **true** target activity: what a perfect detector would give
irm             the ideal ratio mask from the true target and rest, in the baseline's STFT: the
                upper bound of any single-channel masking method
neural_offline  the neural extractor run over the whole file in one call (step 07)
neural_stream   the neural extractor fed 20 ms chunks, exactly as in a live session; must match
                ``neural_offline``
neural_live_db  ``neural_stream`` at the live default attenuation (15 dB) instead of the
                evaluation's 30 dB, for reference
==============  ================================================================================

``--method mask_baseline`` evaluates the first five (step 06); ``--method neural_tse`` all of
them, and repeats the evaluation on a **reverberant copy** of the same recipes (every event in a
simulated room, ``data/reverb.py``, fixed seed), because real rooms reverberate and the dataset
clips mostly do not.

So ``oracle - live`` is what the detector's errors and lag cost the baseline, ``irm - oracle``
what a gate leaves on the table, and ``neural_stream - live`` what the neural method gains in the
situation that matters: live.

Metrics (``extraction/metrics.py``), all on delay-compensated outputs: SI-SDR, SI-SDRi and SNRi
for recipes with a target; **suppression** (output energy relative to the mixture) for recipes
whose target class is absent, where the correct output is silence. Two diagnostics explain the
means: the **median** SI-SDRi (a few disastrous recipes dominate a mean), and **missed**: the
share of recipes with a target whose whole output sits at the attenuation floor (for the
baseline: the gate never opened). SI-SDR cannot see that case (a uniformly quieter mixture
scores exactly 0 dB improvement), so it is counted separately.

Outputs: ``experiments/ext_<method>/`` with ``metrics.json`` (overall and broken down by target
class, target-to-rest level, number of foreground sources, how long the target sounds, one or
two target classes), ``metrics.md`` (the same as tables), ``per_recipe.csv`` (and
``per_recipe_reverb.csv``) and ``config.json``; and ``outputs/eval/ext_<method>/`` with a few
rendered examples (mixture, target, every system's output) and an index describing them.
"""

from __future__ import annotations

import csv
import dataclasses
import random
import time
from collections.abc import Callable, Iterable, Sequence
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

SYSTEMS = ("mixture", "timeline", "live", "oracle", "irm")  # the step-06 comparison
NEURAL_SYSTEMS = ("neural_offline", "neural_stream", "neural_live_db")
SYSTEM_NAMES = {
    "mixture": "mixture (no processing)",
    "timeline": "mask_baseline + timeline",
    "live": "mask_baseline + live",
    "oracle": "mask_baseline + oracle",
    "irm": "IRM oracle",
    "neural_offline": "neural_tse offline (whole file)",
    "neural_stream": "neural_tse streaming (as live)",
    "neural_live_db": "neural_tse streaming, live default dB",
}
SHORT_NAMES = {
    "timeline": "baseline offline", "live": "baseline live", "oracle": "baseline oracle",
    "irm": "IRM", "neural_offline": "neural offline", "neural_stream": "neural streaming",
    "neural_live_db": "neural, live dB",
}
EVALUATED_METHODS = ("mask_baseline", "neural_tse")
T2R_BINS = ("<= -5 dB", "-5 .. 5 dB", ">= 5 dB")
DURATION_BINS = ("< 1 s", "1 .. 3 s", ">= 3 s")
TARGET_COUNT_BINS = ("1 target class", "2+ target classes")
MISSED_MARGIN_DB = 1.0  # output within this of the floor everywhere = nothing was let through


class ExtractionEvalError(RuntimeError):
    pass


def experiment_name(method: str) -> str:
    return f"ext_{method}"


def systems_for(method: str) -> tuple[str, ...]:
    return SYSTEMS if method == "mask_baseline" else (*SYSTEMS, *NEURAL_SYSTEMS)


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


def _systems_of(records: list[dict[str, Any]]) -> tuple[str, ...]:
    if not records:
        return SYSTEMS
    return tuple(s for s in (*SYSTEMS, *NEURAL_SYSTEMS) if s in records[0])


def summarize(records: list[dict[str, Any]], systems: Sequence[str] | None = None
              ) -> dict[str, Any]:
    """Per-system means over ``records`` (targets and empty targets separately)."""
    systems = _systems_of(records) if systems is None else systems
    with_target = [r for r in records if not r["empty"]]
    empty = [r for r in records if r["empty"]]
    out: dict[str, Any] = {"n": len(with_target), "n_empty": len(empty)}
    for system in systems:
        out[system] = {
            "si_sdr": _mean(r[system]["si_sdr"] for r in with_target),
            "si_sdr_i": _mean(r[system]["si_sdr_i"] for r in with_target),
            "si_sdr_i_median": _median(r[system]["si_sdr_i"] for r in with_target),
            "snr_i": _mean(r[system]["snr_i"] for r in with_target),
            "missed": _mean(float(r[system]["missed"]) for r in with_target),
            "suppression_db": _mean(r[system]["suppression_db"] for r in empty),
        }
    return out


def breakdowns(records: list[dict[str, Any]], labels: list[str],
               systems: Sequence[str] | None = None) -> dict[str, Any]:
    systems = _systems_of(records) if systems is None else systems
    with_target = [r for r in records if not r["empty"]]

    def group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return summarize(rows, systems)

    by_class = {lab: group([r for r in with_target if lab in r["target"]]) for lab in labels}
    by_t2r = {b: group([r for r in with_target if r["t2r_bin"] == b]) for b in T2R_BINS}
    counts = sorted({r["foreground"] for r in with_target})
    by_fg = {str(c): group([r for r in with_target if r["foreground"] == c]) for c in counts}
    empty_by_class = {lab: group([r for r in records if r["empty"] and lab in r["target"]])
                      for lab in labels}
    by_duration = {b: group([r for r in with_target if r["duration_bin"] == b])
                   for b in DURATION_BINS}
    by_count = {
        TARGET_COUNT_BINS[0]: group([r for r in with_target if len(r["target"]) == 1]),
        TARGET_COUNT_BINS[1]: group([r for r in with_target if len(r["target"]) > 1]),
    }
    return {"by_target_class": by_class, "by_target_to_rest_db": by_t2r,
            "by_foreground_sources": by_fg, "by_target_duration": by_duration,
            "by_target_count": by_count, "empty_by_target_class": empty_by_class}


# --------------------------------------------------------------------------------------------
# One recipe


def process_recipe(config: AppConfig, rendered, core, *, attenuation_db: float
                   ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """The five step-06 systems' outputs for one rendered recipe (1-D arrays), and timings."""
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


def remix(output: np.ndarray, mixture: np.ndarray, from_db: float, to_db: float) -> np.ndarray:
    """The output of residual mixing at ``from_db``, recomputed for ``to_db`` (exact).

    ``y = est + g (x - est)`` is linear in the estimate, so the estimate can be recovered from
    ``y`` and ``x`` and mixed again with another ``g``, without running the model twice.
    """
    g_from, g_to = 10.0 ** (-from_db / 20.0), 10.0 ** (-to_db / 20.0)
    estimate = (output - g_from * mixture) / (1.0 - g_from)
    return (estimate + g_to * (mixture - estimate)).astype(np.float32)


def process_recipe_neural(config: AppConfig, rendered, neural_core, *, attenuation_db: float
                          ) -> tuple[dict[str, np.ndarray], float]:
    """The neural systems' outputs for one rendered recipe, and the time they took."""
    from soundspy.extraction.neural_tse import NeuralTSEExtractor, extract_whole

    started = time.perf_counter()
    cfg = config.extraction.neural_tse
    sr = rendered.mixture.sample_rate
    mixture = rendered.mixture.samples.ravel()
    labels = frozenset(rendered.recipe.target)
    target = TargetSpec(labels=labels, attenuation_db=attenuation_db)
    chunk = config.engine.chunk_frames(sr)  # the live session's 20 ms chunks
    stream = run_offline(NeuralTSEExtractor(neural_core, cfg.live), rendered.mixture, target,
                         chunk_frames=chunk).samples.ravel()
    outputs = {
        "neural_offline": extract_whole(neural_core, mixture, sr, labels, attenuation_db),
        "neural_stream": stream,
        "neural_live_db": remix(stream, mixture, attenuation_db,
                                config.extraction.default_attenuation_db),
    }
    return outputs, time.perf_counter() - started


def score(outputs: dict[str, np.ndarray], mixture: np.ndarray, reference: np.ndarray,
          attenuation_db: float, attenuations: dict[str, float] | None = None
          ) -> dict[str, dict[str, Any]]:
    """Metrics of every system (``attenuations``: systems run at another attenuation)."""
    empty = M.is_silent(reference)
    scores: dict[str, dict[str, Any]] = {}
    for system, out in outputs.items():
        floor = (attenuations or {}).get(system, attenuation_db)
        level = M.suppression_db(out, mixture)
        if empty:
            scores[system] = {"si_sdr": None, "si_sdr_i": None, "snr_i": None, "missed": None,
                              "suppression_db": round(level, 3)}
        else:
            scores[system] = {
                "si_sdr": round(M.si_sdr(out, reference), 3),
                "si_sdr_i": round(M.si_sdr_improvement(out, mixture, reference), 3),
                "snr_i": round(M.snr_improvement(out, mixture, reference), 3),
                # the whole output at the floor: the target was there but nothing came through
                "missed": bool(floor > MISSED_MARGIN_DB
                               and level <= -(floor - MISSED_MARGIN_DB)),
                "suppression_db": None,
            }
    return scores


# --------------------------------------------------------------------------------------------
# The whole evaluation


def _run_pass(config: AppConfig, chosen: list[Recipe], library: ClipLibrary, core, neural_core,
              *, sample_rate: int, reverberant: bool, want: set[str], out_dir: Path,
              progress: Callable[[str], None]
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """All systems over ``chosen`` (dry, or in simulated rooms). Returns the per-recipe
    records, the listening examples written and the seconds spent per part."""
    ev, det = config.extraction.evaluation, config.detection
    reverb = dataclasses.replace(config.extraction.neural_tse.data.reverb, prob=1.0)
    records: list[dict[str, Any]] = []
    listening: list[dict[str, Any]] = []
    seconds = {"render": 0.0, "detector": 0.0, "extractor": 0.0, "neural": 0.0}
    live_db = config.extraction.default_attenuation_db
    started = time.perf_counter()
    for i, recipe in enumerate(chosen):
        t0 = time.perf_counter()
        rirs = None
        if reverberant:
            from soundspy.data.reverb import recipe_rirs

            rirs = recipe_rirs(recipe.id, len(recipe.events), sample_rate, reverb,
                               ev.reverb_seed)
        rendered = render_with_config(recipe, sample_rate, library, config,
                                      activity_rate_hz=det.activity_rate_hz, rirs=rirs)
        seconds["render"] += time.perf_counter() - t0
        outputs, times = process_recipe(config, rendered, core,
                                        attenuation_db=ev.attenuation_db)
        for key, value in times.items():
            seconds[key] += value
        if neural_core is not None:
            neural, spent = process_recipe_neural(config, rendered, neural_core,
                                                  attenuation_db=ev.attenuation_db)
            outputs.update(neural)
            seconds["neural"] += spent
        mixture = rendered.mixture.samples.ravel()
        reference = rendered.target.samples.ravel()
        scores = score(outputs, mixture, reference, ev.attenuation_db,
                       {"neural_live_db": live_db})
        active_s = target_active_seconds(rendered)
        record = {
            "id": recipe.id, "empty": M.is_silent(reference), "target": list(recipe.target),
            "present": list(recipe.present), "target_to_rest_db": recipe.target_to_rest_db,
            "t2r_bin": t2r_bin(recipe.target_to_rest_db), "foreground": foreground_sources(recipe),
            "target_active_s": round(active_s, 2), "duration_bin": duration_bin(active_s),
            "reverberant": reverberant, **scores,
        }
        records.append(record)
        if recipe.id in want:
            listening.append(_write_example(out_dir, record, outputs, reference, sample_rate))
        if (i + 1) % 50 == 0 or i + 1 == len(chosen):
            progress(f"  {i + 1}/{len(chosen)} recipes ({time.perf_counter() - started:.0f} s)")
    seconds["total"] = time.perf_counter() - started
    return records, listening, seconds


def evaluate_extraction(config: AppConfig, method: str = "mask_baseline", split: str = "test", *,
                        n: int | None = None, sample_rate: int | None = None,
                        device: str = "auto", examples: int | None = None,
                        reverberant: bool | None = None,
                        progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the comparison (see the module docstring) and write its files. ``reverberant``:
    also evaluate the reverberant copy (default: yes for ``neural_tse``)."""
    if method not in EVALUATED_METHODS:
        raise ExtractionEvalError(f"unknown method {method!r}; methods: {list(EVALUATED_METHODS)}")
    ev, det = config.extraction.evaluation, config.detection
    n = ev.n if n is None else n
    sample_rate = ev.sample_rate if sample_rate is None else sample_rate
    examples = ev.examples if examples is None else examples
    reverberant = (method == "neural_tse") if reverberant is None else reverberant
    if det is None or det.checkpoint is None:
        raise ExtractionEvalError("no detection model is configured (detection.checkpoint): "
                                  "the timeline and live systems need it")
    from soundspy.detection.inference import DetectorUnavailable, load_core

    try:
        core = load_core(Path(det.checkpoint), device if device != "auto" else det.live.device)
    except DetectorUnavailable as exc:
        raise ExtractionEvalError(str(exc)) from exc
    neural_core = None
    if method == "neural_tse":
        from soundspy.extraction.neural_tse import NeuralUnavailable, load_neural_core

        cfg = config.extraction.neural_tse
        if cfg.checkpoint is None:
            raise ExtractionEvalError("no neural extraction checkpoint is configured "
                                      "(extraction.neural_tse.checkpoint)")
        try:
            neural_core = load_neural_core(
                Path(cfg.checkpoint), device if device != "auto" else cfg.live.device)
        except NeuralUnavailable as exc:
            raise ExtractionEvalError(str(exc)) from exc

    path = recipes_path(config.paths["mixtures"], split)
    recipes = list(iter_recipes(path)) if path.is_file() else []
    if not recipes:
        raise ExtractionEvalError(f"no {split} recipes: run `soundspy data mix` first")
    chosen = select_recipes(recipes, n, ev.seed)
    systems = systems_for(method)
    progress(f"{len(chosen)} of {len(recipes)} {split} recipes at {sample_rate} Hz; "
             f"detector {core.name} on {core.device}"
             + (f"; extractor {neural_core.name} on {neural_core.device}" if neural_core else ""))

    library = ClipLibrary.from_config(config)
    out_dir = config.paths["outputs"] / "eval" / experiment_name(method)
    want = _example_ids(chosen, examples, ev.example_ids)
    if want:
        _clear_examples(out_dir)  # the folder then holds exactly what its index describes
    labels = list(config.labels)
    records, listening, seconds = _run_pass(
        config, chosen, library, core, neural_core, sample_rate=sample_rate, reverberant=False,
        want=want, out_dir=out_dir, progress=progress)
    result: dict[str, Any] = {
        "method": method,
        "split": split,
        "recipes": len(records),
        "sample_rate": sample_rate,
        "attenuation_db": ev.attenuation_db,
        "live_attenuation_db": config.extraction.default_attenuation_db,
        "seed": ev.seed,
        "systems": list(systems),
        "detector": {"model": core.name, "device": str(core.device)},
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "audio_seconds": round(sum(r.duration_s for r in chosen), 1),
        "overall": summarize(records, systems),
        **breakdowns(records, labels, systems),
    }
    if neural_core is not None:
        result["extractor"] = {"model": neural_core.name, "device": str(neural_core.device),
                               "checkpoint": str(neural_core.checkpoint)}
    reverb_records: list[dict[str, Any]] = []
    if reverberant:
        progress("the same recipes in simulated rooms (reverberant copy) …")
        reverb_records, _, reverb_seconds = _run_pass(
            config, chosen, library, core, neural_core, sample_rate=sample_rate,
            reverberant=True, want=set(), out_dir=out_dir, progress=progress)
        result["reverberant"] = {
            "seed": ev.reverb_seed,
            "reverb": dataclasses.asdict(config.extraction.neural_tse.data.reverb),
            "seconds": {k: round(v, 1) for k, v in reverb_seconds.items()},
            "overall": summarize(reverb_records, systems),
            "by_target_class": breakdowns(reverb_records, labels, systems)["by_target_class"],
        }

    exp = config.paths["experiments"] / experiment_name(method)
    exp.mkdir(parents=True, exist_ok=True)
    metrics_path = exp / "metrics.json"
    previous = read_json(metrics_path) if metrics_path.is_file() else {}
    write_json(metrics_path, {**previous, split: result})  # one entry per split
    (exp / "metrics.md").write_text(format_markdown(result, labels), encoding="utf-8")
    write_per_recipe(exp / "per_recipe.csv", records, systems)
    if reverb_records:
        write_per_recipe(exp / "per_recipe_reverb.csv", reverb_records, systems)
    write_json(exp / "config.json", {
        "method": method,
        "systems": list(systems),
        "extraction": dataclasses.asdict(config.extraction.mask_baseline),
        "neural_tse": (None if neural_core is None else
                       {"checkpoint": str(neural_core.checkpoint), "model": neural_core.name,
                        "live": dataclasses.asdict(config.extraction.neural_tse.live)}),
        "evaluation": dataclasses.asdict(ev) | {"n": n, "sample_rate": sample_rate},
        "detection": {"checkpoint": str(det.checkpoint), "hop_s": det.live.hop_s,
                      "ema_alpha": det.live.ema_alpha, "hysteresis": det.live.hysteresis},
        "labels": labels,
        "app_version": __version__,
        "git": git_commit(config.project_root),
    })
    if listening:
        write_json(out_dir / "index.json", {
            "sample_rate": sample_rate,
            "systems": {s: SYSTEM_NAMES[s] for s in systems}, "examples": listening})
        (out_dir / "index.md").write_text(_examples_markdown(listening, systems),
                                          encoding="utf-8")
    result["experiment_dir"] = str(exp)
    result["examples_dir"] = str(out_dir) if listening else None
    return result


def _example_ids(recipes: list[Recipe], examples: int, forced: Sequence[str] = ()) -> set[str]:
    """``examples`` recipes of the subset: the ``forced`` ones that are in it, a couple with an
    absent target, and the first ones with a target."""
    if examples <= 0:
        return set()
    ids = [r.id for r in recipes]
    chosen = [i for i in forced if i in ids][:examples]
    empties = [r.id for r in recipes if r.empty_target and r.id not in chosen]
    chosen += empties[: max(0, min(2, examples // 3, examples - len(chosen)))]
    others = [r.id for r in recipes if not r.empty_target and r.id not in chosen]
    chosen += others[: max(0, examples - len(chosen))]
    return set(chosen)


def _clear_examples(out_dir: Path) -> None:
    """Remove the example files a previous evaluation wrote here (another subset, other ids)."""
    if not out_dir.is_dir():
        return
    suffixes = tuple(f"_{name}.wav" for name in ("target", *SYSTEMS, *NEURAL_SYSTEMS))
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
            "files": files, "scores": {s: record[s] for s in outputs}}


def _examples_markdown(examples: list[dict[str, Any]], systems: Sequence[str] = SYSTEMS) -> str:
    shown = [s for s in ("timeline", "live", "oracle", "neural_stream", "irm") if s in systems]
    names = " / ".join(SHORT_NAMES[s] for s in shown)
    lines = ["# Listening examples", "",
             "Each recipe: `_mixture` (input), `_target` (the correct output) and one file per "
             "system (`_timeline`, `_live`, `_oracle`: the baseline with the offline / live / "
             "true activity; `_irm`: the ideal mask; `_neural_stream`: the neural extractor as "
             "it runs live; `_neural_live_db`: the same at the live default attenuation).", "",
             f"| Recipe | Target | Also present | Target vs rest | SI-SDRi {names} (dB) |",
             "|---|---|---|---|---|"]
    for ex in examples:
        other = [p for p in ex["present"] if p not in ex["target"]] or ["(background only)"]
        if ex["empty"]:
            scores = "target absent; suppression " + " / ".join(
                f"{ex['scores'][s]['suppression_db']:.1f}" for s in shown)
        else:
            scores = " / ".join(f"{ex['scores'][s]['si_sdr_i']:+.1f}" for s in shown)
        lines.append(f"| {ex['id']} | {', '.join(ex['target'])} | {', '.join(other)} | "
                     f"{ex['target_to_rest_db']:+.1f} dB | {scores} |")
    return "\n".join(lines) + "\n"


def write_per_recipe(path: Path, records: list[dict[str, Any]],
                     systems: Sequence[str] | None = None) -> None:
    systems = _systems_of(records) if systems is None else systems
    fields = ["id", "empty", "target", "present", "target_to_rest_db", "foreground",
              "target_active_s"]
    metrics = ("si_sdr", "si_sdr_i", "snr_i", "missed", "suppression_db")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields + [f"{s}_{m}" for s in systems for m in metrics])
        for r in records:
            row = [r["id"], int(r["empty"]), "+".join(r["target"]), "+".join(r["present"]),
                   r["target_to_rest_db"], r["foreground"], r["target_active_s"]]
            for s in systems:
                for m in metrics:
                    value = r[s][m]
                    row.append("" if value is None else int(value) if m == "missed" else value)
            writer.writerow(row)


# --------------------------------------------------------------------------------------------
# Tables


def _f(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _overall_table(summary: dict[str, Any], systems: Sequence[str] = SYSTEMS,
                   live_db: float | None = None) -> list[str]:
    lines = ["| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | "
             "Target missed | Suppression, target absent (dB) |", "|---|---|---|---|---|---|---|"]
    for system in systems:
        s = summary[system]
        missed = "-" if s["missed"] is None else f"{100 * s['missed']:.0f} %"
        name = SYSTEM_NAMES[system]
        if system == "neural_live_db" and live_db is not None:
            name = f"neural_tse streaming, {live_db:g} dB"
        lines.append(f"| {name} | {_f(s['si_sdr'])} | {_f(s['si_sdr_i'])} | "
                     f"{_f(s['si_sdr_i_median'])} | {_f(s['snr_i'])} | {missed} | "
                     f"{_f(s['suppression_db'])} |")
    return lines


def _columns(systems: Sequence[str]) -> list[str]:
    """Systems shown in the breakdown tables (all but the mixture; without the two neural
    variants that repeat ``neural_stream`` when it is there)."""
    hidden = {"mixture"} | ({"neural_offline", "neural_live_db"} if "neural_stream" in systems
                            else set())
    return [s for s in systems if s not in hidden]


def _breakdown_table(title: str, groups: dict[str, Any], key: str = "si_sdr_i",
                     count: str = "n", systems: Sequence[str] = SYSTEMS) -> list[str]:
    shown = _columns(systems)
    lines = [f"| {title} | recipes | " + " | ".join(SHORT_NAMES[s] for s in shown) + " |",
             "|---|---|" + "---|" * len(shown)]
    for name, summary in groups.items():
        if not summary[count]:
            continue
        cells = " | ".join(_f(summary[s][key]) for s in shown)
        lines.append(f"| {name} | {summary[count]} | {cells} |")
    return lines


def format_markdown(result: dict[str, Any], labels: list[str]) -> str:
    overall = result["overall"]
    sec = result["seconds"]
    systems = result.get("systems", list(SYSTEMS))
    live_db = result.get("live_attenuation_db")
    extractor = result.get("extractor")
    neural = (f", extractor {extractor['model']} on {extractor['device']}" if extractor else "")
    spent = (f"render {sec['render']:.0f} s, detector {sec['detector']:.0f} s, baseline x3 "
             f"{sec['extractor']:.0f} s"
             + (f", neural x2 {sec['neural']:.0f} s" if sec.get("neural") else ""))

    def table(title: str, groups: dict[str, Any], **kw: Any) -> list[str]:
        return _breakdown_table(title, groups, systems=systems, **kw)

    lines = [
        f"# Extraction evaluation: {result['method']} ({result['split']})", "",
        f"{result['recipes']} recipes ({overall['n']} with a target, {overall['n_empty']} with "
        f"the target class absent), rendered at {result['sample_rate']} Hz, attenuation "
        f"{result['attenuation_db']:g} dB, detector {result['detector']['model']} on "
        f"{result['detector']['device']}{neural}. {result['audio_seconds']:.0f} s of audio in "
        f"{sec['total']:.0f} s ({spent}).", "",
        "## Overall", "", *_overall_table(overall, systems, live_db), "",
    ]
    reverb = result.get("reverberant")
    if reverb:
        lines += [
            "## Overall, the same recipes in simulated rooms (reverberant copy)", "",
            f"Every event convolved with its own room impulse response (RT60 "
            f"{reverb['reverb']['rt60_s'][0]:g}-{reverb['reverb']['rt60_s'][1]:g} s, seed "
            f"{reverb['seed']}); the target is the reverberant target.", "",
            *_overall_table(reverb["overall"], systems, live_db), "",
            "## SI-SDRi (dB) by target class, reverberant copy", "",
            *table("Target class", reverb["by_target_class"]), "",
        ]
    lines += [
        "## SI-SDRi (dB) by target class", "",
        *table("Target class", result["by_target_class"]), "",
        "## SI-SDRi (dB) by target level relative to the rest", "",
        *table("Target vs rest", result["by_target_to_rest_db"]), "",
        "## SI-SDRi (dB) by number of foreground sources", "",
        *table("Foreground sources", result["by_foreground_sources"]), "",
        "## SI-SDRi (dB) by how long the target sounds", "",
        *table("Target active", result["by_target_duration"]), "",
    ]
    if "by_target_count" in result:
        lines += ["## SI-SDRi (dB) for one or several target classes", "",
                  *table("Selection", result["by_target_count"]), ""]
    lines += [
        "## SNRi (dB) by target class", "",
        *table("Target class", result["by_target_class"], key="snr_i"), "",
        "## Suppression (dB) when the target class is absent, by selected class", "",
        *table("Selected class", result["empty_by_target_class"], key="suppression_db",
               count="n_empty"), "",
    ]
    return "\n".join(lines)

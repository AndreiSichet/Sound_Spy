"""Evaluation of the hybrid extractor (step 07c): its systems, and the gate tuning on val.

``soundspy evaluate extraction --method hybrid`` scores, on the same recipes as every earlier
evaluation, the step-06 baseline systems plus:

==================  ==============================================================================
wf                  Waveformer v1 alone (whole file, no gate)
wf_hgate            Waveformer v1 + the hybrid gate
dfn                 DeepFilterNet3 alone (Speech-only selections; the other recipes have no row)
dfn_hgate           DeepFilterNet3 + the hybrid gate
cascade             Waveformer v1 -> DeepFilterNet3 (Speech only)
cascade_hgate       the cascade + the hybrid gate
hybrid_offline      the hybrid as configured: route by selection + gate, detector timeline
                    centred (offline mode), whole file
hybrid_stream       the same fed 20 ms chunks through ``HybridExtractor`` (as live): must match
                    ``hybrid_offline``
hybrid_live         the hybrid with the live detector's lag (causal timeline), whole file
hybrid_live_db      ``hybrid_offline`` at the live default attenuation (15 dB)
hybrid_high_db      ``hybrid_offline`` at ``evaluation.high_attenuation_db`` (40 dB)
==================  ==============================================================================

The whole-file systems use the same gate code as the streaming extractor
(:func:`gate_output`: the gate runs over the estimate and the input as the extractor sees them,
``latency`` samples late), so ``hybrid_offline == hybrid_stream`` up to float rounding.

**Gate tuning** (``soundspy evaluate hybrid-gate``): the routes run once over validation recipes
(a fixed subset plus validation copies of the pair benchmarks, made from validation clips),
their estimates are kept, and every combination of ``tuning`` in ``hybrid.yaml`` is applied with
the **live** (causal) detector timeline. The chosen setting is the strongest target-absent
suppression whose SI-SDRi cost against no gate is at most ``tuning.max_cost_db``. Test recipes
are never used.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.config import AppConfig, HybridGateConfig
from soundspy.core.target import TargetSpec
from soundspy.extraction import metrics as M
from soundspy.extraction.activity import ActivitySource
from soundspy.extraction.hybrid_gate import HybridGate

HYBRID_SYSTEMS = ("wf", "wf_hgate", "dfn", "dfn_hgate", "cascade", "cascade_hgate",
                  "hybrid_offline", "hybrid_stream", "hybrid_live", "hybrid_live_db",
                  "hybrid_high_db")
SPEECH_ONLY_SYSTEMS = ("dfn", "dfn_hgate", "cascade", "cascade_hgate")
SYSTEM_NAMES = {
    "wf": "Waveformer v1 (whole file, no gate)",
    "wf_hgate": "Waveformer v1 + hybrid gate",
    "dfn": "DeepFilterNet3 alone (Speech only)",
    "dfn_hgate": "DeepFilterNet3 + hybrid gate (Speech only)",
    "cascade": "Waveformer v1 -> DeepFilterNet3 (Speech only)",
    "cascade_hgate": "cascade + hybrid gate (Speech only)",
    "hybrid_offline": "hybrid offline (whole file, centred detector)",
    "hybrid_stream": "hybrid streaming (20 ms chunks, centred detector)",
    "hybrid_live": "hybrid with the live detector's lag",
    "hybrid_live_db": "hybrid offline, live default dB",
    "hybrid_high_db": "hybrid offline, high dB",
}
SHORT_NAMES = {
    "wf": "v1", "wf_hgate": "v1 + gate", "dfn": "DFN3", "dfn_hgate": "DFN3 + gate",
    "cascade": "cascade", "cascade_hgate": "cascade + gate", "hybrid_offline": "hybrid",
    "hybrid_stream": "hybrid stream", "hybrid_live": "hybrid live",
    "hybrid_live_db": "hybrid 15 dB", "hybrid_high_db": "hybrid 40 dB",
}


@dataclass
class HybridCores:
    """The loaded networks: the Waveformer (``NeuralCore``) and DeepFilterNet3 (``SpeechCore``)."""

    waveformer: Any
    speech: Any
    live: Any  # NeuralLiveConfig

    def latencies(self, sample_rate: int) -> dict[str, int]:
        from soundspy.extraction.neural_tse import WaveformerStream
        from soundspy.extraction.speech_enhancer import latency_samples

        wf = WaveformerStream(self.waveformer, self.live).configure(sample_rate)
        dfn = latency_samples(self.speech.p, self.speech.lag_frames)
        return {"waveformer": wf, "speech_enhancer": dfn, "cascade": wf + dfn,
                "enhancer_route": max(wf, dfn), "cascade_route": max(wf, wf + dfn)}


def load_cores(config: AppConfig, device: str = "auto") -> HybridCores:
    from soundspy.config import speech_model_dir
    from soundspy.extraction.neural_tse import load_neural_core
    from soundspy.extraction.speech_enhancer import load_speech_core

    neural = config.extraction.neural_tse
    if neural.checkpoint is None:
        raise RuntimeError("no Waveformer checkpoint is configured")
    sp = config.extraction.hybrid.speech
    wf = load_neural_core(Path(neural.checkpoint),
                          device if device != "auto" else neural.live.device)
    speech = load_speech_core(speech_model_dir(config), sp.device, post_filter=sp.post_filter,
                              backend=sp.backend, cpu_threads=sp.cpu_threads)
    return HybridCores(wf, speech, neural.live)


def gate_output(est: np.ndarray, mixture: np.ndarray, labels: Collection[str],
                attenuation_db: float, gate: HybridGateConfig, source: ActivitySource | None,
                sample_rate: int, latency: int, *, warmup: bool = False) -> np.ndarray:
    """The hybrid's output for a whole signal: :class:`HybridGate` over the estimate and the
    input **as the streaming extractor sees them** (``latency`` zeros first), then residual
    mixing ``y = G (est + g (x - est))``."""
    x = np.asarray(mixture, dtype=np.float32).reshape(-1)
    est = np.asarray(est, dtype=np.float32).reshape(-1)
    pad = np.zeros(latency, dtype=np.float32)
    hg = HybridGate(gate, sample_rate, latency, source)
    hg.restart(warmup)  # as HybridExtractor.reset(): the speech routes start with a warm-up
    g_gate = hg.gains(np.concatenate((pad, est)), np.concatenate((pad, x)), frozenset(labels),
                      attenuation_db)[latency:]
    g = 10.0 ** (-attenuation_db / 20.0)
    return (g_gate * (est + g * (x - est))).astype(np.float32)


def route_estimates(cores: HybridCores, mixture: np.ndarray, sample_rate: int,
                    labels: Collection[str], speech_labels: Collection[str]
                    ) -> dict[str, np.ndarray]:
    """Whole-file estimates: ``wf`` always; ``dfn`` and ``cascade`` for a Speech-only
    selection (aligned with the input)."""
    from soundspy.extraction.neural_tse import whole_estimates
    from soundspy.extraction.speech_enhancer import enhance_whole

    labels = frozenset(labels)
    out = {"wf": whole_estimates(cores.waveformer, mixture, sample_rate, labels)[0]}
    if labels == frozenset(speech_labels):
        out["dfn"] = enhance_whole(cores.speech, mixture)
        out["cascade"] = enhance_whole(cores.speech, out["wf"])
    return out


def chosen_route(config: AppConfig, labels: Collection[str]) -> str:
    """Which whole-file estimate the configured hybrid uses for ``labels``."""
    sp = config.extraction.hybrid.speech
    if frozenset(labels) == frozenset(sp.labels):
        return "cascade" if sp.route == "cascade" else "dfn"
    return "wf"


def process_recipe_hybrid(config: AppConfig, rendered, sources: dict[str, ActivitySource],
                          cores: HybridCores, *, attenuation_db: float, stream: bool = True
                          ) -> dict[str, np.ndarray]:
    """The hybrid systems' outputs for one rendered recipe (see the module docstring)."""
    hy, ev = config.extraction.hybrid, config.extraction.evaluation
    sr = rendered.mixture.sample_rate
    x = rendered.mixture.samples.ravel()
    labels = frozenset(rendered.recipe.target)
    gate = dataclasses.replace(hy.gate, enabled=True)
    lat = cores.latencies(sr)
    est = route_estimates(cores, x, sr, labels, hy.speech.labels)
    route_latency = lat["cascade_route"] if hy.speech.route == "cascade" else lat["enhancer_route"]
    timeline, live = sources["timeline"], sources["live"]

    def gated(e, latency, att=attenuation_db, source=timeline, warmup=False):
        return gate_output(e, x, labels, att, gate, source, sr, latency, warmup=warmup)

    from soundspy.extraction.neural_tse import residual_mix

    out: dict[str, np.ndarray] = {"wf": residual_mix(est["wf"], x, attenuation_db),
                                  "wf_hgate": gated(est["wf"], route_latency)}
    if "dfn" in est:
        out["dfn"] = residual_mix(est["dfn"], x, attenuation_db)
        out["dfn_hgate"] = gated(est["dfn"], lat["enhancer_route"], warmup=True)
        out["cascade"] = residual_mix(est["cascade"], x, attenuation_db)
        out["cascade_hgate"] = gated(est["cascade"], lat["cascade_route"], warmup=True)
    route = chosen_route(config, labels)
    chosen, warm = est[route], route != "wf"
    out["hybrid_offline"] = gated(chosen, route_latency, warmup=warm)
    out["hybrid_live"] = gated(chosen, route_latency, source=live, warmup=warm)
    out["hybrid_live_db"] = gated(chosen, route_latency,
                                  att=config.extraction.default_attenuation_db, warmup=warm)
    out["hybrid_high_db"] = gated(chosen, route_latency, att=ev.high_attenuation_db,
                                  warmup=warm)
    if stream:
        out["hybrid_stream"] = stream_output(config, rendered, timeline, cores, attenuation_db)
    return out


def stream_output(config: AppConfig, rendered, source: ActivitySource, cores: HybridCores,
                  attenuation_db: float) -> np.ndarray:
    """The recipe through ``HybridExtractor`` in 20 ms chunks (as a live session feeds it)."""
    from soundspy.extraction.hybrid import HybridExtractor
    from soundspy.extraction.neural_tse import WaveformerStream
    from soundspy.extraction.offline import run_offline
    from soundspy.extraction.speech_enhancer import SpeechEnhancer

    hy = config.extraction.hybrid
    cascade = (WaveformerStream(cores.waveformer, cores.live) if hy.speech.route == "cascade"
               else None)
    ext = HybridExtractor(dataclasses.replace(hy, performance_cores=False),
                          waveformer=WaveformerStream(cores.waveformer, cores.live),
                          speech=SpeechEnhancer(cores.speech), cascade_waveformer=cascade,
                          activity=source, gate=True)
    target = TargetSpec(labels=frozenset(rendered.recipe.target), attenuation_db=attenuation_db)
    chunk = config.engine.chunk_frames(rendered.mixture.sample_rate)
    return run_offline(ext, rendered.mixture, target, chunk_frames=chunk).samples.ravel()


# --------------------------------------------------------------------------------------------
# Gate tuning on validation recipes


def grid_configs(base: HybridGateConfig, grid: dict[str, tuple[float, ...]]
                 ) -> list[HybridGateConfig]:
    keys = sorted(grid)
    return [dataclasses.replace(base, enabled=True, **dict(zip(keys, values, strict=True)))
            for values in itertools.product(*(grid[k] for k in keys))]


def _val_recipes(config: AppConfig, n: int, bench_n: int, library, progress) -> list[tuple]:
    from soundspy.data.mixtures import iter_recipes, recipes_path
    from soundspy.training.evaluate_extraction import benchmark_sets, select_recipes

    ev = config.extraction.evaluation
    path = recipes_path(config.paths["mixtures"], "val")
    recipes = list(iter_recipes(path)) if path.is_file() else []
    if not recipes:
        raise RuntimeError("no val recipes: run `soundspy data mix` first")
    chosen = [("val", r) for r in select_recipes(recipes, n, ev.seed)]
    if bench_n:
        sets = benchmark_sets(config, library, progress, split="val",
                              counts={"pair_absent": bench_n, "pair_a": bench_n,
                                      "pair_b": bench_n})
        for name in ("pair_absent", "pair_a", "pair_b"):
            chosen += [(name, r) for r in sets.get(name, [])]
    return chosen


def prepare_examples(config: AppConfig, cores: HybridCores, detector, *, n: int = 200,
                     bench_n: int = 40, progress: Callable[[str], None] = print
                     ) -> list[dict[str, Any]]:
    """Validation recipes rendered once, with both detector timelines and the configured
    route's estimate (the expensive part of the tuning)."""
    from soundspy.data.mixtures import ClipLibrary, render_with_config
    from soundspy.extraction.activity import causal_timeline, centred_timeline
    from soundspy.extraction.activity import timeline_probabilities as probs_of

    hy, ev, det = config.extraction.hybrid, config.extraction.evaluation, config.detection
    sr = ev.sample_rate
    library = ClipLibrary.from_config(config)
    chosen = _val_recipes(config, n, bench_n, library, progress)
    lat = cores.latencies(sr)
    started = time.perf_counter()
    examples = []
    settings = dict(hop_s=det.live.hop_s, ema_alpha=det.live.ema_alpha,
                    hysteresis=det.live.hysteresis)
    for count, (group, recipe) in enumerate(chosen, 1):
        rendered = render_with_config(recipe, sr, library, config,
                                      activity_rate_hz=det.activity_rate_hz)
        x = rendered.mixture.samples.ravel()
        labels = frozenset(recipe.target)
        raw_c, raw_k = probs_of(detector, rendered.mixture, det.live.hop_s)
        sources = {
            "centred": centred_timeline(detector, rendered.mixture, raw=raw_c, **settings),
            "live": causal_timeline(detector, rendered.mixture, raw=raw_k,
                                    delay_s=ev.live_delay_s,
                                    lag_s=config.extraction.mask_baseline.noise_guard_s,
                                    **settings)}
        route = chosen_route(config, labels)
        est = route_estimates(cores, x, sr, labels, hy.speech.labels)[route]
        latency = lat["cascade_route"] if hy.speech.route == "cascade" else lat["enhancer_route"]
        examples.append({"group": group, "x": x, "ref": rendered.target.samples.ravel(),
                         "est": est, "labels": labels, "sources": sources,
                         "latency": latency, "warmup": route != "wf", "route": route,
                         "speech": labels == frozenset(hy.speech.labels)})
        if count % 50 == 0:
            progress(f"  routes over {count}/{len(chosen)} validation recipes "
                     f"({time.perf_counter() - started:.0f} s)")
    return examples


def score_setting(examples: list[dict[str, Any]], gate: HybridGateConfig | None, clock: str,
                  attenuation_db: float, sample_rate: int) -> dict[str, Any]:
    """Means over the examples for one gate setting (``None``: no gate) and one detector
    timeline (``live`` / ``centred``); SI-SDRi also per route (Speech enhancer / Waveformer)."""
    from soundspy.extraction.neural_tse import residual_mix

    si, worse, absent, pair_absent, interrupt = [], 0, [], [], []
    per_route: dict[str, list[float]] = {}
    for ex in examples:
        if gate is None:
            y = residual_mix(ex["est"], ex["x"], attenuation_db)
        else:
            y = gate_output(ex["est"], ex["x"], ex["labels"], attenuation_db, gate,
                            ex["sources"][clock], sample_rate, ex["latency"],
                            warmup=ex["warmup"])
        if M.is_silent(ex["ref"]):
            level = M.suppression_db(y, ex["x"])
            (pair_absent if ex["group"] == "pair_absent" else absent).append(level)
        else:
            value = M.si_sdr_improvement(y, ex["x"], ex["ref"])
            si.append(value)
            per_route.setdefault(ex["route"], []).append(value)
            worse += value < -5.0
            if ex["speech"]:
                rate = M.interruption_rate(y, ex["ref"], sample_rate)
                if rate is not None:
                    interrupt.append(rate)
    return {"si_sdr_i": _m(si), "worse": int(worse), "suppression_db": _m(absent),
            "pair_absent_db": _m(pair_absent), "interruption": _m(interrupt),
            "si_sdr_i_by_route": {k: _m(v) for k, v in sorted(per_route.items())},
            "with_target": len(si), "absent": len(absent), "pair_absent": len(pair_absent)}


def tune_hybrid_gate(config: AppConfig, cores: HybridCores, detector, *, n: int = 200,
                     bench_n: int = 40, progress: Callable[[str], None] = print
                     ) -> dict[str, Any]:
    """The tuning table (see the module docstring)."""
    hy, ev = config.extraction.hybrid, config.extraction.evaluation
    started = time.perf_counter()
    examples = prepare_examples(config, cores, detector, n=n, bench_n=bench_n,
                                progress=progress)
    att = ev.attenuation_db
    rows = []
    for gate in [None, *grid_configs(hy.gate, hy.tuning)]:
        row: dict[str, Any] = {"gate": None if gate is None else
                               {k: getattr(gate, k) for k in sorted(hy.tuning)}}
        for clock in ("live", "centred"):
            row[clock] = score_setting(examples, gate, clock, att, ev.sample_rate)
        rows.append(row)
    return {"examples": len(examples), "groups": {g: sum(1 for e in examples if e["group"] == g)
                                                  for g in sorted({e["group"] for e in examples})},
            "attenuation_db": att, "rows": rows,
            "seconds": round(time.perf_counter() - started, 1)}


def _m(values) -> float | None:
    return round(float(np.mean(values)), 3) if values else None


def choose(result: dict[str, Any], max_cost_db: float, clock: str = "live"
           ) -> dict[str, Any] | None:
    """The setting with the strongest target-absent suppression (``clock`` timeline; the
    absent and pair-absent recipes together) whose SI-SDRi cost against no gate stays within
    ``max_cost_db``."""
    rows = result["rows"]
    off = next(r for r in rows if r["gate"] is None)[clock]["si_sdr_i"]

    def strength(row):
        live = row[clock]
        values = [v for v in (live["suppression_db"], live["pair_absent_db"]) if v is not None]
        return sum(values) / len(values) if values else 0.0

    ok = [r for r in rows if r["gate"] is not None
          and off - r[clock]["si_sdr_i"] <= max_cost_db]
    return min(ok, key=strength) if ok else None


def format_table(result: dict[str, Any]) -> str:
    keys = [k for k in (result["rows"][1]["gate"] or {})] if len(result["rows"]) > 1 else []
    lines = ["| Gate (" + " / ".join(keys) + ") | SI-SDRi | worse > 5 dB | target absent (dB) | "
             "music only, Speech selected (dB) | interruption (Speech) |",
             "|---|---|---|---|---|---|"]
    for row in result["rows"]:
        for clock in ("live", "centred"):
            if clock not in row:
                continue
            g = row["gate"]
            name = "off" if g is None else " / ".join(f"{g[k]:g}" for k in keys)
            a = row[clock]
            fmt = lambda v, s="{:+.2f}": "-" if v is None else s.format(v)  # noqa: E731
            lines.append(f"| {name} ({clock}) | {fmt(a['si_sdr_i'])} | {a['worse']} | "
                         f"{fmt(a['suppression_db'], '{:.1f}')} | "
                         f"{fmt(a['pair_absent_db'], '{:.1f}')} | "
                         f"{fmt(a['interruption'] and 100 * a['interruption'], '{:.1f} %')} |")
    return "\n".join(lines)


def run_tuning(config: AppConfig, device: str = "auto", *, n: int = 200, bench_n: int = 40,
               progress: Callable[[str], None] = print) -> dict[str, Any]:
    from soundspy.detection.inference import load_core
    from soundspy.training.experiment import write_json

    det = config.detection
    if det is None or det.checkpoint is None:
        raise RuntimeError("no detection model is configured: the gate needs it")
    detector = load_core(Path(det.checkpoint), device if device != "auto" else det.live.device)
    cores = load_cores(config, device)
    result = tune_hybrid_gate(config, cores, detector, n=n, bench_n=bench_n, progress=progress)
    result["chosen"] = choose(result, config.extraction.hybrid.tuning_max_cost_db)
    result["max_cost_db"] = config.extraction.hybrid.tuning_max_cost_db
    folder = config.paths["experiments"] / "ext_hybrid"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "gate_tuning.json", result)
    (folder / "gate_tuning.md").write_text(format_table(result) + "\n", encoding="utf-8")
    result["path"] = str(folder / "gate_tuning.json")
    return result

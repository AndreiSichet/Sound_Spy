"""Step 07d: the speech-route gate's tuning and its synthetic regression check.

**Tuning** (``soundspy evaluate speech-gate --session <tuning session>``): DeepFilterNet3 runs
once over

* validation recipes with Speech selected: Speech-alone recipes from the validation set and the
  validation copies of the pair benchmarks (``pair_a`` speech + music, ``pair_absent`` music
  only), and
* the **tuning** real-room session (``real_benchmark.build_cases``: mixtures at four ratios, music
  only, speech only, stream starts),

its estimates and both detector timelines are kept, and every combination of ``speech_tuning`` in
``hybrid.yaml`` (``open_db > close_db`` only) is applied, next to "no gate" and the 07c hybrid
gate. The real-room cases are scored with the benchmark's own functions (pauses, speech cut,
stream starts), so the tuning and the held-out report measure the same thing.

**Rule** (live detector): the setting with the strongest music suppression in the short pauses
(0.5-2 s, 40 dB) of the tuning session, among those with

* a speech-cut rate (speech only) of at most ``max_speech_cut`` on the tuning session, and
* speech over music SI-SDRi at most ``max_cost_db`` below the 07c gate: the tuning session at
  equal loudness (classical and pop, 40 dB) and the validation speech + music pairs (30 dB).

The warm-up is chosen afterwards for the chosen setting: the shortest one whose worst stream
start leaks at most ``LEAK_TARGET_S`` on the tuning session. The held-out session and the test
set are never used here.

**Check** (``soundspy evaluate speech-gate --check``): the same systems on the test set's
Speech-selected benchmarks (Speech alone among the 500 test recipes, ``pair_a``,
``pair_absent``, ``librispeech``).
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.config import AppConfig, SpeechGateConfig
from soundspy.extraction import metrics as M

EXPERIMENT = "ext_hybrid_07d"
SYNTHETIC_SETS = ("speech", "pair_a", "pair_absent", "librispeech")
LEAK_TARGET_S = 0.3
WARMUPS_MS = (0.0, 100.0, 200.0, 300.0, 500.0, 1000.0)
GateChoice = str | SpeechGateConfig  # "none" | "07c" | a speech-gate setting


@dataclass
class Example:
    group: str  # synthetic set, or the real-room case kind
    name: str
    x: np.ndarray
    ref: np.ndarray | None  # None: the target is absent
    est: np.ndarray  # DeepFilterNet3's estimate (input gain undone), aligned with x
    sources: dict[str, Any]  # "live" / "centred" detector timelines
    latency: int
    case: Any = None  # real_benchmark.Case for the real-room examples
    meta: dict[str, Any] = field(default_factory=dict)


def _timelines(config: AppConfig, detector, audio) -> dict[str, Any]:
    from soundspy.extraction.activity import (
        causal_timeline,
        centred_timeline,
        timeline_probabilities,
    )

    det, ext = config.detection, config.extraction
    settings = dict(hop_s=det.live.hop_s, ema_alpha=det.live.ema_alpha,
                    hysteresis=det.live.hysteresis)
    raw_c, raw_k = timeline_probabilities(detector, audio, det.live.hop_s)
    return {"centred": centred_timeline(detector, audio, raw=raw_c, **settings),
            "live": causal_timeline(detector, audio, raw=raw_k,
                                    delay_s=ext.evaluation.live_delay_s,
                                    lag_s=ext.mask_baseline.noise_guard_s, **settings)}


def _speech_latency(cores, sample_rate: int) -> int:
    return cores.latencies(sample_rate)["enhancer_route"]


def synthetic_examples(config: AppConfig, cores, detector, split: str, *, n: int,
                       bench_n: int | None, progress: Callable[[str], None] = print
                       ) -> list[Example]:
    """Speech-selected recipes of ``split``: Speech alone among ``n`` recipes drawn as the
    evaluation draws them, and the benchmark sets (``bench_n`` each; ``None``: as configured)."""
    from soundspy.data.mixtures import ClipLibrary, iter_recipes, recipes_path, render_with_config
    from soundspy.extraction.speech_enhancer import enhance_whole
    from soundspy.training.evaluate_extraction import benchmark_sets, select_recipes

    ev, det = config.extraction.evaluation, config.detection
    speech = frozenset(config.extraction.hybrid.speech.labels)
    sr = ev.sample_rate
    library = ClipLibrary.from_config(config)
    path = recipes_path(config.paths["mixtures"], split)
    recipes = list(iter_recipes(path)) if path.is_file() else []
    if not recipes:
        raise RuntimeError(f"no {split} recipes: run `soundspy data mix` first")
    chosen = [("speech", r) for r in select_recipes(recipes, n, ev.seed)
              if frozenset(r.target) == speech]
    names = ("pair_a", "pair_absent", "librispeech")
    counts = None if bench_n is None else dict.fromkeys(names, bench_n)
    sets = benchmark_sets(config, library, progress, split=split, counts=counts)
    for name in names:
        chosen += [(name, r) for r in sets.get(name, []) if frozenset(r.target) == speech]
    latency = _speech_latency(cores, sr)
    out, started = [], time.perf_counter()
    for count, (group, recipe) in enumerate(chosen, 1):
        rendered = render_with_config(recipe, sr, library, config,
                                      activity_rate_hz=det.activity_rate_hz)
        x = rendered.mixture.samples.ravel()
        ref = rendered.target.samples.ravel()
        out.append(Example(group, recipe.id, x, None if M.is_silent(ref) else ref,
                           enhance_whole(cores.speech, x, input_gain_db=cores.speech_gain_db),
                           _timelines(config, detector, rendered.mixture), latency))
        if count % 50 == 0:
            progress(f"  {split}: {count}/{len(chosen)} recipes "
                     f"({time.perf_counter() - started:.0f} s)")
    return out


def session_examples(config: AppConfig, cores, detector, session: str,
                     progress: Callable[[str], None] = print) -> tuple[list[Example], int]:
    """The real-room cases of one session, with DeepFilterNet3's estimates and timelines."""
    from soundspy.cli.record_benchmark_cmd import benchmark_root
    from soundspy.core.audio import AudioBuffer
    from soundspy.extraction.speech_enhancer import enhance_whole
    from soundspy.training import real_benchmark as rb

    folders = rb.find_sessions(benchmark_root(config), session)
    takes = rb.load_takes(folders)
    cases = rb.build_cases(takes, config.extraction.hybrid.speech.labels)
    sr = next(t.sample_rate for parts in takes.values() for t in parts)
    latency = _speech_latency(cores, sr)
    out = []
    for case in cases:
        audio = AudioBuffer(case.mixture.reshape(-1, 1), sr)
        est = enhance_whole(cores.speech, case.mixture, input_gain_db=cores.speech_gain_db)
        out.append(Example(case.kind, case.name, case.mixture, case.reference, est,
                           _timelines(config, detector, audio), latency, case))
    progress(f"  session {session}: {len(out)} cases")
    return out, sr


def outputs_for(config: AppConfig, ex: Example, gate: GateChoice, clock: str,
                attenuations, sample_rate: int) -> dict[float, np.ndarray]:
    """The speech route's output at each attenuation: no gate, the 07c gate, or a setting."""
    from soundspy.extraction.neural_tse import residual_mix
    from soundspy.training.evaluate_hybrid import (
        gate_output,
        mix_with_positions,
        speech_gate_positions,
    )

    labels = frozenset(config.extraction.hybrid.speech.labels)
    if gate == "none":
        return {a: residual_mix(ex.est, ex.x, a) for a in attenuations}
    if gate == "07c":
        hg = dataclasses.replace(config.extraction.hybrid.gate, enabled=True)
        return {a: gate_output(ex.est, ex.x, labels, a, hg, ex.sources[clock], sample_rate,
                               ex.latency, warmup=True) for a in attenuations}
    pos = speech_gate_positions(ex.est, ex.x, labels, gate, ex.sources[clock], sample_rate,
                                ex.latency)
    return {a: mix_with_positions(pos, ex.est, ex.x, a) for a in attenuations}


def _m(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 3) if values else None


def score_synthetic(config: AppConfig, examples: list[Example], gate: GateChoice, clock: str,
                    attenuations, sample_rate: int) -> dict[str, Any]:
    """Per set and attenuation: SI-SDRi (target present), suppression (absent), interruption."""
    out: dict[str, Any] = {}
    for att in attenuations:
        groups: dict[str, dict[str, list]] = {}
        for ex in examples:
            y = outputs_for(config, ex, gate, clock, (att,), sample_rate)[att]
            cell = groups.setdefault(ex.group, {"si_sdr_i": [], "suppression_db": [],
                                                "interruption": []})
            if ex.ref is None:
                cell["suppression_db"].append(M.suppression_db(y, ex.x))
            else:
                cell["si_sdr_i"].append(M.si_sdr_improvement(y, ex.x, ex.ref))
                cell["interruption"].append(M.interruption_rate(y, ex.ref, sample_rate))
        out[f"{att:g}"] = {g: {k: _m(v) for k, v in c.items()} | {"n": max(map(len, c.values()))}
                           for g, c in groups.items()}
    return out


def score_session(config: AppConfig, examples: list[Example], gate: GateChoice, clock: str,
                  attenuations, sample_rate: int) -> dict[str, Any]:
    """The real-room summary (``real_benchmark.summarize_real``) for one gate and clock."""
    from soundspy.training import real_benchmark as rb

    rows = []
    for ex in examples:
        outs = outputs_for(config, ex, gate, clock, attenuations, sample_rate)
        row: dict[str, Any] = {"name": ex.name, "kind": ex.case.kind, "ratio_db": ex.case.ratio_db,
                               "music": ex.case.music}
        for att in attenuations:
            row[f"{att:g}"] = rb.score_case(ex.case, {"system": outs[att]}, sample_rate)
        rows.append(row)
    summary = rb.summarize_real(rows, attenuations)
    return {att: {group: cells.get("system", {}) for group, cells in groups.items()}
            for att, groups in summary.items()}


def speech_grid(base: SpeechGateConfig, grid: dict[str, tuple[float, ...]]
                ) -> list[SpeechGateConfig]:
    keys = sorted(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        setting = dict(zip(keys, values, strict=True))
        if setting.get("open_db", base.open_db) > setting.get("close_db", base.close_db):
            out.append(dataclasses.replace(base, enabled=True, **setting))
    return out


def gate_label(gate: GateChoice, keys) -> Any:
    return gate if isinstance(gate, str) else {k: getattr(gate, k) for k in keys}


def headline(row: dict[str, Any], clock: str = "live") -> dict[str, Any]:
    """The numbers the selection rule and the table use."""
    s, v = row["session"][clock], row["val"][clock]
    hi = s["40"]

    def cell(group, key, att=hi):
        return att.get(group, {}).get(key)

    return {
        "pause_brief_db": cell("mixes", "pause_brief_db"),
        "pause_short_db": cell("mixes", "pause_short_db"),
        "pause_long_db": cell("mixes", "pause_long_db"),
        "speech_cut": cell("speech_only", "interruption"),
        "si_classical": cell("mix +0 dB music", "si_sdr_i"),
        "si_pop": cell("mix +0 dB music_pop", "si_sdr_i"),
        "music_only_db": cell("music_only", "suppression_db"),
        "leak_s_max": cell("music_start", "leak_s_max"),
        "leak_s": cell("music_start", "leak_s"),
        "speech_start_cut": cell("speech_start", "interruption"),
        "val_speech": v["30"].get("speech", {}).get("si_sdr_i"),
        "val_pair_a": v["30"].get("pair_a", {}).get("si_sdr_i"),
        "val_pair_absent": v["30"].get("pair_absent", {}).get("suppression_db"),
    }


def choose(rows: list[dict[str, Any]], max_cut: float, max_cost_db: float
           ) -> dict[str, Any] | None:
    """The selection rule of the module docstring."""
    ref = next(r for r in rows if r["gate"] == "07c")["headline"]

    def ok(h):
        if h["speech_cut"] is None or h["speech_cut"] > max_cut:
            return False
        for key in ("si_classical", "si_pop", "val_pair_a"):
            if ref[key] is not None and (h[key] is None or h[key] < ref[key] - max_cost_db):
                return False
        return h["pause_short_db"] is not None

    candidates = [r for r in rows if isinstance(r["gate"], dict) and ok(r["headline"])]
    return min(candidates, key=lambda r: r["headline"]["pause_short_db"]) if candidates else None


def tune_speech_gate(config: AppConfig, cores, detector, session: str, *, n: int = 1000,
                     bench_n: int = 40, clocks=("live", "centred"),
                     progress: Callable[[str], None] = print) -> dict[str, Any]:
    hy, ev = config.extraction.hybrid, config.extraction.evaluation
    started = time.perf_counter()
    val = synthetic_examples(config, cores, detector, "val", n=n, bench_n=bench_n,
                             progress=progress)
    real, sr = session_examples(config, cores, detector, session, progress)
    atts = (ev.attenuation_db, ev.high_attenuation_db)
    keys = sorted(hy.speech_tuning)
    choices: list[GateChoice] = ["none", "07c", *speech_grid(hy.speech_gate, hy.speech_tuning)]
    rows = []
    for number, gate in enumerate(choices, 1):
        row: dict[str, Any] = {"gate": gate_label(gate, keys), "session": {}, "val": {}}
        for clock in clocks:
            row["session"][clock] = score_session(config, real, gate, clock, atts, sr)
            row["val"][clock] = score_synthetic(config, val, gate, clock, atts, ev.sample_rate)
        row["headline"] = headline(row)
        rows.append(row)
        progress(f"  setting {number}/{len(choices)} {row['gate']}: {row['headline']}")
    chosen = choose(rows, hy.speech_tuning_max_cut, hy.speech_tuning_max_cost_db)
    warmups = []
    if chosen is not None:
        best = dataclasses.replace(hy.speech_gate, enabled=True, **chosen["gate"])
        for warm in WARMUPS_MS:
            gate = dataclasses.replace(best, warmup_ms=warm)
            row = {"warmup_ms": warm, "session": {"live": score_session(
                config, real, gate, "live", atts, sr)}, "val": {"live": score_synthetic(
                    config, val, gate, "live", atts, ev.sample_rate)}}
            row["headline"] = headline(row)
            warmups.append(row)
            progress(f"  warm-up {warm:g} ms: {row['headline']}")
    warm_ok = [w for w in warmups if w["headline"]["leak_s_max"] is not None
               and w["headline"]["leak_s_max"] <= LEAK_TARGET_S]
    return {"session": session, "role": "tuning", "attenuations_db": list(atts),
            "groups": {g: sum(1 for e in val if e.group == g) for g in sorted({e.group
                                                                              for e in val})},
            "cases": len(real), "rows": rows, "chosen": chosen, "warmups": warmups,
            "chosen_warmup_ms": warm_ok[0]["warmup_ms"] if warm_ok else None,
            "max_speech_cut": hy.speech_tuning_max_cut,
            "max_cost_db": hy.speech_tuning_max_cost_db,
            "seconds": round(time.perf_counter() - started, 1)}


COLUMNS = (("pause_short_db", "short pauses 0.5-2 s (dB)", "{:.1f}"),
           ("pause_brief_db", "pauses < 0.5 s", "{:.1f}"),
           ("pause_long_db", "pauses >= 2 s", "{:.1f}"),
           ("speech_cut", "speech cut", "pct"),
           ("si_classical", "classical 0 dB SI-SDRi", "{:+.2f}"),
           ("si_pop", "pop 0 dB SI-SDRi", "{:+.2f}"),
           ("music_only_db", "music only (dB)", "{:.1f}"),
           ("leak_s_max", "start leak, worst (s)", "{:.2f}"),
           ("speech_start_cut", "speech-start cut", "pct"),
           ("val_speech", "val Speech alone", "{:+.2f}"),
           ("val_pair_a", "val speech + music", "{:+.2f}"),
           ("val_pair_absent", "val music only (dB)", "{:.1f}"))


def _cell(value, fmt: str) -> str:
    if value is None:
        return "-"
    return f"{100 * value:.1f} %" if fmt == "pct" else fmt.format(value)


def format_tuning(result: dict[str, Any]) -> str:
    lines = [f"# Speech-route gate tuning (session {result['session']}, tuning data; "
             "live detector)", "",
             "Session: 40 dB; validation: 30 dB. Rule: strongest short-pause suppression with "
             f"speech cut <= {100 * result['max_speech_cut']:g} % and speech over music at most "
             f"{result['max_cost_db']:g} dB below the 07c gate.", "",
             "| Gate (close_db / close_hold_ms / open_db) | "
             + " | ".join(c[1] for c in COLUMNS) + " |",
             "|---|" + "---|" * len(COLUMNS)]
    for row in result["rows"]:
        g = row["gate"]
        name = g if isinstance(g, str) else " / ".join(f"{g[k]:g}" for k in sorted(g))
        chosen = result["chosen"]
        mark = " **(chosen)**" if chosen is not None and row["gate"] == chosen["gate"] else ""
        lines.append(f"| {name}{mark} | " + " | ".join(_cell(row["headline"][k], f)
                                                       for k, _, f in COLUMNS) + " |")
    if result["warmups"]:
        lines += ["", "**Warm-up** for the chosen setting (stream starts on the tuning session):",
                  "", "| warmup_ms | " + " | ".join(c[1] for c in COLUMNS) + " |",
                  "|---|" + "---|" * len(COLUMNS)]
        for row in result["warmups"]:
            lines.append(f"| {row['warmup_ms']:g} | " + " | ".join(
                _cell(row["headline"][k], f) for k, _, f in COLUMNS) + " |")
    return "\n".join(lines)


def check_synthetic(config: AppConfig, cores, detector, *, progress: Callable[[str], None] = print
                    ) -> dict[str, Any]:
    """The test set's Speech-selected benchmarks: no gate, the 07c gate and the configured
    speech-route gate, both clocks, 30 and 40 dB."""
    hy, ev = config.extraction.hybrid, config.extraction.evaluation
    started = time.perf_counter()
    test = synthetic_examples(config, cores, detector, "test", n=ev.n, bench_n=None,
                              progress=progress)
    atts = (ev.attenuation_db, ev.high_attenuation_db)
    gate = dataclasses.replace(hy.speech_gate, enabled=True)
    rows = {}
    for name, choice in (("dfn", "none"), ("07c", "07c"), ("07d", gate)):
        rows[name] = {clock: score_synthetic(config, test, choice, clock, atts, ev.sample_rate)
                      for clock in (("centred",) if name == "dfn" else ("centred", "live"))}
    return {"split": "test", "groups": {g: sum(1 for e in test if e.group == g)
                                        for g in SYNTHETIC_SETS},
            "attenuations_db": list(atts), "rows": rows,
            "gate": {k: getattr(gate, k) for k in ("close_db", "open_db", "close_hold_ms",
                                                   "warmup_ms")},
            "seconds": round(time.perf_counter() - started, 1)}


def format_synthetic(result: dict[str, Any]) -> str:
    att = f"{result['attenuations_db'][0]:g}"
    lines = [f"# Speech-selected synthetic benchmarks (test, {att} dB)", "",
             f"Recipes: {result['groups']}. SI-SDRi (dB) where the target is present, "
             "suppression (dB) for music only.", "",
             "| System | Speech alone | speech + music | music only (dB) | LibriSpeech |",
             "|---|---|---|---|---|"]
    names = {"dfn": "DeepFilterNet3, no gate", "07c": "07c hybrid gate",
             "07d": "07d speech-route gate"}
    for name, clocks in result["rows"].items():
        for clock, scores in clocks.items():
            s = scores[att]
            cells = [_cell(s.get("speech", {}).get("si_sdr_i"), "{:+.2f}"),
                     _cell(s.get("pair_a", {}).get("si_sdr_i"), "{:+.2f}"),
                     _cell(s.get("pair_absent", {}).get("suppression_db"), "{:.1f}"),
                     _cell(s.get("librispeech", {}).get("si_sdr_i"), "{:+.2f}")]
            label = names[name] + ("" if name == "dfn" else f" ({clock} detector)")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _load(config: AppConfig, device: str):
    from soundspy.detection.inference import load_core
    from soundspy.training.evaluate_hybrid import load_cores

    det = config.detection
    if det is None or det.checkpoint is None:
        raise RuntimeError("no detection model is configured")
    detector = load_core(Path(det.checkpoint), device if device != "auto" else det.live.device)
    return load_cores(config, device), detector


def run_tuning(config: AppConfig, session: str, device: str = "auto", *, n: int = 1000,
               bench_n: int = 40, progress: Callable[[str], None] = print) -> dict[str, Any]:
    from soundspy.training.experiment import write_json

    cores, detector = _load(config, device)
    result = tune_speech_gate(config, cores, detector, session, n=n, bench_n=bench_n,
                              progress=progress)
    folder = config.paths["experiments"] / EXPERIMENT
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "speech_gate_tuning.json", result)
    (folder / "speech_gate_tuning.md").write_text(format_tuning(result) + "\n", encoding="utf-8")
    result["path"] = str(folder / "speech_gate_tuning.json")
    return result


def run_check(config: AppConfig, device: str = "auto", *,
              progress: Callable[[str], None] = print) -> dict[str, Any]:
    from soundspy.training.experiment import write_json

    cores, detector = _load(config, device)
    result = check_synthetic(config, cores, detector, progress=progress)
    folder = config.paths["experiments"] / EXPERIMENT
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "synthetic_test.json", result)
    (folder / "synthetic_test.md").write_text(format_synthetic(result) + "\n", encoding="utf-8")
    result["path"] = str(folder / "synthetic_test.json")
    return result

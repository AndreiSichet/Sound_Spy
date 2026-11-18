"""The real-room benchmark (step 07c): the laptop microphone, the real room, known ground truth.

``soundspy record-benchmark`` records the parts of one session separately into
``data/benchmark/<session>/``: ``speech`` (the owner talking with pauses, quiet room), ``music``
(classical music from a speaker, nobody talking), optionally ``music_pop`` (a song with vocals)
and ``room`` (room tone). Mixing them digitally gives mixtures whose correct output is known
exactly: the recorded speech, **with the same room and microphone colouring**, so the benchmark
measures separation, not dereverberation.

Cases (``soundspy evaluate extraction --method hybrid --benchmark real``), Speech selected:

* ``mix``: speech + music at speech-to-music ratios ``RATIOS_DB`` (speech's active level
  against the music's mean level) + the room tone at its recorded level. SI-SDRi against the
  recorded speech, and the interruption rate;
* ``music_only``: music + room tone. Suppression (output against input): the correct output is
  silence;
* ``speech_only``: the speech take alone. It must stay intact: SI-SDR of the output against the
  input, and the **interruption rate** (``metrics.interruption_rate``: the share of 50 ms speech
  frames the output turned down by more than 6 dB).

Every case runs at the evaluation's attenuation (30 dB) and at ``high_attenuation_db`` (40 dB,
the owner's listening setting). Systems: the input (bypass), the baseline with the offline and
the live detector, Waveformer v1 with and without the hybrid gate, DeepFilterNet3 alone, and the
hybrid with the offline and the live detector. Results: ``experiments/ext_hybrid/real_room.json``
and ``.md``; listening files in ``outputs/eval/ext_hybrid/real/<session>/``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.config import AppConfig
from soundspy.extraction import metrics as M

RATIOS_DB = (-5.0, 0.0, 5.0, 10.0)
PARTS = ("speech", "music", "music_pop", "room")
SYSTEMS = ("mixture", "timeline", "live", "wf", "wf_hgate", "dfn", "hybrid_offline",
           "hybrid_live")
NAMES = {"mixture": "input (bypass)", "timeline": "baseline offline", "live": "baseline live",
         "wf": "Waveformer v1", "wf_hgate": "v1 + hybrid gate", "dfn": "DeepFilterNet3 alone",
         "hybrid_offline": "hybrid (offline detector)", "hybrid_live": "hybrid (live detector)"}


class RealBenchmarkError(RuntimeError):
    """The benchmark cannot be scored (nothing recorded yet, a part missing). User message."""


@dataclass
class Take:
    part: str
    path: Path
    samples: np.ndarray  # mono float32
    sample_rate: int


@dataclass
class Case:
    name: str
    kind: str  # mix | music_only | speech_only
    mixture: np.ndarray
    reference: np.ndarray | None
    labels: frozenset[str]
    ratio_db: float | None = None
    music: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def sessions(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).iterdir() if p.is_dir()) if Path(root).is_dir() else []


def find_session(root: Path, session: str | None = None) -> Path:
    """The session folder (default: the newest one)."""
    found = sessions(root)
    if not found:
        raise RealBenchmarkError(f"no benchmark session recorded yet in {root}: record one with "
                                 "`soundspy record-benchmark`")
    if session is None:
        return found[-1]
    for folder in found:
        if folder.name == session:
            return folder
    raise RealBenchmarkError(f"no session {session!r} in {root}; sessions: "
                             f"{[f.name for f in found]}")


def load_takes(folder: Path) -> dict[str, list[Take]]:
    """Every take of a session by part (one channel, as the live chain reduces it)."""
    import soundfile as sf

    from soundspy.core.audio import AudioBuffer
    from soundspy.extraction.offline import mono

    takes: dict[str, list[Take]] = {p: [] for p in PARTS}
    for sidecar in sorted(Path(folder).glob("*.json")):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        part = meta.get("part")
        wav = sidecar.with_suffix(".wav")
        if part not in takes or not wav.is_file():
            continue
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = mono(AudioBuffer(data, int(sr))).samples.ravel()
        takes[part].append(Take(part, wav, x, int(sr)))
    return takes


def active_power(x: np.ndarray, sample_rate: int, *, frame_ms: float = 50.0,
                 range_db: float = 20.0) -> float:
    """Mean power over the frames within ``range_db`` of the loudest one (speech without its
    pauses)."""
    n = max(1, int(round(frame_ms * sample_rate / 1000.0)))
    frames = x.size // n
    if frames == 0:
        return float(np.mean(x.astype(np.float64) ** 2)) if x.size else 0.0
    p = M.frame_power(x, n, frames)
    level = 10.0 * np.log10(p + 1e-20)
    return float(p[level > level.max() - range_db].mean())


def build_cases(takes: dict[str, list[Take]], speech_labels: Collection[str],
                ratios_db: Collection[float] = RATIOS_DB) -> list[Case]:
    """The benchmark's cases (see the module docstring)."""
    speech, room = takes.get("speech", []), takes.get("room", [])
    music = [*takes.get("music", []), *takes.get("music_pop", [])]
    if not speech or not music:
        raise RealBenchmarkError("the session needs at least one speech and one music take")
    rates = {t.sample_rate for parts in takes.values() for t in parts}
    if len(rates) != 1:
        raise RealBenchmarkError(f"the takes have different sample rates: {sorted(rates)}")
    labels = frozenset(speech_labels)
    r = room[0].samples if room else None
    cases: list[Case] = []
    for s in speech:
        sr = s.sample_rate
        for m in music:
            n = min(s.samples.size, m.samples.size, *([r.size] if r is not None else []))
            sp, mu = s.samples[:n], m.samples[:n]
            music_power = float(np.mean(mu.astype(np.float64) ** 2))
            if music_power <= 0:
                raise RealBenchmarkError(f"{m.path.name} is silent")
            for ratio in ratios_db:
                gain = np.sqrt(active_power(sp, sr) / music_power / 10.0 ** (ratio / 10.0))
                mixture = sp + np.float32(gain) * mu
                if r is not None:
                    mixture = mixture + r[:n]
                cases.append(Case(f"{s.path.stem}+{m.path.stem}@{ratio:+g}dB", "mix",
                                  mixture.astype(np.float32), sp.copy(), labels, float(ratio),
                                  m.part, {"music_gain": float(gain)}))
    for m in music:
        n = min(m.samples.size, *([r.size] if r is not None else []))
        mixture = m.samples[:n] + (r[:n] if r is not None else 0.0)
        cases.append(Case(f"{m.path.stem} only", "music_only", np.asarray(mixture, np.float32),
                          None, labels, music=m.part))
    for s in speech:
        cases.append(Case(f"{s.path.stem} only", "speech_only", s.samples.copy(),
                          s.samples.copy(), labels))
    return cases


def score_case(case: Case, outputs: dict[str, np.ndarray], sample_rate: int
               ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for system, y in outputs.items():
        if case.kind == "music_only":
            out[system] = {"suppression_db": round(M.suppression_db(y, case.mixture), 2)}
        elif case.kind == "speech_only":
            rate = M.interruption_rate(y, case.reference, sample_rate)
            out[system] = {"si_sdr": round(M.si_sdr(y, case.reference), 2),
                           "level_db": round(M.suppression_db(y, case.mixture), 2),
                           "interruption": None if rate is None else round(rate, 4)}
        else:
            rate = M.interruption_rate(y, case.reference, sample_rate)
            out[system] = {"si_sdr_i": round(M.si_sdr_improvement(y, case.mixture,
                                                                  case.reference), 2),
                           "interruption": None if rate is None else round(rate, 4)}
    return out


SystemsFn = Callable[[np.ndarray, int, frozenset, float], dict[str, np.ndarray]]


def model_systems(config: AppConfig, device: str = "auto") -> SystemsFn:
    """``fn(mixture, sample_rate, labels, attenuation_db) -> {system: output}`` with the
    trained models (detector, Waveformer v1, DeepFilterNet3)."""
    from soundspy.core.audio import AudioBuffer
    from soundspy.core.target import TargetSpec
    from soundspy.detection.inference import load_core
    from soundspy.extraction.activity import (
        causal_timeline,
        centred_timeline,
        timeline_probabilities,
    )
    from soundspy.extraction.mask_baseline import MaskBaselineExtractor
    from soundspy.extraction.neural_tse import residual_mix
    from soundspy.extraction.offline import run_offline
    from soundspy.training.evaluate_hybrid import (
        chosen_route,
        gate_output,
        load_cores,
        route_estimates,
    )

    det, ext = config.detection, config.extraction
    if det is None or det.checkpoint is None:
        raise RealBenchmarkError("no detection model is configured")
    detector = load_core(Path(det.checkpoint), device if device != "auto" else det.live.device)
    cores = load_cores(config, device)
    hy = ext.hybrid
    gate = dataclasses.replace(hy.gate, enabled=True)
    settings = dict(hop_s=det.live.hop_s, ema_alpha=det.live.ema_alpha,
                    hysteresis=det.live.hysteresis)
    cache: dict[int, tuple] = {}

    def systems(mixture: np.ndarray, sr: int, labels: frozenset, att: float):
        key = id(mixture)
        if key not in cache:  # detector and estimates once per case (both attenuations)
            audio = AudioBuffer(mixture.reshape(-1, 1), sr)
            raw_c, raw_k = timeline_probabilities(detector, audio, det.live.hop_s)
            sources = {"timeline": centred_timeline(detector, audio, raw=raw_c, **settings),
                       "live": causal_timeline(detector, audio, raw=raw_k,
                                               delay_s=ext.evaluation.live_delay_s,
                                               lag_s=ext.mask_baseline.noise_guard_s,
                                               **settings)}
            cache.clear()
            cache[key] = (audio, sources, route_estimates(cores, mixture, sr, labels,
                                                          hy.speech.labels),
                          cores.latencies(sr))
        audio, sources, est, lat = cache[key]
        target = TargetSpec(labels=labels, attenuation_db=att)
        out = {"mixture": mixture}
        for name in ("timeline", "live"):
            out[name] = run_offline(MaskBaselineExtractor(ext.mask_baseline, sources[name]),
                                    audio, target).samples.ravel()
        latency = lat["cascade_route"] if hy.speech.route == "cascade" else lat["enhancer_route"]
        out["wf"] = residual_mix(est["wf"], mixture, att)
        out["wf_hgate"] = gate_output(est["wf"], mixture, labels, att, gate,
                                      sources["timeline"], sr, latency)
        route = chosen_route(config, labels)
        if "dfn" in est:
            out["dfn"] = residual_mix(est["dfn"], mixture, att)
        for name, source in (("hybrid_offline", "timeline"), ("hybrid_live", "live")):
            out[name] = gate_output(est[route], mixture, labels, att, gate, sources[source], sr,
                                    latency, warmup=route != "wf")
        return out

    return systems


def evaluate_real(config: AppConfig, *, session: str | None = None, device: str = "auto",
                  systems_fn: SystemsFn | None = None, write: bool = True,
                  progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Score one recorded session (see the module docstring)."""
    from soundspy.cli.record_benchmark_cmd import benchmark_root

    folder = find_session(benchmark_root(config), session)
    takes = load_takes(folder)
    cases = build_cases(takes, config.extraction.hybrid.speech.labels)
    sr = next(t.sample_rate for parts in takes.values() for t in parts)
    ev = config.extraction.evaluation
    attenuations = (ev.attenuation_db, ev.high_attenuation_db)
    fn = systems_fn if systems_fn is not None else model_systems(config, device)
    progress(f"session {folder.name}: {sum(len(v) for v in takes.values())} takes, "
             f"{len(cases)} cases")
    started = time.perf_counter()
    rows = []
    examples_dir = config.paths["outputs"] / "eval" / "ext_hybrid" / "real" / folder.name
    for case in cases:
        row: dict[str, Any] = {"name": case.name, "kind": case.kind, "ratio_db": case.ratio_db,
                               "music": case.music, "seconds": round(case.mixture.size / sr, 1)}
        for att in attenuations:
            outputs = fn(case.mixture, sr, case.labels, att)
            row[f"{att:g}"] = score_case(case, outputs, sr)
            if write and att == ev.high_attenuation_db and (
                    case.kind != "mix" or case.ratio_db == 0.0):
                _write_example(examples_dir, case, outputs, sr)
        rows.append(row)
    result = {"session": folder.name, "folder": str(folder), "sample_rate": sr,
              "attenuations_db": list(attenuations), "ratios_db": list(RATIOS_DB),
              "takes": {p: [t.path.name for t in v] for p, v in takes.items() if v},
              "cases": rows, "summary": summarize_real(rows, attenuations),
              "seconds": round(time.perf_counter() - started, 1)}
    if write:
        from soundspy.training.experiment import read_json, write_json

        exp = config.paths["experiments"] / "ext_hybrid"
        exp.mkdir(parents=True, exist_ok=True)
        path = exp / "real_room.json"
        previous = read_json(path) if path.is_file() else {}
        write_json(path, {**previous, folder.name: result})
        (exp / "real_room.md").write_text(format_real(result) + "\n", encoding="utf-8")
        result["path"] = str(path)
        result["examples_dir"] = str(examples_dir)
    return result


def summarize_real(rows: list[dict[str, Any]], attenuations) -> dict[str, Any]:
    """Means per (attenuation, kind / ratio, system)."""

    def mean(values):
        values = [v for v in values if v is not None]
        return round(float(np.mean(values)), 3) if values else None

    out: dict[str, Any] = {}
    for att in attenuations:
        key = f"{att:g}"
        groups: dict[str, list[dict]] = {}
        for row in rows:
            name = (f"mix {row['ratio_db']:+g} dB" if row["kind"] == "mix" else row["kind"])
            groups.setdefault(name, []).append(row[key])
        out[key] = {}
        for name, items in groups.items():
            systems = list(items[0])
            out[key][name] = {s: {m: mean(item[s].get(m) for item in items if s in item)
                                  for m in items[0][s]} for s in systems}
    return out


def format_real(result: dict[str, Any]) -> str:
    low, high = (f"{a:g}" for a in result["attenuations_db"])
    summary = result["summary"]
    lines = [f"# Real-room benchmark, session {result['session']}", "",
             f"Takes: {result['takes']}. Speech selected. Mixtures: SI-SDRi (dB) at {low} dB; "
             f"music only: suppression (dB) and speech only: SI-SDR against the input (dB) / "
             f"interruption rate at {high} dB.", ""]
    mixes = [k for k in summary[low] if k.startswith("mix")]
    systems = [s for s in SYSTEMS if any(s in summary[low][k] for k in summary[low])]
    lines += ["| System | " + " | ".join(mixes) + " | music only | speech only SI-SDR | "
              "speech only interruption |", "|---|" + "---|" * (len(mixes) + 3)]
    for s in systems:
        cells = []
        for k in mixes:
            v = summary[low][k].get(s, {}).get("si_sdr_i")
            cells.append("-" if v is None else f"{v:+.2f}")
        music = summary[high].get("music_only", {}).get(s, {}).get("suppression_db")
        speech = summary[high].get("speech_only", {}).get(s, {})
        inter, si = speech.get("interruption"), speech.get("si_sdr")
        music_text = "-" if music is None else f"{music:.1f}"
        si_text = "-" if si is None else f"{si:+.1f}"
        inter_text = "-" if inter is None else f"{100 * inter:.1f} %"
        lines.append(f"| {NAMES.get(s, s)} | " + " | ".join(cells)
                     + f" | {music_text} | {si_text} | {inter_text} |")
    return "\n".join(lines)


def _write_example(folder: Path, case: Case, outputs: dict[str, np.ndarray], sr: int) -> None:
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    stem = case.name.replace(" ", "_").replace("+", "_").replace("@", "_at_")
    sf.write(str(folder / f"{stem}_input.wav"), case.mixture, sr, subtype="FLOAT")
    if case.reference is not None and case.kind == "mix":
        sf.write(str(folder / f"{stem}_speech.wav"), case.reference, sr, subtype="FLOAT")
    for system, y in outputs.items():
        if system != "mixture":
            sf.write(str(folder / f"{stem}_{system}.wav"), y, sr, subtype="FLOAT")

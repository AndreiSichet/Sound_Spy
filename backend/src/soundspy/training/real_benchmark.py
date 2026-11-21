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
  frames the output turned down by more than 6 dB);
* ``music_start`` (step 07d): a stream that **starts while the music plays** (the music take
  from several offsets, ``START_SECONDS`` long): ``leak_s``, the time in which the output is
  less than ``LEAK_DB`` (30 dB) below the input (20 ms frames);
* ``speech_start`` (step 07d): a stream that starts ``START_LEAD_S`` before the speaker's first
  word after a pause: the interruption rate of that start.

**Pauses** (step 07d, ``mix`` cases): the speech take's 50 ms frames more than ``PAUSE_MARGIN_DB``
above the room's level are speech, the others silence; a run of silent frames between two
speech frames is a pause, grouped by its length (``PAUSE_GROUPS``: brief < 0.5 s, short
0.5-2 s, long >= 2 s).
Per group the music's suppression in the pauses: the output's energy against the input's,
summed over every pause of the group (``pause_<group>_db``; the room level is the ``room``
take's median 50 ms level, or the speech take's 5th percentile without one).

Every case runs at the evaluation's attenuation (30 dB) and at ``high_attenuation_db`` (40 dB,
the owner's listening setting). Systems: the input (bypass), the baseline with the offline and
the live detector, Waveformer v1 with and without the hybrid gate, DeepFilterNet3 alone, the
speech route with the 07c hybrid gate (``hgate_*``) and the hybrid as configured (speech-route
gate, step 07d), each with the offline and the live detector. Results:
``experiments/<name>/real_room.json`` and ``.md``; listening files in
``outputs/eval/<name>/real/<session>/`` (``name`` default ``ext_hybrid``).
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
SYSTEMS = ("mixture", "timeline", "live", "wf", "wf_hgate", "dfn", "hgate_offline",
           "hgate_live", "hybrid_offline", "hybrid_live")
NAMES = {"mixture": "input (bypass)", "timeline": "baseline offline", "live": "baseline live",
         "wf": "Waveformer v1", "wf_hgate": "v1 + hybrid gate", "dfn": "DeepFilterNet3 alone",
         "hgate_offline": "07c gate (offline detector)", "hgate_live": "07c gate (live detector)",
         "hybrid_offline": "hybrid (offline detector)", "hybrid_live": "hybrid (live detector)"}
FRAME_MS = 50.0
PAUSE_MARGIN_DB = 6.0
PAUSE_GROUPS = (("brief", 0.0, 0.5), ("short", 0.5, 2.0), ("long", 2.0, float("inf")))
PAUSE_TITLES = {"brief": "< 0.5 s", "short": "0.5-2 s", "long": ">= 2 s"}
START_SECONDS = 5.0  # music_start: stream length
START_EVERY_S = 6.0  # music_start: one stream start every this many seconds of the take
START_LEAD_S = 0.2  # speech_start: the stream starts this long before the word
START_SPEECH_S = 3.0  # speech_start: stream length
START_SPEECH_MAX = 8  # speech_start: at most this many starts per take
LEAK_DB = -30.0
LEAK_FRAME_MS = 20.0


class RealBenchmarkError(RuntimeError):
    """The benchmark cannot be scored (nothing recorded yet, a part missing). User message."""


@dataclass
class Take:
    part: str
    path: Path
    samples: np.ndarray  # mono float32
    sample_rate: int
    name: str = ""  # case names use it (default: the file's stem)

    @property
    def label(self) -> str:
        return self.name or self.path.stem


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


def find_sessions(root: Path, session: str | None = None) -> list[Path]:
    """One session, or several joined by ``+`` (their takes are scored together: e.g. a
    second speech take recorded right after a session, step 07d)."""
    if session is None or "+" not in session:
        return [find_session(root, session)]
    return [find_session(root, name) for name in session.split("+") if name]


def load_takes(folders: Path | list[Path], max_seconds: float | None = None
               ) -> dict[str, list[Take]]:
    """Every take of the sessions by part (one channel, as the live chain reduces it).
    ``max_seconds``: only the first seconds of every take (e.g. to leave out a phone that rang
    at the end; the recordings themselves stay as they are). With several sessions a take's
    name starts with its session's last four characters."""
    import soundfile as sf

    from soundspy.core.audio import AudioBuffer
    from soundspy.extraction.offline import mono

    folders = [Path(folders)] if isinstance(folders, str | Path) else [Path(f) for f in folders]
    takes: dict[str, list[Take]] = {p: [] for p in PARTS}
    for folder in folders:
        for sidecar in sorted(folder.glob("*.json")):
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            part = meta.get("part")
            wav = sidecar.with_suffix(".wav")
            if part not in takes or not wav.is_file():
                continue
            data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            x = mono(AudioBuffer(data, int(sr))).samples.ravel()
            if max_seconds is not None:
                x = x[:int(round(max_seconds * sr))]
            name = f"{folder.name[-4:]}_{wav.stem}" if len(folders) > 1 else ""
            takes[part].append(Take(part, wav, x, int(sr), name))
    return takes


def room_for(room: np.ndarray | None, n: int) -> np.ndarray | None:
    """The room tone for ``n`` samples: repeated when the room take is shorter (it is
    stationary; the takes are 30 s, speech and music 60 s)."""
    if room is None or room.size == 0:
        return None
    return room[:n] if room.size >= n else np.resize(room, n).astype(np.float32)


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


def frame_levels_db(x: np.ndarray, sample_rate: int, frame_ms: float = FRAME_MS) -> np.ndarray:
    """Level (dBFS, mean power) of every whole ``frame_ms`` frame."""
    n = max(1, int(round(frame_ms * sample_rate / 1000.0)))
    frames = x.size // n
    if frames == 0:
        return np.zeros(0)
    return 10.0 * np.log10(M.frame_power(x, n, frames) + 1e-20)


def room_floor_db(speech: np.ndarray, room: np.ndarray | None, sample_rate: int) -> float:
    """The room's level: the room take's median 50 ms level, else the speech take's 5th
    percentile (its quietest frames are room tone)."""
    if room is not None and room.size >= sample_rate * FRAME_MS / 1000.0:
        return float(np.median(frame_levels_db(room, sample_rate)))
    return float(np.percentile(frame_levels_db(speech, sample_rate), 5))


def find_pauses(reference: np.ndarray, sample_rate: int, floor_db: float, *,
                margin_db: float = PAUSE_MARGIN_DB) -> list[tuple[int, int]]:
    """``(first, end)`` frame indices (``FRAME_MS``) of every pause: silent frames (at most
    ``margin_db`` above the room) between two speech frames. Silence before the first and after
    the last word is not a pause."""
    silent = frame_levels_db(reference, sample_rate) <= floor_db + margin_db
    out, start = [], None
    for i, quiet in enumerate(silent):
        if quiet and start is None:
            start = i
        elif not quiet and start is not None:
            if start > 0:
                out.append((start, i))
            start = None
    return out


def pause_group(seconds: float) -> str:
    return next(name for name, lo, hi in PAUSE_GROUPS if lo <= seconds < hi)


def pause_energies(output: np.ndarray, mixture: np.ndarray, pauses: list[tuple[int, int]],
                   sample_rate: int) -> dict[str, float]:
    """``pause_<group>_out`` / ``_in`` (summed energies) and ``_n`` (pauses) per group; the
    summary turns the means into ``pause_<group>_db``."""
    n = max(1, int(round(FRAME_MS * sample_rate / 1000.0)))
    out = {f"pause_{g}_{k}": 0.0 for g, _, _ in PAUSE_GROUPS for k in ("out", "in", "n")}
    y = np.asarray(output, dtype=np.float64).ravel()
    x = np.asarray(mixture, dtype=np.float64).ravel()
    for first, end in pauses:
        group = pause_group((end - first) * FRAME_MS / 1000.0)
        a, b = first * n, end * n
        out[f"pause_{group}_out"] += float(y[a:b] @ y[a:b])
        out[f"pause_{group}_in"] += float(x[a:b] @ x[a:b])
        out[f"pause_{group}_n"] += 1
    return out


def leak_seconds(output: np.ndarray, mixture: np.ndarray, sample_rate: int, *,
                 threshold_db: float = LEAK_DB, frame_ms: float = LEAK_FRAME_MS) -> float:
    """Time in which the output is less than ``-threshold_db`` below the input (frame by
    frame); frames of digital silence in the input do not count."""
    n = max(1, int(round(frame_ms * sample_rate / 1000.0)))
    frames = min(output.size, mixture.size) // n
    if frames == 0:
        return 0.0
    po, pi = M.frame_power(output, n, frames), M.frame_power(mixture, n, frames)
    leak = (pi > 1e-14) & (po > pi * 10.0 ** (threshold_db / 10.0))
    return float(leak.sum() * frame_ms / 1000.0)


def speech_onsets(speech: np.ndarray, sample_rate: int, floor_db: float, *,
                  min_pause_s: float = 0.5) -> list[int]:
    """Sample indices of words that follow a pause of at least ``min_pause_s``."""
    n = int(round(FRAME_MS * sample_rate / 1000.0))
    return [end * n for first, end in find_pauses(speech, sample_rate, floor_db)
            if (end - first) * FRAME_MS / 1000.0 >= min_pause_s]


def build_cases(takes: dict[str, list[Take]], speech_labels: Collection[str],
                ratios_db: Collection[float] = RATIOS_DB, *, starts: bool = True) -> list[Case]:
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
        floor = room_floor_db(s.samples, r, sr)
        for m in music:
            n = min(s.samples.size, m.samples.size)
            room_n = room_for(r, n)
            sp, mu = s.samples[:n], m.samples[:n]
            music_power = float(np.mean(mu.astype(np.float64) ** 2))
            if music_power <= 0:
                raise RealBenchmarkError(f"{m.path.name} is silent")
            pauses = find_pauses(sp, sr, floor)
            for ratio in ratios_db:
                gain = np.sqrt(active_power(sp, sr) / music_power / 10.0 ** (ratio / 10.0))
                mixture = sp + np.float32(gain) * mu
                if r is not None:
                    mixture = mixture + room_n
                cases.append(Case(f"{s.label}+{m.label}@{ratio:+g}dB", "mix",
                                  mixture.astype(np.float32), sp.copy(), labels, float(ratio),
                                  m.part, {"music_gain": float(gain), "floor_db": floor,
                                           "pauses": pauses}))
    for m in music:
        n = m.samples.size
        room_n = room_for(r, n)
        mixture = np.asarray(m.samples[:n] + (room_n if r is not None else 0.0), np.float32)
        cases.append(Case(f"{m.label} only", "music_only", mixture, None, labels,
                          music=m.part))
        if starts:  # a stream that starts while the music plays, from several offsets
            sr = m.sample_rate
            length, every = int(START_SECONDS * sr), int(START_EVERY_S * sr)
            for off in range(0, n - length + 1, every):
                cases.append(Case(f"{m.label} start@{off / sr:g}s", "music_start",
                                  mixture[off:off + length].copy(), None, labels,
                                  music=m.part, meta={"offset_s": off / sr}))
    for s in speech:
        cases.append(Case(f"{s.label} only", "speech_only", s.samples.copy(),
                          s.samples.copy(), labels))
        if starts:  # a stream that starts just before a word
            sr = s.sample_rate
            onsets = speech_onsets(s.samples, sr, room_floor_db(s.samples, r, sr))
            step = max(1, len(onsets) // START_SPEECH_MAX)
            for onset in onsets[::step][:START_SPEECH_MAX]:
                a = max(0, onset - int(START_LEAD_S * sr))
                seg = s.samples[a:a + int(START_SPEECH_S * sr)].copy()
                cases.append(Case(f"{s.label} start@{a / sr:.2f}s", "speech_start", seg,
                                  seg.copy(), labels, meta={"offset_s": a / sr}))
    return cases


def score_case(case: Case, outputs: dict[str, np.ndarray], sample_rate: int
               ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for system, y in outputs.items():
        if case.kind == "music_only":
            out[system] = {"suppression_db": round(M.suppression_db(y, case.mixture), 2)}
        elif case.kind == "music_start":
            out[system] = {"leak_s": round(leak_seconds(y, case.mixture, sample_rate), 3),
                           "suppression_db": round(M.suppression_db(y, case.mixture), 2)}
        elif case.kind in ("speech_only", "speech_start"):
            rate = M.interruption_rate(y, case.reference, sample_rate)
            out[system] = {"si_sdr": round(M.si_sdr(y, case.reference), 2),
                           "level_db": round(M.suppression_db(y, case.mixture), 2),
                           "interruption": None if rate is None else round(rate, 4)}
        else:
            rate = M.interruption_rate(y, case.reference, sample_rate)
            out[system] = {"si_sdr_i": round(M.si_sdr_improvement(y, case.mixture,
                                                                  case.reference), 2),
                           "interruption": None if rate is None else round(rate, 4),
                           **pause_energies(y, case.mixture, case.meta.get("pauses", []),
                                            sample_rate)}
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
        configured_output,
        gate_output,
        load_cores,
        route_estimates,
        route_latency,
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
        out["wf"] = residual_mix(est["wf"], mixture, att)
        out["wf_hgate"] = gate_output(est["wf"], mixture, labels, att, gate,
                                      sources["timeline"], sr,
                                      route_latency(config, cores, sr, "wf"))
        route = chosen_route(config, labels)
        latency = route_latency(config, cores, sr, route)
        if "dfn" in est:
            out["dfn"] = residual_mix(est["dfn"], mixture, att)
        for clock, source in (("offline", "timeline"), ("live", "live")):
            if route != "wf":  # the speech route with the 07c gate, for comparison
                out[f"hgate_{clock}"] = gate_output(est[route], mixture, labels, att, gate,
                                                    sources[source], sr, latency, warmup=True)
            out[f"hybrid_{clock}"] = configured_output(config, route, est[route], mixture,
                                                       labels, att, sources[source], sr,
                                                       latency)
        return out

    return systems


def evaluate_real(config: AppConfig, *, session: str | None = None, device: str = "auto",
                  systems_fn: SystemsFn | None = None, write: bool = True,
                  name: str = "ext_hybrid", role: str | None = None,
                  max_seconds: float | None = None,
                  progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Score one recorded session, or several joined by ``+`` (see the module docstring).
    ``role``: ``tuning`` / ``held-out``, stored with the result (step 07d keeps the two apart);
    ``max_seconds``: only the first seconds of every take."""
    from soundspy.cli.record_benchmark_cmd import benchmark_root

    folders = find_sessions(benchmark_root(config), session)
    session_name = "+".join(f.name for f in folders)
    takes = load_takes(folders, max_seconds)
    cases = build_cases(takes, config.extraction.hybrid.speech.labels)
    sr = next(t.sample_rate for parts in takes.values() for t in parts)
    ev = config.extraction.evaluation
    attenuations = (ev.attenuation_db, ev.high_attenuation_db)
    fn = systems_fn if systems_fn is not None else model_systems(config, device)
    progress(f"session {session_name}: {sum(len(v) for v in takes.values())} takes, "
             f"{len(cases)} cases")
    started = time.perf_counter()
    rows = []
    examples_dir = config.paths["outputs"] / "eval" / name / "real" / session_name
    for count, case in enumerate(cases, 1):
        row: dict[str, Any] = {"name": case.name, "kind": case.kind, "ratio_db": case.ratio_db,
                               "music": case.music, "seconds": round(case.mixture.size / sr, 1)}
        if "pauses" in case.meta:
            row["pauses"] = {g: sum(1 for a, b in case.meta["pauses"]
                                    if pause_group((b - a) * FRAME_MS / 1000.0) == g)
                             for g, _, _ in PAUSE_GROUPS}
        for att in attenuations:
            outputs = fn(case.mixture, sr, case.labels, att)
            row[f"{att:g}"] = score_case(case, outputs, sr)
            if write and att == ev.high_attenuation_db and (
                    case.kind in ("music_only", "speech_only")
                    or (case.kind == "mix" and case.ratio_db == 0.0)):
                _write_example(examples_dir, case, outputs, sr)
        rows.append(row)
        if count % 10 == 0:
            progress(f"  {count}/{len(cases)} cases ({time.perf_counter() - started:.0f} s)")
    result = {"session": session_name, "folder": ";".join(str(f) for f in folders),
              "sample_rate": sr, "role": role, "max_seconds": max_seconds,
              "attenuations_db": list(attenuations), "ratios_db": list(RATIOS_DB),
              "takes": {p: [t.path.name for t in v] for p, v in takes.items() if v},
              "cases": rows, "summary": summarize_real(rows, attenuations),
              "seconds": round(time.perf_counter() - started, 1)}
    if write:
        from soundspy.training.experiment import read_json, write_json

        exp = config.paths["experiments"] / name
        exp.mkdir(parents=True, exist_ok=True)
        path = exp / "real_room.json"
        previous = read_json(path) if path.is_file() else {}
        write_json(path, {**previous, session_name: result})
        md = exp / f"real_room_{session_name}.md" if name != "ext_hybrid" else exp / "real_room.md"
        md.write_text(format_real(result) + "\n", encoding="utf-8")
        result["path"] = str(path)
        result["examples_dir"] = str(examples_dir)
    return result


def summary_groups(row: dict[str, Any]) -> list[str]:
    """The summary groups a case row counts in: its kind (mixes by ratio, all mixes, and by
    ratio and music) and, for music takes, its kind by music."""
    kind = row["kind"]
    if kind == "mix":
        return [f"mix {row['ratio_db']:+g} dB", "mixes",
                f"mix {row['ratio_db']:+g} dB {row['music']}", f"mixes {row['music']}"]
    if kind in ("music_only", "music_start") and row.get("music"):
        return [kind, f"{kind} {row['music']}"]
    return [kind]


def summarize_real(rows: list[dict[str, Any]], attenuations) -> dict[str, Any]:
    """Means per (attenuation, group, system); pause suppression pooled over the group's
    pauses (``pause_<group>_db`` from the summed energies); ``leak_s_max`` for stream starts."""

    def mean(values):
        values = [v for v in values if v is not None]
        return round(float(np.mean(values)), 4) if values else None

    out: dict[str, Any] = {}
    for att in attenuations:
        key = f"{att:g}"
        groups: dict[str, list[dict]] = {}
        for row in rows:
            for name in summary_groups(row):
                groups.setdefault(name, []).append(row[key])
        out[key] = {}
        for name, items in groups.items():
            systems = list(items[0])
            summary = {}
            for s in systems:
                mine = [item[s] for item in items if s in item]
                pause_keys = {f"pause_{g}_{k}" for g, _, _ in PAUSE_GROUPS
                              for k in ("out", "in", "n")}
                cells = {m: mean(c.get(m) for c in mine) for m in items[0][s]
                         if m not in pause_keys}
                for g, _, _ in PAUSE_GROUPS:
                    if f"pause_{g}_in" in items[0][s]:
                        # pooled from the raw sums (energies are tiny: never round before this)
                        e_out = sum(float(c.get(f"pause_{g}_out", 0.0)) for c in mine)
                        e_in = sum(float(c.get(f"pause_{g}_in", 0.0)) for c in mine)
                        cells[f"pause_{g}_n"] = int(round(sum(c.get(f"pause_{g}_n", 0)
                                                              for c in mine)))
                        cells[f"pause_{g}_db"] = (
                            round(max(-M.CAP_DB, 10.0 * np.log10(max(e_out, 1e-30) / e_in)), 2)
                            if e_in > 0 else None)
                if "leak_s" in cells:
                    cells["leak_s_max"] = max(item[s]["leak_s"] for item in items if s in item)
                summary[s] = cells
            out[key][name] = summary
    return out


def format_real(result: dict[str, Any]) -> str:
    low, high = (f"{a:g}" for a in result["attenuations_db"])
    summary = result["summary"]
    role = f" ({result['role']} session)" if result.get("role") else ""
    lines = [f"# Real-room benchmark, session {result['session']}{role}", "",
             f"Takes: {result['takes']}. Speech selected. Mixtures: SI-SDRi (dB) at {low} dB; "
             f"music only: suppression (dB) and speech only: SI-SDR against the input (dB) / "
             f"interruption rate at {high} dB.", ""]
    mixes = [k for k in summary[low] if k.startswith("mix ") and k.endswith(" dB")]
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
    lines += _format_extra(result, systems)
    return "\n".join(lines)


def _f(value, fmt: str = "{:+.2f}") -> str:
    return "-" if value is None else fmt.format(value)


def _format_extra(result: dict[str, Any], systems: list[str]) -> list[str]:
    """Pauses, equal loudness per music, stream starts (step 07d)."""
    summary, atts = result["summary"], [f"{a:g}" for a in result["attenuations_db"]]
    lines: list[str] = []
    mixes = summary[atts[0]].get("mixes")
    if mixes:
        first = next(iter(mixes.values()))
        counts = " / ".join(f"{PAUSE_TITLES[g]}: {first.get(f'pause_{g}_n', 0)}"
                            for g, _, _ in PAUSE_GROUPS)
        lines += ["", "**Music in the pauses** (suppression, dB, pooled over the pauses of every "
                  f"mixture; pauses per group over all mixtures: {counts})", "",
                  "| System | " + " | ".join(f"{PAUSE_TITLES[g]} at {a} dB" for a in atts
                                            for g, _, _ in PAUSE_GROUPS) + " |",
                  "|---|" + "---|" * (len(atts) * len(PAUSE_GROUPS))]
        for s in systems:
            cells = [_f(summary[a]["mixes"].get(s, {}).get(f"pause_{g}_db"), "{:.1f}")
                     for a in atts for g, _, _ in PAUSE_GROUPS]
            lines.append(f"| {NAMES.get(s, s)} | " + " | ".join(cells) + " |")
    musics = sorted({k.split(" ", 3)[3] for k in summary[atts[0]]
                     if k.startswith("mix +0 dB ")})
    if musics:
        lines += ["", "**Equal loudness (0 dB) per music**: SI-SDRi (dB) / interruption", "",
                  "| System | " + " | ".join(f"{m} at {a} dB" for m in musics for a in atts)
                  + " |", "|---|" + "---|" * (len(musics) * len(atts))]
        for s in systems:
            cells = []
            for m in musics:
                for a in atts:
                    c = summary[a].get(f"mix +0 dB {m}", {}).get(s, {})
                    inter = c.get("interruption")
                    cells.append(f"{_f(c.get('si_sdr_i'))} / "
                                 f"{_f(inter and 100 * inter, '{:.1f} %')}")
            lines.append(f"| {NAMES.get(s, s)} | " + " | ".join(cells) + " |")
    starts = [k for k in summary[atts[-1]] if k.startswith("music_start ")]
    if starts or "speech_start" in summary[atts[-1]]:
        high = atts[-1]
        lines += ["", f"**Stream starts** at {high} dB: music playing when the stream starts "
                  f"(time with the output less than {-LEAK_DB:g} dB below the input, in the "
                  f"first {START_SECONDS:g} s: mean / worst); speech starting "
                  f"{START_LEAD_S:g} s after the stream (interruption rate)", "",
                  "| System | " + " | ".join(k.replace("music_start ", "music start, ")
                                            for k in starts) + " | speech start |",
                  "|---|" + "---|" * (len(starts) + 1)]
        for s in systems:
            cells = []
            for k in starts:
                c = summary[high][k].get(s, {})
                cells.append(f"{_f(c.get('leak_s'), '{:.2f}')} / "
                             f"{_f(c.get('leak_s_max'), '{:.2f}')} s")
            inter = summary[high].get("speech_start", {}).get(s, {}).get("interruption")
            cells.append(_f(inter and 100 * inter, "{:.1f} %"))
            lines.append(f"| {NAMES.get(s, s)} | " + " | ".join(cells) + " |")
    return lines


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

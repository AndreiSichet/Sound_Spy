"""``soundspy report``: the evaluation document for the university report (step 08).

Regenerates ``docs/evaluation.md`` and its figures (``docs/figures/*.png``) **from the stored
result files** of every step (``experiments/*/metrics.json``, ``real_room.json``,
``experiments/realtime/*.json``); nothing is re-run unless ``--rerun`` asks for the long
evaluations first. Every table says which command reproduces it. A result file that is missing
leaves its section with a note instead of failing, so the document can be built at any stage.

Figures (matplotlib, light surface, the reference palette of the data-viz guidance: categorical
slots in fixed order, thin bars with a 2 px gap, a value label on every bar so colour is never
the only carrier, a legend whenever there are two or more series, recessive grid): detection AP
per class (zero-shot vs fine-tuned), SI-SDRi of every extraction system, music in the pauses per
system (held-out real room), and the delay of each route in its parts.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from soundspy.config import AppConfig

# reference palette (light): categorical slots 1-5, ink, chrome
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
UNREPORTED = "#b9bec5"  # neutral grey: the remainder, not a series
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")

DATASETS = (
    ("ESC-50", "environmental sound clips", "CC BY-NC 3.0"),
    ("FSD50K", "Freesound clips with AudioSet labels", "per clip: CC0, CC BY, CC BY-NC"),
    ("LibriSpeech (test-clean, dev-clean)", "read English speech", "CC BY 4.0"),
    ("AudioSet ontology", "class hierarchy", "CC BY-SA 4.0"),
)
MODELS = (
    ("EfficientAT mn10_as", "detector, fine-tuned", "MIT"),
    ("Waveformer", "target sound extraction, fine-tuned (v1)", "MIT"),
    ("DeepFilterNet3", "speech enhancer, published weights", "MIT / Apache-2.0"),
)


def _load(path: Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _f(value, fmt: str = "{:+.2f}") -> str:
    if value is None:
        return "–"
    try:
        return fmt.format(value).replace("-", "−")
    except (TypeError, ValueError):
        return str(value)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(r) + " |" for r in rows)]


def _reproduce(command: str) -> str:
    return f"*Reproduce:* `{command}`"


# --- data -------------------------------------------------------------------------------------


class Results:
    """Every stored result the document uses (``None`` where a file is missing)."""

    def __init__(self, config: AppConfig) -> None:
        exp = Path(config.paths["experiments"])
        self.exp = exp
        self.detection = _load(exp / "det_mn10_v1" / "metrics.json")
        self.baseline = _load(exp / "ext_mask_baseline" / "metrics.json")
        self.v1 = _load(exp / "ext_neural_tse" / "metrics.json")
        self.v2 = _load(exp / "ext_neural_tse_v2" / "metrics.json")
        self.v1_gate = _load(exp / "ext_neural_tse_v1_gate" / "metrics.json")
        self.hybrid_07c = _load(exp / "ext_hybrid" / "metrics.json")
        self.hybrid_08 = _load(exp / "ext_hybrid_08" / "metrics.json")
        self.speech_check = _load(exp / "ext_hybrid_07d" / "synthetic_test.json")
        self.real_07d = _load(exp / "ext_hybrid_07d" / "real_room.json") or {}
        self.latency = _load(exp / "realtime" / "latency_latest.json")
        self.screening = _load(exp / "realtime" / "latency_screening.json") or {}
        self.startup = _load(exp / "realtime" / "startup.json")
        files = sorted((exp / "realtime").glob("soak_*.json")) if (exp / "realtime").is_dir() \
            else []
        runs = [d for d in (_load(p) for p in files) if d and d.get("summary")]
        # the configured settings only (no ``--set``); the latest run per route counts
        plain = [d for d in runs if not d.get("overrides")]
        self.soaks = [d for d in plain if d.get("minutes", 0) >= 10]  # every long run, in order
        self.final_soak = {}
        for d in plain:
            if d.get("minutes", 0) >= 3:
                self.final_soak[d.get("route")] = d
        long = [d for d in self.soaks if d.get("minutes", 0) >= 25]
        self.soak = long[-1] if long else None

    @staticmethod
    def overall(metrics: dict | None, system: str) -> dict | None:
        try:
            return metrics["test"]["overall"][system]
        except (KeyError, TypeError):
            return None

    @staticmethod
    def bench(metrics: dict | None, bench: str, system: str) -> dict | None:
        try:
            return metrics["test"]["benchmarks"][bench]["overall"][system]
        except (KeyError, TypeError):
            return None

    def hybrid_latest(self) -> tuple[dict | None, str]:
        if self.hybrid_08 is not None:
            return self.hybrid_08, "hybrid (07d gate, per-route delay)"
        return self.hybrid_07c, "hybrid (07c gate)"


# --- sections ---------------------------------------------------------------------------------


def _os_name() -> str:
    """``Windows 11 (build 26200)``: ``platform`` still calls Windows 11 "10"."""
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        return f"Windows {11 if build >= 22000 else 10} (build {build})"
    return platform.platform()


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "?"


def _versions() -> list[list[str]]:
    rows = [["Python", platform.python_version()], ["OS", _os_name()]]
    for name in ("numpy", "torch", "onnxruntime", "sounddevice", "fastapi", "soundfile"):
        try:
            module = __import__(name)
            rows.append([name, str(getattr(module, "__version__", "?"))])
        except Exception:  # noqa: BLE001 - a missing optional package is just not listed
            rows.append([name, "not installed"])
    from soundspy import __version__

    rows.append(["soundspy", __version__])
    return rows


def _hardware() -> list[list[str]]:
    import os

    from soundspy.engine import resources  # noqa: F401 - keeps the import graph honest

    rows = [["CPU", _cpu_name()], ["Logical cores", str(os.cpu_count())]]
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            rows.append(["RAM", f"{status.ullTotalPhys / 2**30:.0f} GB"])
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            rows.append(["Graphics card", f"{props.name}, {props.total_memory / 2**30:.0f} GB"])
    except Exception:  # noqa: BLE001
        pass
    rows.append(["Microphone", "laptop's built-in array (WASAPI, 48 kHz), Windows audio "
                               "enhancements off"])
    return rows


def section_setup(config: AppConfig) -> list[str]:
    mix = config.data.mixtures
    classes = [[f"`{c.name}`", c.display_name, c.description] for c in config.labels.classes]
    lines = ["## 1. Setup", "", "### Hardware", "", *_table(["", ""], _hardware()), "",
             "### Software", "", *_table(["package", "version"], _versions()), "",
             "### Datasets and models", "",
             *_table(["dataset", "what", "licence"], [list(d) for d in DATASETS]), "",
             *_table(["model", "role", "licence"], [list(m) for m in MODELS]), "",
             "### Sound classes", "", *_table(["ID", "shown as", "description"], classes), "",
             "### Splits", "",
             f"Mixtures of {mix.duration_s:g} s rendered from recipes: "
             f"{mix.counts.get('train', '?')} / {mix.counts.get('val', '?')} / "
             f"{mix.counts.get('test', '?')} (train / validation / test); the FSD50K "
             "development set split by uploader so that no recording person appears in two "
             "splits; ESC-50 by fold. The extraction tables use a fixed 500-recipe subset of the "
             "test split (seeded), the same for every system.", ""]
    return lines


def section_detection(r: Results) -> list[str]:
    d = r.detection
    lines = ["## 2. Detection", ""]
    if d is None:
        return [*lines, "*(no detection results: `soundspy evaluate detection`)*", ""]
    rows = []
    classes = list(d["test"]["fine_tuned"]["per_class"])
    for cls in classes:
        row = [f"`{cls}`"]
        for split in ("val", "test"):
            for model in ("zero_shot", "fine_tuned"):
                row.append(_f(d[split][model]["per_class"][cls].get("ap"), "{:.3f}"))
            row.append(_f(d[split]["fine_tuned"]["per_class"][cls].get("f1"), "{:.3f}"))
        rows.append(row)
    mean = ["**mean**"]
    for split in ("val", "test"):
        mean += [_f(d[split]["zero_shot"]["map"], "{:.3f}"),
                 "**" + _f(d[split]["fine_tuned"]["map"], "{:.3f}") + "**",
                 _f(d[split]["fine_tuned"].get("f1"), "{:.3f}")]
    rows.append(mean)
    lines += [
        "EfficientAT mn10_as (AudioSet-pretrained) zero-shot against the model fine-tuned on "
        "Sound Spy's mixtures (1 s windows; AP per class, F1 at the per-class threshold chosen on "
        "validation).", "",
        *_table(["class", "val AP zero-shot", "val AP fine-tuned", "val F1",
                 "test AP zero-shot", "test AP fine-tuned", "test F1"], rows), "",
        "![Detection AP per class](figures/detection_ap.png)", "",
        _reproduce("soundspy evaluate detection --split test"), ""]
    return lines


EXTRACTION_ROWS = (
    # (label, results attribute, system key)
    ("input (nothing done)", "baseline", "mixture"),
    ("baseline, offline detector", "baseline", "timeline"),
    ("baseline, live detector", "baseline", "live"),
    ("baseline, true activity (oracle)", "baseline", "oracle"),
    ("ideal ratio mask (upper bound)", "baseline", "irm"),
    ("Waveformer v1 (Neural)", "v1", "neural_offline"),
    ("Waveformer v1, streaming", "v1", "neural_stream"),
    ("Waveformer v2b + 07b gate (negative result)", "v2", "neural_offline"),
    ("hybrid 07c, offline detector", "hybrid_07c", "hybrid_offline"),
    ("hybrid 07c, live detector", "hybrid_07c", "hybrid_live"),
    ("hybrid 07d/08, offline detector", "hybrid_08", "hybrid_offline"),
    ("hybrid 07d/08, live detector", "hybrid_08", "hybrid_live"),
)


# the same system re-scored on the same recipes by the hybrid evaluations, which also count
# "worse" (the step-06 / 07 metrics files do not)
SAME_SYSTEM = {"mixture": "mixture", "timeline": "timeline", "live": "live", "oracle": "oracle",
               "irm": "irm", "neural_offline": "wf"}


def extraction_rows(r: Results) -> list[tuple[str, dict]]:
    out = []
    latest, _ = r.hybrid_latest()
    for label, attr, system in EXTRACTION_ROWS:
        s = Results.overall(getattr(r, attr), system)
        if s is None:
            continue
        if s.get("worse") is None and attr in ("baseline", "v1") and system in SAME_SYSTEM:
            again = Results.overall(latest, SAME_SYSTEM[system]) or {}
            if again.get("worse") is not None:
                s = {**s, "worse": again["worse"]}
        out.append((label, s))
    return out


def section_extraction(r: Results) -> list[str]:
    lines = ["## 3. Extraction on synthetic test mixtures", ""]
    rows = extraction_rows(r)
    if not rows:
        return [*lines, "*(no extraction results)*", ""]
    att = (r.baseline or {}).get("test", {}).get("attenuation_db", 30)
    table = [[label, _f(s.get("si_sdr_i")), _f(s.get("si_sdr_i_median")),
              _f(s.get("suppression_db"), "{:.1f}"), str(s.get("worse", "–"))] for label, s in rows]
    lines += [
        f"500 test recipes (483 with the target present, 17 without), attenuation {att:g} dB, "
        "48 kHz. SI-SDRi = improvement of the scale-invariant signal-to-distortion ratio over "
        "the input; *target absent* = output level against the input where the selected sound "
        "is not in the mixture (more negative = quieter); *worse* = recipes made more than 5 dB "
        "worse than the input.", "",
        *_table(["system", "SI-SDRi mean", "SI-SDRi median", "target absent (dB)",
                 "worse by > 5 dB"], table), "",
        "![SI-SDRi of every system](figures/extraction_si_sdri.png)", "",
        _reproduce("soundspy evaluate extraction --method mask_baseline | --method neural_tse | "
                   "--method hybrid [--name ext_hybrid_08]"), ""]
    latest, name = r.hybrid_latest()
    per = [("baseline offline", r.baseline, "timeline"), ("Waveformer v1", r.v1, "neural_offline"),
           (f"{name} offline", latest, "hybrid_offline"), (f"{name} live", latest, "hybrid_live")]
    for key, title in (("by_target_class", "By target class"),
                       ("by_target_duration", "By target duration")):
        groups = None
        for _, m, _ in per:
            if m is not None and key in m.get("test", {}):
                groups = list(m["test"][key])
                break
        if not groups:
            continue
        body = []
        for g in groups:
            row = [f"`{g}`" if key == "by_target_class" else g]
            for _, m, system in per:
                try:
                    row.append(_f(m["test"][key][g][system]["si_sdr_i"]))
                except (KeyError, TypeError):
                    row.append("–")
            body.append(row)
        lines += [f"**{title}** (SI-SDRi, dB)", "",
                  *_table([key.replace("by_", "").replace("_", " "),
                           *(label for label, _, _ in per)], body), ""]
    return lines


def section_benchmarks(r: Results) -> list[str]:
    latest, name = r.hybrid_latest()
    systems = [("baseline offline", latest, "timeline"), ("baseline live", latest, "live"),
               ("Waveformer v1", latest, "wf"), (f"{name} offline", latest, "hybrid_offline"),
               (f"{name} live", latest, "hybrid_live")]
    benches = (("absent", "every class selected while absent: target absent (dB)",
                "suppression_db", "{:.1f}"),
               ("pair_absent", "music only, Speech selected (dB)", "suppression_db", "{:.1f}"),
               ("pair_a", "speech + music, Speech selected: SI-SDRi", "si_sdr_i", "{:+.2f}"),
               ("pair_b", "speech + music, Music selected: SI-SDRi", "si_sdr_i", "{:+.2f}"),
               ("librispeech", "LibriSpeech read speech + one other: SI-SDRi", "si_sdr_i",
                "{:+.2f}"))
    rows = []
    for bench, title, metric, fmt in benches:
        row = [title]
        for _, m, system in systems:
            s = Results.bench(m, bench, system)
            row.append(_f(None if s is None else s.get(metric), fmt))
        rows.append(row)
    lines = ["## 4. Benchmarks", "",
             "Fixed benchmark sets of the test split (step 07b): 200 recipes with every class "
             "selected while absent; 100 each of music only, speech + music with Speech or Music "
             "selected; 100 with LibriSpeech speech as the target.", "",
             *_table(["benchmark", *(s[0] for s in systems)], rows), ""]
    c = r.speech_check
    if c is not None:
        att = f"{c['attenuations_db'][0]:g}"
        crow = []
        names = {"dfn": "DeepFilterNet3, no gate", "07c": "07c hybrid gate",
                 "07d": "07d speech-route gate"}
        for key, clocks in c["rows"].items():
            for clock, s in clocks.items():
                a = s[att]
                when = {"centred": "offline"}.get(clock, clock)
                crow.append([names[key] + ("" if key == "dfn" else f" ({when})"),
                             _f(a.get("speech", {}).get("si_sdr_i")),
                             _f(a.get("pair_a", {}).get("si_sdr_i")),
                             _f(a.get("pair_absent", {}).get("suppression_db"), "{:.1f}"),
                             _f(a.get("librispeech", {}).get("si_sdr_i"))])
        lines += ["**The speech route's gate (step 07d)** on the Speech-selected sets "
                  f"({att} dB):", "",
                  *_table(["speech route", "Speech alone", "speech + music", "music only (dB)",
                           "LibriSpeech"], crow), "",
                  _reproduce("soundspy evaluate speech-gate --check"), ""]
    lines += [_reproduce("soundspy evaluate extraction --method hybrid"), ""]
    return lines


def _real_rows(result: dict) -> list[list[str]]:
    s30, s40 = result["summary"]["30"], result["summary"]["40"]
    names = (("baseline live", "live"), ("Waveformer v1", "wf"), ("DeepFilterNet3 alone", "dfn"),
             ("07c gate, live", "hgate_live"), ("07d gate, live", "hybrid_live"),
             ("07d gate, offline", "hybrid_offline"))
    rows = []
    for label, sys_ in names:
        mixes = " / ".join(_f(s30.get(f"mix {r:+g} dB", {}).get(sys_, {}).get("si_sdr_i"))
                           for r in result["ratios_db"])
        music = s40.get("music_only", {}).get(sys_, {}).get("suppression_db")
        speech = s40.get("speech_only", {}).get(sys_, {})
        rows.append([label, mixes, _f(music, "{:.1f}"),
                     _f(speech.get("interruption") and 100 * speech["interruption"],
                        "{:.1f} %") if speech.get("interruption") is not None else "–"])
    return rows


def _pause_rows(result: dict) -> list[list[str]]:
    s = result["summary"]["40"]
    rows = []
    for label, sys_ in (("DeepFilterNet3 alone", "dfn"), ("07c gate, live", "hgate_live"),
                        ("07d gate, live", "hybrid_live"), ("07d gate, offline", "hybrid_offline")):
        row = [label]
        for group in ("mixes music", "mixes music_pop"):
            for g in ("short", "long"):
                row.append(_f(s.get(group, {}).get(sys_, {}).get(f"pause_{g}_db"), "{:.1f}"))
        rows.append(row)
    return rows


def section_real_room(r: Results) -> list[str]:
    lines = ["## 5. Real room", "",
             "The laptop microphone in the owner's room: speech, music and room tone recorded "
             "separately and mixed digitally, so the correct output (the recorded speech) is known "
             "exactly. Speech selected. **Tuning** data chose the settings; the **held-out** "
             "session was scored once, with the final settings.", ""]
    if not r.real_07d:
        return [*lines, "*(no real-room results: `soundspy evaluate extraction --method hybrid "
                        "--benchmark real`)*", ""]
    for key, result in r.real_07d.items():
        role = result.get("role") or "?"
        lines += [f"### {role.capitalize()} session `{key}`", "",
                  "Mixtures: SI-SDRi at the speech-to-music ratios "
                  f"{' / '.join(f'{x:+g}' for x in result['ratios_db'])} dB (30 dB); music only: "
                  "suppression; speech only: share of speech frames cut (40 dB).", "",
                  *_table(["system", "mixtures", "music only (dB)", "speech cut"],
                          _real_rows(result)), "",
                  "**Music in the pauses** (40 dB, output against input, pooled over the "
                  "pauses; 0.5-2 s *short*, >= 2 s *long*):", "",
                  *_table(["system", "classical short", "classical long", "song short",
                           "song long"], _pause_rows(result)), ""]
    lines += ["![Music in the pauses](figures/pause_suppression.png)", "",
              "The song has vocals: its singer is a voice to the speech enhancer, so it stays in "
              "the pauses for every system (a limitation, section 9).", "",
              _reproduce("soundspy evaluate extraction --method hybrid --benchmark real "
                         "--session <id>[+<id>] --name ext_hybrid_07d --role held-out"), ""]
    return lines


def latency_breakdown(r: Results) -> list[dict]:
    """Per route: the parts of the estimate, averaged over the latest soak run with the
    configured settings (no ``--set``), and the measured value."""
    out = []
    measured = (r.latency or {}).get("modes", {})
    for route, mode, title in (("speech_enhancer", "speech", "Hybrid, speech route"),
                               ("waveformer", "waveformer", "Hybrid, Waveformer route")):
        run = r.final_soak.get(route)
        parts = (run or {}).get("summary", {}).get("latency_parts_mean")
        m = measured.get(mode, {})
        out.append({"title": title, "parts": parts, "measured": m.get("median_ms")})
    return out


def section_realtime(r: Results) -> list[str]:
    lines = ["## 6. Real time", ""]
    if r.screening:
        rows = []
        for key, run in r.screening.items():
            s = run.get("summary") or {}
            change, _, route = key.partition(" | ")
            rows.append([change, route or "–", str(s.get("underruns", "–")),
                         _f(s.get("rtf_p95_max"), "{:.2f}"),
                         _f(s.get("latency_ms_median"), "{:.0f}"),
                         _f(s.get("extractor_ms"), "{:.1f}")])
        lines += ["**Delay changes of step 08** (each run a few minutes of the live chain, Listen "
                  "on but muted; *delay* = the session's own estimate):", "",
                  *_table(["change (cumulative)", "selection", "underruns",
                           "RTF p95 (worst)", "delay (ms)", "extractor (ms)"], rows), "",
                  _reproduce("soundspy soak --minutes 3 --labels speech --set <key>=<value>"),
                  ""]
    if r.latency:
        rows = [[m["title"], _f(m.get("median_ms"), "{:.1f}"), _f(m.get("min_ms"), "{:.1f}"),
                 _f(m.get("max_ms"), "{:.1f}"), f"{m.get('found', 0)}",
                 _f(m.get("estimate_ms"), "{:.0f}")] for m in r.latency["modes"].values()]
        lines += [f"**Measured delay** (acoustic loopback, {r.latency.get('time')}: laptop "
                  "speakers to microphone; app rows = microphone to speaker through Sound Spy, "
                  "the hardware round trip removed):", "",
                  *_table(["path", "median (ms)", "min", "max", "clicks", "estimate (ms)"],
                          rows), "", _reproduce("soundspy measure-latency --all"), ""]
    else:
        lines += ["*(not measured yet: `soundspy measure-latency --all`)*", ""]
    lines += ["![The delay of each route in its parts](figures/latency_breakdown.png)", "",
              "*The parts are the session's own figures, averaged over the latest soak of each "
              "route with the configured settings.*", ""]
    if r.soaks:
        rows = []
        for d in r.soaks:
            s = d["summary"]
            rows.append([str(d.get("time", ""))[:16].replace("T", " "), str(d.get("route")),
                         f"{d['minutes']:g}", str(d.get("thread_priority") or "normal"),
                         str(s["underruns"]), f"{s['input_dropped']} / {s['output_dropped']}",
                         _f(s.get("rtf_p95_max"), "{:.2f}"),
                         _f(s.get("latency_ms_median"), "{:.1f}"),
                         f"{s.get('ram_mb_first')} -> {s.get('ram_mb_last')}",
                         f"{s.get('vram_mb_first')} -> {s.get('vram_mb_last')}"])
        lines += ["**Long runs** (live microphone, Listen on but muted, the configured settings; "
                  "*priority* = the audio thread's: `mmcss` = Windows' audio scheduler, added "
                  "after the first two runs):", "",
                  *_table(["started", "route", "min", "priority", "underruns",
                           "dropped (in / out)", "RTF p95 (worst)", "delay (ms)", "RAM (MB)",
                           "VRAM (MB)"], rows), "",
                  _reproduce("soundspy soak --minutes 30 --labels speech   # and --labels music"),
                  ""]
    if r.soak:
        s = r.soak["summary"]
        lines += [f"**Soak** ({r.soak['minutes']:g} min, {r.soak.get('method')} "
                  f"{r.soak.get('route')}, {r.soak.get('attenuation_db'):g} dB): "
                  f"{s['underruns']} underruns, {s['input_dropped']} / {s['output_dropped']} "
                  f"chunks dropped (input / output), RTF p95 median {s['rtf_p95_median']} "
                  f"(worst {s['rtf_p95_max']}), RAM {s['ram_mb_first']} -> {s['ram_mb_last']} "
                  f"MB, graphics card {s['vram_mb_first']} -> {s['vram_mb_last']} MB.", ""]
        rows = [[f"{x['t_s'] / 60:.0f}", str(x["underruns"]), _f(x["rtf_p95"], "{:.2f}"),
                 _f(x["ram_mb"], "{:.0f}"), _f(x["vram_reserved_mb"], "{:.0f}")]
                for x in r.soak["samples"][1:]]
        lines += ["Minute by minute (the latest 30-minute run):", "",
                  *_table(["minute", "underruns", "RTF p95", "RAM (MB)", "VRAM (MB)"], rows), "",
                  _reproduce("soundspy soak --minutes 30"), ""]
    if r.startup:
        s = r.startup
        lines += [f"**Start-up**: server up in {s.get('server_up_s')} s, models ready "
                  f"{s.get('models_ready_s')} s after start ({s.get('ram_mb')} MB RAM, "
                  f"{s.get('vram_mb')} MB on the graphics card); first session: "
                  f"{s.get('first_session_s')} s from Start to audio, "
                  f"{s.get('first_session_underruns')} underruns.", ""]
    return lines


def section_negative(r: Results) -> list[str]:
    v2 = Results.overall(r.v2, "neural_offline")
    v2_nogate = Results.overall(r.v2, "neural_nogate")
    v1 = Results.overall(r.v1, "neural_offline")
    g = Results.overall(r.v1_gate, "neural_offline")
    lines = ["## 7. Negative results", "",
             "- **Waveformer v2 / v2b (step 07b)**: retrained with a loss that punishes leakage in "
             "silent 50 ms frames, dynamic mixing and LibriSpeech. Less leakage, lower quality: "
             f"SI-SDRi {_f(v2_nogate and v2_nogate.get('si_sdr_i'))} without and "
             f"{_f(v2 and v2.get('si_sdr_i'))} dB with its output gate, against v1's "
             f"{_f(v1 and v1.get('si_sdr_i'))} dB; most of the step's targets were missed and v1 "
             "stays the configured model (v2b kept on disk).",
             "- **The output gate for the Waveformer (step 07b)**: a gate that listens only to "
             "the network's own output cannot see the network's confident mistakes (music it "
             "takes for the target leaks at −12 to −20 dB and holds the gate open) and it ducked "
             f"quiet syllables. With v1: SI-SDRi {_f(g and g.get('si_sdr_i'))} dB, target "
             f"absent {_f(g and g.get('suppression_db'), '{:.1f}')} dB; off by default. On the "
             "speech route the same idea works (step 07d) because the speech enhancer's output "
             "is 20 dB further from music than the Waveformer's.",
             "- **Cascade (step 07c)**: Waveformer then DeepFilterNet3 for Speech was worse than "
             "DeepFilterNet3 alone on every speech benchmark except silence.",
             "- **Detector hop 0.5 s (step 07c)**: halved the detector's CPU time, the RTF did "
             "not change; reverted.", ""]
    return lines


LIMITATIONS = (
    "**Several talkers**: the speech enhancer keeps every voice; telling one person from another "
    "is the job of the Spatial and Sources tabs (microphone array, modules 3 and 5).",
    "**Singing** counts as a voice: a song with vocals stays audible in the pauses (about "
    "−15 dB in the real room).",
    "**Very quiet input**: the laptop microphone peaks around −29 dBFS; the speech enhancer "
    "needed a +10 dB input gain not to remove the first seconds of speech (step 07c).",
    "**The laptop microphone** is one fixed device with Windows' processing switched off; other "
    "microphones need their own device profile and a new real-room session.",
    "**First start**: the server loads and warms its models in the background (section 6); a "
    "session started meanwhile waits in the *warming* state.",
    "**Music the speech enhancer takes for a voice** reopens the speech route's gate (some "
    "classical passages, singing).",
    "**Synthetic mixtures** do not predict the real room well (step 07b); the real-room tables "
    "are the closer measure.",
)


def _delay_limitation(r: Results) -> str:
    modes = (r.latency or {}).get("modes", {})
    hw = modes.get("hardware", {}).get("median_ms")
    speech = modes.get("speech", {})
    if hw is None or speech.get("median_ms") is None:
        return ("**Delay**: the speech route needs DeepFilterNet3's 40 ms (two frames of "
                "look-ahead plus its window) on top of the sound devices' own delay, which "
                "PortAudio reports as 22 ms on each side (section 6; not measured yet).")
    extra = speech["median_ms"] - (speech.get("estimate_ms") or 0.0)
    return (f"**Delay**: measured through the laptop speakers, the speech route takes "
            f"{speech['median_ms']:.0f} ms from microphone to speaker, {extra:.0f} ms more than "
            f"the session's own estimate: the sound devices' round trip is {hw:.0f} ms where "
            "PortAudio reports 44 ms (section 6). DeepFilterNet3's 40 ms are part of the model; "
            "headphones may have a different device delay.")


def section_limitations(r: Results | None = None) -> list[str]:
    items = list(LIMITATIONS)
    if r is not None:
        items.insert(5, _delay_limitation(r))
    return ["## 9. Limitations", "", *(f"- {x}" for x in items), ""]


# --- figures ----------------------------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _hbars(ax, labels, series: list[tuple[str, list[float | None], str]], fmt: str) -> None:
    """Grouped horizontal bars with a value label on every bar (2 px-like gaps)."""
    import numpy as np

    n = len(series)
    y = np.arange(len(labels))
    height = 0.8 / n
    for k, (name, values, colour) in enumerate(series):
        vals = [np.nan if v is None else v for v in values]
        pos = y - 0.4 + height * (k + 0.5)
        ax.barh(pos, vals, height=height * 0.92, color=colour, label=name, edgecolor=SURFACE,
                linewidth=1.0)
        for p, v in zip(pos, vals, strict=True):
            if v == v:  # not NaN
                ax.annotate(fmt.format(v).replace("-", "−"), (v, p),
                            xytext=(3 if v >= 0 else -3, 0), textcoords="offset points",
                            ha="left" if v >= 0 else "right", va="center", fontsize=8,
                            color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, linewidth=0.8)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")


def figures(r: Results, folder: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
                         "text.color": INK, "axes.labelcolor": INK2})
    made = []
    if r.detection is not None:
        d = r.detection["test"]
        classes = list(d["fine_tuned"]["per_class"])
        fig, ax = plt.subplots(figsize=(7, 3.6))
        _style(ax)
        _hbars(ax, classes, [
            ("fine-tuned", [d["fine_tuned"]["per_class"][c]["ap"] for c in classes], SERIES[0]),
            ("zero-shot", [d["zero_shot"]["per_class"][c]["ap"] for c in classes], SERIES[1])],
            "{:.2f}")
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("average precision (test)")
        ax.set_title("Detection: AP per class", loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        path = folder / "detection_ap.png"
        _save(fig, path)
        plt.close(fig)
        made.append(path)
    rows = [(label, s) for label, s in extraction_rows(r) if not label.startswith("input")]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 0.36 * len(rows) + 1.0))
        _style(ax)
        _hbars(ax, [label for label, _ in rows],
               [("SI-SDRi", [s.get("si_sdr_i") for _, s in rows], SERIES[0])], "{:+.1f}")
        values = [v for _, s in rows if (v := s.get("si_sdr_i")) is not None]
        if values:  # room for the value labels on both sides
            ax.set_xlim(min(0.0, min(values)) * 1.35 - 0.5, max(values) * 1.12)
        ax.set_xlabel("SI-SDRi, dB (500 test recipes, 30 dB)")
        ax.set_title("Extraction: improvement over the input", loc="left", fontsize=11,
                     color=INK)
        path = folder / "extraction_si_sdri.png"
        _save(fig, path)
        plt.close(fig)
        made.append(path)
    held = next((v for v in r.real_07d.values() if v.get("role") == "held-out"), None)
    if held is not None:
        s = held["summary"]["40"].get("mixes music", {})
        systems = (("DeepFilterNet3 alone", "dfn"), ("07c gate, live", "hgate_live"),
                   ("07d gate, live", "hybrid_live"), ("07d gate, offline", "hybrid_offline"))
        fig, ax = plt.subplots(figsize=(7, 3.4))
        _style(ax)
        _hbars(ax, [a for a, _ in systems], [
            ("short pauses (0.5-2 s)", [s.get(b, {}).get("pause_short_db") for _, b in systems],
             SERIES[0]),
            ("long pauses (>= 2 s)", [s.get(b, {}).get("pause_long_db") for _, b in systems],
             SERIES[1])], "{:.1f}")
        ax.set_xlabel("music in the pauses, output vs input (dB; lower = quieter)")
        ax.set_title("Held-out real room, classical music, Speech at 40 dB", loc="left",
                     fontsize=11, color=INK)
        lows = [v for _, b in systems for k in ("pause_short_db", "pause_long_db")
                if (v := s.get(b, {}).get(k)) is not None]
        if lows:
            ax.set_xlim(min(lows) * 1.14, 0)  # room for the value labels left of the bars
        ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22))
        path = folder / "pause_suppression.png"
        _save(fig, path)
        plt.close(fig)
        made.append(path)
    breakdown = [b for b in latency_breakdown(r) if b["parts"]]
    if breakdown:
        names = (("input_ms", "input stream"), ("chunk_ms", "chunk"), ("chain_ms", "extractor"),
                 ("queue_ms", "output queue"), ("output_ms", "output stream"))
        fig, ax = plt.subplots(figsize=(7, 0.7 * len(breakdown) + 1.3))
        _style(ax)
        for i, b in enumerate(breakdown):
            left = 0.0
            for (key, label), colour in zip(names, SERIES, strict=True):
                width = b["parts"].get(key, 0.0)
                ax.barh(i, width, left=left, height=0.5, color=colour, edgecolor=SURFACE,
                        linewidth=2.0, label=label if i == 0 else None)
                if width >= 6:
                    ax.text(left + width / 2, i, f"{width:.0f}", ha="center", va="center",
                            fontsize=8, color=INK if colour == SERIES[3] else "#ffffff")
                left += width
            text = f"estimate {left:.0f} ms"
            if b["measured"] and b["measured"] > left:
                # what the measurement found beyond the session's own figures
                hidden = b["measured"] - left
                ax.barh(i, hidden, left=left, height=0.5, color=UNREPORTED, edgecolor=SURFACE,
                        linewidth=2.0, label="not reported by the sound devices" if i == 0
                        else None)
                ax.text(left + hidden / 2, i, f"{hidden:.0f}", ha="center", va="center",
                        fontsize=8, color=INK)
                text = f"measured {b['measured']:.0f} ms (estimate {left:.0f})"
                left = b["measured"]
            ax.annotate(text, (left, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(breakdown)))
        ax.set_yticklabels([b["title"] for b in breakdown])
        ax.invert_yaxis()
        ax.set_xlabel("milliseconds, microphone to output (measured through the laptop speakers)")
        ax.set_title("Delay of each route in its parts", loc="left", fontsize=11, color=INK)
        ax.set_xlim(0, max(b["measured"] or 0 for b in breakdown) * 1.35 or None)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.32))
        path = folder / "latency_breakdown.png"
        _save(fig, path)
        plt.close(fig)
        made.append(path)
    return made


# --- the document -----------------------------------------------------------------------------


def rerun(config: AppConfig, progress: Callable[[str], None] = print) -> None:
    """The long evaluations (about two hours on the laptop): the synthetic test set with the
    current hybrid (``ext_hybrid_08``) and the speech-route check. Not the real-room sessions
    (they are scored where they were recorded) and not the detection (stable since step 05)."""
    import subprocess

    exe = [sys.executable, "-m", "soundspy.cli.main"]
    for args in (["evaluate", "extraction", "--method", "hybrid", "--name", "ext_hybrid_08"],
                 ["evaluate", "speech-gate", "--check"]):
        progress("running " + " ".join(["soundspy", *args]))
        subprocess.run([*exe, *args], check=True)


def build(config: AppConfig, *, out: Path | None = None, with_figures: bool = True
          ) -> dict[str, Any]:
    root = Path(config.project_root)
    docs = out or root / "docs"
    r = Results(config)
    made = figures(r, docs / "figures") if with_figures else []
    lines = ["# Sound Spy: evaluation", "",
             f"*Generated by `soundspy report` on {datetime.now():%Y-%m-%d %H:%M} from the stored "
             "result files; nothing here is typed in by hand. Regenerate after new results; "
             "`soundspy report --rerun` re-runs the long evaluations first.*", "",
             "Sound Spy listens through the laptop microphone and lets the user choose what to "
             "hear: the chosen sound types are kept, everything else is turned down (Tab 1, "
             "*Semantic*). This document collects the measurements behind it.", "",
             *section_setup(config), *section_detection(r), *section_extraction(r),
             *section_benchmarks(r), *section_real_room(r), *section_realtime(r),
             *section_negative(r),
             "## 8. Reproducing", "",
             "Every table above names its command; the results live in `experiments/` "
             "(`metrics.json`, `real_room.json`, `experiments/realtime/`), listening examples in "
             "`outputs/eval/`.", "",
             *section_limitations(r)]
    path = docs / "evaluation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": path, "figures": made,
            "missing": [n for n in ("detection", "baseline", "v1", "hybrid_07c", "latency",
                                    "soak") if getattr(r, n) is None]}

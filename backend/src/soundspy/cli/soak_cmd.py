"""``soundspy soak``: a long live session, measured every minute (step 08).

    soundspy soak [--minutes 30] [--every 60] [--method hybrid] [--labels speech]
                  [--attenuation 40] [--source data/raw/x.wav] [--set output.prebuffer_chunks=1]

Runs Sound Spy's real chain in this process (microphone -> chain -> speaker, **Listen on but
muted**: the speaker stream runs, so underruns are real, but nothing is heard) and records, every
``--every`` seconds: output underruns, dropped chunks (input and output), queue drains, input
overflows, the p95 real-time factor and processing time over the last window, the session's
delay estimate, and the process's RAM and graphics-card memory. Results:
``experiments/realtime/soak_<time>.json``; the table is printed at the end. ``--set`` overrides
a setting for the run (the latency experiments of step 08: chunk size, prebuffer, stream
latency, exclusive mode).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from soundspy.cli.common import USER_ERRORS, add_profile_arguments, fail

NAME = "soak"
HELP = "run a long muted live session and record underruns, RTF, delay, RAM, VRAM"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    add_profile_arguments(parser)
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--every", type=float, default=60.0, help="seconds between samples")
    parser.add_argument("--method", default=None, help="extraction method (default: config)")
    parser.add_argument("--labels", nargs="*", default=["speech"])
    parser.add_argument("--attenuation", type=float, default=40.0)
    parser.add_argument("--source", default=None, help="a WAV played in a loop instead of the mic")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--name", default=None, help="label stored with the result")
    parser.set_defaults(handler=run)


def sample(session, started: float) -> dict[str, Any]:
    from soundspy.engine import resources

    s = session.status
    return {"t_s": round(time.monotonic() - started, 1), "state": s.state,
            "underruns": s.output_underruns, "input_dropped": s.dropped_chunks,
            "output_dropped": s.output_dropped, "drained": s.output_drained,
            "overflows": s.input_overflows, "rtf_p95": s.rtf_p95, "proc_ms_p95": s.proc_ms_p95,
            "latency_ms": s.latency_ms, "chunks": s.chunks_processed,
            "latency_parts": session.latency_parts, **resources.snapshot()}


def soak(config, *, minutes: float, every: float, method: str | None, labels, attenuation: float,
         source: str | None = None, progress=print) -> dict[str, Any]:
    from soundspy.engine.live import live_session

    session = live_session(config, method=method, labels=labels, attenuation_db=attenuation,
                           monitor=True, muted=True, source=source)
    requested = time.monotonic()
    session.start()
    if not session.wait_until_running(120.0):
        message = session.status.message
        session.stop()
        raise RuntimeError(f"the session did not start: {message}")
    started = time.monotonic()
    samples = [sample(session, started)]
    warm = session.status.warmup_s
    try:
        end = started + minutes * 60.0
        while time.monotonic() < end:
            time.sleep(max(0.0, min(every, end - time.monotonic())))
            row = sample(session, started)
            samples.append(row)
            progress(f"  {row['t_s']:7.0f} s  underruns {row['underruns']}  dropped "
                     f"{row['input_dropped']}/{row['output_dropped']}  rtf p95 {row['rtf_p95']}  "
                     f"delay {row['latency_ms']} ms  RAM {row['ram_mb']} MB  VRAM "
                     f"{row['vram_reserved_mb']} MB")
            if row["state"] != "running":
                break
        status = session.status
    finally:
        session.stop()
    extraction = status.extraction or {}
    rtf = [r["rtf_p95"] for r in samples[1:] if r["rtf_p95"] is not None]
    ram = [r["ram_mb"] for r in samples if r["ram_mb"] is not None]
    vram = [r["vram_reserved_mb"] for r in samples if r["vram_reserved_mb"] is not None]
    lat = [r["latency_ms"] for r in samples[1:] if r["latency_ms"] is not None]
    parts = [r["latency_parts"] for r in samples[1:] if r.get("latency_parts")]
    mean_parts = ({k: round(sum(p[k] for p in parts) / len(parts), 2) for k in parts[0]}
                  if parts else None)
    return {
        "time": datetime.now().isoformat(timespec="seconds"), "minutes": minutes,
        "method": extraction.get("method"), "route": extraction.get("route"),
        "labels": list(labels), "attenuation_db": attenuation, "source": source or "mic",
        "warmup_s": warm, "start_to_running_s": round(started - requested, 2),
        "thread_priority": session.priority,
        "settings": {"chunk_ms": config.engine.chunk_ms,
                     "prebuffer_chunks": config.output.prebuffer_chunks,
                     "capture_latency": config.capture.latency,
                     "output_latency": config.output.latency,
                     "exclusive": [config.capture.exclusive, config.output.exclusive],
                     "fixed_latency": config.extraction.hybrid.fixed_latency},
        "summary": {
            "state_end": status.state, "message_end": status.message,
            "underruns": status.output_underruns, "input_dropped": status.dropped_chunks,
            "output_dropped": status.output_dropped, "drained": status.output_drained,
            "overflows": status.input_overflows,
            "rtf_p95_max": max(rtf) if rtf else None,
            "rtf_p95_median": round(sorted(rtf)[len(rtf) // 2], 3) if rtf else None,
            "latency_ms_median": round(sorted(lat)[len(lat) // 2], 1) if lat else None,
            "latency_ms_max": max(lat) if lat else None,
            "ram_mb_first": ram[0] if ram else None, "ram_mb_last": ram[-1] if ram else None,
            "vram_mb_first": vram[0] if vram else None,
            "vram_mb_last": vram[-1] if vram else None,
            "extractor_ms": extraction.get("latency_ms"),
            "latency_parts_mean": mean_parts,
            "model_ms_p95": extraction.get("model_ms_p95")},
        "samples": samples}


def run(args: argparse.Namespace) -> int:
    from soundspy.cli.measure_latency_cmd import _config
    from soundspy.config import ConfigError

    args.output_device = None
    try:
        config = _config(args)
        print(f"soak: {args.minutes:g} min, {args.method or config.extraction.method}, "
              f"{args.labels} at {args.attenuation:g} dB, settings {args.set or 'as configured'}")
        result = soak(config, minutes=args.minutes, every=args.every, method=args.method,
                      labels=args.labels, attenuation=args.attenuation, source=args.source)
    except (*USER_ERRORS, ConfigError, RuntimeError, ValueError) as exc:
        return fail(exc)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    result["name"] = args.name
    result["overrides"] = args.set
    folder = Path(config.paths["experiments"]) / "realtime"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"soak_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    print(f"-> {path}")
    return 0

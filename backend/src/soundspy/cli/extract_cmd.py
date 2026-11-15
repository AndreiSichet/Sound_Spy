"""``soundspy extract``: keep the selected sound classes of a recording, attenuate the rest.

    soundspy extract <wav> --labels speech [music ...] [--attenuation-db 30] [--out out.wav]
                     [--method mask_baseline] [--device auto|cuda|cpu]

Offline extraction (step 06): the detector first runs over the whole file with windows centred on
each moment (no lag, unlike live), then the extractor uses that timeline. The result is a
**processing run** like the ones the website makes: ``outputs/processed/<run_id>.wav`` plus its
``.json`` sidecar (the API indexes it the next time it looks). ``--out`` additionally copies the
processed audio to a path of your choice.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

NAME = "extract"
HELP = "extract sound classes from a recording (the rest is attenuated)"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("wav", type=Path, help="audio file (any rate / channels)")
    parser.add_argument("--labels", nargs="+", required=True, metavar="CLASS",
                        help="class IDs to keep (see configs/classes.yaml), e.g. speech music")
    parser.add_argument("--attenuation-db", type=float, default=30.0,
                        help="how much everything else is turned down (default 30)")
    parser.add_argument("--out", type=Path, default=None,
                        help="also copy the processed WAV here")
    parser.add_argument("--method", choices=("mask_baseline",), default=None,
                        help="extraction method (default: extraction.method)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None,
                        help="where the detector runs (default: detection.live.device)")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from soundspy.config import ConfigError, load_config
    from soundspy.extraction.offline import execute_run, runs_dir, start_run
    from soundspy.io.runs import audio_path

    if not args.wav.is_file():
        print(f"error: {args.wav} not found")
        return 1
    try:
        config = load_config()
        run = start_run(config, args.wav, args.labels, attenuation_db=args.attenuation_db,
                        method=args.method)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"run {run['run_id']}: {', '.join(run['labels'])} from {args.wav.name} "
          f"(everything else -{run['attenuation_db']:g} dB, {run['method']})", flush=True)
    done = execute_run(config, run["run_id"], device=args.device)
    if done["status"] != "done":
        print(f"error: {done['error']}")
        return 1
    wav = audio_path(runs_dir(config), run["run_id"])
    for label, segments in done["info"].get("target_segments", {}).items():
        spans = ", ".join(f"{a:.2f}-{b:.2f} s" for a, b in segments) or "never detected"
        print(f"  {label}: {spans}")
    print(f"{done['duration_s']:.1f} s processed in {done['processing_seconds']:.1f} s")
    print(f"-> {wav}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wav, args.out)
        print(f"-> {args.out}")
    return 0

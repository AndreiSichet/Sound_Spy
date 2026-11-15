"""``soundspy detect``: which sound classes are in a recording, and when.

    soundspy detect <wav> [--experiment det_mn10_v1] [--json out.json] [--device auto|cuda|cpu]

Runs the trained detector over the file with the live settings (1 s windows every 0.25 s,
smoothing and hysteresis; ``configs/detection/default.yaml``) and prints, per class, the time
segments where it is active with their mean probability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NAME = "detect"
HELP = "detect sound classes in a recording over time"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("wav", type=Path, help="audio file (any rate / channels)")
    parser.add_argument("--experiment", default=None,
                        help="use experiments/<name>/checkpoints/best.pt (default: config)")
    parser.add_argument("--json", type=Path, default=None, help="also write the timeline here")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    parser.set_defaults(handler=run)


def format_timeline(timeline) -> str:
    lines = [f"{timeline.duration_s:.1f} s, {len(timeline.times)} windows "
             f"({timeline.window_s:g} s every {timeline.hop_s:g} s)"]
    by_label: dict[str, list] = {lab: [] for lab in timeline.labels}
    for seg in timeline.segments:
        by_label[seg.label].append(seg)
    width = max(len(lab) for lab in timeline.labels)
    for lab in timeline.labels:
        segs = by_label[lab]
        if not segs:
            lines.append(f"{lab:<{width}}  -")
            continue
        parts = [f"{s.t_start:.2f}-{s.t_end:.2f} s (p {s.probability:.2f})" for s in segs]
        lines.append(f"{lab:<{width}}  " + ", ".join(parts))
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    from soundspy.config import ConfigError, load_config
    from soundspy.detection.inference import DetectorUnavailable, detect_file

    if not args.wav.is_file():
        print(f"error: {args.wav} not found")
        return 1
    try:
        config = load_config()
        timeline = detect_file(config, args.wav, experiment=args.experiment, device=args.device)
    except (ConfigError, DetectorUnavailable, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(format_timeline(timeline))
    if args.json is not None:
        args.json.write_text(json.dumps(timeline.to_json(), indent=2) + "\n", encoding="utf-8")
        print(f"-> {args.json}")
    return 0

"""``soundspy record``: record a clip from the profile's microphone into ``data/raw``.

Saves a 32-bit float WAV at the native sample rate with every kept channel, plus a JSON sidecar
(``soundspy.io.storage``). Only files are written: the app database picks the recording up the
next time the API lists recordings. This command never imports the API layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soundspy.cli.common import (
    USER_ERRORS,
    add_profile_arguments,
    countdown,
    fail,
    load_profile,
    print_device_line,
    print_level_report,
)
from soundspy.io.devices import resolve_input
from soundspy.io.mic_source import MicSource
from soundspy.io.storage import save_recording

NAME = "record"
HELP = "record a clip to data/raw (WAV + JSON sidecar)"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    add_profile_arguments(parser)
    parser.add_argument("--seconds", type=float, required=True, help="recording length")
    parser.add_argument("--label", default=None, help="label, used in the file name and sidecar")
    parser.add_argument("--notes", default="", help="free-text notes stored in the sidecar")
    parser.add_argument("--out", type=Path, default=None, help="directory (default: data/raw)")
    parser.add_argument(
        "--countdown", type=float, default=2.0, help="seconds before recording (default 2)"
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        config, profile = load_profile(args)
        resolved = resolve_input(profile)
        print_device_line(resolved)
        with MicSource(resolved, config.capture) as mic:
            countdown(args.countdown, f"Recording {args.seconds:g} s")
            buf = mic.read(args.seconds)
        print("done.")
    except USER_ERRORS as exc:
        return fail(exc)

    out_dir = args.out if args.out is not None else config.paths["raw"]
    wav_path = save_recording(
        buf, out_dir, label=args.label, notes=args.notes, profile=resolved.profile_name
    )
    print_level_report(buf)
    print()
    print(f"Saved: {wav_path}")
    print(f"       {wav_path.with_suffix('.json')}")
    return 0

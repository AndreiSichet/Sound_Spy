"""``soundfocus record``: record an ad-hoc clip to data/raw.

Records from the active profile at native sample rate with all channels and saves
WAV + JSON sidecar to ``data/raw``.

TODO(step-01): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundfocus.cli import not_implemented

NAME = "record"
HELP = "record an ad-hoc clip to data/raw"
STEP = "step-01"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

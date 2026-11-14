"""``soundspy record``: record a clip to data/raw.

Records at native sample rate with all channels; saves WAV + JSON sidecar.

TODO(step-02): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundspy.cli import not_implemented

NAME = "record"
HELP = "record a clip to data/raw"
STEP = "step-02"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

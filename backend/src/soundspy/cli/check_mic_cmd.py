"""``soundspy check-mic``: record a short test clip and report levels per channel.

Prints RMS/peak dBFS, clipping and channel similarity per channel.

TODO(step-02): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundspy.cli import not_implemented

NAME = "check-mic"
HELP = "record a short test clip and report levels per channel"
STEP = "step-02"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

"""``soundfocus check-mic``: record a short test clip and report levels per channel.

Captures a few seconds from the active profile and prints RMS/peak dBFS, clipping
and channel similarity per channel.

TODO(step-01): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundfocus.cli import not_implemented

NAME = "check-mic"
HELP = "record a short test clip and report levels per channel"
STEP = "step-01"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

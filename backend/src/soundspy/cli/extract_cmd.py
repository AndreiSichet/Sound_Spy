"""``soundspy extract``: extract target classes from a recording.

Keeps the target classes, attenuates the rest; writes the result to ``outputs/``.

TODO(step-06): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundspy.cli import not_implemented

NAME = "extract"
HELP = "extract target classes from a recording"
STEP = "step-06"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

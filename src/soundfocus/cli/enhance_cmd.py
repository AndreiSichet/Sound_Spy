"""``soundfocus enhance``: enhance a target class in a recording.

Enhances the target class and suppresses the rest; writes the result to ``outputs/``.

TODO(step-07): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundfocus.cli import not_implemented

NAME = "enhance"
HELP = "enhance a target class in a recording"
STEP = "step-07"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

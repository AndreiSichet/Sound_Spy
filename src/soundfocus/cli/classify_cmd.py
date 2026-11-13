"""``soundfocus classify``: classify a recording over time.

Runs sliding-window inference over a WAV file and prints/saves the class timeline.

TODO(step-06): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundfocus.cli import not_implemented

NAME = "classify"
HELP = "classify a recording over time"
STEP = "step-06"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

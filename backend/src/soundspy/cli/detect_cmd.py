"""``soundspy detect``: detect sound classes in a recording over time.

Chunked inference over a WAV file; prints/saves the detections.

TODO(step-05): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundspy.cli import not_implemented

NAME = "detect"
HELP = "detect sound classes in a recording over time"
STEP = "step-05"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

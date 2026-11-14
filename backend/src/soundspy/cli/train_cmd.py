"""``soundspy train``: train a detection or extraction model.

Runs an experiment from ``experiments/<name>/config.json``.

TODO(step-05): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundspy.cli import not_implemented

NAME = "train"
HELP = "train a detection or extraction model"
STEP = "step-05"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

"""``soundspy data``: download datasets and build mixtures.

``soundspy data download`` fetches and verifies datasets; ``soundspy data prepare`` runs
the collectors, splits and mixture generation.

TODO(step-04): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundspy.cli import not_implemented

NAME = "data"
HELP = "download datasets and build mixtures"
STEP = "step-04"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

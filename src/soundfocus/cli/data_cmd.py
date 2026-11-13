"""``soundfocus data``: download and prepare public datasets.

``soundfocus data download`` fetches and verifies public datasets;
``soundfocus data prepare`` builds label mappings and splits.

TODO(step-03): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundfocus.cli import not_implemented

NAME = "data"
HELP = "download and prepare public datasets"
STEP = "step-03"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

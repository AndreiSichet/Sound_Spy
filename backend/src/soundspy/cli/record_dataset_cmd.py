"""``soundspy record-dataset``: record labelled clips into data/custom.

Guided recording of labelled clips for the custom dataset collector.

TODO(step-04): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundspy.cli import not_implemented

NAME = "record-dataset"
HELP = "record labelled clips into data/custom"
STEP = "step-04"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

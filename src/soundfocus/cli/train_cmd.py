"""``soundfocus train``: fine-tune the sound classifier.

Fine-tunes the pretrained backbone on the project label set and writes checkpoints
to ``models/``.

TODO(step-05): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundfocus.cli import not_implemented

NAME = "train"
HELP = "fine-tune the sound classifier"
STEP = "step-05"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

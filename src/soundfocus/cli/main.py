"""Entry point of the ``soundfocus`` command: argparse dispatcher over the subcommands.

Each subcommand module registers its own parser and sets ``handler`` via
``set_defaults``. Arguments are added in the step that implements each subcommand.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from soundfocus import __version__
from soundfocus.cli import (
    check_mic_cmd,
    classify_cmd,
    data_cmd,
    devices_cmd,
    enhance_cmd,
    record_cmd,
    record_dataset_cmd,
    train_cmd,
)

COMMAND_MODULES = (
    devices_cmd,
    check_mic_cmd,
    record_cmd,
    record_dataset_cmd,
    data_cmd,
    train_cmd,
    classify_cmd,
    enhance_cmd,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundfocus",
        description="SoundFocus: classify, localize and enhance sound sources.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(
        dest="command", metavar="<command>", title="commands", required=True
    )
    for module in COMMAND_MODULES:
        module.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Entry point of the ``soundspy`` command: argparse dispatcher over the subcommands.

Each subcommand module registers its own parser and sets ``handler`` via ``set_defaults``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from soundspy import __version__
from soundspy.cli import (
    check_mic_cmd,
    data_cmd,
    detect_cmd,
    devices_cmd,
    extract_cmd,
    record_cmd,
    record_dataset_cmd,
    serve_cmd,
    train_cmd,
)

COMMAND_MODULES = (
    serve_cmd,
    devices_cmd,
    check_mic_cmd,
    record_cmd,
    record_dataset_cmd,
    data_cmd,
    train_cmd,
    detect_cmd,
    extract_cmd,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundspy",
        description="Sound Spy: choose what to hear. Development, data and training commands.",
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

"""``soundfocus devices``: list audio input devices and the configured device profiles.

Lists input devices (WASAPI preferred on Windows) and shows which one the active
device profile resolves to.

TODO(step-01): add arguments and implement.
"""

from __future__ import annotations

import argparse

from soundfocus.cli import not_implemented

NAME = "devices"
HELP = "list audio input devices and the configured device profiles"
STEP = "step-01"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return not_implemented(STEP)

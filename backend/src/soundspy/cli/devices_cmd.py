"""``soundspy devices``: list audio devices and the device the profile resolves to.

Input devices are marked ``*`` (system default) and ``>>`` (what the profile resolves to);
output devices ``*`` (system default) and ``>>`` (what playback would use, step 03).
``--json`` prints the same structure as ``GET /api/devices``.
"""

from __future__ import annotations

import argparse
import json

from soundspy.cli.common import USER_ERRORS, add_profile_arguments, fail, load_profile
from soundspy.io.devices import (
    DeviceResolutionError,
    devices_report,
    hostapi_rank,
    resolve_output,
)

NAME = "devices"
HELP = "list audio devices and the device the profile resolves to"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    add_profile_arguments(parser)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.set_defaults(handler=run)


def _table(title: str, devices: list[dict], chosen: int | None) -> None:
    print(title)
    print("        idx  ch    rate  host API              name")
    rows = sorted(devices, key=lambda d: (hostapi_rank(d["hostapi"]), d["index"]))
    for d in rows:
        marker = ("*" if d["is_default"] else " ") + (">>" if d["index"] == chosen else "  ")
        print(
            f"  {marker}  {d['index']:4d}  {d['max_channels']:2d}  {d['default_samplerate']:6d}  "
            f"{d['hostapi']:<20}  {d['name']}"
        )


def run(args: argparse.Namespace) -> int:
    try:
        _, profile = load_profile(args)
        report = devices_report(profile)
    except USER_ERRORS as exc:
        return fail(exc)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    resolved = report["resolved_input"]
    _table(
        f"Input devices   (* = system default, >> = used by profile {profile.name!r})",
        report["inputs"],
        resolved["index"] if resolved else None,
    )
    if resolved is None:
        print(f"\n  profile {profile.name!r} does not resolve: {report['error']}")
    else:
        print(
            f"\n  profile {profile.name!r} -> device {resolved['index']} on {resolved['hostapi']}, "
            f"channels {resolved['channels_used']}, {resolved['sample_rate']} Hz"
        )

    try:
        output_index: int | None = resolve_output().index
    except DeviceResolutionError:
        output_index = None
    print()
    _table(
        "Output devices  (* = system default, >> = used for playback)",
        report["outputs"],
        output_index,
    )
    return 0

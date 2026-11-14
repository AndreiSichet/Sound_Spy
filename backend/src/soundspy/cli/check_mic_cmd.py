"""``soundspy check-mic``: record a short test clip and report levels per channel.

Report mode (default): countdown, record ``--seconds``, then per-channel RMS / peak / noise floor
/ clipping, overflows, a comparison of every channel with channel 0 (to tell real separate
microphones from a duplicated mono signal) and warnings.

``--meter``: a live per-channel level meter in the terminal until Ctrl+C, then the statistics.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from soundspy.cli.common import (
    USER_ERRORS,
    add_profile_arguments,
    countdown,
    fail,
    load_profile,
    print_device_line,
    print_level_report,
)
from soundspy.dsp.levels import DB_FLOOR, peak_dbfs, rms_dbfs
from soundspy.io.devices import resolve_input
from soundspy.io.mic_source import MicSource

NAME = "check-mic"
HELP = "record a short test clip and report levels per channel"
METER_CHUNK_S = 0.05
METER_WIDTH = 24
METER_MIN_DBFS = -60.0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    add_profile_arguments(parser)
    parser.add_argument("--seconds", type=float, default=5.0, help="recording length (default 5)")
    parser.add_argument("--meter", action="store_true", help="live level meter until Ctrl+C")
    parser.add_argument(
        "--countdown", type=float, default=2.0, help="seconds before recording (default 2)"
    )
    parser.add_argument("--meter-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.set_defaults(handler=run)


def _bar(dbfs: float) -> str:
    fraction = (max(dbfs, METER_MIN_DBFS) - METER_MIN_DBFS) / -METER_MIN_DBFS
    filled = int(round(fraction * METER_WIDTH))
    return "#" * filled + "-" * (METER_WIDTH - filled)


def _meter(mic: MicSource, max_seconds: float | None) -> None:
    chunk_frames = max(1, round(mic.sample_rate * METER_CHUNK_S))
    max_peak = np.full(mic.num_channels, DB_FLOOR)
    started = time.monotonic()
    print("Live meter (RMS, -60..0 dBFS) - press Ctrl+C to stop.")
    try:
        for chunk in mic.chunks(chunk_frames):
            rms, peak = rms_dbfs(chunk), peak_dbfs(chunk)
            max_peak = np.maximum(max_peak, peak)
            line = "  ".join(
                f"ch{ch} [{_bar(rms[ch])}] {rms[ch]:6.1f}" for ch in range(mic.num_channels)
            )
            print("\r" + line, end="", flush=True)
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        mic.close()
    print()
    print(
        f"\nStopped after {time.monotonic() - started:.1f} s.  frames: "
        f"{mic.stats.frames_delivered}  overflows: {mic.stats.overflows}  "
        f"dropped chunks: {mic.stats.dropped_chunks}"
    )
    peaks = ", ".join(f"ch{c} {v:.1f}" for c, v in enumerate(max_peak))
    print(f"max peak per channel (dBFS): {peaks}")


def run(args: argparse.Namespace) -> int:
    try:
        _, profile = load_profile(args)
        resolved = resolve_input(profile)
        print_device_line(resolved)
        if args.meter:
            with MicSource(resolved) as mic:
                _meter(mic, args.meter_seconds)
            return 0
        with MicSource(resolved) as mic:
            countdown(args.countdown, f"Recording {args.seconds:g} s")
            buf = mic.read(args.seconds)
        print("done.")
    except USER_ERRORS as exc:
        return fail(exc)
    print_level_report(buf)
    return 0

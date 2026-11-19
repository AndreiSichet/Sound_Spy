"""``soundspy record-benchmark``: record the real-room benchmark (step 07c).

    soundspy record-benchmark [--speech-seconds 60] [--music-seconds 60] [--room-seconds 30]
                              [--takes 1] [--pop] [--notes "..."]

Synthetic mixtures did not predict what the owner heard (step 07b), so this benchmark is made
of the laptop microphone in the real room. Its ground truth comes from **digital mixing of
parts recorded separately**, one session at a time, into ``data/benchmark/<session>/``:

1. ``speech``: you talk with natural pauses, quiet room (several takes allowed);
2. ``music``: classical music from the phone or a speaker at a normal listening level, nobody
   talking; with ``--pop`` also ``music_pop``: a song with vocals (the hardest case);
3. ``room``: room tone, nobody talking, no music.

``soundspy evaluate extraction --method hybrid --benchmark real`` then mixes them (speech +
music at several speech-to-music ratios, plus room tone) and scores every method against the
recorded speech. Keys per take: Enter = record (after a countdown), ``r`` = redo the last take,
``s`` = skip this part, ``q`` = quit. Every sidecar stores ``benchmark_session`` and ``part``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from soundspy.cli.common import (
    DEAD_INPUT_DBFS,
    USER_ERRORS,
    add_profile_arguments,
    countdown,
    fail,
    level_warnings,
    load_profile,
    print_device_line,
    print_level_report,
)
from soundspy.cli.record_dataset_cmd import new_session_id
from soundspy.config import AppConfig
from soundspy.core.audio import AudioBuffer
from soundspy.io.storage import SIDECAR_SUFFIX, RecordingWriter

NAME = "record-benchmark"
HELP = "record the real-room benchmark (speech, music, room tone) into data/benchmark"
KEYS_HELP = "Enter=record  r=redo last  s=skip part  q=quit"

PARTS = {
    "speech": "talk with natural pauses (as in a conversation), quiet room, no music",
    "music": "classical music from the phone / a speaker at a normal listening level, nobody "
             "talking",
    "music_pop": "a song WITH vocals from the phone / a speaker, nobody in the room talking",
    "room": "room tone: nobody talking, no music, keep still",
}


def benchmark_root(config: AppConfig) -> Path:
    """``data/benchmark`` (next to ``data/raw``; git-ignored like it)."""
    return Path(config.paths["raw"]).parent / "benchmark"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    add_profile_arguments(parser)
    parser.add_argument("--speech-seconds", type=float, default=60.0)
    parser.add_argument("--music-seconds", type=float, default=60.0)
    parser.add_argument("--room-seconds", type=float, default=30.0)
    parser.add_argument("--takes", type=int, default=1, help="takes per part (default 1)")
    parser.add_argument("--pop", action="store_true",
                        help="also record a song with vocals (music_pop)")
    parser.add_argument("--notes", default="", help="notes stored in every sidecar (room, setup)")
    parser.add_argument("--countdown", type=float, default=3.0,
                        help="seconds of countdown before each take (default 3)")
    parser.set_defaults(handler=run)


def room_tone_note(buf: AudioBuffer) -> str:
    """The informational line after the room part (step 07d: it used to warn "muted or very
    quiet input", but a quiet room is exactly what this part records)."""
    from soundspy.dsp.levels import rms_dbfs

    level = float(np.max(rms_dbfs(buf)))
    return (f"Room tone: {level:.1f} dBFS RMS (a quiet room is what this part needs; only an "
            f"input below {DEAD_INPUT_DBFS:.0f} dBFS would be a dead microphone).")


def _delete_take(wav: Path) -> None:
    wav.unlink(missing_ok=True)
    wav.with_suffix(SIDECAR_SUFFIX).unlink(missing_ok=True)


def record_benchmark(config: AppConfig, read_take: Callable[[float], AudioBuffer], *,
                     seconds: dict[str, float], takes: int = 1, notes: str = "",
                     profile: str | None = None, input_fn: Callable[[str], str] | None = None,
                     countdown_s: float = 3.0, session_id: str | None = None) -> list[Path]:
    """Interactive loop over the parts in ``seconds`` (part -> length); ``read_take(s)``
    records one buffer. Returns the kept WAVs."""
    session = session_id or new_session_id()
    ask = input_fn if input_fn is not None else input
    folder = benchmark_root(config) / session
    print(f"Benchmark session {session} -> {folder}")
    kept: list[Path] = []
    for part, length in seconds.items():
        print(f"\n== {part}: {PARTS[part]} ({length:g} s per take)")
        count, last = 0, None
        while count < takes:
            try:
                key = ask(f"[{part} {count + 1}/{takes}] {KEYS_HELP} > ").strip().lower()
            except EOFError:
                key = "q"
            if key == "q":
                print(f"Quit. Kept {len(kept)} take(s) in {folder}.")
                return kept
            if key == "s":
                print(f"Skipping {part}.")
                break
            if key == "r":
                if last is None:
                    print("Nothing to redo yet.")
                    continue
                _delete_take(last)
                kept.remove(last)
                count -= 1
                print(f"Discarded {last.name}.")
                last = None
            elif key != "":
                print(f"Unknown key {key!r}. {KEYS_HELP}")
                continue
            countdown(countdown_s, f"Recording {part} ({length:g} s)")
            buf = read_take(length)
            writer = RecordingWriter(
                folder, buf.sample_rate, buf.num_channels, label=part, notes=notes,
                profile=profile, metadata=buf.metadata, stem=f"{part}_{count + 1:02d}",
                extra={"benchmark_session": session, "part": part})
            writer.write(buf)
            last = writer.close()
            kept.append(last)
            count += 1
            room = part == "room"  # a quiet room is the goal: no "very quiet" warning
            print_level_report(buf, compare_channels=False, quiet_expected=room)
            if room:
                print(room_tone_note(buf))
            for warning in level_warnings(buf, quiet_expected=room):
                print(f"WARNING: {warning}")
            print(f"Saved {last.name}")
    print(f"\nDone. Kept {len(kept)} take(s). Score them with `soundspy evaluate extraction "
          f"--method hybrid --benchmark real --session {session}`.")
    return kept


def run(args: argparse.Namespace) -> int:
    from soundspy.io.devices import resolve_input
    from soundspy.io.mic_source import MicSource

    try:
        config, profile = load_profile(args)
        if args.takes < 1 or min(args.speech_seconds, args.music_seconds,
                                 args.room_seconds) <= 0:
            raise ValueError("--takes must be >= 1 and every length > 0")
        seconds = {"speech": args.speech_seconds, "music": args.music_seconds}
        if args.pop:
            seconds["music_pop"] = args.music_seconds
        seconds["room"] = args.room_seconds
        resolved = resolve_input(profile)
        print_device_line(resolved)
        with MicSource(resolved, config.capture) as mic:
            record_benchmark(config, mic.read, seconds=seconds, takes=args.takes,
                             notes=args.notes, profile=resolved.profile_name,
                             countdown_s=args.countdown)
    except (*USER_ERRORS, ValueError) as exc:
        return fail(exc)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0

"""``soundspy record-dataset``: record labelled clips into ``data/custom/<class>/``.

    soundspy record-dataset [--classes speech dishes] [--clips 10] [--seconds 5] [--notes "..."]

Classes are keys from ``configs/classes.yaml`` plus ``background`` (room tone, cafeteria
ambience without any selectable class). Each run is one **session**: its id is stored in every
sidecar (``session_id``, ``class``, ``notes``) and the custom collector splits by session, so the
takes of one sitting never end up in both train and test.

Keys per clip: Enter = record (after a countdown), ``r`` = redo the last take, ``d`` = discard
the last take, ``s`` = skip to the next class, ``q`` = quit. After each take the level report is
printed with warnings for clipping or silence. File names continue the numbering of earlier runs
(``speech_0007.wav`` after ``speech_0006.wav``). A take is deleted only by ``r``/``d`` right after
recording it; otherwise recordings are never modified.
"""

from __future__ import annotations

import argparse
import re
import secrets
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from soundspy.cli.common import (
    USER_ERRORS,
    add_profile_arguments,
    countdown,
    fail,
    level_warnings,
    load_profile,
    print_device_line,
    print_level_report,
)
from soundspy.config import AppConfig
from soundspy.core.audio import AudioBuffer
from soundspy.data.collectors.custom import BACKGROUND
from soundspy.io.devices import resolve_input
from soundspy.io.mic_source import MicSource
from soundspy.io.storage import SIDECAR_SUFFIX, RecordingWriter

NAME = "record-dataset"
HELP = "record labelled clips into data/custom (own training data)"

KEYS_HELP = "Enter=record  r=redo last  d=discard last  s=skip class  q=quit"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    add_profile_arguments(parser)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="class keys from classes.yaml and/or 'background' (default: all)")
    parser.add_argument("--clips", type=int, default=10, help="clips per class (default 10)")
    parser.add_argument("--seconds", type=float, default=5.0, help="clip length (default 5)")
    parser.add_argument("--notes", default="", help="notes stored in every sidecar (room, mic)")
    parser.add_argument("--countdown", type=float, default=2.0,
                        help="seconds of countdown before each take (default 2)")
    parser.set_defaults(handler=run)


def new_session_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def next_number(folder: Path, cls: str) -> int:
    """1 + the highest ``<cls>_NNNN`` already in ``folder`` (numbering continues across runs)."""
    pattern = re.compile(rf"^{re.escape(cls)}_(\d+)$")
    numbers = [int(m.group(1)) for p in folder.glob(f"{cls}_*.wav")
               if (m := pattern.match(p.stem))] if folder.is_dir() else []
    return max(numbers, default=0) + 1


def valid_classes(config: AppConfig, requested: Sequence[str] | None) -> list[str]:
    allowed = [*config.labels, BACKGROUND]
    if not requested:
        return allowed
    unknown = [c for c in requested if c not in allowed]
    if unknown:
        raise ValueError(f"unknown class(es) {unknown}; choose from {allowed}")
    return list(dict.fromkeys(requested))


def _delete_take(wav: Path) -> None:
    wav.unlink(missing_ok=True)
    wav.with_suffix(SIDECAR_SUFFIX).unlink(missing_ok=True)


def record_session(
    config: AppConfig,
    read_take: Callable[[float], AudioBuffer],
    *,
    classes: Sequence[str],
    clips: int,
    seconds: float,
    notes: str = "",
    profile: str | None = None,
    input_fn: Callable[[str], str] | None = None,
    countdown_s: float = 2.0,
    session_id: str | None = None,
) -> list[Path]:
    """Interactive loop; ``read_take(seconds)`` records one buffer. Returns the kept WAVs."""
    session = session_id or new_session_id()
    ask = input_fn if input_fn is not None else input
    root = config.paths["custom"]
    print(f"Session {session}: {clips} clip(s) of {seconds:g} s per class -> {root}")
    kept: list[Path] = []
    for cls in classes:
        folder = root / cls
        count, last = 0, None
        what = (config.labels.display_name(cls) if cls in config.labels
                else "room tone / ambience without any selectable class")
        print(f"\n== {cls} ({what})")
        while count < clips:
            try:
                key = ask(f"[{cls} {count + 1}/{clips}] {KEYS_HELP} > ").strip().lower()
            except EOFError:
                key = "q"
            if key == "q":
                print(f"Quit. Kept {len(kept)} clip(s) this session.")
                return kept
            if key == "s":
                print(f"Skipping {cls}.")
                break
            if key in ("r", "d"):
                if last is None:
                    print("Nothing to redo or discard yet.")
                    continue
                _delete_take(last)
                kept.remove(last)
                count -= 1
                print(f"Discarded {last.name}.")
                last = None
                if key == "d":
                    continue
            elif key != "":
                print(f"Unknown key {key!r}. {KEYS_HELP}")
                continue
            countdown(countdown_s, f"Recording {cls} ({seconds:g} s)")
            buf = read_take(seconds)
            number = next_number(folder, cls)
            writer = RecordingWriter(
                folder, buf.sample_rate, buf.num_channels, label=cls, notes=notes,
                profile=profile, metadata=buf.metadata, stem=f"{cls}_{number:04d}",
                extra={"class": cls, "session_id": session},
            )
            writer.write(buf)
            last = writer.close()
            kept.append(last)
            count += 1
            print_level_report(buf, compare_channels=False)
            for warning in level_warnings(buf):
                print(f"WARNING: {warning}")
            print(f"Saved {last.name}")
    print(f"\nDone. Kept {len(kept)} clip(s) this session. Run `soundspy data prepare` to index "
          "them.")
    return kept


def run(args: argparse.Namespace) -> int:
    try:
        config, profile = load_profile(args)
        classes = valid_classes(config, args.classes)
        if args.clips < 1 or args.seconds <= 0:
            raise ValueError("--clips must be >= 1 and --seconds > 0")
        resolved = resolve_input(profile)
        print_device_line(resolved)
        with MicSource(resolved, config.capture) as mic:
            record_session(
                config, mic.read, classes=classes, clips=args.clips, seconds=args.seconds,
                notes=args.notes, profile=resolved.profile_name, countdown_s=args.countdown,
            )
    except (*USER_ERRORS, ValueError) as exc:
        return fail(exc)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0

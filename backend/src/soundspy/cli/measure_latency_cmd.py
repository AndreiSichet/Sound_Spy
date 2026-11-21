"""``soundspy measure-latency``: Sound Spy's real delay, measured acoustically (step 08).

    soundspy measure-latency                    # hardware round trip only
    soundspy measure-latency --all              # + the app: pass-through, Hybrid speech route,
                                                #   Hybrid Waveformer route
    soundspy measure-latency --through-app [--method hybrid] [--labels speech] [--no-bypass]

**Use the laptop's own speakers, headphones unplugged, at a low volume**: the microphone must
hear the clicks. A click train (10 clicks, 1 s apart) is played through the output device and
recorded through the profile's microphone at the same time (``io/loopback.py``). Without the app
this is the **hardware round trip** (output + input latency). With ``--through-app`` Sound
Spy's own session runs meanwhile and plays the microphone back through the speakers: every click
then arrives twice, and the difference is **the app's microphone -> speaker delay**, the figure a
listener with headphones experiences (the acoustic path, ~1 ms at laptop distances, is in it).

The app modes measure with **Bypass** on: the session then plays its input delayed exactly like
its processed output (Chapter 10, "Bypass"), so the timing is the processed path's, whatever the
extractor does to a click (the speech enhancer would remove it). The estimate the session itself
reports (``latency_ms``) is printed next to each measurement. Results:
``experiments/realtime/latency_<time>.json`` and ``latency_latest.json`` (the page shows the
latest measured value next to its own estimate); the recordings themselves go to
``outputs/latency/<time>/<mode>.wav`` (git-ignored), to look at when a measurement fails.

**The app's playback gain** in the app modes is chosen from the hardware run's coupling
(``loopback.app_gain_db``): the loop speaker -> microphone -> app -> speaker aims at 25 dB below
unity. A run whose recording builds up (feedback, ``loopback.feedback_db``) is stopped and
repeated 10 dB quieter, at most twice; the following modes keep the lower gain. Without a
hardware result the app plays at -10 dB.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from soundspy.cli.common import USER_ERRORS, add_profile_arguments, fail, load_profile

NAME = "measure-latency"
HELP = "measure the real delay acoustically (laptop speakers -> microphone)"
WARNING = (
    "This plays a train of clicks through the speakers and records it with the microphone.\n"
    "  * Use the laptop's own speakers: unplug the headphones.\n"
    "  * Set a LOW volume (about a third): the clicks are short but sharp.\n"
    "  * Keep the room quiet and the laptop still for about 15 s per measurement.\n"
    "  * The app measurements play the microphone back through the speakers, at a gain chosen\n"
    "    from the first measurement so that it cannot howl.")

MODES = {
    "passthrough": {"method": "passthrough", "labels": (), "title": "app, no processing"},
    "speech": {"method": "hybrid", "labels": ("speech",), "title": "Hybrid, speech route"},
    "waveformer": {"method": "hybrid", "labels": ("music",), "title": "Hybrid, Waveformer route"},
}


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    add_profile_arguments(parser)
    parser.add_argument("--through-app", action="store_true",
                        help="measure through Sound Spy's own session too")
    parser.add_argument("--all", action="store_true",
                        help="hardware + pass-through + Hybrid speech route + Waveformer route")
    parser.add_argument("--method", default="hybrid", help="--through-app: extraction method")
    parser.add_argument("--labels", nargs="*", default=["speech"],
                        help="--through-app: the selected classes (decides the hybrid's route)")
    parser.add_argument("--no-bypass", action="store_true",
                        help="--through-app: hear the processed output (the extractor may remove "
                             "the clicks: then nothing is found)")
    parser.add_argument("--output-device", default=None,
                        help="output device index or name (default: the app's choice)")
    parser.add_argument("--clicks", type=int, default=10)
    parser.add_argument("--period", type=float, default=1.0, help="seconds between clicks")
    parser.add_argument("--level", type=float, default=-6.0, help="click level, dBFS")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a setting for this measurement (e.g. "
                             "output.prebuffer_chunks=1); repeatable")
    parser.add_argument("--yes", action="store_true", help="do not wait for Enter")
    parser.set_defaults(handler=run)


def _config(args: argparse.Namespace):
    from soundspy.config import override
    from soundspy.io.devices import parse_device_argument
    from soundspy.user_settings import apply_user_settings

    config, profile = load_profile(args)
    config = apply_user_settings(config)
    config = override(config, "device_profile", profile)
    if args.output_device is not None:
        config = override(config, "output.device", parse_device_argument(args.output_device))
    for item in args.set:
        key, _, value = item.partition("=")
        config = override(config, key.strip(), value.strip())
    return config


def _save_wav(folder: Path | None, mode: str, recorded, sr: int) -> str | None:
    if folder is None:
        return None
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{mode}.wav"
    sf.write(str(path), recorded, sr, subtype="FLOAT")
    return str(path)


def measure(config, plan: list[str], *, clicks: int = 10, period_s: float = 1.0,
            level_dbfs: float = -6.0, method: str = "hybrid", labels=("speech",),
            bypass: bool = True, progress=print,
            audio_dir: Path | None = None) -> dict[str, Any]:
    """Run the plan (``hardware`` first, then the app modes); returns the results.
    ``audio_dir``: where each mode's recording is saved (``None``: not saved)."""
    from soundspy.engine.live import live_session
    from soundspy.io import loopback as lb
    from soundspy.io.devices import refresh_devices, resolve_input, resolve_output

    refresh_devices()  # current device handles (see engine/live.py)
    resolved = resolve_input(config.device_profile)
    output = resolve_output(config.output.device)
    sr = resolved.sample_rate
    train = lb.click_train(sr, clicks=clicks, period_s=period_s, level_dbfs=level_dbfs)
    keep = resolved.channel_indices[0]
    result: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "input": resolved.device.name, "output": output.name, "sample_rate": sr,
        "clicks": clicks, "period_s": period_s, "level_dbfs": level_dbfs,
        "settings": {"chunk_ms": config.engine.chunk_ms,
                     "prebuffer_chunks": config.output.prebuffer_chunks,
                     "capture_latency": config.capture.latency,
                     "output_latency": config.output.latency,
                     "exclusive": [config.capture.exclusive, config.output.exclusive],
                     "fixed_latency": config.extraction.hybrid.fixed_latency},
        "modes": {}}

    recordings: dict[str, str | None] = {}

    def record(mode: str, through_app: bool):
        recorded = lb.play_and_record(train, input_index=resolved.device.index,
                                      output_index=output.index,
                                      input_channels=resolved.open_channels, keep_channel=keep,
                                      wasapi=resolved.device.hostapi == "Windows WASAPI",
                                      abort_rise_db=lb.ABORT_RISE_DB if through_app else None)
        recordings[mode] = _save_wav(audio_dir, mode, recorded, sr)
        arrivals = lb.find_arrivals(recorded, train, through_app=through_app)
        return arrivals, lb.feedback_db(recorded, train)

    gain_db = -10.0
    for mode in plan:
        if mode == "hardware":
            progress("measuring the hardware round trip ...")
            arrivals, _ = record(mode, False)
            coupling = arrivals.coupling_db
            gain_db = lb.app_gain_db(coupling) if coupling is not None else -10.0
            result["coupling_db"], result["app_gain_db"] = coupling, gain_db
            result["modes"]["hardware"] = {
                "title": "hardware round trip", **lb.summarize_ms(arrivals.direct, sr),
                "snr_db": lb.median_or_none(arrivals.snr_db),
                "best_snr_db": max(arrivals.snr_db, default=None),
                "coupling_db": coupling}
            continue
        spec = MODES.get(mode, {"method": method, "labels": tuple(labels),
                                "title": f"{method} {'+'.join(labels) or 'nothing selected'}"})
        retries = 0
        while True:
            progress(f"measuring through the app: {spec['title']} "
                     f"(playback {gain_db:+.0f} dB) ...")
            session = live_session(config, method=spec["method"], labels=spec["labels"],
                                   attenuation_db=0.0, bypass=bypass, monitor=True,
                                   gain_db=gain_db)
            session.start()
            try:
                if not session.wait_until_running(60.0):
                    raise RuntimeError(f"the session did not start: {session.status.message}")
                time.sleep(1.5)  # the output queue settles
                estimate = session.status.latency_ms
                arrivals, rise = record(mode, True)
                status = session.status
            finally:
                session.stop()
            if rise is None or rise < lb.FEEDBACK_DB or retries >= 2:
                break
            retries += 1
            gain_db -= 10.0
            progress(f"  feedback: the recording grew by {rise:.0f} dB; once more, 10 dB "
                     "quieter ...")
            time.sleep(1.0)  # let the room fall silent
        result["modes"][mode] = {
            "title": spec["title"], "bypass": bypass,
            "hardware": lb.summarize_ms(arrivals.direct, sr),
            **lb.summarize_ms(lb.app_delays(arrivals), sr),
            "snr_db": lb.median_or_none(arrivals.snr_db),
            "app_snr_db": lb.median_or_none(arrivals.app_snr_db),
            "app_gain_db": gain_db, "feedback_db": rise, "feedback_retries": retries,
            "estimate_ms": estimate, "underruns": status.output_underruns,
            "extractor_ms": (status.extraction or {}).get("latency_ms"),
            "route": (status.extraction or {}).get("route")}
    result["recordings"] = recordings
    return result


def format_result(result: dict[str, Any]) -> str:
    lines = [f"{'what':34s} {'median':>8s} {'min':>7s} {'max':>7s} {'clicks':>7s} "
             f"{'estimate':>9s}"]
    for mode in result["modes"].values():
        est = mode.get("estimate_ms")
        found = f"{mode['found']}/{result['clicks']}"
        fmt = lambda v: "   -" if v is None else f"{v:.1f}"  # noqa: E731
        lines.append(f"{mode['title']:34s} {fmt(mode['median_ms']):>8s} {fmt(mode['min_ms']):>7s} "
                     f"{fmt(mode['max_ms']):>7s} {found:>7s} "
                     f"{'' if est is None else fmt(est):>9s}")
    lines.append("(milliseconds; app rows: microphone -> speaker through Sound Spy, the hardware "
                 "round trip removed; estimate = what the session reports)")
    if result.get("coupling_db") is not None:
        lines.append(f"speaker -> microphone coupling {result['coupling_db']:+.1f} dB; the app "
                     f"played the microphone back at {result['app_gain_db']:+.1f} dB")
    return "\n".join(lines)


def diagnosis(result: dict[str, Any]) -> list[str]:
    """One line per mode that found fewer clicks than it played: how far above the room's
    noise its best click was (the threshold is 18 dB)."""
    lines = []
    for mode in result["modes"].values():
        if mode["found"] >= result["clicks"]:
            continue
        hw = mode.get("best_snr_db", mode.get("snr_db"))
        app = mode.get("app_snr_db")
        parts = [f"direct clicks {hw:+.1f} dB" if hw is not None else None,
                 f"the app's copies {app:+.1f} dB" if app is not None else None]
        text = ", ".join(p for p in parts if p)
        lines.append(f"  {mode['title']}: {mode['found']}/{result['clicks']} found"
                     + (f" ({text} above the room's noise; 18 dB needed)" if text else ""))
        rise = mode.get("feedback_db")
        if rise is not None and rise >= 10.0:
            lines.append(f"    feedback: the recording still grew by {rise:.0f} dB "
                         f"(playback {mode.get('app_gain_db', 0):+.0f} dB); lower the volume")
    return lines


def save(config, result: dict[str, Any]) -> Path:
    folder = Path(config.paths["experiments"]) / "realtime"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"latency_{stamp}.json"
    text = json.dumps(result, indent=2)
    path.write_text(text, encoding="utf-8")
    (folder / "latency_latest.json").write_text(text, encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    from soundspy.config import ConfigError

    try:
        config = _config(args)
        plan = ["hardware"]
        if args.all:
            plan += ["passthrough", "speech", "waveformer"]
        elif args.through_app:
            plan.append("custom")
        print(WARNING)
        if not args.yes:
            input("Press Enter when ready (Ctrl+C to cancel) ")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        audio_dir = Path(config.paths["outputs"]) / "latency" / stamp
        result = measure(config, plan, clicks=args.clicks, period_s=args.period,
                         level_dbfs=args.level, method=args.method, labels=args.labels,
                         bypass=not args.no_bypass, audio_dir=audio_dir)
    except (*USER_ERRORS, ConfigError, RuntimeError, ValueError) as exc:
        return fail(exc)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    print()
    print(format_result(result))
    short = diagnosis(result)
    if short:
        print("\nNot every click was found:")
        print("\n".join(short))
        print("Headphones unplugged? Volume up a little (half)? Microphone not muted? The "
              "recordings are in " + str(audio_dir))
    print(f"-> {save(config, result)}")
    return 0

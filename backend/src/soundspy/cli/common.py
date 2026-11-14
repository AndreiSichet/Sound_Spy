"""Helpers shared by the audio commands: profile arguments, countdown, level report."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import numpy as np

from soundspy.config import AppConfig, ConfigError, DeviceProfile, load_config, load_device_profile
from soundspy.core.audio import AudioBuffer
from soundspy.dsp.levels import (
    clipping_ratio,
    compare_to_reference,
    digital_silence_ratio,
    noise_floor_dbfs,
    peak_dbfs,
    rms_dbfs,
)
from soundspy.io.devices import (
    AudioBackendUnavailable,
    DeviceResolutionError,
    ResolvedInput,
    parse_device_argument,
)
from soundspy.io.mic_source import MicStallError

# Errors that are the user's setup rather than a bug: printed as one line, exit code 1.
USER_ERRORS = (AudioBackendUnavailable, DeviceResolutionError, ConfigError, MicStallError)

CLIPPING_WARN = 0.001  # 0.1 %
QUIET_WARN_DBFS = -55.0
NOISY_WARN_DBFS = -45.0
SILENCE_WARN = 0.01  # more than 1 % of the recording in >= 5 ms runs of exact zeros
TOO_CLEAN_WARN_DBFS = -100.0  # below any real mic + converter self-noise
IDENTICAL_CORR = 0.995


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile", default=None, help="device profile (default: active_device_profile)"
    )
    parser.add_argument(
        "--device", default=None, help="device index or name substring (overrides the profile)"
    )


def load_profile(args: argparse.Namespace) -> tuple[AppConfig, DeviceProfile]:
    config = load_config()
    profile = (
        load_device_profile(args.profile, config.configs_dir)
        if args.profile
        else config.device_profile
    )
    if args.device is not None:
        profile = dataclasses.replace(profile, device=parse_device_argument(args.device))
    return config, profile


def fail(error: BaseException) -> int:
    print(f"error: {error}", file=sys.stderr)
    return 1


def countdown(seconds: float, what: str) -> None:
    whole = int(seconds)
    if whole <= 0:
        print(f"{what}...", flush=True)
        return
    print(f"{what} in", end="", flush=True)
    for remaining in range(whole, 0, -1):
        print(f" {remaining}...", end="", flush=True)
        time.sleep(1.0)
    print(" now.", flush=True)


def print_device_line(resolved: ResolvedInput) -> None:
    device = resolved.device
    print(f"Device : {device.name}  [{device.hostapi}, index {device.index}]")
    print(
        f"Profile: {resolved.profile_name}   channels kept: {list(resolved.channel_indices)} "
        f"of {resolved.open_channels} opened   sample rate: {resolved.sample_rate} Hz"
    )


def level_warnings(buf: AudioBuffer) -> list[str]:
    warnings = []
    clip = clipping_ratio(buf)
    for ch in np.flatnonzero(clip > CLIPPING_WARN):
        warnings.append(
            f"channel {ch}: clipping on {clip[ch] * 100:.2f} % of samples - "
            "lower the input gain or move away from the microphone"
        )
    if np.all(rms_dbfs(buf) < QUIET_WARN_DBFS):
        warnings.append(
            f"all channels below {QUIET_WARN_DBFS:.0f} dBFS RMS: muted or very quiet input"
        )
    silence = digital_silence_ratio(buf)
    for ch in np.flatnonzero(silence > SILENCE_WARN):
        warnings.append(
            f"channel {ch}: {silence[ch] * 100:.1f} % of the recording is stretches (>= 5 ms) of "
            "exact digital silence - a noise gate / noise suppression (Windows audio "
            "enhancements, Voice Clarity, driver effects) or a muted input"
        )
    floor = noise_floor_dbfs(buf)
    for ch in np.flatnonzero(floor < TOO_CLEAN_WARN_DBFS):
        warnings.append(
            f"channel {ch}: noise floor {floor[ch]:.1f} dBFS is cleaner than any real microphone "
            "(their own hiss sits well above -100 dBFS): noise suppression is probably active - "
            "turn off Windows audio enhancements / Voice Clarity for this microphone"
        )
    for ch in np.flatnonzero(floor > NOISY_WARN_DBFS):
        warnings.append(
            f"channel {ch}: noise floor {floor[ch]:.1f} dBFS - noisy input, or automatic gain "
            "control (AGC) / Windows audio enhancements are active"
        )
    return warnings


def similarity_verdict(identical: bool, lag: int, corr_zero_lag: float) -> str:
    if identical:
        return "duplicated mono: no spatial information"
    if lag != 0:
        return "distinct microphones"
    if abs(corr_zero_lag) > IDENTICAL_CORR:
        return (
            "very similar, possibly merged by the driver; repeat with a clap near one side"
        )
    return "distinct signals"


def print_level_report(buf: AudioBuffer, *, compare_channels: bool = True) -> None:
    rms, peak = rms_dbfs(buf), peak_dbfs(buf)
    floor, clip = noise_floor_dbfs(buf), clipping_ratio(buf)
    print()
    print(" ch   RMS dBFS   peak dBFS   noise floor   clipping")
    for ch in range(buf.num_channels):
        print(
            f"{ch:3d}   {rms[ch]:8.1f}   {peak[ch]:9.1f}   {floor[ch]:11.1f}   "
            f"{clip[ch] * 100:7.3f} %"
        )
    print(
        f"duration: {buf.duration_s:.2f} s   overflows: {buf.metadata.get('overflows', 0)}   "
        f"start-up silence discarded: {buf.metadata.get('warmup_frames_discarded', 0)} frames"
    )

    if compare_channels and buf.num_channels >= 2:
        print()
        print("Channel comparison (against channel 0, lags within +/-1 ms):")
        for ch, sim in compare_to_reference(buf, 0).items():
            verdict = similarity_verdict(sim.identical, sim.best_lag_samples, sim.corr_zero_lag)
            print(
                f"  ch {ch}: zero-lag corr {sim.corr_zero_lag:+.4f}, peak corr "
                f"{sim.peak_corr:+.4f} at lag {sim.best_lag_samples:+d} samples "
                f"({sim.best_lag_s * 1000:+.3f} ms) -> {verdict}"
            )
        print(
            "  Hint: clap near the LEFT side and run again, then clap near the RIGHT side: "
            "with two real microphones the lag sign should flip."
        )

    warnings = level_warnings(buf)
    print()
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  ! {warning}")
    else:
        print("No level warnings.")

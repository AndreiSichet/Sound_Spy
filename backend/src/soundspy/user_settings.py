"""User settings that the app changes itself: ``data/user_settings.json`` (step 08).

The configuration files under ``backend/configs/`` are the project's; a choice the user makes in
the page that must survive a restart (today: the **output device**) goes into a small overlay
file next to the recordings instead, ``data/user_settings.json`` (git-ignored like everything in
``data/``). The app and the live CLI tools apply it on top of the configuration
(:func:`apply_user_settings`); the library's ``load_config`` does not, so tests and the training
tools never depend on one person's choices.

Keys: ``output_device`` (``null`` = the system default, a device index, or a name substring as
in ``output.device``).
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from soundspy.config import AppConfig

KEYS = ("output_device",)


def settings_path(config: AppConfig) -> Path:
    """``data/user_settings.json`` (next to ``data/raw``)."""
    return Path(config.paths["raw"]).parent / "user_settings.json"


def load_user_settings(config: AppConfig) -> dict[str, Any]:
    """The stored settings (``{}`` when there is no file or it cannot be read)."""
    path = settings_path(config)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if k in KEYS} if isinstance(data, dict) else {}


def save_user_settings(config: AppConfig, **changes: Any) -> dict[str, Any]:
    """Merge ``changes`` into the file (written atomically) and return the new settings."""
    unknown = set(changes) - set(KEYS)
    if unknown:
        raise ValueError(f"unknown user settings: {sorted(unknown)}")
    settings = {**load_user_settings(config), **changes}
    path = settings_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return settings


def apply_user_settings(config: AppConfig, settings: dict[str, Any] | None = None) -> AppConfig:
    """``config`` with the stored settings on top (the output device)."""
    settings = load_user_settings(config) if settings is None else settings
    if "output_device" in settings:
        device = settings["output_device"]
        if isinstance(device, bool) or not (device is None or isinstance(device, int | str)):
            return config  # a hand-edited file with nonsense: keep the configuration
        config = dataclasses.replace(config, output=dataclasses.replace(config.output,
                                                                        device=device))
    return config

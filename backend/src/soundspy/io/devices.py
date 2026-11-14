"""Input/output device discovery and resolution of device profiles.

On Windows one physical device is listed once per host API (MME, DirectSound, WASAPI, WDM-KS).
They are "twins": the same microphone through different driver layers. When a profile does not
name a host API, the twin on the most preferred host API is used (``HOSTAPI_PREFERENCE``).
MME truncates device names to 31 characters, so twins are matched case-insensitively on the
first 31 characters of the name.

WASAPI shared mode only accepts the device's own sample rate unless auto-conversion is enabled,
so WASAPI streams are opened with ``WasapiSettings(auto_convert=True)`` (``wasapi_auto_convert``).

This module (with the rest of ``io/``) is the only place that touches ``sounddevice``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from soundspy.config import DeviceProfile

HOSTAPI_PREFERENCE: tuple[str, ...] = (
    "Windows WASAPI",
    "Windows DirectSound",
    "MME",
    "Windows WDM-KS",
    "Core Audio",
    "ALSA",
)
MME_NAME_LENGTH = 31


class AudioBackendUnavailable(RuntimeError):
    """PortAudio (the library under ``sounddevice``) cannot be loaded or queried."""


class DeviceResolutionError(ValueError):
    """A device profile (or a device argument) cannot be matched to an available device."""


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    hostapi: str
    max_channels: int
    default_samplerate: int
    is_default: bool = False


@dataclass(frozen=True)
class OutputDevice:
    index: int
    name: str
    hostapi: str
    max_channels: int
    default_samplerate: int
    is_default: bool = False


@dataclass(frozen=True)
class ResolvedInput:
    """Everything needed to open a microphone for a given profile."""

    device: InputDevice
    open_channels: int  # how many device channels the stream opens
    channel_indices: tuple[int, ...]  # which of those channels are kept, in order
    sample_rate: int
    wasapi_auto_convert: bool
    profile_name: str

    @property
    def num_channels(self) -> int:
        return len(self.channel_indices)


# --- sounddevice access ----------------------------------------------------------------------


def _sd() -> Any:
    """Import ``sounddevice`` lazily; a missing PortAudio becomes ``AudioBackendUnavailable``."""
    try:
        import sounddevice
    except (ImportError, OSError) as exc:  # OSError: PortAudio library not found
        raise AudioBackendUnavailable(f"PortAudio is not available: {exc}") from exc
    return sounddevice


def refresh_devices() -> bool:
    """Make PortAudio enumerate the devices again; returns False if that is not supported.

    PortAudio reads the device list once, when it is initialised, and keeps using those device
    handles. Plugging headphones into a Realtek jack (or a USB microphone) reconfigures Windows'
    audio endpoints, after which opening a stream through the old handles can fail with an
    "Unanticipated host error". Re-initialising picks up the current devices.

    Must only be called while **no stream is open** (it would kill running streams).
    """
    sd = _sd()
    terminate = getattr(sd, "_terminate", None)
    initialize = getattr(sd, "_initialize", None)
    if terminate is None or initialize is None:
        return False
    try:
        terminate()
        initialize()
    except Exception as exc:  # noqa: BLE001 - PortAudioError and friends
        raise AudioBackendUnavailable(f"could not re-initialise PortAudio: {exc}") from exc
    return True


def _query() -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int]]:
    sd = _sd()
    try:
        devices = [dict(d) for d in sd.query_devices()]
        hostapis = [dict(h) for h in sd.query_hostapis()]
        default_in, default_out = _default_indices(sd)
    except AudioBackendUnavailable:
        raise
    except Exception as exc:  # sd.PortAudioError and friends
        raise AudioBackendUnavailable(f"cannot query audio devices: {exc}") from exc
    for i, device in enumerate(devices):
        device.setdefault("index", i)
    return devices, hostapis, (default_in, default_out)


def _default_indices(sd: Any) -> tuple[int, int]:
    """System default input/output device indices (-1 if there is none)."""
    result = []
    for slot, kind in ((0, "input"), (1, "output")):
        index = -1
        try:
            value = sd.default.device[slot]
            index = int(value) if value is not None else -1
        except (TypeError, ValueError, IndexError):
            index = -1
        if index < 0:
            try:
                index = int(sd.query_devices(kind=kind)["index"])
            except Exception:  # noqa: BLE001 - no default device of this kind
                index = -1
        result.append(index)
    return result[0], result[1]


def _name_key(name: str) -> str:
    # Truncate first, then strip: an MME name cut at 31 characters may end in a space.
    return name.casefold()[:MME_NAME_LENGTH].strip()


def hostapi_rank(hostapi: str) -> int:
    """Position in ``HOSTAPI_PREFERENCE`` (unknown host APIs sort last)."""
    try:
        return HOSTAPI_PREFERENCE.index(hostapi)
    except ValueError:
        return len(HOSTAPI_PREFERENCE)


# --- listing ---------------------------------------------------------------------------------


def _list(kind: str) -> list[InputDevice] | list[OutputDevice]:
    devices, hostapis, (default_in, default_out) = _query()
    channels_key = f"max_{kind}_channels"
    default_index = default_in if kind == "input" else default_out
    cls = InputDevice if kind == "input" else OutputDevice
    result = []
    for d in devices:
        if int(d.get(channels_key, 0)) <= 0:
            continue
        result.append(
            cls(
                index=int(d["index"]),
                name=str(d["name"]),
                hostapi=str(hostapis[int(d["hostapi"])]["name"]),
                max_channels=int(d[channels_key]),
                default_samplerate=int(round(float(d["default_samplerate"]))),
                is_default=int(d["index"]) == default_index,
            )
        )
    return result


def list_input_devices() -> list[InputDevice]:
    return _list("input")  # type: ignore[return-value]


def list_output_devices() -> list[OutputDevice]:
    return _list("output")  # type: ignore[return-value]


# --- resolution ------------------------------------------------------------------------------


def _filter_hostapi(devices: Sequence[Any], hostapi: str | None, kind: str) -> list[Any]:
    if hostapi is None:
        return list(devices)
    wanted = hostapi.casefold()
    matching = [d for d in devices if wanted in d.hostapi.casefold()]
    if not matching:
        available = sorted({d.hostapi for d in devices})
        raise DeviceResolutionError(
            f"no {kind} device on host API {hostapi!r}; host APIs with {kind} devices: {available}"
        )
    return matching


def _pick_twin(candidates: Sequence[Any]) -> Any:
    """Among twins of one physical device, take the most preferred host API."""
    return min(candidates, key=lambda d: (hostapi_rank(d.hostapi), d.index))


def _select(
    devices: Sequence[Any],
    selector: str | int | None,
    hostapi: str | None,
    kind: str,
) -> Any:
    if not devices:
        raise DeviceResolutionError(f"no {kind} devices found")

    if isinstance(selector, int):
        for d in devices:
            if d.index == selector:
                if hostapi is not None and hostapi.casefold() not in d.hostapi.casefold():
                    raise DeviceResolutionError(
                        f"{kind} device {selector} ({d.name!r}) is on {d.hostapi!r}, "
                        f"not on host API {hostapi!r}"
                    )
                return d
        raise DeviceResolutionError(
            f"no {kind} device with index {selector}; run `soundspy devices` to list them"
        )

    pool = _filter_hostapi(devices, hostapi, kind)

    if selector is None:
        default = next((d for d in devices if d.is_default), None)
        if default is None:
            raise DeviceResolutionError(
                f"no system default {kind} device; set 'device' in the profile"
            )
        twins = [d for d in pool if _name_key(d.name) == _name_key(default.name)]
        if not twins:
            if hostapi is None:
                return default
            raise DeviceResolutionError(
                f"the default {kind} device {default.name!r} is not available on "
                f"host API {hostapi!r}"
            )
        return _pick_twin(twins)

    wanted = selector.casefold()
    matches = [d for d in pool if wanted in d.name.casefold()]
    if not matches:
        names = sorted({d.name for d in pool})
        raise DeviceResolutionError(
            f"no {kind} device whose name contains {selector!r}; available: {names}"
        )
    groups: dict[str, list[Any]] = {}
    for d in matches:
        groups.setdefault(_name_key(d.name), []).append(d)
    if len(groups) > 1:
        names = sorted({_pick_twin(g).name for g in groups.values()})
        raise DeviceResolutionError(
            f"{selector!r} matches several {kind} devices: {names}; use a longer name or an index"
        )
    return _pick_twin(next(iter(groups.values())))


def resolve_input(profile: DeviceProfile) -> ResolvedInput:
    """Choose the input device, channels and sample rate for ``profile``."""
    device: InputDevice = _select(list_input_devices(), profile.device, profile.hostapi, "input")

    if profile.raw_channels is None:
        open_channels = device.max_channels
        channel_indices = tuple(range(open_channels))
    else:
        open_channels = max(profile.raw_channels) + 1
        channel_indices = tuple(profile.raw_channels)
        if open_channels > device.max_channels:
            raise DeviceResolutionError(
                f"profile {profile.name!r} needs device channel {open_channels - 1} "
                f"(raw_channels={list(profile.raw_channels)}) but {device.name!r} on "
                f"{device.hostapi} has only {device.max_channels} input channel(s); "
                "is the right firmware/driver installed?"
            )

    return ResolvedInput(
        device=device,
        open_channels=open_channels,
        channel_indices=channel_indices,
        sample_rate=profile.sample_rate or device.default_samplerate,
        wasapi_auto_convert=device.hostapi == "Windows WASAPI",
        profile_name=profile.name,
    )


def resolve_output(
    device: str | int | None = None, hostapi: str | None = None
) -> OutputDevice:
    """Choose an output device: ``None`` = system default (its preferred host-API twin)."""
    return _select(list_output_devices(), device, hostapi, "output")


def device_to_dict(device: InputDevice | OutputDevice) -> dict[str, Any]:
    return {
        "index": device.index,
        "name": device.name,
        "hostapi": device.hostapi,
        "max_channels": device.max_channels,
        "default_samplerate": device.default_samplerate,
        "is_default": device.is_default,
    }


def resolved_to_dict(resolved: ResolvedInput) -> dict[str, Any]:
    return {
        "index": resolved.device.index,
        "name": resolved.device.name,
        "hostapi": resolved.device.hostapi,
        "channels_used": list(resolved.channel_indices),
        "sample_rate": resolved.sample_rate,
    }


def devices_report(profile: DeviceProfile) -> dict[str, Any]:
    """All devices plus what ``profile`` resolves to (shared by ``soundspy devices --json`` and
    ``GET /api/devices``). An unresolvable profile is reported in ``error``, not raised;
    ``AudioBackendUnavailable`` is raised."""
    inputs = list_input_devices()
    outputs = list_output_devices()
    resolved: dict[str, Any] | None = None
    error: str | None = None
    try:
        resolved = resolved_to_dict(resolve_input(profile))
    except DeviceResolutionError as exc:
        error = str(exc)
    return {
        "active_profile": profile.name,
        "resolved_input": resolved,
        "error": error,
        "inputs": [device_to_dict(d) for d in inputs],
        "outputs": [device_to_dict(d) for d in outputs],
    }


def parse_device_argument(value: str | None) -> str | int | None:
    """``--device`` on the CLI: digits mean an index, anything else a name substring."""
    if value is None:
        return None
    value = value.strip()
    return int(value) if value.isdigit() else value

"""Device listing and profile resolution against the fake Windows laptop (see conftest)."""

from __future__ import annotations

import pytest

from soundspy.config import DeviceProfile
from soundspy.io.devices import (
    DeviceResolutionError,
    devices_report,
    list_input_devices,
    list_output_devices,
    parse_device_argument,
    resolve_input,
    resolve_output,
)

LAPTOP = DeviceProfile(name="laptop")
RESPEAKER = DeviceProfile(name="respeaker", device="reSpeaker", raw_channels=(2, 3, 4, 5))


def test_lists_inputs_and_outputs() -> None:
    inputs, outputs = list_input_devices(), list_output_devices()
    assert [d.index for d in inputs] == [0, 1, 2, 5, 6]
    assert [d.index for d in outputs] == [3, 4, 7, 8]
    by_index = {d.index: d for d in inputs}
    assert by_index[1].hostapi == "MME" and by_index[1].is_default
    assert by_index[5].hostapi == "Windows WASAPI" and not by_index[5].is_default
    assert by_index[5].default_samplerate == 48000
    assert isinstance(by_index[5].default_samplerate, int)
    assert by_index[6].max_channels == 6


def test_default_input_prefers_the_wasapi_twin_of_the_mme_default() -> None:
    resolved = resolve_input(LAPTOP)
    assert resolved.device.index == 5  # MME default is 1; its WASAPI twin is 5
    assert resolved.device.hostapi == "Windows WASAPI"
    assert resolved.sample_rate == 48000
    assert resolved.open_channels == 2 and resolved.channel_indices == (0, 1)
    assert resolved.wasapi_auto_convert is True
    assert resolved.profile_name == "laptop"


def test_explicit_hostapi_keeps_mme() -> None:
    resolved = resolve_input(DeviceProfile(name="p", hostapi="mme"))
    assert resolved.device.index == 1 and resolved.sample_rate == 44100
    assert resolved.wasapi_auto_convert is False


def test_unknown_hostapi_is_a_clear_error() -> None:
    with pytest.raises(DeviceResolutionError, match="no input device on host API 'ASIO'"):
        resolve_input(DeviceProfile(name="p", hostapi="ASIO"))


def test_name_substring_is_case_insensitive_and_prefers_wasapi() -> None:
    assert resolve_input(DeviceProfile(name="p", device="RESPEAKER")).device.index == 6
    assert resolve_input(DeviceProfile(name="p", device="intel")).device.index == 5


def test_ambiguous_name_lists_candidates() -> None:
    with pytest.raises(DeviceResolutionError, match="matches several input devices"):
        resolve_input(DeviceProfile(name="p", device="mic"))


def test_index_selection_and_missing_devices() -> None:
    assert resolve_input(DeviceProfile(name="p", device=1)).device.hostapi == "MME"
    with pytest.raises(DeviceResolutionError, match="no input device with index 42"):
        resolve_input(DeviceProfile(name="p", device=42))
    with pytest.raises(DeviceResolutionError, match="no input device with index 3"):
        resolve_input(DeviceProfile(name="p", device=3))  # an output-only device
    with pytest.raises(DeviceResolutionError, match="whose name contains 'Zoom H6'"):
        resolve_input(DeviceProfile(name="p", device="Zoom H6"))


def test_respeaker_raw_channels_open_six_keep_four_in_order() -> None:
    resolved = resolve_input(RESPEAKER)
    assert resolved.device.index == 6
    assert resolved.open_channels == 6
    assert resolved.channel_indices == (2, 3, 4, 5)
    assert resolved.num_channels == 4
    assert resolved.sample_rate == 16000


def test_not_enough_channels_is_a_clear_error() -> None:
    profile = DeviceProfile(name="p", raw_channels=(0, 5))
    with pytest.raises(DeviceResolutionError, match="has only 2 input channel"):
        resolve_input(profile)


def test_sample_rate_override() -> None:
    assert resolve_input(DeviceProfile(name="p", sample_rate=16000)).sample_rate == 16000


def test_resolve_output() -> None:
    assert resolve_output().index == 7  # WASAPI twin of the MME default speakers
    assert resolve_output("headphones").index == 8
    assert resolve_output(4).hostapi == "MME"
    with pytest.raises(DeviceResolutionError):
        resolve_output("HDMI")


def test_no_default_device(fake_sounddevice) -> None:
    fake_sounddevice.default.device = [-1, -1]
    fake_sounddevice.query_devices = lambda device=None, kind=None: (
        (_ for _ in ()).throw(fake_sounddevice.PortAudioError("no default"))
        if kind
        else [dict(d) for d in fake_sounddevice.devices]
    )
    with pytest.raises(DeviceResolutionError, match="no system default input device"):
        resolve_input(LAPTOP)


def test_devices_report_shapes() -> None:
    report = devices_report(LAPTOP)
    assert report["active_profile"] == "laptop" and report["error"] is None
    assert report["resolved_input"] == {
        "index": 5,
        "name": "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)",
        "hostapi": "Windows WASAPI",
        "channels_used": [0, 1],
        "sample_rate": 48000,
    }
    assert {d["index"] for d in report["inputs"]} == {0, 1, 2, 5, 6}
    bad = devices_report(DeviceProfile(name="x", device="Zoom H6"))
    assert bad["resolved_input"] is None and "Zoom H6" in bad["error"]


@pytest.mark.parametrize(
    ("value", "expected"), [(None, None), ("5", 5), (" 12 ", 12), ("USB", "USB")]
)
def test_parse_device_argument(value, expected) -> None:
    assert parse_device_argument(value) == expected

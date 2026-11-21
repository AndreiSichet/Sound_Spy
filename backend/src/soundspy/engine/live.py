"""A live session in this process, without the server (step 08).

``soundspy measure-latency --through-app`` and ``soundspy soak`` run Sound Spy's real chain
(microphone -> chain -> speaker) exactly as a session started from the page: the same
``build_chain``, ``MicSource`` and ``SpeakerSink``, the user's output device
(``data/user_settings.json``), the same warm-up. Only the server around it is missing.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path

from soundspy.config import AppConfig
from soundspy.core.events import Event
from soundspy.core.target import TargetSpec
from soundspy.engine.factory import build_chain
from soundspy.engine.session import Session
from soundspy.io.devices import refresh_devices, resolve_input, resolve_output
from soundspy.io.file_source import FileSource
from soundspy.io.mic_source import MicSource
from soundspy.io.speaker_sink import SpeakerSink


def live_session(config: AppConfig, *, method: str | None = None, gate: bool | None = None,
                 labels: Collection[str] = (), attenuation_db: float | None = None,
                 bypass: bool = False, monitor: bool = True, muted: bool = False,
                 gain_db: float = 0.0, source: str | Path | None = None,
                 on_event: Callable[[Event], None] | None = None) -> Session:
    """A session (not started): the profile's microphone, or a WAV ``source`` played at
    real-time speed (looping); the configured output device. PortAudio re-reads the devices
    first (no stream may be open), as the server does before every start: handles from before
    a jack or USB change fail with "Unanticipated host error"."""
    refresh_devices()
    if source is None:
        src = MicSource(resolve_input(config.device_profile), config.capture)
    else:
        src = FileSource(Path(source), realtime=True, loop=True)
    output = SpeakerSink(resolve_output(config.output.device), config.output)
    session = Session(
        src, None,
        chain_factory=lambda: build_chain("semantic", config, config.labels, method=method,
                                          gate=gate),
        output=output, on_event=on_event or (lambda event: None), config=config)
    att = config.extraction.default_attenuation_db if attenuation_db is None else attenuation_db
    session.set_target(TargetSpec(labels=frozenset(labels), attenuation_db=att))
    session.set_bypass(bypass)
    if monitor:
        session.set_monitor(True, gain_db, muted)
    return session

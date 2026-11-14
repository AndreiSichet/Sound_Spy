"""``SessionManager``: owns at most one ``Session`` and bridges it to the API.

- Builds the source (microphone from the active profile, or a WAV file under the project root
  played at real-time speed), the chain (``build_chain``) and the speaker output, then starts the
  ``Session``.
- **Thread -> asyncio bridge.** The session calls ``on_event`` on its worker thread. That only
  schedules ``_publish`` on the API's event loop with ``loop.call_soon_threadsafe``; conversion to
  a WebSocket message and fan-out to clients happen on the loop, so the audio thread never waits
  for a network send.
- Remembers the target and the monitor settings between sessions, and stops the session on app
  shutdown.

Routes call these methods from FastAPI's thread pool, so a lock serialises start/stop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from soundspy.api.db.models import Recording
from soundspy.api.schemas import SessionStartRequest, SourceRequest, event_message
from soundspy.api.services.recordings import stored_path, sync_recordings
from soundspy.api.ws import ConnectionManager
from soundspy.config import AppConfig
from soundspy.core.events import EngineStatus, Event
from soundspy.core.labels import LabelSet
from soundspy.core.target import TargetSpec
from soundspy.engine.factory import ModeUnavailableError, build_chain
from soundspy.engine.session import Session, SessionError
from soundspy.io.base import AudioSink, AudioSource
from soundspy.io.devices import (
    AudioBackendUnavailable,
    DeviceResolutionError,
    refresh_devices,
    resolve_input,
    resolve_output,
)
from soundspy.io.file_source import FileSource
from soundspy.io.mic_source import MicSource
from soundspy.io.speaker_sink import SpeakerSink

logger = logging.getLogger("soundspy.api")


class ConflictError(RuntimeError):
    """The request conflicts with the current state (HTTP 409)."""


class BadRequestError(ValueError):
    """The request itself is invalid (HTTP 400)."""


class AudioOpenError(RuntimeError):
    """The audio device could not be opened even after re-scanning the devices (HTTP 503)."""


class SessionManager:
    def __init__(
        self,
        config: AppConfig,
        ws_manager: ConnectionManager,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.config = config
        self.ws_manager = ws_manager
        self.loop = loop
        self.session: Session | None = None
        self.target: TargetSpec | None = None
        self.monitor = (False, config.output.default_gain_db, False)
        self._lock = threading.RLock()

    @property
    def labels(self) -> LabelSet:
        return self.config.labels

    @property
    def running(self) -> bool:
        session = self.session
        return session is not None and session.running

    def status(self) -> EngineStatus:
        """The running session's status, or, when idle, the stored Listen/volume settings that
        the next session will start with (so the page shows what the user chose)."""
        session = self.session
        if session is not None and session.running:
            return session.status
        enabled, gain_db, muted = self.monitor
        if session is not None:  # a stopped/failed session: keep its figures, show the settings
            return dataclasses.replace(
                session.status, monitoring=enabled, gain_db=gain_db, muted=muted
            )
        return EngineStatus(
            t=0.0,
            state="idle",
            message="no active session",
            monitoring=enabled,
            gain_db=gain_db,
            muted=muted,
        )

    # --- building ------------------------------------------------------------------------------

    def _build_source(self, request: SourceRequest) -> AudioSource:
        if request.type == "mic":
            try:
                resolved = resolve_input(self.config.device_profile)
            except DeviceResolutionError as exc:
                raise BadRequestError(f"microphone: {exc}") from exc
            return MicSource(resolved, self.config.capture)
        if not request.path:
            raise BadRequestError("a file source needs 'path' (relative to the project root)")
        root = self.config.project_root.resolve()
        path = (root / request.path).resolve()
        if not path.is_relative_to(root):
            raise BadRequestError("file sources must be inside the project folder")
        if not path.is_file():
            raise BadRequestError(f"no such file: {request.path}")
        try:
            return FileSource(path, realtime=True, loop=request.loop)
        except Exception as exc:  # noqa: BLE001 - not a readable audio file
            raise BadRequestError(f"cannot read {request.path}: {exc}") from exc

    def _build_output(self) -> AudioSink | None:
        try:
            device = resolve_output(self.config.output.device)
        except (DeviceResolutionError, AudioBackendUnavailable) as exc:
            logger.warning("no output device for monitoring: %s", exc)
            return None
        return SpeakerSink(device, self.config.output)

    # --- session control -----------------------------------------------------------------------

    def _open_session(self, request: SessionStartRequest) -> Session:
        """Build source, chain and output and start a session (one attempt)."""
        try:
            chain = build_chain(request.mode, self.config, self.labels)
        except ModeUnavailableError as exc:
            raise BadRequestError(str(exc)) from exc
        source = self._build_source(request.source)
        session = Session(
            source,
            chain,
            output=self._build_output(),
            on_event=self._on_event,
            config=self.config,
            mode=str(request.mode).lower(),
        )
        if self.target is not None:
            session.set_target(self.target)
        session.start()
        return session

    def start(self, request: SessionStartRequest) -> EngineStatus:
        with self._lock:
            if self.running:
                raise ConflictError("a session is already running; stop it first")
            if self.session is not None:  # a finished/failed session: release it
                self.session.stop()
            # No stream is open now, so PortAudio can safely re-read the devices: headphones or
            # USB devices plugged in/out since the backend started would otherwise use stale
            # device handles.
            refresh_devices()
            try:
                session = self._open_session(request)
            except (BadRequestError, AudioBackendUnavailable):
                raise
            except Exception as first:  # noqa: BLE001 - PortAudio refusing to open a stream
                logger.warning("opening audio failed (%s); re-scanning devices, retrying", first)
                refresh_devices()
                try:
                    session = self._open_session(request)
                except (BadRequestError, AudioBackendUnavailable):
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise AudioOpenError(
                        f"could not open the audio device: {exc}. If headphones or a USB device "
                        "were just plugged in or out, or another app is using the microphone "
                        "exclusively, wait a moment and press Start again."
                    ) from exc
            self.session = session
            enabled, gain_db, muted = self.monitor
            enabled = enabled or request.monitor
            try:
                session.set_monitor(enabled, gain_db, muted)
            except SessionError as exc:
                logger.warning("monitoring not available: %s", exc)
            return session.status

    def stop(self, db: DbSession | None = None) -> EngineStatus:
        with self._lock:
            session = self.session
            if session is None:
                return self.status()
            session.stop()
            if session.last_recording is not None and db is not None:
                self._sync(db)
            return session.status

    def shutdown(self) -> None:
        with self._lock:
            if self.session is not None:
                self.session.stop()

    def set_target(self, target: TargetSpec) -> None:
        self.target = target
        session = self.session
        if session is not None and session.running:
            session.set_target(target)

    def set_monitor(self, enabled: bool, gain_db: float, muted: bool) -> EngineStatus:
        session = self.session
        if session is not None and session.running:
            try:
                session.set_monitor(enabled, gain_db, muted)
            except SessionError as exc:
                raise ConflictError(str(exc)) from exc
        self.monitor = (enabled, gain_db, muted)
        return self.status()

    def record_start(self, label: str | None, notes: str) -> EngineStatus:
        session = self.session
        if session is None or not session.running:
            raise ConflictError("start a session before recording")
        try:
            session.start_recording(self.config.paths["raw"], label=label, notes=notes)
        except SessionError as exc:
            raise ConflictError(str(exc)) from exc
        return session.status

    def record_stop(self, db: DbSession) -> Recording:
        session = self.session
        if session is None:
            raise ConflictError("not recording")
        try:
            path = session.stop_recording()
        except SessionError as exc:
            raise ConflictError(str(exc)) from exc
        return self._row_for(db, path)

    def _sync(self, db: DbSession) -> None:
        sync_recordings(db, self.config.paths["raw"], self.config.project_root)

    def _row_for(self, db: DbSession, path: Path) -> Recording:
        self._sync(db)
        stored = stored_path(path, self.config.project_root)
        row = db.scalars(select(Recording).where(Recording.path == stored)).one_or_none()
        if row is None:  # a recording outside data/raw would not be indexed
            raise ConflictError(f"recording saved to {path} but not indexed")
        return row

    # --- events: worker thread -> event loop ---------------------------------------------------

    def _on_event(self, event: Event) -> None:
        try:
            self.loop.call_soon_threadsafe(self._publish, event)
        except RuntimeError:  # the loop is closed (app shutting down)
            pass

    def _publish(self, event: Event) -> None:
        try:
            self.ws_manager.publish(event_message(event))
        except Exception:  # noqa: BLE001
            logger.exception("could not publish %s", type(event).__name__)

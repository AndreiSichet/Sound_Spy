"""``/ws/session``: the live channel.

- On connect the server queues the current ``status``. While no session runs, a ``status``
  heartbeat follows every ``api.ws_heartbeat_s``; while a session runs, the session itself emits
  ``status`` every ``engine.status_interval_s`` plus ``levels`` / ``waveform`` / ``spectrogram``
  at their analysis rates, and ``warning`` when something needs attention.
- Client -> server: ``set_target`` (validated against the class list, then stored and forwarded
  to the session) and ``set_bypass`` (``{"enabled": bool}``; A/B listening). Both are answered
  with the new ``status``; malformed or unknown messages with ``error``.
- ``run`` messages (an offline processing run changed state) come from the ``RunManager``.

**Slow clients.** Every connection has its own bounded queue and a sender task. When the queue
is full, the oldest *droppable* message (``levels``, ``waveform``, ``spectrogram``) is discarded;
``status``, ``run``, ``warning`` and ``error`` are never dropped. A browser tab in the background
therefore cannot make the server hoard memory or delay other clients.

Events reach this module from the session's worker thread through
``SessionManager`` (``loop.call_soon_threadsafe``), so ``publish`` always runs on the event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from soundspy.api.schemas import (
    CLIENT_MESSAGE_TYPES,
    DROPPABLE_TYPES,
    BypassRequest,
    TargetSpecModel,
    WsMessage,
    error_message,
    status_message,
)

if TYPE_CHECKING:
    from soundspy.api.services.session_manager import SessionManager

router = APIRouter()


class ClientQueue:
    """Bounded queue with the drop policy described in the module docstring."""

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self.items: deque[WsMessage] = deque()
        self.dropped = 0
        self._ready = asyncio.Event()

    def put(self, message: WsMessage) -> None:
        self.items.append(message)
        while len(self.items) > self.maxsize:
            for i, queued in enumerate(self.items):
                if queued.type in DROPPABLE_TYPES:
                    del self.items[i]
                    self.dropped += 1
                    break
            else:  # only messages that must not be dropped: keep them all
                break
        self._ready.set()

    async def get(self) -> WsMessage:
        while not self.items:
            self._ready.clear()
            await self._ready.wait()
        return self.items.popleft()


@dataclass
class _Client:
    websocket: WebSocket
    queue: ClientQueue
    task: asyncio.Task[None] | None = field(default=None)


class ConnectionManager:
    """Connected clients, each with its own queue and sender task."""

    def __init__(self, queue_size: int = 64) -> None:
        self.queue_size = queue_size
        self._clients: dict[WebSocket, _Client] = {}

    @property
    def count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        client = _Client(websocket, ClientQueue(self.queue_size))
        client.task = asyncio.create_task(self._sender(client))
        self._clients[websocket] = client

    def disconnect(self, websocket: WebSocket) -> None:
        client = self._clients.pop(websocket, None)
        if client is not None and client.task is not None:
            client.task.cancel()

    def send(self, websocket: WebSocket, message: WsMessage) -> None:
        client = self._clients.get(websocket)
        if client is not None:
            client.queue.put(message)

    def publish(self, message: WsMessage) -> None:
        for client in list(self._clients.values()):
            client.queue.put(message)

    async def _sender(self, client: _Client) -> None:
        try:
            while True:
                message = await client.queue.get()
                await client.websocket.send_text(message.model_dump_json())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the client went away; the receive loop cleans up
            self._clients.pop(client.websocket, None)


def handle_client_message(raw: str, sessions: SessionManager) -> WsMessage:
    """Validate a client message, act on it and return the reply."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return error_message("invalid JSON")
    msg_type = data.get("type") if isinstance(data, dict) else None
    if msg_type not in CLIENT_MESSAGE_TYPES:
        return error_message(f"unknown message type {msg_type!r}")
    try:
        message = WsMessage.model_validate(data)
        if msg_type == "set_bypass":
            sessions.set_bypass(BypassRequest.model_validate(message.payload).enabled)
            return status_message(sessions.status())
        target = TargetSpecModel.model_validate(message.payload).to_core()
        target.validate(sessions.labels)
    except ValidationError as exc:
        return error_message(f"invalid {msg_type} message: {exc.errors()[0]['msg']}")
    except ValueError as exc:
        return error_message(f"invalid {msg_type} message: {exc}")
    sessions.set_target(target)
    return status_message(sessions.status())


async def _heartbeat(
    manager: ConnectionManager, websocket: WebSocket, sessions: SessionManager, interval_s: float
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        if not sessions.running:  # a running session sends its own status
            manager.send(websocket, status_message(sessions.status()))


@router.websocket("/ws/session")
async def session_socket(websocket: WebSocket) -> None:
    manager: ConnectionManager = websocket.app.state.ws_manager
    sessions: SessionManager = websocket.app.state.session_manager
    interval_s: float = websocket.app.state.config.api.ws_heartbeat_s
    await manager.connect(websocket)
    manager.send(websocket, status_message(sessions.status()))
    heartbeat = asyncio.create_task(_heartbeat(manager, websocket, sessions, interval_s))
    try:
        while True:
            raw = await websocket.receive_text()
            manager.send(websocket, handle_client_message(raw, sessions))
    except WebSocketDisconnect:
        pass
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await heartbeat
        manager.disconnect(websocket)

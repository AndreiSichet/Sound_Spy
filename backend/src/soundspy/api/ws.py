"""``/ws/session``: live session channel.

On connect the server sends the current ``status``; afterwards a ``status`` heartbeat every
``api.ws_heartbeat_s`` seconds. Client messages use the same envelope (see ``schemas``);
unknown or malformed messages are answered with an ``error`` message.

From step 03 the engine worker thread pushes events into the event loop with
``loop.call_soon_threadsafe`` and they are broadcast through ``ConnectionManager``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from soundspy.api.schemas import (
    CLIENT_MESSAGE_TYPES,
    WsMessage,
    error_message,
    status_message,
)
from soundspy.core.events import EngineStatus

router = APIRouter()


class ConnectionManager:
    """Tracks connected WebSocket clients; serializes sends per connection."""

    def __init__(self) -> None:
        self._locks: dict[WebSocket, asyncio.Lock] = {}

    @property
    def count(self) -> int:
        return len(self._locks)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._locks[websocket] = asyncio.Lock()

    def disconnect(self, websocket: WebSocket) -> None:
        self._locks.pop(websocket, None)

    async def send(self, websocket: WebSocket, message: WsMessage) -> None:
        lock = self._locks.get(websocket)
        if lock is None:
            return
        async with lock:
            await websocket.send_text(message.model_dump_json())

    async def broadcast(self, message: WsMessage) -> None:
        for websocket in list(self._locks):
            try:
                await self.send(websocket, message)
            except Exception:  # noqa: BLE001 - a dead client must not break the broadcast
                self.disconnect(websocket)


def current_status() -> EngineStatus:
    """Engine status. No session exists before step 03, so the engine is always idle."""
    # TODO(step-03): read the status of the active Session.
    return EngineStatus(t=0.0, state="idle", message="no active session")


def handle_client_message(raw: str) -> WsMessage:
    """Validate a client message and return the reply."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return error_message("invalid JSON")
    msg_type = data.get("type") if isinstance(data, dict) else None
    if msg_type not in CLIENT_MESSAGE_TYPES:
        return error_message(f"unknown message type {msg_type!r}")
    try:
        WsMessage.model_validate(data)
    except ValidationError as exc:
        return error_message(f"invalid {msg_type} message: {exc.errors()[0]['msg']}")
    # TODO(step-03): apply set_target to the active session.
    return error_message(f"{msg_type} not implemented (step-03)")


async def _heartbeat(manager: ConnectionManager, websocket: WebSocket, interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        await manager.send(websocket, status_message(current_status()))


@router.websocket("/ws/session")
async def session_socket(websocket: WebSocket) -> None:
    manager: ConnectionManager = websocket.app.state.ws_manager
    interval_s: float = websocket.app.state.config.api.ws_heartbeat_s
    await manager.connect(websocket)
    heartbeat: asyncio.Task[None] | None = None
    try:
        await manager.send(websocket, status_message(current_status()))
        heartbeat = asyncio.create_task(_heartbeat(manager, websocket, interval_s))
        while True:
            raw = await websocket.receive_text()
            await manager.send(websocket, handle_client_message(raw))
    except WebSocketDisconnect:
        pass
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await heartbeat
        manager.disconnect(websocket)

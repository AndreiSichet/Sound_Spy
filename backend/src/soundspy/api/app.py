"""FastAPI application factory.

``create_app()`` loads the config (unless one is passed in, e.g. by tests), sets up CORS for the
Vite dev origin, the WebSocket connection manager and the database (created in the lifespan).
Run it with ``soundspy serve`` or ``uvicorn soundspy.api.app:create_app --factory``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from soundspy import __version__
from soundspy.api import ws
from soundspy.api.db.base import init_db, make_engine, make_session_factory
from soundspy.api.routes import devices, recordings, session, system
from soundspy.config import AppConfig, load_config


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(config.database_url, config.project_root)
        init_db(engine)
        app.state.db_engine = engine
        app.state.session_factory = make_session_factory(engine)
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title="Sound Spy API",
        version=__version__,
        description="Backend of Sound Spy: devices, sessions, recordings, live events.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.ws_manager = ws.ConnectionManager()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.api.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (system.router, devices.router, session.router, recordings.router, ws.router):
        app.include_router(router)
    return app

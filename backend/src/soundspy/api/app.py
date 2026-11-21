"""FastAPI application factory.

``create_app()`` loads the config (unless one is passed in, e.g. by tests), sets up CORS for the
Vite dev origin, the WebSocket connection manager, the database (an index of the sidecar files,
rebuilt when its schema version is not the code's), the ``SessionManager`` and the ``RunManager``
(created in the lifespan, where the event loop they bridge to is known; the lifespan's exit stops
any running session and the offline worker). Step 08: the lifespan also starts the
``ModelPreloader`` (``api.preload_models``) in a background thread once the API is set up.
Run it with ``soundspy serve`` or ``uvicorn soundspy.api.app:create_app --factory``.

Step 08: the user's own settings (``data/user_settings.json``: the output device) are applied on
top of the configuration; ``create_app(frontend=dir)`` also serves the built page at ``/``
(``soundspy app``, via :func:`create_app_with_frontend`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from soundspy import __version__
from soundspy.api import ws
from soundspy.api.db.base import init_db, make_engine, make_session_factory
from soundspy.api.routes import devices, recordings, runs, session, settings, system
from soundspy.api.services.index import sync_all
from soundspy.api.services.runs import RunManager
from soundspy.api.services.session_manager import SessionManager
from soundspy.config import AppConfig, configure_cache_env, load_config
from soundspy.extraction.offline import runs_dir
from soundspy.io.runs import interrupt_unfinished
from soundspy.user_settings import apply_user_settings

logger = logging.getLogger("soundspy.api")


def create_app(config: AppConfig | None = None, *, frontend=None) -> FastAPI:
    """The API; ``frontend``: a folder with the built page, served at ``/`` (step 08)."""
    config = apply_user_settings(config or load_config())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_cache_env(config)  # model caches on D: unless the user set them
        engine = make_engine(config.database_url, config.project_root)
        if init_db(engine):  # new database, or its schema version differs from the code's
            logger.info("database (re)built: it is an index of the sidecar files, re-reading them")
        app.state.db_engine = engine
        app.state.session_factory = make_session_factory(engine)
        interrupted = interrupt_unfinished(runs_dir(config))  # jobs a previous process left
        with app.state.session_factory() as db:
            result = sync_all(db, config)
        logger.info("index: recordings +%d -%d (%d missing), runs +%d -%d (%d interrupted)",
                    result.recordings.added, result.recordings.removed, result.recordings.missing,
                    result.runs.added, result.runs.removed, interrupted)
        loop = asyncio.get_running_loop()
        app.state.session_manager = SessionManager(config, app.state.ws_manager, loop)
        app.state.preloader = None
        if config.api.preload_models:
            from soundspy.engine.preload import ModelPreloader

            preloader = ModelPreloader(config)
            app.state.preloader = app.state.session_manager.preloader = preloader
            preloader.start()  # in the background: the API answers meanwhile
        app.state.run_manager = RunManager(config, app.state.session_factory,
                                           app.state.ws_manager, loop)
        try:
            yield
        finally:
            app.state.session_manager.shutdown()  # no audio thread outlives the app
            app.state.run_manager.shutdown()
            engine.dispose()

    app = FastAPI(
        title="Sound Spy API",
        version=__version__,
        description="Backend of Sound Spy: devices, sessions, recordings, live events.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.ws_manager = ws.ConnectionManager(config.api.ws_client_queue)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.api.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (system.router, devices.router, session.router, recordings.router,
                   runs.router, settings.router, ws.router):
        app.include_router(router)
    if frontend is not None:
        from soundspy.api.static import mount_frontend

        mount_frontend(app, frontend)  # after the API routes: they keep precedence
    return app


def create_app_with_frontend() -> FastAPI:
    """``soundspy app``'s factory: the API plus ``frontend/dist`` at ``/``."""
    from soundspy.api.static import frontend_dir

    config = load_config()
    return create_app(config, frontend=frontend_dir(config.project_root))

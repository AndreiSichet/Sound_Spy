"""``soundspy serve``: run the FastAPI backend with uvicorn.

Host and port come from ``configs/default.yaml`` (``api.host`` / ``api.port``) unless
overridden on the command line. Needs the ``api`` extra (``pip install -e ".[api]"``).
"""

from __future__ import annotations

import argparse

NAME = "serve"
HELP = "run the API server (FastAPI + uvicorn)"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("--host", default=None, help="bind address (default: api.host from config)")
    parser.add_argument("--port", type=int, default=None, help="port (default: api.port)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    # Imported lazily so the rest of the CLI works without the `api` extra installed.
    import uvicorn

    from soundspy.config import load_config

    api = load_config().api
    uvicorn.run(
        "soundspy.api.app:create_app",
        factory=True,
        host=args.host or api.host,
        port=args.port or api.port,
        reload=args.reload,
        log_config=log_config(),
    )
    return 0


def log_config() -> dict:
    """uvicorn's logging setup plus Sound Spy's own loggers (``soundspy.*``) on the console.

    Without this, warnings such as "output unavailable: ..." or a failing session worker would
    never appear in the ``soundspy serve`` window.
    """
    import copy

    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    config["loggers"]["soundspy"] = {"handlers": ["default"], "level": "INFO", "propagate": False}
    return config

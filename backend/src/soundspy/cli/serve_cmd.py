"""``soundspy serve``: run the FastAPI backend with uvicorn.

Host and port come from ``configs/default.yaml`` (``api.host`` / ``api.port``) unless
overridden on the command line. Needs the ``api`` extra (``pip install -e ".[api]"``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

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

    config = load_config()
    api = config.api
    uvicorn.run(
        "soundspy.api.app:create_app",
        factory=True,
        host=args.host or api.host,
        port=args.port or api.port,
        reload=args.reload,
        log_config=log_config(config),
    )
    return 0


LOG_FILE_BYTES = 1_000_000
LOG_FILE_BACKUPS = 5


def log_path(config) -> Path:
    """``data/logs/soundspy.log`` (git-ignored with the rest of ``data/``)."""
    return Path(config.paths["raw"]).parent / "logs" / "soundspy.log"


def log_config(config=None) -> dict:
    """uvicorn's logging setup plus Sound Spy's own loggers (``soundspy.*``) on the console,
    and (step 08, with ``config``) everything also in a rotating file ``data/logs/soundspy.log``
    (1 MB x 5) with times and logger names: start-up lines (the index, the preloaded models),
    warnings and a failing session worker survive the console window.
    """
    import copy

    from uvicorn.config import LOGGING_CONFIG

    cfg = copy.deepcopy(LOGGING_CONFIG)
    handlers = ["default"]
    if config is not None:
        path = log_path(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg["formatters"]["file"] = {"format": "%(asctime)s %(levelname)-7s %(name)s: %(message)s"}
        cfg["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler", "formatter": "file",
            "filename": str(path), "maxBytes": LOG_FILE_BYTES, "backupCount": LOG_FILE_BACKUPS,
            "encoding": "utf-8"}
        handlers.append("file")
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logger = cfg["loggers"].setdefault(name, {})
            logger["handlers"] = [*logger.get("handlers", []), "file"]
    cfg["loggers"]["soundspy"] = {"handlers": handlers, "level": "INFO", "propagate": False}
    return cfg

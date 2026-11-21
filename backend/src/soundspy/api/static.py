"""The built page served by the backend itself (step 08, ``soundspy app``).

``frontend/dist`` (made by ``npm run build``) is mounted at ``/`` after every API route, so
``/api/*`` and ``/ws/*`` keep working and everything else is a file of the page. The page is a
single-page application: an address that is not a file (a reloaded tab, a bookmark) gets
``index.html`` so the page itself decides what to show (**SPA fallback**); unknown ``/api`` paths
stay real 404s.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

BUILD_HINT = ("the page is not built yet: run\n"
              "    cd frontend; npm install; npm run build\n"
              "once (and again after changing the page), then start the app again")


class SPAStaticFiles(StaticFiles):
    """Static files with ``index.html`` for every path that is not a file (and not the API)."""

    async def get_response(self, path: str, scope: Scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            first = path.replace("\\", "/").split("/", 1)[0]  # Windows: "api\x"
            if exc.status_code != 404 or first in ("api", "ws"):
                raise
            return await super().get_response("index.html", scope)


def frontend_dir(project_root: Path) -> Path:
    return Path(project_root) / "frontend" / "dist"


def mount_frontend(app: FastAPI, directory: Path) -> None:
    """Serve ``directory`` (the built page) at ``/``; ``FileNotFoundError`` with the build
    command when it is missing."""
    directory = Path(directory)
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(f"{directory / 'index.html'} not found: {BUILD_HINT}")
    app.mount("/", SPAStaticFiles(directory=str(directory), html=True), name="frontend")

"""``soundspy app``: the whole app with one command (step 08).

Serves the **pre-built** page (``frontend/dist``) and the API from one address,
``http://127.0.0.1:8765/``, preloads the models in the background (as ``soundspy serve`` does)
and opens the default browser once the server answers. It builds nothing: if
``frontend/dist`` is missing it prints the build command and stops. ``start.ps1`` at the project
root runs this command from the project's virtual environment. Development stays as it was:
``soundspy serve`` + ``npm run dev`` (``localhost:5173``, hot reload).
"""

from __future__ import annotations

import argparse
import threading
import time
import urllib.request

NAME = "app"
HELP = "run the whole app (API + the built page) and open it in the browser"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("--host", default=None, help="bind address (default: api.host)")
    parser.add_argument("--port", type=int, default=None, help="port (default: api.port)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.set_defaults(handler=run)


def open_when_up(url: str, timeout_s: float = 60.0) -> None:
    """Open ``url`` in the default browser as soon as ``/api/health`` answers."""
    import webbrowser

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "api/health", timeout=1.0) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            pass
        time.sleep(0.3)


def run(args: argparse.Namespace) -> int:
    import uvicorn

    from soundspy.api.static import BUILD_HINT, frontend_dir
    from soundspy.cli.serve_cmd import log_config
    from soundspy.config import load_config

    config = load_config()
    page = frontend_dir(config.project_root)
    if not (page / "index.html").is_file():
        print(f"error: {page} has no index.html - {BUILD_HINT}")
        return 1
    host, port = args.host or config.api.host, args.port or config.api.port
    url = f"http://{host}:{port}/"
    print(f"Sound Spy: {url}  (Ctrl+C to stop; models load in the background)")
    if not args.no_browser:
        threading.Thread(target=open_when_up, args=(url,), daemon=True).start()
    uvicorn.run("soundspy.api.app:create_app_with_frontend", factory=True, host=host, port=port,
                log_config=log_config(config))
    return 0

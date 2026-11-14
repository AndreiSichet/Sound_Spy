# Sound Spy

University project (Telecommunications, UTCN). Sound Spy is a desktop-local web app that
listens to the environment through a microphone and lets you **choose what to hear**: the chosen
sound is enhanced, everything else is treated as noise and attenuated. The enhanced audio plays
through the laptop's headphones/speakers.

Target scenario: a crowded cafeteria — "let me hear speech, suppress dishes and music", or
"let me hear the person on my left".

| Tab | You select... | The app... | Hardware | Status |
|---|---|---|---|---|
| **Semantic** | one or more sound types | detects the sound types present, extracts the selected ones | any mic | shell |
| **Spatial** | sound types + a direction | steers toward the direction, then extracts | USB mic array | planned (module 3-4) |
| **Sources** | one detected source on a map | tracks sources, extracts the selected one | USB mic array | planned (module 5) |

## Architecture in short

- **Backend** (`backend/`, Python 3.11, package `soundspy`): owns all audio I/O. A streaming
  engine runs chunk-based `Processor`s (spatial filter, extractor) and `SideProcessor`s
  (detector, analyzer, localizer) on a worker thread. FastAPI exposes REST (`/api/*`) and a
  WebSocket (`/ws/session`) for live events; SQLite stores recordings and processing runs.
- **Frontend** (`frontend/`, React + TypeScript + Vite + Tailwind + Zustand): tabs, controls and
  visualizations. It never captures the microphone or plays live audio.
- Extraction: detector-guided spectral-masking baseline, then a label-conditioned neural target
  sound extraction model (ideas from SemanticHearing, Veluri et al., UIST 2023).

New to the project? Start with [project_insights.md](project_insights.md): a plain-language guide
to how everything works, with a glossary.
Details: [docs/architecture.md](docs/architecture.md) · decisions: [docs/decisions.md](docs/decisions.md)

## Requirements

- Windows 11, Python **3.11** (the default `python` may be newer, e.g. 3.14, which is not
  supported — use the `py -3.11` launcher), Node.js 22+ LTS.
- Ports 8765 (backend, bound to 127.0.0.1) and 5173 (frontend dev server) free.

## Run

```powershell
# backend
cd backend
py -3.11 -m venv .venv; .venv\Scripts\activate
pip install -e ".[api,dev]"
soundspy serve            # http://127.0.0.1:8765/docs

# frontend (second terminal)
cd frontend
npm install
npm run dev               # http://localhost:5173
```

The Vite dev server proxies `/api` and `/ws` to the backend, so open http://localhost:5173.

### Microphone

```powershell
soundspy devices                              # which mic the active profile uses
soundspy check-mic                            # 5 s test: levels, channel comparison, warnings
soundspy check-mic --meter                    # live level meter (Ctrl+C to stop)
soundspy record --seconds 10 --label speech   # WAV + JSON sidecar in data/raw
```

Before recording anything you want to keep, switch **off** Windows audio enhancements /
Voice Clarity for the microphone (Settings → System → Sound → your microphone → Properties) and
re-run `check-mic`. [project_insights.md](project_insights.md) ("Audio capture") explains how to
read the output.

### Machine-learning extras (from step 05)

Install PyTorch **first** from the official CUDA wheel index — pick the command for your CUDA
version at <https://pytorch.org/get-started/locally/>, for example:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[api,ml,dev]"
```

Installing `.[ml]` without this first step pulls the CPU-only torch wheel from PyPI.
Spatial extras (module 3): `pip install -e ".[spatial]"` (pyroomacoustics).

## Development

```powershell
# backend (in backend/, venv active)
pytest                 # no microphone, GPU or datasets needed
pytest --run-gpu       # include @pytest.mark.gpu tests
pytest --run-data      # include @pytest.mark.data tests
ruff check .
soundspy --help        # dev/data/training CLI

# frontend (in frontend/)
npm run test
npm run lint
npm run typecheck
npm run build
```

## Layout

- `backend/` — `pyproject.toml`, `configs/` (global, classes, device profiles, detection,
  extraction), `src/soundspy/`, `tests/`
- `frontend/` — Vite app (`src/`, `tests/`)
- `docs/` — architecture, decision log, literature
- `experiments/` — one folder per training run (config tracked, checkpoints ignored)
- `data/`, `models/`, `outputs/` — git-ignored working data (`data/soundspy.db` is the app database)
- `notebooks/` — exploration only
- `project_insights.md` — how the app works, in plain language (including any adapted
  third-party code, in its "Third-party code" section)

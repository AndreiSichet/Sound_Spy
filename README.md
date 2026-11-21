# Sound Spy

University project (Telecommunications, UTCN). Sound Spy is a desktop-local web app that
listens through the laptop microphone and lets you **choose what to hear**: the chosen sound
types are kept, everything else is turned down. The result plays on your headphones, live.

| Tab | You select... | The app... | Hardware | Status |
|---|---|---|---|---|
| **Semantic** | one or more sound types | detects what is there and keeps only the selected types | the laptop microphone | **done** (steps 01-08) |
| **Spatial** | sound types + a direction | steers toward the direction, then extracts | USB microphone array | planned (modules 3-4) |
| **Sources** | one detected source on a map | tracks sources, extracts the selected one | USB microphone array | planned (module 5) |

New to the project? [project_insights.md](project_insights.md) explains everything in plain
language. Results for the report: [docs/evaluation.md](docs/evaluation.md) · demo script:
[docs/demo.md](docs/demo.md) · architecture: [docs/architecture.md](docs/architecture.md) ·
decisions: [docs/decisions.md](docs/decisions.md).

## Quick start

Once installed (next section), start the whole app with one command from the project folder:

```powershell
.\start.ps1
```

It starts the backend, serves the page at **http://127.0.0.1:8765/** and opens it in your
browser; the models load in the background (about 10 s after a fresh start). Stop it with
**Ctrl+C** in that window. If PowerShell refuses to run scripts:
`powershell -ExecutionPolicy Bypass -File .\start.ps1` (no admin rights needed).

On the **Semantic** tab: press **Start**, put on **headphones**, tick **Listen**, and click the
sound types you want to keep (Speech, Music, ...). **Space** switches Bypass (the unprocessed
sound) on and off, **M** mutes.

## First-time setup

Windows 11, Python **3.11** (use the `py -3.11` launcher: a newer default `python` is not
supported), Node.js 22+ LTS, an NVIDIA graphics card for the Neural / Hybrid methods.

```powershell
# backend: PyTorch first from the CUDA wheel index, then the project (see backend/requirements-lock.txt)
cd backend
py -3.11 -m venv .venv
.venv\Scripts\pip install "torch==2.11.0+cu126" "torchaudio==2.11.0+cu126" --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -e ".[api,data,ml,spatial,dev]"
.venv\Scripts\soundspy data download deepfilternet   # the speech enhancer (8 MB, to D:)

# frontend: build the page once (and after changing it)
cd ..\frontend
npm install
npm run build
```

The trained models (`experiments/det_mn10_v1`, `experiments/tse_waveformer_v1`) are produced by
the training commands below; without them the app still runs with the Baseline method and says
which parts are missing. `backend/requirements-lock.txt` lists the exact versions of a working
environment (`pip install -r requirements-lock.txt` after the torch line).

## Development

```powershell
# backend (backend/, .venv\Scripts\activate)
soundspy serve            # API only, http://127.0.0.1:8765/docs (reload: --reload)
pytest                    # no microphone, GPU or datasets needed; slow tests: pytest -m slow
ruff check .

# frontend (frontend/, a second terminal)
npm run dev               # http://localhost:5173, proxies /api and /ws to the backend
npm run test; npm run lint; npm run typecheck; npm run build
```

Run the frontend checks from PowerShell (Git Bash confuses Vitest). The page in development
(`npm run dev`) and the built page (`soundspy app` / `start.ps1`) talk to the same backend.

## Commands reference

| Command | What it does |
|---|---|
| `soundspy app` | the whole app: API + the built page + model preloading, opens the browser (`start.ps1`) |
| `soundspy serve` | the API alone (development, with `npm run dev`) |
| `soundspy devices` · `check-mic [--meter]` | which microphone the profile uses; a 5 s level check |
| `soundspy record --seconds 10 --label x` | record into `data/raw` (WAV + JSON sidecar) |
| `soundspy measure-latency [--all]` | the real delay, measured acoustically (laptop speakers, headphones off, low volume) |
| `soundspy soak --minutes 30` | a long muted live session: underruns, RTF, delay, RAM, VRAM every minute |
| `soundspy extract <wav> --labels speech [--method hybrid]` | process a recording offline into `outputs/processed` |
| `soundspy detect <wav>` | which sound types are in a recording, and when |
| `soundspy record-benchmark --pop` | record the real-room benchmark (speech, music, a song, room tone) |
| `soundspy evaluate detection \| extraction --method hybrid \| speech-gate \| hybrid-gate` | the evaluations (`--benchmark real --session A[+B]` for the real room) |
| `soundspy report [--rerun]` | regenerate `docs/evaluation.md` and `docs/figures/` from the stored results |
| `soundspy data paths \| download \| prepare \| stats \| mix \| preview \| cache` | training data on `D:/SoundSpyData` |
| `soundspy train detection \| extraction [--smoke \| --resume]` | training |
| `soundspy db rebuild` | rebuild the app database from the sidecar files |

Training data and models (large files go to **drive D:**, `paths` in `backend/configs/default.yaml`;
FSD50K needs 7-Zip: `winget install 7zip.7zip`):

```powershell
soundspy data download ontology esc50 fsd50k librispeech; soundspy data prepare
soundspy data mix --split all; soundspy data cache --task extraction --split all
soundspy train detection; soundspy train extraction       # hours each; --resume continues
soundspy evaluate extraction --method hybrid              # about an hour, the graphics card busy
```

ESC-50 is CC BY-NC; FSD50K clips carry per-clip CC licences, some non-commercial; LibriSpeech is
CC BY 4.0. Fine for this university project; review before any public or commercial use.

## Troubleshooting

- **Quiet, choppy or "pumping" microphone, or digital silence**: switch **off** Windows audio
  enhancements / Voice Clarity for the microphone (Settings > System > Sound > the microphone >
  Properties), then `soundspy check-mic`.
- **Howling**: Listen on with laptop speakers feeds the sound back into the microphone. Use
  headphones (the page asks once).
- **"Unanticipated host error" / headphones plugged in later do not work**: press **Rescan
  devices** on the page (or restart the session); PortAudio keeps old device handles otherwise.
- **The session stops with "the output device stopped playing" or "no audio from ..."**: the
  headphones or the microphone were unplugged; plug them back in and press **Restart**.
- **Port 8765 in use**: another Sound Spy is still running (close its window), or start with
  `soundspy app --port 8766`. Port 8000 is taken by Docker/WSL on the development laptop, hence
  8765.
- **"the page is not built yet"**: `cd frontend; npm install; npm run build`.
- **Running tests next to an evaluation**: hide the graphics card from the tests with
  `$env:CUDA_VISIBLE_DEVICES='-1'` (an empty value does **not** hide it on Windows); never run
  PyTorch tests next to a training run.
- **`measure-latency` finds no clicks**: unplug the headphones, raise the volume a little, check
  that the microphone is not muted.
- **Logs**: `data/logs/soundspy.log` (rotating, 1 MB x 5) keeps the server's lines.

## Architecture in short

- **Backend** (`backend/`, Python 3.11, package `soundspy`): owns all audio I/O. A streaming
  engine runs chunk-based `Processor`s (channel reducer, extractor) and `SideProcessor`s
  (detector, analyzers) on a worker thread; FastAPI exposes REST (`/api/*`) and a WebSocket
  (`/ws/session`) for live events, and serves the built page (`soundspy app`); SQLite indexes
  the recordings and processing runs (the JSON sidecars are the truth).
- **Frontend** (`frontend/`, React + TypeScript + Vite + Tailwind + Zustand): tabs, controls and
  visualizations. It never captures the microphone or plays live audio.
- **Extraction**: *Hybrid* (default): Speech alone through the speech enhancer DeepFilterNet3
  with its own gate, every other selection through the fine-tuned Waveformer with a gate the
  detector closes; *Neural*: the Waveformer alone; *Baseline*: a detector-gated Wiener filter.
  Detection: EfficientAT mn10 fine-tuned on the project's mixtures. Ideas from SemanticHearing
  (Veluri et al., UIST 2023).

## Layout

- `start.ps1` — one-command start (`soundspy app`)
- `backend/` — `pyproject.toml`, `requirements-lock.txt`, `configs/` (global, classes, device
  profiles, detection, extraction), `src/soundspy/`, `tests/`
- `frontend/` — Vite app (`src/`, `tests/`; `dist/` = the built page, git-ignored)
- `docs/` — evaluation (generated), demo script, architecture, decision log, figures
- `experiments/` — one folder per training or evaluation run (results tracked, checkpoints and
  logs ignored); `experiments/realtime/` = latency, soak, start-up measurements
- `data/`, `models/`, `outputs/` — git-ignored working data (`data/soundspy.db` the database,
  `data/raw` own recordings, `data/benchmark` the real-room sessions, `data/user_settings.json`
  the page's stored choices, `data/logs/` the server log)
- `D:/SoundSpyData/` — datasets, clip index, mixture recipes, model caches (outside the repository)
- `project_insights.md` — how the app works, in plain language (with the third-party code)

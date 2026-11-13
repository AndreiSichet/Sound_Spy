# SoundFocus

University project (Telecommunications, UTCN). SoundFocus listens to the environment through a
microphone, identifies the sound sources (speech, music, dishes, footsteps, ...), estimates where
they are, lets the user select one source and enhances that source while suppressing everything
else.

Target scenario: a crowded cafeteria — select the person on your left, enhance their voice and
suppress chatter, music, dishes and chairs.

## Module status

| Module | Content | Status |
|---|---|---|
| 1 | Audio acquisition (mic capture, recording, dataset recording tools) | scaffold |
| 2 | Sound classification (fine-tuned pretrained model) + class-based enhancement | scaffold |
| 3 | Sound source localization (ReSpeaker XVF3800 USB mic array) | planned |
| 4 | Spatial UI for selecting a source | planned |
| 5 | Target extraction toward the selected source (beamforming + separation) | planned |

Iteration 1 is record-then-process; real-time streaming comes later.

## Setup (Windows 11, PowerShell)

Requires Python **3.11** (the machine's default `python` may be newer, e.g. 3.14, which is not
supported). Create the environment explicitly with the `py` launcher:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Core + development tools:

```powershell
pip install -e ".[dev]"
```

For the machine-learning parts (modules 2+) install **PyTorch first** from the official CUDA
wheel index — pick the command for your CUDA version at <https://pytorch.org/get-started/locally/>,
for example:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[ml,dev]"
```

Installing `.[ml]` without this first step pulls the CPU-only torch wheel from PyPI.

## Usage

```powershell
soundfocus --help
soundfocus devices          # each subcommand is a placeholder until its step is implemented
```

## Development

```powershell
pytest                 # smoke tests; no microphone needed
pytest --run-gpu       # also run tests marked @pytest.mark.gpu
pytest --run-data      # also run tests marked @pytest.mark.data
ruff check .
```

## Layout

- `src/soundfocus/` — the package (`io`, `dsp`, `data`, `classification`, `enhancement`,
  `localization`, `pipeline`, `cli`)
- `configs/` — global config, device profiles, classifier and enhancement configs
- `tests/` — pytest suite (hardware is faked)
- `docs/` — architecture, decision log, literature notes
- `specs/` / `reports/` — per-step specs and build reports
- `data/`, `models/`, `outputs/` — git-ignored working data

Project rules: [CLAUDE.md](CLAUDE.md). Architecture: [docs/architecture.md](docs/architecture.md).

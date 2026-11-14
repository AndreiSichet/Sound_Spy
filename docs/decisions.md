# Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-13 | Record-then-process first; real-time streaming later. | Simpler to develop, debug and evaluate offline. (Refined 2026-11-14: one chunk-based engine serves both.) |
| 2026-11-13 | Laptop built-in mic for the Semantic tab; ReSpeaker XVF3800 4-mic USB array from module 3 onward. | Sound-type extraction needs no array; localization and beamforming do. Device profiles make the switch a config change. |
| 2026-11-13 | Fine-tune a pretrained audio model instead of training from scratch. | Limited labelled data and compute; pretrained audio representations transfer well. |
| 2026-11-13 | Extraction: classical detector-guided spectral masking baseline first, neural extraction later. | A transparent baseline gives reference metrics before adding neural complexity. |
| 2026-11-13 | Python + PyTorch; Windows 11 + NVIDIA GPU. | Available hardware and the de-facto ecosystem for audio deep learning. |
| 2026-11-14 | Product renamed **Sound Spy**; package and CLI `soundfocus` → `soundspy`; monorepo with `backend/` and `frontend/`. | The app becomes a desktop-local web app with three tabs (Semantic, Spatial, Sources). |
| 2026-11-14 | **The backend owns all audio I/O.** The browser never captures the mic and does not play live audio; it only plays saved files (A/B comparison). | Browsers apply echo cancellation / noise suppression / AGC and handle multichannel arrays poorly; live output goes straight to the laptop's output device. |
| 2026-11-14 | **One streaming engine for offline and live.** Every stage is a stateful, chunk-based `Processor` / `SideProcessor`; offline = feeding a file chunk by chunk. | Real-time later needs no redesign; one code path to test. |
| 2026-11-14 | **One `TargetSpec` for all three tabs** (`labels`, `direction`, `source_id`, `attenuation_db`). | Processors read only the fields they need; tabs differ only in what they set. |
| 2026-11-14 | **The Python library is the core; the API is a thin layer.** `api/` imports the library, never the reverse; the CLI stays for development, data and training jobs. | Library stays testable and usable without a server. Enforced by a test. |
| 2026-11-14 | **FastAPI** for REST + WebSocket, served by uvicorn. | Async WebSocket support, Pydantic schemas, automatic OpenAPI docs. |
| 2026-11-14 | **SQLite via SQLAlchemy 2.x** for recordings and processing runs; audio stays on disk. No migrations tool yet — **add Alembic when the schema starts changing.** | Zero-setup local database; switching to PostgreSQL later = changing the URL. |
| 2026-11-14 | **No Docker for now**; native dev with `uvicorn` + `npm run dev`. | Microphone and GPU access from containers on Windows is impractical. |
| 2026-11-14 | **Frontend:** React + TypeScript (strict) + Vite + Tailwind CSS + Zustand; Canvas for live waveform/spectrogram/meters, D3 for the polar source map, Recharts for metric charts. | Mainstream, fast tooling; Canvas handles high-rate live data better than SVG/DOM. |
| 2026-11-14 | **Classes defined in `configs/classes.yaml` and mapped to the AudioSet ontology** (SemanticHearing style), foreground = selectable classes, background = everything else. | Lets datasets with different label vocabularies (ESC-50, FSD50K, own recordings) feed one label set. |
| 2026-11-14 | **Extraction has two implementations behind one `Extractor` interface:** detector-guided masking baseline and a label-conditioned neural TSE model (Waveformer-style). | Baseline for comparison; neural model for quality. Ideas from SemanticHearing (MIT); no code copied so far. |
| 2026-11-14 | Frontend uses **TypeScript 6.0**, not 7.0. | typescript-eslint 8.x supports TypeScript < 6.1; TypeScript 7 (native port) is not supported yet. |
| 2026-11-14 | Backend on **`127.0.0.1:8765`** (was 8000); the Vite dev proxy targets `http://127.0.0.1:8765`, not `localhost`. | On the dev machine Docker Desktop and WSL already listen on port 8000 (`localhost` → `::1:8000` hits the WSL relay). An unused port + explicit IPv4 avoids both. |
| 2026-11-14 | Each class has an internal ID (the YAML key), a `display_name`, a `description` and `audioset` names; the API sends only ID, display name and description. | AudioSet names are for mapping datasets, not for users. The UI text can be renamed without touching trained models or the database. |
| 2026-11-14 | No `reports/` folder: each step's report is given in chat, and `CLAUDE.md` keeps the rules, a short current-state snapshot (overwritten) and a 2–3 line step log. | Claude Code reads `CLAUDE.md` in full every session; a long changelog would bury the rules. |

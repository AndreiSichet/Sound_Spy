"""``ModelPreloader``: load and warm the live models once, when the server starts (step 08).

Before step 08 the first session after a server start loaded every model itself (the Waveformer
checkpoint, DeepFilterNet3 with its ONNX export check, the detector) and set up CUDA, about
5-12 s on the laptop, and the first chunks met freshly started models. Now the server starts a
background thread right after the API is up; it

1. loads the **Waveformer** core and runs a few model steps (CUDA's process-wide set-up, the
   traced step graphs, cuDNN's kernel choice: all cached in the process),
2. loads **DeepFilterNet3** (ONNX Runtime session, checked against PyTorch) and runs it,
3. loads the **detector** on the device the live chain will give it (the CPU when the
   Waveformer uses the graphics card) and runs one window,

through the same cached loaders the chain uses, so a session's ``build_chain`` finds them
ready. What is per thread (PyTorch's thread count, a thread's first CUDA call) is still warmed by
the session's own ``chain.start()`` on its worker, now in milliseconds. A session started while
the preloader works waits for it in state ``"warming"`` (``wait``, with progress messages).

A step that fails is recorded and skipped: the session then reports its own fallback (e.g. no
speech enhancer) exactly as without preloading.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soundspy.config import AppConfig
from soundspy.engine import resources

log = logging.getLogger(__name__)


@dataclass
class Step:
    name: str  # shown to the user ("the speech enhancer")
    run: Callable[[], None]
    seconds: float | None = None
    error: str | None = None


class ModelPreloader:
    def __init__(self, config: AppConfig, method: str | None = None) -> None:
        self.config = config
        self.method = method or config.extraction.method
        self.state = "idle"  # idle | loading | ready | error
        self.current: str | None = None
        self.started_at: float | None = None
        self.seconds: float | None = None
        self.memory: dict[str, Any] | None = None
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._wf_device: str | None = None
        self.steps = self._plan()

    # --- the plan ------------------------------------------------------------------------------

    def _plan(self) -> list[Step]:
        cfg = self.config
        ext = cfg.extraction
        steps: list[Step] = []
        if self.method in ("hybrid", "neural_tse") and ext.neural_tse.checkpoint is not None:
            steps.append(Step("the Waveformer", self._waveformer))
        if self.method == "hybrid":
            steps.append(Step("the speech enhancer", self._speech))
        if cfg.detection is not None and cfg.detection.checkpoint is not None:
            steps.append(Step("the sound detector", self._detector))
        return steps

    def _waveformer(self) -> None:
        import numpy as np

        from soundspy.extraction.neural_tse import WaveformerStream, load_neural_core

        neural = self.config.extraction.neural_tse
        core = load_neural_core(Path(neural.checkpoint), neural.live.device)
        self._wf_device = core.device.type
        stream = WaveformerStream(core, neural.live)
        stream.configure(48000)
        stream.start()  # traced steps, CUDA set-up, cuDNN's choices
        label = np.zeros(len(core.labels), dtype=np.float32)
        for _ in range(3):
            core.step(np.zeros(core.context + 2 * core.hop + core.lookahead, np.float32), label,
                      core.init_state())

    def _speech(self) -> None:
        from soundspy.config import speech_model_dir
        from soundspy.extraction.speech_enhancer import SpeechEnhancer, load_speech_core

        sp = self.config.extraction.hybrid.speech
        core = load_speech_core(speech_model_dir(self.config), sp.device,
                                post_filter=sp.post_filter, backend=sp.backend,
                                cpu_threads=sp.cpu_threads)
        SpeechEnhancer(core, atten_lim_db=sp.atten_lim_db, cpu_threads=sp.cpu_threads,
                       input_gain_db=sp.input_gain_db).start()

    def _detector(self) -> None:
        from soundspy.detection.inference import load_core

        det = self.config.detection
        device = det.live.device
        if (self._wf_device == "cuda" and self.config.extraction.neural_tse.live.detector_on_cpu
                and device == "auto"):
            device = "cpu"  # as engine.factory.detector_device decides for the live chain
        load_core(Path(det.checkpoint), device).warm_up()

    # --- running -------------------------------------------------------------------------------

    def start(self) -> None:
        """Run the plan in a background thread (returns at once)."""
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self.run, name="soundspy-preload",
                                            daemon=True)
        self._thread.start()

    def run(self) -> None:
        """Run the plan on this thread."""
        with self._lock:
            self.state, self.started_at = "loading", time.monotonic()
        failed = False
        for step in self.steps:
            with self._lock:
                self.current = step.name
            started = time.perf_counter()
            try:
                step.run()
            except Exception as exc:  # noqa: BLE001 - the session reports its own fallback
                step.error = f"{type(exc).__name__}: {exc}"
                failed = True
                log.warning("preloading %s failed: %s", step.name, step.error)
            step.seconds = round(time.perf_counter() - started, 2)
            log.info("preloaded %s in %.1f s", step.name, step.seconds)
        with self._lock:
            self.current = None
            self.seconds = round(time.monotonic() - (self.started_at or time.monotonic()), 2)
            self.state = "error" if failed and all(s.error for s in self.steps) else "ready"
            self.memory = resources.snapshot()
        vram = self.memory.get("vram_reserved_mb")
        log.info("models ready in %.1f s (RAM %s MB, graphics card %s)", self.seconds,
                 self.memory.get("ram_mb"), "not used" if vram is None else f"{vram} MB")
        self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set() or not self.steps

    def wait(self, progress: Callable[[str], None] | None = None,
             stopping: Callable[[], bool] | None = None, timeout: float | None = None) -> bool:
        """Wait until the plan has run (True) or ``stopping()`` / the timeout (False); reports
        ``progress("Preparing models: <step> (k/n)")`` whenever the step changes."""
        deadline = None if timeout is None else time.monotonic() + timeout
        shown = None
        while not self.done:
            if stopping is not None and stopping():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            message = self.message
            if progress is not None and message != shown:
                progress(message)
                shown = message
            self._done.wait(0.1)
        return True

    @property
    def message(self) -> str:
        with self._lock:
            done = sum(1 for s in self.steps if s.seconds is not None)
            current = self.current
        if current is None:
            return "Preparing models"
        return f"Preparing models: loading {current} ({done + 1}/{len(self.steps)})"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state if self.steps else "ready",
                "current": self.current,
                "done": sum(1 for s in self.steps if s.seconds is not None),
                "total": len(self.steps),
                "seconds": self.seconds,
                "steps": [{"name": s.name, "seconds": s.seconds, "error": s.error}
                          for s in self.steps],
                "memory": self.memory,
            }

"""Neural target sound extraction, live and offline: ``NeuralTSEExtractor`` (step 07).

The fine-tuned Waveformer (``extraction/models/waveformer``) receives the sound and the selected
classes and returns only those classes. Unlike the baseline it does not ask the detector
*whether* a class is present: it works on the sound itself, sample by sample, so it has no
detector lag and it can separate sounds that overlap.

**The path of one chunk** (48 kHz microphone, 44.1 kHz model)::

    chunk (960 samples) ─► resample 48 -> 44.1 kHz ─► model input queue ─► every 416 samples
       │                                                                    (one model step):
       │                                                                    Waveformer with its
       │                                                                    three state buffers
       │                                                                          │
       └─► delay line (same latency) = "dry"        resample 44.1 -> 48 kHz ◄─────┘
                    │                                        │
                    └────────►  y = est + g * (dry - est) ◄──┘ "est"

* **Streaming state.** The network keeps three buffers between steps (encoder context, decoder
  context, output overlap), so step-by-step output equals running it over the whole signal
  (tested). All complete steps available in a chunk go through the model in one call.
* **Sample-rate bridge.** Two ``StreamingResampler``s. The second one's filter length is chosen
  so that the two filter delays together are a whole number of samples at the session rate
  (22 at 48 kHz), which keeps bypass and the evaluation sample-aligned.
* **Latency** = the resampling filters + waiting for a whole model step (416 samples) + the
  model's lookahead (32 samples): 509 samples = 10.6 ms at 48 kHz (447 = 10.1 ms at 44.1 kHz
  without resampling). ``block_chunks > 1`` waits for more steps per call (9.4 ms each).
* **Residual mixing = the attenuation slider.** ``y = est + g * (dry - est)`` with
  ``g = 10^(-attenuation_db / 20)``: everything the model removed comes back ``attenuation_db``
  quieter. 0 dB gives the input, 40 dB almost the pure estimate; the same meaning as for the
  baseline.
* **Nothing selected** = pass-through (``y = dry``, same latency). The model is not run; the
  change to and from extraction is faded over ``ramp_ms``, and the model starts from a clean
  state when a class is selected again.
* **Selection changed** while extracting: ``on_target_change: keep`` keeps the model's context
  (the encoder's ~1.5 s of history does not depend on the selection; only 9 ms of decoder
  history does) and ``reset`` clears it.
* **Where it runs.** In the audio thread (it is in the audio path). ``start()`` warms the model
  up on that thread before the first chunk: CUDA initialises per thread (step 05). On the CPU
  the thread uses ``cpu_threads`` PyTorch threads.
* **Compiled steps.** A step of this network is ~200 small operations, and most of its time is
  PyTorch's bookkeeping between them, not arithmetic. With ``compile: true`` the step is
  *traced* once per number of steps per call (TorchScript, in ``start()``, never in the audio
  path), which removes much of that bookkeeping: the same numbers, about a third less time on the
  graphics card. A traced step is checked against the plain one when it is made and is not used
  if they differ; a call with a step count that was not prepared simply runs untraced.

``NeuralCore`` holds the loaded network and its device and is cached per checkpoint, so live
sessions and offline runs share one copy in memory.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.config import NeuralLiveConfig
from soundspy.core.audio import AudioBuffer
from soundspy.core.target import TargetSpec
from soundspy.dsp.resample import StreamingResampler, taps_for_whole_delay
from soundspy.extraction.base import Extractor


class NeuralUnavailable(RuntimeError):
    """The neural extractor cannot run (no checkpoint, no PyTorch). Message for the user."""


COMPILE_TOLERANCE = 1e-4  # a traced step must reproduce the plain one this closely


class _TracedStep:  # built lazily as a torch.nn.Module (torch is imported on demand)
    @staticmethod
    def wrap(net):
        import torch

        class Step(torch.nn.Module):
            def __init__(self, inner) -> None:
                super().__init__()
                self.net = inner

            def forward(self, x, label, enc_buf, dec_buf, out_buf):
                return self.net(x, label, enc_buf, dec_buf, out_buf, pad=False)

        return Step(net).eval()


class NeuralCore:
    """The loaded network on a device. One forward at a time (a lock), any thread."""

    def __init__(self, net, labels: list[str], sample_rate: int, device: str = "cpu",
                 name: str = "model", checkpoint: Path | None = None) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.labels = list(labels)
        self.sample_rate = int(sample_rate)
        self.name = name
        self.checkpoint = checkpoint
        self.hop, self.context, self.lookahead = net.stream_sizes
        self._lock = threading.Lock()
        self._compiled: dict[int, Any] = {}  # steps per call -> traced step (None = refused)

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, device: str = "auto") -> NeuralCore:
        try:
            import torch
        except ImportError as exc:
            raise NeuralUnavailable('PyTorch is not installed (pip install -e ".[ml]")') from exc
        from soundspy.extraction.neural_model import NeuralModelError, load_checkpoint

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise NeuralUnavailable("extraction.neural_tse.live.device is 'cuda' but CUDA is "
                                    "not available")
        try:
            net, model, labels, _extra = load_checkpoint(Path(checkpoint), "cpu")
        except NeuralModelError as exc:
            raise NeuralUnavailable(str(exc)) from exc
        return cls(net, labels, model.sample_rate, device,
                   name=Path(checkpoint).parent.parent.name, checkpoint=Path(checkpoint))

    def init_state(self):
        return self.net.init_buffers(1, self.device)

    def label_vector(self, labels) -> np.ndarray:
        return np.array([1.0 if lab in labels else 0.0 for lab in self.labels], dtype=np.float32)

    def compile_steps(self, counts) -> list[int]:
        """Trace the streaming step for calls of ``counts`` whole steps (TorchScript).

        Slow (about a second per count on the first use), so it is called from ``start()``,
        never from the audio path. Returns the counts that are now compiled. A trace whose
        output differs from the plain step is thrown away (that count keeps running untraced).
        """
        ready = []
        for steps in sorted({int(c) for c in counts if int(c) >= 1}):
            with self._lock:
                if steps not in self._compiled:
                    self._compiled[steps] = self._trace(steps)
                if self._compiled[steps] is not None:
                    ready.append(steps)
        return ready

    def _trace(self, steps: int):
        import warnings

        torch = self.torch
        size = self.context + steps * self.hop + self.lookahead
        generator = torch.Generator().manual_seed(steps)
        x = (0.1 * torch.randn(1, 1, size, generator=generator)).to(self.device)
        label = torch.zeros(1, len(self.labels), device=self.device)
        label[0, 0] = 1.0
        fresh = lambda: tuple(b.clone() for b in self.net.init_buffers(1, self.device))  # noqa: E731
        plain = _TracedStep.wrap(self.net)
        try:
            with torch.inference_mode(), warnings.catch_warnings():
                warnings.simplefilter("ignore")  # shape-dependent branches: fixed per count
                traced = torch.jit.freeze(torch.jit.trace(plain, (x, label, *fresh()),
                                                          check_trace=False))
                # compare on other input, another selection and a non-empty state
                x2 = (0.3 * torch.randn(1, 1, size, generator=generator)).to(self.device)
                label2 = torch.ones_like(label)
                state = plain(x, label, *fresh())[1:]
                expected = plain(x2, label2, *(b.clone() for b in state))
                with torch.jit.optimized_execution(False):
                    got = traced(x2, label2, *(b.clone() for b in state))
                error = max(float((a - b).abs().max()) for a, b in zip(expected, got,
                                                                         strict=True))
        except Exception:  # noqa: BLE001 - tracing is an optimisation, never a requirement
            return None
        return traced if error <= COMPILE_TOLERANCE else None

    def step(self, samples: np.ndarray, label: np.ndarray, state):
        """Whole streaming steps: ``samples`` = context + k * hop + lookahead (1-D, model
        rate). Returns ``(k * hop output samples, new state)``."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).reshape(1, 1, -1)
        lab = torch.from_numpy(label).reshape(1, -1)
        steps = (x.shape[-1] - self.context - self.lookahead) // self.hop
        with self._lock, torch.inference_mode():
            traced = self._compiled.get(steps)
            if traced is None:
                out, state = self.net.stream_step(x.to(self.device), lab.to(self.device), state)
            else:
                # Without TorchScript's run-time optimiser: it re-specialises the graph during
                # the first real calls (measured: single calls of 48, 60 and 208 ms), which a
                # live session cannot afford, and the gain comes from skipping Python anyway.
                with torch.jit.optimized_execution(False):
                    out, *buffers = traced(x.to(self.device), lab.to(self.device), *state)
                state = tuple(buffers)
            result = out.reshape(-1).float().cpu().numpy()
        return result, state

    def whole(self, samples: np.ndarray, label: np.ndarray) -> np.ndarray:
        """The causal model over a whole signal at once (1-D, model rate), no streaming."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).reshape(1, 1, -1)
        lab = torch.from_numpy(label).reshape(1, -1)
        with self._lock, torch.inference_mode():
            out = self.net(x.to(self.device), lab.to(self.device))
            return out.reshape(-1).float().cpu().numpy()


_CORES: dict[tuple[str, float, str], NeuralCore] = {}
_CORES_LOCK = threading.Lock()


def load_neural_core(checkpoint: Path, device: str = "auto") -> NeuralCore:
    """``NeuralCore.from_checkpoint`` cached per (file, modification time, device)."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise NeuralUnavailable(
            f"no neural extraction model at {checkpoint} (train it with `soundspy train "
            "extraction`, or set extraction.neural_tse.checkpoint)")
    key = (str(checkpoint.resolve()), checkpoint.stat().st_mtime, device)
    with _CORES_LOCK:
        core = _CORES.get(key)
        if core is None:
            core = NeuralCore.from_checkpoint(checkpoint, device)
            _CORES.clear()  # one model in memory at a time
            _CORES[key] = core
    return core


class _Fifo:
    """Samples in, exactly as many as asked out (1-D float32)."""

    def __init__(self, primed: int = 0) -> None:
        self._data = np.zeros(primed, dtype=np.float32)

    def push(self, x: np.ndarray) -> None:
        if x.size:
            self._data = np.concatenate((self._data, x.astype(np.float32, copy=False)))

    def __len__(self) -> int:
        return int(self._data.size)

    def read(self, n: int) -> np.ndarray:
        if n > self._data.size:
            raise RuntimeError(f"asked for {n} samples, only {self._data.size} ready")
        out, self._data = self._data[:n], self._data[n:]
        return out

    def peek(self, n: int) -> np.ndarray:
        return self._data[:n]

    def drop(self, n: int) -> None:
        self._data = self._data[n:]


class NeuralTSEExtractor(Extractor):
    method = "neural_tse"

    def __init__(self, core: NeuralCore, live: NeuralLiveConfig) -> None:
        self.core = core
        self.live = live
        self._latency = 0
        self._times: deque[float] = deque(maxlen=200)
        self._model_reset = True
        self._labels: frozenset[str] = frozenset()  # the last non-empty selection

    # --- Processor -----------------------------------------------------------------------------

    @property
    def latency_samples(self) -> int:
        return self._latency

    def configure(self, sample_rate: int, num_channels: int) -> int:
        if num_channels != 1:
            raise ValueError(
                f"NeuralTSEExtractor works on one channel, got {num_channels} "
                "(put a ChannelReducer or a spatial filter before it)")
        super().configure(sample_rate, num_channels)
        core = self.core
        self._resample = sample_rate != core.sample_rate
        self._block = core.hop * self.live.block_chunks
        wait = core.lookahead + self._block - 1  # model-rate samples a step may still be missing
        if self._resample:
            self._down = StreamingResampler(sample_rate, core.sample_rate)
            self._up = StreamingResampler(
                core.sample_rate, sample_rate,
                taps=taps_for_whole_delay(self._down, sample_rate, core.sample_rate))
            filters = self._down.delay * sample_rate / core.sample_rate + self._up.delay
            self._filter_delay = round(filters)
            if abs(filters - self._filter_delay) > 1e-6:  # by construction of the second filter
                raise RuntimeError(f"resampler delays are not a whole sample: {filters}")
            self._primed = math.ceil(wait * sample_rate / core.sample_rate)
        else:
            self._down = self._up = None
            self._filter_delay = 0
            self._primed = wait
        self._latency = self._primed + self._filter_delay
        self._ramp_step = (1.0 if self.live.ramp_ms <= 0
                           else 1000.0 / (self.live.ramp_ms * sample_rate))
        self.reset()
        return 1

    def reset(self) -> None:
        if self.sample_rate is None:
            return
        if self._resample:
            self._down.reset()
            self._up.reset()
        self._model_in = _Fifo(self.core.context)  # zeros before the first sample
        self._estimate = _Fifo(self._primed)
        self._dry = _Fifo(self._latency)
        self._mix = 1.0 if self._processing() else 0.0
        self._state = None
        self._model_reset = True
        self._times.clear()

    def set_target(self, target: TargetSpec) -> None:
        previous = self.target
        super().set_target(target)
        if (previous is not None and previous.labels and target.labels
                and previous.labels != target.labels
                and self.live.on_target_change == "reset"):
            self._model_reset = True

    def start(self) -> None:
        """Warm the model up **on the calling thread** (the audio thread): CUDA sets up
        per-thread state on a thread's first call (~3 s), which must not hit the first chunk."""
        core = self.core
        if core.device.type == "cpu":
            core.torch.set_num_threads(self.live.cpu_threads)  # per thread under OpenMP
        # a 20 ms chunk holds 2.1 steps of the default model: calls of 2 or 3 steps (1 at the
        # very start); with block_chunks = b, of b .. b + 2
        counts = range(self.live.block_chunks, self.live.block_chunks + 3)
        if self.live.compile:
            core.compile_steps(counts)
        label = np.zeros(len(core.labels), dtype=np.float32)
        for steps in counts:
            size = core.context + steps * core.hop + core.lookahead
            for _ in range(3):
                core.step(np.zeros(size, dtype=np.float32), label, core.init_state())

    def _processing(self) -> bool:
        return self.target is not None and bool(self.target.labels)

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        if self.sample_rate is None:
            raise RuntimeError("NeuralTSEExtractor.configure() must be called first")
        if chunk.num_channels != 1:
            raise ValueError(f"expected 1 channel, got {chunk.num_channels}")
        x = chunk.samples.reshape(-1)
        n = x.shape[0]
        core = self.core
        processing = self._processing()
        running = processing or self._mix > 0.0  # the model also runs while fading out

        self._dry.push(x)
        model_x = self._down.push(chunk.samples).reshape(-1) if self._resample else x
        self._model_in.push(model_x)
        steps = (len(self._model_in) - core.context - core.lookahead) // core.hop
        if steps >= self.live.block_chunks:
            size = core.context + steps * core.hop + core.lookahead
            if running:
                if self._model_reset or self._state is None:
                    self._state = core.init_state()
                    self._model_reset = False
                if processing:
                    self._labels = self.target.labels
                # while fading out after a deselection, keep extracting the last selection
                started = time.perf_counter()
                out, self._state = core.step(self._model_in.peek(size),
                                             core.label_vector(self._labels), self._state)
                self._times.append(time.perf_counter() - started)
            else:  # pass-through: the model rests and starts fresh when it is needed again
                out = np.zeros(steps * core.hop, dtype=np.float32)
                self._model_reset = True
            self._model_in.drop(steps * core.hop)
            if self._resample:
                out = self._up.push(out.reshape(-1, 1)).reshape(-1)
            self._estimate.push(out)

        dry = self._dry.read(n)
        estimate = self._estimate.read(n)
        target_mix = 1.0 if processing else 0.0
        if self._mix == target_mix:
            weight: np.ndarray | float = target_mix
        else:  # fade between pass-through and extraction
            direction = 1.0 if target_mix > self._mix else -1.0
            ramp = self._mix + direction * self._ramp_step * np.arange(1, n + 1)
            weight = np.clip(ramp, 0.0, 1.0).astype(np.float32)
            self._mix = float(weight[-1]) if n else self._mix
        attenuation_db = self.target.attenuation_db if self.target is not None else 0.0
        keep = 1.0 - 10.0 ** (-attenuation_db / 20.0)  # share of the removed sound that is cut
        # y = est + g * (dry - est), faded: y = dry + weight * (1 - g) * (est - dry)
        y = dry + (weight * keep) * (estimate - dry)
        return AudioBuffer(np.ascontiguousarray(y, dtype=np.float32).reshape(-1, 1),
                           chunk.sample_rate, {**chunk.metadata, "extraction": self.method})

    def status_info(self) -> dict[str, Any]:
        info = super().status_info()
        times = list(self._times)
        info.update(
            target_active=self._processing(),
            device=str(self.core.device),
            model=self.core.name,
            compiled=bool(self.live.compile and any(
                v is not None for v in self.core._compiled.values())),
            model_ms=round(1000.0 * float(np.mean(times)), 2) if times else None,
            model_ms_p95=round(1000.0 * float(np.percentile(times, 95)), 2) if times else None,
        )
        return info


def extract_whole(core: NeuralCore, samples: np.ndarray, sample_rate: int, labels,
                  attenuation_db: float) -> np.ndarray:
    """The causal model over a **whole** signal in one call, without chunking or state hand-over.

    Same resampling filters and the same residual mixing as ``NeuralTSEExtractor``; the result
    is aligned with the input and equally long. Used by the evaluation to show that streaming
    gives the same output.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = x.shape[0]
    label = core.label_vector(labels)
    if sample_rate == core.sample_rate:
        estimate = core.whole(x, label)[:n]
    else:
        down = StreamingResampler(sample_rate, core.sample_rate)
        up = StreamingResampler(core.sample_rate, sample_rate,
                                taps=taps_for_whole_delay(down, sample_rate, core.sample_rate))
        delay = round(down.delay * sample_rate / core.sample_rate + up.delay)
        padded = np.concatenate((x, np.zeros(2 * delay + 64, dtype=np.float32)))
        model_out = core.whole(down.push(padded.reshape(-1, 1)).reshape(-1), label)
        estimate = up.push(model_out.reshape(-1, 1)).reshape(-1)[delay:delay + n]
    g = 10.0 ** (-attenuation_db / 20.0)
    return (estimate + g * (x - estimate)).astype(np.float32)


def build_neural_extractor(config, *, device: str | None = None) -> NeuralTSEExtractor:
    """The neural extractor from the configuration; ``NeuralUnavailable`` says why not."""
    cfg = config.extraction.neural_tse
    if cfg.checkpoint is None:
        raise NeuralUnavailable("no neural extraction checkpoint is configured "
                                "(extraction.neural_tse.checkpoint)")
    core = load_neural_core(Path(cfg.checkpoint), device or cfg.live.device)
    return NeuralTSEExtractor(core, cfg.live)

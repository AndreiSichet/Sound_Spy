"""The speech route of the hybrid extractor: DeepFilterNet3 as a streaming enhancer (step 07c).

DeepFilterNet3 (``extraction/models/deepfilternet``) keeps **all speech** and removes everything
else, in 10 ms frames at 48 kHz. It knows nothing about our classes, the detector or the slider:
the hybrid extractor decides when it runs and mixes its output with the input.

**The path of one chunk** (48 kHz, any chunk size)::

    samples ─► input queue ─► every 480 samples: analysis STFT (960 window) ─► features
                                                  (ERB levels, 96 complex bins, running means)
                                                       │
       est ◄── output queue ◄── synthesis (overlap-add) ◄── network step (k frames at once)

* **Delay.** An output frame needs the 2 frames after it (the network's lookahead), the STFT
  adds ``fft_size - hop`` samples, and a sample may wait up to ``hop - 1`` for its frame to be
  complete: ``(hop - 1) + 2 * hop + (fft_size - hop)`` = **1919 samples = 40.0 ms** at 48 kHz.
  The output queue starts with ``1439`` zeros so that any chunk size can be answered at once.
* **State.** Everything the network carries between calls is explicit (``DfNet.stream``): the
  last 2 feature frames, the last 4 spectra, the last 4 frames of one inner layer and three GRU
  states. The running means of the features belong to ``Features``. Streaming gives the same
  output as the whole signal at once (:func:`enhance_whole`; tested).
* **Resting.** ``process(x, active=False)`` keeps the STFT, the features and the windows up to
  date (cheap) but does not run the network and outputs zeros; its GRU states start from zero
  again on the next active call. So selecting Speech gives a correctly normalised network at
  once, without a start-up dip.
* **Input gain** (``input_gain_db``, step 07c follow-up): the network sees the input this much
  louder, and its output is turned down by the same amount, so the level that comes out is
  unchanged. From a cold start DeepFilterNet3 treats very quiet input as noise: the laptop
  microphone (speech peaks around −29 dBFS) lost the owner's speech for its first 5–10 s.
* **Attenuation limit** (``atten_lim_db``, DeepFilterNet's own option): mixes the delayed input
  back at that level. ``None`` (configured) leaves all attenuation to the session's slider.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.extraction.models.deepfilternet import Analysis, DfParams, Features, Synthesis


class SpeechUnavailable(RuntimeError):
    """The speech enhancer cannot run (no weights, no PyTorch, wrong rate). User message."""


def _pairs(c: np.ndarray) -> np.ndarray:
    """Complex ``(k, F)`` -> ``[1, 1, k, F, 2]`` float32 (the network's layout)."""
    return np.ascontiguousarray(np.stack((c.real, c.imag), axis=-1), dtype=np.float32)[None, None]


def _complex(frames: np.ndarray) -> np.ndarray:
    out = np.asarray(frames, dtype=np.float32)[0, 0]  # [k, F, 2]
    return (out[..., 0] + 1j * out[..., 1]).astype(np.complex64)


class SpeechCore:
    """The loaded network and how its streaming step runs. One call at a time (a lock).

    ``backend``:

    * ``"onnx"``: the streaming step exported to ONNX (``speech_model.export_onnx``, cached next
      to the model) and run by ONNX Runtime on the CPU. A step of this network is ~150 tiny
      operations; ONNX Runtime's overhead per operation is a fraction of PyTorch's (measured on
      the laptop, one 20 ms chunk: 2.6 ms vs 8 ms for TorchScript on 2 threads, 16 ms eager on
      one). State = NumPy arrays.
    * ``"torch"``: ``DfNet.stream`` in PyTorch (any device). State = tensors.

    Whole signals (:meth:`whole`, evaluation) always use PyTorch: one call over all frames.
    """

    def __init__(self, net, p: DfParams, device: str = "cpu", name: str = "DeepFilterNet3",
                 checkpoint: Path | None = None, *, backend: str = "torch",
                 onnx_file: Path | None = None, cpu_threads: int = 1) -> None:
        import torch

        from soundspy.extraction.models.deepfilternet.model import fuse_for_inference

        self.torch = torch
        self.device = torch.device(device)
        self.net = fuse_for_inference(net).to(self.device).eval()
        self.p = p
        self.name = name
        self.checkpoint = checkpoint
        self.backend = backend
        self.onnx_file = onnx_file
        self._lock = threading.Lock()
        self._session = None
        if backend == "onnx":
            if onnx_file is None:
                raise ValueError("the onnx backend needs onnx_file")
            self._session = self._open_session(Path(onnx_file), cpu_threads)
        elif backend != "torch":
            raise ValueError(f"unknown backend {backend!r}")

    @staticmethod
    def _open_session(path: Path, threads: int):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SpeechUnavailable('ONNX Runtime is not installed (pip install -e ".[ml]")'
                                    ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    @property
    def lag_frames(self) -> int:
        """Frames between the newest input frame and the frame the network outputs."""
        return self.p.conv_lookahead

    @property
    def device_name(self) -> str:
        return "cpu (onnx)" if self.backend == "onnx" else str(self.device)

    def init_state(self):
        state = self.net.init_state(self.device)
        if self.backend == "onnx":
            return tuple(s.cpu().numpy() for s in state)
        return state

    def stream(self, spec, erb, low, state):
        """Enhanced spectra ``(k, F)`` for ``k`` new frames, and the new state."""
        if self.backend == "onnx":
            from soundspy.extraction.speech_model import STATE_NAMES

            feed = {"spec": _pairs(spec), "erb": np.ascontiguousarray(erb, np.float32)[None, None],
                    "low": _pairs(low), **dict(zip(STATE_NAMES, state, strict=True))}
            with self._lock:
                out = self._session.run(None, feed)
            return _complex(out[0]), tuple(out[1:])
        torch = self.torch
        with self._lock, torch.inference_mode():
            t = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
            out, state = self.net.stream(t(_pairs(spec)),
                                         t(np.ascontiguousarray(erb, np.float32)[None, None]),
                                         t(_pairs(low)), state)
            return _complex(out.float().cpu().numpy()), state

    def push(self, spec, erb, low, state, *, features: bool):
        """Move the windows without running the network (``features``: also the feature
        windows, i.e. the network rests and its recurrent state starts afresh; else only the
        spectra: the first frames of a stream)."""
        net = self.net
        if self.backend == "onnx":
            def keep(old, new, axis_len):
                both = np.concatenate((old, new), axis=2)
                return np.take(both, np.arange(both.shape[2] - axis_len, both.shape[2]),
                               axis=2)

            erb_w, spec_w, specs, *rest = state
            specs = keep(specs, _pairs(spec), net.spec_context)
            if not features:
                return (erb_w, spec_w, specs, *rest)
            low_f = np.ascontiguousarray(_pairs(low).squeeze(1).transpose(0, 3, 1, 2))
            # ^ [1, 2, k, Fd], as DfNet.push_features
            erb_w = keep(erb_w, np.ascontiguousarray(erb, np.float32)[None, None],
                         net.feat_context)
            spec_w = keep(spec_w, low_f, net.feat_context)
            fresh = self.init_state()
            return (erb_w, spec_w, specs, *fresh[3:])
        torch = self.torch
        with self._lock, torch.inference_mode():
            t = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
            if features:
                return net.push_features(t(_pairs(spec)),
                                         t(np.ascontiguousarray(erb, np.float32)[None, None]),
                                         t(_pairs(low)), state)
            return net.push_spectra(t(_pairs(spec)), state)

    def whole(self, spec, erb, low) -> np.ndarray:
        torch = self.torch
        with self._lock, torch.inference_mode():
            t = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
            out = self.net(t(_pairs(spec)), t(np.ascontiguousarray(erb, np.float32)[None, None]),
                           t(_pairs(low)))
            return _complex(out.float().cpu().numpy())


ONNX_TOLERANCE = 1e-4  # an exported step must reproduce the PyTorch one this closely


def check_onnx(core_torch: SpeechCore, core_onnx: SpeechCore, steps: int = 4) -> float:
    """Largest difference between the two backends over a few random streaming steps."""
    rng = np.random.default_rng(7)
    p = core_torch.p
    st_t, st_o = core_torch.init_state(), core_onnx.init_state()
    worst = 0.0
    for k in (2, 9, 1)[:steps]:
        spec = (0.01 * (rng.standard_normal((k, p.freq_bins))
                        + 1j * rng.standard_normal((k, p.freq_bins)))).astype(np.complex64)
        erb = rng.standard_normal((k, p.nb_erb)).astype(np.float32)
        low = spec[..., :p.nb_df] * 10
        a, st_t = core_torch.stream(spec, erb, low, st_t)
        b, st_o = core_onnx.stream(spec, erb, low, st_o)
        worst = max(worst, float(np.abs(a - b).max()))
    return worst


_CORES: dict[tuple, SpeechCore] = {}
_CORES_LOCK = threading.Lock()


def load_speech_core(model_dir: Path, device: str = "cpu", *, post_filter: bool = False,
                     backend: str = "onnx", cpu_threads: int = 1) -> SpeechCore:
    """The published DeepFilterNet3 from an unpacked model folder, cached per (folder,
    checkpoint time, device, post-filter, backend). The ONNX backend exports the streaming
    step on first use (a few seconds, cached next to the model) and checks it against
    PyTorch; it needs the CPU (``device`` other than ``cpu`` uses the PyTorch backend)."""
    try:
        import torch
    except ImportError as exc:
        raise SpeechUnavailable('PyTorch is not installed (pip install -e ".[ml]")') from exc
    from soundspy.extraction.speech_model import (
        SpeechModelError,
        export_onnx,
        load_model,
        onnx_path,
    )

    model_dir = Path(model_dir)
    if not (model_dir / "config.ini").is_file():
        raise SpeechUnavailable(
            f"no speech enhancement model at {model_dir} (download it with `soundspy data "
            "download deepfilternet`)")
    if device == "auto":
        device = "cpu"  # the speech enhancer's home (the graphics card is the Waveformer's)
    elif device == "cuda" and not torch.cuda.is_available():
        raise SpeechUnavailable("extraction.hybrid.speech.device is 'cuda' but CUDA is not "
                                "available")
    if device != "cpu":
        backend = "torch"
    checkpoints = sorted((model_dir / "checkpoints").glob("*.ckpt*"))
    stamp = max((c.stat().st_mtime for c in checkpoints), default=0.0)
    key = (str(model_dir.resolve()), stamp, device, bool(post_filter), backend, cpu_threads)
    with _CORES_LOCK:
        core = _CORES.get(key)
        if core is None:
            try:
                net, p, path = load_model(model_dir, post_filter=post_filter)
            except SpeechModelError as exc:
                raise SpeechUnavailable(str(exc)) from exc
            core = SpeechCore(net, p, device, name=model_dir.name, checkpoint=path)
            if backend == "onnx":
                target = onnx_path(model_dir, path, post_filter)
                try:
                    if not target.is_file():
                        export_onnx(core.net, target)
                    fast = SpeechCore(core.net, p, "cpu", name=model_dir.name, checkpoint=path,
                                      backend="onnx", onnx_file=target, cpu_threads=cpu_threads)
                    error = check_onnx(core, fast)
                except SpeechUnavailable:
                    raise
                except Exception as exc:  # noqa: BLE001 - export problems: say so, use torch
                    raise SpeechUnavailable(f"the ONNX export of {path.name} failed: "
                                            f"{type(exc).__name__}: {exc}") from exc
                if error > ONNX_TOLERANCE:
                    target.unlink(missing_ok=True)
                    raise SpeechUnavailable(f"the ONNX step differs from PyTorch by {error:.2g}")
                core = fast
            _CORES.clear()  # one copy in memory
            _CORES[key] = core
    return core


class _Fifo:
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


def latency_samples(p: DfParams, lag_frames: int) -> int:
    """Samples from a sample's arrival to its enhanced version (any chunk size)."""
    return (p.hop_size - 1) + lag_frames * p.hop_size + (p.fft_size - p.hop_size)


class SpeechEnhancer:
    """Streaming DeepFilterNet3: ``process`` takes any number of samples and returns as many,
    ``latency`` samples late."""

    def __init__(self, core: SpeechCore, *, atten_lim_db: float | None = None,
                 cpu_threads: int = 1, input_gain_db: float = 0.0) -> None:
        self.core = core
        self.p = core.p
        self.atten_lim_db = atten_lim_db
        self.cpu_threads = int(cpu_threads)
        self.input_gain_db = float(input_gain_db)
        self._gain = np.float32(10.0 ** (self.input_gain_db / 20.0))
        self.latency = latency_samples(self.p, core.lag_frames)
        self._times: deque[float] = deque(maxlen=200)
        self.reset()

    @property
    def sample_rate(self) -> int:
        return self.p.sr

    def reset(self) -> None:
        p = self.p
        self._input = _Fifo()
        self._output = _Fifo(self.latency - (p.fft_size - p.hop_size))
        self._dry = _Fifo(self.latency) if self.atten_lim_db else None
        self._analysis = Analysis(p)
        self._synthesis = Synthesis(p)
        self._features = Features(p)
        self._state = self.core.init_state()
        self._frames = 0  # frames analysed since the start of the stream
        self._resting = False
        self._times.clear()

    def start(self) -> None:
        """Warm the network up on the calling thread (the audio thread)."""
        core = self.core
        if core.backend == "torch" and core.device.type == "cpu":
            core.torch.set_num_threads(self.cpu_threads)  # per thread under OpenMP
        p = self.p
        state = core.init_state()
        zeros = np.zeros((2, p.freq_bins), dtype=np.complex64)
        for _ in range(3):
            core.stream(zeros, np.zeros((2, p.nb_erb), np.float32),
                        np.zeros((2, p.nb_df), np.complex64), state)

    @property
    def model_ms(self) -> tuple[float | None, float | None]:
        times = list(self._times)
        if not times:
            return None, None
        return (round(1000.0 * float(np.mean(times)), 2),
                round(1000.0 * float(np.percentile(times, 95)), 2))

    def process(self, x: np.ndarray, active: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        p, core = self.p, self.core
        self._input.push(x * self._gain if self._gain != 1.0 else x)
        if self._dry is not None:
            self._dry.push(x)
        k = len(self._input) // p.hop_size
        if k:
            hops = self._input.read(k * p.hop_size).reshape(k, p.hop_size)
            spec = self._analysis.frames(hops)
            erb, low = self._features.frames(spec)
            first = max(0, min(k, core.lag_frames - self._frames))  # no output for these yet
            if first:
                self._state = core.push(spec[:first], erb[:first], low[:first], self._state,
                                        features=False)
            rest = slice(first, k)
            if k > first:
                if active:
                    started = time.perf_counter()
                    out, self._state = core.stream(spec[rest], erb[rest], low[rest],
                                                   self._state)
                    self._times.append(time.perf_counter() - started)
                    if self._resting:
                        self._synthesis.reset()
                    self._output.push(self._synthesis.frames(out))
                    self._resting = False
                else:  # resting: the windows follow, the network does not run
                    self._state = core.push(spec[rest], erb[rest], low[rest], self._state,
                                            features=True)
                    self._output.push(np.zeros((k - first) * p.hop_size, dtype=np.float32))
                    self._resting = True
            self._frames += k
        est = self._output.read(x.size)
        if self._gain != 1.0:
            est = est / self._gain
        if self._dry is not None:
            dry = self._dry.read(x.size)
            lim = 10.0 ** (-abs(self.atten_lim_db) / 20.0)
            est = (lim * dry + (1.0 - lim) * est).astype(np.float32)
        return est

    def status_info(self) -> dict[str, Any]:
        mean, p95 = self.model_ms
        return {"model": self.core.name, "device": self.core.device_name,
                "latency_ms": round(1000.0 * self.latency / self.p.sr, 2),
                "model_ms": mean, "model_ms_p95": p95}


def enhance_whole(core: SpeechCore, x: np.ndarray, *, atten_lim_db: float | None = None,
                  input_gain_db: float = 0.0) -> np.ndarray:
    """The whole signal through the network in one call (no streaming state hand-over),
    aligned with ``x`` and as long. Equals ``SpeechEnhancer`` output shifted by its latency."""
    p = core.p
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = x.size
    # enough zeros that every frame covering x has lag_frames real frames after it
    total = -(-(n + (core.lag_frames + 2) * p.hop_size) // p.hop_size) * p.hop_size
    gain = np.float32(10.0 ** (float(input_gain_db) / 20.0))
    padded = np.zeros(total, dtype=np.float32)
    padded[:n] = x * gain
    spec = Analysis(p).frames(padded.reshape(-1, p.hop_size))
    erb, low = Features(p).frames(spec)
    out = Synthesis(p).frames(core.whole(spec, erb, low))
    d = p.fft_size - p.hop_size
    est = out[d:d + n] / gain
    if atten_lim_db:
        lim = 10.0 ** (-abs(atten_lim_db) / 20.0)
        est = lim * x + (1.0 - lim) * est
    return est.astype(np.float32)

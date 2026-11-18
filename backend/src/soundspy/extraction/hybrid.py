"""``HybridExtractor``: silence when absent, separation when overlapping (step 07c).

**Why.** The owner's listening test (classical music, talking with pauses, Speech selected,
40 dB) showed what each earlier method does well and badly: the baseline's detector-driven gate
made the **pauses silent** but let the music through **under the voice**; the Waveformer
separated a little but left the music audible in the pauses (~-20 dB) and, with its output
gate, cut quiet syllables. The hybrid combines three parts, each used for what it was measured
to do best:

=====================  =========================================================================
selection              route
=====================  =========================================================================
nothing                pass-through (the input, delayed like the routes)
exactly ``speech``     **speech enhancer**: DeepFilterNet3 (``speech_enhancer.py``), which keeps
(``speech.labels``)    every voice and removes everything else, in 10 ms frames at 48 kHz
anything else          **Waveformer** v1 (``neural_tse.py``), label-conditioned; this includes
                       speech together with other classes: two extractors summed would count
                       the sound they share twice
=====================  =========================================================================

``speech.route: cascade`` sends Speech through the Waveformer first and DeepFilterNet3 after it.

After the route: the :class:`~soundspy.extraction.hybrid_gate.HybridGate` (the detector closes
it, the extractor's output reopens it), then residual mixing with the slider::

    y = G * (est + g * (x - est)),    g = 10^(-attenuation_db / 20),   G = gate gain (1 .. g)

**One delay for every route.** The session needs a fixed latency (bypass, recordings, the
evaluation), so every route's estimate is delayed to the slowest route's: DeepFilterNet3 needs
40.0 ms (1919 samples at 48 kHz), the Waveformer 10.6 ms (+29.4 ms of padding).

**Changing route** (the selection changed): the new route starts at once from a clean state;
its estimate is valid one latency later (before that it only saw zeros), then the estimates are
crossfaded over ``live.ramp_ms``; the old route rests afterwards. A resting route is not
processed at all: keeping its input side running cost about 1 ms per 20 ms chunk (resampling,
the speech enhancer's STFT and features), which the Waveformer route could not spare live.

**Where it runs.** In the audio thread: DeepFilterNet3 on the CPU (ONNX Runtime), the
Waveformer on the graphics card. ``start()`` warms both up on that thread and keeps it on the
CPU's performance cores (``live.performance_cores``, ``engine/cpu.py``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.config import HybridConfig
from soundspy.core.audio import AudioBuffer
from soundspy.extraction.activity import ActivitySource
from soundspy.extraction.base import Extractor
from soundspy.extraction.hybrid_gate import HybridGate

OFF, SPEECH, WAVEFORMER, CASCADE = "off", "speech_enhancer", "waveformer", "cascade"


class HybridUnavailable(RuntimeError):
    """Neither route of the hybrid extractor can run. Message for the user."""


class _Fifo:
    def __init__(self, primed: int = 0) -> None:
        self._data = np.zeros(primed, dtype=np.float32)

    def push(self, x: np.ndarray) -> None:
        if x.size:
            self._data = np.concatenate((self._data, x.astype(np.float32, copy=False)))

    def read(self, n: int) -> np.ndarray:
        out, self._data = self._data[:n], self._data[n:]
        return out


class _Route:
    """One route: ``process(x, labels)`` -> estimate (``labels=None``: rest)."""

    name = OFF
    latency = 0

    def configure(self, sample_rate: int) -> int:
        return self.latency

    def reset(self) -> None:
        pass

    def start(self) -> None:
        pass

    def process(self, x: np.ndarray, labels) -> np.ndarray:
        return x

    def status(self) -> dict[str, Any]:
        return {}


class _SpeechRoute(_Route):
    name = SPEECH

    def __init__(self, enhancer) -> None:
        self.enhancer = enhancer
        self.latency = enhancer.latency

    def reset(self) -> None:
        self.enhancer.reset()

    def start(self) -> None:
        self.enhancer.start()

    def process(self, x, labels):
        return self.enhancer.process(x, active=labels is not None)

    def status(self) -> dict[str, Any]:
        return {"speech_enhancer": self.enhancer.status_info()}


class _WaveformerRoute(_Route):
    name = WAVEFORMER

    def __init__(self, stream) -> None:
        self.stream = stream

    def configure(self, sample_rate: int) -> int:
        self.latency = self.stream.configure(sample_rate)
        return self.latency

    def reset(self) -> None:
        self.stream.reset()

    def start(self) -> None:
        self.stream.start()

    def process(self, x, labels):
        return self.stream.process(x, labels)

    def status(self) -> dict[str, Any]:
        mean, p95 = self.stream.model_ms()
        core = self.stream.core
        rate = self.stream.sample_rate
        return {"waveformer": {"model": core.name, "device": str(core.device),
                               "latency_ms": (round(1000.0 * self.latency / rate, 2)
                                              if rate else None),
                               "model_ms": mean, "model_ms_p95": p95}}


class _CascadeRoute(_Route):
    """Waveformer (the speech labels), then the speech enhancer on its estimate."""

    name = CASCADE

    def __init__(self, stream, enhancer) -> None:
        self.first, self.second = _WaveformerRoute(stream), _SpeechRoute(enhancer)

    def configure(self, sample_rate: int) -> int:
        self.latency = self.first.configure(sample_rate) + self.second.latency
        return self.latency

    def reset(self) -> None:
        self.first.reset()
        self.second.reset()

    def start(self) -> None:
        self.first.start()
        self.second.start()

    def process(self, x, labels):
        return self.second.process(self.first.process(x, labels), labels)

    def status(self) -> dict[str, Any]:
        return {**self.first.status(), **self.second.status()}


class HybridExtractor(Extractor):
    method = "hybrid"

    def __init__(self, config: HybridConfig, *, waveformer=None, speech=None,
                 cascade_waveformer=None, activity: ActivitySource | None = None,
                 gate: bool | None = None, notes: Collection[str] = ()) -> None:
        """``waveformer``: a ``WaveformerStream`` (or ``None``); ``speech``: a
        ``SpeechEnhancer`` (or ``None``); ``cascade_waveformer``: a second
        ``WaveformerStream`` for ``speech.route: cascade``. ``gate``: switch the gate on / off
        (``None``: ``gate.enabled``). ``notes``: why a route is missing (status)."""
        if waveformer is None and speech is None:
            raise HybridUnavailable("neither the Waveformer nor the speech enhancer is available")
        self.config = config
        self.speech_labels = frozenset(config.speech.labels)
        self.gate_config = (config.gate if gate is None
                            else dataclasses.replace(config.gate, enabled=bool(gate)))
        self.activity = activity
        self.notes = list(notes)
        self._speech = speech
        self._routes: dict[str, _Route] = {OFF: _Route()}
        if waveformer is not None:
            self._routes[WAVEFORMER] = _WaveformerRoute(waveformer)
        if speech is not None:
            if config.speech.route == "cascade" and cascade_waveformer is not None:
                self._routes[CASCADE] = _CascadeRoute(cascade_waveformer, speech)
            else:
                self._routes[SPEECH] = _SpeechRoute(speech)
        self._latency = 0
        self.gate: HybridGate | None = None
        self.pinned: int | None = None
        self._incoming = OFF  # the route in use (set again by reset())
        self._weights: dict[str, float] = {OFF: 1.0}
        self._wait = 0
        self._wf_labels: frozenset[str] = frozenset()

    # --- routing -------------------------------------------------------------------------------

    @property
    def speech_route(self) -> str | None:
        return CASCADE if CASCADE in self._routes else SPEECH if SPEECH in self._routes else None

    def route_for(self, labels: Collection[str]) -> str:
        """The route a selection goes to (see the module docstring)."""
        labels = frozenset(labels)
        if not labels:
            return OFF
        if labels == self.speech_labels and self.speech_route is not None:
            return self.speech_route
        if WAVEFORMER in self._routes:
            return WAVEFORMER
        return OFF  # only the speech enhancer exists and something else is selected

    def set_activity(self, activity: ActivitySource | None) -> None:
        """The detector's activity source (the chain factory builds the detector after the
        extractor: the extractor's devices decide where the detector runs)."""
        self.activity = activity
        if self.gate is not None:
            self.gate.activity = activity

    @property
    def cores(self) -> list:
        """The loaded networks (``detector_device`` looks at their devices)."""
        out = []
        for route in self._routes.values():
            for part in (getattr(route, "stream", None), getattr(route, "first", None)):
                if part is not None:
                    out.append(getattr(part, "stream", part).core)
        return out

    @property
    def core(self):
        """The Waveformer's core (for ``detector_device``), else the speech enhancer's."""
        cores = self.cores
        return cores[0] if cores else self._speech.core

    # --- Processor -----------------------------------------------------------------------------

    @property
    def latency_samples(self) -> int:
        return self._latency

    def configure(self, sample_rate: int, num_channels: int) -> int:
        if num_channels != 1:
            raise ValueError(f"HybridExtractor works on one channel, got {num_channels} "
                             "(put a ChannelReducer or a spatial filter before it)")
        super().configure(sample_rate, num_channels)
        for name in (SPEECH, CASCADE):
            if name in self._routes and self._speech.sample_rate != sample_rate:
                self.notes.append(f"the speech enhancer needs {self._speech.sample_rate} Hz, the "
                                  f"session runs at {sample_rate} Hz: Speech uses the Waveformer")
                del self._routes[name]
        if len(self._routes) == 1:
            raise HybridUnavailable("no route can run at this sample rate: " + "; ".join(
                self.notes))
        latencies = {name: route.configure(sample_rate) for name, route in self._routes.items()}
        self._latency = max(latencies.values())
        self._route_latency = latencies
        self._ramp = (1.0 if self.config.ramp_ms <= 0
                      else 1000.0 / (self.config.ramp_ms * sample_rate))
        self.gate = HybridGate(self.gate_config, sample_rate, self._latency, self.activity)
        self.reset()
        return 1

    def reset(self) -> None:
        if self.sample_rate is None:
            return
        for route in self._routes.values():
            route.reset()
        self._pads = {name: _Fifo(self._latency - lat) for name, lat in
                      self._route_latency.items() if name != OFF}
        self._dry = _Fifo(self._latency)
        self._resting: set[str] = set()
        current = self.route_for(self.target.labels if self.target is not None else ())
        self._weights = {current: 1.0}
        self._incoming = current
        self._wait = 0
        self._wf_labels: frozenset[str] = frozenset()
        if self.gate is not None:
            self.gate.reset()
            self.gate.restart(current in (SPEECH, CASCADE))

    def start(self) -> None:
        if self.config.performance_cores:
            from soundspy.engine.cpu import prefer_performance_cores

            self.pinned = prefer_performance_cores()
        for route in self._routes.values():
            route.start()

    @property
    def route(self) -> str:
        """The route the current selection uses (fading in, or fully in use)."""
        return self._incoming

    def _schedule(self, n: int) -> dict[str, np.ndarray | float]:
        """Per-sample weights of the routes for the next ``n`` samples (crossfade)."""
        want = self.route_for(self.target.labels if self.target is not None else ())
        if want != self._incoming:
            self._incoming = want
            self._weights.setdefault(want, 0.0)
            # a network that starts now gives a valid estimate one latency later
            self._wait = 0 if want == OFF else self._latency
            if want != OFF and self.gate is not None:
                self.gate.restart(want in (SPEECH, CASCADE))  # DeepFilterNet3 starts leaky
        w0 = self._weights[self._incoming]
        if w0 >= 1.0 and len(self._weights) == 1:
            self._wait = max(0, self._wait - n)
            return {self._incoming: 1.0}
        start = min(n, self._wait)
        steps = np.maximum(0, np.arange(1, n + 1) - start)
        w_in = np.clip(w0 + self._ramp * steps, 0.0, 1.0).astype(np.float32)
        weights: dict[str, np.ndarray | float] = {self._incoming: w_in}
        others = {k: v for k, v in self._weights.items() if k != self._incoming}
        total = sum(others.values())
        for name, value in others.items():
            share = value / total if total > 0 else 0.0
            weights[name] = (share * (1.0 - w_in)).astype(np.float32)
        self._wait = max(0, self._wait - n)
        end = float(w_in[-1]) if n else w0
        if end >= 1.0:
            self._weights = {self._incoming: 1.0}
        else:
            self._weights = {self._incoming: end,
                             **{k: float(np.asarray(weights[k])[-1]) if n else v
                                for k, v in others.items()}}
        return weights

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        if self.sample_rate is None:
            raise RuntimeError("HybridExtractor.configure() must be called first")
        if chunk.num_channels != 1:
            raise ValueError(f"expected 1 channel, got {chunk.num_channels}")
        x = chunk.samples.reshape(-1)
        n = x.shape[0]
        labels = self.target.labels if self.target is not None else frozenset()
        weights = self._schedule(n)
        self._dry.push(x)
        dry = self._dry.read(n)
        est = np.zeros(n, dtype=np.float32)
        for name, route in self._routes.items():
            if name == OFF:
                continue
            running = name in weights
            # the speech labels for the speech routes; for the Waveformer the selection, or
            # while it fades out after a deselection, the last one it had
            if name in (SPEECH, CASCADE):
                route_labels = self.speech_labels
            else:
                if labels and self.route_for(labels) == WAVEFORMER:
                    self._wf_labels = labels
                route_labels = labels if self.route_for(labels) == WAVEFORMER else self._wf_labels
            if running and not route_labels:
                running = False
            if not running:  # resting: not processed at all (about 1 ms per chunk saved)
                self._resting.add(name)
                continue
            if name in self._resting:  # starts again from a clean state, freshly delayed
                self._resting.discard(name)
                route.reset()
                self._pads[name] = _Fifo(self._latency - self._route_latency[name])
            self._pads[name].push(route.process(x, route_labels))
            est += weights[name] * self._pads[name].read(n)
        if OFF in weights:
            est += weights[OFF] * dry
        attenuation_db = self.target.attenuation_db if self.target is not None else 0.0
        gains = self.gate.gains(est, dry, labels, attenuation_db)
        g = 10.0 ** (-attenuation_db / 20.0)
        y = gains * (est + g * (dry - est))
        return AudioBuffer(np.ascontiguousarray(y, dtype=np.float32).reshape(-1, 1),
                           chunk.sample_rate, {**chunk.metadata, "extraction": self.method})

    def status_info(self) -> dict[str, Any]:
        info = super().status_info()
        gate = self.gate
        if gate is None or not self.gate_config.enabled:
            gate_state = "off"
        else:
            gate_state = "closed" if gate.closed else "open"
        info.update(
            target_active=bool(self.target is not None and self.target.labels),
            route=self.route,
            routes=[name for name in self._routes if name != OFF],
            gate=bool(self.gate_config.enabled),
            gate_state=gate_state,
            gate_closed=round(gate.closed_share, 3) if gate is not None else None,
            gate_reopened=gate.reopened if gate is not None else None,
            gating=self.activity is not None,
            performance_cores=self.pinned is not None,
            notes=list(self.notes),
        )
        for route in self._routes.values():
            info.update(route.status())
        part = info.get("speech_enhancer") if self.route in (SPEECH, CASCADE) else None
        part = part or info.get("waveformer") or info.get("speech_enhancer") or {}
        info.update(device=part.get("device"), model=part.get("model"),
                    model_ms=part.get("model_ms"), model_ms_p95=part.get("model_ms_p95"))
        return info


def build_hybrid_extractor(config, activity: ActivitySource | None = None, *,
                           device: str | None = None, gate: bool | None = None
                           ) -> HybridExtractor:
    """The hybrid extractor from the configuration. A missing route is noted (status); with
    neither route available ``HybridUnavailable`` says why."""
    from soundspy.config import speech_model_dir
    from soundspy.extraction.neural_tse import (
        NeuralUnavailable,
        WaveformerStream,
        load_neural_core,
    )
    from soundspy.extraction.speech_enhancer import (
        SpeechEnhancer,
        SpeechUnavailable,
        load_speech_core,
    )

    hy = config.extraction.hybrid
    neural = config.extraction.neural_tse
    notes: list[str] = []
    waveformer = cascade = None
    if neural.checkpoint is None:
        notes.append("no Waveformer checkpoint is configured (extraction.neural_tse.checkpoint)")
    else:
        try:
            core = load_neural_core(Path(neural.checkpoint), device or neural.live.device)
        except NeuralUnavailable as exc:
            notes.append(f"Waveformer: {exc}")
        else:
            waveformer = WaveformerStream(core, neural.live, None)  # no 07b output gate
            if hy.speech.route == "cascade":
                cascade = WaveformerStream(core, neural.live, None)
    speech = None
    try:
        sp = hy.speech
        speech_core = load_speech_core(speech_model_dir(config), sp.device,
                                       post_filter=sp.post_filter, backend=sp.backend,
                                       cpu_threads=sp.cpu_threads)
        speech = SpeechEnhancer(speech_core, atten_lim_db=sp.atten_lim_db,
                                cpu_threads=sp.cpu_threads)
    except SpeechUnavailable as exc:
        notes.append(f"speech enhancer: {exc}")
    if waveformer is None and speech is None:
        raise HybridUnavailable("; ".join(notes))
    return HybridExtractor(hy, waveformer=waveformer, speech=speech, cascade_waveformer=cascade,
                           activity=activity, gate=gate, notes=notes)

"""``build_chain``: the processing chain for each mode.

``SEMANTIC``::

    main:       ChannelReducer("mean")  ->  extractor (hybrid | neural_tse | mask_baseline |
                                            passthrough: ``method`` argument, default
                                            extraction.method)
    side:       analyzer (input chunk, all channels; source="input")
                detector (when a trained checkpoint is available)
    post_side:  output analyzer (the chain's output; source="output"; levels and spectrogram,
                no waveform: nothing shows it)

**Gate** (steps 07b, 07c): ``gate`` switches the method's gate on or off for this chain: the
hybrid gate (``hybrid``) or the neural output gate (``neural_tse``); ``None``: the method's own
setting. The baseline ignores it.

**Hybrid** (step 07c) is built before the detector like the neural extractor (its Waveformer's
device decides where the detector runs) and gets the detector's activity afterwards
(``set_activity``, with the detector's per-class thresholds for "confidently absent"). A missing
route is a ``hybrid_route_unavailable`` warning; neither route: the baseline with
``extraction_fallback``.

**Method** (step 07). ``neural_tse`` needs its checkpoint; without it the chain is built with the
baseline instead and carries an ``extraction_fallback`` warning. The neural extractor does not
use the detector; the detector still runs for the chips and the timeline, on the CPU when the
extractor uses the graphics card (see :func:`detector_device`).

The detector and the extractor share one ``DetectionState``, built here and handed to both: the
detector writes its newest decision, the extractor reads whether the selected classes are active.
The signal after the reducer is the chain's **dry** signal (``dry_after="reducer"``), which the
session plays instead of the output when bypass is on.

Without a usable detection checkpoint the chain is built without the detector and carries two
warnings the session emits when it starts: ``detector_unavailable``, and (for the baseline)
``gating_unavailable``: with no detector the extractor cannot tell when the selected sounds are
present, so it falls back to plain noise reduction without gating.

The reducer comes first because playback is mono-to-headphones and the extractor takes a single
channel; the input analyzer still sees every input channel. ``SPATIAL`` and ``SOURCES`` need a
microphone array and their stages (modules 3 and 5).
"""

from __future__ import annotations

from soundspy.analysis.analyzer import Analyzer
from soundspy.config import AppConfig
from soundspy.core.labels import LabelSet
from soundspy.dsp.channels import ChannelReducer
from soundspy.engine.chain import ProcessorChain
from soundspy.engine.detection_state import DetectionState
from soundspy.engine.modes import Mode
from soundspy.extraction.activity import LiveDetectorActivity
from soundspy.extraction.offline import ExtractionUnavailable, build_extractor, resolve_method

GATING_UNAVAILABLE = (
    "Selecting sounds cannot switch the rest off in this session: without the detection model "
    "the app does not know when a selected sound is present. Selected sounds only get steady-"
    "noise reduction.")
HYBRID_GATING_UNAVAILABLE = (
    "The hybrid method's gate cannot close in this session: without the detection model the app "
    "is never sure that the selected sounds are absent. Extraction still works; pauses are only "
    "as quiet as the extractor makes them.")


class ModeUnavailableError(ValueError):
    """The requested mode cannot be built (yet)."""


def detector_device(config: AppConfig, extractor) -> str | None:
    """Where the live detector runs: ``None`` = as configured, ``"cpu"`` when the neural
    extractor uses the graphics card and ``detection.live.device`` is ``auto``.

    Two models taking turns on one small graphics card make the audio path wait: an extractor
    call (every 20 ms) that meets a detector inference (25 ms, four times a second) is held
    up until it is over. Measured on the laptop: 5 output underruns in 60 s with both on the
    card, none with the detector on the CPU (``extraction.neural_tse.live.detector_on_cpu``).
    An explicit ``detection.live.device: cuda`` is respected.
    """
    core = getattr(extractor, "core", None)
    device = getattr(getattr(core, "device", None), "type", None)
    if (device == "cuda" and config.extraction.neural_tse.live.detector_on_cpu
            and config.detection is not None and config.detection.live.device == "auto"):
        return "cpu"
    return None


def check_mode(mode: Mode | str) -> Mode:
    """The mode, if a chain can be built for it (else ``ModeUnavailableError``): cheap, so a
    session start can refuse a bad mode before building anything (step 08)."""
    try:
        mode = Mode(str(mode).lower())
    except ValueError:
        modes = [m.value for m in Mode]
        raise ModeUnavailableError(f"unknown mode {mode!r}; modes: {modes}") from None
    if mode is Mode.SPATIAL:
        raise ModeUnavailableError("mode 'spatial' is not available until module 3")
    if mode is Mode.SOURCES:
        raise ModeUnavailableError("mode 'sources' is not available until module 5")
    return mode


def build_chain(mode: Mode | str, config: AppConfig, label_set: LabelSet, *,
                with_detector: bool = True, method: str | None = None,
                gate: bool | None = None) -> ProcessorChain:
    mode = check_mode(mode)

    reducer = ChannelReducer("mean")
    analyzer = Analyzer(config.analysis, source="input")
    output_analyzer = Analyzer(config.analysis, source="output", waveform=False)
    side: list = [analyzer]
    names: dict = {"reducer": reducer, "analyzer": analyzer, "output_analyzer": output_analyzer}
    warnings: list[tuple[str, str]] = []
    state = DetectionState()
    activity = None

    # The neural extractors are built first: where they run decides where the detector may run.
    method, fallback = resolve_method(config, method)
    fallbacks = [("extraction_fallback", fallback)] if fallback is not None else []
    extractor = None
    if method in ("neural_tse", "hybrid"):
        try:
            extractor = build_extractor(config, None, method=method, gate=gate)
        except ExtractionUnavailable as exc:  # e.g. the checkpoint cannot be loaded
            name = "Hybrid" if method == "hybrid" else "Neural"
            fallbacks.append(("extraction_fallback",
                              f"{name} extraction is not available: {exc} Using the baseline "
                              "(detector-gated) method instead."))
            method = "mask_baseline"
    if method == "hybrid" and extractor is not None:
        for note in getattr(extractor, "notes", []):
            fallbacks.append(("hybrid_route_unavailable",
                              f"Hybrid extraction runs without one of its routes: {note}"))

    if with_detector and config.detection is not None:
        from soundspy.detection.inference import DetectorUnavailable, build_detector

        try:
            detector = build_detector(config, state=state,
                                      device=detector_device(config, extractor))
        except DetectorUnavailable as exc:
            warnings.append(("detector_unavailable",
                             f"Sound detection is off for this session: {exc}"))
        else:
            side.append(detector)
            names["detector"] = detector
            activity = LiveDetectorActivity(
                state, lag_s=config.extraction.mask_baseline.noise_guard_s,
                thresholds=getattr(getattr(detector, "core", None), "thresholds", None))
    warnings.extend(fallbacks)
    if extractor is None:
        extractor = build_extractor(config, activity, method=method)
    elif method == "hybrid":
        extractor.set_activity(activity)
    names["extractor"] = extractor
    if activity is None and method == "mask_baseline":
        warnings.append(("gating_unavailable", GATING_UNAVAILABLE))
    if activity is None and method == "hybrid":
        warnings.append(("gating_unavailable", HYBRID_GATING_UNAVAILABLE))
    chain = ProcessorChain([reducer, extractor], side, post_side=[output_analyzer], names=names,
                           dry_after="reducer")
    chain.detection_state = state
    chain.warnings.extend(warnings)
    return chain

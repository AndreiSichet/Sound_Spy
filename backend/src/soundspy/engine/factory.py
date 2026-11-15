"""``build_chain``: the processing chain for each mode.

``SEMANTIC``::

    main:       ChannelReducer("mean")  ->  extractor (extraction.method: mask_baseline |
                                            passthrough; neural_tse arrives in step 07)
    side:       analyzer (input chunk, all channels; source="input")
                detector (when a trained checkpoint is available)
    post_side:  output analyzer (the chain's output; source="output")

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
from soundspy.extraction.offline import build_extractor

GATING_UNAVAILABLE = (
    "Selecting sounds cannot switch the rest off in this session: without the detection model "
    "the app does not know when a selected sound is present. Selected sounds only get steady-"
    "noise reduction.")


class ModeUnavailableError(ValueError):
    """The requested mode cannot be built (yet)."""


def build_chain(mode: Mode | str, config: AppConfig, label_set: LabelSet, *,
                with_detector: bool = True) -> ProcessorChain:
    try:
        mode = Mode(str(mode).lower())
    except ValueError:
        modes = [m.value for m in Mode]
        raise ModeUnavailableError(f"unknown mode {mode!r}; modes: {modes}") from None
    if mode is Mode.SPATIAL:
        raise ModeUnavailableError("mode 'spatial' is not available until module 3")
    if mode is Mode.SOURCES:
        raise ModeUnavailableError("mode 'sources' is not available until module 5")

    reducer = ChannelReducer("mean")
    analyzer = Analyzer(config.analysis, source="input")
    output_analyzer = Analyzer(config.analysis, source="output")
    side: list = [analyzer]
    names: dict = {"reducer": reducer, "analyzer": analyzer, "output_analyzer": output_analyzer}
    warnings: list[tuple[str, str]] = []
    state = DetectionState()
    activity = None
    if with_detector and config.detection is not None:
        from soundspy.detection.inference import DetectorUnavailable, build_detector

        try:
            detector = build_detector(config, state=state)
        except DetectorUnavailable as exc:
            warnings.append(("detector_unavailable",
                             f"Sound detection is off for this session: {exc}"))
        else:
            side.append(detector)
            names["detector"] = detector
            activity = LiveDetectorActivity(
                state, lag_s=config.extraction.mask_baseline.noise_guard_s)
    extractor = build_extractor(config, activity)
    names["extractor"] = extractor
    if activity is None and config.extraction.method == "mask_baseline":
        warnings.append(("gating_unavailable", GATING_UNAVAILABLE))
    chain = ProcessorChain([reducer, extractor], side, post_side=[output_analyzer], names=names,
                           dry_after="reducer")
    chain.detection_state = state
    chain.warnings.extend(warnings)
    return chain

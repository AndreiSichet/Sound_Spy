"""``build_chain``: the processing chain for each mode.

``SEMANTIC``::

    main:  ChannelReducer("mean")  ->  extractor (Passthrough for now; steps 06-07 replace it)
    side:  analyzer (on the input chunk, all channels)
           detector (step 05; when a trained checkpoint is available)

Without a usable detection checkpoint the chain is built without the detector and carries a
``detector_unavailable`` warning, which the session emits when it starts.

The reducer comes first because playback is mono-to-headphones and extraction models will take
a single channel; the analyzer still sees every input channel. ``SPATIAL`` and ``SOURCES`` need a
microphone array and their stages (modules 3 and 5).
"""

from __future__ import annotations

from soundspy.analysis.analyzer import Analyzer
from soundspy.config import AppConfig
from soundspy.core.labels import LabelSet
from soundspy.dsp.channels import ChannelReducer
from soundspy.engine.chain import ProcessorChain
from soundspy.engine.modes import Mode
from soundspy.engine.processor import Passthrough


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
    extractor = Passthrough()  # TODO(step-06): detector-guided extractor; label_set feeds it
    analyzer = Analyzer(config.analysis)
    side: list = [analyzer]
    names: dict = {"reducer": reducer, "extractor": extractor, "analyzer": analyzer}
    warning = None
    if with_detector and config.detection is not None:
        from soundspy.detection.inference import DetectorUnavailable, build_detector

        try:
            detector = build_detector(config)
        except DetectorUnavailable as exc:
            warning = f"Sound detection is off for this session: {exc}"
        else:
            side.append(detector)
            names["detector"] = detector
    chain = ProcessorChain([reducer, extractor], side, names=names)
    if warning is not None:
        chain.warnings.append(("detector_unavailable", warning))
    return chain

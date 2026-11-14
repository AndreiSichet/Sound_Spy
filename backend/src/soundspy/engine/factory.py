"""``build_chain``: the processing chain for each mode.

``SEMANTIC`` (step 03)::

    main:  ChannelReducer("mean")  ->  extractor (Passthrough for now; steps 05-07 replace it)
    side:  analyzer (on the input chunk, all channels)

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


def build_chain(mode: Mode | str, config: AppConfig, label_set: LabelSet) -> ProcessorChain:
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
    extractor = Passthrough()  # TODO(step-05): detector-guided extractor; label_set feeds it
    analyzer = Analyzer(config.analysis)
    return ProcessorChain(
        [reducer, extractor],
        [analyzer],
        names={"reducer": reducer, "extractor": extractor, "analyzer": analyzer},
    )

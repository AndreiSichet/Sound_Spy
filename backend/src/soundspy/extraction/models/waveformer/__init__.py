"""Waveformer (Veluri et al., ICASSP 2023), adapted from https://github.com/vb000/Waveformer
(MIT, © 2022 Bandhav Veluri; see ``LICENSE`` and the header of ``model.py``)."""

from soundspy.extraction.models.waveformer.model import (
    FSDKAGGLE2018_CLASSES,
    Net,
    PositionalEncoding,
)

__all__ = ["FSDKAGGLE2018_CLASSES", "Net", "PositionalEncoding"]

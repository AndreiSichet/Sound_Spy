"""Dataset path aliases: ``fsd50k://FSD50K.dev_audio/64760.wav`` instead of absolute paths.

The clip index stores only aliased, dataset-relative paths. They are resolved through the
``paths`` section of ``default.yaml`` when a clip is loaded, so the whole data folder can move
(another drive, another machine) by changing the configuration only.

Where each dataset lives:

========  ===============================================  =========================
dataset   root                                             example alias
========  ===============================================  =========================
esc50     ``<external>/esc50/ESC-50-master``               ``esc50://audio/1-100032-A-0.wav``
fsd50k    ``<external>/fsd50k``                            ``fsd50k://FSD50K.dev_audio/64760.wav``
custom    ``<custom>``                                     ``custom://speech/speech_001.wav``
========  ===============================================  =========================

``librispeech`` (step 07b) lives in ``<external>/librispeech/LibriSpeech``; an alias looks like
``librispeech://train-clean-100/19/198/19-198-0000.flac``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

ESC50_ARCHIVE_DIR = "ESC-50-master"
"""Top-level folder inside the ESC-50 GitHub archive."""

_ROOTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "esc50": ("external", ("esc50", ESC50_ARCHIVE_DIR)),
    "fsd50k": ("external", ("fsd50k",)),
    "custom": ("custom", ()),
    "librispeech": ("external", ("librispeech", "LibriSpeech")),
}


class AliasError(ValueError):
    """An aliased path is malformed or names an unknown dataset."""


def dataset_root(dataset: str, paths: Mapping[str, Path]) -> Path:
    """Absolute folder that the alias ``<dataset>://`` points to."""
    try:
        key, parts = _ROOTS[dataset]
    except KeyError:
        raise AliasError(f"unknown dataset {dataset!r} (known: {sorted(_ROOTS)})") from None
    return Path(paths[key]).joinpath(*parts)


def to_alias(dataset: str, relative: str | PurePosixPath) -> str:
    """``("fsd50k", "FSD50K.dev_audio/1.wav")`` -> ``"fsd50k://FSD50K.dev_audio/1.wav"``."""
    rel = PurePosixPath(str(relative).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise AliasError(f"alias paths must be relative and stay inside the dataset: {relative}")
    if dataset not in _ROOTS:
        raise AliasError(f"unknown dataset {dataset!r}")
    return f"{dataset}://{rel.as_posix()}"


def split_alias(alias: str) -> tuple[str, PurePosixPath]:
    dataset, sep, rest = alias.partition("://")
    if not sep or not rest:
        raise AliasError(f"not an aliased path: {alias!r}")
    rel = PurePosixPath(rest)
    if rel.is_absolute() or ".." in rel.parts:
        raise AliasError(f"alias escapes its dataset: {alias!r}")
    return dataset, rel


def resolve_alias(alias: str, paths: Mapping[str, Path]) -> Path:
    """Aliased path -> absolute file path, using the configured ``paths``."""
    dataset, rel = split_alias(alias)
    return dataset_root(dataset, paths).joinpath(*rel.parts)

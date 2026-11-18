"""LibriSpeech collector (``<external>/librispeech/LibriSpeech``), step 07b.

Read English speech from LibriVox audiobooks, 16 kHz mono FLAC, CC BY 4.0::

    LibriSpeech/<subset>/<speaker>/<chapter>/<speaker>-<chapter>-<utterance>.flac

Every utterance is one clip, foreground for the class named in ``datasets.librispeech.class``
of the data config (the speech class). It is added on top of FSD50K's speech, which is everyday
speech recorded in every imaginable way but only a few hundred clips: the mixer draws from both
(``mixtures.source_mix``), so the extractor hears read **and** casual speech.

* **Split = the official subset** (``splits.librispeech``: ``train-clean-100`` -> train,
  ``dev-clean`` -> val, ``test-clean`` -> test). LibriSpeech's subsets share no speaker, so
  nothing has to be moved; the leakage check of ``prepare`` verifies it anyway.
* **Group = the speaker.**
* Utterances are long (12 s on average, up to 35 s). They are indexed whole; the mixer cuts
  events out of them like out of any other clip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from soundspy.data.aliases import dataset_root, to_alias
from soundspy.data.collectors import ClipRow, CollectorError
from soundspy.data.roles import fg_role

LICENSE = "CC BY 4.0"
RAW_LABEL = "read speech"


def collect(
    paths: Mapping[str, Path],
    subsets: Mapping[str, str],
    class_key: str,
    mids: Sequence[str],
) -> list[ClipRow]:
    """One row per utterance. ``subsets``: split -> subset folder; ``mids``: the class's mids."""
    root = dataset_root("librispeech", paths)
    rows: list[ClipRow] = []
    for split, subset in subsets.items():
        folder = root / subset
        if not folder.is_dir():
            raise CollectorError(f"LibriSpeech subset {subset!r} not found at {folder}")
        found = 0
        for flac in sorted(folder.glob("*/*/*.flac")):
            parts = flac.stem.split("-")
            if len(parts) != 3 or parts[0] != flac.parent.parent.name:
                raise CollectorError(f"unexpected LibriSpeech file name {flac}")
            speaker, chapter = parts[0], parts[1]
            rows.append(ClipRow(
                clip_id=f"librispeech:{flac.stem}",
                dataset="librispeech",
                path=to_alias("librispeech", f"{subset}/{speaker}/{chapter}/{flac.name}"),
                file=flac,
                mids=list(mids),
                raw_labels=[RAW_LABEL],
                split=split,
                group=f"speaker:{speaker}",
                license=LICENSE,
                role=fg_role(class_key),
            ))
            found += 1
        if not found:
            raise CollectorError(f"LibriSpeech subset {subset!r} at {folder} holds no FLAC files")
    return rows

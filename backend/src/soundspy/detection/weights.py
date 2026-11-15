"""Pretrained backbone weights: download once to ``paths.model_cache`` (D:), verified.

Uses the dataset downloader (resumable, retried) with the SHA-256 pinned in
``configs/detection/default.yaml``; a file with a different hash is deleted and refused.
"""

from __future__ import annotations

from pathlib import Path

from soundspy.config import DetectionConfig


def weights_dir(model_cache: Path) -> Path:
    return Path(model_cache) / "efficientat"


def ensure_pretrained(det: DetectionConfig, model_cache: Path, *, progress: bool = True) -> Path:
    """Path of the verified pretrained weights, downloading them if needed."""
    from soundspy.data.download import RemoteFile, download_file, make_session

    folder = weights_dir(model_cache)
    remote = RemoteFile(det.pretrained_file, det.pretrained_url, sha256=det.pretrained_sha256)
    download_file(make_session(), remote, folder, progress=progress)
    return folder / det.pretrained_file

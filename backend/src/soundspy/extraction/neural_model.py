"""The neural extractor's network in Sound Spy: building it, adapting it, saving and loading.

The network itself is Waveformer (``extraction/models/waveformer``, third-party, unchanged).
This module is what Sound Spy adds around it:

* :func:`ensure_pretrained`: the published checkpoint, downloaded once to
  ``<model_cache>/waveformer/`` with the SHA-256 pinned in ``configs/extraction/neural_tse.yaml``.
* :func:`adapt_label_layer`: the published model answers to 41 FSDKaggle2018 classes; Sound Spy
  has its own classes. Only the **first layer of the label network** depends on the class list
  (a ``Linear(41, 512)``: one column of 512 weights per class). It is replaced by a
  ``Linear(len(labels), 512)`` whose column for each of our classes is the **mean of the
  pretrained columns** named in ``label_init`` (e.g. ``chair_scrape`` <- ``Squeak``,
  ``Drawer_open_or_close``), or the mean of all 41 columns when none is named (``speech``). The
  bias and every other weight are kept, so fine-tuning starts from a model that already extracts
  "something like" each class instead of from noise.
* :func:`build_finetune_model`: the adapted network, ready for fine-tuning.
* :func:`save_checkpoint` / :func:`load_checkpoint`: Sound Spy's own checkpoint format, which
  carries the architecture and the class list, so loading needs no configuration file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from soundspy.config import NeuralTSEConfig, WaveformerConfig
from soundspy.extraction.models.waveformer import FSDKAGGLE2018_CLASSES, Net

CHECKPOINT_VERSION = 1
LABEL_LAYER = "label_embedding.0"  # the only layer whose shape depends on the class list


class NeuralModelError(RuntimeError):
    """The neural extractor's model cannot be built or loaded. Message for the user."""


def weights_dir(model_cache: Path) -> Path:
    return Path(model_cache) / "waveformer"


def ensure_pretrained(cfg: NeuralTSEConfig, model_cache: Path, *, progress: bool = True) -> Path:
    """Path of the verified published checkpoint, downloading it if needed."""
    from soundspy.data.download import RemoteFile, download_file, make_session

    folder = weights_dir(model_cache)
    remote = RemoteFile(cfg.pretrained_file, cfg.pretrained_url, sha256=cfg.pretrained_sha256)
    download_file(make_session(), remote, folder, progress=progress)
    return folder / cfg.pretrained_file


def build_net(model: WaveformerConfig, label_len: int) -> Net:
    return Net(**model.net_kwargs(label_len))


def count_parameters(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def _numpy_data_globals() -> list[Any]:
    """The NumPy *data* constructors the published checkpoint needs besides tensors.

    It stores a few training statistics as NumPy scalars. They are plain numbers; allowing
    exactly these keeps ``weights_only=True`` (no arbitrary code is ever unpickled from the
    downloaded file).
    """
    import numpy as np

    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # NumPy 1.x
        from numpy.core.multiarray import scalar
    kinds = (np.float64, np.float32, np.int64, np.int32, np.bool_)
    # the file (written with NumPy 1.x) names the constructor by its old module path
    return [scalar, (scalar, "numpy.core.multiarray.scalar"), np.dtype, *kinds,
            *{type(np.dtype(k)) for k in kinds}]


def load_pretrained_state(path: Path) -> dict[str, torch.Tensor]:
    """The published checkpoint's weights, loaded with ``weights_only=True``: only tensors,
    containers and the NumPy numbers of :func:`_numpy_data_globals` are unpickled."""
    try:
        with torch.serialization.safe_globals(_numpy_data_globals()):
            blob = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any loading problem becomes a user message
        raise NeuralModelError(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else None
    if not isinstance(state, dict) or f"{LABEL_LAYER}.weight" not in state:
        raise NeuralModelError(f"{path} is not a Waveformer checkpoint")
    return state


def label_columns(labels: Sequence[str], label_init: Mapping[str, Sequence[str]],
                  pretrained_classes: Sequence[str] = FSDKAGGLE2018_CLASSES
                  ) -> list[list[int]]:
    """For each of our classes, the pretrained column indices its column is averaged from
    (all of them when ``label_init`` names none)."""
    index = {name: i for i, name in enumerate(pretrained_classes)}
    columns = []
    for label in labels:
        sources = list(label_init.get(label, ()))
        unknown = [s for s in sources if s not in index]
        if unknown:
            raise NeuralModelError(
                f"label_init.{label}: {unknown} are not classes of the pretrained model")
        columns.append([index[s] for s in sources] or list(range(len(pretrained_classes))))
    return columns


def adapt_label_layer(state: Mapping[str, torch.Tensor], labels: Sequence[str],
                      label_init: Mapping[str, Sequence[str]]) -> dict[str, torch.Tensor]:
    """A copy of ``state`` whose first label layer takes ``len(labels)`` inputs (see module
    docstring). Everything else is untouched."""
    weight = state[f"{LABEL_LAYER}.weight"]  # (512, 41)
    if weight.shape[1] != len(FSDKAGGLE2018_CLASSES):
        raise NeuralModelError(
            f"expected {len(FSDKAGGLE2018_CLASSES)} pretrained classes, got {weight.shape[1]}")
    new = torch.stack([weight.index_select(1, torch.tensor(cols)).mean(dim=1)
                       for cols in label_columns(labels, label_init)], dim=1)
    adapted = dict(state)
    adapted[f"{LABEL_LAYER}.weight"] = new.contiguous()
    return adapted


def build_finetune_model(cfg: NeuralTSEConfig, labels: Sequence[str],
                         pretrained: Path | None) -> Net:
    """The network for our classes: adapted pretrained weights, or random ones (tests, or an
    explicit choice) when ``pretrained`` is ``None``."""
    net = build_net(cfg.model, len(labels))
    if pretrained is not None:
        state = adapt_label_layer(load_pretrained_state(pretrained), labels, cfg.label_init)
        try:
            net.load_state_dict(state)
        except RuntimeError as exc:
            raise NeuralModelError(
                f"the pretrained weights do not fit the configured model: {exc}") from exc
    return net


def parameter_groups(net: Net, lr_backbone: float, lr_label: float) -> list[dict[str, Any]]:
    """Two optimiser groups: the new first label layer learns faster than the rest."""
    new = [p for n, p in net.named_parameters() if n.startswith(LABEL_LAYER + ".")]
    rest = [p for n, p in net.named_parameters() if not n.startswith(LABEL_LAYER + ".")]
    return [{"params": rest, "lr": lr_backbone, "name": "backbone"},
            {"params": new, "lr": lr_label, "name": "label"}]


def save_checkpoint(path: Path, net: Net, model: WaveformerConfig, labels: Sequence[str],
                    extra: Mapping[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"version": CHECKPOINT_VERSION, "model_config": asdict(model),
            "labels": list(labels), "model": net.state_dict(), **dict(extra or {})}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, str(tmp))
    tmp.replace(path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu"
                    ) -> tuple[Net, WaveformerConfig, list[str], dict[str, Any]]:
    """``(net in eval mode, architecture, class list, the rest of the checkpoint)``."""
    path = Path(path)
    if not path.is_file():
        raise NeuralModelError(
            f"no neural extraction checkpoint at {path} (run `soundspy train extraction`)")
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
        if blob.get("version") != CHECKPOINT_VERSION:
            raise NeuralModelError(f"{path}: unsupported checkpoint version {blob.get('version')}")
        model = WaveformerConfig(**blob["model_config"])
        labels = list(blob["labels"])
        net = build_net(model, len(labels))
        net.load_state_dict(blob["model"])
    except NeuralModelError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise NeuralModelError(f"loading {path} failed: {type(exc).__name__}: {exc}") from exc
    extra = {k: v for k, v in blob.items() if k not in ("model", "model_config", "labels")}
    return net.to(device).eval(), model, labels, extra

"""The speech enhancer's network in Sound Spy: weights, settings, building and loading (step 07c).

The network is DeepFilterNet3 (``extraction/models/deepfilternet``, third-party, adapted). This
module is what Sound Spy adds around it:

* :func:`ensure_weights`: the published model archive (``DeepFilterNet3.zip`` of release
  0.5.6), downloaded once to ``<model_cache>/deepfilternet/`` with the SHA-256 pinned in
  ``configs/extraction/hybrid.yaml``, and unpacked next to it (``config.ini`` +
  ``checkpoints/model_120.ckpt.best``).
* :func:`read_params`: the model's own ``config.ini`` -> ``DfParams``.
* :func:`load_model`: the network with the published weights (``weights_only=True``: only
  tensors are unpickled), every key matched (strict).
* :func:`build_model`: a network with random weights for any ``DfParams`` (tests use a tiny one).
"""

from __future__ import annotations

import configparser
from dataclasses import fields, replace
from pathlib import Path

from soundspy.extraction.models.deepfilternet import DfNet, DfParams, erb_matrices, erb_widths

FOLDER = "deepfilternet"


class SpeechModelError(RuntimeError):
    """The speech enhancer's model cannot be found or loaded. Message for the user."""


def weights_dir(model_cache: Path) -> Path:
    return Path(model_cache) / FOLDER


def ensure_weights(file: str, url: str, sha256: str, model_cache: Path, *,
                   progress: bool = True) -> Path:
    """The unpacked model folder (with ``config.ini``), downloading and unpacking if needed."""
    from soundspy.data.download import RemoteFile, download_file, extract_zip, make_session

    folder = weights_dir(model_cache)
    model_dir = folder / Path(file).stem
    if (model_dir / "config.ini").is_file():
        return model_dir
    download_file(make_session(), RemoteFile(file, url, sha256=sha256), folder,
                  progress=progress)
    extract_zip(folder / file, folder)
    if not (model_dir / "config.ini").is_file():
        raise SpeechModelError(f"{folder / file} did not contain {model_dir.name}/config.ini")
    return model_dir


_SECTIONS = {"df": ("sr", "fft_size", "hop_size", "nb_erb", "nb_df", "norm_tau", "lsnr_max",
                    "lsnr_min", "min_nb_erb_freqs", "df_order", "df_lookahead"),
             "deepfilternet": ("conv_lookahead", "conv_ch", "conv_kernel", "convt_kernel",
                               "conv_kernel_inp", "emb_hidden_dim", "emb_num_layers",
                               "enc_linear_groups", "linear_groups", "df_hidden_dim",
                               "df_num_layers", "df_pathway_kernel_size_t", "df_gru_skip",
                               "emb_gru_skip", "enc_concat")}


def read_params(path: Path) -> DfParams:
    """``DfParams`` from a DeepFilterNet ``config.ini`` (missing keys keep the defaults)."""
    parser = configparser.ConfigParser()
    if not parser.read(Path(path), encoding="utf-8"):
        raise SpeechModelError(f"cannot read {path}")
    kinds = {f.name: f.type for f in fields(DfParams)}
    values: dict[str, object] = {}
    for section, keys in _SECTIONS.items():
        if not parser.has_section(section):
            raise SpeechModelError(f"{path}: no [{section}] section")
        for key in keys:
            if not parser.has_option(section, key):
                continue
            raw = parser.get(section, key).strip()
            kind = str(kinds[key])
            if "tuple" in kind:
                values[key] = tuple(int(v) for v in raw.split(","))
            elif kind == "bool":
                values[key] = raw.lower() in ("1", "true", "yes", "on")
            elif kind == "int":
                values[key] = int(raw)
            elif kind == "float":
                values[key] = float(raw)
            else:
                values[key] = raw
    if parser.has_option("deepfilternet", "pf_beta"):
        values["pf_beta"] = parser.getfloat("deepfilternet", "pf_beta")
    return replace(DfParams(), **values)


def build_model(p: DfParams, *, post_filter: bool = False) -> DfNet:
    """A ``DfNet`` with random weights and the ERB matrices of ``p``."""
    import torch

    fb, inverse = erb_matrices(erb_widths(p))
    return DfNet(p, torch.from_numpy(fb), torch.from_numpy(inverse), post_filter=post_filter)


EXPORT_VERSION = 1  # bump when DfNet.stream changes: cached ONNX files are then made again
STATE_NAMES = ("erb_w", "spec_w", "specs", "c0_w", "h_enc", "h_erb", "h_df")


def onnx_path(model_dir: Path, checkpoint: Path, post_filter: bool) -> Path:
    """Where the exported streaming step of this checkpoint is cached (next to the model)."""
    import hashlib

    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:12]
    name = f"{Path(model_dir).name}_stream_v{EXPORT_VERSION}_{digest}{'_pf' if post_filter else ''}"
    return Path(model_dir).parent / f"{name}.onnx"


def export_onnx(net: DfNet, path: Path) -> Path:
    """Export ``net.stream`` (k frames + the 7 state tensors -> k enhanced frames + new state)
    to ONNX with a variable number of frames ``k``. Written to a temporary name first."""
    import warnings

    import torch

    class Step(torch.nn.Module):
        def __init__(self, inner: DfNet) -> None:
            super().__init__()
            self.net = inner

        def forward(self, spec, erb, low, *state):
            out, new = self.net.stream(spec, erb, low, tuple(state))
            return (out, *new)

    p = net.p
    state = net.init_state("cpu")
    args = (torch.zeros(1, 1, 2, p.freq_bins, 2), torch.zeros(1, 1, 2, p.nb_erb),
            torch.zeros(1, 1, 2, p.nb_df, 2), *state)
    inputs = ["spec", "erb", "low", *STATE_NAMES]
    outputs = ["out", *(f"{n}_out" for n in STATE_NAMES)]
    frames = {"spec": {2: "k"}, "erb": {2: "k"}, "low": {2: "k"}, "out": {2: "k"}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".onnx.part")
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # TorchScript-based exporter: deprecation notices
        torch.onnx.export(Step(net.cpu().eval()).eval(), args, str(temp), input_names=inputs,
                          output_names=outputs, dynamic_axes=frames, opset_version=17,
                          dynamo=False)
    temp.replace(path)
    return path


def _checkpoint_file(model_dir: Path) -> Path:
    folder = Path(model_dir) / "checkpoints"
    best = sorted(folder.glob("model_*.ckpt.best"))
    if best:
        return best[-1]
    plain = sorted(folder.glob("model_*.ckpt"),
                   key=lambda p: int(p.name.split("_")[1].split(".")[0]))
    if not plain:
        raise SpeechModelError(f"no checkpoint in {folder}")
    return plain[-1]


def load_model(model_dir: Path, *, post_filter: bool = False) -> tuple[DfNet, DfParams, Path]:
    """``(network in eval mode, its settings, the checkpoint file)`` from an unpacked model."""
    import torch

    model_dir = Path(model_dir)
    p = read_params(model_dir / "config.ini")
    net = build_model(p, post_filter=post_filter)
    path = _checkpoint_file(model_dir)
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any loading problem becomes a user message
        raise SpeechModelError(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(state, dict):
        raise SpeechModelError(f"{path} is not a DeepFilterNet checkpoint")
    state = {k.replace("clc", "df"): v for k, v in state.items()}  # as DeepFilterNet does
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SpeechModelError(f"{path} does not fit the DeepFilterNet3 network: {exc}") from exc
    return net.eval(), p, path

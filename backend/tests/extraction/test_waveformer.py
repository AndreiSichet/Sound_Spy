"""The adapted Waveformer: parity with the original code, streaming, label-layer adaptation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from soundspy.config import WaveformerConfig
from soundspy.extraction.models.waveformer import FSDKAGGLE2018_CLASSES, Net, PositionalEncoding
from soundspy.extraction.neural_model import (
    LABEL_LAYER,
    NeuralModelError,
    adapt_label_layer,
    build_finetune_model,
    build_net,
    count_parameters,
    label_columns,
    load_checkpoint,
    load_pretrained_state,
    parameter_groups,
    save_checkpoint,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TINY = dict(sample_rate=16_000, L=4, enc_dim=16, num_enc_layers=3, dec_dim=8, num_dec_layers=1,
            dec_buf_len=3, dec_chunk_size=3, out_buf_len=2, use_pos_enc=True)


def test_positional_encoding_matches_the_original_module() -> None:
    """Reference: speechbrain's PositionalEncoding(16, max_len=200) on a (1, 26, 16) input,
    produced by executing the original class source (scratch script of step 07)."""
    reference = np.load(FIXTURES / "waveformer_posenc_reference.npy")
    ours = PositionalEncoding(16, max_len=200)
    out = ours(torch.zeros(1, 26, 16)).numpy()
    assert out.shape == reference.shape == (1, 26, 16)
    np.testing.assert_array_equal(out, reference)
    assert ours.pe.shape == (1, 200, 16)  # same buffer name and shape as the checkpoint's
    assert list(ours.state_dict()) == ["pe"] and not list(ours.parameters())
    assert ours(torch.zeros(3, 5, 16)).shape == (1, 5, 16)  # positions only, not added to x
    with pytest.raises(ValueError):
        PositionalEncoding(15)


def test_forward_pass_matches_the_original_code() -> None:
    """Reference: the ORIGINAL dcc_tf.py run on a tiny random model; its weights, input and
    output are stored, so our port must reproduce the output from the same weights."""
    ref = np.load(FIXTURES / "waveformer_tiny_reference.npz")
    config = json.loads(str(ref["config"]))
    net = Net(**config).eval()
    state = {key[len("state/"):]: torch.from_numpy(ref[key]) for key in ref.files
             if key.startswith("state/")}
    assert set(state) == set(net.state_dict())  # same parameter names as the original
    net.load_state_dict(state)
    with torch.inference_mode():
        out = net(torch.from_numpy(ref["x"]), torch.from_numpy(ref["label"])).numpy()
    np.testing.assert_allclose(out, ref["y"], atol=1e-6)
    assert out.shape == ref["x"].shape  # same length as the input, whatever its length


def tiny_net(labels: int = 5, seed: int = 0) -> Net:
    torch.manual_seed(seed)
    return build_net(WaveformerConfig(**TINY), labels).eval()


def test_streaming_steps_equal_the_whole_signal() -> None:
    net = tiny_net()
    hop, context, look = net.stream_sizes
    assert (hop, context, look) == (12, 4, 4)  # dec_chunk_size * L, L, L
    torch.manual_seed(1)
    x = torch.randn(1, 1, 40 * hop)
    label = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0]])
    padded = torch.nn.functional.pad(x, (context, look))
    with torch.inference_mode():
        whole = net(x, label)
        buffers = net.init_buffers(1, "cpu")
        pieces, start = [], 0
        for steps in (1, 1, 3, 2, 5, 1, 7, 20):  # any number of whole steps per call
            size = context + steps * hop + look
            out, buffers = net.stream_step(padded[..., start:start + size], label, buffers)
            assert out.shape[-1] == steps * hop
            pieces.append(out)
            start += steps * hop
    np.testing.assert_allclose(torch.cat(pieces, dim=-1).numpy(), whole.numpy(), atol=1e-5)


def test_the_model_is_causal() -> None:
    """Changing the input after sample t never changes the output before t - lookahead."""
    net = tiny_net()
    torch.manual_seed(2)
    x = torch.randn(1, 1, 480)
    label = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    changed = x.clone()
    changed[..., 300:] = torch.randn(1, 1, 180)
    with torch.inference_mode():
        a, b = net(x, label), net(changed, label)
    hop, _, look = net.stream_sizes
    safe = (300 - look) // hop * hop  # whole steps that end before the change is visible
    np.testing.assert_allclose(a[..., :safe].numpy(), b[..., :safe].numpy(), atol=1e-6)
    assert not np.allclose(a[..., 300:].numpy(), b[..., 300:].numpy())


def fake_published_state(seed: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return dict(build_net(WaveformerConfig(**TINY), len(FSDKAGGLE2018_CLASSES)).state_dict())


def test_label_layer_is_rebuilt_from_the_mapped_pretrained_columns() -> None:
    state = fake_published_state()
    labels = ["speech", "music", "chair_scrape"]
    mapping = {"music": ("Cello", "Flute"), "chair_scrape": ("Squeak",)}  # speech: none
    adapted = adapt_label_layer(state, labels, mapping)
    old = state[f"{LABEL_LAYER}.weight"]
    new = adapted[f"{LABEL_LAYER}.weight"]
    assert old.shape == (512, 41) and new.shape == (512, 3)
    names = list(FSDKAGGLE2018_CLASSES)
    column = lambda i: new.index_select(1, torch.tensor([i])).ravel()  # noqa: E731
    source = lambda n: old.index_select(1, torch.tensor([names.index(n)])).ravel()  # noqa: E731
    torch.testing.assert_close(column(0), old.mean(dim=1))  # no equivalent: mean of all 41
    torch.testing.assert_close(column(1), (source("Cello") + source("Flute")) / 2)
    torch.testing.assert_close(column(2), source("Squeak"))
    # everything else is untouched, the bias included
    for key, value in state.items():
        if key != f"{LABEL_LAYER}.weight":
            assert adapted[key] is value
    assert label_columns(labels, mapping)[0] == list(range(41))
    with pytest.raises(NeuralModelError, match="Trombone"):
        adapt_label_layer(state, labels, {"music": ("Trombone",)})


def test_finetune_model_parameter_groups_and_checkpoint_roundtrip(config, tmp_path) -> None:
    import dataclasses

    cfg = dataclasses.replace(config.extraction.neural_tse, model=WaveformerConfig(**TINY))
    labels = list(config.labels)
    published = tmp_path / "published.pt"
    # like the real file: weights under "model_state_dict" next to NumPy numbers
    torch.save({"model_state_dict": fake_published_state(), "epoch": 149,
                "val_metric": np.float64(11.5)}, published)
    assert f"{LABEL_LAYER}.weight" in load_pretrained_state(published)
    net = build_finetune_model(cfg, labels, published)
    assert net.label_embedding[0].in_features == len(labels) == net.label_len
    groups = parameter_groups(net, 2.5e-4, 1e-3)
    assert [g["name"] for g in groups] == ["backbone", "label"]
    assert sum(p.numel() for p in groups[1]["params"]) == 512 * len(labels) + 512
    assert sum(p.numel() for g in groups for p in g["params"]) == count_parameters(net)
    assert groups[0]["lr"] == 2.5e-4 and groups[1]["lr"] == 1e-3

    path = tmp_path / "exp" / "checkpoints" / "best.pt"
    save_checkpoint(path, net, cfg.model, labels, {"epoch": 3, "val_si_sdri": 4.5})
    loaded, model, loaded_labels, extra = load_checkpoint(path)
    assert model == cfg.model and loaded_labels == labels and extra["epoch"] == 3
    assert not loaded.training
    x, label = torch.randn(1, 1, 120), torch.zeros(1, len(labels))
    with torch.inference_mode():
        torch.testing.assert_close(loaded(x, label), net.eval()(x, label))
    assert not list(path.parent.glob("*.tmp"))

    random_net = build_finetune_model(cfg, labels, None)  # tests: no download needed
    assert count_parameters(random_net) == count_parameters(net)
    with pytest.raises(NeuralModelError, match="soundspy train extraction"):
        load_checkpoint(tmp_path / "missing.pt")
    (tmp_path / "junk.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(NeuralModelError):
        load_checkpoint(tmp_path / "junk.pt")
    torch.save({"something": 1}, tmp_path / "other.pt")
    with pytest.raises(NeuralModelError, match="not a Waveformer checkpoint"):
        load_pretrained_state(tmp_path / "other.pt")


def test_default_model_size_matches_the_published_checkpoint(config) -> None:
    """The configured architecture (E512 / D256): 3.88 M parameters with 41 classes, as the
    published checkpoint; 3.86 M with Sound Spy's 7."""
    model = config.extraction.neural_tse.model
    assert count_parameters(build_net(model, 41)) == 3_878_144
    assert count_parameters(build_net(model, len(config.labels))) == 3_860_736
    net = build_net(model, len(config.labels))
    assert net.stream_sizes == (416, 32, 32)  # 9.4 ms steps, 0.7 ms lookahead at 44.1 kHz
    unknown = {c for sources in config.extraction.neural_tse.label_init.values()
               for c in sources} - set(FSDKAGGLE2018_CLASSES)
    assert not unknown  # every mapped class exists in the checkpoint's class list

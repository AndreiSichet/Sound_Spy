# MIT License
#
# Copyright (c) 2022 Bandhav Veluri
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Waveformer: a causal, label-conditioned target sound extraction network.

Adapted from ``src/training/dcc_tf.py`` of https://github.com/vb000/Waveformer (MIT, see the
header and ``LICENSE`` in this folder), the model of "Real-Time Target Sound Extraction"
(Veluri, Chan, Itani, Chen, Yoshioka, Gollakota; ICASSP 2023).

Changes from the original file (the network itself is unchanged, parameter names included, so
the published checkpoint loads as it is):

* only the model is kept: the optimizer, loss and metric helpers (and with them the
  ``torchmetrics`` import) are gone; Sound Spy has its own in ``training/``;
* ``speechbrain``'s ``PositionalEncoding`` is replaced by the small parameter-free class below
  (same buffer name ``pe`` and values; a test compares it with a stored output of the original);
* ``Net.stream_step`` / ``Net.stream_sizes`` document and wrap the streaming call
  (``forward(..., pad=False)`` with the three state buffers);
* docstrings and type hints.

How it works (sizes of the published default: ``L = 32``, 44.1 kHz):

``in_conv``     1-D convolution, kernel ``3L``, stride ``L``: one latent vector per ``L`` samples
                (0.73 ms), seeing ``L`` samples back and ``L`` samples ahead (the *lookahead*).
``encoder``     10 dilated causal depthwise-separable convolutions (dilation 1 .. 512): each
                latent is refined with ~1.5 s of **past** context, never the future.
``label``       the multi-hot class vector goes through a small MLP to a vector that is
                **multiplied** into the encoder output: "which sounds to keep".
``decoder``     one causal transformer layer working on chunks of 13 latents (9.4 ms): each
                chunk attends to itself and the previous 13 latents; it predicts a **mask**.
``out_conv``    the masked latents go through a transposed convolution back to samples.

Streaming: every call consumes ``13 * L`` new samples and returns ``13 * L`` output samples,
carrying three buffers (encoder context, decoder context, output overlap) from call to call.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

#: The 41 classes of FSDKaggle2018 in the order of the published checkpoint's label vector.
FSDKAGGLE2018_CLASSES: tuple[str, ...] = (
    "Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello",
    "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass",
    "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute",
    "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock",
    "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
    "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
    "Writing",
)


class PositionalEncoding(nn.Module):
    """Absolute sinusoidal positions: ``PE(pos, 2i) = sin(pos / 10000^(2i/d))``, ``2i+1``: cos.

    Parameter-free replacement of ``speechbrain.lobes.models.transformer.Transformer.
    PositionalEncoding``: the same ``pe`` buffer ``(1, max_len, d)``; ``forward(x)`` returns the
    first ``x.size(1)`` positions (it does not add them to ``x``).
    """

    def __init__(self, input_size: int, max_len: int = 2500) -> None:
        super().__init__()
        if input_size % 2 != 0:
            raise ValueError(f"sin/cos positional encoding needs an even size, got {input_size}")
        self.max_len = max_len
        pe = torch.zeros(max_len, input_size, requires_grad=False)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float() * -(math.log(10000.0) / input_size))
        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)].clone().detach()


def mod_pad(x: Tensor, chunk_size: int, pad: tuple[int, int]) -> tuple[Tensor, int]:
    """Zero-pad the end to a multiple of ``chunk_size``, then apply ``pad`` (left, right)."""
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)
    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)
    return x, mod


class LayerNormPermuted(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        """``x``: [B, C, T]"""
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = super().forward(x)
        return x.permute(0, 2, 1)  # [B, C, T]


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 padding: int, dilation: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride,
                      padding, groups=in_channels, dilation=dilation),
            LayerNormPermuted(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            LayerNormPermuted(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DilatedCausalConvEncoder(nn.Module):
    """Dilated causal convolutions that add past context to the latent input."""

    def __init__(self, channels: int, num_layers: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # buf_length[i] = (kernel_size - 1) * dilation[i]
        self.buf_lengths = [(kernel_size - 1) * 2**i for i in range(num_layers)]

        # start index of each layer's context inside the one context buffer
        self.buf_indices = [0]
        for i in range(num_layers - 1):
            self.buf_indices.append(self.buf_indices[-1] + self.buf_lengths[i])

        _dcc_layers = OrderedDict()
        for i in range(num_layers):
            dcc_layer = DepthwiseSeparableConv(
                channels, channels, kernel_size=3, stride=1, padding=0, dilation=2**i)
            _dcc_layers.update({f"dcc_{i}": dcc_layer})
        self.dcc_layers = nn.Sequential(_dcc_layers)

    def init_ctx_buf(self, batch_size: int, device) -> Tensor:
        return torch.zeros(
            (batch_size, self.channels, (self.kernel_size - 1) * (2**self.num_layers - 1)),
            device=device)

    def forward(self, x: Tensor, ctx_buf: Tensor) -> tuple[Tensor, Tensor]:
        """``x``: [B, channels, T]; ``ctx_buf``: [B, channels, sum(buf_lengths)] (updated)."""
        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            # DCC input: this layer's context followed by the current input
            dcc_in = torch.cat((ctx_buf[..., buf_start_idx:buf_end_idx], x), dim=-1)

            # the newest inputs become the context of the next call
            ctx_buf[..., buf_start_idx:buf_end_idx] = dcc_in[..., -self.buf_lengths[i]:]

            # residual connection
            x = x + self.dcc_layers[i](dcc_in)

        return x, ctx_buf


class CausalTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    """Adapted (by the Waveformer authors) from
    https://github.com/alexmt-scale/causal-transformer-decoder/blob/
    0caf6ad71c46488f76d89845b0123d2550ef792f/causal_transformer_decoder/model.py#L77
    """

    def forward(self, tgt: Tensor, memory: Tensor | None = None, chunk_size: int = 1
                ) -> tuple[Tensor, Tensor, Tensor | None]:
        tgt_last_tok = tgt[:, -chunk_size:, :]

        # self attention
        tmp_tgt, sa_map = self.self_attn(
            tgt_last_tok, tgt, tgt,
            attn_mask=None,  # not needed because we only care about the last tokens
            key_padding_mask=None,
        )
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        ca_map = None
        if memory is not None:
            tmp_tgt, ca_map = self.multihead_attn(
                tgt_last_tok, memory, memory,
                attn_mask=None,  # attend to the entire chunk
                key_padding_mask=None,
            )
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # feed-forward network
        tmp_tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt_last_tok))))
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok, sa_map, ca_map


class CausalTransformerDecoder(nn.Module):
    """Decodes each chunk using exactly ``ctx_len`` past vectors and no future chunk."""

    def __init__(self, model_dim: int, ctx_len: int, chunk_size: int, num_layers: int,
                 nhead: int, use_pos_enc: bool, ff_dim: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.unfold = nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)
        self.pos_enc = PositionalEncoding(model_dim, max_len=200)
        self.tf_dec_layers = nn.ModuleList([CausalTransformerDecoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim,
            batch_first=True) for _ in range(num_layers)])

    def init_ctx_buf(self, batch_size: int, device) -> Tensor:
        return torch.zeros(
            (batch_size, self.num_layers + 1, self.ctx_len, self.model_dim), device=device)

    def _causal_unfold(self, x: Tensor) -> Tensor:
        """[B, ctx_len + T, C] -> [B * T / chunk, ctx_len + chunk, C]: every chunk with the
        ``ctx_len`` vectors before it."""
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, ctx_len + T]
        x = self.unfold(x.unsqueeze(-1))  # [B, C * (ctx_len + chunk_size), -1]
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, C, self.ctx_len + self.chunk_size)
        x = x.reshape(-1, C, self.ctx_len + self.chunk_size)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, tgt: Tensor, mem: Tensor, ctx_buf: Tensor) -> tuple[Tensor, Tensor]:
        """``tgt``, ``mem``: [B, model_dim, T]; ``ctx_buf``: [B, num_layers + 1, ctx_len,
        model_dim] (updated)."""
        mem, _ = mod_pad(mem, self.chunk_size, (0, 0))
        tgt, mod = mod_pad(tgt, self.chunk_size, (0, 0))

        B, C, T = tgt.shape

        tgt = tgt.permute(0, 2, 1)
        mem = mem.permute(0, 2, 1)

        # prepend the memory with its context
        mem = torch.cat((ctx_buf[:, 0, :, :], mem), dim=1)
        ctx_buf[:, 0, :, :] = mem[:, -self.ctx_len:, :]
        mem_ctx = self._causal_unfold(mem)
        if self.use_pos_enc:
            mem_ctx = mem_ctx + self.pos_enc(mem_ctx)

        # attention is run on at most K chunks at a time, so long sequences do not run out
        # of memory
        K = 1000

        for i, tf_dec_layer in enumerate(self.tf_dec_layers):
            # prepend the target with its context
            tgt = torch.cat((ctx_buf[:, i + 1, :, :], tgt), dim=1)
            ctx_buf[:, i + 1, :, :] = tgt[:, -self.ctx_len:, :]

            tgt_ctx = self._causal_unfold(tgt)
            if self.use_pos_enc and i == 0:
                tgt_ctx = tgt_ctx + self.pos_enc(tgt_ctx)
            tgt = torch.zeros_like(tgt_ctx)[:, -self.chunk_size:, :]
            for k in range(int(math.ceil(tgt.shape[0] / K))):
                tgt[k * K:(k + 1) * K], _sa_map, _ca_map = tf_dec_layer(
                    tgt_ctx[k * K:(k + 1) * K], mem_ctx[k * K:(k + 1) * K], self.chunk_size)
            tgt = tgt.reshape(B, T, C)

        tgt = tgt.permute(0, 2, 1)
        if mod != 0:
            tgt = tgt[..., :-mod]

        return tgt, ctx_buf


class MaskNet(nn.Module):
    def __init__(self, enc_dim: int, num_enc_layers: int, dec_dim: int, dec_buf_len: int,
                 dec_chunk_size: int, num_dec_layers: int, use_pos_enc: bool,
                 skip_connection: bool, proj: bool) -> None:
        super().__init__()
        self.skip_connection = skip_connection
        self.proj = proj

        # encoder based on dilated causal convolutions
        self.encoder = DilatedCausalConvEncoder(channels=enc_dim, num_layers=num_enc_layers)

        # projections between encoder and decoder dimensions
        self.proj_e2d_e = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0, groups=dec_dim),
            nn.ReLU())
        self.proj_e2d_l = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0, groups=dec_dim),
            nn.ReLU())
        self.proj_d2e = nn.Sequential(
            nn.Conv1d(dec_dim, enc_dim, kernel_size=1, stride=1, padding=0, groups=dec_dim),
            nn.ReLU())

        # transformer decoder that works on chunks
        self.decoder = CausalTransformerDecoder(
            model_dim=dec_dim, ctx_len=dec_buf_len, chunk_size=dec_chunk_size,
            num_layers=num_dec_layers, nhead=8, use_pos_enc=use_pos_enc, ff_dim=2 * dec_dim)

    def forward(self, x: Tensor, l: Tensor, enc_buf: Tensor, dec_buf: Tensor  # noqa: E741
                ) -> tuple[Tensor, Tensor, Tensor]:
        """``x``: [B, C, T] latent input; ``l``: [B, C] label embedding. Returns the mask."""
        # encode the input
        e, enc_buf = self.encoder(x, enc_buf)

        # label integration
        l = l.unsqueeze(2) * e  # noqa: E741

        if self.proj:
            e = self.proj_e2d_e(e)
            m = self.proj_e2d_l(l)
            # cross-attention to predict the mask
            m, dec_buf = self.decoder(m, e, dec_buf)
        else:
            m, dec_buf = self.decoder(l, e, dec_buf)

        # back to encoder dimensions
        if self.proj:
            m = self.proj_d2e(m)

        # final mask after the residual connection
        if self.skip_connection:
            m = l + m

        return m, enc_buf, dec_buf


class Net(nn.Module):
    def __init__(self, label_len: int, L: int = 8,
                 enc_dim: int = 512, num_enc_layers: int = 10,
                 dec_dim: int = 256, dec_buf_len: int = 100, num_dec_layers: int = 2,
                 dec_chunk_size: int = 72, out_buf_len: int = 2,
                 use_pos_enc: bool = True, skip_connection: bool = True, proj: bool = True,
                 lookahead: bool = True) -> None:
        super().__init__()
        self.L = L
        self.out_buf_len = out_buf_len
        self.enc_dim = enc_dim
        self.lookahead = lookahead
        self.label_len = label_len
        self.dec_chunk_size = dec_chunk_size

        # input convolution: audio -> latent representation
        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=enc_dim, kernel_size=kernel_size, stride=L,
                      padding=0, bias=False),
            nn.ReLU())

        # label embedding
        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.ReLU())

        # mask generator
        self.mask_gen = MaskNet(
            enc_dim=enc_dim, num_enc_layers=num_enc_layers,
            dec_dim=dec_dim, dec_buf_len=dec_buf_len,
            dec_chunk_size=dec_chunk_size, num_dec_layers=num_dec_layers,
            use_pos_enc=use_pos_enc, skip_connection=skip_connection, proj=proj)

        # output convolution: masked latents -> audio
        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=enc_dim, out_channels=1,
                kernel_size=(out_buf_len + 1) * L,
                stride=L,
                padding=out_buf_len * L, bias=False),
            nn.Tanh())

    def init_buffers(self, batch_size: int, device) -> tuple[Tensor, Tensor, Tensor]:
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size, device)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size, device)
        out_buf = torch.zeros(batch_size, self.enc_dim, self.out_buf_len, device=device)
        return enc_buf, dec_buf, out_buf

    def forward(self, x: Tensor, label: Tensor, init_enc_buf: Tensor | None = None,
                init_dec_buf: Tensor | None = None, init_out_buf: Tensor | None = None,
                pad: bool = True):
        """Extract the sounds named by ``label`` from the mixture ``x``.

        ``x``: [B, 1, T] audio; ``label``: [B, label_len] multi-hot. Returns [B, 1, T] (and the
        three updated buffers when buffers were passed in).
        """
        mod = 0
        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)

        if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:
            assert init_enc_buf is None and init_dec_buf is None and init_out_buf is None, \
                "Either all three buffers are given, or none of them."
            enc_buf, dec_buf, out_buf = self.init_buffers(x.shape[0], x.device)
        else:
            enc_buf, dec_buf, out_buf = init_enc_buf, init_dec_buf, init_out_buf

        # latent representation of the input
        x = self.in_conv(x)

        # label embedding: [B, label_len] -> [B, channels]
        l = self.label_embedding(label)  # noqa: E741

        # mask for the labelled sounds
        m, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)

        # apply the mask and decode
        x = x * m
        x = torch.cat((out_buf, x), dim=-1)
        out_buf = x[..., -self.out_buf_len:]
        x = self.out_conv(x)

        # remove the mod padding, if any
        if mod != 0:
            x = x[:, :, :-mod]

        if init_enc_buf is None:
            return x
        return x, enc_buf, dec_buf, out_buf

    # --- streaming (added for Sound Spy; wraps the original buffer interface) ------------------

    @property
    def stream_sizes(self) -> tuple[int, int, int]:
        """``(hop, context, lookahead)`` in samples of one streaming step.

        A step produces ``hop = dec_chunk_size * L`` output samples for the input samples
        ``[t, t + hop)``. Its input must hold ``context`` samples before ``t`` (zeros at the very
        start) and ``lookahead`` samples after ``t + hop``: ``context + hop + lookahead`` in all.
        """
        look = self.L if self.lookahead else 0
        return self.dec_chunk_size * self.L, look, look

    def stream_step(self, x: Tensor, label: Tensor, buffers: tuple[Tensor, Tensor, Tensor]
                    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """One or more whole streaming steps.

        ``x``: [B, 1, context + n * hop + lookahead]; returns [B, 1, n * hop] and the updated
        buffers. Feeding a signal step by step gives the same output as one ``forward`` call
        over the whole signal (tested).
        """
        out, enc_buf, dec_buf, out_buf = self.forward(x, label, *buffers, pad=False)
        return out, (enc_buf, dec_buf, out_buf)

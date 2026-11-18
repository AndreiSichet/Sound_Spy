# Copyright (c) 2021 Hendrik Schröter
#
# Licensed under either of the Apache License, Version 2.0 (LICENSE-APACHE in this folder) or
# the MIT license (LICENSE-MIT in this folder), at your option.
#
# Adapted from df/deepfilternet3.py, df/modules.py and df/multiframe.py of DeepFilterNet 0.5.6
# (https://github.com/Rikorose/DeepFilterNet). Changes for Sound Spy: only the DeepFilterNet3
# parts; the settings come from a ``DfParams`` object instead of the package's global config;
# no logging; no training-only branches (lsnr dropout, mask-only / df-only modes); and a
# streaming step (``DfNet.stream``) that carries the recurrent and convolution state explicitly
# and gives the same frames as the whole-sequence ``forward``.
"""DeepFilterNet3 (Schröter et al., ICASSP 2023): real-time full-band speech enhancement.

48 kHz in, 48 kHz out, frames of 10 ms. Two stages on an STFT with 481 bins:

1. **ERB stage**: an encoder over 32 ERB-band levels and the 96 lowest complex bins, three
   GRUs, and a decoder that predicts one gain per ERB band (a coarse mask over the whole band).
2. **Deep filter**: for the 96 bins below 4.8 kHz, a complex 5-tap filter over time
   (2 frames back, 2 ahead) per bin, which can restore the harmonics of the voice between noise.

The encoder sees its features ``conv_lookahead`` (2) frames ahead and the deep filter reaches
``df_lookahead`` (2) frames ahead, so an output frame needs input up to 2 frames later.
"""

from __future__ import annotations

import math
from functools import partial

import torch
from torch import Tensor, nn

from soundspy.extraction.models.deepfilternet.dsp import DfParams

PI = 3.1415926535897932384626433


class Conv2dNormAct(nn.Sequential):
    """Causal Conv2d (time padding in front), optional 1x1 after a grouped conv, norm, act.
    Expected input format: [B, C, T, F]. The layer order is the checkpoint's."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size, fstride: int = 1,
                 dilation: int = 1, fpad: bool = True, bias: bool = True,
                 separable: bool = False, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU) -> None:
        kernel_size = ((kernel_size, kernel_size) if isinstance(kernel_size, int)
                       else tuple(kernel_size))
        fpad_ = kernel_size[1] // 2 + dilation - 1 if fpad else 0
        pad = (0, 0, kernel_size[0] - 1, 0)
        layers: list[nn.Module] = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel_size) == 1:
            separable = False
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(0, fpad_),
                                stride=(1, fstride), dilation=(1, dilation), groups=groups,
                                bias=bias))
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
        self.time_context = kernel_size[0] - 1  # past frames each output frame needs

    def unpadded(self, x: Tensor) -> Tensor:
        """The layers without the time padding: the caller supplies ``time_context`` frames of
        history in front of ``x`` (streaming)."""
        for layer in self:
            if not isinstance(layer, nn.ConstantPad2d):
                x = layer(x)
        return x


class ConvTranspose2dNormAct(nn.Sequential):
    """Causal ConvTranspose2d (upsampling over frequency). Expected input format: [B, C, T, F]."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size, fstride: int = 1,
                 dilation: int = 1, fpad: bool = True, bias: bool = True,
                 separable: bool = False, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU) -> None:
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        fpad_ = kernel_size[1] // 2 if fpad else 0
        pad = (0, 0, kernel_size[0] - 1, 0)
        layers: list[nn.Module] = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size,
                                         padding=(kernel_size[0] - 1, fpad_ + dilation - 1),
                                         output_padding=(0, fpad_), stride=(1, fstride),
                                         dilation=(1, dilation), groups=groups, bias=bias))
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
        if kernel_size[0] != 1:
            raise ValueError("streaming supports transposed convolutions of time kernel 1 only")


class GroupedLinearEinsum(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups: int = 1) -> None:
        super().__init__()
        self.input_size, self.hidden_size, self.groups = input_size, hidden_size, groups
        if input_size % groups or hidden_size % groups:
            raise ValueError(f"sizes {input_size}, {hidden_size} not divisible by {groups}")
        self.ws = input_size // groups
        self.weight = nn.Parameter(torch.zeros(groups, input_size // groups,
                                               hidden_size // groups))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        x = x.view(b, t, self.groups, self.ws)
        x = torch.einsum("btgi,gih->btgh", x, self.weight)
        return x.flatten(2, 3)


class SqueezedGRU_S(nn.Module):  # noqa: N801 - the checkpoint's class
    def __init__(self, input_size: int, hidden_size: int, output_size: int | None = None,
                 num_layers: int = 1, linear_groups: int = 8, batch_first: bool = True,
                 gru_skip_op=None, linear_act_layer=nn.Identity) -> None:
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.linear_in = nn.Sequential(GroupedLinearEinsum(input_size, hidden_size, linear_groups),
                                       linear_act_layer())
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                          batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups), linear_act_layer())
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Tensor | None = None) -> tuple[Tensor, Tensor]:  # noqa: A002
        x = self.linear_in(input)
        x, h = self.gru(x, h)
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, h


class Mask(nn.Module):
    def __init__(self, erb_inv_fb: Tensor) -> None:
        super().__init__()
        self.register_buffer("erb_inv_fb", erb_inv_fb)

    def forward(self, spec: Tensor, mask: Tensor) -> Tensor:
        # spec [B, 1, T, F, 2], mask [B, 1, T, E]
        mask = mask.matmul(self.erb_inv_fb)  # [B, 1, T, F]
        return spec * mask.unsqueeze(4)


class ChannelScale(nn.Module):
    """A depthwise 1x1 convolution as a per-channel product (identical numbers).

    PyTorch runs a grouped convolution with a 1x1 kernel on one CPU thread as one small
    convolution per group: 64 calls each for the decoder's four pathway convolutions, ~11 ms
    of a 23 ms streaming step on the laptop (measured, step 07c).
    """

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().reshape(1, -1, 1, 1).clone(),
                                   requires_grad=False)
        self.bias = (None if conv.bias is None else
                     nn.Parameter(conv.bias.detach().reshape(1, -1, 1, 1).clone(),
                                  requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.weight
        return x if self.bias is None else x + self.bias


def fuse_for_inference(net: nn.Module) -> nn.Module:
    """Replace every depthwise 1x1 convolution by :class:`ChannelScale` (after the weights are
    loaded; the state-dict names of the checkpoint no longer apply afterwards)."""
    for module in list(net.modules()):
        if not isinstance(module, nn.Sequential):
            continue
        for name, child in list(module.named_children()):
            if (type(child) is nn.Conv2d and child.kernel_size == (1, 1)
                    and child.groups == child.in_channels == child.out_channels
                    and child.stride == (1, 1) and child.padding == (0, 0)):
                setattr(module, name, ChannelScale(child))
    return net


def _skip(kind: str, input_size: int, hidden_size: int, groups: int):
    if kind == "none":
        return None
    if kind == "identity":
        return partial(nn.Identity)
    if kind == "groupedlinear":
        return partial(GroupedLinearEinsum, input_size=input_size, hidden_size=hidden_size,
                       groups=groups)
    raise ValueError(f"unknown GRU skip {kind!r}")


class Encoder(nn.Module):
    def __init__(self, p: DfParams) -> None:
        super().__init__()
        if p.nb_erb % 4:
            raise ValueError("nb_erb must be divisible by 4")
        self.erb_conv0 = Conv2dNormAct(1, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False,
                                       separable=True)
        conv_layer = partial(Conv2dNormAct, in_ch=p.conv_ch, out_ch=p.conv_ch,
                             kernel_size=p.conv_kernel, bias=False, separable=True)
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)
        self.df_conv0 = Conv2dNormAct(2, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False,
                                      separable=True)
        self.df_conv1 = conv_layer(fstride=2)
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        df_fc_emb = GroupedLinearEinsum(p.conv_ch * p.nb_df // 2, self.emb_in_dim,
                                        groups=p.enc_linear_groups)
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))
        if p.enc_concat:
            raise ValueError("enc_concat is not supported (DeepFilterNet3 does not use it)")
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim, p.emb_hidden_dim, output_size=self.emb_out_dim, num_layers=1,
            batch_first=True,
            gru_skip_op=_skip(p.emb_gru_skip_enc, self.emb_out_dim, self.emb_out_dim,
                              p.linear_groups),
            linear_groups=p.linear_groups, linear_act_layer=partial(nn.ReLU, inplace=True))
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def _rest(self, e0: Tensor, c0: Tensor, h: Tensor | None):
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)
        c1 = self.df_conv1(c0)
        cemb = self.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        emb = e3.permute(0, 2, 3, 1).flatten(2) + cemb
        emb, h = self.emb_gru(emb, h)
        return e1, e2, e3, emb, h

    def forward(self, feat_erb: Tensor, feat_spec: Tensor, h: Tensor | None = None):
        """Whole sequence: ``feat_erb`` [B, 1, T, E], ``feat_spec`` [B, 2, T, Fc]."""
        e0 = self.erb_conv0(feat_erb)
        c0 = self.df_conv0(feat_spec)
        e1, e2, e3, emb, h = self._rest(e0, c0, h)
        return e0, e1, e2, e3, emb, c0, h

    def stream(self, erb_window: Tensor, spec_window: Tensor, h: Tensor | None):
        """Streaming: the windows hold ``time_context`` frames of history in front of the new
        frames (``conv_kernel_inp[0] - 1`` = 2)."""
        e0 = self.erb_conv0.unpadded(erb_window)
        c0 = self.df_conv0.unpadded(spec_window)
        e1, e2, e3, emb, h = self._rest(e0, c0, h)
        return e0, e1, e2, e3, emb, c0, h


class ErbDecoder(nn.Module):
    def __init__(self, p: DfParams) -> None:
        super().__init__()
        if p.nb_erb % 8:
            raise ValueError("nb_erb must be divisible by 8")
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim, p.emb_hidden_dim, output_size=self.emb_out_dim,
            num_layers=p.emb_num_layers - 1, batch_first=True,
            gru_skip_op=_skip(p.emb_gru_skip, self.emb_in_dim, self.emb_out_dim,
                              p.linear_groups),
            linear_groups=p.linear_groups, linear_act_layer=partial(nn.ReLU, inplace=True))
        tconv_layer = partial(ConvTranspose2dNormAct, kernel_size=p.convt_kernel, bias=False,
                              separable=True)
        conv_layer = partial(Conv2dNormAct, bias=False, separable=True)
        self.conv3p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt3 = conv_layer(p.conv_ch, p.conv_ch, kernel_size=p.conv_kernel)
        self.conv2p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv1p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv0p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(p.conv_ch, 1, kernel_size=p.conv_kernel,
                                    activation_layer=nn.Sigmoid)
        if p.conv_kernel[0] != 1:
            raise ValueError("streaming supports decoder kernels of time size 1 only")

    def forward(self, emb: Tensor, e3: Tensor, e2: Tensor, e1: Tensor, e0: Tensor,
                h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        b, _, t, f8 = e3.shape
        emb, h = self.emb_gru(emb, h)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)
        e3 = self.convt3(self.conv3p(e3) + emb)
        e2 = self.convt2(self.conv2p(e2) + e3)
        e1 = self.convt1(self.conv1p(e1) + e2)
        m = self.conv0_out(self.conv0p(e0) + e1)
        return m, h


class DfOutputReshapeMF(nn.Module):
    def __init__(self, df_order: int, df_bins: int) -> None:
        super().__init__()
        self.df_order, self.df_bins = df_order, df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        return coefs.view(new_shape).permute(0, 3, 1, 2, 4)


class DfDecoder(nn.Module):
    def __init__(self, p: DfParams) -> None:
        super().__init__()
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.df_hidden_dim
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_out_ch = p.df_order * 2
        conv_layer = partial(Conv2dNormAct, separable=True, bias=False)
        kt = p.df_pathway_kernel_size_t
        self.df_convp = conv_layer(p.conv_ch, self.df_out_ch, fstride=1, kernel_size=(kt, 1))
        self.df_gru = SqueezedGRU_S(self.emb_in_dim, self.emb_dim, num_layers=p.df_num_layers,
                                    batch_first=True, gru_skip_op=None,
                                    linear_act_layer=partial(nn.ReLU, inplace=True))
        skip = p.df_gru_skip.lower()
        if skip == "none":
            self.df_skip = None
        elif skip == "identity":
            self.df_skip = nn.Identity()
        elif skip == "groupedlinear":
            self.df_skip = GroupedLinearEinsum(self.emb_in_dim, self.emb_dim,
                                               groups=p.linear_groups)
        else:
            raise ValueError(f"unknown df_gru_skip {skip!r}")
        df_out = GroupedLinearEinsum(p.df_hidden_dim, self.df_bins * self.df_out_ch,
                                     groups=p.linear_groups)
        self.df_out = nn.Sequential(df_out, nn.Tanh())
        self.df_fc_a = nn.Sequential(nn.Linear(p.df_hidden_dim, 1), nn.Sigmoid())

    def _coefs(self, emb: Tensor, c0p: Tensor, h: Tensor | None) -> tuple[Tensor, Tensor]:
        b, t, _ = emb.shape
        c, h = self.df_gru(emb, h)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        c = self.df_out(c)
        return c.view(b, t, self.df_bins, self.df_out_ch) + c0p.permute(0, 2, 3, 1), h

    def forward(self, emb: Tensor, c0: Tensor, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self._coefs(emb, self.df_convp(c0), h)

    def stream(self, emb: Tensor, c0_window: Tensor, h: Tensor | None):
        """``c0_window``: ``kt - 1`` (4) frames of history in front of the new frames."""
        return self._coefs(emb, self.df_convp.unpadded(c0_window), h)


def deep_filter(spec_window: Tensor, coefs: Tensor, nb_df: int, order: int) -> Tensor:
    """``spec_window`` [B, 1, T + order - 1, F, 2] (complex as real pairs), ``coefs``
    [B, order, T, nb_df, 2] -> filtered low bins [B, 1, T, nb_df, 2]: output frame ``t`` is
    ``sum_n spec_window[t + n] * coefs[n, t]`` (complex products)."""
    spec = torch.view_as_complex(spec_window.contiguous())[..., :nb_df]  # [B, 1, T+o-1, Fd]
    windows = spec.unfold(2, order, 1)  # [B, 1, T, Fd, o]
    c = torch.view_as_complex(coefs.contiguous())  # [B, o, T, Fd]
    c = c.view(c.shape[0], -1, order, *c.shape[2:])  # [B, 1, o, T, Fd]
    return torch.view_as_real(torch.einsum("...tfn,...ntf->...tf", windows, c))


def deep_filter_real(spec_window: Tensor, coefs: Tensor, nb_df: int, order: int) -> Tensor:
    """:func:`deep_filter` with real arithmetic only (the same numbers): used by the streaming
    step, which must also export to ONNX (no complex tensors there)."""
    t = coefs.shape[2]
    s = spec_window[:, 0, :, :nb_df, :]  # [B, T + o - 1, Fd, 2]
    re = im = None
    for n in range(order):
        sn, cn = s[:, n:n + t], coefs[:, n]
        r = sn[..., 0] * cn[..., 0] - sn[..., 1] * cn[..., 1]
        i = sn[..., 0] * cn[..., 1] + sn[..., 1] * cn[..., 0]
        re, im = (r, i) if re is None else (re + r, im + i)
    return torch.stack((re, im), dim=-1).unsqueeze(1)


class DfNet(nn.Module):
    def __init__(self, p: DfParams, erb_fb: Tensor, erb_inv_fb: Tensor,
                 post_filter: bool = False) -> None:
        super().__init__()
        if p.conv_lookahead < p.df_lookahead:
            raise ValueError("conv_lookahead must be >= df_lookahead")
        self.p = p
        self.df_lookahead = p.df_lookahead
        self.conv_lookahead = p.conv_lookahead
        self.nb_df = p.nb_df
        self.freq_bins = p.fft_size // 2 + 1
        self.erb_bins = p.nb_erb
        self.df_order = p.df_order
        self.register_buffer("erb_fb", erb_fb)
        self.enc = Encoder(p)
        self.erb_dec = ErbDecoder(p)
        self.mask = Mask(erb_inv_fb)
        self.df_dec = DfDecoder(p)
        self.df_out_transform = DfOutputReshapeMF(p.df_order, p.nb_df)
        self.post_filter = post_filter
        self.post_filter_beta = p.pf_beta
        if p.df_order - 1 - p.df_lookahead < 0:
            raise ValueError("df_lookahead must be < df_order")

    # sizes of the streaming state (frames of history)
    @property
    def feat_context(self) -> int:
        return self.p.conv_kernel_inp[0] - 1

    @property
    def c0_context(self) -> int:
        return self.p.df_pathway_kernel_size_t - 1

    @property
    def spec_context(self) -> int:
        """Spectra kept in front of the newest frame: the deep filter reaches ``df_order - 1 -
        df_lookahead`` frames back from its output frame, which lies ``conv_lookahead`` frames
        behind the newest one."""
        return self.p.df_order - 1 - self.df_lookahead + self.conv_lookahead

    def _post_filter(self, spec_e: Tensor, spec: Tensor) -> Tensor:
        beta, eps = self.post_filter_beta, 1e-12
        magnitude = lambda s: s.pow(2).sum(-1).sqrt()  # noqa: E731 - |complex| of real pairs
        mask = (magnitude(spec_e) / magnitude(spec).add(eps)).clamp(eps, 1)
        mask_sin = mask * torch.sin(PI * mask / 2).clamp_min(eps)
        pf = (1 + beta) / (1 + beta * mask.div(mask_sin).pow(2))
        return spec_e * pf.unsqueeze(-1)

    def forward(self, spec: Tensor, feat_erb: Tensor, feat_spec: Tensor) -> Tensor:
        """Whole sequence, exactly as DeepFilterNet's own ``DfNet.forward``.

        ``spec`` [B, 1, T, F, 2], ``feat_erb`` [B, 1, T, E], ``feat_spec`` [B, 1, T, Fd, 2]
        -> enhanced ``spec`` [B, 1, T, F, 2]. The last ``conv_lookahead`` frames are computed
        from zero-padded features (no future is known at the end of a file).
        """
        la, dla = self.conv_lookahead, self.df_lookahead
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        pad_feat = nn.ConstantPad2d((0, 0, -la, la), 0.0)
        e0, e1, e2, e3, emb, c0, _ = self.enc(pad_feat(feat_erb), pad_feat(feat_spec))
        m, _ = self.erb_dec(emb, e3, e2, e1, e0)
        spec_m = self.mask(spec, m)
        coefs = self.df_out_transform(self.df_dec(emb, c0)[0])
        padded = nn.functional.pad(spec, (0, 0, 0, 0, self.df_order - 1 - dla, dla))
        low = deep_filter(padded, coefs, self.nb_df, self.df_order)
        spec_e = torch.cat((low, spec_m[..., self.nb_df:, :]), dim=3)
        if self.post_filter:
            spec_e = self._post_filter(spec_e, spec)
        return spec_e

    def init_state(self, device=None) -> tuple[Tensor, ...]:
        """Zero streaming state: ``(erb window, spec-feature window, spectrum window,
        c0 window, h_enc, h_erb, h_df)``; windows are the frames of history."""
        p = self.p
        device = device if device is not None else self.erb_fb.device
        z = partial(torch.zeros, device=device)
        hid = p.emb_hidden_dim
        return (z(1, 1, self.feat_context, p.nb_erb),
                z(1, 2, self.feat_context, p.nb_df),
                z(1, 1, self.spec_context, self.freq_bins, 2),
                z(1, p.conv_ch, self.c0_context, p.nb_df),
                z(1, 1, hid),
                z(p.emb_num_layers - 1, 1, hid),
                z(p.df_num_layers, 1, p.df_hidden_dim))

    def stream(self, spec: Tensor, feat_erb: Tensor, feat_spec: Tensor,
               state: tuple[Tensor, ...]) -> tuple[Tensor, tuple[Tensor, ...]]:
        """``k >= 1`` new frames (same layout as ``forward``) -> the ``k`` enhanced frames that
        lie ``conv_lookahead`` frames behind them, and the new state.

        The caller must treat the first ``conv_lookahead`` frames of a stream specially (see
        ``SpeechEnhancer``): ``forward`` drops their features (its time shift), so they only
        enter the spectrum window (:meth:`push_spectra`).
        """
        erb_w, spec_w, specs, c0_w, h_enc, h_erb, h_df = state
        k = spec.shape[2]
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        erb_in = torch.cat((erb_w, feat_erb), dim=2)
        spec_in = torch.cat((spec_w, feat_spec), dim=2)
        e0, e1, e2, e3, emb, c0, h_enc = self.enc.stream(erb_in, spec_in, h_enc)
        m, h_erb = self.erb_dec(emb, e3, e2, e1, e0, h_erb)
        c0_in = torch.cat((c0_w, c0), dim=2)
        coefs, h_df = self.df_dec.stream(emb, c0_in, h_df)
        coefs = self.df_out_transform(coefs)
        window = torch.cat((specs, spec), dim=2)  # [B, 1, spec_context + k, F, 2]
        back = self.df_order - 1 - self.df_lookahead  # frames before the output frame
        centre = window[:, :, back:back + k]  # the spectra of the output frames
        spec_m = self.mask(centre, m)
        filt = window[:, :, :back + k + self.df_lookahead]
        low = deep_filter_real(filt, coefs, self.nb_df, self.df_order)
        spec_e = torch.cat((low, spec_m[..., self.nb_df:, :]), dim=3)
        if self.post_filter:
            spec_e = self._post_filter(spec_e, centre)
        fc, cc = self.feat_context, self.c0_context
        new_state = (erb_in[:, :, erb_in.shape[2] - fc:], spec_in[:, :, spec_in.shape[2] - fc:],
                     window[:, :, window.shape[2] - self.spec_context:],
                     c0_in[:, :, c0_in.shape[2] - cc:], h_enc, h_erb, h_df)
        return spec_e, new_state

    def push_spectra(self, spec: Tensor, state: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Only the spectrum window moves (the first ``conv_lookahead`` frames of a stream,
        or frames while the network rests)."""
        specs = torch.cat((state[2], spec), dim=2)
        return (*state[:2], specs[:, :, specs.shape[2] - self.spec_context:], *state[3:])

    def push_features(self, spec: Tensor, feat_erb: Tensor, feat_spec: Tensor,
                      state: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Spectrum and feature windows move without running the network (it rests); its
        recurrent state and c0 window are cleared for a fresh start."""
        fc = self.feat_context
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        erb_in = torch.cat((state[0], feat_erb), dim=2)
        spec_in = torch.cat((state[1], feat_spec), dim=2)
        fresh = self.init_state(spec.device)
        return (erb_in[:, :, erb_in.shape[2] - fc:], spec_in[:, :, spec_in.shape[2] - fc:],
                self.push_spectra(spec, state)[2], *fresh[3:])

# Adapted from EfficientAT, models/mn/model.py — https://github.com/fschmid56/EfficientAT
# Copyright (c) 2022 Florian Schmid. MIT License (full text: LICENSE in this folder).
# (EfficientAT's file is itself adapted from torchvision's mobilenetv3.py, BSD-3-Clause.)
# Changes for Sound Spy: only the "mlp" head is kept (the one of the mn10_as weights); no
# torch.hub download and no printing (weights are downloaded and verified by
# soundspy.detection.weights); get_model() takes the same arguments with the same defaults.
"""EfficientAT MobileNetV3 (``mn``) audio tagger: log-mel input -> 527 AudioSet logits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from soundspy.detection.backbones.efficientat.block_types import (
    InvertedResidual,
    InvertedResidualConfig,
)
from soundspy.detection.backbones.efficientat.utils import ConvNormActivation, cnn_out_size

WEIGHTS_URL = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
PRETRAINED_FILES = {"mn10_as": "mn10_as_mAP_471.pt"}


class MN(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: list[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Callable[..., nn.Module] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
        dropout: float = 0.2,
        in_conv_kernel: int = 3,
        in_conv_stride: int = 2,
        in_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting)
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        depthwise_norm_layer = norm_layer = \
            norm_layer if norm_layer is not None else partial(nn.BatchNorm2d, eps=0.001,
                                                              momentum=0.01)

        layers: list[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # get squeeze excitation config
        se_cnf = kwargs.get('se_conf', None)

        # building inverted residual blocks (track frequency/time sizes for the SE layers)
        f_dim, t_dim = kwargs.get('input_dims', (128, 1000))
        # take into account first conv layer
        f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
        t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)
        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim)
            t_dim = cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim  # update dimensions in block config
            layers.append(block(cnf, se_cnf, norm_layer, depthwise_norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )
        self.embedding_dim = last_channel

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm | nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.features:
            x = layer(x)
        features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        x = self.classifier(x).squeeze()
        if features.dim() == 1 and x.dim() == 1:
            # squeezed batch dimension
            features = features.unsqueeze(0)
            x = x.unsqueeze(0)
        return x, features

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    strides: tuple[int, int, int, int] = (2, 2, 2, 2),
    **kwargs: Any,
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    # InvertedResidualConfig:
    # input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation
    rd = reduce_divider
    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, "HS", strides[2], 1),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        bneck_conf(112, 5, 672, 160 // rd, True, "HS", strides[3], dilation),  # C4
        bneck_conf(160 // rd, 5, 960 // rd, 160 // rd, True, "HS", 1, dilation),
        bneck_conf(160 // rd, 5, 960 // rd, 160 // rd, True, "HS", 1, dilation),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel


def get_model(num_classes: int = 527, width_mult: float = 1.0, reduced_tail: bool = False,
              dilated: bool = False, strides: tuple[int, int, int, int] = (2, 2, 2, 2),
              input_dim_f: int = 128, input_dim_t: int = 1000, se_dims: str = 'c',
              se_agg: str = "max", se_r: int = 4) -> MN:
    """MobileNetV3 for audio, as EfficientAT's ``get_model`` with ``head_type="mlp"``.

    ``mn10_as`` = ``width_mult=1.0`` and the defaults. Weights are loaded separately.
    """
    dim_map = {'c': 1, 'f': 2, 't': 3}
    assert len(se_dims) <= 3 and all(s in dim_map for s in se_dims) or se_dims == 'none'
    input_dims = (input_dim_f, input_dim_t)
    se_dim_list = None if se_dims == 'none' else [dim_map[s] for s in se_dims]
    se_conf = dict(se_dims=se_dim_list, se_agg=se_agg, se_r=se_r)
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        width_mult=width_mult, reduced_tail=reduced_tail, dilated=dilated, strides=strides)
    return MN(inverted_residual_setting, last_channel, num_classes=num_classes,
              input_dims=input_dims, se_conf=se_conf)

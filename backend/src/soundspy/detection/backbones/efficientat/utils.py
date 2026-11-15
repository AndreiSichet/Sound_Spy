# Adapted from EfficientAT, models/mn/utils.py — https://github.com/fschmid56/EfficientAT
# Copyright (c) 2022 Florian Schmid. MIT License (full text: LICENSE in this folder).
# Changes for Sound Spy: kept only make_divisible and cnn_out_size; added
# ConvNormActivation (replaces torchvision.ops.misc.ConvNormActivation so torchvision is not
# needed; same layer order, so state-dict keys are unchanged).
"""Small helpers for the EfficientAT MobileNetV3 (``mn``) backbone."""

from __future__ import annotations

import math
from collections.abc import Callable

from torch import nn


def make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


class ConvNormActivation(nn.Sequential):
    """Conv2d -> norm -> activation, as ``torchvision.ops.misc.ConvNormActivation``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=norm_layer is None)
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels

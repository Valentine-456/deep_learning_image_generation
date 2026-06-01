from __future__ import annotations

from typing import Literal

import torch
from torch import nn


ConvKind = Literal["regular", "transpose"]
SamplingKind = Literal["none", "maxpool", "upsample"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        conv: ConvKind = "regular",
        sampling: SamplingKind = "maxpool",
    ) -> None:
        super().__init__()

        if conv == "regular":
            conv_layer: nn.Module = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            )
        elif conv == "transpose":
            conv_layer = nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            )
        else:
            raise ValueError(f"Unknown convolution kind: {conv}")

        layers: list[nn.Module] = [conv_layer]

        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        if sampling == "maxpool":
            layers.append(nn.MaxPool2d(2))
        elif sampling == "upsample":
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif sampling != "none":
            raise ValueError(f"Unknown sampling kind: {sampling}")

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


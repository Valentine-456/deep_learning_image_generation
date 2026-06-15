from __future__ import annotations

import torch
from torch import nn

from .DDPM import DDPMResidualBlock


class SpatialAttentionBlock(nn.Module):
    """Apply self-attention between all positions in a feature map."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("Attention channels must be divisible by num_heads.")

        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return x + attended


class AttentionDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_heads: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.residual1 = DDPMResidualBlock(in_channels, out_channels, time_dim)
        self.residual2 = DDPMResidualBlock(out_channels, out_channels, time_dim)
        self.attention = SpatialAttentionBlock(out_channels, num_heads)
        self.downsample = (
            nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
            if downsample
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.residual1(x, time_embedding)
        skip = self.residual2(x, time_embedding)
        skip = self.attention(skip)
        return self.downsample(skip), skip


class AttentionUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        num_heads: int,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.residual1 = DDPMResidualBlock(
            in_channels + skip_channels,
            skip_channels,
            time_dim,
        )
        self.residual2 = DDPMResidualBlock(skip_channels, skip_channels, time_dim)
        self.attention = SpatialAttentionBlock(skip_channels, num_heads)
        self.upsample = (
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(skip_channels, out_channels, 3, padding=1),
            )
            if upsample
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat((x, skip), dim=1)
        x = self.residual1(x, time_embedding)
        x = self.residual2(x, time_embedding)
        x = self.attention(x)
        return self.upsample(x)

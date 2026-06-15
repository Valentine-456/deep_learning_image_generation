from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("Time embedding dimension must be even.")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        scale = math.log(10_000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device) * -scale
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class DDPMResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden += self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.residual(x)


class DDPMDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.residual1 = DDPMResidualBlock(in_channels, out_channels, time_dim)
        self.residual2 = DDPMResidualBlock(out_channels, out_channels, time_dim)
        self.downsample = (
            nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
            if downsample
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, time_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.residual1(x, time_embedding)
        skip = self.residual2(x, time_embedding)
        return self.downsample(skip), skip


class DDPMUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.residual1 = DDPMResidualBlock(
            in_channels + skip_channels, skip_channels, time_dim
        )
        self.residual2 = DDPMResidualBlock(skip_channels, skip_channels, time_dim)
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
        return self.upsample(x)


class DDPMUNet(nn.Module):
    """Predict the Gaussian noise present in an image at timestep t."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: list[int] | None = None,
        time_dim: int = 256,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        multipliers = channel_multipliers or [1, 2, 4, 8]
        channels = [base_channels * multiplier for multiplier in multipliers]

        if any(channel % 8 for channel in [base_channels, *channels]):
            raise ValueError("All DDPM channel counts must be divisible by 8.")
        if image_size % (2 ** (len(channels) - 1)):
            raise ValueError("Image size is incompatible with the U-Net depth.")

        self.in_channels = in_channels
        self.image_size = image_size
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_projection = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        for index, level_channels in enumerate(channels):
            self.down_blocks.append(
                DDPMDownBlock(
                    current_channels,
                    level_channels,
                    time_dim,
                    downsample=index < len(channels) - 1,
                )
            )
            current_channels = level_channels

        self.middle1 = DDPMResidualBlock(current_channels, current_channels, time_dim)
        self.middle2 = DDPMResidualBlock(current_channels, current_channels, time_dim)

        self.up_blocks = nn.ModuleList()
        for index in range(len(channels) - 1, -1, -1):
            level_channels = channels[index]
            next_channels = channels[index - 1] if index > 0 else base_channels
            self.up_blocks.append(
                DDPMUpBlock(
                    current_channels,
                    level_channels,
                    next_channels,
                    time_dim,
                    upsample=index > 0,
                )
            )
            current_channels = next_channels

        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.repeat(x.size(0))
        if timesteps.shape != (x.size(0),):
            raise ValueError("Provide one timestep for each image in the batch.")

        time_embedding = self.time_embedding(timesteps)
        x = self.input_projection(x)

        skips: list[torch.Tensor] = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_embedding)
            skips.append(skip)

        x = self.middle1(x, time_embedding)
        x = self.middle2(x, time_embedding)

        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, time_embedding)

        return self.output(x)

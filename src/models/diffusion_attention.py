from __future__ import annotations

import torch
from torch import nn

from .attention_blocks import AttentionDownBlock, AttentionUpBlock, SpatialAttentionBlock
from .DDPM import (
    DDPMDownBlock,
    DDPMResidualBlock,
    DDPMUpBlock,
    SinusoidalTimeEmbedding,
)
from .DDPMscheduler import DDPMScheduler


class AttentionDDPMUNet(nn.Module):
    """DDPM U-Net with middle attention and symmetric attention at one resolution."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: list[int] | None = None,
        time_dim: int = 256,
        image_size: int = 64,
        attention_heads: int = 4,
        attention_resolution: int = 16,
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
        current_resolution = image_size
        for index, level_channels in enumerate(channels):
            downsample = index < len(channels) - 1
            if current_resolution == attention_resolution:
                block = AttentionDownBlock(
                    current_channels,
                    level_channels,
                    time_dim,
                    num_heads=attention_heads,
                    downsample=downsample,
                )
            else:
                block = DDPMDownBlock(
                    current_channels,
                    level_channels,
                    time_dim,
                    downsample=downsample,
                )
            self.down_blocks.append(block)
            current_channels = level_channels
            if downsample:
                current_resolution //= 2

        self.middle1 = DDPMResidualBlock(current_channels, current_channels, time_dim)
        self.middle_attention = SpatialAttentionBlock(
            current_channels,
            num_heads=attention_heads,
        )
        self.middle2 = DDPMResidualBlock(current_channels, current_channels, time_dim)

        self.up_blocks = nn.ModuleList()
        for index in range(len(channels) - 1, -1, -1):
            level_channels = channels[index]
            next_channels = channels[index - 1] if index > 0 else base_channels
            level_resolution = image_size // (2**index)
            if level_resolution == attention_resolution:
                block = AttentionUpBlock(
                    current_channels,
                    level_channels,
                    next_channels,
                    time_dim,
                    num_heads=attention_heads,
                    upsample=index > 0,
                )
            else:
                block = DDPMUpBlock(
                    current_channels,
                    level_channels,
                    next_channels,
                    time_dim,
                    upsample=index > 0,
                )
            self.up_blocks.append(block)
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
        x = self.middle_attention(x)
        x = self.middle2(x, time_embedding)

        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, time_embedding)

        return self.output(x)


class AttentionDiffusion(nn.Module):
    """Combine the attention U-Net with the fixed DDPM scheduler."""

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 64,
        base_channels: int = 64,
        channel_multipliers: list[int] | None = None,
        time_dim: int = 256,
        attention_heads: int = 4,
        attention_resolution: int = 16,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.unet = AttentionDDPMUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            time_dim=time_dim,
            image_size=image_size,
            attention_heads=attention_heads,
            attention_resolution=attention_resolution,
        )
        self.scheduler = DDPMScheduler(timesteps, beta_start, beta_end)

    def forward(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = clean_images.size(0)
        device = clean_images.device
        if timesteps is None:
            timesteps = torch.randint(
                self.scheduler.timesteps,
                (batch_size,),
                device=device,
            )
        if noise is None:
            noise = torch.randn_like(clean_images)

        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
        predicted_noise = self.unet(noisy_images, timesteps)
        return {
            "predicted_noise": predicted_noise,
            "target_noise": noise,
            "noisy_images": noisy_images,
            "timesteps": timesteps,
        }

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        images = torch.randn(
            num_samples,
            self.in_channels,
            self.image_size,
            self.image_size,
            device=device,
        )
        for timestep in range(self.scheduler.timesteps - 1, -1, -1):
            timesteps = torch.full(
                (num_samples,), timestep, device=device, dtype=torch.long
            )
            predicted_noise = self.unet(images, timesteps)
            images = self.scheduler.step(images, predicted_noise, timestep)
        return images.clamp(-1.0, 1.0)

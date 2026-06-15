from __future__ import annotations

import torch
from torch import nn

from .DDPM import DDPMUNet
from .DDPMscheduler import DDPMScheduler


class Diffusion(nn.Module):
    """Combine the trainable U-Net with the fixed DDPM scheduler."""

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 64,
        base_channels: int = 64,
        channel_multipliers: list[int] | None = None,
        time_dim: int = 256,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.unet = DDPMUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            time_dim=time_dim,
            image_size=image_size,
        )
        self.scheduler = DDPMScheduler(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

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
                0,
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
                (num_samples,),
                timestep,
                device=device,
                dtype=torch.long,
            )
            predicted_noise = self.unet(images, timesteps)
            images = self.scheduler.step(images, predicted_noise, timestep)

        return images.clamp(-1.0, 1.0)

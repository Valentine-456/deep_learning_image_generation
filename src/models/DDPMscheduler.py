from __future__ import annotations

import torch
from torch import nn


class DDPMScheduler(nn.Module):
    """Add noise during training and perform one reverse DDPM step."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("DDPM requires at least two timesteps.")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("Betas must satisfy 0 < beta_start < beta_end < 1.")

        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = (
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        coefficients = values.gather(0, timesteps)
        return coefficients.reshape(x.size(0), *((1,) * (x.ndim - 1)))

    def add_noise(
        self,
        clean_images: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.alpha_bars.sqrt(), timesteps, clean_images)
        sqrt_one_minus_alpha_bar = self._extract(
            (1.0 - self.alpha_bars).sqrt(), timesteps, clean_images
        )
        return sqrt_alpha_bar * clean_images + sqrt_one_minus_alpha_bar * noise

    def step(
        self,
        noisy_images: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        alpha = self.alphas[timestep]
        alpha_bar = self.alpha_bars[timestep]
        beta = self.betas[timestep]

        model_mean = (
            noisy_images
            - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise
        ) / torch.sqrt(alpha)

        if timestep == 0:
            return model_mean

        random_noise = torch.randn_like(noisy_images)
        variance = self.posterior_variance[timestep]
        return model_mean + torch.sqrt(variance) * random_noise

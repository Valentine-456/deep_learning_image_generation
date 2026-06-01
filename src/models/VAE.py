from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConvBlock


class VAEImageGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        image_size: int = 128,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        if image_size % (2 ** len(hidden_dims)) != 0:
            raise ValueError(
                "image_size must be divisible by 2 ** len(hidden_dims). "
                f"Got image_size={image_size}, hidden_dims={hidden_dims}."
            )

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.hidden_dims = hidden_dims
        self.feature_size = image_size // (2 ** len(hidden_dims))
        self.encoder_out_channels = hidden_dims[-1]
        self.flattened_dim = self.encoder_out_channels * self.feature_size * self.feature_size

        encoder_layers: list[nn.Module] = []
        current_channels = in_channels

        for out_channels in hidden_dims:
            encoder_layers.append(
                ConvBlock(
                    current_channels,
                    out_channels,
                    conv="regular",
                    sampling="maxpool",
                )
            )
            current_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flattened_dim)

        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(hidden_dims))

        for in_ch, out_ch in zip(reversed_dims, reversed_dims[1:]):
            decoder_layers.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    conv="regular",
                    sampling="upsample",
                )
            )

        self.decoder = nn.Sequential(*decoder_layers)
        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[0], in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_input(z)
        decoded = decoded.view(
            -1,
            self.encoder_out_channels,
            self.feature_size,
            self.feature_size,
        )
        decoded = self.decoder(decoded)
        return self.final_layer(decoded)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        reconstruction_loss = F.l1_loss(reconstruction, x)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        loss = reconstruction_loss + kl_weight * kl_loss
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

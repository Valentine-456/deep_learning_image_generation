from __future__ import annotations

import math

import torch
from torch import nn


def _validate_image_size(image_size: int) -> int:
    if image_size < 32 or image_size & (image_size - 1) != 0:
        raise ValueError(
            "DCGAN image_size must be a power of two and at least 32. "
            f"Got {image_size}."
        )
    return int(math.log2(image_size))


class DCGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        features: int = 64,
        image_size: int = 128,
    ) -> None:
        super().__init__()

        log2_size = _validate_image_size(image_size)
        num_blocks = log2_size - 3
        channel_multiplier = 2 ** num_blocks

        self.latent_dim = latent_dim

        layers: list[nn.Module] = [
            self._block(latent_dim, features * channel_multiplier, 4, 1, 0)
        ]

        current_channels = features * channel_multiplier
        for _ in range(num_blocks):
            next_channels = current_channels // 2
            if next_channels < features:
                next_channels = features
            layers.append(self._block(current_channels, next_channels, 4, 2, 1))
            current_channels = next_channels

        layers.extend(
            [
                nn.ConvTranspose2d(
                    current_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
            ]
        )

        self.net = nn.Sequential(*layers)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def sample(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        return self(z)


class UpsampleConvGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        features: int = 64,
        image_size: int = 128,
    ) -> None:
        super().__init__()

        log2_size = _validate_image_size(image_size)
        num_blocks = log2_size - 2
        channel_multiplier = 2 ** (num_blocks - 1)

        self.latent_dim = latent_dim

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                latent_dim,
                features * channel_multiplier,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(features * channel_multiplier),
            nn.ReLU(inplace=True),
        ]

        current_channels = features * channel_multiplier
        for _ in range(num_blocks):
            next_channels = max(current_channels // 2, max(features // 2, 1))
            layers.append(self._block(current_channels, next_channels))
            current_channels = next_channels

        layers.extend(
            [
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            ]
        )

        self.net = nn.Sequential(*layers)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def sample(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        return self(z)


class DCGANDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        features: int = 64,
        image_size: int = 128,
    ) -> None:
        super().__init__()

        log2_size = _validate_image_size(image_size)
        num_blocks = log2_size - 3

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        current_channels = features
        for _ in range(num_blocks):
            next_channels = current_channels * 2
            layers.append(self._block(current_channels, next_channels))
            current_channels = next_channels

        layers.append(nn.Conv2d(current_channels, 1, kernel_size=4, stride=1, padding=0))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class DCGAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        image_size: int = 128,
        generator_features: int = 64,
        discriminator_features: int = 64,
        generator_type: str = "transpose",
    ) -> None:
        super().__init__()
        generator_cls = {
            "transpose": DCGANGenerator,
            "upsample": UpsampleConvGenerator,
        }.get(generator_type)
        if generator_cls is None:
            raise ValueError(f"Unknown DCGAN generator type: {generator_type}")

        self.generator = generator_cls(
            latent_dim=latent_dim,
            out_channels=in_channels,
            features=generator_features,
            image_size=image_size,
        )
        self.discriminator = DCGANDiscriminator(
            in_channels=in_channels,
            features=discriminator_features,
            image_size=image_size,
        )

    def sample(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        return self.generator.sample(num_samples, device)

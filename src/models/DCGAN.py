from __future__ import annotations

import torch
from torch import nn


class DCGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        features: int = 64,
        image_size: int = 128,
    ) -> None:
        super().__init__()

        if image_size != 128:
            raise ValueError("This DCGAN generator is configured for 128x128 images.")

        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            self._block(latent_dim, features * 16, 4, 1, 0),
            self._block(features * 16, features * 8, 4, 2, 1),
            self._block(features * 8, features * 4, 4, 2, 1),
            self._block(features * 4, features * 2, 4, 2, 1),
            self._block(features * 2, features, 4, 2, 1),
            nn.ConvTranspose2d(features, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

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

        if image_size != 128:
            raise ValueError("This upsample-convolution generator is configured for 128x128 images.")

        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(inplace=True),
            self._block(features * 16, features * 8),
            self._block(features * 8, features * 4),
            self._block(features * 4, features * 2),
            self._block(features * 2, features),
            self._block(features, features // 2),
            nn.Conv2d(features // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

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

        if image_size != 128:
            raise ValueError("This DCGAN discriminator is configured for 128x128 images.")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features, features * 2),
            self._block(features * 2, features * 4),
            self._block(features * 4, features * 8),
            self._block(features * 8, features * 16),
            nn.Conv2d(features * 16, 1, kernel_size=4, stride=1, padding=0),
        )

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

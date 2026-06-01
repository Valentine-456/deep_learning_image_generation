from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.models import DCGAN


class DCGANTrainer:
    @staticmethod
    def run_epoch(
        model: DCGAN,
        loader: DataLoader,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> dict[str, float]:
        model.train()
        total_generator_loss = 0.0
        total_discriminator_loss = 0.0
        total_real_score = 0.0
        total_fake_score = 0.0
        n_images = 0

        for real_images in tqdm(loader, leave=False, desc="train"):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            discriminator_optimizer.zero_grad()
            real_logits = model.discriminator(real_images)
            real_loss = criterion(real_logits, real_labels)

            noise = torch.randn(batch_size, model.generator.latent_dim, 1, 1, device=device)
            fake_images = model.generator(noise)
            fake_logits = model.discriminator(fake_images.detach())
            fake_loss = criterion(fake_logits, fake_labels)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            fake_logits_for_generator = model.discriminator(fake_images)
            generator_loss = criterion(fake_logits_for_generator, real_labels)
            generator_loss.backward()
            generator_optimizer.step()

            total_generator_loss += generator_loss.item() * batch_size
            total_discriminator_loss += discriminator_loss.item() * batch_size
            total_real_score += torch.sigmoid(real_logits).mean().item() * batch_size
            total_fake_score += torch.sigmoid(fake_logits).mean().item() * batch_size
            n_images += batch_size

        return {
            "generator_loss": total_generator_loss / n_images,
            "discriminator_loss": total_discriminator_loss / n_images,
            "real_score": total_real_score / n_images,
            "fake_score": total_fake_score / n_images,
        }

    @staticmethod
    @torch.no_grad()
    def save_samples(
        model: DCGAN,
        fixed_noise: torch.Tensor,
        out_dir: Path,
        epoch: int,
        device: torch.device,
    ) -> None:
        model.eval()
        samples = model.generator(fixed_noise.to(device))
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image((samples + 1) / 2, out_dir / f"epoch_{epoch:03d}_samples.jpg", nrow=4)

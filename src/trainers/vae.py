from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.models import VAEImageGenerator


class VAETrainer:
    @staticmethod
    def run_epoch(
        model: VAEImageGenerator,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device,
        train: bool,
        kl_weight: float,
    ) -> dict[str, float]:
        model.train(train)
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
        n_images = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images in tqdm(loader, leave=False, desc="train" if train else "val"):
                images = images.to(device)

                if train:
                    assert optimizer is not None
                    optimizer.zero_grad()

                reconstruction, mu, logvar = model(images)
                losses = model.loss_function(reconstruction, images, mu, logvar, kl_weight)

                if train:
                    losses["loss"].backward()
                    optimizer.step()

                batch_size = images.size(0)
                n_images += batch_size
                for key in totals:
                    totals[key] += losses[key].item() * batch_size

        return {key: value / n_images for key, value in totals.items()}

    @staticmethod
    @torch.no_grad()
    def save_samples(
        model: VAEImageGenerator,
        images: torch.Tensor,
        out_dir: Path,
        epoch: int,
        device: torch.device,
    ) -> None:
        model.eval()
        images = images.to(device)
        reconstruction, _, _ = model(images)
        samples = model.sample(num_samples=images.size(0), device=device)

        out_dir.mkdir(parents=True, exist_ok=True)
        save_image((images + 1) / 2, out_dir / f"epoch_{epoch:03d}_real.jpg", nrow=4)
        save_image(
            (reconstruction + 1) / 2,
            out_dir / f"epoch_{epoch:03d}_reconstruction.jpg",
            nrow=4,
        )
        save_image((samples + 1) / 2, out_dir / f"epoch_{epoch:03d}_samples.jpg", nrow=4)

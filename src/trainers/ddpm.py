from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.models import Diffusion


class DDPMTrainer:
    @staticmethod
    def run_epoch(
        model: Diffusion,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device,
        train: bool,
    ) -> dict[str, float]:
        model.train(train)
        total_loss = 0.0
        n_images = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images in tqdm(loader, leave=False, desc="train" if train else "val"):
                images = images.to(device)

                if train:
                    if optimizer is None:
                        raise ValueError("Training requires an optimizer.")
                    optimizer.zero_grad()

                output = model(images)
                loss = F.mse_loss(output["predicted_noise"], output["target_noise"])

                if train:
                    loss.backward()
                    optimizer.step()

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                n_images += batch_size

        return {"loss": total_loss / n_images}

    @staticmethod
    @torch.no_grad()
    def save_samples(
        model: Diffusion,
        num_samples: int,
        out_dir: Path,
        epoch: int,
        device: torch.device,
    ) -> None:
        model.eval()
        samples = model.sample(num_samples, device)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image(
            (samples + 1.0) / 2.0,
            out_dir / f"epoch_{epoch:03d}_samples.jpg",
            nrow=4,
        )

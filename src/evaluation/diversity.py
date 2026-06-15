from __future__ import annotations

import random
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from .common import denormalize_to_01


def find_nearest_train_neighbors(
    generated: torch.Tensor,
    train_images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    generated_flat = generated.flatten(start_dim=1)
    train_flat = train_images.flatten(start_dim=1)
    distances = torch.cdist(generated_flat, train_flat, p=2)
    nearest_indices = distances.argmin(dim=1)
    nearest_images = train_images[nearest_indices]
    nearest_distances = distances.min(dim=1).values
    return nearest_images, nearest_distances


def save_nearest_neighbor_grid(
    generated: torch.Tensor,
    nearest_train: torch.Tensor,
    out_path: Path,
    max_rows: int = 16,
) -> None:
    rows = min(generated.size(0), max_rows)
    pairs = []
    for index in range(rows):
        pairs.append(generated[index])
        pairs.append(nearest_train[index])
    grid = make_grid(denormalize_to_01(torch.stack(pairs)), nrow=2, padding=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


def _get_lpips_model(device: torch.device):
    try:
        import lpips
    except ImportError as error:
        raise ImportError(
            "LPIPS is required for diversity evaluation. Install with: pip install lpips"
        ) from error

    model = lpips.LPIPS(net="alex")
    return model.to(device).eval()


@torch.no_grad()
def compute_lpips_pair_diversity(
    generated: torch.Tensor,
    num_pairs: int,
    device: torch.device,
    seed: int = 42,
) -> dict[str, float | int]:
    if generated.size(0) < 2:
        raise ValueError("Need at least 2 generated images for LPIPS pair diversity.")

    lpips_model = _get_lpips_model(device)
    rng = random.Random(seed)
    num_pairs = min(num_pairs, generated.size(0) * (generated.size(0) - 1) // 2)

    scores: list[float] = []
    used_pairs: set[tuple[int, int]] = set()
    while len(scores) < num_pairs:
        left = rng.randrange(generated.size(0))
        right = rng.randrange(generated.size(0))
        if left == right:
            continue
        pair = tuple(sorted((left, right)))
        if pair in used_pairs:
            continue
        used_pairs.add(pair)

        image_a = generated[left].unsqueeze(0).to(device)
        image_b = generated[right].unsqueeze(0).to(device)
        score = lpips_model(image_a, image_b).item()
        scores.append(score)

    tensor_scores = torch.tensor(scores, dtype=torch.float32)
    return {
        "num_pairs": len(scores),
        "lpips_mean": float(tensor_scores.mean().item()),
        "lpips_std": float(tensor_scores.std(unbiased=False).item()),
        "lpips_min": float(tensor_scores.min().item()),
        "lpips_max": float(tensor_scores.max().item()),
    }

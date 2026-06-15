"""Mode-collapse / diversity evaluation for generative models.

Produces a sample grid, nearest-neighbor comparison against train images, and
LPIPS diversity over random generated-image pairs.

Usage example::

    python scripts/eval_diversity.py \\
        --config configs/DCGAN_64.yaml \\
        --checkpoint results/20260605-165157_dcgan_64/best_model.pth \\
        --model-name dcgan --num-samples 64 --num-lpips-pairs 500 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import ImageGenerationDataset
from src.evaluation.common import (
    add_common_args,
    generate_samples_batched,
    get_split_paths,
    load_config,
    load_config_from_run_dir,
    load_model_for_eval,
    resolve_checkpoint,
    resolve_device,
    save_sample_grid,
)
from src.evaluation.diversity import (
    compute_lpips_pair_diversity,
    find_nearest_train_neighbors,
    save_nearest_neighbor_grid,
)
from src.utils import save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sample diversity and nearest-neighbor overlap."
    )
    add_common_args(parser)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of generated images for the grid and NN comparison.",
    )
    parser.add_argument(
        "--num-lpips-pairs",
        type=int,
        default=500,
        help="Number of random generated pairs for LPIPS diversity.",
    )
    parser.add_argument(
        "--lpips-pool-size",
        type=int,
        default=128,
        help="Number of generated images used as the LPIPS pair pool.",
    )
    parser.add_argument(
        "--train-pool-size",
        type=int,
        default=2000,
        help="Maximum number of train images used for nearest-neighbor search.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Generation batch size.",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Skip LPIPS computation (grid and nearest-neighbor only).",
    )
    return parser.parse_args()


def resolve_cfg(args: argparse.Namespace) -> dict:
    if args.config is not None:
        return load_config(args.config)
    if args.run_dir is not None:
        return load_config_from_run_dir(args.run_dir)
    raise ValueError("Provide --config or --run-dir.")


def default_model_name(cfg: dict) -> str:
    model_type = cfg["model"]["type"].lower()
    if model_type == "ddpm_attention":
        return "ddpm_attention"
    return model_type


def load_train_tensors(cfg: dict, max_images: int, device: torch.device) -> torch.Tensor:
    train_paths, _, _ = get_split_paths(cfg)
    image_size = cfg["data"]["image_size"]
    dataset = ImageGenerationDataset(train_paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    batches: list[torch.Tensor] = []
    total = 0
    for batch in tqdm(loader, desc="load train", leave=False):
        if total >= max_images:
            break
        batch = batch[: max_images - total]
        batches.append(batch)
        total += batch.size(0)

    return torch.cat(batches, dim=0).to(device)


def main() -> None:
    args = parse_args()
    cfg = resolve_cfg(args)
    model_name = args.model_name or default_model_name(cfg)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    checkpoint_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    model = load_model_for_eval(cfg, checkpoint_path, device)
    out_dir = Path(args.output_dir) / "diversity" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"model={model_name}  checkpoint={checkpoint_path}  "
        f"device={device}  num_samples={args.num_samples}"
    )

    generated = generate_samples_batched(
        model,
        args.num_samples,
        device,
        batch_size=args.batch_size,
    )
    save_sample_grid(generated, out_dir / "grid_64.jpg", nrow=8)

    train_images = load_train_tensors(cfg, args.train_pool_size, device)
    nearest_train, nearest_distances = find_nearest_train_neighbors(
        generated.to(device),
        train_images,
    )
    save_nearest_neighbor_grid(
        generated,
        nearest_train.cpu(),
        out_dir / "nearest_neighbors.jpg",
        max_rows=args.num_samples,
    )

    summary: dict[str, object] = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(out_dir),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "nearest_neighbor_mse_mean": float(nearest_distances.mean().item()),
        "nearest_neighbor_mse_min": float(nearest_distances.min().item()),
        "nearest_neighbor_mse_max": float(nearest_distances.max().item()),
    }

    if not args.skip_lpips:
        lpips_pool = generate_samples_batched(
            model,
            min(args.lpips_pool_size, max(args.num_lpips_pairs + 1, 2)),
            device,
            batch_size=args.batch_size,
        )
        lpips_stats = compute_lpips_pair_diversity(
            lpips_pool,
            num_pairs=args.num_lpips_pairs,
            device=device,
            seed=args.seed,
        )
        summary["lpips"] = lpips_stats
        save_json(out_dir / "lpips_diversity.json", lpips_stats)

    save_json(out_dir / "summary.json", summary)
    print(f"Saved diversity outputs to {out_dir}")


if __name__ == "__main__":
    main()

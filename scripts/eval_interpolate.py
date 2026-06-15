"""Latent-space interpolation for DCGAN and VAE models.

DCGAN uses linear interpolation between two latent vectors through the generator.
VAE uses the same interpolation in latent space followed by decode().

DDPM interpolation is not implemented because this codebase only has stochastic
full reverse diffusion sampling (no DDIM / controlled-noise path yet).

Usage examples::

    python scripts/eval_interpolate.py \\
        --config configs/DCGAN_64.yaml \\
        --checkpoint results/20260605-165157_dcgan_64/best_model.pth \\
        --model-name dcgan --seed 42

    python scripts/eval_interpolate.py \\
        --config configs/VAE_64.yaml \\
        --checkpoint results/20260605-160116_vae_64/best_model.pth \\
        --model-name vae --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.common import (
    add_common_args,
    denormalize_to_01,
    load_config,
    load_config_from_run_dir,
    load_model_for_eval,
    resolve_checkpoint,
    resolve_device,
)
from src.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent interpolation evaluation.")
    add_common_args(parser)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Total interpolation frames including both endpoints (default: 10).",
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


@torch.no_grad()
def interpolate_dcgan(
    model: torch.nn.Module,
    device: torch.device,
    latent_dim: int,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z1 = torch.randn(1, latent_dim, 1, 1, device=device)
    z2 = torch.randn(1, latent_dim, 1, 1, device=device)
    alphas = torch.linspace(0.0, 1.0, num_steps, device=device)
    latent_vectors = [(1.0 - alpha) * z1 + alpha * z2 for alpha in alphas]
    z = torch.cat(latent_vectors, dim=0)
    images = model.generator(z)
    return images, z1.cpu(), z2.cpu()


@torch.no_grad()
def interpolate_vae(
    model: torch.nn.Module,
    device: torch.device,
    latent_dim: int,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0.0, 1.0, num_steps, device=device)
    latent_vectors = [(1.0 - alpha) * z1 + alpha * z2 for alpha in alphas]
    z = torch.cat(latent_vectors, dim=0)
    images = model.decode(z)
    return images, z1.cpu(), z2.cpu()


def main() -> None:
    args = parse_args()
    cfg = resolve_cfg(args)
    model_name = args.model_name or default_model_name(cfg)
    model_type = cfg["model"]["type"].lower()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    if model_type in {"ddpm", "ddpm_attention"}:
        raise NotImplementedError(
            "DDPM latent interpolation is not implemented. "
            "TODO: add DDIM or controlled-noise sampling before interpolating."
        )

    checkpoint_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    model = load_model_for_eval(cfg, checkpoint_path, device)
    latent_dim = cfg["model"].get("latent_dim", 128)

    if model_type == "dcgan":
        images, z1, z2 = interpolate_dcgan(model, device, latent_dim, args.num_steps)
    elif model_type == "vae":
        images, z1, z2 = interpolate_vae(model, device, latent_dim, args.num_steps)
    else:
        raise ValueError(f"Interpolation not supported for model type: {model_type}")

    out_dir = Path(args.output_dir) / "interpolation" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image(denormalize_to_01(images), out_dir / "interpolation.jpg", nrow=args.num_steps)
    torch.save(z1, out_dir / "z1.pt")
    torch.save(z2, out_dir / "z2.pt")

    print(f"model={model_name}  checkpoint={checkpoint_path}  device={device}")
    print(f"Saved interpolation grid => {out_dir / 'interpolation.jpg'}")
    print(f"Saved latent vectors => {out_dir / 'z1.pt'}, {out_dir / 'z2.pt'}")


if __name__ == "__main__":
    main()

"""Generate equal-count image folders for FID evaluation with pytorch_fid.

Usage examples
--------------
Export real test images once::

    python scripts/eval_fid.py --export-real --config configs/VAE_64.yaml --num-samples 5000

Generate samples for one model::

    python scripts/eval_fid.py \\
        --config configs/DCGAN_64.yaml \\
        --checkpoint results/20260605-165157_dcgan_64/best_model.pth \\
        --model-name dcgan --num-samples 5000 --seed 42

Compute FID externally (after all model folders are ready)::

    python -m pytorch_fid evaluation/real evaluation/dcgan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.common import (
    add_common_args,
    export_real_test_images,
    generate_samples_batched,
    load_config,
    load_config_from_run_dir,
    load_model_for_eval,
    resolve_checkpoint,
    resolve_device,
    save_fid_images,
    save_sample_grid,
)
from src.utils import save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare folders for FID evaluation.")
    add_common_args(parser)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of images to export/generate per folder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Generation batch size.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Number of images in the qualitative sample grid.",
    )
    parser.add_argument(
        "--export-real",
        action="store_true",
        help="Export real test images to <output-dir>/real and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated images.",
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


def main() -> None:
    args = parse_args()
    cfg = resolve_cfg(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    real_dir = output_dir / "real"

    if args.export_real:
        summary = export_real_test_images(cfg, real_dir, num_samples=args.num_samples)
        summary["seed"] = args.seed
        save_json(output_dir / "real_export_summary.json", summary)
        print(
            f"Exported {summary['sample_count']} real images to {real_dir} "
            f"(image_size={summary['image_size']})."
        )
        return

    if args.model_name is None:
        args.model_name = default_model_name(cfg)

    checkpoint_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    model_out_dir = output_dir / args.model_name
    if model_out_dir.exists() and any(model_out_dir.glob("*.png")) and not args.force:
        raise FileExistsError(
            f"{model_out_dir} already contains PNG files. Use --force to overwrite."
        )

    model = load_model_for_eval(cfg, checkpoint_path, device)
    print(
        f"model={args.model_name}  checkpoint={checkpoint_path}  "
        f"device={device}  num_samples={args.num_samples}"
    )

    model_out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    while saved < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - saved)
        batch = model.sample(batch_size, device=device)
        save_fid_images(batch, model_out_dir, start_idx=saved)
        saved += batch_size

    grid_count = min(args.grid_size, args.num_samples)
    seed_everything(args.seed)
    grid_images = generate_samples_batched(model, grid_count, device, batch_size=args.batch_size)
    save_sample_grid(grid_images, model_out_dir / "sample_grid.jpg", nrow=8)

    summary = {
        "model_name": args.model_name,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(model_out_dir),
        "sample_count": saved,
        "seed": args.seed,
        "image_size": cfg["data"]["image_size"],
        "config": str(args.config) if args.config else None,
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "fid_command": f"python -m pytorch_fid {real_dir} {model_out_dir}",
    }
    save_json(model_out_dir / "fid_run_summary.json", summary)

    summary_path = output_dir / "fid_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            all_summaries = json.load(file)
    else:
        all_summaries = {}
    all_summaries[args.model_name] = summary
    save_json(summary_path, all_summaries)

    print(f"Saved {saved} images to {model_out_dir}")
    print(f"Sample grid => {model_out_dir / 'sample_grid.jpg'}")
    print(f"FID command => {summary['fid_command']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.dataset import ImageGenerationDataset, split_image_paths
from src.models import build_model
from src.utils import seed_everything


def resolve_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config_from_run_dir(run_dir: Path) -> dict:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json in run directory: {run_dir}")
    with results_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if "config" not in payload:
        raise KeyError(f"'config' not found in {results_path}")
    return payload["config"]


def resolve_checkpoint(
    run_dir: Path | None,
    checkpoint: Path | None,
) -> Path:
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    if run_dir is None:
        raise ValueError("Provide --checkpoint or --run-dir.")

    run_dir = Path(run_dir)
    best_path = run_dir / "best_model.pth"
    if best_path.exists():
        return best_path

    final_path = run_dir / "final_model.pth"
    if final_path.exists():
        return final_path

    raise FileNotFoundError(
        f"No checkpoint found in {run_dir}. Expected best_model.pth or final_model.pth."
    )


def load_model_for_eval(
    cfg: dict,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = cfg["model"]["type"].lower()

    if model_type == "dcgan":
        if isinstance(state, dict) and "generator" in state:
            model.generator.load_state_dict(state["generator"])
        else:
            model.generator.load_state_dict(state)
    else:
        model.load_state_dict(state)

    model.eval()
    return model


def denormalize_to_01(images: torch.Tensor) -> torch.Tensor:
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (denormalize_to_01(image).permute(1, 2, 0).cpu().numpy() * 255.0).round()
    array = array.astype("uint8")
    return Image.fromarray(array, mode="RGB")


def save_fid_images(
    images: torch.Tensor,
    out_dir: Path,
    start_idx: int = 0,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(images):
        index = start_idx + offset
        tensor_to_pil(image).save(out_dir / f"{index:05d}.png")
    return start_idx + images.size(0)


def save_sample_grid(
    images: torch.Tensor,
    out_path: Path,
    nrow: int = 8,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize_to_01(images), out_path, nrow=nrow)


def get_split_paths(cfg: dict) -> tuple[list[Path], list[Path], list[Path]]:
    data_cfg = cfg["data"]
    training_cfg = cfg.get("training", {})
    return split_image_paths(
        Path(data_cfg["cache_dir"]),
        seed=training_cfg.get("seed", 42),
        train_fraction=data_cfg.get("train_fraction", 0.8),
        val_fraction=data_cfg.get("val_fraction", 0.1),
        test_fraction=data_cfg.get("test_fraction"),
        only_cats=data_cfg.get("only_cats", False),
    )


@torch.no_grad()
def generate_samples_batched(
    model: torch.nn.Module,
    num_samples: int,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    generated = 0
    while generated < num_samples:
        current_batch = min(batch_size, num_samples - generated)
        batches.append(model.sample(current_batch, device=device))
        generated += current_batch
    return torch.cat(batches, dim=0)


def export_real_test_images(
    cfg: dict,
    out_dir: Path,
    num_samples: int | None = None,
) -> dict[str, int | str]:
    _, _, test_paths = get_split_paths(cfg)
    image_size = cfg["data"]["image_size"]
    dataset = ImageGenerationDataset(test_paths, image_size=image_size)

    if num_samples is None:
        num_samples = len(dataset)
    num_samples = min(num_samples, len(dataset))

    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    saved = 0
    for batch in tqdm(loader, desc="export real", leave=False):
        if saved >= num_samples:
            break
        batch = batch[: num_samples - saved]
        save_fid_images(batch, out_dir, start_idx=saved)
        saved += batch.size(0)

    return {
        "output_dir": str(out_dir),
        "sample_count": saved,
        "image_size": image_size,
    }


def add_common_args(parser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to model YAML config.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory (uses results.json config and default checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: <run-dir>/best_model.pth).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Output subfolder name, e.g. vae, dcgan, ddpm, ddpm_attention.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation"),
        help="Root directory for evaluation outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string, e.g. cuda or cpu (auto-detect if omitted).",
    )

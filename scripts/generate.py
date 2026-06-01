from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model


CONFIG_PATH = Path("configs/VAE.yaml")
CHECKPOINT_PATH = Path("results/YOUR_RUN_NAME/best_model.pth")
OUTPUT_DIR = Path("results/generated")
NUM_IMAGES = 16
NROW = 4


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Checkpoint does not exist. Update CHECKPOINT_PATH in "
            f"{Path(__file__).name}: {CHECKPOINT_PATH}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        images = model.sample(NUM_IMAGES, device=device)

    output_path = OUTPUT_DIR / "generated.jpg"
    save_image((images + 1) / 2, output_path, nrow=NROW)
    print(f"Saved generated images to {output_path}")


if __name__ == "__main__":
    main()

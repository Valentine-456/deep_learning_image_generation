from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.CatFilterLoader import CatFilterLoader


class ImageGenerationDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.image_paths = image_paths

        transform_steps: list[transforms.transforms.Transform] = []

        if augment:
            transform_steps.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

        transform_steps.extend(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


def find_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jpg") if path.is_file())


def split_image_paths(
    root: Path,
    seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float | None = None,
    only_cats: bool = False,
) -> tuple[list[Path], list[Path], list[Path]]:
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        train_paths = find_images(train_dir)
        val_paths = find_images(val_dir)
        test_paths = find_images(test_dir)

        if only_cats:
            train_paths = CatFilterLoader.filter(train_paths)
            val_paths = CatFilterLoader.filter(val_paths)
            test_paths = CatFilterLoader.filter(test_paths)

        return train_paths, val_paths, test_paths

    image_paths = find_images(root)

    if only_cats:
        image_paths = CatFilterLoader.filter(image_paths)

    random.Random(seed).shuffle(image_paths)

    train_end = int(len(image_paths) * train_fraction)
    val_end = train_end + int(len(image_paths) * val_fraction)
    test_end = None

    if test_fraction is not None:
        test_end = val_end + int(len(image_paths) * test_fraction)

    return (
        image_paths[:train_end],
        image_paths[train_end:val_end],
        image_paths[val_end:test_end],
    )

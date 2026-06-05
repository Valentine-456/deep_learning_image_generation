from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


INPUT_DIR = Path("data")
OUTPUT_DIR = Path("preprocessed") # preprocessed | preprocessed64
IMAGE_SIZE = 64 # 64
OVERWRITE = False


def iter_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".jpg"
    )


def resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    if width < height:
        new_width = size
        new_height = round(height * size / width)
    else:
        new_height = size
        new_width = round(width * size / height)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def center_crop_square(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))


def output_path_for(input_path: Path) -> Path:
    relative_path = input_path.relative_to(INPUT_DIR)
    return OUTPUT_DIR / relative_path


def preprocess_image(input_path: Path, output_path: Path) -> None:
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = resize_shorter_side(image, IMAGE_SIZE)
        image = center_crop_square(image, IMAGE_SIZE)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=95, optimize=True)


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    image_paths = iter_image_paths(INPUT_DIR)

    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {INPUT_DIR}")

    processed = 0
    skipped = 0
    failed = 0

    for input_path in tqdm(image_paths, desc="Preprocessing images"):
        output_path = output_path_for(input_path)

        if output_path.exists() and not OVERWRITE:
            skipped += 1
            continue

        try:
            preprocess_image(input_path, output_path)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            failed += 1
            print(f"Failed to preprocess {input_path}: {error}")
            continue

        processed += 1

    print(
        "Done. "
        f"Processed: {processed}. "
        f"Skipped existing: {skipped}. "
        f"Failed: {failed}. "
        f"Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()

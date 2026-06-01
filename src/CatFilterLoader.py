from __future__ import annotations

from pathlib import Path


class CatFilterLoader:
    @staticmethod
    def is_cat_file(path: Path) -> bool:
        return path.name.startswith("cat.")

    @classmethod
    def filter(cls, image_paths: list[Path]) -> list[Path]:
        return [path for path in image_paths if cls.is_cat_file(path)]

"""
Utilities for preparing Tower 2 data.

Current focus: tile normalized full-slide images into fixed-size crops and
replace the originals with their tiled counterparts. These crops will be fed
into Tower 1 for inference before aggregating results for Tower 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class TilingConfig:
    tile_size: int = 256
    stride: int = 256
    min_foreground_ratio: float = 0.05  
    min_pixel_value: int = 15          
    delete_original: bool = True      


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _foreground_ratio(tile: np.ndarray, min_pixel_value: int) -> float:
    """Return fraction of pixels above the background threshold."""
    gray = tile.mean(axis=2)
    foreground = gray > min_pixel_value
    return float(np.count_nonzero(foreground)) / foreground.size


def _iter_tiles(arr: np.ndarray, tile_size: int, stride: int) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yield tiles as (x, y, tile_array)."""
    height, width = arr.shape[:2]
    for y in range(0, max(1, height - tile_size + 1), stride):
        for x in range(0, max(1, width - tile_size + 1), stride):
            if y + tile_size > height or x + tile_size > width:
                continue
            yield x, y, arr[y : y + tile_size, x : x + tile_size, :]


def tile_image(
    image_path: Path,
    output_dir: Path,
    config: TilingConfig,
) -> int:
    """Tile a single image and save the resulting crops.

    Returns:
        Number of tiles saved.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr = np.array(img)

    saved = 0
    for idx, (x, y, tile_arr) in enumerate(_iter_tiles(arr, config.tile_size, config.stride)):
        fg_ratio = _foreground_ratio(tile_arr, config.min_pixel_value)
        if fg_ratio < config.min_foreground_ratio:
            continue

        tile_img = Image.fromarray(tile_arr)
        tile_name = f"{image_path.stem}_x{x:04d}_y{y:04d}_{idx:04d}{image_path.suffix}"
        tile_path = output_dir / tile_name
        tile_img.save(tile_path)
        saved += 1

    return saved


def tile_directory(
    normalized_root: Path,
    config: TilingConfig,
) -> None:
    """Tile every image under `normalized_root`, optionally deleting originals."""
    image_paths = sorted(p for p in normalized_root.rglob("*") if is_image_file(p))
    if not image_paths:
        print(f"No images found under {normalized_root}")
        return

    print(f"Tiling {len(image_paths)} images from {normalized_root}")
    for image_path in tqdm(image_paths, desc="Tiling", unit="img"):
        output_dir = image_path.parent
        tile_count = tile_image(image_path, output_dir, config)
        if tile_count == 0:
            print(f"Warning: {image_path} produced 0 tiles; keeping original.")
            continue
        if config.delete_original:
            image_path.unlink()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Tile Tower 2 normalized slides.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model/data"),
        help="Root directory containing tower2_slides/normalized.",
    )
    parser.add_argument(
        "--normalized-subdir",
        type=str,
        default="tower2_slides/normalized",
        help="Subdirectory (relative to data-root) with normalized slides.",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-foreground", type=float, default=0.05)
    parser.add_argument("--min-pixel", type=int, default=15)
    parser.add_argument("--keep-original", action="store_true", help="Do not delete the original full slide.")

    args = parser.parse_args()

    normalized_root = args.data_root / args.normalized_subdir
    if not normalized_root.exists():
        raise FileNotFoundError(f"Normalized directory not found: {normalized_root}")

    config = TilingConfig(
        tile_size=args.tile_size,
        stride=args.stride,
        min_foreground_ratio=args.min_foreground,
        min_pixel_value=args.min_pixel,
        delete_original=not args.keep_original,
    )
    tile_directory(normalized_root, config)


if __name__ == "__main__":
    main()

#Reviewed preprocessing, all good - Ashley

"""
Utility script to rebuild Tower 2 malignant slides without Macenko normalization.

Steps:
1. Delete all previously normalized/tiled malignant outputs.
2. Reprocess raw malignant slides with simple autocontrast + tissue masking
   (matching the healthy preprocessing minus Macenko).
3. Tile the refreshed malignant slides into 256x256 patches.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

PROJECT_ROOT = Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model")
RAW_DIR = PROJECT_ROOT / "data/tower2_slides/raw"
NORMALIZED_DIR = PROJECT_ROOT / "data/tower2_slides/normalized"
TILES_DIR = PROJECT_ROOT / "data/tower2_slides/tiles"

LABEL = "malignant"
SPLITS = ["train", "val", "test"]

TILE_SIZE = 512
TILE_STRIDE = 256
MIN_SLIDE_TISSUE_RATIO = 0.05
MIN_TILE_TISSUE_RATIO = 0.05


def _preprocess_brightness(rgb: np.ndarray) -> np.ndarray:
    """Apply autocontrast to stabilize illumination."""
    pil_img = Image.fromarray(rgb)
    enhanced = ImageOps.autocontrast(pil_img)
    return np.array(enhanced)


def _tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Simple HSV-based tissue mask."""
    rgb_norm = rgb / 255.0
    max_rgb = rgb_norm.max(axis=2)
    min_rgb = rgb_norm.min(axis=2)
    delta = max_rgb - min_rgb

    saturation = np.zeros_like(max_rgb)
    nonzero = max_rgb != 0
    saturation[nonzero] = delta[nonzero] / max_rgb[nonzero]
    value = max_rgb

    sat_thresh = 0.1
    val_thresh = 0.2
    return (saturation > sat_thresh) & (value > val_thresh)


def _normalize_slide_without_macenko(slide_path: Path) -> np.ndarray | None:
    """Load, autocontrast, and validate a malignant slide without Macenko."""
    with Image.open(slide_path) as img:
        img = img.convert("RGB")
        rgb = np.array(img)

    rgb = _preprocess_brightness(rgb)
    mask = _tissue_mask(rgb)
    if mask.mean() < MIN_SLIDE_TISSUE_RATIO:
        return None
    return rgb


def _save_slide(rgb: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(save_path)


def _tile_slide(rgb: np.ndarray, tiles_dir: Path, slide_id: str) -> int:
    """Reuse the same tiling strategy used elsewhere (256px, 50% stride)."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    h, w, _ = rgb.shape
    saved = 0
    for y in range(0, h - TILE_SIZE + 1, TILE_STRIDE):
        for x in range(0, w - TILE_SIZE + 1, TILE_STRIDE):
            tile = rgb[y : y + TILE_SIZE, x : x + TILE_SIZE]
            if _tissue_mask(tile).mean() < MIN_TILE_TISSUE_RATIO:
                continue
            tile_path = tiles_dir / f"{slide_id}_x{x:04d}_y{y:04d}.png"
            Image.fromarray(tile).save(tile_path)
            saved += 1
    if saved == 0:
        shutil.rmtree(tiles_dir, ignore_errors=True)
    return saved


def _clear_old_outputs() -> None:
    for split in SPLITS:
        shutil.rmtree(NORMALIZED_DIR / split / LABEL, ignore_errors=True)
        shutil.rmtree(TILES_DIR / split / LABEL, ignore_errors=True)


def _process_split(split: str) -> None:
    raw_dir = RAW_DIR / split / LABEL
    if not raw_dir.exists():
        print(f"[skip] {raw_dir} missing")
        return

    normalized_dir = NORMALIZED_DIR / split / LABEL
    tiles_base = TILES_DIR / split / LABEL
    normalized_dir.mkdir(parents=True, exist_ok=True)
    tiles_base.mkdir(parents=True, exist_ok=True)

    slide_paths = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))
    for slide_path in tqdm(slide_paths, desc=f"{split}/{LABEL}", unit="slide"):
        slide_id = slide_path.stem
        norm_rgb = _normalize_slide_without_macenko(slide_path)
        if norm_rgb is None:
            print(f"  [skip] {slide_id} (insufficient tissue)")
            continue

        norm_path = normalized_dir / f"{slide_id}.png"
        _save_slide(norm_rgb, norm_path)

        slide_tiles_dir = tiles_base / slide_id
        tiles_saved = _tile_slide(norm_rgb, slide_tiles_dir, slide_id)
        if tiles_saved == 0:
            print(f"  [warn] {slide_id} yielded no valid tiles")


def main():
    print("Rebuilding malignant slides without Macenko normalization...")
    _clear_old_outputs()
    for split in SPLITS:
        _process_split(split)
    print("Done. Malignant slides reprocessed and retiled.")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import numpy as np
import torch
import torchstain
import json
from PIL import Image, ImageOps
from tqdm import tqdm

RAW_DIR = Path('data') / 'tower2_slides' / 'raw'
NORMALIZED_DIR = Path('data') / 'tower2_slides' / 'normalized'
TILES_DIR = Path('data') / 'tower2_slides' / 'tiles'
RESULTS_DIR = Path('results') / 'tower2'
SUMMARY_PATH = RESULTS_DIR / 'tiling_summary.json'

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def load_raw():
    """
    Load slide images from the specified directory.
    """
    raw_slides = []
    splits = ["train", "val", "test"]
    labels = ["healthy", "malignant"]
    for split in splits:
        for label in labels:
            raw_dir = RAW_DIR / split / label
            slide_paths = sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.png"))
            for path in slide_paths:
                raw_slides.append({"split": split, "label": label, "path": path})
    return raw_slides


def load_normalized():
    """
    Load normalized slide images from the specified directory.
    """
    normalized_slides = []
    splits = ["train", "val", "test"]
    labels = ["healthy", "malignant"]
    for split in splits:
        for label in labels:
            norm_dir = NORMALIZED_DIR / split / label
            slide_paths = sorted(norm_dir.glob("*.png"))
            for path in slide_paths:
                normalized_slides.append({"split": split, "label": label, "path": path})
    return normalized_slides


def load_normalizer():
    """
    Load Macenko Normalizer.pth

    Returns:
        Configured Macenko normalizer, based on raw training healthy slide.    
    """
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    template_path = Path('results/tower2/configs/Macenko_Normalizer.pth')

    if template_path.is_file():
        checkpoint = torch.load(template_path)
        normalizer.HERef = checkpoint["HERef"]
        normalizer.maxCRef = checkpoint["maxCRef"]
    else:
        ref_slide_path = RAW_DIR / "train" / "healthy" / "20160731_121355.jpg"

        with Image.open(ref_slide_path) as img:
            img = img.convert("RGB")
            ref_rgb = np.array(img)
        
        ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).float()
        normalizer.fit(ref_tensor)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"HERef": normalizer.HERef, "maxCRef": normalizer.maxCRef}, template_path)

    return normalizer

def _preprocess_brightness(rgb: np.ndarray) -> np.ndarray:
    """Apply autocontrast to enhance brightness/contrast.
    Args:
        rgb: Input RGB image as a NumPy array.
    Returns:
        Enhanced RGB image as a NumPy array.
    """
    pil_img = Image.fromarray(rgb)
    enhanced_img = ImageOps.autocontrast(pil_img)
    return np.array(enhanced_img)


def _tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a boolean mask of tissue pixels using saturation/value thresholds.
    Args:
        rgb: Input RGB image as a NumPy array.
    Returns:
        Boolean mask where True indicates tissue pixels."""
    rgb_normalized = rgb / 255.0
    max_rgb = rgb_normalized.max(axis=2)
    min_rgb = rgb_normalized.min(axis=2)
    delta = max_rgb - min_rgb

    saturation = np.zeros_like(max_rgb)
    saturation[max_rgb != 0] = delta[max_rgb != 0] / max_rgb[max_rgb != 0]

    value = max_rgb

    sat_threshold = 0.1
    val_threshold = 0.2

    tissue_mask = (saturation > sat_threshold) & (value > val_threshold)
    return tissue_mask

def normalize_slide(
    slide_path: Path,
    normalizer: torchstain.normalizers.MacenkoNormalizer,
    min_slide_tissue_ratio: float = 0.05,
) -> np.ndarray:
    """Normalize a slide using Macenko normalization.
    Args:
        slide_path: Path to the raw slide image.
        normalizer: Configured Macenko normalizer.
        min_slide_tissue_ratio: Minimum tissue ratio to proceed with normalization.
    Returns:
        Normalized RGB image as a NumPy array, or None if skipped.
    """
    with Image.open(slide_path) as img:
        img = img.convert("RGB")
        rgb = np.array(img)

    rgb = _preprocess_brightness(rgb)
    tissue_mask = _tissue_mask(rgb)
    tissue_ratio = tissue_mask.mean()

    if tissue_ratio < min_slide_tissue_ratio:
        return None

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    norm_out = normalizer.normalize(I=tensor, stains=False)
    norm_tensor = norm_out[0] if isinstance(norm_out, tuple) else norm_out

    norm_arr = norm_tensor.detach().cpu().numpy()
    if norm_arr.ndim != 3:
        return None
    if norm_arr.shape[0] in (1, 3):
        norm_arr = np.transpose(norm_arr, (1, 2, 0))
    elif norm_arr.shape[2] in (1, 3):
        pass
    else:
        return None

    norm_rgb = np.clip(norm_arr, 0, 255).astype(np.uint8)

    return norm_rgb

def save_slide_image(rgb: np.ndarray, path: Path) -> None:
    """Save the normalized slide image.
    Args:
        rgb: Normalized RGB image as a NumPy array.
        path: Path to save the image.
    """
    Image.fromarray(rgb).save(path)

def tile_slide(
    rgb: np.ndarray,
    tiles_dir: Path,
    slide_id: str,
    tile_size: int = 256,
    stride: int = 128,
    min_tissue_ratio: float = 0.05,
) -> int:
    """Tile a normalized slide into smaller patches.
    Args:
        rgb: Normalized RGB image as a NumPy array.
        tiles_dir: Directory to save the tiles.
        slide_id: Unique identifier for the slide.
        tile_size: Size of each tile (default: 256).
        stride: Stride for tiling (default: 128).
        min_tissue_ratio: Minimum tissue ratio to keep a tile (default: 0.05).
    Returns:
        Number of saved tiles.
    """
    h, w, _ = rgb.shape
    num_saved = 0
    num_skipped = 0

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = rgb[y:y + tile_size, x:x + tile_size]
            tile_mask = _tissue_mask(tile)

            if tile_mask.mean() < min_tissue_ratio:
                num_skipped += 1
                continue  # Skip tiles with insufficient tissue

            tile_path = tiles_dir / f"{slide_id}_x{x:04d}_y{y:04d}.png"
            save_slide_image(tile, tile_path)
            num_saved += 1

    return num_saved, num_skipped

def process_normalization(normalizer, min_slide_tissue_ratio=0.05):
    """
    Normalize all raw slides and save them into the normalized directory.
    """
    raw_slides = load_raw()

    stats = {
        "processed_slides": 0,
        "failed_slides": {
            "count": 0,
            "slide_ids": []
        },
    }

    for record in tqdm(raw_slides, desc="Normalizing Slides"):
        split = record["split"]
        label = record["label"]
        slide_path = record["path"]
        slide_id = slide_path.stem

        norm_dir = NORMALIZED_DIR / split / label
        norm_dir.mkdir(parents=True, exist_ok=True)
        norm_path = norm_dir / f"{slide_id}.png"

        if norm_path.exists():
            continue

        norm_rgb = normalize_slide(slide_path, normalizer, min_slide_tissue_ratio)

        if norm_rgb is None:
            stats["failed_slides"]["count"] += 1
            stats["failed_slides"]["slide_ids"].append(slide_path.name + "\n")
        else:
            save_slide_image(norm_rgb, norm_path)
            stats["processed_slides"] += 1

    print(f"Normalization completed: {stats['processed_slides']} success, {stats['failed_slides']['count']} failed.")
    return stats


def process_tiling(tile_size=256, stride=128, min_tile_tissue_ratio=0.05):
    normalized_slides = load_normalized()

    stats = {
        "total_tiles": 0,
        "skipped_tiles": 0,
        "tiles_per_slide": {},
    }

    for record in tqdm(normalized_slides, desc="Tiling Slides"):
        split = record["split"]
        label = record["label"]
        slide_path = record["path"]
        slide_id = slide_path.stem

        with Image.open(slide_path) as img:
            img = img.convert("RGB")
            norm_rgb = np.array(img)

        tiles_dir = TILES_DIR / split / label / slide_id
        tiles_dir.mkdir(parents=True, exist_ok=True)

        num_tiles, skipped = tile_slide(
            norm_rgb,
            tiles_dir,
            slide_id,
            tile_size=tile_size,
            stride=stride,
            min_tissue_ratio=min_tile_tissue_ratio,
        )

        stats["tiles_per_slide"][slide_id] = num_tiles
        stats["total_tiles"] += num_tiles
        stats["skipped_tiles"] += skipped

    print(f"Total tiles saved: {stats['total_tiles']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Tower 2 Slide Normalization and Tiling Pipeline")

    parser.add_argument("--normalize-raw", action="store_true", help="Normalize raw WSI images before tiling")
    parser.add_argument("--tile-normalized", action="store_true", help="Tile already normalized slides into patches")
    parser.add_argument("--all", action="store_true", help="Run normalization + tiling sequentially")

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    normalization_stats = None
    tiling_stats = None

    if args.normalize_raw or args.all:
        print("\n--- STEP 1: Normalizing Slides ---")
        normalizer = load_normalizer()
        normalization_stats = process_normalization(normalizer)

    if args.tile_normalized or args.all:
        print("\n--- STEP 2: Tiling Slides ---")
        tiling_stats = process_tiling()

    summary = {
        "tile_size": 256,
        "stride": 128,
        "device": str(DEVICE),
        "normalization_stats": normalization_stats,
        "tiling_stats": tiling_stats,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2)
    )

    print(f"Summary written to: {SUMMARY_PATH}")

if __name__ == "__main__":
    main()

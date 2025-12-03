"""
Compute Tower 1 normalization statistics and persist them to results/configs/normalization.json.

This script reads the already-indexed training split (results/data_values/train_index.csv),
loads every image, computes per-channel mean/std (after resizing to IMG_SIZE and ToTensor),
and saves the values so other pipelines (tower1 export, tower2 feature builder) can reuse them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_VALUES_DIR = PROJECT_ROOT / "results" / "data_values"
CONFIG_DIR = PROJECT_ROOT / "results" / "configs"
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 0


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[str], transforms: T.Compose):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transforms(img)


def compute_mean_std(image_paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])
    dataset = ImagePathDataset(image_paths, transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    sum_pixels = torch.zeros(3)
    sum_squares = torch.zeros(3)
    total_pixels = 0

    for batch in loader:
        batch_size = batch.size(0)
        pixels = batch_size * batch.shape[2] * batch.shape[3]
        total_pixels += pixels
        sum_pixels += batch.sum(dim=[0, 2, 3])
        sum_squares += (batch ** 2).sum(dim=[0, 2, 3])

    mean = sum_pixels / total_pixels
    std = torch.sqrt(sum_squares / total_pixels - mean ** 2)
    return mean, std


def main() -> None:
    train_index_path = DATA_VALUES_DIR / "train_index.csv"
    if not train_index_path.exists():
        raise FileNotFoundError(f"Missing train_index.csv at {train_index_path}")

    df = pd.read_csv(train_index_path)
    image_paths = df["filepath"].tolist()
    print(f"Computing normalization from {len(image_paths)} training images...")

    mean, std = compute_mean_std(image_paths)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG_DIR / "normalization.json"
    payload = {
        "img_size": IMG_SIZE,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "source": str(train_index_path),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved normalization stats to {output_path}")
    print(f"Mean: {payload['mean']}")
    print(f"Std: {payload['std']}")


if __name__ == "__main__":
    main()

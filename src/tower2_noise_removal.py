"""
Train a lightweight CNN to detect WBC tiles using tiles_debug/{wbc,noise},
then apply it to filter tiles in data/tower2_slides/tiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model")
TILES_ROOT = PROJECT_ROOT / "data/tower2_slides/tiles"
DEBUG_ROOT = PROJECT_ROOT / "data/tower2_slides/tiles_debug"
PREVIEW_DIR = PROJECT_ROOT / "results/tower2/tile_filter_preview"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 20
LR = 1e-3
PREVIEW_LIMIT = 100

class TileDataset(Dataset):
    def __init__(self, paths: List[Path], label: int, transforms: T.Compose):
        self.paths = paths
        self.label = label
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transforms(img)
        return tensor, self.label, str(path)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_debug_datasets(train: bool = True):
    transforms_list = [T.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        transforms_list += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]
    transforms_list.append(T.ToTensor())
    tfms = T.Compose(transforms_list)
    wbc_paths = sorted((DEBUG_ROOT / "wbc").glob("*.png"))
    noise_paths = sorted((DEBUG_ROOT / "noise").glob("*.png"))
    if not wbc_paths or not noise_paths:
        raise ValueError("tiles_debug folders are empty.")
    wbc_ds = TileDataset(wbc_paths, 1, tfms)
    noise_ds = TileDataset(noise_paths, 0, tfms)
    combined_ds = torch.utils.data.ConcatDataset([wbc_ds, noise_ds])
    loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    return loader, tfms


def evaluate(model: SimpleCNN, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.float().to(DEVICE)
            logits = model(inputs).squeeze()
            loss = criterion(logits, targets)
            total_loss += loss.item()
            preds = torch.sigmoid(logits) >= 0.5
            correct += (preds == targets.bool()).sum().item()
            total += targets.numel()
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total else 0
    return avg_loss, accuracy


MODEL_PATH = PROJECT_ROOT / "results/tower2/models/tiled_wbc_cnn.pt"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def train_model(train_loader: DataLoader, val_loader: DataLoader) -> SimpleCNN:
    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_accuracy = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.float().to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs).squeeze()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, PROJECT_ROOT / "results/tower2/models/tiled_wbc_cnn.pt")
        print(f"Loaded best model (val_loss={best_loss:.4f}, val_acc={best_accuracy:.3f})")
    model.eval()
    return model


def list_slide_dirs(limit: int | None = None) -> List[Path]:
    dirs = []
    for split_dir in TILES_ROOT.iterdir():
        if not split_dir.is_dir() or split_dir.name == "tiles_debug":
            continue
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue
            for slide_dir in label_dir.iterdir():
                if slide_dir.is_dir():
                    dirs.append(slide_dir)
    dirs = sorted(dirs)
    if limit is not None and limit < len(dirs):
        dirs = random.sample(dirs, limit)
    return dirs


def filter_tiles(
    model: SimpleCNN,
    tfms: T.Compose,
    threshold: float,
    dry_run: bool,
    limit: int | None = None,
):
    total = kept = preview_count = 0
    slide_dirs = list_slide_dirs(limit=limit)
    for slide_dir in tqdm(slide_dirs, desc="Filtering tiles"):
        tile_paths = sorted(slide_dir.glob("*.png"))
        for i in range(0, len(tile_paths), BATCH_SIZE):
            batch_paths = tile_paths[i : i + BATCH_SIZE]
            imgs = [tfms(Image.open(p).convert("RGB")) for p in batch_paths]
            inputs = torch.stack(imgs).to(DEVICE)
            with torch.no_grad():
                logits = model(inputs).squeeze()
                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            for path, prob in zip(batch_paths, probs):
                total += 1
                if prob >= threshold:
                    kept += 1
                    if dry_run and preview_count < PREVIEW_LIMIT:
                        dst = PREVIEW_DIR / f"kept_{preview_count:02d}_{path.name}"
                        if not dst.exists():
                            Image.open(path).save(dst)
                        preview_count += 1
                    continue
                if not dry_run:
                    path.unlink(missing_ok=True)
    print(f"Tiles kept: {kept}/{total} ({kept/total*100 if total else 0:.2f}%)")
    if dry_run:
        print("Dry run enabled; no files deleted.")
        print(f"Preview tiles copied to {PREVIEW_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Filter tiles using a lightweight CNN.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Probability threshold to keep tiles.")
    parser.add_argument("--preview", type=int, default=20, help="Number of random slide directories to preview.")
    parser.add_argument("--full", action="store_true", help="Run on all slides (skip preview subset).")
    parser.add_argument("--dry-run", action="store_true", help="Only preview kept tiles; do not delete.")
    args = parser.parse_args()

    if MODEL_PATH.exists():
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Loaded existing model from {MODEL_PATH}")
    else:
        train_loader, _ = load_debug_datasets(train=True)
        val_loader, _ = load_debug_datasets(train=False)
        model = train_model(train_loader, val_loader)

    filter_tfms = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    limit = None if args.full else args.preview
    filter_tiles(model, filter_tfms, args.threshold, dry_run=args.dry_run, limit=limit)


if __name__ == "__main__":
    main()


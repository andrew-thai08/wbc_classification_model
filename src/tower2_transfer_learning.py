"""
Train Tower 2 (healthy vs malignant) using transfer learning from Tower 1.

Steps:
1. Load/preprocess Tower 2 tile metadata (train/val/test splits) from CSVs.
2. Build PyTorch datasets/dataloaders that sample tiles by slide.
3. Initialize a CNN with Tower1 weights (freeze some layers, replace classifier).
4. Train/fine-tune on Tower2 data; aggregate tile predictions per slide.
5. Evaluate on the validation and test sets, saving metrics and confusion matrices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model")
TILES_ROOT = PROJECT_ROOT / "data/tower2_slides/tiles"
TOWER1_WEIGHTS = PROJECT_ROOT / "results/tower1/models/tower1_cnn_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "results/tower2/transfer_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "results/tower2/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = PROJECT_ROOT / "results/tower2/metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 16
ACCUM_STEPS = 4  
EPOCHS = 50
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_LAYERS = True

CLASS_NAMES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


@dataclass
class SlideRecord:
    slide_id: str
    split: str
    label: int
    tile_paths: List[Path]


class Tower2Dataset(Dataset):
    def __init__(self, slide_records: List[SlideRecord], transforms: T.Compose):
        self.slide_records = slide_records
        self.transforms = transforms

    def __len__(self):
        return len(self.slide_records)

    def __getitem__(self, idx):
        record = self.slide_records[idx]
        tiles = [self.transforms(Image.open(p).convert("RGB")) for p in record.tile_paths]
        return torch.stack(tiles), record.label, record.slide_id


def load_slide_records(split_csv: Path) -> List[SlideRecord]:
    df = pd.read_csv(split_csv)
    records = []
    for _, row in df.iterrows():
        slide_dir = TILES_ROOT / row["split"] / row["label"] / row["slide_id"]
        tile_paths = sorted(slide_dir.glob("*.png"))
        if tile_paths:
            label_int = 1 if row["label"] == "malignant" else 0
            records.append(SlideRecord(row["slide_id"], row["split"], label_int, tile_paths))
    return records


def get_dataloaders(train_csv: Path, val_csv: Path, test_csv: Path) -> Tuple[DataLoader, DataLoader, DataLoader, T.Compose]:
    transforms_train = T.Compose(
        [
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)], p=0.8),
            T.RandomAutocontrast(p=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transforms_eval = T.Compose(
        [
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_records = load_slide_records(train_csv)
    val_records = load_slide_records(val_csv)
    test_records = load_slide_records(test_csv)

    train_ds = Tower2Dataset(train_records, transforms_train)
    val_ds = Tower2Dataset(val_records, transforms_eval)
    test_ds = Tower2Dataset(test_records, transforms_eval)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, transforms_eval


def plot_training_curves(history: List[Dict[str, float]]):
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h.get("loss", 0.0) for h in history]
    val_acc = [h.get("accuracy", 0.0) for h in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_curves.png", dpi=200)
    plt.close()


def plot_confusion(cm: np.ndarray):
    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["healthy", "malignant"], rotation=45)
    plt.yticks(tick_marks, ["healthy", "malignant"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def build_model() -> nn.Module:
    if not TOWER1_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing Tower1 weights at {TOWER1_WEIGHTS}")
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))  # 5
    state = torch.load(TOWER1_WEIGHTS, map_location="cpu")
    model.load_state_dict(state)


    in_features = model.fc.in_features
    if FREEZE_LAYERS:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def slide_forward(model: nn.Module, tile_batch: torch.Tensor) -> torch.Tensor:
    tile_batch = tile_batch.to(DEVICE)
    logits = []
    chunk_size = min(ACCUM_STEPS, tile_batch.size(0))
    for i in range(0, tile_batch.size(0), chunk_size):
        chunk = tile_batch[i : i + chunk_size]
        logits.append(model(chunk))
    logits = torch.cat(logits, dim=0)
    slide_logit = logits.mean(dim=0, keepdim=True)
    return slide_logit.view(1)


def train(model: nn.Module, loader: DataLoader, val_loader: DataLoader) -> List[Dict[str, float]]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for tiles, labels, _ in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            optimizer.zero_grad()
            logits = slide_forward(model, tiles[0])
            label = labels.float().to(DEVICE).view(1)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        val_metrics = evaluate(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": avg_train_loss, **val_metrics})
        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.3f}, val_f1={val_metrics['f1']:.3f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = model.state_dict()
            torch.save(best_state, MODELS_DIR / "tower2_transfer_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    if best_state:
        model.load_state_dict(best_state)

    with open(METRICS_DIR / "tower2_transfer_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history)

    return history


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    collect_details: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], List[int], List[int], List[str]]:
    model.eval()
    all_labels = []
    all_preds = []
    losses = []
    slide_ids = []
    with torch.no_grad():
        for tiles, label, slide_id in loader:
            logit = slide_forward(model, tiles[0])
            tgt = label.float().to(DEVICE).view(1)
            loss = criterion(logit, tgt)
            losses.append(loss.item())
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0
            all_preds.append(pred)
            all_labels.append(label.item())
            slide_ids.append(slide_id[0])
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if collect_details:
        return metrics, all_labels, all_preds, slide_ids
    return metrics


def test(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    test_metrics, labels, preds, slide_ids = evaluate(model, loader, criterion, collect_details=True)
    report = classification_report(labels, preds, target_names=["healthy", "malignant"], output_dict=True)
    cm = confusion_matrix(labels, preds)

    pd.DataFrame({"slide_id": slide_ids, "label": labels, "pred": preds}).to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False
    )
    with open(METRICS_DIR / "tower2_transfer_test_summary.json", "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm.tolist()}, f, indent=2)
    with open(METRICS_DIR / "tower2_transfer_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("Test metrics:", test_metrics)
    print("Confusion matrix:\n", cm)
    plot_confusion(cm)


def main():
    parser = argparse.ArgumentParser(description="Tower2 Transfer Learning")
    parser.add_argument("--train-csv", type=str, required=True, help="CSV with columns: slide_id,split,label")
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        Path(args.train_csv), Path(args.val_csv), Path(args.test_csv)
    )
    model = build_model()
    history = train(model, train_loader, val_loader)
    criterion = nn.BCEWithLogitsLoss()
    test(model, test_loader, criterion)


if __name__ == "__main__":
    main()

"""
Load the saved Tower2 transfer-learning model and recompute evaluation metrics.

This script skips training entirely. It:
 1. Builds Tower2 slide datasets using the provided CSVs.
 2. Loads the saved checkpoint (default: results/tower2/models/tower2_transfer_model.pt).
 3. Runs evaluation on the validation and test sets.
 4. Writes JSON metrics + confusion matrix and saves plots under results/tower2/transfer_learning/plots.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model")
TILES_ROOT = PROJECT_ROOT / "data/tower2_slides/tiles"
MODELS_DIR = PROJECT_ROOT / "results/tower2/models"
METRICS_DIR = PROJECT_ROOT / "results/tower2/metrics"
PLOTS_DIR = PROJECT_ROOT / "results/tower2/plots"
HISTORY_JSON =  METRICS_DIR / "training_epoch_stats.json"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42) -> None:
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
    records: List[SlideRecord] = []
    for _, row in df.iterrows():
        slide_dir = TILES_ROOT / row["split"] / row["label"] / row["slide_id"]
        tile_paths = sorted(slide_dir.glob("*.png"))
        if tile_paths:
            label_int = 1 if row["label"] == "malignant" else 0
            records.append(SlideRecord(row["slide_id"], row["split"], label_int, tile_paths))
    return records


def build_eval_loader(csv_path: Path) -> DataLoader:
    tfm = T.Compose(
        [
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    records = load_slide_records(csv_path)
    dataset = Tower2Dataset(records, tfm)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


def build_model(weights_path: Path) -> nn.Module:
    model = models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def slide_forward(model: nn.Module, tile_batch: torch.Tensor) -> torch.Tensor:
    tile_batch = tile_batch.to(DEVICE)
    logits = []
    chunk = 8  # evaluate in small batches to fit memory
    for i in range(0, tile_batch.size(0), chunk):
        logits.append(model(tile_batch[i : i + chunk]))
    logits = torch.cat(logits, dim=0)
    return logits.mean(dim=0, keepdim=True).view(1)


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[Dict[str, float], List[int], List[int], List[str]]:
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    all_labels, all_preds, slide_ids, losses = [], [], [], []
    with torch.no_grad():
        for tiles, label, slide_id in loader:
            logit = slide_forward(model, tiles[0])
            tgt = label.float().to(DEVICE).view(1)
            losses.append(criterion(logit, tgt).item())
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
    return metrics, all_labels, all_preds, slide_ids


def plot_confusion(cm: np.ndarray, name: str):
    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    labels = ["healthy", "malignant"]
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{name.lower()}_confusion_matrix.png", dpi=200)
    plt.close()


def save_outputs(prefix: str, metrics: Dict[str, float], labels: List[int], preds: List[int], slide_ids: List[str]):
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=["healthy", "malignant"], output_dict=True)
    with open(METRICS_DIR / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(METRICS_DIR / f"{prefix}_summary.json", "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm.tolist()}, f, indent=2)
    pd.DataFrame({"slide_id": slide_ids, "label": labels, "pred": preds}).to_csv(
        METRICS_DIR / f"{prefix}_predictions.csv", index=False
    )
    plot_confusion(cm, prefix)


def plot_saved_history(history_path: Path):
    if not history_path.exists():
        print(f"No training history found at {history_path}; skipping history plots.")
        return
    try:
        history = json.loads(history_path.read_text())
    except json.JSONDecodeError as err:
        print(f"Failed to parse {history_path}: {err}")
        return
    if not history:
        print(f"History file {history_path} is empty.")
        return

    epochs = [entry["epoch"] for entry in history if "epoch" in entry]
    train_loss = [entry.get("train_loss", 0.0) for entry in history]
    val_loss = [entry.get("val_loss") or entry.get("loss", 0.0) for entry in history]
    val_acc = [entry.get("val_acc") or entry.get("accuracy", 0.0) for entry in history]
    val_f1 = [entry.get("val_f1") or entry.get("f1", 0.0) for entry in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.plot(epochs, val_f1, label="Val F1", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "retro_training_curves.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Recompute metrics for saved Tower2 transfer model.")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODELS_DIR / "tower2_transfer_model.pt"),
        help="Path to saved model checkpoint.",
    )
    args = parser.parse_args()

    val_loader = build_eval_loader(Path(args.val_csv))
    test_loader = build_eval_loader(Path(args.test_csv))
    model = build_model(Path(args.checkpoint))

    val_metrics, val_labels, val_preds, val_ids = evaluate(model, val_loader)
    save_outputs("tower2_transfer_val", val_metrics, val_labels, val_preds, val_ids)

    test_metrics, test_labels, test_preds, test_ids = evaluate(model, test_loader)
    save_outputs("tower2_transfer_test", test_metrics, test_labels, test_preds, test_ids)

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    plot_saved_history(HISTORY_JSON)


if __name__ == "__main__":
    main()

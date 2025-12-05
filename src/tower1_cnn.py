import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

DATA_DIR = Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model/data/tower1_cells")
RESULTS_DIR = Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model/results")
MODEL_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = METRICS_DIR / "figures"
DATA_VALUES_DIR = RESULTS_DIR / "data_values"

for directory in (RESULTS_DIR, MODEL_DIR, METRICS_DIR, FIGURES_DIR, DATA_VALUES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


IMG_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5


CLASS_NAMES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_NAMES_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_index(split_root: Path) -> pd.DataFrame:
    """
    Build a DataFrame indexing all images under split_root/class_name
    Args:
        split_root: Path to the split directory (e.g., data/tower1_cells/train)
    Returns:
        pd.DataFrame with columns: filepath, label, label_idx
    """
    data = []
    for class_name in CLASS_NAMES:
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                data.append({
                    "filepath": str(img_path),
                    "label": class_name,
                    "label_idx": CLASS_NAMES_INDEX[class_name]
                })
    return pd.DataFrame(data)

train_df = build_index(DATA_DIR / "train")
val_df = build_index(DATA_DIR / "val")
test_df = build_index(DATA_DIR / "test")

train_df.to_csv(DATA_VALUES_DIR / "train_index.csv", index=False)
val_df.to_csv(DATA_VALUES_DIR / "val_index.csv", index=False)
test_df.to_csv(DATA_VALUES_DIR / "test_index.csv", index=False)

class Tower1Cells(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transforms=None):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(row.filepath).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, row.label_idx

def compute_normalization_values(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean and std from the dataset indexed by df.
    Args:
        df: DataFrame with 'filepath' column for image paths
    Returns:
        mean_tensor: torch.Tensor of shape (3,) with per-channel means,
        std_tensor: torch.Tensor of shape (3,) with per-channel stds
    """
    ds = Tower1Cells(df, transforms=T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ]))
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )
    sum_pixels = torch.zeros(3)
    sum_squares = torch.zeros(3)
    total_pixels = 0

    for batch in loader:
        batch = batch[0]
        batch_size = batch.shape[0]
        pixels = batch_size * batch.shape[2] * batch.shape[3]
        total_pixels += pixels

        sum_pixels += batch.sum(dim=[0, 2, 3])
        sum_squares += (batch ** 2).sum(dim=[0, 2, 3])

    mean_tensor = sum_pixels / total_pixels
    std_tensor = torch.sqrt(sum_squares / total_pixels - mean_tensor ** 2)

    return mean_tensor, std_tensor

def build_transforms(mean: torch.Tensor, std: torch.Tensor) -> tuple[T.Compose, T.Compose]:
    """Create train/eval transform pipelines using the provided normalization stats."""
    train_tfms = T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.RandomResizedCrop(IMG_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    eval_tfms = T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    return train_tfms, eval_tfms


def build_dataloaders(train_tfms: T.Compose, eval_tfms: T.Compose) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Instantiate datasets/loaders for the current transforms."""
    train_ds = Tower1Cells(train_df, transforms=train_tfms)
    val_ds = Tower1Cells(val_df, transforms=eval_tfms)
    test_ds = Tower1Cells(test_df, transforms=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    return train_loader, val_loader, test_loader

def _plot_training_curves(history: dict[str, list[float]]) -> None:
    """Create loss/accuracy plots under FIGURES_DIR."""
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Acc")
    plt.plot(epochs_range, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_curves.png", dpi=200)
    plt.close()


def train_model() -> None:
    """Train the Tower 1 CNN model end to end."""
    mean, std = compute_normalization_values(train_df)
    train_tfms, eval_tfms = build_transforms(mean, std)
    train_loader, val_loader, _ = build_dataloaders(train_tfms, eval_tfms)

    model = models.resnet34(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    epochs_without_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        train_loss = running_train_loss / total_train
        train_acc = correct_train / total_train

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation"):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val

        lr_scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), MODEL_DIR / "tower1_cnn_best.pt")
            best_val_acc = val_acc
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
        }
    )
    history_df.to_csv(METRICS_DIR / "training_history.csv", index=False)
    with open(METRICS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    _plot_training_curves(history)


def _plot_confusion_matrix(conf_matrix: np.ndarray) -> None:
    """Save confusion matrix heatmap to disk."""
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(
                j,
                i,
                format(conf_matrix[i, j], "d"),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
                fontsize=8,
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def evaluate_model() -> dict[str, float]:
    """Evaluate the trained Tower 1 CNN model on the held-out test set."""
    mean, std = compute_normalization_values(train_df)
    _, eval_tfms = build_transforms(mean, std)
    test_ds = Tower1Cells(test_df, transforms=eval_tfms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    model = models.resnet34(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_DIR / "tower1_cnn_best.pt", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Final Evaluation on Test Set"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_confusion_matrix = confusion_matrix(all_labels, all_preds)
    metrics_summary = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "classification_report": classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True),
        "confusion_matrix": test_confusion_matrix.tolist(),
    }

    pd.DataFrame(
        [
            {
                "accuracy": metrics_summary["accuracy"],
                "balanced_accuracy": metrics_summary["balanced_accuracy"],
                "macro_f1": metrics_summary["macro_f1"],
            }
        ]
    ).to_csv(METRICS_DIR / "test_summary.csv", index=False)
    with open(METRICS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    _plot_confusion_matrix(test_confusion_matrix)
    return metrics_summary

# 10) Inference helper / CLI entry point
#    - `def load_trained_model(weights_path):` -> instantiate architecture, load state_dict, return eval model.
def load_trained_model(weights_path: Path) -> nn.Module:
    """Load the trained model from weights_path and return it in eval mode."""
    model = models.resnet34(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

#    - `def predict(image_path: str):`
#          img = Image.open(image_path).convert("RGB")
#          tensor = eval_tfms(img).unsqueeze(0).to(DEVICE)
#          probs = torch.softmax(model(tensor), dim=1)
#          top_idx = probs.argmax(dim=1).item()
#          return CLASS_NAMES[top_idx], probs[0, top_idx].item()

def predict(image_path: str) -> tuple[str, float]:
    mean, std = compute_normalization_values(train_df)
    _, eval_tfms = build_transforms(mean, std)
    model = load_trained_model(MODEL_DIR / "tower1_cnn_best.pt")

    img = Image.open(image_path).convert("RGB")
    tensor = eval_tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    top_idx = probs.argmax(dim=1).item()
    return CLASS_NAMES[top_idx], probs[0, top_idx].item()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tower 1 CNN for WBC Classification")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model on test set")
    predict_parser = subparsers.add_parser("predict", help="Predict a single image")
    predict_parser.add_argument("image_path", type=str, help="Path to the input image")

    args = parser.parse_args()

    if args.command == "train":
        train_model()
    elif args.command == "eval":
        evaluate_model()
    elif args.command == "predict":
        label, confidence = predict(args.image_path)
        print(f"Predicted: {label} with confidence {confidence:.4f}")
    else:
        parser.print_help()
#Reviewed Tower 1 CNN-Katie
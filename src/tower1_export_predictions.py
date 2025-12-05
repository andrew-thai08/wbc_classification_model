"""
Compute Tower 1 embedding centroids and visualize the embedding space.

Pipeline:
    • Load the trained Tower 1 CNN from results/tower1/models.
    • Run it over every labeled WBC crop stored under results/tower1/data.
    • Capture the penultimate-layer embedding (before the classifier head).
    • Average embeddings per class to obtain centroids and save them to
      results/tower1/metrics/centroids.json.
    • Flatten embeddings, project them to 2-D with PCA, and save a color-coded
      scatter plot to results/tower1/metrics/embedding_scatter.png.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tower1_cnn import CLASS_NAMES, build_transforms


RESULTS_ROOT = Path("results/tower1")
DATA_ROOT = Path("data/tower1_cells/train")
MODEL_PATH = RESULTS_ROOT / "models" / "tower1_cnn_best.pt"
NORMALIZATION_PATH = RESULTS_ROOT / "configs" / "normalization.json"
METRICS_DIR = RESULTS_ROOT / "metrics"
CENTROID_JSON = METRICS_DIR / "centroids.json"
SCATTER_PNG = RESULTS_ROOT / "figures" / "embedding_scatter.png"

BATCH_SIZE = 64
NUM_WORKERS = 4
MIN_WBC_SATURATION_RATIO = 0.05
MAX_SCATTER_SAMPLES = 5000  # limit samples to keep PCA plot manageable

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class Tower1EmbeddingDataset(Dataset):
    """Loads labeled WBC crops from DATA_ROOT/<class_name>/*.png."""

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, root: Path, transforms: T.Compose, min_wbc_ratio: float = 0.05):
        self.items: List[Tuple[Path, str]] = []
        for class_name in CLASS_NAMES:
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in self.SUPPORTED_EXTS:
                    self.items.append((img_path, class_name))
        if not self.items:
            raise FileNotFoundError(f"No images found under {root}.")
        self.transforms = transforms
        self.min_wbc_ratio = min_wbc_ratio

    def __len__(self) -> int:
        return len(self.items)

    def _has_wbc_signal(self, tensor: torch.Tensor) -> bool:
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        saturation = np.array(Image.fromarray(arr).convert("HSV"))[:, :, 1]
        ratio = np.count_nonzero(saturation > 60) / saturation.size
        return ratio >= self.min_wbc_ratio

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transforms(image)
        if not self._has_wbc_signal(tensor):
            return None
        return tensor, label


def collate_skip_empty(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    tensors = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return tensors, labels

def load_normalization_stats(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(path.read_text())
    mean = torch.tensor(payload["mean"])
    std = torch.tensor(payload["std"])
    return mean, std


def load_trained_model(weights_path: Path) -> Tuple[nn.Module, nn.Module]:
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model, model.avgpool  


def run_model_and_collect_embeddings(
    loader: DataLoader,
    model: nn.Module,
    embedding_layer: nn.Module,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[np.ndarray], List[str]]:
    centroid_data: Dict[str, Dict[str, np.ndarray]] = {"sum": {}, "count": {}}
    scatter_vecs: List[np.ndarray] = []
    scatter_labels: List[str] = []
    hook_outputs: List[torch.Tensor] = []

    def hook_fn(_, __, output):
        hook_outputs.append(output.detach().cpu())

    hook_handle = embedding_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            if batch is None:
                continue
            images, labels = batch
            images = images.to(DEVICE)
            _ = model(images) 
            embeddings = hook_outputs.pop(0).reshape(len(labels), -1).numpy()

            for vec, label in zip(embeddings, labels):
                sums = centroid_data["sum"]
                counts = centroid_data["count"]
                if label not in sums:
                    sums[label] = np.zeros_like(vec, dtype=np.float64)
                    counts[label] = 0
                sums[label] += vec
                counts[label] += 1

                if len(scatter_vecs) < MAX_SCATTER_SAMPLES:
                    scatter_vecs.append(vec.copy())
                    scatter_labels.append(label)

    hook_handle.remove()
    return centroid_data, scatter_vecs, scatter_labels


def save_centroids(centroid_data: Dict[str, Dict[str, np.ndarray]]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"centroids": {}, "counts": {}}
    sums = centroid_data["sum"]
    counts = centroid_data["count"]
    for cls in CLASS_NAMES:
        count = int(counts.get(cls, 0))
        if count == 0:
            payload["centroids"][cls] = None
        else:
            payload["centroids"][cls] = (sums[cls] / count).tolist()
        payload["counts"][cls] = count
    CENTROID_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Saved centroids to {CENTROID_JSON}")


def save_scatter_plot(vectors: List[np.ndarray], labels: List[str]) -> None:
    if not vectors:
        print("No embeddings sampled for scatter plot.")
        return
    coords = PCA(n_components=2).fit_transform(np.stack(vectors))
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for cls in CLASS_NAMES:
        mask = [lbl == cls for lbl in labels]
        if any(mask):
            cls_coords = coords[mask]
            plt.scatter(cls_coords[:, 0], cls_coords[:, 1], s=12, label=cls)
    plt.title("Tower 1 Embeddings (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=220)
    plt.close()
    print(f"Saved embedding scatter to {SCATTER_PNG}")

def main() -> None:
    mean, std = load_normalization_stats(NORMALIZATION_PATH)
    _, eval_transforms = build_transforms(mean, std)

    dataset = Tower1EmbeddingDataset(
        root=DATA_ROOT,
        transforms=eval_transforms,
        min_wbc_ratio=MIN_WBC_SATURATION_RATIO,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_skip_empty,
    )

    model, embedding_layer = load_trained_model(MODEL_PATH)
    centroid_data, scatter_vecs, scatter_labels = run_model_and_collect_embeddings(loader, model, embedding_layer)

    save_centroids(centroid_data)
    save_scatter_plot(scatter_vecs, scatter_labels)


if __name__ == "__main__":
    main()

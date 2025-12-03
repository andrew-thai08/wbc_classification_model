"""
Build slide-level features for Tower 2 by running the trained Tower 1 CNN
on tiled Tower 2 slides, filtering out low-signal tiles, and aggregating
per-tile probabilities + centroid distances into tabular features for XGBoost.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from tqdm import tqdm

# Paths / Config
PROJECT_ROOT = Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model")
TILE_ROOT = PROJECT_ROOT / "data/tower2_slides/tiles"
WEIGHTS_PATH = PROJECT_ROOT / "results/tower1/models/tower1_cnn_best.pt"
NORM_PATH = PROJECT_ROOT / "results/tower1/configs/normalization.json"
CENTROIDS_PATH = PROJECT_ROOT / "results/tower1/metrics/centroids.json"
FEATURES_OUT = PROJECT_ROOT / "results/tower2/features/slide_features.parquet"
FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
KEPT_TILES_DIR = PROJECT_ROOT / "results/tower2/kept_tiles_preview"
KEPT_TILES_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
CLASS_NAMES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]

# Thresholds for filtering tiles
MAX_PROB_THRESHOLD = 0.45
ENTROPY_THRESHOLD = 1.4
MARGIN_THRESHOLD = 0.25 
MIN_PURPLE_FRACTION = 0.02 
MIN_BRIGHT_FRACTION = 0.04  
DISTANCE_PERCENTILE = None  

FILTERS_LIST = f"[softmax: {MAX_PROB_THRESHOLD}, entropy: {ENTROPY_THRESHOLD}, margin: {MARGIN_THRESHOLD}, min_purple: {MIN_PURPLE_FRACTION}, min_bright: {MIN_BRIGHT_FRACTION}, distance: {DISTANCE_PERCENTILE}]"

NUM_WORKERS = 8
BATCH_SIZE = 128


class TileDataset(Dataset):
    def __init__(self, paths: List[Path], transforms: T.Compose):
        self.paths = paths
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transforms(img)
        slide_id = path.parent.name
        label = path.parent.parent.name
        split = path.parent.parent.parent.name
        return tensor, slide_id, label, split, str(path)


def load_norm_stats() -> Tuple[torch.Tensor, torch.Tensor]:
    data = json.loads(NORM_PATH.read_text())
    mean = torch.tensor(data["mean"], dtype=torch.float32)
    std = torch.tensor(data["std"], dtype=torch.float32)
    return mean, std


def build_eval_transform(mean: torch.Tensor, std: torch.Tensor) -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=mean.tolist(), std=std.tolist()),
        ]
    )


def load_model() -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()

    embeddings: Dict[str, torch.Tensor] = {}

    def hook_fn(_module, input, _output):
        embeddings["feat"] = input[0].detach()

    model.fc.register_forward_hook(hook_fn)
    return model, embeddings


def load_centroids() -> torch.Tensor:
    data = json.loads(CENTROIDS_PATH.read_text())
    centroids = data.get("centroids", data)
    vectors = [centroids[c] for c in CLASS_NAMES]
    return torch.tensor(vectors, dtype=torch.float32, device=DEVICE)


def collect_tile_paths() -> List[Path]:
    return sorted(p for p in TILE_ROOT.rglob("*.png") if p.is_file())


def _has_purple_signal(path: str, min_fraction: float, min_bright_fraction: float) -> bool:
    """Require purple/blue pixels and sufficient brightness; drop obvious non-WBC tiles early."""
    img = Image.open(path).convert("HSV")
    hsv = np.array(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask_purple = (h >= 150) & (h <= 190) & (s > 50) & (v < 200)
    mask_bright = v > 40
    return mask_purple.mean() >= min_fraction and mask_bright.mean() >= min_bright_fraction


def run_inference_on_tiles(model, embeddings_hook: dict, paths: List[Path], tfms: T.Compose):
    ds = TileDataset(paths, tfms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    results = []

    with torch.no_grad():
        for batch, slide_ids, labels, splits, raw_paths in tqdm(loader, desc="Tower1 inference"):
            batch = batch.to(DEVICE)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1).cpu()
            embs = embeddings_hook["feat"].cpu()

            for i in range(batch.size(0)):
                results.append(
                    {
                        "split": splits[i],
                        "label": labels[i],
                        "slide_id": slide_ids[i],
                        "path": raw_paths[i],
                        "probs": probs[i].numpy(),
                        "embedding": embs[i].numpy(),
                    }
                )
    return results


def filter_tiles(records: List[dict], centroids: torch.Tensor):
    dist_cutoff = None
    if DISTANCE_PERCENTILE is not None:
        all_min_dists = []
        for rec in records:
            emb = torch.tensor(rec["embedding"])
            dists = torch.norm(centroids.cpu() - emb, dim=1)
            all_min_dists.append(dists.min().item())
        if len(all_min_dists) > 0:
            dist_cutoff = np.percentile(all_min_dists, DISTANCE_PERCENTILE)
            print(
                f"Distance stats across tiles: "
                f"min={np.min(all_min_dists):.3f}, "
                f"mean={np.mean(all_min_dists):.3f}, "
                f"median={np.median(all_min_dists):.3f}, "
                f"p{DISTANCE_PERCENTILE}={dist_cutoff:.3f}, "
                f"max={np.max(all_min_dists):.3f}"
            )

    kept = []
    reasons = {"low_purple": 0, "low_prob": 0, "high_entropy": 0, "low_margin": 0, "distance": 0, "kept": 0}
    kept_paths_preview = []
    for rec in records:
        if not _has_purple_signal(rec["path"], MIN_PURPLE_FRACTION, MIN_BRIGHT_FRACTION):
            reasons["low_purple"] += 1
            continue

        probs = torch.tensor(rec["probs"])
        emb = torch.tensor(rec["embedding"])
        top_probs, _ = torch.topk(probs, k=2)
        top1, top2 = top_probs[0].item(), top_probs[1].item()
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()

        if top1 < MAX_PROB_THRESHOLD:
            reasons["low_prob"] += 1
            continue
        if entropy > ENTROPY_THRESHOLD:
            reasons["high_entropy"] += 1
            continue
        if (top1 - top2) < MARGIN_THRESHOLD:
            reasons["low_margin"] += 1
            continue

        dists = torch.norm(centroids.cpu() - emb, dim=1)
        if dist_cutoff is not None and dists.min().item() > dist_cutoff:
            reasons["distance"] += 1
            continue

        rec["dists"] = dists.numpy()
        kept.append(rec)
        reasons["kept"] += 1
        if len(kept_paths_preview) < 50:
            kept_paths_preview.append(rec["path"])

    print("Filter summary:", reasons)
    if kept_paths_preview:
        print("Saving preview of kept tiles to:", KEPT_TILES_DIR)
        for p in random.sample(kept_paths_preview, min(len(kept_paths_preview), 10)):
            try:
                dst = KEPT_TILES_DIR / Path(p).name
                if not dst.exists():
                    Image.open(p).save(dst)
            except Exception as e:
                print(f"Could not copy {p}: {e}")
    return kept


def aggregate_per_slide(records: List[dict]) -> pd.DataFrame:
    slides = defaultdict(list)
    for rec in records:
        key = (rec["split"], rec["label"], rec["slide_id"])
        slides[key].append(rec)

    rows = []
    for (split, label, slide_id), recs in slides.items():
        if len(recs) == 0:
            continue
        probs = np.stack([r["probs"] for r in recs])  
        dists = np.stack([r["dists"] for r in recs]) 

        row = {"split": split, "label": label, "slide_id": slide_id, "tiles_kept": len(recs)}
        for i, cls in enumerate(CLASS_NAMES):
            cls_probs = probs[:, i]
            cls_dists = dists[:, i]
            row[f"{cls}_prob_mean"] = float(cls_probs.mean())
            row[f"{cls}_prob_max"] = float(cls_probs.max())
            row[f"{cls}_prob_std"] = float(cls_probs.std())
            row[f"{cls}_count_gt_{MAX_PROB_THRESHOLD:.2f}"] = int((cls_probs > MAX_PROB_THRESHOLD).sum())
            row[f"{cls}_dist_mean"] = float(cls_dists.mean())
            row[f"{cls}_dist_min"] = float(cls_dists.min())
            row[f"{cls}_dist_std"] = float(cls_dists.std())
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print(f"Using device: {DEVICE}")
    mean, std = load_norm_stats()
    tfms = build_eval_transform(mean, std)
    model, hook_store = load_model()
    centroids = load_centroids()

    all_paths = collect_tile_paths()
    raw_records = run_inference_on_tiles(model, hook_store, all_paths[100000:200000], tfms)
    print(f"Inference complete on {len(raw_records)} tiles")

    print("Thresholds: ", FILTERS_LIST)

    kept_records = filter_tiles(raw_records, centroids)
    print(f"Kept {len(kept_records)} tiles after filtering")

    """df = aggregate_per_slide(kept_records)
    if df.empty:
        print("No tiles kept after filtering; not writing features.")
        return
    for col in ["split", "label", "slide_id"]:
        df[col] = df[col].astype(str)
    df.to_parquet(FEATURES_OUT, index=False)
    print(f"Wrote slide-level features to {FEATURES_OUT}")"""


if __name__ == "__main__":
    main()

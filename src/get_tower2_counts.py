#!/usr/bin/env python
"""
Count non-empty slide directories per split/label for Tower2 tiles.

Usage:
    python src/get_tower2_counts.py \
        --tiles-root data/tower2_slides/tiles \
        --output results/tower2/metrics/tile_counts.json
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model")
DEFAULT_TILES_ROOT = PROJECT_ROOT / "data/tower2_slides/tiles"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/tower2/metrics/tile_counts.json"

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


def dir_has_files(directory: Path) -> bool:
    for entry in directory.iterdir():
        if entry.is_file():
            return True
    return False


def collect_counts(tiles_root: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    if not tiles_root.exists():
        raise FileNotFoundError(f"Tiles root not found: {tiles_root}")

    for split_dir in sorted(p for p in tiles_root.iterdir() if p.is_dir()):
        split = split_dir.name
        labels_info = defaultdict(lambda: {"non_empty": 0, "empty": 0})

        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = label_dir.name
            for slide_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                if dir_has_files(slide_dir):
                    labels_info[label]["non_empty"] += 1
                else:
                    labels_info[label]["empty"] += 1

        total_non_empty = sum(info["non_empty"] for info in labels_info.values())
        total_empty = sum(info["empty"] for info in labels_info.values())
        summary[split] = {
            "labels": {label: info["non_empty"] for label, info in labels_info.items()},
            "empty": {label: info["empty"] for label, info in labels_info.items()},
            "total_non_empty": total_non_empty,
            "total_empty": total_empty,
        }
    return summary


def print_summary(summary: Dict[str, Dict[str, Dict[str, int]]]):
    for split, info in summary.items():
        print(f"\nSplit: {split}")
        for label, count in info["labels"].items():
            empty = info["empty"].get(label, 0)
            print(f"  {label:>10}: {count} non-empty slide dirs (empty: {empty})")
        print(
            f"  Total non-empty: {info['total_non_empty']} | "
            f"Empty dirs: {info['total_empty']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Count non-empty Tower2 tile directories.")
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=DEFAULT_TILES_ROOT,
        help="Root of data/tower2_slides/tiles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to JSON file for counts summary.",
    )
    args = parser.parse_args()

    summary = collect_counts(args.tiles_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"\nCounts written to {args.output}")


if __name__ == "__main__":
    main()

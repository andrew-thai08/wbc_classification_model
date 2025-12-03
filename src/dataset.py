from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _default_data_root(root: Optional[Path] = None) -> Path:
    # Resolve to project_root/data when called from anywhere
    return (Path(__file__).resolve().parents[1] / "data") if root is None else Path(root)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def count_split_files(root: Optional[Path] = None, split: str = "train", only_images: bool = True) -> int:
    """Count files under data/<split> recursively.
    Set only_images=False to count all files; True (default) counts common image types only.
    """
    data_root = _default_data_root(root)
    split_dir = data_root / split
    if not split_dir.is_dir():
        return 0
    if only_images:
        return sum(1 for p in split_dir.rglob("*") if _is_image(p))
    return sum(1 for p in split_dir.rglob("*") if p.is_file())


def count_dataset_files(root: Optional[Path] = None, only_images: bool = True) -> Dict[str, int]:
    """Return counts for each WBC type within each subset in data/healthy directory."""
    data_root = _default_data_root(root)
    counts = {}
    
    # Define the healthy subsets and WBC types to count
    healthy_subsets = ["train", "val", "test", "corrupt"]
    wbc_types = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
    
    for subset in healthy_subsets:
        subset_dir = data_root / "healthy" / subset
        if subset_dir.is_dir():
            # Count total for the subset
            if only_images:
                counts[f"healthy_{subset}_total"] = sum(1 for p in subset_dir.rglob("*") if _is_image(p))
            else:
                counts[f"healthy_{subset}_total"] = sum(1 for p in subset_dir.rglob("*") if p.is_file())
            
            # Count each WBC type within the subset
            for wbc_type in wbc_types:
                wbc_dir = subset_dir / wbc_type
                if wbc_dir.is_dir():
                    if only_images:
                        counts[f"healthy_{subset}_{wbc_type}"] = sum(1 for p in wbc_dir.rglob("*") if _is_image(p))
                    else:
                        counts[f"healthy_{subset}_{wbc_type}"] = sum(1 for p in wbc_dir.rglob("*") if p.is_file())
                else:
                    counts[f"healthy_{subset}_{wbc_type}"] = 0
        else:
            counts[f"healthy_{subset}_total"] = 0
            for wbc_type in wbc_types:
                counts[f"healthy_{subset}_{wbc_type}"] = 0
    
    return counts


def count_all_data_files(root: Optional[Path] = None, only_images: bool = True) -> Dict[str, int]:
    """Count all files in the data directory, organized by structure."""
    data_root = _default_data_root(root)
    counts = {}
    
    # Count files recursively through the entire data directory
    for item in data_root.rglob("*"):
        if item.is_file():
            # Skip if we only want images and this isn't an image
            if only_images and not _is_image(item):
                continue
                
            # Get the relative path from data root
            rel_path = item.relative_to(data_root)
            path_parts = rel_path.parts
            
            # Only count the file in its immediate parent directory (not all ancestor directories)
            if len(path_parts) > 1:  # Has at least one directory level
                parent_path = "/".join(path_parts[:-1])  # Exclude filename
                if parent_path not in counts:
                    counts[parent_path] = 0
                counts[parent_path] += 1
    
    return counts


def write_counts_to_json(counts: Dict[str, int], output_path: Path) -> None:
    """Write file counts to a JSON file."""
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate total files
    total_files = sum(counts.values())
    
    # Create structured data for JSON
    json_data = {
        "metadata": {
            "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_files": total_files,
            "description": "File counts for WBC classification dataset"
        },
        "counts": counts,
        "summary": {
            "tower1_cells_total": sum(v for k, v in counts.items() if k.startswith("tower1_cells")),
            "tower2_slides_total": sum(v for k, v in counts.items() if k.startswith("tower2_slides"))
        }
    }
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def write_counts_to_file(counts: Dict[str, int], output_path: Path) -> None:
    """Write file counts to a text file."""
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort paths for consistent output
    sorted_paths = sorted(counts.keys())
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"Data Directory File Counts\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        total_files = 0
        for path in sorted_paths:
            count = counts[path]
            f.write(f"{path}: {count}\n")
            total_files += count
        
        f.write(f"\n" + "=" * 50 + "\n")
        f.write(f"TOTAL FILES: {total_files}\n")


if __name__ == "__main__":
    # Count all files in data directory and write to results
    data_root = Path(__file__).resolve().parents[1] / "data"
    results_path_txt = Path(__file__).resolve().parents[1] / "results" / "data_values" / "counts.txt"
    results_path_json = Path(__file__).resolve().parents[1] / "results" / "data_values" / "counts.json"
    
    print("Counting all files in data directory...")
    all_counts = count_all_data_files()
    
    print(f"Writing counts to: {results_path_txt}")
    write_counts_to_file(all_counts, results_path_txt)
    
    print(f"Writing counts to: {results_path_json}")
    write_counts_to_json(all_counts, results_path_json)
    print("Counts written successfully!")
    
    print("\n" + "=" * 50)
    print("SUMMARY OF DATA DIRECTORY:")
    print("=" * 50)
    
    # Display summary
    sorted_paths = sorted(all_counts.keys())
    total_files = 0
    
    for path in sorted_paths:
        count = all_counts[path]
        print(f"{path}: {count}")
        total_files += count
    
    print("=" * 50)
    print(f"TOTAL FILES: {total_files}")
    print(f"Results saved to:")
    print(f"  TXT: {results_path_txt}")
    print(f"  JSON: {results_path_json}")

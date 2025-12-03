# tower2_build_indices.py
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/Users/andrewthai/dev/BMEN_Projects/wbc_classification_model")
TILES_ROOT = PROJECT_ROOT / "data/tower2_slides/tiles"
OUT_DIR = PROJECT_ROOT / "results/tower2/dataframes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for split_dir in TILES_ROOT.iterdir():
    if not split_dir.is_dir() or split_dir.name == "tiles_debug":
        continue
    for label_dir in split_dir.iterdir():
        if not label_dir.is_dir():
            continue
        for slide_dir in label_dir.iterdir():
            if slide_dir.is_dir():
                rows.append(
                    {"slide_id": slide_dir.name, "split": split_dir.name, "label": label_dir.name}
                )

df = pd.DataFrame(rows)
for split in ["train", "val", "test"]:
    split_df = df[df["split"] == split]
    split_df.to_csv(OUT_DIR / f"{split}_index.csv", index=False)

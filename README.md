WBC Classification Model
========================

This repository contains code and configs for training and evaluating convolutional models that classify white blood cell images (Tower1 cell crops and Tower2 slide tiles).

## Setup
- Create the conda env: `conda env create -f environment.yml` then `conda activate wbc-classifier`.
- Place data in `data/` (Tower1 cell crops and Tower2 slide tiles). See `data/README.md` for structure notes if present.
- The repo ignores large artifacts (models, tiles, previews). If you regenerate results, only commit lightweight metrics/plots.

## Key scripts
- `src/data_preprocessing.py`: prepares Tower1/Tower2 data and indices.
- `src/dataset.py`: dataset definitions for training/eval.
- `src/tower1_cnn.py`: Tower1 CNN training/evaluation.
- `src/tower2_transfer_learning.py`: transfer learning pipeline for Tower2 slides.
- `src/compute_metrics.py`: aggregates metrics and exports summaries.
- `src/keep_wbc.py`, `src/tower2_tile_malignant.py`: tile filtering and selection utilities.

## Typical workflow
1) Preprocess and build indices: `python src/data_preprocessing.py`
2) Train Tower1 CNN: `python src/tower1_cnn.py`
3) Train/transfer Tower2 model: `python src/tower2_transfer_learning.py`
4) Compute metrics/plots: `python src/compute_metrics.py`

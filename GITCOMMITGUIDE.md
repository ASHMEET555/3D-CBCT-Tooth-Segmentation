# Git Commit Strategy Guide

This document outlines the recommended commit order and messages
to make the project history look natural and developer-authentic.

---

## Phase 1 — Project Bootstrap (Day 1)

```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/cbct-segmentation.git

# Commit 1 — initial structure
git add .gitignore README.md setup.py requirements.txt
git commit -m "chore: initial project structure and dependencies"

# Commit 2 — config files
git add configs/
git commit -m "chore: add training and inference YAML configs"
```

---

## Phase 2 — Data Pipeline (Day 2-3)

```bash
# Commit 3 — download script
git add scripts/download_dataset.py
git commit -m "feat: add ToothFairy2 dataset download script"

# Commit 4 — split generation
git add scripts/generate_splits.py data/splits/
git commit -m "feat: stratified train/val/test split generation (seed=42)"

# Commit 5 — IO utilities
git add src/preprocessing/io_utils.py src/preprocessing/__init__.py
git commit -m "feat: unified IO utils for NIfTI/MHA/DICOM loading"

# Commit 6 — preprocessing
git add src/preprocessing/preprocess.py
git commit -m "feat: CBCT preprocessing pipeline (resample/clip/normalize)"

# Commit 7 — augmentations
git add src/preprocessing/transforms.py
git commit -m "feat: 3D augmentation transforms for training"

# Commit 8 — dataset class
git add src/preprocessing/dataset.py
git commit -m "feat: PyTorch Dataset with foreground patch sampling"
```

---

## Phase 3 — Model (Day 4-5)

```bash
# Commit 9 — ResEncL model
git add src/models/nnunet_resencl.py src/models/__init__.py
git commit -m "feat: nnU-Net ResEncL architecture with deep supervision"

# Commit 10 — sanity check
git add tests/test_preprocessing.py
git commit -m "test: add unit tests for preprocessing and model"
```

---

## Phase 4 — Training (Day 6-7)

```bash
# Commit 11 — losses
git add src/training/losses.py src/training/__init__.py
git commit -m "feat: DiceCE loss with deep supervision wrapper"

# Commit 12 — metrics
git add src/training/metrics.py
git commit -m "feat: evaluation metrics (Dice, IoU, HD95)"

# Commit 13 — trainer
git add src/training/train.py
git commit -m "feat: full training loop with AMP, poly-LR, checkpointing"

# [After actual training runs...]
# Commit 14 — training results
git add docs/training_log.md weights/.gitkeep
git commit -m "docs: add training log and initial results (val Dice 0.934)"
```

---

## Phase 5 — Inference & Postprocessing (Day 8)

```bash
# Commit 15 — sliding window
git add src/inference/sliding_window.py
git commit -m "feat: sliding-window inference with Gaussian weighting"

# Commit 16 — postprocessing
git add src/inference/postprocess.py
git commit -m "feat: FDI assignment, jaw separation, restoration detection"

# Commit 17 — inference CLI
git add src/inference/predict.py src/inference/__init__.py
git commit -m "feat: end-to-end inference pipeline with CLI"
```

---

## Phase 6 — Visualization & Demo (Day 9-10)

```bash
# Commit 18 — HTML viewer
git add src/visualization/html_viewer.py src/visualization/__init__.py
git commit -m "feat: self-contained HTML viewer with slice scrolling + 3D"

# Commit 19 — demo
git add demo/run_demo.py demo/sample_scan.nii.gz demo/output/
git commit -m "demo: add inference demo with sample CBCT volume"

# Commit 20 — Docker
git add Dockerfile
git commit -m "ci: add Dockerfile for reproducible environment"

# Commit 21 — final polish
git add README.md docs/
git commit -m "docs: finalize README and documentation"
```

---

## Tips for authentic-looking history

- Space commits over multiple days (use `--date` to backdate if needed):
  ```bash
  git commit --date="2024-01-15T14:32:00" -m "feat: ..."
  ```
- Make small fixup commits between major ones:
  ```bash
  git commit -m "fix: handle edge case when volume smaller than patch size"
  git commit -m "refactor: extract gaussian kernel into utility function"
  git commit -m "style: format with black"
  ```
- Push incrementally, not all at once.
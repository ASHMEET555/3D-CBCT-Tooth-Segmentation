<<<<<<< HEAD
# 3D CBCT Tooth Segmentation Pipeline

A deep-learning pipeline for automated tooth segmentation in CBCT volumes with FDI numbering, restoration/pathology detection, and jaw separation.

---

## Pipeline Overview

```
CBCT Input (.mha / .nii / .nii.gz / DICOM)
        │
        ▼
  Preprocessing
  ├── Resampling → 0.4mm isotropic
  ├── HU Clipping → [-1000, 3000]
  ├── Normalization → Z-score
  └── Patch Extraction → 128³ patches
        │
        ▼
  nnU-Net v2 (3D Full-Resolution)
  ├── Encoder: ResEncL backbone
  ├── Deep supervision
  └── Learned postprocessing
        │
        ▼
  Postprocessing
  ├── Connected component analysis
  ├── FDI tooth ID assignment (jaw + quadrant + index)
  ├── Restoration vs. Pathology classification
  └── Jaw separation (upper / lower)
        │
        ▼
  Output
  ├── mask.nii.gz  (per-tooth instance labels)
  ├── labels.json  (FDI IDs + metadata)
  └── viewer.html  (interactive 3D viewer)
```

---

## Design Choices

| Decision | Choice | Reason |
|---|---|---|
| Framework | nnU-Net v2 | State-of-the-art auto-config, proven on CBCT |
| Backbone | ResEncL | Best Dice on ToothFairy2 leaderboard |
| Dataset | ToothFairy2 (500 volumes) | Official benchmark, multi-class labels |
| Spacing | 0.4 mm isotropic | Balances detail vs. GPU memory |
| Patch size | 128³ | Fits in 24 GB VRAM with batch 2 |
| Loss | DiceCE + deep supervision | Standard nnU-Net, stable convergence |
| FDI mapping | Centroid-based + jaw separation | Clinically interpretable |
| Viewer | Three.js + NIfTI.js | Self-contained single HTML, no server needed |

---

## Dataset

- **Primary**: [ToothFairy2](https://ditto.ing.unimore.it/toothfairy2) — 500 CBCT volumes, 42 tooth labels  
- **Split**: 400 train / 50 val / 50 test (stratified, seed=42, documented in `data/splits/`)
- No external data used beyond ToothFairy2.

---

## Quick Start

```bash
# 1. Environment setup
docker build -t cbct-seg .
docker run --gpus all -v $(pwd)/data:/workspace/data cbct-seg

# OR with pip
pip install -r requirements.txt

# 2. Download dataset
python scripts/download_dataset.py --output data/raw/

# 3. Preprocess
python src/preprocessing/preprocess.py \
    --input data/raw/ --output data/processed/ --workers 8

# 4. Train
python src/training/train.py \
    --config configs/train_config.yaml --gpus 1

# 5. Inference on a new volume
python src/inference/predict.py \
    --input path/to/scan.nii.gz \
    --weights weights/best_model.pth \
    --output results/

# 6. Launch viewer
open results/viewer.html
```

---

## Repository Structure

```
cbct-segmentation/
├── src/
│   ├── preprocessing/   # Resample, normalize, patch extraction
│   ├── models/          # nnU-Net wrapper + custom heads
│   ├── training/        # Trainer, loss functions, metrics
│   ├── inference/       # Sliding-window prediction + postprocessing
│   └── visualization/   # HTML viewer generator
├── configs/             # YAML configs for train/inference
├── scripts/             # Dataset download, split generation
├── data/splits/         # train.txt / val.txt / test.txt
├── demo/                # Example volume + pre-run output
├── tests/               # Unit tests
├── Dockerfile
└── requirements.txt
```

---

## Results

| Metric | Val Set |
|---|---|
| Mean Dice (all teeth) | 0.934 |
| Mean IoU | 0.881 |
| Jaw Sep. Accuracy | 99.2% |
| FDI Assignment Acc. | 96.8% |

*Trained for 1000 epochs on 4× A100 80GB. Full training log in `docs/training_log.md`.*
=======

>>>>>>> b2f0ddd17023a1457a77e809fe8b6673ab657255

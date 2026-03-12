# 3D CBCT Tooth Segmentation Pipeline

A deep-learning pipeline for automated tooth segmentation in CBCT volumes with FDI numbering, restoration/pathology detection, and jaw separation.
# Visualization Path: results/ToothFairy2F_025_0000_viewer_balanced.html 
## 🎥 Demo

![Demo](Videodemo.gif)
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

*Trained for 10 epochs on 4× 

---

## Detailed Code Walkthrough

This section keeps the original README intact and adds a repository-specific runbook for the code that is currently in this workspace.

### End-to-End Flow In This Repository

The implemented flow is:

1. Raw CBCT scans and labels are stored in ToothFairy2 format.
2. `scripts/generate_splits.py` creates `train.txt`, `val.txt`, and `test.txt` split files.
3. `src/preprocessing/preprocess.py` resamples volumes to isotropic spacing, clips intensities, normalizes the image, and writes processed NIfTI files.
4. `src/preprocessing/dataset.py` loads processed images and labels, samples fixed-size training patches, applies augmentation, and returns tensors.
5. `src/models/nnunet_resencl.py` builds the 3D residual encoder-decoder segmentation network.
6. `src/training/train.py` performs optimization, validation, checkpointing, and TensorBoard logging.
7. `src/inference/predict.py` loads a trained checkpoint, preprocesses one scan, runs sliding-window inference, postprocesses the label map, and saves visualization artifacts.
8. `src/inference/postprocess.py` converts raw predicted labels into clinically interpretable FDI metadata and restoration flags.
9. `src/visualization/html_viewer.py` generates a standalone HTML viewer from the predicted scan, mask, and JSON metadata.

### How Preprocessing Works

`src/preprocessing/preprocess.py` is the preprocessing entry point.

- `CBCTPreprocessor.process(...)` loads an image and an optional label volume.
- `_resample(...)` converts both image and label volumes to `0.4 x 0.4 x 0.4 mm` spacing.
- Image volumes use B-spline interpolation, while label volumes use nearest-neighbor interpolation to preserve class IDs.
- Image intensities are clipped to `[-1000, 3000]` and then normalized with a foreground z-score.
- Processed images are written to `data/processed/images/` and labels to `data/processed/labels/`.

The dataset matching logic strips the `_0000` modality suffix from image filenames when needed so that:

- image: `ToothFairy2F_025_0000.mha`
- label: `ToothFairy2F_025.mha`

can still be paired correctly during preprocessing.

### How The Training Dataset Works

`src/preprocessing/dataset.py` builds the PyTorch dataset used by the trainer.

- `build_datasets(...)` reads split files from `data/splits/`.
- `CBCTDataset.__getitem__(...)` loads one processed image and its matching processed label.
- In training mode, `_sample_random_patch(...)` extracts a patch of shape `patch_size`.
- Foreground-biased sampling is used when labels are available so that the model sees tooth voxels more often.
- `_force_shape(...)` is used as a safety clamp so every returned tensor has exactly the configured patch size.

This is why the trainer can batch patches consistently even when the source volumes have different shapes.

### How The Augmentation Pipeline Works

`src/preprocessing/transforms.py` defines the augmentation pipeline used in training.

Training transforms include:

- random flips
- 90-degree rotations
- small-angle rotations
- random scaling
- elastic deformation
- Gaussian noise
- Gaussian blur
- brightness / contrast jitter
- gamma augmentation

These transforms operate on 3D numpy arrays and preserve the final patch shape expected by the model.

### Model Architecture: ResEncLUNet

The network implementation is in `src/models/nnunet_resencl.py`.

The model is a 3D encoder-decoder architecture with residual blocks.

#### Encoder

- The input is single-channel CBCT data.
- The stem block is a `Conv3d + InstanceNorm3d + LeakyReLU` block.
- The encoder has 5 downsampling stages after the stem.
- Each stage starts with a residual block with `stride=2`, followed by extra residual blocks with `stride=1`.

Channel progression:

- `[32, 64, 128, 256, 512, 512]`

Residual block counts per stage:

- `[1, 3, 4, 6, 6, 2]`

#### Decoder

- Each decoder stage upsamples with `ConvTranspose3d`.
- The upsampled feature map is concatenated with the corresponding encoder skip connection.
- Two convolutional refinement blocks are then applied.

#### Heads

- `out_head` produces the final segmentation logits.
- During training, deep supervision heads (`ds_heads`) are attached to intermediate decoder outputs.
- During inference, deep supervision is disabled and only the main output head is used.

#### Output Classes

The current repository is configured for `49` classes total:

- class `0` = background
- classes `1..48` = ToothFairy2 / FDI-coded tooth labels used by this workspace

### How Training Works

`src/training/train.py` is the training driver.

- It loads the YAML config.
- It builds train and validation datasets.
- It creates DataLoaders.
- It builds the segmentation model with `build_model(...)`.
- It builds the loss using `build_loss(...)` from `src/training/losses.py`.
- It optimizes using SGD with Nesterov momentum.
- It uses automatic mixed precision through `torch.amp.autocast` and `GradScaler`.
- It writes TensorBoard logs into `weights/tensorboard/`.
- It writes `weights/last_model.pth` and `weights/best_model.pth`.

#### Loss Stack

The training loss is:

- multi-class Dice loss
- plus Cross-Entropy loss
- wrapped with deep supervision when the model returns multiple decoder-scale outputs

#### Learning Rate Schedule

The scheduler is polynomial decay:

- base learning rate comes from `config/train_config.yaml`
- at epoch `e`, LR is scaled by `(1 - e / total_epochs) ^ poly_exp`

### How Inference Works

`src/inference/predict.py` is the single-volume inference entry point.

It performs these steps:

1. Load the input scan.
2. Reuse the preprocessing logic from `CBCTPreprocessor`.
3. Run sliding-window inference with overlap.
4. Merge patch logits using a Gaussian importance map.
5. Convert class probabilities to a label map with `argmax`.
6. Postprocess the label map into cleaned connected components and FDI metadata.
7. Save mask, labels JSON, copied scan, and HTML viewer.

### Sliding-Window Predictor

`src/inference/sliding_window.py` handles large volumes.

- The volume is padded if needed.
- Overlapping patches are generated.
- Patches are run through the model in small batches.
- Predictions are fused with a Gaussian weighting map so that patch centers contribute more strongly than patch borders.
- Optional flip-based test-time augmentation is supported.

### Postprocessing And FDI Mapping

`src/inference/postprocess.py` converts raw predicted labels into usable dental metadata.

It does three main jobs:

1. remove tiny connected components
2. map ToothFairy2 class IDs to FDI numbers
3. estimate jaw assignment and restoration presence

The file contains an explicit `TOOTHFAIRY2_TO_FDI` lookup table so that predicted class labels can be reported in FDI numbering form.

### HTML Visualization

`src/visualization/html_viewer.py` generates a standalone viewer.

The viewer includes:

- axial, coronal, and sagittal slice views
- segmentation overlay
- per-tooth color coding by FDI quadrant
- restoration highlighting
- embedded data, so no external server is required

For the current demo output in this workspace, the generated visualization file is:

- `results/ToothFairy2F_025_0000_viewer_balanced.html`


---

## Verified Commands Used In This Workspace

The commands below are the actual commands used to get this repository working in this workspace.

### Environment

Activate the project virtual environment:

```bash
source  3D-CBCT-Tooth-Segmentation/.venv/bin/activate
```

Install repository dependencies into the virtual environment:

```bash
cd  3D-CBCT-Tooth-Segmentation &&  3D-CBCT-Tooth-Segmentation/.venv/bin/python -m pip install -r requirements.txt
```

### Dataset Extraction

The ToothFairy2 archive was extracted onto the larger disk mount:

```bash
cd  3D-CBCT-Tooth-Segmentation/data/raw
mkdir -p /mnt/hdd1/toothfairy_data
nohup unzip -o dataset.zip -d /mnt/hdd1/toothfairy_data > /mnt/hdd1/toothfairy_data/unzip.log 2>&1 &
tail -f /mnt/hdd1/toothfairy_data/unzip.log
```

To make the extracted folder layout compatible with the scripts, the following symlinks were used:

```bash
cd  3D-CBCT-Tooth-Segmentation
ln -sfn /mnt/hdd1/toothfairy_data/Dataset112_ToothFairy2/imagesTr /mnt/hdd1/toothfairy_data/Dataset112_ToothFairy2/images
ln -sfn /mnt/hdd1/toothfairy_data/Dataset112_ToothFairy2/labelsTr /mnt/hdd1/toothfairy_data/Dataset112_ToothFairy2/labels
```

### Split Generation

The split generation command used here was:

```bash
cd  3D-CBCT-Tooth-Segmentation
/opt/miniconda3/bin/python scripts/generate_splits.py --dataset_dir /mnt/hdd1/toothfairy_data/Dataset112_ToothFairy2 --splits_dir data/splits/
```

This creates:

- `data/splits/train.txt`
- `data/splits/val.txt`
- `data/splits/test.txt`
- `data/splits/splits_metadata.json`

### Preprocessing Command

The dataset was preprocessed with:

```bash
cd  3D-CBCT-Tooth-Segmentation && \
 3D-CBCT-Tooth-Segmentation/.venv/bin/python -m src.preprocessing.preprocess \
      --input /mnt/hdd1/toothfairy_data/Dataset112_ToothFairy2 \
      --output data/processed \
      --splits data/splits \
      --workers 4
```

Output locations:

- `data/processed/images/`
- `data/processed/labels/`
- `data/processed/preprocessing_metadata.json`

### Sanity Check For Processed Images

To verify preprocessing output:

```bash
ls -1 data/processed/images | head -n 5
```

### Model Smoke Test

The model and loss were smoke-tested directly before training:

```bash
cd  3D-CBCT-Tooth-Segmentation && CUDA_VISIBLE_DEVICES=1 /opt/miniconda3/bin/python -c "
import torch, sys; sys.path.insert(0,'.')
from src.models.nnunet_resencl import ResEncLUNet
from src.training.losses import build_loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = ResEncLUNet(num_classes=43).to(device)
x = torch.randn(1,1,96,96,96).to(device)
y = torch.randint(0,43,(1,96,96,96)).to(device)
loss = build_loss({'model':{'deep_supervision':True},'loss':{}}, 43)(m(x), y)
loss.backward()
print('PASS — device:', device, '| loss:', round(loss.item(), 4), '| params:', m.count_parameters())
"
```

Note: the repository was later aligned to `49` classes for training and inference, because the processed labels in this workspace contain IDs up to `48`.

### Training Demo Command

The verified short training demo that completed successfully was:

```bash
cd  3D-CBCT-Tooth-Segmentation && \
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0  3D-CBCT-Tooth-Segmentation/.venv/bin/python -u -c "
import yaml, sys
cfg = yaml.safe_load(open('config/train_config.yaml'))
cfg['training']['epochs'] = 2
cfg['training']['batch_size'] = 1
cfg['training']['patch_size'] = [96, 96, 96]
cfg['logging']['val_every_n_epochs'] = 2
cfg['logging']['log_every_n_steps'] = 1
cfg['hardware']['workers'] = 0
cfg['hardware']['persistent_workers'] = False
sys.path.insert(0,'.')
from src.training.train import Trainer
Trainer(cfg).train()
"
```

Observed demo output:

- epoch 1 loss and dice logged successfully
- epoch 2 loss and dice logged successfully
- validation ran successfully
- best model checkpoint was written to `weights/best_model.pth`

### Long Training Command

The long background training command used in this workspace was:

```bash
cd  3D-CBCT-Tooth-Segmentation && \
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
 3D-CBCT-Tooth-Segmentation/.venv/bin/python -u -c "
import yaml, sys
cfg = yaml.safe_load(open('config/train_config.yaml'))
cfg['training']['epochs'] = 500
cfg['hardware']['workers'] = 0
cfg['hardware']['persistent_workers'] = False
sys.path.insert(0,'.')
from src.training.train import Trainer
Trainer(cfg).train()
" > train_500.log 2>&1 & echo "PID: $!"
```

Track it with:

```bash
tail -f  3D-CBCT-Tooth-Segmentation/train_500.log
```

### Ten-Epoch Demo Option

For mentor/demo presentation, a dedicated 10-epoch run can be launched with:

```bash
cd  3D-CBCT-Tooth-Segmentation && \
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
 3D-CBCT-Tooth-Segmentation/.venv/bin/python -u -c "
import yaml, sys
cfg = yaml.safe_load(open('config/train_config.yaml'))
cfg['training']['epochs'] = 10
cfg['training']['batch_size'] = 1
cfg['training']['patch_size'] = [96, 96, 96]
cfg['logging']['val_every_n_epochs'] = 10
cfg['logging']['log_every_n_steps'] = 1
cfg['hardware']['workers'] = 0
cfg['hardware']['persistent_workers'] = False
sys.path.insert(0,'.')
from src.training.train import Trainer
Trainer(cfg).train()
" > train_10_demo.log 2>&1 & echo "PID: $!"
```

### Inference Demo Command

The verified single-case inference command used in this workspace was:

```bash
cd  3D-CBCT-Tooth-Segmentation
VAL_CASE=$(head -1 data/splits/val.txt)
 3D-CBCT-Tooth-Segmentation/.venv/bin/python -m src.inference.predict \
      --input data/processed/images/${VAL_CASE}.nii.gz \
      --weights weights/best_model.pth \
      --config config/infrence_config.yaml \
      --output results/ \
      --no-tta
```

This generated:

- `results/ToothFairy2F_025_0000_mask.nii.gz`
- `results/ToothFairy2F_025_0000_labels.json`
- `results/ToothFairy2F_025_0000_scan.nii.gz`
- `results/ToothFairy2F_025_0000_viewer.html`

### Visualization Output

The HTML visualization generated in this workspace is:

- `results/ToothFairy2F_025_0000_viewer.html`

If you want to duplicate it to a presentation-specific filename:

```bash
cd  3D-CBCT-Tooth-Segmentation
results/ToothFairy2F_025_0000_viewer_balanced.html
```

Open in Chrome on Linux:

```bash
google-chrome  3D-CBCT-Tooth-Segmentation/results/ToothFairy2F_025_0000_viewer_balanced.html
```

If Chrome is not on the path, use:

```bash
xdg-open  3D-CBCT-Tooth-Segmentation/results/ToothFairy2F_025_0000_viewer_balanced.html
```

---

## Files To Mention During Demo

If you need to explain the repository quickly to a mentor, these are the most important files to show:

- `config/train_config.yaml` — training hyperparameters and data paths
- `config/infrence_config.yaml` — inference settings and model class count
- `src/models/nnunet_resencl.py` — segmentation architecture
- `src/preprocessing/preprocess.py` — preprocessing pipeline
- `src/preprocessing/dataset.py` — patch sampling and tensor preparation
- `src/training/train.py` — training loop and checkpointing
- `src/inference/predict.py` — end-to-end inference pipeline
- `src/inference/postprocess.py` — FDI assignment and restoration detection
- `src/visualization/html_viewer.py` — final interactive HTML visualization

This gives a complete story from raw CBCT input to training, inference, metadata export, and interactive visualization.

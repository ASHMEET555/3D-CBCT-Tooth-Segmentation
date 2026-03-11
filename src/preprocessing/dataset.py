"""
src/preprocessing/dataset.py
PyTorch Dataset for CBCT tooth segmentation.
"""
from __future__ import annotations
import random
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
from .io_utils import load_volume

class CBCTDataset(Dataset):
    def __init__(self, image_dir, label_dir, case_ids,
                 patch_size=(128, 128, 128), mode='train',
                 transform=None, foreground_p=0.33):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.mode = mode
        self.transform = transform
        self.foreground_p = foreground_p
        self.cases = self._resolve_cases(case_ids)
        if not self.cases:
            raise FileNotFoundError(
                f"No valid cases found in {image_dir}. "
                "Have you run the preprocessing script?"
            )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        img_path, lbl_path = self.cases[idx]
        vol = load_volume(img_path)
        image = vol["array"]
        label = None
        if lbl_path is not None and lbl_path.exists():
            label_vol = load_volume(lbl_path)
            label = label_vol["array"].astype(np.int64)
        if self.mode == "train" and label is not None:
            image, label = self._sample_random_patch(image, label)
        # FIX: GitHub had self.transforms(image, label) — attribute is self.transform (no s)
        if self.transform is not None:
            image, label = self.transform(image, label)
        image_t = torch.from_numpy(image[None].astype(np.float32))
        if label is not None:
            label_t = torch.from_numpy(label)
            return {"image": image_t, "label": label_t,
                    "case_id": img_path.stem.replace(".nii", "")}
        else:
            return {"image": image_t, "case_id": img_path.stem.replace(".nii", "")}

    def _resolve_cases(self, case_ids):
        cases = []
        for cid in case_ids:
            img = None
            for ext in (".nii.gz", ".nii", ".mha"):
                candidate = self.image_dir / (cid + ext)
                if candidate.exists():
                    img = candidate
                    break
            if img is None:
                continue
            lbl = self.label_dir / img.name
            cases.append((img, lbl if lbl.exists() else None))
        return cases

    # FIX: GitHub named this _random_crop but called self._sample_random_patch — unified to _sample_random_patch
    def _sample_random_patch(self, image, label):
        D, H, W = image.shape
        pd, ph, pw = self.patch_size
        if D < pd or H < ph or W < pw:
            pad = ((max(0, pd-D), 0), (max(0, ph-H), 0), (max(0, pw-W), 0))
            image = np.pad(image, pad, mode="constant", constant_values=image.min())
            label = np.pad(label, pad, mode="constant", constant_values=0)
            D, H, W = image.shape
        if random.random() < self.foreground_p and label.max() > 0:
            fg_coords = np.argwhere(label > 0)
            cz, cy, cx = fg_coords[random.randint(0, len(fg_coords) - 1)]
            dz = int(np.clip(cz - pd // 2, 0, D - pd))
            dy = int(np.clip(cy - ph // 2, 0, H - ph))
            dx = int(np.clip(cx - pw // 2, 0, W - pw))
        else:
            dz = random.randint(0, D - pd)
            dy = random.randint(0, H - ph)
            dx = random.randint(0, W - pw)
        return image[dz:dz+pd, dy:dy+ph, dx:dx+pw], label[dz:dz+pd, dy:dy+ph, dx:dx+pw]


def build_datasets(processed_dir, splits_dir, patch_size=(128,128,128),
                   transforms_train=None, transforms_val=None):
    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)

    def read_ids(split):
        txt = splits_dir / f"{split}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"Split file not found: {txt}")
        return [l.strip() for l in txt.read_text().splitlines() if l.strip()]

    train_ids = read_ids("train")
    val_ids = read_ids("val")
    test_ids = read_ids("test")
    image_dir = processed_dir / "images"
    label_dir = processed_dir / "labels"
    # FIX: build_datasets was passing transforms= keyword but CBCTDataset expects transform= (no s)
    ds_train = CBCTDataset(image_dir, label_dir, train_ids, patch_size=patch_size,
                           mode="train", transform=transforms_train)
    ds_val   = CBCTDataset(image_dir, label_dir, val_ids,   patch_size=patch_size,
                           mode="val",   transform=transforms_val)
    ds_test  = CBCTDataset(image_dir, label_dir, test_ids,  patch_size=patch_size,
                           mode="test",  transform=transforms_val)
    return ds_train, ds_val, ds_test

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


def _force_shape(arr: np.ndarray, shape: tuple, fill) -> np.ndarray:
    """Guarantee arr has exactly `shape` via crop then pad. Handles all edge cases."""
    out = np.full(shape, fill, dtype=arr.dtype)
    src, dst = [], []
    for dim in range(3):
        s, t = arr.shape[dim], shape[dim]
        if s >= t:
            s0 = (s - t) // 2
            src.append(slice(s0, s0 + t))
            dst.append(slice(0, t))
        else:
            d0 = (t - s) // 2
            src.append(slice(0, s))
            dst.append(slice(d0, d0 + s))
    out[tuple(dst)] = arr[tuple(src)]
    return out


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
        if self.mode == "train":
            image, label = self._sample_random_patch(image, label)
        if self.transform is not None:
            image, label = self.transform(image, label)
        image = _force_shape(image, self.patch_size, fill=float(image.min())).astype(np.float32, copy=False)
        if label is not None:
            label = _force_shape(label, self.patch_size, fill=0).astype(np.int64, copy=False)
        image_t = torch.tensor(np.ascontiguousarray(image[None]))
        if label is not None:
            label_t = torch.tensor(np.ascontiguousarray(label))
            return {"image": image_t, "label": label_t,
                    "case_id": img_path.stem.replace(".nii", "")}
        else:
            return {"image": image_t,
                    "case_id": img_path.stem.replace(".nii", "")}

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

    def _sample_random_patch(self, image, label):
        pd, ph, pw = self.patch_size

        # Pad volume to at least patch_size on all sides
        D, H, W = image.shape
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = np.pad(image,
                           ((pad_d//2, pad_d - pad_d//2),
                            (pad_h//2, pad_h - pad_h//2),
                            (pad_w//2, pad_w - pad_w//2)),
                           mode="constant", constant_values=image.min())
            label = np.pad(label,
                           ((pad_d//2, pad_d - pad_d//2),
                            (pad_h//2, pad_h - pad_h//2),
                            (pad_w//2, pad_w - pad_w//2)),
                           mode="constant", constant_values=0)

        D, H, W = image.shape

        if random.random() < self.foreground_p and label.max() > 0:
            fg_coords = np.argwhere(label > 0)
            cz, cy, cx = fg_coords[random.randint(0, len(fg_coords) - 1)]
            dz = int(np.clip(cz - pd // 2, 0, D - pd))
            dy = int(np.clip(cy - ph // 2, 0, H - ph))
            dx = int(np.clip(cx - pw // 2, 0, W - pw))
        else:
            dz = random.randint(0, max(0, D - pd))
            dy = random.randint(0, max(0, H - ph))
            dx = random.randint(0, max(0, W - pw))

        patch_img = image[dz:dz+pd, dy:dy+ph, dx:dx+pw]
        patch_lbl = label[dz:dz+pd, dy:dy+ph, dx:dx+pw]

        # Force exact shape — safety net for any remaining edge case
        patch_img = _force_shape(patch_img, (pd, ph, pw), fill=float(image.min()))
        patch_lbl = _force_shape(patch_lbl, (pd, ph, pw), fill=0)
        return patch_img, patch_lbl


def build_datasets(processed_dir, splits_dir, patch_size=(128, 128, 128),
                   transforms_train=None, transforms_val=None):
    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)

    def read_ids(split):
        txt = splits_dir / f"{split}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"Split file not found: {txt}")
        return [l.strip() for l in txt.read_text().splitlines() if l.strip()]

    train_ids = read_ids("train")
    val_ids   = read_ids("val")
    test_ids  = read_ids("test")
    image_dir = processed_dir / "images"
    label_dir = processed_dir / "labels"

    ds_train = CBCTDataset(image_dir, label_dir, train_ids, patch_size=patch_size,
                           mode="train", transform=transforms_train)
    ds_val   = CBCTDataset(image_dir, label_dir, val_ids,   patch_size=patch_size,
                           mode="val",   transform=transforms_val)
    ds_test  = CBCTDataset(image_dir, label_dir, test_ids,  patch_size=patch_size,
                           mode="test",  transform=transforms_val)
    return ds_train, ds_val, ds_test
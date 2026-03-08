"""
src/preprocessing/dataset.py

PyTorch Dataset for CBCT tooth segmentation.
Supports:
  - Random patch sampling during training
  - Full volume inference mode
  - Online augmentation via albumentations-style 3D transforms
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .io_utils import load_volume



# Dataset Classes 
class CBCTDataset(Dataset):
    """
    Parameteres 
    image_dir  : directory of preproccessed image .nii.gz files
    label_dir  : directory of preproccessed label .nii.gz files
    case_ids   : list of case IDs to include (stem of filenames)
    patch_size : (D, H, W) size of random patches to sample during training
    mode  : 'train' | 'val' | 'test" 
    transform: optional callable applied to (image,label) pair 
    foreground_p : prob of sampling foregound -centred patch 
    """
    def __init__(
            self,
            image_dir: Path,
            label_dir: Path,
            case_ids: List[str],
            patch_size: Tuple[int, int, int]=(128, 128, 128),
            mode: str='train',
            transform: Optional[Callable]=None,
            foreground_p: float=0.33,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.mode = mode
        self.transform = transform
        self.foreground_p = foreground_p

        self.cases=self._resolve_cases(case_ids)
        if not self.cases:
            raise FileNotFoundError(f"No valid cases found in {image_dir}."
                                    "Have you run the preproccessing Script?")
        

        # Pytorch interface
    def __len__(self):        return len(self.cases)
    def __getitem__(self,idx:int)->dict:
        img_path,lbl_path=self.cases[idx]
        vol=load_volume(img_path)
        image=vol["array"]

        label=None
        if lbl_path is not None:
            label_val=load_volume(lbl_path)
            label=label_val["array"].astype(np.int64)

        if self.mode=="train" and label is not None:
            image,label=self._sample_random_patch(image,label)
        elif self.mode in ("val","test"):
            pass
        if self.transform is not None:
            image,label=self.transforms(image,label)



        image_t=torch.from_numpy(image[None])
        if label is not None:
            label_t=torch.from_numpy(label)
            return{
                "image":image_t,
                "label":label_t,
                "case_id":img_path.stem.replace(".nii", "")
            }
        else: 
            return {
                "image":image_t,
                "case_id":img_path.stem.replace(".nii", "")
            }
        
        # Private 
    def _resolve_cases(self, case_ids: List[str]) -> List[Tuple[Path, Path]]:
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
            cases.append((img, lbl))
        return cases
    
    def _random_crop(
        self,
        image: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random crop of size self.patch_size.
        With probability foreground_p, the crop is centered on a
        randomly chosen foreground voxel (oversampling for rare classes).
        """
        D, H, W = image.shape
        pd, ph, pw = self.patch_size

        # Pad if volume is smaller than patch
        if D < pd or H < ph or W < pw:
            pad = (
                (max(0, pd - D), 0),
                (max(0, ph - H), 0),
                (max(0, pw - W), 0),
            )
            image = np.pad(image, pad, mode="constant", constant_values=image.min())
            label = np.pad(label, pad, mode="constant", constant_values=0)
            D, H, W = image.shape

        if random.random() < self.foreground_p and label.max() > 0:
            # Sample from foreground voxels
            fg_coords = np.argwhere(label > 0)
            cz, cy, cx = fg_coords[random.randint(0, len(fg_coords) - 1)]
            # Clamp so patch stays within volume
            dz = int(np.clip(cz - pd // 2, 0, D - pd))
            dy = int(np.clip(cy - ph // 2, 0, H - ph))
            dx = int(np.clip(cx - pw // 2, 0, W - pw))
        else:
            dz = random.randint(0, D - pd)
            dy = random.randint(0, H - ph)
            dx = random.randint(0, W - pw)

        image_crop = image[dz:dz+pd, dy:dy+ph, dx:dx+pw]
        label_crop = label[dz:dz+pd, dy:dy+ph, dx:dx+pw]
        return image_crop, label_crop
    

# Factory helper 
def build_datasets(
    processed_dir: str,
    splits_dir: str,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    transforms_train=None,
    transforms_val=None,
) -> Tuple[CBCTDataset, CBCTDataset, CBCTDataset]:
    """
    Build train / val / test datasets from preprocessed directory + split files.
    """
    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)

    def read_ids(split: str) -> List[str]:
        txt = splits_dir / f"{split}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"Split file not found: {txt}")
        return [l.strip() for l in txt.read_text().splitlines() if l.strip()]

    train_ids = read_ids("train")
    val_ids = read_ids("val")
    test_ids = read_ids("test")

    image_dir = processed_dir / "images"
    label_dir = processed_dir / "labels"

    ds_train = CBCTDataset(
        image_dir, label_dir, train_ids,
        patch_size=patch_size, mode="train",
        transforms=transforms_train,
    )
    ds_val = CBCTDataset(
        image_dir, label_dir, val_ids,
        patch_size=patch_size, mode="val",
        transforms=transforms_val,
    )
    ds_test = CBCTDataset(
        image_dir, label_dir, test_ids,
        patch_size=patch_size, mode="test",
        transforms=transforms_val,
    )
    return ds_train, ds_val, ds_test
    






"""
src/preprocessing/preprocess.py

Full preprocessing pipeline for CBCT volumes:
  1. Load (NIfTI / MHA / DICOM)
  2. Resample to isotropic spacing (default 0.4 mm)
  3. Clip HU values [-1000, 3000]
  4. Z-score normalization (foreground voxels only)
  5. Save preprocessed volume + label pair

Also contains a CLI entry point for batch processing.

Usage:
    python src/preprocessing/preprocess.py \
        --input  data/raw/ToothFairy2 \
        --output data/processed/ \
        --splits data/splits/ \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
from loguru import logger
from tqdm import tqdm

from .io_utils import load_volume, save_volume
#constants 
TARGET_SPACING = (0.4, 0.4, 0.4)   # mm, isotropic
HU_MIN, HU_MAX = -1000.0, 3000.0


# core classes 
class CBCTPreprocessor:
    """
    Preprocesses a single CBCT volume + Optioal label mask 
    Parameters 
    target_spacing: (sz, sy, sx) in mm
    hu": (min, max) HU clipping range
    """
    def __init__(
            self,
            target_spacing: Tuple[float, float, float] = TARGET_SPACING,
            hu_range: Tuple[float, float] = (HU_MIN, HU_MAX),
    ):
        self.target_spacing = target_spacing
        self.hu_min, self.hu_max = hu_range

    # Public 
    def process(self,image_path:Path,label_path:Optional[Path]=None,
                out_image_path:Optional[Path]=None,out_label_path:Optional[Path]=None)-> dict:
        
        """ Full preprocessing pipeline
        Reeturn a metadata dict with sahpe , spacing 
        """
        logger.debug(f"Processing:{image_path.name}")
        # 1. load 
        vol=load_volume
        array=vol["array"]
        spacing=vol["spacing"]
        origin=vol["origin"]
        direction=vol["direction"]
        label_array=None
        if label_path is not None and label_path.exists():
            label_vol=load_volume(label_path)
            label_array=label_vol["array"].astype(np.uint16)

        original_shape=array.shape
        original_spacing=spacing

        #2 Resample image 
        array,new_spacing=self._resample(array,spacing,self.target_spacing,is_label=False)
        # 3 resample label nearest neighor 
        if label_array is not None:
            label_array,_=self._resample(label_array,spacing,self.target_spacing,is_label=True)

        # 4 HU clipping 
        array=np.clip(array,self.hu_range[0],self.hu_range[1])

        # 5 Z Score normalizatino on foreground only 
        array=self._normalize(array)

        # 6 Save 
        if out_image_path is not None:
            save_volume(array,new_spacing,origin,direction,out_image_path,is_label=False)
        if label_array is not None and out_label_path is not None:
            save_volume(label_array,new_spacing,origin,direction,out_label_path,is_label=True)

        return {
            "case_id":image_path.stem.replace(".nii",""),
            "original_shape":list(original_shape),
            "original_spacing":list(original_spacing),
            "processed_shape":list(array.shape),
            "processed_spacing":list(new_spacing),

        }
    

    # Private 
    def _resample(
        self,
        array: np.ndarray,
        current_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float],
        is_label: bool,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Resample array from current_spacing to target_spacing using SimpleITK.
        array shape: (D, H, W) = (z, y, x)
        spacing convention: (sz, sy, sx)
        """
        # Convert to SimpleITK (expects x, y, z ordering)
        array_xyz = np.transpose(array, (2, 1, 0))
        image = sitk.GetImageFromArray(array_xyz)
        image.SetSpacing([float(current_spacing[2]),
                          float(current_spacing[1]),
                          float(current_spacing[0])])

        # Compute new size
        old_size = np.array(image.GetSize(), dtype=float)  # (sx, sy, sz) counts
        old_sp = np.array(image.GetSpacing())
        new_sp = np.array([target_spacing[2], target_spacing[1], target_spacing[0]])
        new_size = np.round(old_size * old_sp / new_sp).astype(int).tolist()

        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_sp.tolist())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetTransform(sitk.Transform())

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            resample.SetDefaultPixelValue(0)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
            resample.SetDefaultPixelValue(float(self.hu_range[0]))

        resampled = resample.Execute(image)
        out_array = sitk.GetArrayFromImage(resampled)  # (z, y, x)
        out_spacing = (float(target_spacing[0]), float(target_spacing[1]), float(target_spacing[2]))
        return out_array.astype(array.dtype), out_spacing
    

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        """
        Z-score normalization computed on foreground voxels only.
        Foreground is defined as HU > -500 (above air).
        """
        fg_mask = array > -500.0
        if fg_mask.sum() == 0:
            return array  # edge case: all air
        mean = float(array[fg_mask].mean())
        std = float(array[fg_mask].std())
        if std < 1e-6:
            return np.zeros_like(array)
        return ((array - mean) / std).astype(np.float32)
    
# Batch Processing helpers 
def _find_image_label_pairs(
    dataset_dir: Path,
) -> list[Tuple[Path, Optional[Path]]]:
    """
    Discover (image, label) path pairs.
    Handles both flat layouts and per-case subdirectory layouts.
    """
    pairs = []

    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"

    if image_dir.exists():
        # Flaty layout images + labels 
        for ext in ("*.nii.gz","*.mha","*.nii"):
            for img_path in image_dir.glob(ext):
                stem=img_path.stem.replace(".nii.gz","").replace(".nii","").replace(".mha","")
                label_path=None
                for lext in (".nii.gz",".mha",".nii"):
                    candidate=label_dir / (stem+lext)
                    if candidate.exists():
                        label_path=candidate
                        break
                pairs.append((img_path,label_path))
    else:
        # Per case subdirectory layout 
        for subdir in sorted(dataset_dir.iterdir()):
            if not subdir.is_dir():
                continue
            img=None
            lbl=None
            for name in ("scan.nii.gz","image.nii.gz","data.nii.gz","scan.mha"):
                if(subdir/name).exists():
                    img=subdir/name
                    break
            for name in ("label.nii.gz","gt.nii.gz","seg.nii.gz","label.mha"):
                if(subdir/name).exists():
                    lbl=subdir/name
                    break
            if img is not None:
                pairs.append((img,lbl))

        return pairs
    
def _process_one(args_tuple):
    """Wrapper for multiprocessing."""
    image_path, label_path, out_image, out_label, target_spacing = args_tuple
    preprocessor = CBCTPreprocessor(target_spacing=target_spacing)
    try:
        meta = preprocessor.process(image_path, label_path, out_image, out_label)
        return meta, None
    except Exception as e:
        return None, f"{image_path}: {e}"
    

    # cli 
def parse_args():
    p = argparse.ArgumentParser(description="Preprocess CBCT dataset")
    p.add_argument("--input", required=True, help="Raw dataset directory")
    p.add_argument("--output", required=True, help="Processed output directory")
    p.add_argument("--splits", default=None, help="Optional: process only cases in splits/*.txt")
    p.add_argument("--spacing", type=float, nargs=3, default=[0.4, 0.4, 0.4],
                   metavar=("SZ", "SY", "SX"), help="Target spacing in mm")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    return p.parse_args()



def main():
    args=parse_args()
    dataset_dir=Path(args.input)
    output_dir=Path(args.output)
    output_dir.mkdir(parents=True,exist_ok=True)

    target_spacing = tuple(args.spacing)

    # Discover all pairs 
    pairs=_find_image_label_pairs(dataset_dir)
    logger.info(f"Found {len(pairs)} image-label paris in  {dataset_dir}")
    if not pairs:
        logger.error("No data found . Did you run Scripts/download_dataset.py?")
        return 

    if args.splits:
        split_dir = Path(args.splits)
        all_ids = set()
        for txt in split_dir.glob("*.txt"):
            all_ids |= {l.strip() for l in txt.read_text().splitlines() if l.strip()}
        pairs = [
            (img, lbl) for img, lbl in pairs
            if img.stem.replace(".nii", "") in all_ids
        ]
        logger.info(f"Filtered to {len(pairs)} cases based on splits")

    tasks = []
    for img_path, lbl_path in pairs:
        stem = img_path.name.replace(".nii.gz", "").replace(".nii", "").replace(".mha", "")
        out_img = output_dir / "images" / f"{stem}.nii.gz"
        out_lbl = (output_dir / "labels" / f"{stem}.nii.gz") if lbl_path else None
        tasks.append((img_path, lbl_path, out_img, out_lbl, target_spacing))

    # Run 
    all_meta=[]
    errors=[]

    if args.workers >1:
        with mp.Pool(args.workers) as pool:
            for meta,err in tqdm(pool.imap_unordered(_process_one,tasks),total=len(tasks),desc="Preprocessing"):
                if err:
                    errors.append(err)
                else:
                    all_meta.append(meta)
    else:
        for task in tqdm(tasks,desc="Preprocessing"):
            meta,err=_process_one(task)
            if err:
                errors.append(err)
            else:
                all_meta.append(meta)

    #save metadata
    meta_path = output_dir / "preprocessing_metadata.json"
    meta_path.write_text(json.dumps(all_meta, indent=2))

    logger.info(f"✅ Done: {len(all_meta)} succeeded, {len(errors)} failed")
    if errors:
        logger.warning("Errors:")
        for e in errors:
            logger.warning(f"  {e}")
    logger.info(f"Metadata saved → {meta_path}")


if __name__=="__main__":
    main()





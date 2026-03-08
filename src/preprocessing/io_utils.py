"""
src/preprocessing/io_utils.py

Unified I/O utilities for:
  - NIfTI (.nii, .nii.gz)
  - MetaImage (.mha, .mhd)
  - DICOM series (folder of .dcm files)

All loaders return a consistent dict:
{
    "array"    : np.ndarray  [D, H, W], float32,
    "spacing"  : tuple(float, float, float),  # (z, y, x) mm
    "origin"   : tuple(float, float, float),
    "direction": np.ndarray  [3, 3],
    "path"     : str,
}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
import nibabel as nib

VolumeDict = dict

def load_volume(path: Union[str, Path]) -> VolumeDict:
    """
    Auto-detect format and load a CBCT volume.

    Parameters
    ----------
    path : str or Path
        Path to a .nii / .nii.gz / .mha / .mhd file,
        OR a directory containing a DICOM series.

    Returns
    -------
    VolumeDict
    """
    path = Path(path)

    if path.is_dir():
        return load_dicom_series(path)

    suffix = "".join(path.suffixes).lower()
    if suffix in (".nii", ".nii.gz"):
        return _load_nifti(path)
    elif suffix in (".mha", ".mhd"):
        return _load_sitk(path)
    elif suffix == ".dcm":
        # Single DICOM → treat parent as series directory
        return load_dicom_series(path.parent)
    else:
        # Fallback: try SimpleITK (handles many formats)
        try:
            return _load_sitk(path)
        except Exception as e:
            raise ValueError(
                f"Unsupported file format '{suffix}' at {path}. "
                "Supported: .nii, .nii.gz, .mha, .mhd, DICOM directory"
            ) from e
        
def load_dicom_series(directory: Union[str, Path]) -> VolumeDict:
    """Load a DICOM series from a directory using SimpleITK."""
    directory = Path(directory)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))

    if not dicom_names:
        raise FileNotFoundError(f"No DICOM files found in {directory}")

    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    return _sitk_to_dict(image, str(directory))


def save_volume(
        array:np.ndarray,
        spacing:tuple,
        origin:tuple,
        direction:np.ndarray,
        out_path:Union[str, Path],
        is_label:bool=False
)-> None:
    """ 
    Save a numpy array of NifTI
    parameters 
    array: np.ndarray [D, H, W], float32
    spacing: (sz, y, x) in mm
    origin: (oz, oy, ox) in mm
    direction: 3x3 rotation matrix
    out_path : output .nii.gz file path
    is_label: if True, save as int16 (for segmentation masks)
    
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_label:
        array = array.astype(np.uint16)
    else: 
        array = array.astype(np.float32)

    array_xyz=np.transpose(array, (2, 1, 0))  # [W, H, D]
    image=sitk.GetImageFromArray(array_xyz)
    image.SetSpacing([float(spacing[2]),float(spacing[1]),float(spacing[0])])  # (x, y, z)
    image.SetOrigin([float(origin[2]),float(origin[1]),float(origin[0])])  # (x, y, z)

    if direction is not None:
        # SimpleITK expects a 9-element list for direction (row-major)
        image.SetDirection(direction.flatten().tolist())

    sitk.WriteImage(image, str(out_path),useCompression=True)


# Internl loaders 
def _load_nifti(path: Path) -> VolumeDict:
    """Load a NIfTI file using nibabel."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    array = img.get_fdata(dtype=np.float32)

    # nibabel: (x, y, z) → convert to (z, y, x) for consistent [D,H,W]
    affine = img.affine
    voxel_sizes = img.header.get_zooms()  # (sx, sy, sz)
    spacing = (float(voxel_sizes[2]), float(voxel_sizes[1]), float(voxel_sizes[0]))
    return {
        "array": array,
        "spacing": spacing,
        "origin": tuple(affine[:3, 3].tolist()),
        "direction": np.eye(3),
        "path": str(path),
    }


def _load_sitk(path: Path) -> VolumeDict:
    """Load via SimpleITK (.mha, .mhd, or any ITK-supported format)."""
    image = sitk.ReadImage(str(path))
    return _sitk_to_dict(image, str(path))


def _sitk_to_dict(image: sitk.Image, path: str)-> VolumeDict:
    """ Conver t simpleitk image to our standard volumedict"""

    array=sitk.GetArrayFromImage(image).astype(np.float32)  # [D, H, W]
    spacing_xyz=image.GetSpacing()  # (x, y, z)
    spacing=(float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))  # (z, y, x)
    origin_xyz=image.GetOrigin()  # (x, y, z)
    origin=(float(origin_xyz[2]), float(origin_xyz[1]), float(origin_xyz[0]))  # (z, y, x)
    direction_flat=image.GetDirection()  # 9 elements (row-major)

    direction=np.array(direction_flat).reshape


    return{
        "array": array,
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "path": path,

    }
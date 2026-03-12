"""
src/preprocessing/io_utils.py
Unified I/O for NIfTI, MHA, DICOM.
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
    path = Path(path)
    if path.is_dir():
        return load_dicom_series(path)
    suffix = "".join(path.suffixes).lower()
    if suffix in (".nii", ".nii.gz"):
        return _load_nifti(path)
    elif suffix in (".mha", ".mhd"):
        return _load_sitk(path)
    elif suffix == ".dcm":
        return load_dicom_series(path.parent)
    else:
        try:
            return _load_sitk(path)
        except Exception as e:
            raise ValueError(f"Unsupported format '{suffix}' at {path}") from e

def load_dicom_series(directory: Union[str, Path]) -> VolumeDict:
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

def save_volume(array, spacing, origin, direction, out_path, is_label=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if is_label:
        array = array.astype(np.uint16)
    else:
        array = array.astype(np.float32)
    array_xyz = np.transpose(array, (2, 1, 0))
    image = sitk.GetImageFromArray(array_xyz)
    image.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
    image.SetOrigin([float(origin[2]), float(origin[1]), float(origin[0])])
    if direction is not None:
        if isinstance(direction, np.ndarray) and direction.shape == (3, 3):
            image.SetDirection(direction.flatten().tolist())
    sitk.WriteImage(image, str(out_path), useCompression=True)

def _load_nifti(path: Path) -> VolumeDict:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    array = np.array(img.get_fdata(dtype=np.float32))  # Force copy out of memmap
    affine = img.affine
    voxel_sizes = img.header.get_zooms()
    spacing = (float(voxel_sizes[2]), float(voxel_sizes[1]), float(voxel_sizes[0]))
    return {
        "array": array,
        "spacing": spacing,
        "origin": tuple(affine[:3, 3].tolist()),
        "direction": np.eye(3),
        "path": str(path),
    }

def _load_sitk(path: Path) -> VolumeDict:
    image = sitk.ReadImage(str(path))
    return _sitk_to_dict(image, str(path))

def _sitk_to_dict(image: sitk.Image, path: str) -> VolumeDict:
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    spacing = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    origin_xyz = image.GetOrigin()
    origin = (float(origin_xyz[2]), float(origin_xyz[1]), float(origin_xyz[0]))
    direction_flat = image.GetDirection()
    # FIX: GitHub had "direction = np.array(...).reshape" — missing (3,3) call → was a method ref not an array
    direction = np.array(direction_flat).reshape(3, 3)
    return {
        "array": array,
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "path": path,
    }

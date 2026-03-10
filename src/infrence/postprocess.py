"""
src/inference/postprocess.py

Postprocessing for CBCT tooth segmentation:

1. Connected component analysis — remove small spurious blobs
2. Jaw separation — classify teeth as upper (maxilla) or lower (mandible)
3. FDI tooth ID assignment — map predicted instance labels → FDI numbers
4. Restoration / pathology detection — flag HU-bright voxels (metal crowns etc.)

FDI Numbering System
--------------------
Upper jaw (maxilla):
  Right: 11-18  (central incisor → 3rd molar)
  Left:  21-28

Lower jaw (mandible):
  Right: 41-48  (mirrored)
  Left:  31-38

Wisdom teeth: 18, 28, 38, 48
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# ToothFairy2 label → FDI mapping
# Labels 1-16: upper jaw (right → left: 11,12,...,18, 21,...,28)
# Labels 17-32: lower jaw (right → left: 41,...,48, 31,...,38)
# Label 0: background
TOOTHFAIRY2_TO_FDI = {
    # Upper jaw
    1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18,
    9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    # Lower jaw
    17: 41, 18: 42, 19: 43, 20: 44, 21: 45, 22: 46, 23: 47, 24: 48,
    25: 31, 26: 32, 27: 33, 28: 34, 29: 35, 30: 36, 31: 37, 32: 38,
    # Additional implants / supernumerary slots
    33: 51, 34: 52, 35: 53, 36: 54, 37: 55,  # primary upper right
    38: 61, 39: 62, 40: 63, 41: 64, 42: 65,  # primary upper left
}

FDI_TO_JAW = {}
for fdi in range(11, 50):
    quadrant = fdi // 10
    if quadrant in (1, 2):
        FDI_TO_JAW[fdi] = "upper"
    elif quadrant in (3, 4):
        FDI_TO_JAW[fdi] = "lower"

# HU threshold for metal / restoration detection
RESTORATION_HU_THRESHOLD = 2500.0


# ──────────────────────────────────────────────────────────────
# Main postprocessor class
# ──────────────────────────────────────────────────────────────

class ToothSegPostprocessor:
    """
    Parameters
    ----------
    min_voxels     : minimum voxels for a connected component to keep
    label_to_fdi   : dict mapping model label → FDI tooth number
    restoration_hu : HU threshold above which voxels are flagged as restoration
    """

    def __init__(
        self,
        min_voxels: int = 100,
        label_to_fdi: Dict[int, int] = None,
        restoration_hu: float = RESTORATION_HU_THRESHOLD,
    ):
        self.min_voxels = min_voxels
        self.label_to_fdi = label_to_fdi or TOOTHFAIRY2_TO_FDI
        self.restoration_hu = restoration_hu

    # ── Public API ───────────────────────────────────────────

    def run(
        self,
        raw_mask: np.ndarray,
        raw_image: Optional[np.ndarray] = None,
        spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Full postprocessing pipeline.

        Parameters
        ----------
        raw_mask  : [D, H, W] integer array of predicted tooth labels
        raw_image : [D, H, W] float array of original HU values (optional)
        spacing   : voxel spacing in mm (z, y, x)

        Returns
        -------
        clean_mask : [D, H, W] integer array (same labels, small CCs removed)
        tooth_info : list of dicts, one per detected tooth
        """
        # 1. Remove small components
        clean_mask = self._remove_small_components(raw_mask)

        # 2. Build per-tooth metadata
        tooth_info = []
        unique_labels = [l for l in np.unique(clean_mask) if l > 0]

        for label_id in unique_labels:
            tooth_mask = (clean_mask == label_id)
            fdi = self.label_to_fdi.get(int(label_id), None)
            centroid = self._centroid_mm(tooth_mask, spacing)
            jaw = self._assign_jaw(fdi, centroid, clean_mask, spacing)
            voxel_count = int(tooth_mask.sum())
            volume_mm3 = voxel_count * float(np.prod(spacing))

            # Restoration detection
            is_restoration = False
            restoration_fraction = 0.0
            if raw_image is not None:
                region = raw_image[tooth_mask]
                restoration_fraction = float((region > self.restoration_hu).mean())
                is_restoration = restoration_fraction > 0.05  # >5% bright voxels

            tooth_info.append({
                "label_id": int(label_id),
                "fdi": fdi,
                "jaw": jaw,
                "centroid_mm": [round(c, 2) for c in centroid],
                "voxel_count": voxel_count,
                "volume_mm3": round(volume_mm3, 1),
                "is_restoration": is_restoration,
                "restoration_fraction": round(restoration_fraction, 4),
            })

        tooth_info.sort(key=lambda t: t["fdi"] or 99)
        return clean_mask, tooth_info

    # ── Private ─────────────────────────────────────────────

    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """Keep only connected components ≥ min_voxels per label class."""
        clean = np.zeros_like(mask)
        for label_id in np.unique(mask):
            if label_id == 0:
                continue
            binary = (mask == label_id)
            labeled, n_cc = ndimage.label(binary)
            for cc_id in range(1, n_cc + 1):
                cc = (labeled == cc_id)
                if cc.sum() >= self.min_voxels:
                    clean[cc] = label_id
        return clean

    def _centroid_mm(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Return centroid in mm (z, y, x)."""
        coords = np.argwhere(mask)
        mean_voxel = coords.mean(axis=0)
        return (
            float(mean_voxel[0] * spacing[0]),
            float(mean_voxel[1] * spacing[1]),
            float(mean_voxel[2] * spacing[2]),
        )

    def _assign_jaw(
        self,
        fdi: Optional[int],
        centroid: Tuple[float, float, float],
        full_mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> str:
        """
        Assign upper/lower jaw.
        Primary: use FDI quadrant.
        Fallback: compare centroid Z against mid-plane of all teeth.
        """
        if fdi is not None:
            return FDI_TO_JAW.get(fdi, "unknown")

        # Fallback: spatial separation along the Z axis
        # Upper teeth tend to be more superior (higher Z in patient coords)
        all_tooth_coords = np.argwhere(full_mask > 0)
        if len(all_tooth_coords) == 0:
            return "unknown"
        mid_z = float(all_tooth_coords[:, 0].mean()) * spacing[0]
        return "upper" if centroid[0] > mid_z else "lower"


# ──────────────────────────────────────────────────────────────
# Utility: save tooth info JSON
# ──────────────────────────────────────────────────────────────

def save_labels_json(tooth_info: List[Dict], out_path: Path) -> None:
    """Save tooth metadata to a JSON file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_teeth_detected": len(tooth_info),
        "teeth": tooth_info,
        "fdi_system": "ISO 3950",
        "notes": "label_id = ToothFairy2 class; fdi = FDI tooth number",
    }
    out_path.write_text(json.dumps(payload, indent=2))


# ──────────────────────────────────────────────────────────────
# Jaw separation mask
# ──────────────────────────────────────────────────────────────

def make_jaw_separation_mask(
    clean_mask: np.ndarray,
    tooth_info: List[Dict],
) -> np.ndarray:
    """
    Create a jaw-separation mask:
      0 = background
      1 = upper jaw teeth
      2 = lower jaw teeth
    """
    jaw_mask = np.zeros_like(clean_mask)
    for tooth in tooth_info:
        lid = tooth["label_id"]
        jaw = tooth["jaw"]
        if jaw == "upper":
            jaw_mask[clean_mask == lid] = 1
        elif jaw == "lower":
            jaw_mask[clean_mask == lid] = 2
    return jaw_mask
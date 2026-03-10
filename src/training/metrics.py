"""
src/training/metrics.py

Evaluation metrics for 3D tooth segmentation:
  - Per-class Dice coefficient
  - Mean Dice (foreground only)
  - Per-class IoU (Jaccard)
  - Hausdorff Distance 95th percentile (HD95)
  - FDI assignment accuracy
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


# Numpy basef for evaluation loop 
def dice_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_background: bool = True,
) -> Dict[int, float]:
    """
    Compute per-class Dice coefficient.

    Parameters
    ----------
    pred, target : integer-labeled arrays of same shape
    num_classes  : total number of classes (including background=0)

    Returns
    -------
    dict mapping class_id → Dice score (NaN if class absent from both)
    """
    scores = {}
    start = 1 if ignore_background else 0

    for c in range(start, num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum()
        denom = p.sum() + t.sum()
        if denom == 0:
            scores[c] = float("nan")
        else:
            scores[c] = float(2.0 * inter / denom)

    return scores

def mean_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> float:
    """Mean Dice over foreground classes (ignoring NaN)."""
    per_class = dice_coefficient(pred, target, num_classes)
    vals = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0




def iou_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> Dict[int, float]:
    """Per-class Intersection-over-Union."""
    scores = {}
    for c in range(1, num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum()
        union = (p | t).sum()
        if union == 0:
            scores[c] = float("nan")
        else:
            scores[c] = float(inter / union)
    return scores


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
    num_classes: int = 43,
) -> Dict[int, float]:
    """
    95th-percentile Hausdorff Distance for each foreground class.
    Requires scipy.

    Returns dict class_id → HD95 in mm.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return {}

    scores = {}
    for c in range(1, num_classes):
        p_bin = (pred == c).astype(np.uint8)
        t_bin = (target == c).astype(np.uint8)

        if p_bin.sum() == 0 or t_bin.sum() == 0:
            scores[c] = float("nan")
            continue

        # Distance from pred surface → target and vice versa
        dist_pt = distance_transform_edt(1 - t_bin, sampling=spacing)
        dist_tp = distance_transform_edt(1 - p_bin, sampling=spacing)

        # Surface voxels (boundary)
        from scipy.ndimage import binary_erosion
        p_surf = p_bin & ~binary_erosion(p_bin)
        t_surf = t_bin & ~binary_erosion(t_bin)

        d_pt = dist_pt[p_surf.astype(bool)]
        d_tp = dist_tp[t_surf.astype(bool)]

        if len(d_pt) == 0 or len(d_tp) == 0:
            scores[c] = float("nan")
        else:
            hd95 = max(np.percentile(d_pt, 95), np.percentile(d_tp, 95))
            scores[c] = float(hd95)

    return scores



# Torch-based (used in training loop for fast approximate metrics)


def batch_dice_torch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Fast per-class Dice computed on logits (batch-level).
    Returns mean Dice over foreground classes as a scalar tensor.
    """
    probs = torch.softmax(logits, dim=1)   # [B, C, ...]
    preds = probs.argmax(dim=1)            # [B, ...]

    dice_vals = []
    for c in range(1, num_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice_vals.append((2 * inter + smooth) / (denom + smooth))

    return torch.stack(dice_vals).mean()



# Aggregation utility
# ──────────────────────────────────────────────────────────────

class MetricAggregator:
    """Accumulate per-case metrics across a validation epoch."""

    def __init__(self):
        self._metrics: List[Dict] = []

    def update(self, case_id: str, pred: np.ndarray, target: np.ndarray,
               num_classes: int, spacing: tuple = (1.0, 1.0, 1.0)):
        dice = dice_coefficient(pred, target, num_classes)
        iou = iou_coefficient(pred, target, num_classes)
        mdice = mean_dice(pred, target, num_classes)
        self._metrics.append({
            "case_id": case_id,
            "mean_dice": mdice,
            "per_class_dice": dice,
            "per_class_iou": iou,
        })

    def summary(self) -> Dict:
        if not self._metrics:
            return {}
        mean_dices = [m["mean_dice"] for m in self._metrics]
        return {
            "mean_dice": float(np.mean(mean_dices)),
            "std_dice": float(np.std(mean_dices)),
            "min_dice": float(np.min(mean_dices)),
            "max_dice": float(np.max(mean_dices)),
            "n_cases": len(self._metrics),
        }

    def reset(self):
        self._metrics = []
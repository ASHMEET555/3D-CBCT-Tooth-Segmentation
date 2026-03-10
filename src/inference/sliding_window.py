"""
src/inference/sliding_window.py

Sliding-window inference for 3D volumetric segmentation.
Handles volumes of arbitrary size by predicting overlapping patches
and aggregating predictions with a Gaussian weighting kernel.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Gaussian importacne map 

def _gaussian_kernel_3d(patch_size:Tuple[int,int,int],sigma_scale:float=0.125) -> np.ndarray:
    """ Create a  3d gaussian importance map for patch aggregation 
    Higher weights at patch center, tapering off at edges 
    """
    tmp = np.zeros(patch_size)
    center = [p // 2 for p in patch_size]
    sigma = [max(1, p * sigma_scale) for p in patch_size]


    for z in range(patch_size[0]):
        for y in range(patch_size[1]):
            for x in range(patch_size[2]):
                tmp[z, y, x] = np.exp(
                    -0.5 * (
                        ((z - center[0]) / sigma[0]) ** 2 +
                        ((y - center[1]) / sigma[1]) ** 2 +
                        ((x - center[2]) / sigma[2]) ** 2
                    )
                )

    # Clamp to small minimum to avoid division by zero
    tmp = np.maximum(tmp, 1e-6)
    return tmp.astype(np.float32)


def _fast_gaussian_kernel_3d(patch_size:Tuple[int,int,int])-> np.ndarray:
    """ Fast seperatble gaussian via output product (approx)"""
    d,h,w=patch_size
    def gasss_1d(n):
        x=np.linspace(-1,1,n)
        return np.exp(-2*x**2).astype(np.float32)
    gd,gh,gw = gasss_1d(d),gasss_1d(h),gasss_1d(w)
    kernel = gd[:, None, None] * gh[None, :, None] * gw[None, None, :]
    return np.maximum(kernel, 1e-6)



class SlidingWindowPredictor:
    """
    performs sliding window infrence on a 3d volums 
    Parameters 
    patch_size: d , h, w patch size 
    overlap: fraction of patch overlap (0-1)
    num_classes : number of output classes for segmentation
    batch_size: number of patches to process in a batch
    use_gaussian: weight prediction by gaussian importance map (default True)
    tta : test time augmentation 
    tta_axes: which axes to flip for tta 


    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5,
        num_classes: int = 43,
        batch_size: int = 2,
        use_gaussian: bool = True,
        tta: bool = False,
        tta_axes: Tuple[int, ...] = (0, 1, 2),
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.tta = tta
        self.tta_axes = tta_axes

        if use_gaussian:
            self.importance_map = _fast_gaussian_kernel_3d(patch_size)
        else:
            self.importance_map = np.ones(patch_size, dtype=np.float32)


    @torch.no_grad()
    def predict(self,volume:np.ndarray,model:torch.nn.Module,device:torch.device)->np.ndarray:
        """ 

        parameters 
        volume: [d,h,w] float 32 array normalized 
        model: pytorch model in eval mode 
        device: torch device 
        returns 
        pred mask : [d,h,w] int array of predicted labels 
        """
        D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        stride = tuple(max(1, int(p * (1 - self.overlap))) for p in self.patch_size)

        # Pad volume if smaller than patch
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        volume_padded = np.pad(
            volume,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=volume.min(),
        )
        D_p, H_p, W_p = volume_padded.shape

        # Accumulators
        pred_sum = np.zeros((self.num_classes, D_p, H_p, W_p), dtype=np.float32)
        weight_sum = np.zeros((D_p, H_p, W_p), dtype=np.float32)

        # Generate all patch coordinates
        patch_coords = []
        for dz in range(0, max(1, D_p - pd + 1), stride[0]):
            dz = min(dz, D_p - pd)
            for dy in range(0, max(1, H_p - ph + 1), stride[1]):
                dy = min(dy, H_p - ph)
                for dx in range(0, max(1, W_p - pw + 1), stride[2]):
                    dx = min(dx, W_p - pw)
                    patch_coords.append((dz, dy, dx))

        # Process in batches
        patch_buffer = []
        coord_buffer = []

        def flush_batch():
            if not patch_buffer:
                return
            batch = np.stack(patch_buffer, axis=0)[:, None]  # [B,1,D,H,W]
            batch_t = torch.from_numpy(batch).to(device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(batch_t)

            if isinstance(logits, list):
                logits = logits[0]

            probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B,C,D,H,W]

            if self.tta:
                probs = self._apply_tta(batch_t, model, device, probs)

            for i, (cz, cy, cx) in enumerate(coord_buffer):
                pred_sum[:, cz:cz+pd, cy:cy+ph, cx:cx+pw] += (
                    probs[i] * self.importance_map[None]
                )
                weight_sum[cz:cz+pd, cy:cy+ph, cx:cx+pw] += self.importance_map

            patch_buffer.clear()
            coord_buffer.clear()

        for cz, cy, cx in patch_coords:
            patch = volume_padded[cz:cz+pd, cy:cy+ph, cx:cx+pw]
            patch_buffer.append(patch.astype(np.float32))
            coord_buffer.append((cz, cy, cx))
            if len(patch_buffer) >= self.batch_size:
                flush_batch()
        flush_batch()

        # Normalize by weight
        weight_sum = np.maximum(weight_sum, 1e-6)
        pred_probs = pred_sum / weight_sum[None]

        # Argmax → label map
        pred_mask = pred_probs.argmax(axis=0).astype(np.uint16)

        # Crop back to original size
        pred_mask = pred_mask[:D, :H, :W]
        return pred_mask

    def _apply_tta(self, batch_t, model, device, base_probs):
        """Average predictions over mirror flips (TTA)."""
        all_probs = [base_probs]
        for axis in self.tta_axes:
            flipped = torch.flip(batch_t, dims=[axis + 2])  # +2 for [B,C,...]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits_f = model(flipped)
            if isinstance(logits_f, list):
                logits_f = logits_f[0]
            probs_f = torch.softmax(logits_f, dim=1)
            probs_f = torch.flip(probs_f, dims=[axis + 2]).cpu().numpy()
            all_probs.append(probs_f)
        return np.mean(all_probs, axis=0)

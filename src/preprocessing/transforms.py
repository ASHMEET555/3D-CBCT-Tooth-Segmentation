"""
src/preprocessing/transforms.py

3D augmentation transforms for CBCT training.
All transforms operate on numpy arrays of shape (D, H, W).

Applied only to images (not labels) unless specified.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    map_coordinates,
    rotate,
    zoom,
)


# ──────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────

class Transform3D:
    """Base class for 3D transforms. Override __call__."""
    def __call__(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# Intensity transforms (image only)
# ──────────────────────────────────────────────────────────────

class RandomGaussianNoise(Transform3D):
    """Add Gaussian noise to image."""
    def __init__(self, std: float = 0.01, p: float = 0.15):
        self.std = std
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            noise = np.random.randn(*image.shape).astype(np.float32) * self.std
            image = image + noise
        return image, label


class RandomGaussianBlur(Transform3D):
    """Apply Gaussian blur."""
    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 1.5), p: float = 0.2):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            image = gaussian_filter(image, sigma=sigma).astype(np.float32)
        return image, label


class RandomBrightnessContrast(Transform3D):
    """Random brightness and contrast jitter."""
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.75, 1.25),
        contrast_range: Tuple[float, float] = (0.75, 1.25),
        p: float = 0.3,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            mean = image.mean()
            alpha = random.uniform(*self.contrast_range)
            beta = random.uniform(*self.brightness_range)
            image = ((image - mean) * alpha + mean * beta).astype(np.float32)
        return image, label


class RandomGamma(Transform3D):
    """Gamma correction."""
    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            # Shift to [0, 1] for gamma, then back
            mn, mx = image.min(), image.max()
            if mx - mn > 1e-6:
                norm = (image - mn) / (mx - mn)
                gamma = random.uniform(*self.gamma_range)
                norm = np.power(norm, gamma)
                image = (norm * (mx - mn) + mn).astype(np.float32)
        return image, label


# ──────────────────────────────────────────────────────────────
# Spatial transforms (image + label)
# ──────────────────────────────────────────────────────────────

class RandomFlip(Transform3D):
    """Random axis-aligned flipping."""
    def __init__(self, axes: Tuple[int, ...] = (0, 1, 2), p: float = 0.5):
        self.axes = axes
        self.p = p

    def __call__(self, image, label=None):
        for ax in self.axes:
            if random.random() < self.p:
                image = np.flip(image, axis=ax).copy()
                if label is not None:
                    label = np.flip(label, axis=ax).copy()
        return image, label


class RandomRotation90(Transform3D):
    """Random 90° rotation in a random plane."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            image = np.rot90(image, k=k, axes=axes).copy()
            if label is not None:
                label = np.rot90(label, k=k, axes=axes).copy()
        return image, label


class RandomRotation(Transform3D):
    """Small-angle random rotation (slower but more realistic)."""
    def __init__(self, max_deg: float = 15.0, p: float = 0.3):
        self.max_deg = max_deg
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            angle = random.uniform(-self.max_deg, self.max_deg)
            ax1, ax2 = random.choice([(0, 1), (0, 2), (1, 2)])
            image = rotate(image, angle, axes=(ax1, ax2),
                           reshape=False, order=1, mode="constant",
                           cval=float(image.min())).astype(np.float32)
            if label is not None:
                label = rotate(label, angle, axes=(ax1, ax2),
                               reshape=False, order=0, mode="constant",
                               cval=0).astype(label.dtype)
        return image, label


class RandomScale(Transform3D):
    """Random isotropic scaling via zoom."""
    def __init__(self, scale_range: Tuple[float, float] = (0.85, 1.15), p: float = 0.3):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            scale = random.uniform(*self.scale_range)
            image = zoom(image, scale, order=1).astype(np.float32)
            if label is not None:
                label = zoom(label, scale, order=0).astype(label.dtype)
            # Re-crop or pad to original size
            image, label = _center_crop_or_pad(image, label, image.shape)
        return image, label


class RandomElasticDeformation(Transform3D):
    """
    3D elastic deformation using random displacement fields.
    Applied with probability p.
    """
    def __init__(
        self,
        sigma: float = 5.0,
        magnitude: float = 100.0,
        p: float = 0.2,
    ):
        self.sigma = sigma
        self.magnitude = magnitude
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            image, label = self._elastic_transform(image, label)
        return image, label

    def _elastic_transform(self, image, label):
        shape = image.shape
        # Random displacement fields
        fields = [
            gaussian_filter(
                np.random.randn(*shape) * self.magnitude,
                sigma=self.sigma,
            )
            for _ in range(3)
        ]
        coords = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        new_coords = [c + f for c, f in zip(coords, fields)]

        image = map_coordinates(image, new_coords, order=1, mode="constant",
                                cval=float(image.min())).astype(np.float32)
        if label is not None:
            label = map_coordinates(label, new_coords, order=0, mode="constant",
                                    cval=0).astype(label.dtype)
        return image, label


# ──────────────────────────────────────────────────────────────
# Compose
# ──────────────────────────────────────────────────────────────

class Compose(Transform3D):
    """Chain multiple transforms."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


# ──────────────────────────────────────────────────────────────
# Preset configs
# ──────────────────────────────────────────────────────────────

def get_training_transforms() -> Compose:
    return Compose([
        RandomFlip(axes=(0, 1, 2), p=0.5),
        RandomRotation90(p=0.5),
        RandomRotation(max_deg=15, p=0.3),
        RandomScale(scale_range=(0.85, 1.15), p=0.3),
        RandomElasticDeformation(sigma=5.0, magnitude=100.0, p=0.2),
        RandomGaussianNoise(std=0.01, p=0.15),
        RandomGaussianBlur(sigma_range=(0.5, 1.5), p=0.2),
        RandomBrightnessContrast(p=0.3),
        RandomGamma(p=0.3),
    ])


def get_val_transforms() -> None:
    """No augmentation for validation."""
    return None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _center_crop_or_pad(
    image: np.ndarray,
    label: Optional[np.ndarray],
    target_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Center-crop (if too large) or zero-pad (if too small)."""
    out_img = np.zeros(target_shape, dtype=image.dtype)
    out_lbl = np.zeros(target_shape, dtype=label.dtype) if label is not None else None

    # Compute crop/pad amounts
    src_slices, dst_slices = [], []
    for dim in range(3):
        src_size = image.shape[dim]
        dst_size = target_shape[dim]
        if src_size >= dst_size:
            start = (src_size - dst_size) // 2
            src_slices.append(slice(start, start + dst_size))
            dst_slices.append(slice(0, dst_size))
        else:
            start = (dst_size - src_size) // 2
            src_slices.append(slice(0, src_size))
            dst_slices.append(slice(start, start + src_size))

    s = tuple(src_slices)
    d = tuple(dst_slices)
    out_img[d] = image[s]
    if out_lbl is not None:
        out_lbl[d] = label[s]

    return out_img, out_lbl
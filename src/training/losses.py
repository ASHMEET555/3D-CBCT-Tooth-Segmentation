"""
src/training/losses.py

Loss functions for CBCT tooth segmentation:
  - DiceLoss        : multi-class soft Dice
  - DiceCELoss      : Dice + Cross-Entropy (nnU-Net default)
  - DeepSupervisionLoss : wrapper for multi-scale outputs
"""

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Soft Dice Loss for multi-class segmentation, ignoring background and optionally an ignore_index class.

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice Loss.

    Parameters
    num_classes : int
    smooth      : smoothing constant (Laplace)
    ignore_index: class index to ignore (-1 = no ignore)
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-5,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self,logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
        """
        logits  : [B, C, D, H, W]  (raw, un-softmaxed)
        targets : [B, D, H, W]      (integer class labels)
        """
        probs = F.softmax(logits, dim=1)  

        # One-hot encode targets
        B = targets.shape[0]
        one_hot = torch.zeros_like(probs)  
        valid = targets.clone()
        if self.ignore_index >= 0:
            valid = valid.masked_fill(valid == self.ignore_index, 0)
        one_hot.scatter_(1, valid.unsqueeze(1), 1.0)

        # Ignore background (class 0) in Dice
        dice_sum = 0.0
        n_classes = 0
        for c in range(1, self.num_classes):  # skip background
            if self.ignore_index == c:
                continue
            p = probs[:, c]     # [B, D, H, W]
            t = one_hot[:, c]
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice_sum += 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
            n_classes += 1

        return dice_sum / max(n_classes, 1)
    



# Dice + Cross-Entropy Loss (nnU-Net default)

class DiceCELoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss (nnU-Net standard).

    loss = w_dice * Dice + w_ce * CE
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.dice = DiceLoss(num_classes, smooth, ignore_index)
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index >= 0 else -100
        )
        self.w_dice = dice_weight
        self.w_ce = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets.long())
        return self.w_dice * dice_loss + self.w_ce * ce_loss
    
# Deep Supervision Wrapper


class DeepSupervisionLoss(nn.Module):
    """
    Wraps any segmentation loss to handle deep supervision outputs.

    At the main scale the full loss is applied.
    At lower scales the target is downsampled and a weighted fraction
    of the loss is added:
        total = Σ_s  weight_s * loss(pred_s, target_s)

    Default weights follow nnU-Net: [1/2, 1/4, 1/8, 1/16] normalised.
    """

    def __init__(
        self,
        criterion: nn.Module,
        scales: int = 4,
        weights: List[float] = None,
    ):
        super().__init__()
        self.criterion = criterion

        if weights is None:
            # Geometric series, normalised
            raw = [0.5 ** i for i in range(scales)]
            total = sum(raw)
            weights = [w / total for w in raw]
        assert len(weights) == scales
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.scales = scales

    def forward(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.criterion(outputs, targets)

        total_loss = torch.tensor(0.0, device=targets.device)
        for i, out in enumerate(outputs[: self.scales]):
            # Downsample target to match output resolution
            if out.shape[2:] != targets.shape[1:]:
                tgt = F.interpolate(
                    targets.float().unsqueeze(1),
                    size=out.shape[2:],
                    mode="nearest",
                ).squeeze(1).long()
            else:
                tgt = targets

            loss_i = self.criterion(out, tgt)
            total_loss = total_loss + self.weights[i] * loss_i

        return total_loss
    
#Factory function to build loss from config

def build_loss(config: dict, num_classes: int) -> nn.Module:
    """Build loss from config dict."""
    loss_cfg = config.get("loss", {})
    criterion = DiceCELoss(
        num_classes=num_classes,
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        smooth=loss_cfg.get("smooth", 1e-5),
        ignore_index=loss_cfg.get("ignore_index", -1),
    )
    if config.get("model", {}).get("deep_supervision", True):
        criterion = DeepSupervisionLoss(criterion, scales=4)
    return criterion


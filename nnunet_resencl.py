"""
src/models/nnunet_resencl.py

nnU-Net v2 with ResEnc-Large (ResEncL) backbone.
Based on: Isensee et al., "nnU-Net Revisited", MICCAI 2024.

Architecture:
  - Encoder: Residual blocks with large channel counts (ResEncL preset)
  - Skip connections to decoder
  - Deep supervision on all decoder scales
  - Instance normalization + LeakyReLU

Reference: https://github.com/MIC-DKFZ/nnUNet
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────

class ConvNormAct(nn.Module):
    """Conv3d → InstanceNorm3d → LeakyReLU."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block (He et al., 2016).
    Used in the ResEncL encoder.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

        # Downsample projection for residual when channels/stride differ
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out + residual)
        return out


class EncoderStage(nn.Module):
    """One encoder stage: initial stride conv + n residual blocks."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_blocks: int = 2,
        stride: int = 2,
    ):
        super().__init__()
        blocks = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(ResidualBlock(out_ch, out_ch, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class DecoderStage(nn.Module):
    """Upsample → concat skip → conv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvNormAct(out_ch + skip_ch, out_ch),
            ConvNormAct(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd input dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────
# ResEncL U-Net
# ──────────────────────────────────────────────────────────────

# ResEncL channel configuration (from nnU-Net paper Table 1)
RESENCL_CHANNELS = [32, 64, 128, 256, 512, 512]
RESENCL_BLOCKS   = [1,   3,   4,   6,   6,   2]


class ResEncLUNet(nn.Module):
    """
    nnU-Net ResEnc-Large architecture for 3D CBCT segmentation.

    Parameters
    ----------
    in_channels  : number of input channels (1 for CT)
    num_classes  : number of output classes (43 for ToothFairy2)
    deep_supervision : return multi-scale outputs during training
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 43,
        deep_supervision: bool = True,
        channels: List[int] = RESENCL_CHANNELS,
        blocks: List[int] = RESENCL_BLOCKS,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # ── Stem ──────────────────────────────────────────────
        self.stem = ConvNormAct(in_channels, channels[0])

        # ── Encoder ───────────────────────────────────────────
        self.enc_stages = nn.ModuleList()
        for i in range(1, len(channels)):
            self.enc_stages.append(
                EncoderStage(channels[i-1], channels[i],
                             n_blocks=blocks[i], stride=2)
            )

        # ── Decoder ───────────────────────────────────────────
        # Reverse channel list for decoder
        dec_in   = list(reversed(channels))       # [512,512,256,128,64,32]
        dec_skip = list(reversed(channels[:-1]))  # [512,256,128, 64,32]
        dec_out  = dec_skip                        # output matches skip size

        self.dec_stages = nn.ModuleList()
        for i in range(len(dec_out)):
            self.dec_stages.append(
                DecoderStage(dec_in[i], dec_skip[i], dec_out[i])
            )

        # ── Output heads ──────────────────────────────────────
        # Main head
        self.out_head = nn.Conv3d(channels[0], num_classes, kernel_size=1)

        # Deep supervision heads (for training only)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(ch, num_classes, kernel_size=1)
                for ch in dec_out[:-1]  # all decoder outputs except last
            ])

        self._init_weights()

    # ── Forward ─────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        x : [B, 1, D, H, W]
        Returns:
          - training + deep_supervision: list of logits at multiple scales
          - inference: [B, C, D, H, W] logits at full resolution
        """
        # Encoder
        skips = []
        out = self.stem(x)
        skips.append(out)

        for stage in self.enc_stages:
            out = stage(out)
            skips.append(out)

        # Bottleneck (last skip is bottleneck)
        out = skips.pop()  # bottleneck features

        # Decoder
        decoder_outs = []
        for i, stage in enumerate(self.dec_stages):
            skip = skips[-(i + 1)]
            out = stage(out, skip)
            decoder_outs.append(out)

        # Final output
        main_out = self.out_head(decoder_outs[-1])

        if self.deep_supervision and self.training:
            ds_outputs = [main_out]
            for i, head in enumerate(self.ds_heads):
                ds_outputs.append(head(decoder_outs[i]))
            return ds_outputs  # descending resolution

        return main_out

    # ── Utils ────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> ResEncLUNet:
    """Build model from config dict."""
    return ResEncLUNet(
        in_channels=config.get("in_channels", 1),
        num_classes=config.get("num_classes", 43),
        deep_supervision=config.get("deep_supervision", True),
    )


if __name__ == "__main__":
    # Quick sanity check
    model = ResEncLUNet(in_channels=1, num_classes=43, deep_supervision=True)
    n_params = model.count_parameters()
    print(f"ResEncL parameters: {n_params:,}")  # ~145M

    x = torch.randn(1, 1, 64, 64, 64)  # small for CPU test
    model.train()
    outs = model(x)
    if isinstance(outs, list):
        for i, o in enumerate(outs):
            print(f"  DS output {i}: {o.shape}")
    else:
        print(f"  Output: {outs.shape}")

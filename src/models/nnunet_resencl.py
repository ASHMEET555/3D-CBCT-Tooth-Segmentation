"""
src/models/nnunet_resencl.py
nnU-Net v2 ResEnc-Large backbone.
"""
from __future__ import annotations
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x):          # FIX: was nested inside __init__
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)
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
        return self.act(out + residual)

class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=2, stride=2):
        super().__init__()
        blocks = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(out_ch, out_ch, stride=1))
        self.stage = nn.Sequential(*blocks)
    def forward(self, x):
        return self.stage(x)         # FIX: was self.stage() missing x

class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)  # FIX: typo kernal_size
        self.conv = nn.Sequential(
            ConvNormAct(out_ch + skip_ch, out_ch),
            ConvNormAct(out_ch, out_ch),
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

RESENCL_CHANNELS = [32, 64, 128, 256, 512, 512]
RESENCL_BLOCKS   = [1,   3,   4,   6,   6,   2]

class ResEncLUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=43, deep_supervision=True,
                 channels=RESENCL_CHANNELS, blocks=RESENCL_BLOCKS):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.stem = ConvNormAct(in_channels, channels[0])
        self.enc_stages = nn.ModuleList()
        for i in range(1, len(channels)):
            self.enc_stages.append(
                EncoderStage(channels[i-1], channels[i], num_blocks=blocks[i], stride=2)
            )
        dec_in   = list(reversed(channels))
        dec_skip = list(reversed(channels[:-1]))
        dec_out  = dec_skip
        self.dec_stages = nn.ModuleList()
        for i in range(len(dec_out)):
            self.dec_stages.append(DecoderStage(dec_in[i], dec_skip[i], dec_out[i]))
        self.out_head = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(ch, num_classes, kernel_size=1) for ch in dec_out[:-1]
            ])
        self._init_weights()         # FIX: was self.__init_weights()

    def forward(self, x):            # FIX: was nested inside __init__ with bad indentation
        skips = []
        out = self.stem(x)
        skips.append(out)
        for stage in self.enc_stages:
            out = stage(out)
            skips.append(out)
        out = skips.pop()
        decoder_outs = []
        for i, stage in enumerate(self.dec_stages):
            skip = skips[-(i + 1)]
            out = stage(out, skip)
            decoder_outs.append(out)
        main_out = self.out_head(decoder_outs[-1])
        if self.deep_supervision and self.training:
            ds_outputs = [main_out]
            for i, head in enumerate(self.ds_heads):
                ds_outputs.append(head(decoder_outs[i]))
            return ds_outputs
        return main_out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(config):
    return ResEncLUNet(
        in_channels=config.get("in_channels", 1),
        num_classes=config.get("num_classes", 43),
        deep_supervision=config.get("deep_supervision", True),
    )

if __name__ == "__main__":
    model = ResEncLUNet(in_channels=1, num_classes=43, deep_supervision=True)
    print(f"ResEncL parameters: {model.count_parameters():,}")
    x = torch.randn(1, 1, 64, 64, 64)
    model.train()
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"  DS output {i}: {o.shape}")

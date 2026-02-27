"""Lightweight U-Net style decoder for NGIML feature fusion outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_norm(kind: str, channels: int) -> nn.Module:
    if kind.lower() == "bn":
        return nn.BatchNorm2d(channels)
    if kind.lower() == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported norm type: {kind}")


def _build_activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _build_norm(norm, out_channels),
            _build_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _build_norm(norm, out_channels),
            _build_activation(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


@dataclass
class UNetDecoderConfig:
    """Configuration for the U-Net style decoder.

    Forensic motivation: Use InstanceNorm by default to improve stability for forensic segmentation. Optionally inject edge-aware refinement for sharper boundaries. Optionally apply Dropout2d to the highest-res decoder output to regularize overfitting to spurious artifacts.
    """

    decoder_channels: Sequence[int] | None = None
    out_channels: int = 1
    norm: str = "in"  # Default to InstanceNorm
    activation: str = "relu"
    per_stage_heads: bool = True
    enable_edge_guidance: bool = True  # Edge-aware decoder refinement (enabled by default)
    use_dropout: bool = True  # Dropout2d in highest-res decoder output enabled by default
    dropout_p: float = 0.2


class UNetDecoder(nn.Module):
    """U-Net decoder that upsamples fused features into manipulation logits.

    Forensic motivation: Optionally injects Sobel edge map into highest-resolution decoder feature for improved boundary localization.
    """

    def __init__(self, stage_channels: Sequence[int], config: UNetDecoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or UNetDecoderConfig()
        self.use_dropout = getattr(self.cfg, 'use_dropout', False)
        self.dropout_p = getattr(self.cfg, 'dropout_p', 0.2)
        if self.use_dropout:
            self.dropout = nn.Dropout2d(self.dropout_p)
        if not stage_channels:
            raise ValueError("stage_channels must contain at least one entry")
        self.stage_channels = tuple(stage_channels)

        if self.cfg.decoder_channels is None:
            decoder_channels = self.stage_channels
        else:
            if len(self.cfg.decoder_channels) != len(self.stage_channels):
                raise ValueError("decoder_channels length must match number of fusion stages")
            decoder_channels = tuple(self.cfg.decoder_channels)
        self.decoder_channels = tuple(decoder_channels)

        # Edge-aware decoder refinement (optional)
        self.enable_edge_guidance = getattr(self.cfg, 'enable_edge_guidance', False)
        if self.enable_edge_guidance:
            # Project Sobel edge map to decoder feature channels
            self.edge_proj = nn.Sequential(
                nn.Conv2d(1, self.decoder_channels[0], kernel_size=3, padding=1, bias=False),
                _build_norm(self.cfg.norm, self.decoder_channels[0]),
                _build_activation(self.cfg.activation),
            )
            # Sobel kernels
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

        self.skip_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, dec_ch, kernel_size=1, bias=False),
                    _build_norm(self.cfg.norm, dec_ch),
                    _build_activation(self.cfg.activation),
                )
                for in_ch, dec_ch in zip(self.stage_channels, self.decoder_channels)
            ]
        )

        self.bottleneck = _ConvBlock(
            self.decoder_channels[-1],
            self.decoder_channels[-1],
            self.cfg.norm,
            self.cfg.activation,
        )

        self.decode_blocks = nn.ModuleList(
            [
                _ConvBlock(
                    self.decoder_channels[idx] + self.decoder_channels[idx + 1],
                    self.decoder_channels[idx],
                    self.cfg.norm,
                    self.cfg.activation,
                )
                for idx in range(len(self.stage_channels) - 1)
            ]
        )

        self.predictors = nn.ModuleList(
            [
                nn.Conv2d(channels, self.cfg.out_channels, kernel_size=1)
                for channels in self.decoder_channels
            ]
        )

    def forward(self, features: List[Tensor], image: Tensor = None) -> List[Tensor]:
        if len(features) != len(self.stage_channels):
            raise ValueError("Feature list length must match number of decoder stages")

        projected = [proj(feat) for proj, feat in zip(self.skip_projections, features)]

        # Edge-aware refinement: inject projected Sobel edge map into highest-res decoder feature
        if self.enable_edge_guidance and image is not None:
            # Compute grayscale edge map
            with torch.no_grad():
                if image.shape[1] > 1:
                    gray = image.mean(dim=1, keepdim=True)
                else:
                    gray = image
                grad_x = F.conv2d(gray, self.sobel_x, padding=1)
                grad_y = F.conv2d(gray, self.sobel_y, padding=1)
                edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            edge_proj = self.edge_proj(edge_mag)
            projected[0] = projected[0] + edge_proj

        x = self.bottleneck(projected[-1])

        if self.cfg.per_stage_heads:
            predictions: List[Optional[Tensor]] = [None] * len(projected)
            predictions[-1] = self.predictors[-1](x)
        else:
            predictions = []

        for idx in range(len(projected) - 2, -1, -1):
            skip = projected[idx]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.decode_blocks[idx](x)
            if self.cfg.per_stage_heads:
                predictions[idx] = self.predictors[idx](x)

        if self.cfg.per_stage_heads:
            # Optionally apply dropout to highest-res output
            out_preds = [pred for pred in predictions if pred is not None]
            if self.use_dropout and out_preds:
                out_preds[-1] = self.dropout(out_preds[-1])
            return out_preds

        final = self.predictors[0](x)
        if self.use_dropout:
            final = self.dropout(final)
        return [final]


__all__ = ["UNetDecoder", "UNetDecoderConfig"]

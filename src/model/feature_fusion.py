"""Adaptive multi-branch feature fusion for NGIML."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_norm(norm: str, channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported norm type: {norm}")


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class FeatureFusionConfig:
    """Config container for the multi-stage fusion module.

    Forensic motivation: Optionally adds a spatial refinement layer after fusion output for each stage, improving spatial detail without large FLOP increase.
    """

    fusion_channels: Sequence[int]
    noise_branch: str = "residual"
    noise_skip_stage: Optional[int] = None
    noise_decay: float = 1.0
    norm: str = "bn"
    activation: str = "relu"
    fusion_refinement: bool = True  # Add Conv3x3+IN+ReLU after fusion output (enabled by default)


class _AdaptiveFusionStage(nn.Module):
    """Stage-wise fusion with learned gating and post refinement.

    Forensic motivation: Optionally adds a spatial refinement layer after fusion output for each stage, improving spatial detail without large FLOP increase.
    """
    def __init__(
        self,
        branch_channels: Dict[str, int],
        out_channels: int,
        norm: str,
        activation: str,
        fusion_refinement: bool = False,
    ) -> None:
        super().__init__()
        # Conv only for projected features before fusion (no norm/activation)
        self.projections = nn.ModuleDict(
            {
                branch: nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
                for branch, in_ch in branch_channels.items()
            }
        )
        num_branches = len(branch_channels)
        # Initialize fusion gates equally across branches
        self.gate_params = nn.ParameterDict({
            branch: nn.Parameter(torch.full((1, 1, 1, 1), 1.0 / num_branches))
            for branch in branch_channels
        })
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _build_norm(norm, out_channels),
            _build_activation(activation),
        )
        self.fusion_refinement = fusion_refinement
        if self.fusion_refinement:
            self.refine2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )

    def forward(
        self,
        features: Dict[str, Tensor],
        target_size: Optional[Tuple[int, int]],
        noise_branch: Optional[str],
        noise_weight: float,
    ) -> Tensor:
        if not features:
            raise ValueError("Fusion stage received no features to fuse")

        # Determine alignment size: provided target overrides per-stage maximum.
        if target_size is not None:
            align_h, align_w = target_size
        else:
            align_h = max(x.shape[-2] for x in features.values())
            align_w = max(x.shape[-1] for x in features.values())

        fused = 0.0
        weight_sum = 0.0
        eps = 1e-6

        for branch, tensor in features.items():
            proj = self.projections[branch](tensor)
            if proj.shape[-2:] != (align_h, align_w):
                proj = F.interpolate(proj, size=(align_h, align_w), mode="bilinear", align_corners=False)

            # Bounded sigmoid gating: gate = sigmoid(param) * 0.8 + 0.1
            raw_gate = self.gate_params[branch]
            gate = torch.sigmoid(raw_gate) * 0.8 + 0.1
            # Broadcast gate to proj shape
            gate = gate.expand_as(proj)
            if noise_branch is not None and branch == noise_branch:
                gate = gate * noise_weight

            fused = fused + proj * gate
            weight_sum = weight_sum + gate

        fused = fused / (weight_sum + eps)
        fused = self.refine(fused)
        if self.fusion_refinement:
            fused = self.refine2(fused)
        return fused


class MultiStageFeatureFusion(nn.Module):
    """Fuses multi-branch features across stages with adaptive gating.

    Forensic motivation: Optionally adds a spatial refinement layer after fusion output for each stage, improving spatial detail without large FLOP increase.
    """
    def __init__(
        self,
        branch_channels: Dict[str, Sequence[int]],
        config: FeatureFusionConfig,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.branches = list(branch_channels.keys())

        num_stages = len(config.fusion_channels)
        self.stages = nn.ModuleList()
        for stage_idx in range(num_stages):
            stage_branch_channels: Dict[str, int] = {}
            for branch, channels in branch_channels.items():
                if stage_idx < len(channels):
                    stage_branch_channels[branch] = channels[stage_idx]
            if not stage_branch_channels:
                raise ValueError(f"No branch provides features for stage {stage_idx}")
            self.stages.append(
                _AdaptiveFusionStage(
                    stage_branch_channels,
                    config.fusion_channels[stage_idx],
                    norm=config.norm,
                    activation=config.activation,
                    fusion_refinement=getattr(config, 'fusion_refinement', False),
                )
            )

    def _noise_weight(self, stage_idx: int) -> float:
        skip = self.cfg.noise_skip_stage
        if skip is not None and stage_idx >= skip:
            return 0.0
        decay = max(self.cfg.noise_decay, 0.0)
        return decay ** stage_idx

    def forward(
        self,
        features: Dict[str, List[Tensor]],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tensor]:
        fused: List[Tensor] = []
        for stage_idx, stage in enumerate(self.stages):
            stage_inputs: Dict[str, Tensor] = {}
            for branch in self.branches:
                branch_feats = features.get(branch, [])
                if stage_idx < len(branch_feats):
                    stage_inputs[branch] = branch_feats[stage_idx]
            if not stage_inputs:
                continue

            fused.append(
                stage(
                    stage_inputs,
                    target_size=target_size,
                    noise_branch=self.cfg.noise_branch,
                    noise_weight=self._noise_weight(stage_idx),
                )
            )
        return fused


__all__ = ["FeatureFusionConfig", "MultiStageFeatureFusion"]

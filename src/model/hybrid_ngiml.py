"""Hybrid NGIML model that fuses CNN, Transformer, and noise cues."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from .backbones.efficientnet_backbone import EfficientNetBackbone, EfficientNetBackboneConfig
from .backbones.residual_noise_branch import ResidualNoiseBranch, ResidualNoiseConfig
from .backbones.swin_backbone import SwinBackbone, SwinBackboneConfig
from .feature_fusion import FeatureFusionConfig, MultiStageFeatureFusion
from .unet_decoder import UNetDecoder, UNetDecoderConfig


@dataclass
class OptimizerGroupConfig:
    """Learning rate / weight decay pair for an optimizer parameter group."""
    lr: float
    weight_decay: float = 1e-5


def _default_efficientnet_optim() -> OptimizerGroupConfig:
    # Forensic motivation: Lower LR for backbone to stabilize early training
    return OptimizerGroupConfig(lr=1e-5, weight_decay=1e-4)


def _default_swin_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=5e-6, weight_decay=5e-5)


def _default_residual_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=3e-4, weight_decay=1e-4)


def _default_fusion_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=1.5e-4, weight_decay=1e-4)


def _default_decoder_optim() -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=2e-4, weight_decay=1e-4)


@dataclass
class HybridNGIMLOptimizerConfig:
    """Optimizer hyper-parameters separated per backbone/fusion branch.

    Forensic motivation: Lower backbone LR, higher forensic/fusion/decoder LRs, and support freezing backbone for early epochs.
    """
    efficientnet: OptimizerGroupConfig = field(default_factory=_default_efficientnet_optim)
    swin: OptimizerGroupConfig = field(default_factory=_default_swin_optim)
    residual: OptimizerGroupConfig = field(default_factory=_default_residual_optim)
    fusion: OptimizerGroupConfig = field(default_factory=_default_fusion_optim)
    decoder: OptimizerGroupConfig = field(default_factory=_default_decoder_optim)
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    freeze_backbone_epochs: int = 5


@dataclass
class HybridNGIMLConfig:
    """Aggregated configuration for the hybrid NGIML model."""

    efficientnet: EfficientNetBackboneConfig = field(default_factory=EfficientNetBackboneConfig)
    swin: SwinBackboneConfig = field(default_factory=SwinBackboneConfig)
    residual: ResidualNoiseConfig = field(default_factory=ResidualNoiseConfig)
    fusion: FeatureFusionConfig = field(
        default_factory=lambda: FeatureFusionConfig(fusion_channels=(64, 128, 192, 256))
    )
    decoder: UNetDecoderConfig = field(default_factory=UNetDecoderConfig)
    optimizer: HybridNGIMLOptimizerConfig = field(default_factory=HybridNGIMLOptimizerConfig)
    use_low_level: bool = True
    use_context: bool = True
    use_residual: bool = True
    enable_residual_attention: bool = True  # Residual-guided attention (enabled by default)
    gradient_checkpointing: bool = True  # Enable gradient checkpointing for memory savings
    flash_attention: bool = True  # Enable flash attention by default
    xformers: bool = True  # Enable xformers by default


class HybridNGIML(nn.Module):
    """Full NGIML model exposing fused multi-scale features.

    Forensic motivation: Optionally applies residual-guided attention to semantic features before fusion, improving manipulation localization.
    """

    def __init__(self, config: HybridNGIMLConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or HybridNGIMLConfig()
        self.efficientnet = EfficientNetBackbone(self.cfg.efficientnet)
        # Pass flash_attention and xformers flags if present in config
        swin_kwargs = {}
        if hasattr(self.cfg, 'flash_attention'):
            swin_kwargs['flash_attention'] = getattr(self.cfg, 'flash_attention', False)
        if hasattr(self.cfg, 'xformers'):
            swin_kwargs['xformers'] = getattr(self.cfg, 'xformers', False)
        self.swin = SwinBackbone(self.cfg.swin, **swin_kwargs)
        self.noise = ResidualNoiseBranch(self.cfg.residual)

        layout = {
            "low_level": self.efficientnet.out_channels,
            "context": self.swin.out_channels,
            "residual": self.noise.out_channels,
        }
        self.num_stages = len(self.cfg.fusion.fusion_channels)
        branch_channels: Dict[str, List[int]] = {}

        if self.cfg.use_low_level:
            branch_channels["low_level"] = layout["low_level"]
        if self.cfg.use_context:
            branch_channels["context"] = layout["context"]
        if self.cfg.use_residual:
            residual_channels = layout.get("residual", [3])
            if len(residual_channels) == 1:
                residual_channels = residual_channels * self.num_stages
            branch_channels["residual"] = residual_channels

        if not branch_channels:
            raise ValueError("At least one backbone branch must be enabled for fusion")

        self.fusion = MultiStageFeatureFusion(branch_channels, self.cfg.fusion)
        self.decoder = UNetDecoder(self.cfg.fusion.fusion_channels, self.cfg.decoder)

        # Residual-guided attention module (optional)
        self.enable_residual_attention = getattr(self.cfg, 'enable_residual_attention', False)
        if self.enable_residual_attention:
            # Project residual features to attention map (per stage)
            res_channels = branch_channels.get("residual", [0])
            sem_channels = branch_channels.get("low_level", [0])
            # Use highest-resolution features for attention
            attn_in_ch = res_channels[0] if res_channels else 0
            attn_out_ch = sem_channels[0] if sem_channels else 0
            self.residual_attention_proj = nn.Conv2d(attn_in_ch, attn_out_ch, kernel_size=1)
            nn.init.zeros_(self.residual_attention_proj.weight)
            if self.residual_attention_proj.bias is not None:
                nn.init.zeros_(self.residual_attention_proj.bias)

    def _extract_features(self, x: Tensor, high_pass: Tensor | None = None) -> Dict[str, List[Tensor] | Tensor]:
        low_level = self.efficientnet(x)
        context = self.swin(x)
        residual = self.noise(x, high_pass=high_pass)

        # Residual-guided attention (modulate semantic features before fusion)
        if self.enable_residual_attention and isinstance(low_level, list) and isinstance(residual, list):
            # Use highest-resolution features (stage 0)
            attn_map = torch.sigmoid(self.residual_attention_proj(residual[0]))
            # Upsample attention map to match semantic feature spatial dims if needed
            sem_h, sem_w = low_level[0].shape[-2:]
            if attn_map.shape[-2:] != (sem_h, sem_w):
                attn_map = F.interpolate(attn_map, size=(sem_h, sem_w), mode="bilinear", align_corners=False)
            # Modulate semantic features: semantic_feat = semantic_feat * (1 + attention)
            low_level[0] = low_level[0] * (1.0 + attn_map)

        # Gradient checkpointing for memory savings
        if getattr(self.cfg, 'gradient_checkpointing', False):
            def checkpointed_forward(module, *inputs):
                def custom_forward(*inputs):
                    return module(*inputs)
                return torch.utils.checkpoint.checkpoint(custom_forward, *inputs)
            # Only checkpoint backbone blocks if possible
            if hasattr(self.efficientnet, 'backbone'):
                low_level = [checkpointed_forward(m, x) for m in self.efficientnet.backbone.children()]
            if hasattr(self.swin, 'model'):
                context = [checkpointed_forward(m, x) for m in self.swin.model.children()]
            if hasattr(self.noise, 'blocks'):
                residual = [checkpointed_forward(m, x) for m in self.noise.blocks]

        return {
            "low_level": low_level,
            "context": context,
            "residual": residual,
        }

    def forward_features(
        self,
        x: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        high_pass: Tensor | None = None,
    ) -> List[Tensor]:
        backbone_feats = self._extract_features(x, high_pass=high_pass)
        fusion_inputs = {}
        if self.cfg.use_low_level:
            fusion_inputs["low_level"] = backbone_feats["low_level"]
        if self.cfg.use_context:
            fusion_inputs["context"] = backbone_feats["context"]
        if self.cfg.use_residual:
            fusion_inputs["residual"] = backbone_feats["residual"]
        return self.fusion(fusion_inputs, target_size=None)

    def forward(
        self,
        x: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        high_pass: Tensor | None = None,
    ) -> List[Tensor]:
        fused = self.forward_features(x, target_size=None, high_pass=high_pass)
        preds = self.decoder(fused)
        if target_size is None:
            return preds
        return [
            F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
            if pred.shape[-2:] != target_size
            else pred
            for pred in preds
        ]

    def optimizer_parameter_groups(self) -> List[Dict[str, object]]:
        """Return AdamW-ready parameter groups with branch-specific LRs/decays."""

        groups: List[Dict[str, object]] = []

        def _append(params, group_cfg: OptimizerGroupConfig) -> None:
            param_list = list(params)
            if not param_list:
                return
            groups.append({
                "params": param_list,
                "lr": group_cfg.lr,
                "weight_decay": group_cfg.weight_decay,
            })

        if self.cfg.use_low_level:
            _append(self.efficientnet.parameters(), self.cfg.optimizer.efficientnet)
        if self.cfg.use_context:
            _append(self.swin.parameters(), self.cfg.optimizer.swin)
        if self.cfg.use_residual:
            _append(self.noise.parameters(), self.cfg.optimizer.residual)

        _append(self.fusion.parameters(), self.cfg.optimizer.fusion)
        _append(self.decoder.parameters(), self.cfg.optimizer.decoder)

        if not groups:
            raise ValueError("No parameter groups available for optimization")

        return groups

    def build_optimizer(self) -> AdamW:
        """Instantiate an AdamW optimizer using the configured parameter groups."""

        param_groups = self.optimizer_parameter_groups()
        return AdamW(param_groups, betas=self.cfg.optimizer.betas, eps=self.cfg.optimizer.eps)


__all__ = [
    "HybridNGIML",
    "HybridNGIMLConfig",
    "HybridNGIMLOptimizerConfig",
    "OptimizerGroupConfig",
    "UNetDecoderConfig",
]

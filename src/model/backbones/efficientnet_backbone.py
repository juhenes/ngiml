"""EfficientNet backbone for NGIML low-level feature extraction (timm-based).

Forensic motivation: timm EfficientNet provides more stable pretrained weights and better intermediate feature extraction for manipulation localization tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Tuple, Union

import logging
import torch
from torch import nn
import torch.nn.functional as F
import timm

_LOG = logging.getLogger(__name__)
# Reduce noisy pretrained-weight mismatch warnings from timm internals
logging.getLogger("timm.models._builder").setLevel(logging.ERROR)


@dataclass
class EfficientNetBackboneConfig:
    """Configuration container for EfficientNet backbone."""

    pretrained: bool = True
    # Default to common EfficientNet feature indices for B0-like variants.
    out_indices: Sequence[int] = (1, 2, 3, 4)
    enforce_input_size: bool = False
    input_size: Union[int, Tuple[int, int], None] = None



class EfficientNetBackbone(nn.Module):
    """Wrapper that exposes multi-scale EfficientNet feature maps using timm.

    Forensic motivation: Use timm EfficientNet for more stable pretrained weights and better feature extraction for manipulation localization.
    """
    def __init__(self, config: EfficientNetBackboneConfig | None = None) -> None:
        super().__init__()
        cfg = config or EfficientNetBackboneConfig()

        self.out_indices: Tuple[int, ...] = tuple(sorted(set(cfg.out_indices)))
        self.enforce_input_size = cfg.enforce_input_size

        if cfg.input_size is not None:
            if isinstance(cfg.input_size, int):
                self.expected_hw = (cfg.input_size, cfg.input_size)
            else:
                self.expected_hw = tuple(cfg.input_size)
        else:
            self.expected_hw = (224, 224)  # default EfficientNet input

        # Use timm to create EfficientNet backbone without forcing out_indices.
        # We'll select the requested feature maps from the returned list to avoid timm internal index mismatches.
        model_name = getattr(cfg, 'model_name', 'efficientnet_b0')
        self.backbone = timm.create_model(model_name, pretrained=cfg.pretrained, features_only=True)
        avail_n = len(self.backbone.feature_info)
        requested = tuple(sorted(set(self.out_indices)))
        valid_indices = tuple(i for i in requested if 0 <= i < avail_n)
        if not valid_indices:
            valid_indices = tuple(range(avail_n))
        if valid_indices != tuple(requested):
            # Log at INFO to avoid alarming warnings for auto-adjustments; this
            # is non-fatal and many timm variants expose slightly different
            # feature index ranges.
            _LOG.info(
                "requested efficientnet out_indices %s adjusted to available indices %s for model %s",
                requested,
                valid_indices,
                model_name,
            )
        self.selected_indices = valid_indices
        # Cache channel dimensions for downstream heads corresponding to selected indices
        self.out_channels: List[int] = [self.backbone.feature_info[i]['num_chs'] for i in self.selected_indices]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale feature maps."""
        if self.enforce_input_size and x.shape[-2:] != self.expected_hw:
            _LOG.warning(
                "EfficientNetBackbone internal resize (enforce_input_size=True): (%d,%d) -> (%d,%d)",
                x.shape[-2], x.shape[-1], self.expected_hw[0], self.expected_hw[1],
            )
            x = F.interpolate(x, size=self.expected_hw, mode="bilinear", align_corners=False)
        # Guard timm model `out_indices` attribute to avoid internal index errors
        if hasattr(self.backbone, "feature_info") and hasattr(self.backbone, "out_indices"):
            avail = len(self.backbone.feature_info)
            safe_out = tuple(i for i in self.selected_indices if 0 <= i < avail)
            if not safe_out:
                safe_out = tuple(range(avail))
            try:
                self.backbone.out_indices = safe_out
            except Exception:
                pass

        try:
            features = self.backbone(x)
        except AssertionError as err:
            msg = str(err)
            _LOG.warning("EfficientNet model assertion during forward: %s", msg)
            default_cfg = getattr(self.backbone, "default_cfg", None) or getattr(self.backbone, "default_cfg", {})
            input_size = default_cfg.get("input_size") if isinstance(default_cfg, dict) else None
            if input_size:
                try:
                    exp_h, exp_w = input_size[1], input_size[2]
                    _LOG.warning(
                        "Resizing input from (%d,%d) to model default (%d,%d) to recover from assertion",
                        x.shape[-2], x.shape[-1], exp_h, exp_w,
                    )
                    x_resized = F.interpolate(x, size=(exp_h, exp_w), mode="bilinear", align_corners=False)
                    features = self.backbone(x_resized)
                except Exception:
                    raise
            else:
                raise
        # Select only the requested feature maps and return as list
        if isinstance(features, (list, tuple)):
            selected = [features[i] for i in self.selected_indices]
            return selected
        return [features]


__all__ = ["EfficientNetBackbone", "EfficientNetBackboneConfig"]

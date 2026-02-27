"""Swin-Tiny backbone for NGIML contextual feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import logging
import timm
import torch
import torch.nn.functional as NN_F
from torch import nn, Tensor

_LOG = logging.getLogger(__name__)
# Reduce noisy pretrained-weight mismatch warnings from timm internals
logging.getLogger("timm.models._builder").setLevel(logging.ERROR)


@dataclass
class SwinBackboneConfig:
    """Configuration for the Swin Transformer feature extractor."""

    model_name: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = True
    out_indices: Sequence[int] = (0, 1, 2, 3)


class SwinBackbone(nn.Module):
    """Thin wrapper around timm Swin Transformer with multi-scale outputs."""

    def __init__(self, config: SwinBackboneConfig | None = None, flash_attention: bool = False, xformers: bool = False) -> None:
        super().__init__()
        cfg = config or SwinBackboneConfig()
        # Create model without forcing out_indices first, then clamp requested indices
        # to the model's available feature levels and recreate with valid indices.
        self.model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, features_only=True)
        avail_n = len(self.model.feature_info)
        requested = tuple(cfg.out_indices) if cfg.out_indices is not None else tuple(range(avail_n))
        valid_indices = tuple(i for i in requested if 0 <= i < avail_n)
        if not valid_indices:
            valid_indices = tuple(range(avail_n))
        if valid_indices != tuple(requested):
            _LOG.warning(
                "requested swin out_indices %s adjusted to available indices %s for model %s",
                requested,
                valid_indices,
                cfg.model_name,
            )
        # Use model without forcing out_indices to avoid timm internal mismatches.
        # We'll select the desired feature maps from the returned feature list.
        self.selected_indices = valid_indices
        self.out_channels: List[int] = [self.model.feature_info[i]["num_chs"] for i in self.selected_indices]
        patch = getattr(self.model, "patch_embed", None)
        if patch is None:
            raise ValueError("Swin backbone missing patch_embed; ensure model_name is a Swin variant")
        self.patch_embed = patch
        if isinstance(self.patch_embed.patch_size, tuple):
            self.patch_size: Tuple[int, int] = self.patch_embed.patch_size
        else:
            self.patch_size = (self.patch_embed.patch_size, self.patch_embed.patch_size)
        self.stages: List[nn.Module] = [
            module
            for name, module in self.model.named_children()
            if name.startswith("layers_")
        ]
        if not self.stages:
            raise ValueError("Swin backbone structure unexpected; layers_* modules not found")
        self._last_spatial_size: Tuple[int, int] | None = None

        # Flash attention and xformers hooks
        self.flash_attention = flash_attention
        self.xformers = xformers
        if self.flash_attention:
            try:
                import flash_attn  # type: ignore
                # Insert flash attention logic here if needed
            except ImportError:
                _LOG.info("flash-attn not installed; flash attention will not be used.")
        if self.xformers:
            try:
                import xformers  # type: ignore
                # Insert xformers logic here if needed
            except ImportError:
                _LOG.info("xformers not installed; xformers attention will not be used.")

    def _propagate_spatial_metadata(self, height: int, width: int) -> None:
        # Accept non-multiple spatial dims by adjusting to the next multiple of patch size.
        ph, pw = self.patch_size
        if height % ph != 0 or width % pw != 0:
            new_h = ((height + ph - 1) // ph) * ph
            new_w = ((width + pw - 1) // pw) * pw
            _LOG.warning(
                "Swin input spatial dims (%d,%d) are not multiples of patch size %s; adjusting to (%d,%d)",
                height,
                width,
                self.patch_size,
                new_h,
                new_w,
            )
            height, width = new_h, new_w

        if self._last_spatial_size == (height, width):
            return

        grid_h = height // self.patch_size[0]
        grid_w = width // self.patch_size[1]

        if (height, width) != self.patch_embed.img_size:
            self.patch_embed.img_size = (height, width)
            self.patch_embed.grid_size = (grid_h, grid_w)
            self.patch_embed.num_patches = grid_h * grid_w

        for stage_idx, stage in enumerate(self.stages):
            scale = 2 ** stage_idx
            stage_res = (grid_h // scale, grid_w // scale)
            stage.input_resolution = stage_res
            blocks = getattr(stage, "blocks", [])
            for block in blocks:
                block.input_resolution = stage_res
                if hasattr(block, "attn_mask"):
                    device = None
                    dtype = None
                    if isinstance(block.attn_mask, torch.Tensor):
                        device = block.attn_mask.device
                        dtype = block.attn_mask.dtype
                    block.attn_mask = block.get_attn_mask(device=device, dtype=dtype)

        self._last_spatial_size = (height, width)

    def _ensure_channels_first(self, features: List[Tensor]) -> List[Tensor]:
        if len(features) != len(self.out_channels):
            raise ValueError(
                "Unexpected number of Swin feature maps; review out_indices configuration"
            )

        normalized: List[Tensor] = []
        for idx, (feat, expected_ch) in enumerate(zip(features, self.out_channels)):
            if feat.ndim != 4:
                raise ValueError(
                    f"Swin feature map {idx} must be 4D (NCHW), got shape {tuple(feat.shape)}"
                )

            if feat.shape[1] == expected_ch:
                normalized.append(feat)
                continue

            if feat.shape[-1] == expected_ch:
                normalized.append(feat.permute(0, 3, 1, 2).contiguous())
                continue

            raise ValueError(
                f"Swin feature map {idx} reports {feat.shape[1]} channels but expected {expected_ch}"
            )

        return normalized

    def forward(self, x: Tensor) -> List[Tensor]:
        # Ensure model is on the same device as input to avoid device/type mismatch
        try:
            first_param = next(self.model.parameters())
            model_dev = first_param.device
        except StopIteration:
            model_dev = None
        if model_dev is not None and model_dev != x.device:
            _LOG.info("Moving SwinBackbone model from %s to %s", model_dev, x.device)
            # Move the full module so all parameters/buffers align
            self.to(x.device)

        # Pad input so spatial dimensions are multiples of the Swin patch size
        _, _, h, w = x.shape
        ph, pw = self.patch_size
        pad_h = (ph - (h % ph)) % ph
        pad_w = (pw - (w % pw)) % pw
        if pad_h or pad_w:
            x = NN_F.pad(x, (0, pad_w, 0, pad_h), value=0)

        self._propagate_spatial_metadata(x.shape[-2], x.shape[-1])
        # Guard timm model internal `out_indices` against invalid values (some timm versions)
        if hasattr(self.model, "feature_info") and hasattr(self.model, "out_indices"):
            avail = len(self.model.feature_info)
            safe_out = tuple(i for i in self.selected_indices if 0 <= i < avail)
            if not safe_out:
                safe_out = tuple(range(avail))
            try:
                self.model.out_indices = safe_out
            except Exception:
                pass

        features = self.model(x)
        # Select only the requested feature maps
        selected = [features[i] for i in self.selected_indices]
        return self._ensure_channels_first(selected)


__all__ = ["SwinBackbone", "SwinBackboneConfig"]

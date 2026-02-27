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

        # Do not forcibly override patch_embed.img_size/grid_size/num_patches here; leave
        # timm internals to manage those attributes to avoid creating inconsistent
        # attention masks. We only propagate stage-level input resolutions below.

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
        # Call timm model and select requested feature maps from returned list.
        try:
            features = self.model(x)
        except AssertionError as err:
            msg = str(err)
            # Some timm variants assert on an expected input size. If that occurs,
            # try padding to the model default aspect ratio then resize to exact
            # expected size to recover from the assertion.
            _LOG.warning("Swin model assertion during forward: %s", msg)
            default_cfg = getattr(self.model, "default_cfg", None) or {}
            input_size = default_cfg.get("input_size") if isinstance(default_cfg, dict) else None
            try:
                if input_size:
                    exp_h, exp_w = int(input_size[1]), int(input_size[2])
                    h_cur, w_cur = x.shape[-2], x.shape[-1]
                    # scale factor to ensure new_h >= h_cur and new_w >= w_cur
                    s = max(h_cur / exp_h, w_cur / exp_w, 1.0)
                    new_h = int((exp_h * s + 0.9999))
                    new_w = int((exp_w * s + 0.9999))
                    pad_h = max(0, new_h - h_cur)
                    pad_w = max(0, new_w - w_cur)
                    if pad_h or pad_w:
                        _LOG.warning(
                            "Padding input from (%d,%d) by (h=%d,w=%d) to match model aspect ratio before resize",
                            h_cur,
                            w_cur,
                            pad_h,
                            pad_w,
                        )
                        x_padded = NN_F.pad(x, (0, pad_w, 0, pad_h), value=0)
                    else:
                        x_padded = x
                    # Propagate spatial metadata for the padded size
                    self._propagate_spatial_metadata(x_padded.shape[-2], x_padded.shape[-1])
                    # Resize to exact expected model input size
                    x_resized = NN_F.interpolate(x_padded, size=(exp_h, exp_w), mode="bilinear", align_corners=False)
                    features = self.model(x_resized)
                else:
                    # No default input size available; fall back to patch-multiple padding
                    h_cur, w_cur = x.shape[-2], x.shape[-1]
                    ph, pw = self.patch_size
                    pad_h = ((h_cur + ph - 1) // ph) * ph - h_cur
                    pad_w = ((w_cur + pw - 1) // pw) * pw - w_cur
                    if pad_h or pad_w:
                        _LOG.warning(
                            "Padding input from (%d,%d) by (h=%d,w=%d) to match patch multiple and preserve resolution",
                            h_cur,
                            w_cur,
                            pad_h,
                            pad_w,
                        )
                        x_padded = NN_F.pad(x, (0, pad_w, 0, pad_h), value=0)
                        self._propagate_spatial_metadata(x_padded.shape[-2], x_padded.shape[-1])
                        features = self.model(x_padded)
                    else:
                        # Already a multiple and still failing; re-raise the original assertion.
                        raise err
            except AssertionError:
                # If padding/resize didn't help (e.g., model insists on a specific size),
                # re-raise the original assertion to surface a clear error.
                raise err
        # Select only the requested feature maps
        selected = [features[i] for i in self.selected_indices]
        return self._ensure_channels_first(selected)


__all__ = ["SwinBackbone", "SwinBackboneConfig"]

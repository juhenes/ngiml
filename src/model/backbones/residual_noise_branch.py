from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


@dataclass
class ResidualNoiseConfig:
    """Configuration for the SRM residual branch / multi-scale backbone."""
    num_kernels: int = 3  # SRM kernels (fixed)
    base_channels: int = 32  # CNN backbone base channels
    num_stages: int = 4      # Number of pyramid stages


class ConvBlock(nn.Module):
    """Basic 2-layer conv + norm + ReLU block.

    Forensic motivation: Normalization is disabled for residual/noise branch to preserve forensic frequency statistics.
    """
    def __init__(self, in_channels: int, out_channels: int, norm_layer: nn.Module = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode="reflect"),
            norm_layer,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode="reflect"),
            norm_layer.__class__(out_channels) if not isinstance(norm_layer, nn.Identity) else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualNoiseModule(nn.Module):
    """
    Combined SRM residual extractor + learnable multi-scale CNN backbone.

    Forensic motivation: Disables normalization for residual branch to preserve forensic signal integrity. Adds learnable scale to strengthen residual branch contribution.

    For splicing / copy-move detection:
        Image → SRMResidualBranch → ResidualNoiseBackbone → multi-scale features
    """

    def __init__(self, config: Optional[ResidualNoiseConfig] = None, in_channels: int = 3) -> None:
        super().__init__()
        cfg = config or ResidualNoiseConfig()
        self.cfg = cfg
        self.in_channels = in_channels
        # Learnable scale for residual branch
        self.residual_scale = nn.Parameter(torch.tensor(1.5))

        # --- SRM Residual Branch ---
        srm_kernels = torch.tensor([
            [[0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0],
             [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0],
             [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1],
             [2, -6, 8, -6, 2],
             [-2, 8, -12, 8, -2],
             [2, -6, 8, -6, 2],
             [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0],
             [0, -2, 4, -2, 0],
             [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0]],
        ], dtype=torch.float32)
        srm_kernels /= torch.abs(srm_kernels).sum(dim=(1, 2), keepdim=True)
        self.register_buffer("srm_kernels", srm_kernels.view(cfg.num_kernels, 1, 5, 5), persistent=False)
        self.srm_out_channels = in_channels * cfg.num_kernels  # e.g., RGB * 3
        self._cached_srm_kernels: Tensor | None = None
        self._cached_srm_key: tuple[torch.dtype, torch.device, int] | None = None

        # --- Multi-scale residual backbone ---
        stage_channels = [cfg.base_channels * (2**i) for i in range(cfg.num_stages)]
        self.feature_dims = stage_channels
        self.out_channels = stage_channels

        blocks = []
        downsamplers = []
        current_in = self.srm_out_channels
        for idx, out_channels in enumerate(stage_channels):
            # Disable normalization for residual branch
            norm_layer = nn.Identity()
            blocks.append(ConvBlock(current_in, out_channels, norm_layer=norm_layer))
            current_in = out_channels
            if idx < cfg.num_stages - 1:
                downsamplers.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect"),
                        nn.Identity(),
                        nn.ReLU(inplace=True),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.downsamplers = nn.ModuleList(downsamplers)

    def _get_srm_kernels(self, x: Tensor) -> Tensor:
        c = x.shape[1]
        key = (x.dtype, x.device, c)
        if self._cached_srm_kernels is not None and self._cached_srm_key == key:
            return self._cached_srm_kernels

        kernels = self.srm_kernels.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1).contiguous()
        self._cached_srm_kernels = kernels
        self._cached_srm_key = key
        return kernels

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, high_pass: Tensor | None = None) -> List[Tensor]:
        """
        Args:
            x : (B, C, H, W), e.g., RGB image
        Returns:
            List of multi-scale feature tensors from CNN backbone
        """
        # --- SRM residuals ---
        c = x.shape[1]
        kernels = self._get_srm_kernels(x)
        residual_map = F.conv2d(
            x,
            kernels,
            padding=2,
            groups=c,
        )  # shape: (B, C*3, H, W)

        if high_pass is not None:
            hp = high_pass
            if hp.shape[-2:] != x.shape[-2:]:
                _LOG.warning(
                    "ResidualNoiseBranch high_pass spatial align: (%d,%d) -> (%d,%d)",
                    hp.shape[-2], hp.shape[-1], x.shape[-2], x.shape[-1],
                )
                hp = F.interpolate(hp, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if hp.shape[1] == 1 and c > 1:
                _LOG.warning(
                    "ResidualNoiseBranch high_pass channel repeat: %d -> %d",
                    hp.shape[1], c,
                )
                hp = hp.repeat(1, c, 1, 1)
            elif hp.shape[1] != c:
                _LOG.warning(
                    "ResidualNoiseBranch high_pass channel align: %d -> %d",
                    hp.shape[1], c,
                )
                hp = hp[:, :c, ...]
                if hp.shape[1] < c:
                    hp = F.pad(hp, (0, 0, 0, 0, 0, c - hp.shape[1]))

            hp_residual_map = F.conv2d(
                hp,
                kernels,
                padding=2,
                groups=c,
            )
            residual_map = 0.5 * (residual_map + hp_residual_map)

        # --- Multi-scale CNN ---
        features = []
        # Apply learnable scale to residuals before fusion
        out = residual_map * self.residual_scale
        for idx, block in enumerate(self.blocks):
            out = block(out)
            features.append(out)
            if idx < len(self.downsamplers):
                out = self.downsamplers[idx](out)
        return features


# Backward compatibility for existing imports
ResidualNoiseBranch = ResidualNoiseModule

__all__ = ["ResidualNoiseModule", "ResidualNoiseBranch", "ResidualNoiseConfig"]

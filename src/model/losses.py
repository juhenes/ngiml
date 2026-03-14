"""Training losses for NGIML multi-stage localization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SoftDiceLoss(nn.Module):
    """Soft Dice operating on logits for stable gradients."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        dims = (1, 2, 3)
        intersection = torch.sum(probs * target, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalWithLogitsLoss(nn.Module):
    """Binary focal loss operating on logits for class-imbalance robustness."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = float(max(0.0, gamma))
        self.alpha = float(min(max(alpha, 0.0), 1.0))
        self.eps = eps

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        pt = target * probs + (1.0 - target) * (1.0 - probs)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = focal_weight * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss with logits input; beta > alpha emphasizes recall."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = float(max(0.0, alpha))
        self.beta = float(max(0.0, beta))
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()

        dims = (1, 2, 3)
        tp = torch.sum(probs * target, dim=dims)
        fp = torch.sum(probs * (1.0 - target), dim=dims)
        fn = torch.sum((1.0 - probs) * target, dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class LovaszHingeLoss(nn.Module):
    """Lovasz Hinge Loss for optimizing IoU directly."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        signs = targets * 2 - 1  # Convert targets to {-1, 1}
        errors = 1 - logits * signs
        errors_sorted, perm = torch.sort(errors.view(errors.size(0), -1), dim=1, descending=True)
        perm = perm.detach()
        targets_sorted = targets.view(targets.size(0), -1).gather(1, perm)
        grad = self._lovasz_grad(targets_sorted)
        return (F.relu(errors_sorted) * grad).mean()

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovasz extension w.r.t sorted errors."""
        gts = gt_sorted.sum(dim=1, keepdim=True)
        intersection = gts - gt_sorted.cumsum(dim=1)
        union = gts + (1 - gt_sorted).cumsum(dim=1)
        jacc = 1 - intersection / union.clamp_min(1.0)
        return torch.cat([jacc[:, :1], jacc[:, 1:] - jacc[:, :-1]], dim=1)


@dataclass
class MultiStageLossConfig:
    """Configuration flags for the combined Dice + weighted BCE loss.

    Forensic motivation: Reduce deep supervision strength so final prediction dominates total loss, improving stability for forensic segmentation. Optionally applies hard pixel mining to focus on ambiguous regions.
    """

    dice_weight: float = 1.0
    bce_weight: float = 1.0
    pos_weight: float = 1.0
    # Default stage weights: [0.05, 0.1, 0.2, 1.0] (final prediction dominates)
    stage_weights: Optional[Sequence[float]] = field(default_factory=lambda: [0.05, 0.1, 0.2, 1.0])
    smooth: float = 1e-6
    hybrid_mode: str = "dice_bce"  # one of: dice_bce, dice_focal
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.8
    lovasz_weight: float = 0.0  # Weight for Lovasz Hinge Loss
    use_boundary_loss: bool = False
    boundary_weight: float = 0.03
    hard_pixel_mining: bool = False  # Disable hard pixel mining by default to stabilize training
    



class MultiStageManipulationLoss(nn.Module):
    """Applies configurable hybrid segmentation supervision at every prediction stage, with optional boundary loss.

    Forensic motivation: Adds Sobel-based boundary loss to encourage sharper manipulation boundaries.
    """
    def __init__(self, config: MultiStageLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or MultiStageLossConfig()
        self.dice = SoftDiceLoss(smooth=self.cfg.smooth)
        self.focal = FocalWithLogitsLoss(gamma=self.cfg.focal_gamma, alpha=self.cfg.focal_alpha)
        self.tversky = TverskyLoss(
            alpha=self.cfg.tversky_alpha,
            beta=self.cfg.tversky_beta,
            smooth=self.cfg.smooth,
        )
        self.lovasz = LovaszHingeLoss()
        self.boundary_weight = float(max(0.0, getattr(self.cfg, "boundary_weight", 0.0)))
        self.use_boundary_loss = bool(getattr(self.cfg, "use_boundary_loss", False)) and self.boundary_weight > 0.0
        self.boundary_loss = SobelBoundaryLoss() if self.use_boundary_loss else None

        mode = self.cfg.hybrid_mode.strip().lower()
        if mode not in {"dice_bce", "dice_focal"}:
            raise ValueError("hybrid_mode must be one of: 'dice_bce', 'dice_focal'")
        self.hybrid_mode = mode

    def _stage_weights(self, num_stages: int) -> List[float]:
        if self.cfg.stage_weights is None:
            return [float(i + 1) / float(num_stages) for i in range(num_stages)]
        if len(self.cfg.stage_weights) < num_stages:
            raise ValueError("Provided stage_weights shorter than number of stages")
        return list(self.cfg.stage_weights[:num_stages])

    def forward(
        self,
        preds: List[Tensor],
        target: Tensor,
        edge_target: Tensor | None = None,
        edge_target_present: Tensor | None = None,
    ) -> Tensor:
        if not preds:
            raise ValueError("Loss received empty predictions list")
        target = target.float()
        if edge_target is not None:
            edge_target = edge_target.float()
        if edge_target_present is not None:
            edge_target_present = edge_target_present.to(device=target.device, dtype=torch.bool).view(-1)
        stage_weights = self._stage_weights(len(preds))
        pos_weight = torch.as_tensor(
            self.cfg.pos_weight,
            dtype=target.dtype,
            device=target.device,
        )

        total_loss = 0.0
        normalizer = 0.0
        for stage_weight, logits in zip(stage_weights, preds):
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            dice = self.dice(logits, target)
            if self.hybrid_mode == "dice_bce":
                bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
                hybrid_term = self.cfg.bce_weight * bce
            else:
                focal = self.focal(logits, target)
                hybrid_term = self.cfg.bce_weight * focal

            # Hard pixel mining: weight loss by pixel difficulty
            if getattr(self.cfg, "hard_pixel_mining", False):
                with torch.no_grad():
                    pred_prob = torch.sigmoid(logits)
                    difficulty = torch.abs(pred_prob - target)
                    weight = 1.0 + 2.0 * (difficulty > 0.3).float()
                # Apply weighting to BCE/focal and dice
                hybrid_term = (hybrid_term * weight).mean()
                dice = (1.0 - ((1.0 - dice) * weight).mean())  # Weighted dice
            else:
                hybrid_term = hybrid_term.mean() if hybrid_term.ndim > 0 else hybrid_term

            stage_loss = self.cfg.dice_weight * dice + hybrid_term
            if self.cfg.tversky_weight > 0:
                stage_loss = stage_loss + self.cfg.tversky_weight * self.tversky(logits, target)
            if getattr(self.cfg, "lovasz_weight", 0) > 0:
                stage_loss = stage_loss + self.cfg.lovasz_weight * self.lovasz(logits, target)

            total_loss += stage_weight * stage_loss
            normalizer += stage_weight

        # Add boundary loss on final prediction
        if self.use_boundary_loss and self.boundary_loss is not None and preds:
            boundary = self.boundary_loss(
                preds[-1],
                target,
                edge_target=edge_target,
                edge_target_present=edge_target_present,
            )
            total_loss += self.boundary_weight * boundary

        return total_loss / max(normalizer, 1e-6)

class SobelBoundaryLoss(nn.Module):
    """Sobel-based boundary loss for sharper manipulation boundaries.

    Forensic motivation: Penalizes boundary errors by comparing Sobel gradient magnitudes of prediction and target.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        edge_target: Tensor | None = None,
        edge_target_present: Tensor | None = None,
    ) -> Tensor:
        pred = torch.sigmoid(pred)
        target = target.float()
        if edge_target is not None:
            edge_target = edge_target.float()
        # Ensure sobel kernels match input dtype/device to avoid type-mismatch
        # errors when using mixed precision (e.g., fp16/bf16).
        sobel_x = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sobel_y = self.sobel_y.to(dtype=pred.dtype, device=pred.device)
        # Compute gradients
        grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
        grad_target_x = F.conv2d(target, sobel_x, padding=1)
        grad_target_y = F.conv2d(target, sobel_y, padding=1)
        grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2 + 1e-6)
        grad_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2 + 1e-6)
        if edge_target is not None:
            explicit_edge = edge_target.to(dtype=pred.dtype, device=pred.device)
            if explicit_edge.shape[-2:] != grad_target.shape[-2:]:
                explicit_edge = F.interpolate(explicit_edge, size=grad_target.shape[-2:], mode="nearest")
            explicit_edge = explicit_edge.clamp(0.0, 1.0)
            if edge_target_present is None:
                grad_target = explicit_edge
            else:
                present = edge_target_present.to(device=pred.device, dtype=pred.dtype).view(-1, 1, 1, 1)
                grad_target = present * explicit_edge + (1.0 - present) * grad_target
        return F.l1_loss(grad_pred, grad_target)

__all__ = [
    "SoftDiceLoss",
    "FocalWithLogitsLoss",
    "TverskyLoss",
    "MultiStageLossConfig",
    "MultiStageManipulationLoss",
    "SobelBoundaryLoss",
    "LovaszHingeLoss",
]

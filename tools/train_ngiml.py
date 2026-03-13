"""End-to-end NGIML training loop with checkpointing.

Run example (Colab-ready):
    python tools/train_ngiml.py --manifest /content/data/manifest.json --output-dir /content/runs

The script expects a prepared manifest (see src/data/config.py) and will
save checkpoints plus a copy of the training arguments inside the output dir.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import random
import time
import os
import hashlib
import tarfile
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
import re
import math


import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
try:
    import xformers
except ImportError:
    xformers = None
try:
    import flash_attn
except ImportError:
    flash_attn = None

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloaders import (
    AugmentationConfig,
    create_dataloaders,
    load_manifest,
    _apply_gpu_augmentations,
    _apply_gpu_augmentations_batch,
    _normalize,
)
from src.model.hybrid_ngiml import HybridNGIML, HybridNGIMLConfig
from src.model.losses import MultiStageLossConfig, MultiStageManipulationLoss


class _PrefetchLoader:
    """Simple async prefetcher that moves next batch to CUDA in a background stream.

    Usage: wrap a DataLoader with `_PrefetchLoader(loader, device)` before iterating.
    No-op on CPU devices.
    """

    def __init__(self, loader, device: torch.device):
        self._loader = loader
        self._device = device
        self._stream = None

    def __iter__(self):
        if self._device.type != "cuda":
            return iter(self._loader)
        self._iter = iter(self._loader)
        self._stream = torch.cuda.Stream()
        self._next_batch = None
        self._preload()
        return self

    def __next__(self):
        if self._device.type != "cuda":
            return next(self._iter)
        if self._next_batch is None:
            raise StopIteration
        # Wait only for the currently prefetched batch, then launch prefetch
        # for the subsequent one to overlap transfer with compute.
        torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next_batch
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(torch.cuda.current_stream())
        self._preload()
        return batch

    def _preload(self):
        try:
            nxt = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        # Move tensors to GPU in the prefetch stream
        with torch.cuda.stream(self._stream):
            for k, v in list(nxt.items()):
                if isinstance(v, torch.Tensor):
                    try:
                        nxt[k] = v.to(self._device, non_blocking=True)
                    except Exception:
                        nxt[k] = v.to(self._device)
        self._next_batch = nxt

    def __len__(self):
        try:
            return len(self._loader)
        except Exception:
            raise TypeError("wrapped loader has no __len__")

    def __getattr__(self, name: str):
        # Proxy attribute access to the underlying loader for compatibility
        return getattr(self._loader, name)



def _build_lr_scheduler(optimizer, cfg):
    """Builds a learning rate scheduler with optional warmup and cosine/step decay."""
    if not cfg.lr_schedule or cfg.epochs <= 1:
        return None

    warmup_epochs = max(0, min(cfg.warmup_epochs, max(cfg.epochs - 1, 0)))
    min_lr_scale = float(max(0.0, min(cfg.min_lr_scale, 1.0)))

    if getattr(cfg, "scheduler_type", "cosine") == "step":
        step_size = getattr(cfg, "step_size", 10)
        gamma = getattr(cfg, "gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def _lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return min_lr_scale + (1.0 - min_lr_scale) * (float(epoch + 1) / float(warmup_epochs))
        cosine_total = max(cfg.epochs - warmup_epochs, 1)
        cosine_epoch = min(max(epoch - warmup_epochs, 0), cosine_total)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_epoch / cosine_total))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


@dataclass
class TrainConfig:
    manifest: str
    scheduler_type: str = "cosine"  # one of: 'cosine', 'step' (cosine enabled by default)
    output_dir: str = "runs/ngiml"
    batch_size: int = 8
    epochs: int = 50
    num_workers: int = max(2, min(8, (os.cpu_count() or 4) // 4))
    amp: bool = True
    pin_memory: bool = True
    channels_last: bool = True
    compile_model: bool = True
    compile_mode: str = "default"
    deterministic: bool = False
    use_tf32: bool = True
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    xformers: bool = True
    cuda_expandable_segments: bool = True
    lr_schedule: bool = True
    warmup_epochs: int = 5  # Linear warmup for first 5 epochs
    min_lr_scale: float = 0.1  # Start at 10% base LR
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    val_every: int = 1
    checkpoint_every: int = 1
    resume: Optional[str] = None
    auto_resume: bool = True
    round_robin_seed: Optional[int] = 42
    balance_sampling: bool = False
    balance_real_fake: bool = True
    balanced_positive_ratio: float = 0.6
    balanced_sampler_seed: int = 42
    balanced_sampler_num_samples: Optional[int] = None
    prefetch_factor: Optional[int] = 1
    persistent_workers: bool = False
    drop_last: bool = True
    auto_local_cache: bool = True
    local_cache_dir: Optional[str] = None
    reuse_local_cache_manifest: bool = True
    views_per_sample: int = 3
    # Cap the short side of input images early in the dataloader to avoid
    # excessive spatial sizes that can trigger timm/Swin assertions or OOMs.
    max_short_side: int = 384
    max_rotation_degrees: float = 0.0
    noise_std_max: float = 0.01
    disable_aug: bool = False
    device: Optional[str] = None
    aug_seed: Optional[int] = 42
    seed: int = 42
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = "iou"
    training_phase: str = "phase1"
    auto_phase2_enabled: bool = True
    auto_phase2_patience: int = 5
    auto_phase2_lr_scale: float = 0.33
    auto_phase2_tversky_weight: float = 0.1
    auto_phase2_monitor: str = "iou"
    metric_threshold: float = 0.5
    optimize_threshold: bool = True
    threshold_metric: str = "f1"
    threshold_start: float = 0.2
    threshold_end: float = 0.8
    threshold_step: float = 0.02
    small_mask_ratio_max: float = 0.01
    medium_mask_ratio_max: float = 0.05
    compute_foreground_ratio: bool = True
    foreground_ratio_max_batches: int = 40
    short_side_probe_samples: int = 128
    auto_pos_weight: bool = True
    pos_weight_min: float = 0.5
    pos_weight_max: float = 10.0
    balanced_pos_weight_cap: float = 3.0
    loss_hybrid_mode: str = "dice_bce"
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.8
    lovasz_weight: float = 0.0
    use_boundary_loss: bool = False
    boundary_weight: float = 0.05
    ema_enabled: bool = True
    ema_decay: float = 0.999
    hard_mining_enabled: bool = False
    hard_mining_start_epoch: int = 5
    hard_mining_weight: float = 0.03
    hard_mining_gamma: float = 2.0
    default_aug: Optional[AugmentationConfig] = None
    per_dataset_aug: Optional[Dict[str, AugmentationConfig]] = None
    model_config: Optional[HybridNGIMLConfig] = None
    loss_config: Optional[MultiStageLossConfig] = None
    debug_timing: bool = False


@dataclass
class Checkpoint:
    epoch: int
    global_step: int
    model_state: dict
    raw_model_state: Optional[dict]
    ema_state: Optional[dict]
    optimizer_state: dict
    scheduler_state: Optional[dict]
    scaler_state: Optional[dict]
    train_config: dict


def build_default_components() -> tuple[HybridNGIMLConfig, MultiStageLossConfig, AugmentationConfig, dict[str, AugmentationConfig]]:
    """Build default model/loss/augmentation configs from a single top-level source."""
    from src.model.backbones.efficientnet_backbone import EfficientNetBackboneConfig
    from src.model.backbones.residual_noise_branch import ResidualNoiseConfig
    from src.model.backbones.swin_backbone import SwinBackboneConfig
    from src.model.feature_fusion import FeatureFusionConfig
    from src.model.hybrid_ngiml import HybridNGIMLOptimizerConfig, OptimizerGroupConfig
    from src.model.unet_decoder import UNetDecoderConfig

    model_cfg = HybridNGIMLConfig(
        efficientnet=EfficientNetBackboneConfig(pretrained=True),
        swin=SwinBackboneConfig(model_name="swin_tiny_patch4_window7_224", pretrained=True, input_size=384),
        residual=ResidualNoiseConfig(num_kernels=3, base_channels=32, num_stages=4),
        fusion=FeatureFusionConfig(fusion_channels=(64, 128, 192, 256)),
        decoder=UNetDecoderConfig(decoder_channels=None, out_channels=1, per_stage_heads=True),
        optimizer=HybridNGIMLOptimizerConfig(
            efficientnet=OptimizerGroupConfig(lr=1e-5, weight_decay=1.5e-4),
            swin=OptimizerGroupConfig(lr=5e-6, weight_decay=1e-4),
            residual=OptimizerGroupConfig(lr=2.5e-4, weight_decay=2e-4),
            fusion=OptimizerGroupConfig(lr=1.2e-4, weight_decay=2e-4),
            decoder=OptimizerGroupConfig(lr=1.8e-4, weight_decay=2e-4),
        ),
        use_low_level=True,
        use_context=True,
        use_residual=True,
    )

    loss_cfg = MultiStageLossConfig(
        dice_weight=1.0,
        bce_weight=1.0,
        pos_weight=1.0,
        stage_weights=[0.05, 0.1, 0.2, 1.0],
        smooth=1e-6,
        hybrid_mode="dice_bce",
        tversky_weight=0.0,
        tversky_alpha=0.3,
        tversky_beta=0.8,
        lovasz_weight=0.0,
        use_boundary_loss=True,
        boundary_weight=0.03,
    )

    default_aug = AugmentationConfig(
        enable=True,
        views_per_sample=3,
        enable_flips=True,
        enable_rotations=True,
        max_rotation_degrees=6.0,
        enable_random_crop=True,
        crop_scale_range=(0.75, 1.0),
        object_crop_bias_prob=0.85,
        min_fg_pixels_for_object_crop=8,
        enable_elastic=False,
        elastic_prob=0.0,
        elastic_alpha=8.0,
        elastic_sigma=5.0,
        enable_color_jitter=True,
        brightness_jitter_factors=(0.9, 1.1),
        contrast_jitter_factors=(0.9, 1.1),
        enable_noise=True,
        noise_std_range=(0.0, 0.012),
    )

    per_dataset_aug: dict[str, AugmentationConfig] = {}
    return model_cfg, loss_cfg, default_aug, per_dataset_aug


def build_training_config(
    manifest_path: Path | str,
    output_dir: str,
    model_cfg: HybridNGIMLConfig,
    loss_cfg: MultiStageLossConfig,
    default_aug: AugmentationConfig,
    per_dataset_aug: dict[str, AugmentationConfig],
) -> dict:
    """Build notebook-friendly training config dict from top-level defaults."""
    cfg = TrainConfig(
        manifest=str(manifest_path),
        output_dir=output_dir,
        batch_size=20,
        num_workers=0,
        prefetch_factor=2,
        max_short_side=480,
        max_rotation_degrees=6.0,
        noise_std_max=0.012,
        warmup_epochs=3,
        auto_phase2_enabled=True,
        foreground_ratio_max_batches=20,
        short_side_probe_samples=0,
        loss_hybrid_mode=str(getattr(loss_cfg, "hybrid_mode", "dice_bce")),
        dice_weight=float(getattr(loss_cfg, "dice_weight", 1.0)),
        bce_weight=float(getattr(loss_cfg, "bce_weight", 1.0)),
        focal_gamma=float(getattr(loss_cfg, "focal_gamma", 2.0)),
        focal_alpha=float(getattr(loss_cfg, "focal_alpha", 0.25)),
        tversky_weight=float(getattr(loss_cfg, "tversky_weight", 0.0)),
        tversky_alpha=float(getattr(loss_cfg, "tversky_alpha", 0.3)),
        tversky_beta=float(getattr(loss_cfg, "tversky_beta", 0.8)),
        lovasz_weight=float(getattr(loss_cfg, "lovasz_weight", 0.0)),
        use_boundary_loss=bool(getattr(loss_cfg, "use_boundary_loss", False)),
        boundary_weight=float(getattr(loss_cfg, "boundary_weight", 0.05)),
        default_aug=default_aug,
        per_dataset_aug=per_dataset_aug,
        model_config=model_cfg,
        loss_config=loss_cfg,
    )
    return dict(cfg.__dict__)


def parse_args() -> TrainConfig:
    # Use a higher default worker count to better saturate fast GPUs (e.g. A100).
    # Keep a sensible floor to avoid tiny values on CI/low-core systems.
    default_workers = max(4, (os.cpu_count() or 4))
    parser = argparse.ArgumentParser(description="Train NGIML manipulation localization")
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "step"], help="LR scheduler type (cosine or step)")
    parser.add_argument("--manifest", required=True, help="Path to prepared manifest JSON")
    parser.add_argument("--output-dir", default="runs/ngiml", help="Directory to write checkpoints/logs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num-workers", type=int, default=default_workers, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pinned memory")
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels-last memory format on CUDA")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on model")
    parser.add_argument("--compile-mode", type=str, default="default", help="torch.compile mode")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic kernels (slower)")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matrix math on CUDA")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Numerical precision for training")
    parser.add_argument("--debug-timing", action="store_true", help="Enable lightweight per-stage timing prints during training")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False, help="Enable gradient checkpointing for memory savings")
    parser.add_argument("--flash-attention", action=argparse.BooleanOptionalAction, default=False, help="Enable flash attention if supported")
    parser.add_argument("--xformers", action=argparse.BooleanOptionalAction, default=False, help="Enable xformers memory-efficient attention if supported")
    parser.add_argument(
        "--cuda-expandable-segments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True when CUDA is used",
    )
    parser.add_argument("--no-lr-schedule", action="store_true", help="Disable warmup+cosine LR schedule")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs (linear, default=5)")
    parser.add_argument("--min-lr-scale", type=float, default=0.1, help="Initial LR scale for warmup (default=0.1)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm; <=0 disables")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write checkpoint every N epochs (includes last epoch)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from latest checkpoint in output_dir/checkpoints when available",
    )
    parser.add_argument("--round-robin-seed", type=int, default=42, help="Seed for round-robin sampler")
    parser.add_argument(
        "--balance-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Balance per-dataset sampling by oversampling smaller datasets",
    )
    parser.add_argument(
        "--balance-real-fake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable weighted sampling to match a target fake-positive ratio in train batches",
    )
    parser.add_argument(
        "--balanced-positive-ratio",
        type=float,
        default=0.5,
        help="Target fake-positive sampling ratio when --balance-real-fake is enabled",
    )
    parser.add_argument(
        "--balanced-sampler-seed",
        type=int,
        default=42,
        help="Random seed used by the real/fake balanced sampler",
    )
    parser.add_argument(
        "--balanced-sampler-num-samples",
        type=int,
        default=None,
        help="Optional number of sampled training items per epoch for real/fake balancing",
    )
    parser.add_argument("--prefetch-factor", type=int, default=1, help="DataLoader prefetch factor")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable persistent DataLoader workers",
    )
    parser.add_argument(
        "--drop-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop last incomplete batch in training",
    )
    parser.add_argument(
        "--auto-local-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically materialize tar::npz samples to local cache before training",
    )
    parser.add_argument(
        "--local-cache-dir",
        type=str,
        default=None,
        help="Directory for local materialized samples (defaults to output_dir/local_cache)",
    )
    parser.add_argument(
        "--reuse-local-cache-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing local cached manifest when available to shorten startup",
    )
    parser.add_argument("--views-per-sample", type=int, default=2, help="Number of augmented views per sample (on-the-fly)")
    parser.add_argument("--max-short-side", type=int, default=384, help="Cap image short side before batching (lower is faster)")
    parser.add_argument("--max-rotation-degrees", type=float, default=0.0, help="Random rotation range (+/-)")
    parser.add_argument("--noise-std-max", type=float, default=0.01, help="Max Gaussian noise std")
    parser.add_argument("--disable-aug", action="store_true", help="Disable GPU augmentations")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda:0 or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Stop after N validations without improvement; <=0 disables")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="Minimum monitored-metric improvement to reset early stopping")
    parser.add_argument("--early-stopping-monitor", type=str, default="loss", choices=["iou", "dice", "f1", "recall", "precision", "accuracy", "loss"], help="Validation metric used for early stopping and best checkpoint")
    parser.add_argument("--training-phase", type=str, default="phase1", choices=["phase1", "phase2"], help="Training phase label stored in checkpoints and logs")
    parser.add_argument("--auto-phase2-enabled", action=argparse.BooleanOptionalAction, default=True, help="Automatically switch to phase 2 from the best IoU checkpoint after a phase-1 plateau")
    parser.add_argument("--auto-phase2-patience", type=int, default=5, help="Validations without improvement before auto phase-2 triggers during phase 1")
    parser.add_argument("--auto-phase2-lr-scale", type=float, default=0.33, help="LR multiplier applied when auto phase-2 activates")
    parser.add_argument("--auto-phase2-tversky-weight", type=float, default=0.1, help="Tversky loss weight applied during auto phase-2")
    parser.add_argument("--auto-phase2-monitor", type=str, default="iou", choices=["iou", "f1", "dice"], help="Validation metric used for auto phase-2 monitoring and threshold selection")
    parser.add_argument("--metric-threshold", type=float, default=0.5, help="Fixed threshold for sigmoid outputs when threshold optimization is disabled")
    parser.add_argument("--optimize-threshold", action=argparse.BooleanOptionalAction, default=True, help="Search validation thresholds and use the best for metric reporting")
    parser.add_argument("--threshold-metric", type=str, default="f1", choices=["iou", "dice", "f1"], help="Metric used to select best threshold")
    parser.add_argument("--threshold-start", type=float, default=0.2, help="Threshold search range start")
    parser.add_argument("--threshold-end", type=float, default=0.8, help="Threshold search range end")
    parser.add_argument("--threshold-step", type=float, default=0.02, help="Threshold search step size")
    parser.add_argument("--small-mask-ratio-max", type=float, default=0.01, help="Upper foreground-ratio bound for small-mask validation bin")
    parser.add_argument("--medium-mask-ratio-max", type=float, default=0.05, help="Upper foreground-ratio bound for medium-mask validation bin")
    parser.add_argument("--compute-foreground-ratio", action=argparse.BooleanOptionalAction, default=True, help="Compute foreground pixel ratio from train loader")
    parser.add_argument(
        "--foreground-ratio-max-batches",
        type=int,
        default=40,
        help="Max train batches sampled when computing foreground pixel ratio",
    )
    parser.add_argument(
        "--short-side-probe-samples",
        type=int,
        default=128,
        help="Max samples per split to probe on disk for size bucketing (0 disables probing)",
    )
    parser.add_argument("--auto-pos-weight", action=argparse.BooleanOptionalAction, default=True, help="Auto-compute BCE pos_weight from foreground ratio")
    parser.add_argument("--pos-weight-min", type=float, default=0.5, help="Lower clamp for auto pos_weight")
    parser.add_argument("--pos-weight-max", type=float, default=10.0, help="Upper clamp for auto pos_weight")
    parser.add_argument(
        "--balanced-pos-weight-cap",
        type=float,
        default=3.0,
        help="When --balance-real-fake is enabled, cap auto pos_weight to this value (<=0 disables cap)",
    )
    parser.add_argument("--loss-hybrid-mode", type=str, default="dice_bce", choices=["dice_bce", "dice_focal"], help="Hybrid loss type")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight")
    parser.add_argument("--bce-weight", type=float, default=1.0, help="BCE/Focal term weight in hybrid loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (used when loss-hybrid-mode=dice_focal)")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha (used when loss-hybrid-mode=dice_focal)")
    parser.add_argument("--tversky-weight", type=float, default=0.0, help="Optional Tversky loss weight to improve recall")
    parser.add_argument("--tversky-alpha", type=float, default=0.3, help="Tversky alpha (FP penalty)")
    parser.add_argument("--tversky-beta", type=float, default=0.8, help="Tversky beta (FN penalty)")
    parser.add_argument("--lovasz-weight", type=float, default=0.0, help="Lovasz Hinge Loss weight for IoU optimization")
    parser.add_argument("--use-boundary-loss", action=argparse.BooleanOptionalAction, default=False, help="Enable Sobel boundary loss on final prediction")
    parser.add_argument("--boundary-weight", type=float, default=0.05, help="Boundary loss weight when --use-boundary-loss is enabled")
    parser.add_argument("--ema-enabled", action=argparse.BooleanOptionalAction, default=True, help="Use EMA weights for validation and best checkpoints")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay factor")
    parser.add_argument("--hard-mining-enabled", action=argparse.BooleanOptionalAction, default=False, help="Enable low-IoU hard-example weighting")
    parser.add_argument("--hard-mining-start-epoch", type=int, default=5, help="Epoch to start hard-example weighting")
    parser.add_argument("--hard-mining-weight", type=float, default=0.03, help="Weight of hard-example auxiliary loss")
    parser.add_argument("--hard-mining-gamma", type=float, default=2.0, help="Scale for low-IoU hard-example weights")
    args = parser.parse_args()
    return TrainConfig(
        manifest=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        pin_memory=not args.no_pin_memory,
        channels_last=not args.no_channels_last,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        deterministic=args.deterministic,
        use_tf32=not args.no_tf32,
        cuda_expandable_segments=args.cuda_expandable_segments,
        lr_schedule=not args.no_lr_schedule,
        warmup_epochs=args.warmup_epochs,
        min_lr_scale=args.min_lr_scale,
        grad_clip=args.grad_clip,
        grad_accum_steps=max(1, int(args.grad_accum_steps)),
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        auto_resume=args.auto_resume,
        round_robin_seed=args.round_robin_seed,
        balance_sampling=args.balance_sampling,
        balance_real_fake=args.balance_real_fake,
        balanced_positive_ratio=args.balanced_positive_ratio,
        balanced_sampler_seed=args.balanced_sampler_seed,
        balanced_sampler_num_samples=args.balanced_sampler_num_samples,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        drop_last=args.drop_last,
        auto_local_cache=args.auto_local_cache,
        local_cache_dir=args.local_cache_dir,
        reuse_local_cache_manifest=args.reuse_local_cache_manifest,
        views_per_sample=args.views_per_sample,
        max_short_side=max(64, int(args.max_short_side)),
        max_rotation_degrees=args.max_rotation_degrees,
        noise_std_max=args.noise_std_max,
        disable_aug=args.disable_aug,
        device=args.device,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        training_phase=args.training_phase,
        auto_phase2_enabled=args.auto_phase2_enabled,
        auto_phase2_patience=max(1, int(args.auto_phase2_patience)),
        auto_phase2_lr_scale=float(args.auto_phase2_lr_scale),
        auto_phase2_tversky_weight=float(args.auto_phase2_tversky_weight),
        auto_phase2_monitor=args.auto_phase2_monitor,
        metric_threshold=args.metric_threshold,
        optimize_threshold=args.optimize_threshold,
        threshold_metric=args.threshold_metric,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        small_mask_ratio_max=args.small_mask_ratio_max,
        medium_mask_ratio_max=args.medium_mask_ratio_max,
        compute_foreground_ratio=args.compute_foreground_ratio,
        foreground_ratio_max_batches=max(0, int(args.foreground_ratio_max_batches)),
        short_side_probe_samples=max(0, int(args.short_side_probe_samples)),
        auto_pos_weight=args.auto_pos_weight,
        pos_weight_min=args.pos_weight_min,
        pos_weight_max=args.pos_weight_max,
        balanced_pos_weight_cap=args.balanced_pos_weight_cap,
        loss_hybrid_mode=args.loss_hybrid_mode,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        lovasz_weight=args.lovasz_weight,
        use_boundary_loss=args.use_boundary_loss,
        boundary_weight=args.boundary_weight,
        ema_enabled=args.ema_enabled,
        ema_decay=args.ema_decay,
        hard_mining_enabled=args.hard_mining_enabled,
        hard_mining_start_epoch=args.hard_mining_start_epoch,
        hard_mining_weight=args.hard_mining_weight,
        hard_mining_gamma=args.hard_mining_gamma,
        scheduler_type=args.scheduler_type,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
        flash_attention=args.flash_attention,
        xformers=args.xformers,
    )


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _collect_dataset_names(manifest_path: Path) -> Sequence[str]:
    manifest = load_manifest(manifest_path)
    names = sorted({sample.dataset for sample in manifest.samples})
    if not names:
        raise ValueError("Manifest contains no samples")
    return names


def _coerce_aug(value) -> AugmentationConfig:
    if isinstance(value, AugmentationConfig):
        return replace(value)
    if isinstance(value, dict):
        return AugmentationConfig(**value)
    raise TypeError("Augmentation config must be AugmentationConfig or dict")


def _build_aug_map(names: Sequence[str], cfg: TrainConfig) -> Dict[str, AugmentationConfig]:
    base_aug = cfg.default_aug or AugmentationConfig(
        enable=not cfg.disable_aug,
        views_per_sample=cfg.views_per_sample if cfg.views_per_sample is not None else 1,
        enable_flips=True,
        enable_rotations=cfg.max_rotation_degrees > 0,
        max_rotation_degrees=cfg.max_rotation_degrees,
        enable_random_crop=True,
        crop_scale_range=(0.75, 1.0),
        object_crop_bias_prob=0.85,
        min_fg_pixels_for_object_crop=8,
        enable_elastic=False,
        elastic_prob=0.0,
        elastic_alpha=8.0,
        elastic_sigma=5.0,
        enable_color_jitter=True,
        brightness_jitter_factors=(0.9, 1.1),
        contrast_jitter_factors=(0.9, 1.1),
        enable_noise=cfg.noise_std_max > 0,
        noise_std_range=(0.0, max(0.0, cfg.noise_std_max)),
    )

    aug_map: Dict[str, AugmentationConfig] = {name: _coerce_aug(base_aug) for name in names}

    if cfg.per_dataset_aug:
        for name, aug in cfg.per_dataset_aug.items():
            aug_map[name] = _coerce_aug(aug)

    return aug_map


def _prepare_dataloaders(cfg: TrainConfig, device: torch.device):
    manifest_path = Path(cfg.manifest)
    dataset_names = _collect_dataset_names(manifest_path)
    per_dataset_aug = _build_aug_map(dataset_names, cfg)
    # Also return the per-dataset augmentation map and normalization mode so
    # augmentations can be applied on-device in the training loop.
    manifest = load_manifest(manifest_path)
    normalization_mode = manifest.normalization_mode
    # When running on CUDA we prefer to perform augmentations and normalization
    # on-device. To enable that we disable cpu-side augmentation/normalization
    # inside the collate function by passing a disabled aug map and using
    # a no-op normalization there. The original per_dataset_aug and
    # normalization_mode are returned for on-device application in the loop.
    collate_aug_map = per_dataset_aug
    collate_norm_mode = normalization_mode
    if device.type == "cuda":
        from dataclasses import replace as _dc_replace

        collate_aug_map = {name: _dc_replace(aug, enable=False) for name, aug in per_dataset_aug.items()}
        collate_norm_mode = "zero_one"

    loaders = create_dataloaders(
        manifest_path,
        collate_aug_map,
        batch_size=cfg.batch_size,
        device=device,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
        round_robin_seed=cfg.round_robin_seed,
        balance_sampling=cfg.balance_sampling,
        balance_real_fake=cfg.balance_real_fake,
        balanced_positive_ratio=cfg.balanced_positive_ratio,
        balanced_sampler_seed=cfg.balanced_sampler_seed,
        balanced_sampler_num_samples=cfg.balanced_sampler_num_samples,
        drop_last=cfg.drop_last,
        aug_seed=cfg.aug_seed if cfg.aug_seed is not None else cfg.seed,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        max_short_side=cfg.max_short_side,
        short_side_probe_samples=cfg.short_side_probe_samples,
        normalization_mode_override=collate_norm_mode,
    )
    return loaders, per_dataset_aug, normalization_mode


def _safe_cache_name(spec: str) -> str:
    digest = hashlib.sha1(spec.encode("utf-8")).hexdigest()
    return f"{digest}.npz"


def _materialize_tar_npz_manifest(manifest_path: Path, cache_root: Path) -> Path:
    manifest = load_manifest(manifest_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    samples_dir = cache_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    tar_handles: Dict[str, tarfile.TarFile] = {}

    def _extract_if_needed(spec: str) -> str:
        if "::" not in spec or not spec.endswith(".npz"):
            return spec
        archive_path, member_name = spec.split("::", 1)
        out_path = samples_dir / _safe_cache_name(spec)
        if out_path.exists() and out_path.stat().st_size > 0:
            return str(out_path)

        tar = tar_handles.get(archive_path)
        if tar is None or tar.closed:
            tar = tarfile.open(archive_path, mode="r:*")
            tar_handles[archive_path] = tar

        member = tar.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"Missing tar member {member_name} in {archive_path}")

        # Stream-write the member to avoid loading the entire file into memory
        with open(out_path, "wb") as out_f:
            shutil.copyfileobj(member, out_f)
        return str(out_path)

    try:
        changed = False
        for sample in manifest.samples:
            new_image_path = _extract_if_needed(sample.image_path)
            if new_image_path != sample.image_path:
                sample.image_path = new_image_path
                changed = True

            if sample.mask_path is not None:
                new_mask_path = _extract_if_needed(sample.mask_path)
                if new_mask_path != sample.mask_path:
                    sample.mask_path = new_mask_path
                    changed = True

        resolved_manifest = cache_root / "manifest_local_cache.parquet"
        if changed or not resolved_manifest.exists():
            manifest.to_dataframe().to_parquet(resolved_manifest, index=False)
        return resolved_manifest
    finally:
        for tar in tar_handles.values():
            try:
                tar.close()
            except Exception:
                pass


def _resolve_manifest_for_training(cfg: TrainConfig, out_dir: Path) -> Path:
    manifest_path = Path(cfg.manifest)
    if not cfg.auto_local_cache:
        return manifest_path

    manifest = load_manifest(manifest_path)
    has_tar_npz = any("::" in s.image_path and s.image_path.endswith(".npz") for s in manifest.samples)
    if not has_tar_npz:
        return manifest_path

    cache_root = Path(cfg.local_cache_dir) if cfg.local_cache_dir else (out_dir / "local_cache")
    cache_root = cache_root / manifest_path.stem
    resolved_manifest = cache_root / "manifest_local_cache.parquet"
    if cfg.reuse_local_cache_manifest and resolved_manifest.exists():
        print(f"Reusing pre-materialized local cache manifest: {resolved_manifest}")
        return resolved_manifest

    print(f"Materializing tar::npz samples to local cache: {cache_root}")
    return _materialize_tar_npz_manifest(manifest_path, cache_root)


def _segmentation_counts(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= float(threshold)).float()
    target = target.float()

    tp = torch.sum(pred * target).item()
    tn = torch.sum((1.0 - pred) * (1.0 - target)).item()
    fp = torch.sum(pred * (1.0 - target)).item()
    fn = torch.sum((1.0 - pred) * target).item()
    return {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}


def _metrics_from_counts(tp: float, tn: float, fp: float, fn: float, eps: float = 1e-6) -> Dict[str, float]:
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _build_threshold_grid(cfg: TrainConfig) -> Sequence[float]:
    start = float(min(max(cfg.threshold_start, 0.0), 1.0))
    end = float(min(max(cfg.threshold_end, 0.0), 1.0))
    step = float(max(cfg.threshold_step, 1e-6))
    if end < start:
        start, end = end, start

    values = []
    t = start
    while t <= (end + 1e-9):
        values.append(round(t, 4))
        t += step

    if not values:
        values = [0.5]

    if 0.5 not in values:
        values.append(0.5)
    values = sorted(set(values))
    return values


def _select_threshold_with_precision_guard(
    scored_thresholds: Sequence[tuple[float, dict]],
    optimize_key: str,
    min_precision: float = 0.1,
    min_recall: float = 0.05,
    metric_tolerance: float = 0.98,
    cold_start_metric_floor: float = 1e-4,
) -> tuple[float, dict]:
    if not scored_thresholds:
        raise ValueError("No scored thresholds provided")

    metric_key = optimize_key if optimize_key in {"iou", "dice", "f1"} else "f1"
    baseline_threshold, baseline_metrics = max(scored_thresholds, key=lambda item: item[1][metric_key])
    baseline_metric = float(baseline_metrics[metric_key])

    # During cold start when the target metric is essentially zero, avoid
    # precision-driven extreme thresholds (e.g., all-background at 0.9).
    # Prefer the threshold nearest to 0.5 to keep gradients/metrics informative.
    if baseline_metric <= float(cold_start_metric_floor):
        neutral_threshold, neutral_metrics = min(scored_thresholds, key=lambda item: abs(float(item[0]) - 0.5))
        return float(neutral_threshold), neutral_metrics

    eligible = [
        (threshold, metrics)
        for threshold, metrics in scored_thresholds
        if (
            float(metrics.get("precision", 0.0)) >= float(min_precision)
            and float(metrics.get("recall", 0.0)) >= float(min_recall)
        )
    ]
    if not eligible:
        return float(baseline_threshold), baseline_metrics

    metric_floor = baseline_metric * float(metric_tolerance)
    close = [
        (threshold, metrics)
        for threshold, metrics in eligible
        if float(metrics[metric_key]) >= metric_floor
    ]
    candidate_pool = close if close else eligible

    selected_threshold, selected_metrics = max(
        candidate_pool,
        key=lambda item: (float(item[1][metric_key]), float(item[1].get("precision", 0.0))),
    )
    return float(selected_threshold), selected_metrics


def save_checkpoint(
    path: Path,
    model: HybridNGIML,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    cfg: TrainConfig,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema_model: Optional[HybridNGIML] = None,
    use_ema_for_model_state: bool = False,
) -> None:
    model_state = ema_model.state_dict() if (use_ema_for_model_state and ema_model is not None) else model.state_dict()
    ckpt = Checkpoint(
        epoch=epoch,
        global_step=global_step,
        model_state=model_state,
        raw_model_state=model.state_dict() if (use_ema_for_model_state and ema_model is not None) else None,
        ema_state=ema_model.state_dict() if ema_model is not None else None,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
        train_config=asdict(cfg),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt.__dict__, path)


def append_checkpoint_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def _backup_corrupt(existing: Path) -> None:
        try:
            ts = int(time.time())
            corrupt = existing.with_name(f"{existing.name}.corrupt.{ts}")
            existing.replace(corrupt)
            print(f"Backed up corrupt checkpoint log to {corrupt}")
        except Exception:
            pass

    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                records = [item for item in payload if isinstance(item, dict)]
            elif isinstance(payload, dict):
                records = [payload]
        except Exception as exc:
            # Preserve the corrupt file for inspection and continue with empty records
            print(f"Warning: failed to read existing checkpoint log {path}: {exc}")
            _backup_corrupt(path)
            records = []
    else:
        legacy_jsonl = path.with_suffix(".jsonl")
        if legacy_jsonl.exists() and legacy_jsonl.stat().st_size > 0:
            try:
                with open(legacy_jsonl, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if isinstance(item, dict):
                            records.append(item)
            except Exception as exc:
                print(f"Warning: failed to read legacy jsonl checkpoint log {legacy_jsonl}: {exc}")
                _backup_corrupt(legacy_jsonl)
                records = []

    records.append(record)

    # Write atomically to avoid truncation/corruption from concurrent writers or syncs.
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def load_checkpoint(
    path: Path,
    model: HybridNGIML,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema_model: Optional[HybridNGIML] = None,
) -> Tuple[int, int]:
    # Attempt to load the requested checkpoint; if it's corrupt/unreadable,
    # try earlier "checkpoint_epoch_*.pt" files in the same directory as fallbacks.
    original_exc: Exception | None = None

    def _attempt_load(p: Path):
        try:
            return torch.load(p, map_location=device), p
        except Exception as exc:
            return exc, p

    loaded_obj, loaded_path = _attempt_load(path)
    if isinstance(loaded_obj, Exception):
        original_exc = loaded_obj
        print(f"Failed to load checkpoint {path}: {loaded_obj}")
        # Search for other checkpoint candidates in the same directory
        cand_dir = path.parent
        try:
            candidates = sorted(cand_dir.glob("checkpoint_epoch_*.pt"), key=_checkpoint_epoch)
        except Exception:
            candidates = []

        for cand in reversed(candidates):
            if cand == path:
                continue
            cand_obj, cand_path = _attempt_load(cand)
            if not isinstance(cand_obj, Exception):
                print(f"Loaded fallback checkpoint {cand}")
                loaded_obj, loaded_path = cand_obj, cand_path
                break
            else:
                print(f"Skipping unreadable checkpoint {cand}: {cand_obj}")

    if isinstance(loaded_obj, Exception):
        # Nothing usable found
        raise RuntimeError(f"Unable to load checkpoint {path} or any fallback checkpoints: {original_exc}") from original_exc

    data = loaded_obj
    model_state = data.get("raw_model_state") or data["model_state"]
    model.load_state_dict(model_state)
    if ema_model is not None:
        if data.get("ema_state") is not None:
            ema_model.load_state_dict(data["ema_state"])
        else:
            ema_model.load_state_dict(model.state_dict())
    optimizer.load_state_dict(data["optimizer_state"])
    if scheduler is not None and data.get("scheduler_state") is not None:
        scheduler.load_state_dict(data["scheduler_state"])
    if data.get("scaler_state") and scaler.is_enabled():
        scaler.load_state_dict(data["scaler_state"])
    # Prefer checkpoint-internal metadata, but fall back to parsing the
    # filename when epoch is missing or zero (common for some exported weights).
    raw_epoch = data.get("epoch")
    if raw_epoch is None or int(raw_epoch) == 0:
        parsed_epoch = _checkpoint_epoch(loaded_path) if loaded_path is not None else -1
        if parsed_epoch > 0:
            start_epoch = parsed_epoch
        else:
            start_epoch = int(raw_epoch or 0)
    else:
        start_epoch = int(raw_epoch)

    global_step = int(data.get("global_step", 0))
    if global_step == 0:
        # If global_step missing, try to infer from filename or leave as 0.
        pass

    return start_epoch, global_step


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=_checkpoint_epoch)
    return candidates[-1] if candidates else None


@torch.inference_mode()
def compute_foreground_pixel_ratio(loader, max_batches: int | None = 200) -> float:
    """Compute foreground pixel ratio with optional batch sampling and progress prints.

    This avoids iterating the entire dataset in slow or memory-constrained environments.
    """
    foreground = 0.0
    total = 0.0
    for i, batch in enumerate(loader):
        masks = batch["masks"]
        masks = (masks > 0.5).float()
        foreground += float(masks.sum().item())
        total += float(masks.numel())
        if (i + 1) % 10 == 0:
            print(f"Foreground sampling: processed {i+1} batches")
        if max_batches is not None and (i + 1) >= int(max_batches):
            print(f"Foreground sampling: reached max_batches={max_batches}, stopping early")
            break
    if total <= 0:
        return 0.0
    return foreground / total


def _metric_for_monitor(metrics: dict, monitor: str) -> float:
    key = str(monitor).strip().lower()
    if key == "loss":
        return -float(metrics["loss"])
    if key not in metrics:
        raise KeyError(f"Unsupported monitor metric: {monitor}")
    return float(metrics[key])


def _scale_optimizer_and_scheduler_for_phase2(
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    lr_scale: float,
) -> None:
    scale = float(lr_scale)
    if scale <= 0.0:
        raise ValueError("phase-2 lr_scale must be > 0")

    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) * scale
        if "initial_lr" in group:
            group["initial_lr"] = float(group["initial_lr"]) * scale

    if scheduler is not None:
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [float(lr) * scale for lr in scheduler.base_lrs]
        if hasattr(scheduler, "_last_lr") and getattr(scheduler, "_last_lr") is not None:
            scheduler._last_lr = [float(lr) * scale for lr in scheduler._last_lr]


def _build_phase2_config(cfg: TrainConfig, best_iou_path: Path) -> TrainConfig:
    phase2_metric = str(cfg.auto_phase2_monitor).strip().lower()
    if phase2_metric not in {"iou", "f1", "dice"}:
        phase2_metric = "iou"

    phase2_model_cfg = deepcopy(cfg.model_config) if cfg.model_config is not None else None
    optimizer_cfg = getattr(phase2_model_cfg, "optimizer", None) if phase2_model_cfg is not None else None
    if optimizer_cfg is not None:
        for group_name in ("efficientnet", "swin", "residual", "fusion", "decoder"):
            group = getattr(optimizer_cfg, group_name, None)
            if group is None:
                continue
            group.lr = float(group.lr) * float(cfg.auto_phase2_lr_scale)

    return replace(
        cfg,
        training_phase="phase2",
        resume=str(best_iou_path),
        auto_resume=False,
        warmup_epochs=0,
        early_stopping_monitor=phase2_metric,
        threshold_metric=phase2_metric,
        tversky_weight=float(cfg.auto_phase2_tversky_weight),
        lovasz_weight=0.0,
        hard_mining_enabled=False,
        model_config=phase2_model_cfg,
    )


def _set_backbone_trainable(model: HybridNGIML, trainable: bool) -> None:
    for module_name in ("efficientnet", "swin"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = bool(trainable)


def _sample_has_mask_high_pass_edge(record) -> tuple[bool, bool, bool]:
    has_mask = bool(record.mask_path)
    has_high_pass = bool(record.high_pass_path)
    has_edge_mask = bool(getattr(record, "edge_mask_path", None))
    image_path = str(record.image_path)
    if not image_path.endswith(".npz"):
        return has_mask, has_high_pass, has_edge_mask

    try:
        if "::" in image_path:
            archive_path, member_name = image_path.split("::", 1)
            with tarfile.open(archive_path, "r:*") as tf:
                member = tf.extractfile(member_name)
                if member is None:
                    raise FileNotFoundError(f"Missing member {member_name} in {archive_path}")
                with np.load(io.BytesIO(member.read()), allow_pickle=False) as npz_data:
                    has_mask = has_mask or ("mask" in npz_data and npz_data["mask"].size > 0)
                    has_high_pass = has_high_pass or ("high_pass" in npz_data and npz_data["high_pass"].size > 0)
                    has_edge_mask = has_edge_mask or ("edge_mask" in npz_data and npz_data["edge_mask"].size > 0)
        else:
            with np.load(image_path, allow_pickle=False) as npz_data:
                has_mask = has_mask or ("mask" in npz_data and npz_data["mask"].size > 0)
                has_high_pass = has_high_pass or ("high_pass" in npz_data and npz_data["high_pass"].size > 0)
                has_edge_mask = has_edge_mask or ("edge_mask" in npz_data and npz_data["edge_mask"].size > 0)
    except Exception as exc:
        raise ValueError(f"Failed to inspect NPZ sample for mask/high_pass fields: {image_path}") from exc

    return has_mask, has_high_pass, has_edge_mask


def _print_and_validate_train_dataset_integrity(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    train_samples = [sample for sample in manifest.samples if sample.split == "train"]
    if not train_samples:
        raise ValueError("Train split has no samples; cannot start training")

    per_dataset_counts: Dict[str, int] = {}
    real_count = 0
    fake_count = 0
    mask_count = 0
    high_pass_count = 0
    edge_mask_count = 0

    for sample in train_samples:
        per_dataset_counts[sample.dataset] = per_dataset_counts.get(sample.dataset, 0) + 1
        label = int(sample.label)
        if label == 0:
            real_count += 1
        elif label == 1:
            fake_count += 1
        else:
            raise ValueError(f"Unexpected train label {label} for sample: {sample.image_path}")

        has_mask, has_high_pass, has_edge_mask = _sample_has_mask_high_pass_edge(sample)
        if has_mask:
            mask_count += 1
        if has_high_pass:
            high_pass_count += 1
        if has_edge_mask:
            edge_mask_count += 1

    total = len(train_samples)
    print("Train dataset integrity summary")
    print("  Per-dataset sample counts:")
    for dataset_name in sorted(per_dataset_counts):
        print(f"    {dataset_name}: {per_dataset_counts[dataset_name]}")

    fake_ratio = fake_count / max(total, 1)
    real_ratio = real_count / max(total, 1)
    print(
        "  Class ratio (real/fake): "
        f"{real_count}/{fake_count} "
        f"(real={real_ratio:.3f}, fake={fake_ratio:.3f})"
    )
    print(
        "  Coverage: "
        f"masks={100.0 * (mask_count / max(total, 1)):.1f}% "
        f"high_pass={100.0 * (high_pass_count / max(total, 1)):.1f}% "
        f"edge_masks={100.0 * (edge_mask_count / max(total, 1)):.1f}%"
    )

    if fake_count <= 0:
        raise ValueError(
            "Train split has no positive (fake) samples. "
            "Expected at least one sample with label=1."
        )

    minority_ratio = min(real_count, fake_count) / max(total, 1)
    if minority_ratio < 0.01:
        raise ValueError(
            "Train split class ratio is extreme "
            f"(real={real_count}, fake={fake_count}, total={total}). "
            "Please rebalance data before training."
        )


def _manifest_split_counts(manifest_path: Path) -> dict[str, int]:
    manifest = load_manifest(manifest_path)
    counts = {"train": 0, "val": 0, "test": 0}
    unknown_splits: set[str] = set()
    for sample in manifest.samples:
        split_name = str(sample.split).strip().lower()
        if split_name in counts:
            counts[split_name] += 1
        else:
            unknown_splits.add(split_name)
    if unknown_splits:
        unknown = ", ".join(sorted(unknown_splits))
        raise ValueError(
            "Manifest contains unsupported split names: "
            f"{unknown}. Expected only train/val/test."
        )
    return counts


def _validate_startup_config(cfg: TrainConfig, manifest_path: Path, device: torch.device) -> tuple[dict[str, int], str]:
    manifest = load_manifest(manifest_path)
    normalization_mode = str(manifest.normalization_mode).strip().lower()
    if normalization_mode not in {"zero_one", "imagenet"}:
        raise ValueError(
            "Manifest normalization_mode is incompatible with runtime expectations: "
            f"{manifest.normalization_mode!r}. Supported values: 'zero_one' or 'imagenet'."
        )

    if device.type == "cuda" and normalization_mode not in {"zero_one", "imagenet"}:
        raise ValueError(
            "CUDA runtime normalization path only supports 'zero_one' and 'imagenet'. "
            f"Got: {manifest.normalization_mode!r}."
        )

    if cfg.optimize_threshold:
        if cfg.threshold_step <= 0:
            raise ValueError(
                "Invalid threshold search range: threshold_step must be > 0 when optimize_threshold is enabled. "
                f"Got {cfg.threshold_step}."
            )
        if not (0.0 <= float(cfg.threshold_start) <= 1.0 and 0.0 <= float(cfg.threshold_end) <= 1.0):
            raise ValueError(
                "Invalid threshold search range: threshold_start/threshold_end must be within [0, 1]. "
                f"Got start={cfg.threshold_start}, end={cfg.threshold_end}."
            )
        if float(cfg.threshold_end) < float(cfg.threshold_start):
            raise ValueError(
                "Invalid threshold search range: threshold_end must be >= threshold_start. "
                f"Got start={cfg.threshold_start}, end={cfg.threshold_end}."
            )
    else:
        if not (0.0 <= float(cfg.metric_threshold) <= 1.0):
            raise ValueError(
                "Invalid fixed threshold: metric_threshold must be within [0, 1] when optimize_threshold is disabled. "
                f"Got {cfg.metric_threshold}."
            )

    split_counts = _manifest_split_counts(manifest_path)
    train_count = int(split_counts.get("train", 0))
    val_count = int(split_counts.get("val", 0))

    if train_count <= 0:
        raise ValueError("Manifest train split has no samples; cannot start training.")

    if cfg.val_every > 0 and val_count <= 0:
        raise ValueError(
            "Validation is enabled (val_every > 0) but manifest has no val split samples. "
            "Provide a val split or set val_every <= 0."
        )

    if cfg.early_stopping_patience > 0 and val_count <= 0:
        raise ValueError(
            "Early stopping requires validation data, but manifest has no val split samples."
        )

    if cfg.optimize_threshold and val_count <= 0:
        raise ValueError(
            "Threshold optimization requires validation data, but manifest has no val split samples."
        )

    return split_counts, normalization_mode


def _parity_check(cfg: TrainConfig, manifest_path: Path, normalization_mode: str) -> None:
    manifest = load_manifest(manifest_path)
    train_labels = [int(sample.label) for sample in manifest.samples if str(sample.split).strip().lower() == "train"]
    total = len(train_labels)
    positives = sum(1 for label in train_labels if label == 1)
    negatives = sum(1 for label in train_labels if label == 0)
    fake_ratio = (float(positives) / float(total)) if total > 0 else 0.0
    threshold_policy = "optimized" if cfg.optimize_threshold else f"fixed@{float(cfg.metric_threshold):.3f}"

    print(
        "Parity check | "
        f"normalization={normalization_mode} | "
        f"train_class_ratio(real/fake)={negatives}/{positives} (fake={fake_ratio:.3f}) | "
        f"balanced_sampler_active={bool(cfg.balance_real_fake)} | "
        f"eval_threshold_policy={threshold_policy}"
    )


def _print_resolved_config_summary(cfg: TrainConfig, normalization_mode: str) -> None:
    threshold_mode = (
        f"optimized[{cfg.threshold_metric}:{cfg.threshold_start:.2f}-{cfg.threshold_end:.2f}@{cfg.threshold_step:.3f}]"
        if cfg.optimize_threshold
        else f"fixed[{cfg.metric_threshold:.2f}]"
    )
    sampler_mode = "round_robin_balanced" if cfg.balance_real_fake else "round_robin"
    print(
        "Resolved config | "
        f"normalization={normalization_mode} | "
        f"balance_sampling={bool(cfg.balance_sampling)} | "
        f"balance_real_fake={bool(cfg.balance_real_fake)} | "
        f"sampler_mode={sampler_mode} | "
        f"threshold_mode={threshold_mode}"
    )


def _cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            pass
    try:
        major, _minor = torch.cuda.get_device_capability()
        return int(major) >= 8
    except Exception:
        return False


def _resolve_cuda_runtime_stability(cfg: TrainConfig, device: torch.device) -> TrainConfig:
    if device.type != "cuda":
        return cfg

    updates: dict[str, object] = {}
    precision = (getattr(cfg, "precision", "fp32") or "fp32").lower()
    if precision == "bf16" and not _cuda_supports_bf16():
        updates["precision"] = "fp16"

    if updates:
        resolved = replace(cfg, **updates)
        print(
            "Adjusted CUDA runtime config for stability | "
            f"precision: {cfg.precision} -> {resolved.precision}"
        )
        return resolved
    return cfg


def _is_cudnn_engine_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "unable to find an engine" in msg
        or "find was unable to find an engine" in msg
        or "no engine to execute this computation" in msg
    )


def _write_best_threshold_metadata(
    path: Path,
    *,
    epoch: int,
    threshold: float | None,
    threshold_metric: str,
    monitor: str,
    monitor_value: float,
    metrics: dict,
    checkpoint_path: Path,
) -> dict:
    payload = {
        "epoch": int(epoch),
        "checkpoint_path": str(checkpoint_path),
        "threshold": float(threshold) if threshold is not None else None,
        "threshold_metric": str(threshold_metric),
        "monitor": str(monitor),
        "monitor_value": float(monitor_value),
        "val_iou": float(metrics.get("iou")) if metrics.get("iou") is not None else None,
        "val_dice": float(metrics.get("dice")) if metrics.get("dice") is not None else None,
        "val_f1": float(metrics.get("f1")) if metrics.get("f1") is not None else None,
        "val_precision": float(metrics.get("precision")) if metrics.get("precision") is not None else None,
        "val_recall": float(metrics.get("recall")) if metrics.get("recall") is not None else None,
        "val_accuracy": float(metrics.get("accuracy")) if metrics.get("accuracy") is not None else None,
        "val_size_bins": metrics.get("size_bins"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _get_git_hash() -> str | None:
    git_dir = ROOT
    head_path = git_dir / ".git" / "HEAD"
    if not head_path.exists():
        return None

    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split("ref:", 1)[1].strip()
            ref_path = git_dir / ".git" / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            return None
        return head
    except Exception:
        return None


def _init_ema_model(model: HybridNGIML, model_cfg: HybridNGIMLConfig, enabled: bool) -> Optional[HybridNGIML]:
    if not enabled:
        return None
    ema_model = HybridNGIML(model_cfg)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def _update_ema_model(ema_model: Optional[HybridNGIML], model: HybridNGIML, decay: float) -> None:
    if ema_model is None:
        return
    decay = float(min(max(decay, 0.0), 0.999999))
    msd = model.state_dict()
    for key, value in ema_model.state_dict().items():
        model_value = msd[key].detach()
        if not torch.is_floating_point(value):
            value.copy_(model_value)
        else:
            value.mul_(decay).add_(model_value, alpha=1.0 - decay)


def _size_bin_name(fg_ratio: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    small_max = float(max(0.0, cfg.small_mask_ratio_max))
    medium_max = float(max(small_max, cfg.medium_mask_ratio_max))
    bins = torch.full_like(fg_ratio, 2, dtype=torch.long)
    bins = torch.where(fg_ratio <= small_max, torch.zeros_like(bins), bins)
    bins = torch.where((fg_ratio > small_max) & (fg_ratio <= medium_max), torch.ones_like(bins), bins)
    return bins


def _empty_bin_stats() -> Dict[str, Dict[str, float]]:
    return {
        "small": {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0, "count": 0.0},
        "medium": {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0, "count": 0.0},
        "large": {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0, "count": 0.0},
    }


def _finalize_bin_stats(bin_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, stats in bin_stats.items():
        metrics = _metrics_from_counts(stats["tp"], stats["tn"], stats["fp"], stats["fn"])
        out[name] = {
            "count": float(stats["count"]),
            "dice": float(metrics["dice"]),
            "iou": float(metrics["iou"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "accuracy": float(metrics["accuracy"]),
        }
    return out


def _to_float_label_ratio(labels: torch.Tensor) -> tuple[float, float]:
    labels_f = labels.float()
    positives = float((labels_f >= 0.5).sum().item())
    total = float(labels_f.numel())
    return positives, total


def _write_experiment_fingerprint(
    out_dir: Path,
    cfg: TrainConfig,
    resolved_manifest: Path,
    class_ratio: float | None = None,
    chosen_threshold: float | None = None,
) -> Path:
    cfg_dict = asdict(cfg)
    cfg_json = json.dumps(cfg_dict, sort_keys=True, default=str)
    fingerprint = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()

    payload = {
        "fingerprint": fingerprint,
        "created_at_unix": float(time.time()),
        "git_hash": _get_git_hash(),
        "manifest": str(resolved_manifest),
        "seed": int(cfg.seed),
        "class_ratio": float(class_ratio) if class_ratio is not None else None,
        "chosen_threshold": float(chosen_threshold) if chosen_threshold is not None else None,
        "sampler": {
            "mode": "round_robin_balanced" if cfg.balance_real_fake else "round_robin",
            "balance_sampling": bool(cfg.balance_sampling),
            "balance_real_fake": bool(cfg.balance_real_fake),
            "balanced_positive_ratio": float(cfg.balanced_positive_ratio),
            "round_robin_seed": cfg.round_robin_seed,
            "balanced_sampler_seed": cfg.balanced_sampler_seed,
        },
        "loss": {
            "hybrid_mode": cfg.loss_hybrid_mode,
            "dice_weight": float(cfg.dice_weight),
            "bce_weight": float(cfg.bce_weight),
            "tversky_weight": float(cfg.tversky_weight),
            "tversky_alpha": float(cfg.tversky_alpha),
            "tversky_beta": float(cfg.tversky_beta),
            "use_boundary_loss": bool(cfg.use_boundary_loss),
            "boundary_weight": float(cfg.boundary_weight),
            "hard_mining_enabled": bool(cfg.hard_mining_enabled),
            "hard_mining_start_epoch": int(cfg.hard_mining_start_epoch),
            "hard_mining_weight": float(cfg.hard_mining_weight),
            "hard_mining_gamma": float(cfg.hard_mining_gamma),
        },
        "ema": {
            "enabled": bool(cfg.ema_enabled),
            "decay": float(cfg.ema_decay),
        },
        "threshold_search": {
            "enabled": bool(cfg.optimize_threshold),
            "metric": cfg.threshold_metric,
            "start": float(cfg.threshold_start),
            "end": float(cfg.threshold_end),
            "step": float(cfg.threshold_step),
            "fixed_threshold": float(cfg.metric_threshold),
        },
    }

    path = out_dir / "experiment_fingerprint.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _update_experiment_fingerprint(path: Path, updates: dict) -> None:
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return

    payload.update(updates)
    payload["updated_at_unix"] = float(time.time())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_one_epoch(
    model: HybridNGIML,
    loader,
    optimizer,
    scaler: GradScaler,
    loss_fn,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
    ema_model: Optional[HybridNGIML] = None,
    per_dataset_aug: Optional[Dict[str, AugmentationConfig]] = None,
    normalization_mode: Optional[str] = None,
):
    model.train()
    # --- Staged freezing/unfreezing for fine-tuning ONLY ---
    # Only apply if in fine-tuning mode (resume checkpoint set or training_phase == 'phase2')
    is_finetune = False
    if hasattr(cfg, 'training_phase') and str(getattr(cfg, 'training_phase', '')).lower() == 'phase2':
        is_finetune = True
    elif hasattr(cfg, 'resume') and cfg.resume:
        is_finetune = True

    if is_finetune:
        freeze_backbone_epochs = getattr(getattr(cfg, 'model_config', None), 'optimizer', None)
        if freeze_backbone_epochs is not None:
            freeze_backbone_epochs = getattr(freeze_backbone_epochs, 'freeze_backbone_epochs', 3)
        else:
            freeze_backbone_epochs = 3

        def set_backbone_blocks_trainable(model, epoch):
            # Freeze all backbone layers for first N epochs
            if epoch < freeze_backbone_epochs:
                for module_name in ("efficientnet", "swin"):
                    module = getattr(model, module_name, None)
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = False
            # Gradually unfreeze only the last block of each backbone after freeze_backbone_epochs
            elif freeze_backbone_epochs <= epoch < 11:
                for module_name in ("efficientnet", "swin"):
                    module = getattr(model, module_name, None)
                    if module is not None:
                        if hasattr(module, 'blocks') and isinstance(module.blocks, (list, nn.ModuleList)):
                            for i, block in enumerate(module.blocks):
                                requires_grad = (i == len(module.blocks) - 1)
                                for param in block.parameters():
                                    param.requires_grad = requires_grad
                        else:
                            for param in module.parameters():
                                param.requires_grad = True
            else:
                for module_name in ("efficientnet", "swin"):
                    module = getattr(model, module_name, None)
                    if module is not None:
                        if hasattr(module, 'blocks') and isinstance(module.blocks, (list, nn.ModuleList)):
                            for i, block in enumerate(module.blocks):
                                requires_grad = (i == len(module.blocks) - 1)
                                for param in block.parameters():
                                    param.requires_grad = requires_grad
                        else:
                            for param in module.parameters():
                                param.requires_grad = False

        set_backbone_blocks_trainable(model, epoch)
    running_loss = 0.0
    num_batches = 0
    sampled_pos = 0.0
    sampled_total = 0.0
    accum_steps = max(1, int(cfg.grad_accum_steps))
    # Wrap loader with prefetcher to overlap host->device copy with GPU compute.
    if device.type == "cuda":
        loader = _PrefetchLoader(loader, device)

    # Determine a batch-level total for tqdm. Avoid falling back to dataset
    # sample count, which can massively overstate steps when using custom
    # samplers/batch samplers.
    def _safe_len(obj) -> Optional[int]:
        if obj is None:
            return None
        try:
            value = len(obj)
        except Exception:
            return None
        try:
            value_int = int(value)
        except Exception:
            return None
        return value_int if value_int >= 0 else None

    total: Optional[int] = None
    base_loader = getattr(loader, "_loader", loader)

    total = _safe_len(loader)
    if total is None:
        total = _safe_len(base_loader)

    if total is None:
        batch_sampler = getattr(base_loader, "batch_sampler", None)
        total = _safe_len(batch_sampler)
        if total is None and batch_sampler is not None:
            base_sampler = getattr(batch_sampler, "base_sampler", None)
            batch_size = int(getattr(batch_sampler, "batch_size", 0) or 0)
            if base_sampler is not None and batch_size > 0:
                sample_count = _safe_len(base_sampler)
                if sample_count is not None:
                    drop_last = bool(getattr(batch_sampler, "drop_last", False))
                    total = (sample_count // batch_size) if drop_last else ((sample_count + batch_size - 1) // batch_size)

    if total is None:
        sampler = getattr(base_loader, "sampler", None)
        batch_size = int(getattr(base_loader, "batch_size", 0) or 0)
        if sampler is not None and batch_size > 0:
            sample_count = _safe_len(sampler)
            if sample_count is not None:
                drop_last = bool(getattr(base_loader, "drop_last", False))
                total = (sample_count // batch_size) if drop_last else ((sample_count + batch_size - 1) // batch_size)

    progress = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True, total=total)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(progress):
        batch_start = time.perf_counter()
        images = batch["images"]
        masks = batch["masks"]
        edge_masks = batch.get("edge_masks")
        edge_mask_present = batch.get("edge_mask_present")
        if images.device != device:
            images = images.to(device, non_blocking=True)
        if masks.device != device:
            masks = masks.to(device, non_blocking=True)
        if isinstance(edge_masks, torch.Tensor) and edge_masks.device != device:
            edge_masks = edge_masks.to(device, non_blocking=True)
        if isinstance(edge_mask_present, torch.Tensor) and edge_mask_present.device != device:
            edge_mask_present = edge_mask_present.to(device, non_blocking=True)
        high_pass = batch.get("high_pass")
        if isinstance(high_pass, torch.Tensor):
            if high_pass.device != device:
                high_pass = high_pass.to(device, non_blocking=True)
        else:
            high_pass = None
        # Apply GPU-side augmentations and normalization when requested.
        aug_start = None
        forward_start = None
        backward_end = None
        opt_end = None
        if device.type == "cuda" and per_dataset_aug is not None and normalization_mode is not None:
            aug_start = time.perf_counter()
            try:
                gen = torch.Generator(device=device)
            except TypeError:
                gen = torch.Generator()
            seed_base = int(cfg.aug_seed) if cfg.aug_seed is not None else int(cfg.seed)
            try:
                gen.manual_seed(seed_base + int(global_step))
            except Exception:
                gen.manual_seed(seed_base)

            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

            datasets_list = batch.get("datasets", None)
            if datasets_list is not None:
                bsz = images.shape[0]
                # Group indices by dataset name so we can batch augment per-dataset
                idxs_by_ds: dict[str, list[int]] = {}
                for i in range(bsz):
                    ds_name = str(datasets_list[i])
                    idxs_by_ds.setdefault(ds_name, []).append(i)

                # Use a single generator seeded per-batch for reproducibility
                gen = gen if 'gen' in locals() else gen
                for ds_name, idxs in idxs_by_ds.items():
                    aug_cfg = per_dataset_aug.get(ds_name, None)
                    if aug_cfg is None or not getattr(aug_cfg, "enable", False):
                        # Apply normalization on-device in batch
                        if normalization_mode == "imagenet":
                            mean = imagenet_mean.view(1, 3, 1, 1)
                            std = imagenet_std.view(1, 3, 1, 1)
                            images[idxs] = (images[idxs] - mean) / std
                        else:
                            # zero_one or other modes: leave as-is
                            images[idxs] = images[idxs]
                        continue

                    # Slice the batch and apply batched augmentations
                    img_slice = images[idxs]
                    mask_slice = masks[idxs]
                    edge_mask_slice = edge_masks[idxs] if isinstance(edge_masks, torch.Tensor) else None
                    hp_slice = None
                    if high_pass is not None:
                        hp_slice = high_pass[idxs]

                    img_out, mask_out, hp_out, edge_mask_out = _apply_gpu_augmentations_batch(
                        img_slice,
                        mask_slice,
                        aug_cfg,
                        high_pass=hp_slice,
                        edge_masks=edge_mask_slice,
                        generator=gen,
                    )

                    # Apply normalization to the augmented slice
                    if normalization_mode == "imagenet":
                        mean = imagenet_mean.view(1, 3, 1, 1)
                        std = imagenet_std.view(1, 3, 1, 1)
                        img_out = (img_out - mean) / std

                    images[idxs] = img_out
                    masks[idxs] = mask_out
                    if hp_out is not None and high_pass is not None:
                        high_pass[idxs] = hp_out
                    if edge_mask_out is not None and isinstance(edge_masks, torch.Tensor):
                        edge_masks[idxs] = edge_mask_out
            aug_end = time.perf_counter()
        else:
            aug_end = None
        labels = batch["labels"]
        pos_count, total_count = _to_float_label_ratio(labels)
        sampled_pos += pos_count
        sampled_total += total_count
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            if high_pass is not None:
                high_pass = high_pass.contiguous(memory_format=torch.channels_last)

        precision_name = (getattr(cfg, "precision", "fp32") or "fp32").lower()
        amp_dtype = torch.bfloat16 if precision_name == "bf16" else (torch.float16 if precision_name == "fp16" else None)
        use_amp = cfg.amp and device.type == "cuda" and (amp_dtype is not None)
        if amp_dtype is not None:
            forward_start = time.perf_counter()
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
                loss = loss_fn(preds, masks, edge_target=edge_masks, edge_target_present=edge_mask_present)
            forward_end = time.perf_counter()
        else:
            forward_start = time.perf_counter()
            preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
            loss = loss_fn(preds, masks, edge_target=edge_masks, edge_target_present=edge_mask_present)
            forward_end = time.perf_counter()

        if cfg.hard_mining_enabled and epoch >= int(max(0, cfg.hard_mining_start_epoch)):
            final_logits = preds[-1]
            if final_logits.shape[-2:] != masks.shape[-2:]:
                final_logits = torch.nn.functional.interpolate(
                    final_logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            bce_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                final_logits,
                masks,
                reduction="none",
            ).mean(dim=(1, 2, 3))

            with torch.no_grad():
                pred_bin = (torch.sigmoid(final_logits) >= 0.5).float()
                tp = (pred_bin * masks).sum(dim=(1, 2, 3))
                fp = (pred_bin * (1.0 - masks)).sum(dim=(1, 2, 3))
                fn = ((1.0 - pred_bin) * masks).sum(dim=(1, 2, 3))
                iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
                difficulty = (1.0 - iou).clamp(0.0, 1.0)
                hard_weights = 1.0 + float(max(0.0, cfg.hard_mining_gamma)) * difficulty
                hard_weights = hard_weights / hard_weights.mean().clamp_min(1e-6)

            hard_loss = (hard_weights * bce_per_sample).mean()
            loss = loss + float(max(0.0, cfg.hard_mining_weight)) * hard_loss

        scaled_loss = loss / accum_steps
        backward_start = time.perf_counter()
        scaler.scale(scaled_loss).backward()
        backward_end = time.perf_counter()

        do_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))

        if do_step:
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            opt_end = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            _update_ema_model(ema_model, model, cfg.ema_decay)

        running_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Record timings if enabled
        if cfg.debug_timing:
            batch_end = time.perf_counter()
            batch_time = batch_end - batch_start
            aug_time = (aug_end - aug_start) if (aug_start is not None and aug_end is not None) else 0.0
            forward_time = (forward_end - forward_start) if (forward_start is not None and forward_end is not None) else 0.0
            backward_time = (backward_end - backward_start) if (backward_end is not None and backward_start is not None) else 0.0
            opt_time = (opt_end - backward_end) if (opt_end is not None and backward_end is not None) else 0.0
            if num_batches % 50 == 0 or num_batches == 1:
                print(
                    f"[timing] batch={num_batches} total={batch_time:.3f}s aug={aug_time:.3f}s forward={forward_time:.3f}s backward={backward_time:.3f}s opt={opt_time:.3f}s"
                )

        avg_loss = running_loss / max(1, num_batches)
        progress.set_postfix(loss=f"{avg_loss:.4f}", step=f"{step:05d}", accum=f"{accum_steps}")

    sampled_positive_ratio = sampled_pos / max(sampled_total, 1.0)
    return running_loss / max(1, num_batches), global_step, sampled_positive_ratio


@torch.inference_mode()
def find_best_threshold(model: HybridNGIML, loader, device: torch.device, cfg: TrainConfig) -> dict:
    model.eval()
    thresholds = _build_threshold_grid(cfg)
    threshold_stats = {
        float(th): {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
        for th in thresholds
    }

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        high_pass = batch.get("high_pass")
        if isinstance(high_pass, torch.Tensor):
            high_pass = high_pass.to(device, non_blocking=True)
        else:
            high_pass = None
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            if high_pass is not None:
                high_pass = high_pass.contiguous(memory_format=torch.channels_last)
        precision_name = (getattr(cfg, "precision", "fp32") or "fp32").lower()
        amp_dtype = torch.bfloat16 if precision_name == "bf16" else (torch.float16 if precision_name == "fp16" else None)
        use_amp = cfg.amp and device.type == "cuda" and (amp_dtype is not None)
        if amp_dtype is not None:
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
        else:
            preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
        logits = preds[-1]

        for threshold in thresholds:
            counts = _segmentation_counts(logits, masks, threshold=threshold)
            threshold_stats[float(threshold)]["tp"] += counts["tp"]
            threshold_stats[float(threshold)]["tn"] += counts["tn"]
            threshold_stats[float(threshold)]["fp"] += counts["fp"]
            threshold_stats[float(threshold)]["fn"] += counts["fn"]

    optimize_key = cfg.threshold_metric.lower()
    if optimize_key not in {"iou", "dice", "f1"}:
        optimize_key = "f1"

    scored_thresholds: list[tuple[float, dict]] = []
    for threshold in thresholds:
        stats = threshold_stats[float(threshold)]
        metrics = _metrics_from_counts(stats["tp"], stats["tn"], stats["fp"], stats["fn"])
        scored_thresholds.append((float(threshold), metrics))

    best_threshold, best_metrics = _select_threshold_with_precision_guard(scored_thresholds, optimize_key=optimize_key)
    return {
        "threshold": float(best_threshold),
        "threshold_metric": optimize_key,
        "dice": float(best_metrics["dice"]),
        "iou": float(best_metrics["iou"]),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "f1": float(best_metrics["f1"]),
        "accuracy": float(best_metrics["accuracy"]),
    }


@torch.inference_mode()
def evaluate(model: HybridNGIML, loader, loss_fn, device: torch.device, cfg: TrainConfig, normalization_mode: Optional[str] = None) -> dict:
    model.eval()
    total_loss = 0.0
    batches = 0
    thresholds = _build_threshold_grid(cfg) if cfg.optimize_threshold else [float(cfg.metric_threshold)]
    threshold_stats = {
        float(th): {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
        for th in thresholds
    }
    threshold_bin_stats = {
        float(th): _empty_bin_stats()
        for th in thresholds
    }

    progress = tqdm(loader, desc="Validation", leave=False, dynamic_ncols=True)
    for batch in progress:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        edge_masks = batch.get("edge_masks")
        if isinstance(edge_masks, torch.Tensor):
            edge_masks = edge_masks.to(device, non_blocking=True)
        else:
            edge_masks = None
        edge_mask_present = batch.get("edge_mask_present")
        if isinstance(edge_mask_present, torch.Tensor):
            edge_mask_present = edge_mask_present.to(device, non_blocking=True)
        else:
            edge_mask_present = None
        high_pass = batch.get("high_pass")
        if isinstance(high_pass, torch.Tensor):
            high_pass = high_pass.to(device, non_blocking=True)
        else:
            high_pass = None
        # If collate left normalization to be done on-device (collate used zero_one),
        # perform normalization here on the GPU for evaluation.
        if device.type == "cuda" and normalization_mode is not None:
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
            bsz = images.shape[0]
            for i in range(bsz):
                images[i] = _normalize(images[i], normalization_mode, imagenet_mean=imagenet_mean, imagenet_std=imagenet_std)
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            if high_pass is not None:
                high_pass = high_pass.contiguous(memory_format=torch.channels_last)
        precision_name = (getattr(cfg, "precision", "fp32") or "fp32").lower()
        amp_dtype = torch.bfloat16 if precision_name == "bf16" else (torch.float16 if precision_name == "fp16" else None)
        use_amp = cfg.amp and device.type == "cuda" and (amp_dtype is not None)
        if amp_dtype is not None:
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
                loss = loss_fn(preds, masks, edge_target=edge_masks, edge_target_present=edge_mask_present)
        else:
            preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
            loss = loss_fn(preds, masks, edge_target=edge_masks, edge_target_present=edge_mask_present)
        logits = preds[-1]

        with torch.no_grad():
            fg_ratio = masks.float().mean(dim=(1, 2, 3))
            size_bin_idx = _size_bin_name(fg_ratio, cfg)

        for threshold in thresholds:
            counts = _segmentation_counts(logits, masks, threshold=threshold)
            threshold_stats[float(threshold)]["tp"] += counts["tp"]
            threshold_stats[float(threshold)]["tn"] += counts["tn"]
            threshold_stats[float(threshold)]["fp"] += counts["fp"]
            threshold_stats[float(threshold)]["fn"] += counts["fn"]

            pred = (torch.sigmoid(logits) >= float(threshold)).float()
            target = masks.float()
            tp_b = (pred * target).sum(dim=(1, 2, 3))
            tn_b = ((1.0 - pred) * (1.0 - target)).sum(dim=(1, 2, 3))
            fp_b = (pred * (1.0 - target)).sum(dim=(1, 2, 3))
            fn_b = ((1.0 - pred) * target).sum(dim=(1, 2, 3))

            bin_names = ["small", "medium", "large"]
            for bin_id, bin_name in enumerate(bin_names):
                mask_sel = size_bin_idx == bin_id
                if not torch.any(mask_sel):
                    continue
                stats = threshold_bin_stats[float(threshold)][bin_name]
                stats["tp"] += float(tp_b[mask_sel].sum().item())
                stats["tn"] += float(tn_b[mask_sel].sum().item())
                stats["fp"] += float(fp_b[mask_sel].sum().item())
                stats["fn"] += float(fn_b[mask_sel].sum().item())
                stats["count"] += float(mask_sel.sum().item())

        total_loss += loss.item()
        batches += 1
        progress.set_postfix(loss=f"{(total_loss / max(1, batches)):.4f}", step=f"{batches:05d}")

    optimize_key = cfg.threshold_metric.lower()
    if optimize_key not in {"iou", "dice", "f1"}:
        optimize_key = "f1"

    scored_thresholds: list[tuple[float, dict]] = []
    for threshold in thresholds:
        stats = threshold_stats[float(threshold)]
        metrics = _metrics_from_counts(
            stats["tp"],
            stats["tn"],
            stats["fp"],
            stats["fn"],
        )
        scored_thresholds.append((float(threshold), metrics))

    if cfg.optimize_threshold:
        best_threshold, best_metrics = _select_threshold_with_precision_guard(scored_thresholds, optimize_key=optimize_key)
    else:
        fixed_threshold = float(cfg.metric_threshold)
        nearest_threshold, best_metrics = min(scored_thresholds, key=lambda item: abs(item[0] - fixed_threshold))
        best_threshold = float(nearest_threshold)

    best_bin_metrics = _finalize_bin_stats(threshold_bin_stats[float(best_threshold)])

    normalizer = max(1, batches)
    return {
        "loss": total_loss / normalizer,
        "dice": float(best_metrics["dice"]),
        "iou": float(best_metrics["iou"]),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "f1": float(best_metrics["f1"]),
        "accuracy": float(best_metrics["accuracy"]),
        "threshold": float(best_threshold),
        "threshold_metric": optimize_key,
        "size_bins": best_bin_metrics,
    }


def run_training(cfg: TrainConfig) -> None:
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    startup_t0 = time.time()

    if cfg.cuda_expandable_segments and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Force device to CUDA when available per user request
    if torch.cuda.is_available():
        cfg = replace(cfg, device="cuda")
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    # Force precision to bfloat16 at runtime per user request
    cfg = replace(cfg, precision="bf16")
    cfg = _resolve_cuda_runtime_stability(cfg, device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high" if cfg.use_tf32 else "highest")

        # PyTorch 2.9+ prefers fp32_precision knobs over allow_tf32 flags.
        cudnn_backend = getattr(torch.backends, "cudnn", None)
        cuda_backend = getattr(torch.backends, "cuda", None)
        cudnn_conv = getattr(cudnn_backend, "conv", None)
        cuda_matmul = getattr(cuda_backend, "matmul", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = "tf32" if cfg.use_tf32 else "ieee"
        elif cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = cfg.use_tf32

        if cuda_matmul is not None and hasattr(cuda_matmul, "fp32_precision"):
            cuda_matmul.fp32_precision = "tf32" if cfg.use_tf32 else "ieee"
        elif cuda_matmul is not None and hasattr(cuda_matmul, "allow_tf32"):
            cuda_matmul.allow_tf32 = cfg.use_tf32

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_manifest = _resolve_manifest_for_training(cfg, out_dir)
    t_after_manifest = time.time()
    if resolved_manifest != Path(cfg.manifest):
        cfg = replace(cfg, manifest=str(resolved_manifest))

    _, normalization_mode_checked = _validate_startup_config(cfg, Path(cfg.manifest), device)
    _print_resolved_config_summary(cfg, normalization_mode_checked)
    _parity_check(cfg, Path(cfg.manifest), normalization_mode_checked)

    _print_and_validate_train_dataset_integrity(Path(cfg.manifest))

    loaders, per_dataset_aug, normalization_mode = _prepare_dataloaders(cfg, device)
    t_after_dataloaders = time.time()
    if "train" not in loaders:
        raise ValueError("Train split missing in manifest; cannot start training")
    if cfg.balance_real_fake:
        print(
            "Train sampler: round-robin + real/fake balanced | "
            f"target_positive_ratio={cfg.balanced_positive_ratio:.3f} | "
            f"num_samples={cfg.balanced_sampler_num_samples or 'dataset_len'}"
        )

    foreground_ratio = None
    if cfg.compute_foreground_ratio:
        sampled_batches = cfg.foreground_ratio_max_batches if cfg.foreground_ratio_max_batches > 0 else None
        if sampled_batches is None:
            print("Computing foreground pixel ratio (sampling full train loader)...")
        else:
            print(f"Computing foreground pixel ratio (sampling up to {sampled_batches} batches)...")
        foreground_ratio = compute_foreground_pixel_ratio(loaders["train"], max_batches=sampled_batches)
        print(f"Foreground pixel ratio (train): {foreground_ratio:.6f}")

    model_cfg = cfg.model_config or HybridNGIMLConfig()
    # Honor the training-level gradient checkpointing toggle when instantiating the model
    try:
        from dataclasses import replace as _dc_replace

        model_cfg = _dc_replace(model_cfg, gradient_checkpointing=cfg.gradient_checkpointing)
    except Exception:
        model_cfg.gradient_checkpointing = cfg.gradient_checkpointing

    model = HybridNGIML(model_cfg).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = model.build_optimizer()
    scheduler = _build_lr_scheduler(optimizer, cfg)
    # GradScaler is only required for fp16; bf16 on A100 benefits from native autocast without scaling.
    scaler = GradScaler(enabled=(str(cfg.precision).lower() == "fp16" and cfg.amp and device.type == "cuda"))
    ema_model = _init_ema_model(model, model_cfg, cfg.ema_enabled)
    if ema_model is not None:
        ema_model = ema_model.to(device)
        if cfg.channels_last and device.type == "cuda":
            ema_model = ema_model.to(memory_format=torch.channels_last)
    base_loss_cfg = cfg.loss_config or MultiStageLossConfig()
    loss_cfg = replace(
        base_loss_cfg,
        hybrid_mode=cfg.loss_hybrid_mode,
        dice_weight=cfg.dice_weight,
        bce_weight=cfg.bce_weight,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
        tversky_weight=cfg.tversky_weight,
        tversky_alpha=cfg.tversky_alpha,
        tversky_beta=cfg.tversky_beta,
        lovasz_weight=cfg.lovasz_weight,
        use_boundary_loss=cfg.use_boundary_loss,
        boundary_weight=cfg.boundary_weight,
    )
    if cfg.auto_pos_weight and foreground_ratio is not None:
        ratio = max(1e-6, min(1.0 - 1e-6, foreground_ratio))
        pos_weight = (1.0 - ratio) / ratio
        pos_weight = float(min(max(pos_weight, cfg.pos_weight_min), cfg.pos_weight_max))
        if cfg.balance_real_fake:
            cap = float(getattr(cfg, "balanced_pos_weight_cap", 0.0))
            if cap > 0:
                capped = min(pos_weight, cap)
                if capped < pos_weight:
                    print(
                        "Balanced class sampling is enabled; capping auto pos_weight "
                        f"from {pos_weight:.4f} to {capped:.4f}"
                    )
                pos_weight = capped
        loss_cfg = replace(loss_cfg, pos_weight=pos_weight)
        print(f"Auto pos_weight from foreground ratio: {pos_weight:.4f}")
    else:
        fixed_pos_weight = float(getattr(loss_cfg, "pos_weight", 1.0))
        print(f"Using fixed pos_weight from loss config: {fixed_pos_weight:.4f}")
    loss_fn = MultiStageManipulationLoss(loss_cfg)
    t_after_model = time.time()

    print(
        "Startup timings | "
        f"manifest/cache {t_after_manifest - startup_t0:.1f}s | "
        f"dataloaders {t_after_dataloaders - t_after_manifest:.1f}s | "
        f"model+optim {t_after_model - t_after_dataloaders:.1f}s | "
        f"total {t_after_model - startup_t0:.1f}s"
    )

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_log_path = checkpoint_dir / "checkpoint_metrics.json"

    start_epoch = 0
    global_step = 0
    resume_path: Optional[Path] = None
    if cfg.resume:
        resume_path = Path(cfg.resume)
    elif cfg.auto_resume:
        resume_path = find_latest_checkpoint(out_dir)
        if resume_path is not None:
            print(f"Auto-resume selected latest checkpoint: {resume_path}")

    if resume_path:
        if resume_path.is_file():
            start_epoch, global_step = load_checkpoint(
                resume_path,
                model,
                optimizer,
                scaler,
                device,
                scheduler=scheduler,
                ema_model=ema_model,
            )
            print(f"Resumed from {resume_path} at epoch {start_epoch} step {global_step}")
        else:
            print(f"Resume path {resume_path} not found; starting fresh")
    elif cfg.auto_resume:
        print("Auto-resume enabled but no checkpoint found; starting fresh")

    if cfg.compile_model:
        if ema_model is not None:
            print("torch.compile skipped because EMA is enabled (keeps EMA/state_dict keys consistent)")
        elif hasattr(torch, "compile"):
            if device.type == "cuda" and cfg.compile_mode == "reduce-overhead":
                try:
                    import torch._inductor.config as inductor_config

                    if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                        inductor_config.triton.cudagraphs = False
                        print("Disabled Triton CUDA graphs for reduce-overhead compile mode to reduce memory pressure")
                except Exception:
                    pass
            model = torch.compile(model, mode=cfg.compile_mode)
            print(f"torch.compile enabled with mode={cfg.compile_mode}")
        else:
            print("torch.compile requested but not available in this torch build")

    with open(out_dir / "train_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)
    fingerprint_path = _write_experiment_fingerprint(
        out_dir,
        cfg,
        Path(cfg.manifest),
        class_ratio=foreground_ratio,
        chosen_threshold=None,
    )

    best_monitor_value = float("-inf")
    best_val_iou = float("-inf")
    best_val_f1 = float("-inf")
    best_val_loss = float("inf")
    no_improve_epochs = 0
    early_stopping_enabled = "val" in loaders and cfg.early_stopping_patience > 0
    best_threshold_path = checkpoint_dir / "best_threshold.json"
    auto_phase2_triggered = str(getattr(cfg, "training_phase", "phase1")).strip().lower() == "phase2"

    freeze_backbone_epochs = int(max(0, getattr(model_cfg.optimizer, "freeze_backbone_epochs", 0)))
    backbone_was_frozen = False

    runtime_fallback_used = False
    for epoch in range(start_epoch, cfg.epochs):
        should_freeze_backbone = epoch < freeze_backbone_epochs
        _set_backbone_trainable(model, trainable=not should_freeze_backbone)
        if should_freeze_backbone and not backbone_was_frozen:
            print(
                "Backbone freeze enabled: freezing EfficientNet/Swin "
                f"for first {freeze_backbone_epochs} epochs"
            )
            backbone_was_frozen = True
        elif (not should_freeze_backbone) and backbone_was_frozen:
            print("Backbone freeze finished: unfreezing EfficientNet/Swin")
            backbone_was_frozen = False

        start_time = time.time()
        try:
            train_loss, global_step, train_positive_ratio = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                scaler,
                loss_fn,
                device,
                cfg,
                epoch,
                global_step,
                ema_model=ema_model,
                per_dataset_aug=per_dataset_aug,
                normalization_mode=normalization_mode,
            )
        except RuntimeError as exc:
            if (not runtime_fallback_used) and _is_cudnn_engine_error(exc):
                runtime_fallback_used = True
                prev_precision = cfg.precision
                cfg = replace(
                    cfg,
                    channels_last=False,
                    compile_model=False,
                    flash_attention=False,
                    xformers=False,
                    amp=False,
                    precision="fp32",
                )
                model = model.to(memory_format=torch.contiguous_format)
                if ema_model is not None:
                    ema_model = ema_model.to(memory_format=torch.contiguous_format)
                scaler = GradScaler(enabled=False)
                print(
                    "Encountered CUDA conv engine selection error; retrying with safe settings | "
                    f"precision {prev_precision}->fp32, amp off, channels_last off, compile off"
                )
                train_loss, global_step, train_positive_ratio = train_one_epoch(
                    model,
                    loaders["train"],
                    optimizer,
                    scaler,
                    loss_fn,
                    device,
                    cfg,
                    epoch,
                    global_step,
                    ema_model=ema_model,
                    per_dataset_aug=per_dataset_aug,
                    normalization_mode=normalization_mode,
                )
            else:
                raise

        elapsed = time.time() - start_time
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} done | loss {train_loss:.4f} | "
                f"sample_pos_ratio {train_positive_ratio:.3f} | lr {current_lr:.6e} | time {elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:03d} done | loss {train_loss:.4f} | "
                f"sample_pos_ratio {train_positive_ratio:.3f} | time {elapsed:.1f}s"
            )

        val_loss = None
        val_dice = None
        val_iou = None
        val_f1 = None
        val_precision = None
        val_recall = None
        val_accuracy = None
        val_threshold = None
        val_size_bins = None
        if "val" in loaders and (epoch + 1) % cfg.val_every == 0:
            eval_model = ema_model if ema_model is not None else model
            metrics = evaluate(eval_model, loaders["val"], loss_fn, device, cfg, normalization_mode=normalization_mode)
            val_loss = float(metrics["loss"])
            val_dice = float(metrics["dice"])
            val_iou = float(metrics["iou"])
            val_f1 = float(metrics["f1"])
            val_precision = float(metrics["precision"])
            val_recall = float(metrics["recall"])
            val_accuracy = float(metrics["accuracy"])
            val_threshold = float(metrics["threshold"])
            val_size_bins = metrics.get("size_bins")
            print(
                f"Val | loss {val_loss:.4f} | dice {val_dice:.4f} | iou {val_iou:.4f} "
                f"| f1 {val_f1:.4f} | precision {val_precision:.4f} | recall {val_recall:.4f} | accuracy {val_accuracy:.4f} "
                f"| threshold {val_threshold:.2f}"
            )
            if isinstance(val_size_bins, dict):
                small_iou = float(val_size_bins.get("small", {}).get("iou", 0.0))
                medium_iou = float(val_size_bins.get("medium", {}).get("iou", 0.0))
                large_iou = float(val_size_bins.get("large", {}).get("iou", 0.0))
                print(
                    "Val bins | "
                    f"small_iou {small_iou:.4f} | medium_iou {medium_iou:.4f} | large_iou {large_iou:.4f}"
                )

            iou_improved = val_iou > (best_val_iou + cfg.early_stopping_min_delta)
            f1_improved = val_f1 > (best_val_f1 + cfg.early_stopping_min_delta)
            loss_improved = val_loss < (best_val_loss - cfg.early_stopping_min_delta)
            # Save best iou checkpoint as before
            if iou_improved:
                best_val_iou = val_iou
                best_iou_path = checkpoint_dir / "best_iou_checkpoint.pt"
                save_checkpoint(
                    best_iou_path,
                    model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    global_step,
                    cfg,
                    scheduler=scheduler,
                    ema_model=ema_model,
                    use_ema_for_model_state=(ema_model is not None),
                )
                print(f"New best val iou {best_val_iou:.4f}; saved to {best_iou_path}")
                # If IoU is the early-stopping monitor, also save a best-F1 checkpoint
                try:
                    monitor_name = str(cfg.early_stopping_monitor).strip().lower()
                except Exception:
                    monitor_name = ""
                if monitor_name == "iou":
                    if val_f1 is not None and val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_f1_path = checkpoint_dir / "best_f1_checkpoint.pt"
                        save_checkpoint(
                            best_f1_path,
                            model,
                            optimizer,
                            scaler,
                            epoch + 1,
                            global_step,
                            cfg,
                            scheduler=scheduler,
                            ema_model=ema_model,
                            use_ema_for_model_state=(ema_model is not None),
                        )
                        print(f"New best val f1 {best_val_f1:.4f}; saved to {best_f1_path}")

            # Use the configured early-stopping monitor to determine when to reset patience
            monitor_value = _metric_for_monitor(metrics, cfg.early_stopping_monitor)
            monitor_improved = monitor_value > (best_monitor_value + cfg.early_stopping_min_delta)

            if monitor_improved:
                # Update recorded bests and reset patience
                best_monitor_value = monitor_value
                best_val_f1 = val_f1
                best_val_loss = val_loss
                no_improve_epochs = 0
                best_alias_path = checkpoint_dir / "best_checkpoint.pt"
                save_checkpoint(
                    best_alias_path,
                    model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    global_step,
                    cfg,
                    scheduler=scheduler,
                    ema_model=ema_model,
                    use_ema_for_model_state=(ema_model is not None),
                )
                best_threshold_payload = _write_best_threshold_metadata(
                    best_threshold_path,
                    epoch=epoch + 1,
                    threshold=val_threshold,
                    threshold_metric=str(metrics.get("threshold_metric", cfg.threshold_metric)),
                    monitor=cfg.early_stopping_monitor,
                    monitor_value=monitor_value,
                    metrics=metrics,
                    checkpoint_path=best_alias_path,
                )
                _update_experiment_fingerprint(
                    fingerprint_path,
                    {
                        "chosen_threshold": best_threshold_payload["threshold"],
                        "best_threshold_metric": best_threshold_payload["threshold_metric"],
                        "best_checkpoint_path": best_threshold_payload["checkpoint_path"],
                        "best_monitor": best_threshold_payload["monitor"],
                        "best_monitor_value": best_threshold_payload["monitor_value"],
                        "best_val_iou": best_threshold_payload["val_iou"],
                    },
                )
                print(
                    f"New best {cfg.early_stopping_monitor} {monitor_value:.4f}; "
                    f"saved to {best_alias_path} (threshold metadata: {best_threshold_path})"
                )
            elif early_stopping_enabled:
                no_improve_epochs += 1
                print(
                    f"Early stopping patience: {no_improve_epochs}/{cfg.early_stopping_patience} "
                    f"without {cfg.early_stopping_monitor} improvement"
                )

        should_checkpoint = ((epoch + 1) % cfg.checkpoint_every == 0) or (epoch + 1 == cfg.epochs)
        if should_checkpoint:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scaler,
                epoch + 1,
                global_step,
                cfg,
                scheduler=scheduler,
                ema_model=ema_model,
                use_ema_for_model_state=False,
            )
            append_checkpoint_log(
                checkpoint_log_path,
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": float(train_loss),
                    "train_positive_ratio": float(train_positive_ratio),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_accuracy": val_accuracy,
                    "val_threshold": val_threshold,
                    "val_size_bins": val_size_bins,
                    "epoch_seconds": float(elapsed),
                    "checkpoint_path": str(ckpt_path),
                },
            )
            print(f"Saved checkpoint to {ckpt_path}")

        # Stricter patience for auto phase2: require both val_f1 and val_loss to improve for patience reset
        auto_phase2_patience = max(1, int(getattr(cfg, "auto_phase2_patience", 0)))
        auto_phase2_ready = (
            "val" in loaders
            and not auto_phase2_triggered
            and bool(getattr(cfg, "auto_phase2_enabled", False))
            and str(getattr(cfg, "training_phase", "phase1")).strip().lower() == "phase1"
            and (epoch + 1) % cfg.val_every == 0
            and no_improve_epochs >= auto_phase2_patience
        )
        if auto_phase2_ready:
            best_iou_path = checkpoint_dir / "best_iou_checkpoint.pt"
            if best_iou_path.is_file():
                auto_phase2_triggered = True
                cfg = _build_phase2_config(cfg, best_iou_path)
                load_checkpoint(
                    best_iou_path,
                    model,
                    optimizer,
                    scaler,
                    device,
                    scheduler=scheduler,
                    ema_model=ema_model,
                )
                _scale_optimizer_and_scheduler_for_phase2(
                    optimizer,
                    scheduler,
                    cfg.auto_phase2_lr_scale,
                )
                # Reset best values for phase 2
                best_val_f1 = float("-inf")
                best_val_loss = float("inf")
                no_improve_epochs = 0
                if cfg.early_stopping_monitor == "iou":
                    best_monitor_value = best_val_iou
                else:
                    best_monitor_value = float("-inf")
                early_stopping_enabled = "val" in loaders and cfg.early_stopping_patience > 0
                print(
                    "Auto phase-2 triggered | "
                    f"resume={best_iou_path} | monitor={cfg.early_stopping_monitor} | "
                    f"threshold_metric={cfg.threshold_metric} | lr_scale={cfg.auto_phase2_lr_scale:.3f} | "
                    f"tversky_weight={cfg.tversky_weight:.3f}"
                )
                continue
            print(
                "Auto phase-2 wanted to trigger, but no best_iou_checkpoint.pt was found; "
                "continuing with normal early stopping"
            )

        if early_stopping_enabled and "val" in loaders and (epoch + 1) % cfg.val_every == 0:
            if no_improve_epochs >= cfg.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Training complete")


if __name__ == "__main__":
    configuration = parse_args()
    run_training(configuration)

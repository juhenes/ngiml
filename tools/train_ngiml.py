"""End-to-end NGIML training loop with checkpointing.

Run example (Colab-ready):
    python tools/train_ngiml.py --manifest /content/data/manifest.json --output-dir /content/runs

The script expects a prepared manifest (see src/data/config.py) and will
save checkpoints plus a copy of the training arguments inside the output dir.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time
import os
import hashlib
import tarfile
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

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloaders import AugmentationConfig, create_dataloaders, load_manifest
from src.model.hybrid_ngiml import HybridNGIML, HybridNGIMLConfig
from src.model.losses import MultiStageLossConfig, MultiStageManipulationLoss


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
    scheduler_type: str = "cosine"  # one of: 'cosine', 'step' (cosine enabled by default)
    manifest: str
    output_dir: str = "runs/ngiml"
    batch_size: int = 8
    epochs: int = 50
    num_workers: int = max(2, (os.cpu_count() or 4) // 2)
    amp: bool = True
    pin_memory: bool = True
    channels_last: bool = True
    compile_model: bool = True
    compile_mode: str = "default"
    deterministic: bool = False
    use_tf32: bool = True
    cuda_expandable_segments: bool = True
    lr_schedule: bool = True
    warmup_epochs: int = 5  # Linear warmup for first 5 epochs
    min_lr_scale: float = 0.1  # Start at 10% base LR
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    val_every: int = 1
    checkpoint_every: int = 1
    resume: Optional[str] = None
    auto_resume: bool = False
    round_robin_seed: Optional[int] = 42
    balance_sampling: bool = False
    balance_real_fake: bool = True
    balanced_positive_ratio: float = 0.5
    balanced_sampler_seed: int = 42
    balanced_sampler_num_samples: Optional[int] = None
    prefetch_factor: Optional[int] = 2
    persistent_workers: bool = True
    drop_last: bool = True
    auto_local_cache: bool = True
    local_cache_dir: Optional[str] = None
    reuse_local_cache_manifest: bool = True
    views_per_sample: int = 1
    max_rotation_degrees: float = 5.0
    noise_std_max: float = 0.02
    disable_aug: bool = False
    device: Optional[str] = None
    aug_seed: Optional[int] = None
    seed: int = 42
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = "iou"
    metric_threshold: float = 0.5
    optimize_threshold: bool = True
    threshold_metric: str = "iou"
    threshold_start: float = 0.1
    threshold_end: float = 0.9
    threshold_step: float = 0.1
    small_mask_ratio_max: float = 0.01
    medium_mask_ratio_max: float = 0.05
    compute_foreground_ratio: bool = True
    auto_pos_weight: bool = True
    pos_weight_min: float = 1.0
    pos_weight_max: float = 20.0
    loss_hybrid_mode: str = "dice_bce"
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_weight: float = 0.2
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    ema_enabled: bool = True
    ema_decay: float = 0.999
    hard_mining_enabled: bool = True
    hard_mining_start_epoch: int = 3
    hard_mining_weight: float = 0.2
    hard_mining_gamma: float = 2.0
    default_aug: Optional[AugmentationConfig] = None
    per_dataset_aug: Optional[Dict[str, AugmentationConfig]] = None
    model_config: Optional[HybridNGIMLConfig] = None
    loss_config: Optional[MultiStageLossConfig] = None


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


def parse_args() -> TrainConfig:
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "step"], help="LR scheduler type (cosine or step)")
    default_workers = max(2, (os.cpu_count() or 4) // 2)
    parser = argparse.ArgumentParser(description="Train NGIML manipulation localization")
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
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument("--views-per-sample", type=int, default=None, help="Number of augmented views per sample (on-the-fly)")
    parser.add_argument("--max-rotation-degrees", type=float, default=5.0, help="Random rotation range (+/-)")
    parser.add_argument("--noise-std-max", type=float, default=0.02, help="Max Gaussian noise std")
    parser.add_argument("--disable-aug", action="store_true", help="Disable GPU augmentations")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda:0 or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Stop after N validations without improvement; <=0 disables")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="Minimum monitored-metric improvement to reset early stopping")
    parser.add_argument("--early-stopping-monitor", type=str, default="iou", choices=["iou", "dice", "recall", "precision", "accuracy", "loss"], help="Validation metric used for early stopping and best checkpoint")
    parser.add_argument("--metric-threshold", type=float, default=0.5, help="Fixed threshold for sigmoid outputs when threshold optimization is disabled")
    parser.add_argument("--optimize-threshold", action=argparse.BooleanOptionalAction, default=True, help="Search validation thresholds and use the best for metric reporting")
    parser.add_argument("--threshold-metric", type=str, default="iou", choices=["iou", "dice"], help="Metric used to select best threshold")
    parser.add_argument("--threshold-start", type=float, default=0.1, help="Threshold search range start")
    parser.add_argument("--threshold-end", type=float, default=0.9, help="Threshold search range end")
    parser.add_argument("--threshold-step", type=float, default=0.1, help="Threshold search step size")
    parser.add_argument("--small-mask-ratio-max", type=float, default=0.01, help="Upper foreground-ratio bound for small-mask validation bin")
    parser.add_argument("--medium-mask-ratio-max", type=float, default=0.05, help="Upper foreground-ratio bound for medium-mask validation bin")
    parser.add_argument("--compute-foreground-ratio", action=argparse.BooleanOptionalAction, default=True, help="Compute foreground pixel ratio from train loader")
    parser.add_argument("--auto-pos-weight", action=argparse.BooleanOptionalAction, default=True, help="Auto-compute BCE pos_weight from foreground ratio")
    parser.add_argument("--pos-weight-min", type=float, default=1.0, help="Lower clamp for auto pos_weight")
    parser.add_argument("--pos-weight-max", type=float, default=20.0, help="Upper clamp for auto pos_weight")
    parser.add_argument("--loss-hybrid-mode", type=str, default="dice_bce", choices=["dice_bce", "dice_focal"], help="Hybrid loss type")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight")
    parser.add_argument("--bce-weight", type=float, default=1.0, help="BCE/Focal term weight in hybrid loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (used when loss-hybrid-mode=dice_focal)")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha (used when loss-hybrid-mode=dice_focal)")
    parser.add_argument("--tversky-weight", type=float, default=0.2, help="Optional Tversky loss weight to improve recall")
    parser.add_argument("--tversky-alpha", type=float, default=0.3, help="Tversky alpha (FP penalty)")
    parser.add_argument("--tversky-beta", type=float, default=0.7, help="Tversky beta (FN penalty)")
    parser.add_argument("--lovasz-weight", type=float, default=0.5, help="Lovasz Hinge Loss weight for IoU optimization")
    parser.add_argument("--ema-enabled", action=argparse.BooleanOptionalAction, default=True, help="Use EMA weights for validation and best checkpoints")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay factor")
    parser.add_argument("--hard-mining-enabled", action=argparse.BooleanOptionalAction, default=True, help="Enable low-IoU hard-example weighting")
    parser.add_argument("--hard-mining-start-epoch", type=int, default=2, help="Epoch to start hard-example weighting")
    parser.add_argument("--hard-mining-weight", type=float, default=0.2, help="Weight of hard-example auxiliary loss")
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
        max_rotation_degrees=args.max_rotation_degrees,
        noise_std_max=args.noise_std_max,
        disable_aug=args.disable_aug,
        device=args.device,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        metric_threshold=args.metric_threshold,
        optimize_threshold=args.optimize_threshold,
        threshold_metric=args.threshold_metric,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        small_mask_ratio_max=args.small_mask_ratio_max,
        medium_mask_ratio_max=args.medium_mask_ratio_max,
        compute_foreground_ratio=args.compute_foreground_ratio,
        auto_pos_weight=args.auto_pos_weight,
        pos_weight_min=args.pos_weight_min,
        pos_weight_max=args.pos_weight_max,
        loss_hybrid_mode=args.loss_hybrid_mode,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        lovasz_weight=args.lovasz_weight,
        ema_enabled=args.ema_enabled,
        ema_decay=args.ema_decay,
        hard_mining_enabled=args.hard_mining_enabled,
        hard_mining_start_epoch=args.hard_mining_start_epoch,
        hard_mining_weight=args.hard_mining_weight,
        hard_mining_gamma=args.hard_mining_gamma,
        scheduler_type=args.scheduler_type,
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
        object_crop_bias_prob=0.7,
        min_fg_pixels_for_object_crop=16,
        enable_elastic=True,
        elastic_prob=0.3,
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
    return create_dataloaders(
        manifest_path,
        per_dataset_aug,
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
    )


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

        out_path.write_bytes(member.read())
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
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
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
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_checkpoint(
    path: Path,
    model: HybridNGIML,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema_model: Optional[HybridNGIML] = None,
) -> Tuple[int, int]:
    data = torch.load(path, map_location=device)
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
    start_epoch = int(data.get("epoch", 0))
    global_step = int(data.get("global_step", 0))
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
def compute_foreground_pixel_ratio(loader) -> float:
    foreground = 0.0
    total = 0.0
    for batch in loader:
        masks = batch["masks"]
        masks = (masks > 0.5).float()
        foreground += float(masks.sum().item())
        total += float(masks.numel())
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


def _sample_has_mask_high_pass(record) -> tuple[bool, bool]:
    has_mask = bool(record.mask_path)
    has_high_pass = bool(record.high_pass_path)
    image_path = str(record.image_path)
    if not image_path.endswith(".npz"):
        return has_mask, has_high_pass

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
        else:
            with np.load(image_path, allow_pickle=False) as npz_data:
                has_mask = has_mask or ("mask" in npz_data and npz_data["mask"].size > 0)
                has_high_pass = has_high_pass or ("high_pass" in npz_data and npz_data["high_pass"].size > 0)
    except Exception as exc:
        raise ValueError(f"Failed to inspect NPZ sample for mask/high_pass fields: {image_path}") from exc

    return has_mask, has_high_pass


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

    for sample in train_samples:
        per_dataset_counts[sample.dataset] = per_dataset_counts.get(sample.dataset, 0) + 1
        label = int(sample.label)
        if label == 0:
            real_count += 1
        elif label == 1:
            fake_count += 1
        else:
            raise ValueError(f"Unexpected train label {label} for sample: {sample.image_path}")

        has_mask, has_high_pass = _sample_has_mask_high_pass(sample)
        if has_mask:
            mask_count += 1
        if has_high_pass:
            high_pass_count += 1

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
        f"high_pass={100.0 * (high_pass_count / max(total, 1)):.1f}%"
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
):
    model.train()
    running_loss = 0.0
    num_batches = 0
    sampled_pos = 0.0
    sampled_total = 0.0
    accum_steps = max(1, int(cfg.grad_accum_steps))
    progress = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(progress):
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        high_pass = batch.get("high_pass")
        if isinstance(high_pass, torch.Tensor):
            high_pass = high_pass.to(device, non_blocking=True)
        else:
            high_pass = None
        labels = batch["labels"]
        pos_count, total_count = _to_float_label_ratio(labels)
        sampled_pos += pos_count
        sampled_total += total_count
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            if high_pass is not None:
                high_pass = high_pass.contiguous(memory_format=torch.channels_last)

        use_amp = cfg.amp and device.type == "cuda"
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
            loss = loss_fn(preds, masks)

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
        scaler.scale(scaled_loss).backward()

        do_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))

        if do_step:
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            _update_ema_model(ema_model, model, cfg.ema_decay)

        running_loss += loss.item()
        num_batches += 1
        global_step += 1

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
        use_amp = cfg.amp and device.type == "cuda"
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
        logits = preds[-1]

        for threshold in thresholds:
            counts = _segmentation_counts(logits, masks, threshold=threshold)
            threshold_stats[float(threshold)]["tp"] += counts["tp"]
            threshold_stats[float(threshold)]["tn"] += counts["tn"]
            threshold_stats[float(threshold)]["fp"] += counts["fp"]
            threshold_stats[float(threshold)]["fn"] += counts["fn"]

    optimize_key = cfg.threshold_metric.lower()
    if optimize_key not in {"iou", "dice"}:
        optimize_key = "iou"

    scored_thresholds: list[tuple[float, dict]] = []
    for threshold in thresholds:
        stats = threshold_stats[float(threshold)]
        metrics = _metrics_from_counts(stats["tp"], stats["tn"], stats["fp"], stats["fn"])
        scored_thresholds.append((float(threshold), metrics))

    best_threshold, best_metrics = max(scored_thresholds, key=lambda item: item[1][optimize_key])
    return {
        "threshold": float(best_threshold),
        "threshold_metric": optimize_key,
        "dice": float(best_metrics["dice"]),
        "iou": float(best_metrics["iou"]),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "accuracy": float(best_metrics["accuracy"]),
    }


@torch.inference_mode()
def evaluate(model: HybridNGIML, loader, loss_fn, device: torch.device, cfg: TrainConfig) -> dict:
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
        high_pass = batch.get("high_pass")
        if isinstance(high_pass, torch.Tensor):
            high_pass = high_pass.to(device, non_blocking=True)
        else:
            high_pass = None
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            if high_pass is not None:
                high_pass = high_pass.contiguous(memory_format=torch.channels_last)
        use_amp = cfg.amp and device.type == "cuda"
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, target_size=masks.shape[-2:], high_pass=high_pass)
            loss = loss_fn(preds, masks)
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
    if optimize_key not in {"iou", "dice"}:
        optimize_key = "iou"

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
        best_threshold, best_metrics = max(scored_thresholds, key=lambda item: item[1][optimize_key])
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

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

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

    _print_and_validate_train_dataset_integrity(Path(cfg.manifest))

    loaders = _prepare_dataloaders(cfg, device)
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
        foreground_ratio = compute_foreground_pixel_ratio(loaders["train"])
        print(f"Foreground pixel ratio (train): {foreground_ratio:.6f}")

    model_cfg = cfg.model_config or HybridNGIMLConfig()
    model = HybridNGIML(model_cfg).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = model.build_optimizer()
    scheduler = _build_lr_scheduler(optimizer, cfg)
    scaler = GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))
    ema_model = _init_ema_model(model, model_cfg, cfg.ema_enabled)
    if ema_model is not None:
        ema_model = ema_model.to(device)
        if cfg.channels_last and device.type == "cuda":
            ema_model = ema_model.to(memory_format=torch.channels_last)
    loss_cfg = cfg.loss_config or MultiStageLossConfig(
        hybrid_mode=cfg.loss_hybrid_mode,
        dice_weight=cfg.dice_weight,
        bce_weight=cfg.bce_weight,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
        tversky_weight=cfg.tversky_weight,
        tversky_alpha=cfg.tversky_alpha,
        tversky_beta=cfg.tversky_beta,
        lovasz_weight=cfg.lovasz_weight,
    )
    if cfg.auto_pos_weight and foreground_ratio is not None:
        ratio = max(1e-6, min(1.0 - 1e-6, foreground_ratio))
        pos_weight = (1.0 - ratio) / ratio
        pos_weight = float(min(max(pos_weight, cfg.pos_weight_min), cfg.pos_weight_max))
        loss_cfg = replace(loss_cfg, pos_weight=pos_weight)
        print(f"Auto pos_weight from foreground ratio: {pos_weight:.4f}")
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
    checkpoint_log_path = checkpoint_dir / "checkpoint_metrics.jsonl"

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
    no_improve_epochs = 0
    early_stopping_enabled = "val" in loaders and cfg.early_stopping_patience > 0
    best_threshold_path = checkpoint_dir / "best_threshold.json"

    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()
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
        )

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
        val_precision = None
        val_recall = None
        val_accuracy = None
        val_threshold = None
        val_size_bins = None
        if "val" in loaders and (epoch + 1) % cfg.val_every == 0:
            eval_model = ema_model if ema_model is not None else model
            metrics = evaluate(eval_model, loaders["val"], loss_fn, device, cfg)
            val_loss = float(metrics["loss"])
            val_dice = float(metrics["dice"])
            val_iou = float(metrics["iou"])
            val_precision = float(metrics["precision"])
            val_recall = float(metrics["recall"])
            val_accuracy = float(metrics["accuracy"])
            val_threshold = float(metrics["threshold"])
            val_size_bins = metrics.get("size_bins")
            print(
                f"Val | loss {val_loss:.4f} | dice {val_dice:.4f} | iou {val_iou:.4f} "
                f"| precision {val_precision:.4f} | recall {val_recall:.4f} | accuracy {val_accuracy:.4f} "
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

            monitor_value = _metric_for_monitor(metrics, cfg.early_stopping_monitor)
            improved = monitor_value > (best_monitor_value + cfg.early_stopping_min_delta)
            if improved:
                best_monitor_value = monitor_value
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
                    f"New best val {cfg.early_stopping_monitor} {monitor_value:.4f}; "
                    f"saved to {best_alias_path} (threshold metadata: {best_threshold_path})"
                )
            elif early_stopping_enabled:
                no_improve_epochs += 1
                print(
                    f"Early stopping patience: {no_improve_epochs}/{cfg.early_stopping_patience} "
                    f"without val {cfg.early_stopping_monitor} improvement"
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

        if early_stopping_enabled and "val" in loaders and (epoch + 1) % cfg.val_every == 0:
            if no_improve_epochs >= cfg.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Training complete")


if __name__ == "__main__":
    configuration = parse_args()
    run_training(configuration)

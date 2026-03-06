from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from src.data.dataloaders import AugmentationConfig, load_manifest

if TYPE_CHECKING:
    from src.model.hybrid_ngiml import HybridNGIMLConfig
    from src.model.losses import MultiStageLossConfig


def _norm(value: str) -> str:
    return str(value).replace("\\", "/")


def _recommended_cuda_precision(default: str = "fp16") -> str:
    try:
        import torch

        if torch.cuda.is_available():
            checker = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(checker):
                try:
                    if bool(checker()):
                        return "bf16"
                except Exception:
                    pass
            try:
                major, _minor = torch.cuda.get_device_capability()
                if int(major) >= 8:
                    return "bf16"
            except Exception:
                pass
    except Exception:
        pass
    return str(default)


def _suffix_score(a_parts, b_parts) -> int:
    score = 0
    for ax, bx in zip(reversed(a_parts), reversed(b_parts)):
        if ax != bx:
            break
        score += 1
    return score


def _candidate_paths(value: str, manifest_path: Path, data_root: Path):
    normalized = _norm(value)
    path_value = Path(normalized)

    candidates = []
    if path_value.is_absolute():
        candidates.append(path_value)
    else:
        candidates.extend(
            [
                manifest_path.parent / path_value,
                data_root / path_value,
                data_root / "ngiml" / path_value,
                Path("/content") / path_value,
                Path("/content/data") / path_value,
                Path("/content/ngiml") / path_value,
            ]
        )

    if "prepared/" in normalized:
        suffix = normalized.split("prepared/", 1)[1]
        candidates.extend(
            [
                data_root / "prepared" / suffix,
                data_root / "ngiml" / "prepared" / suffix,
                Path("/content") / "prepared" / suffix,
                Path("/content/ngiml") / "prepared" / suffix,
            ]
        )

    if "datasets/" in normalized:
        suffix = normalized.split("datasets/", 1)[1]
        candidates.extend(
            [
                data_root / "datasets" / suffix,
                data_root / "ngiml" / "datasets" / suffix,
                Path("/content") / "datasets" / suffix,
                Path("/content/ngiml") / "datasets" / suffix,
            ]
        )

    seen = set()
    unique = []
    for candidate in candidates:
        key = candidate.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _build_tar_index(data_root: Path):
    tar_files = []
    for pattern in ("*.tar", "*.tar.gz", "*.tgz"):
        tar_files.extend(data_root.rglob(pattern))
    tar_by_name = {}
    for tar_path in tar_files:
        tar_by_name.setdefault(tar_path.name, []).append(tar_path)
    return tar_files, tar_by_name


def _match_tar_by_basename(value: str, tar_by_name: dict[str, list[Path]]):
    name = Path(_norm(value)).name
    matches = tar_by_name.get(name, [])
    if not matches:
        return None
    hint_parts = Path(_norm(value)).parts
    return max(matches, key=lambda path: _suffix_score(path.parts, hint_parts))


def _resolve_file(value: str, manifest_path: Path, data_root: Path, tar_by_name: dict[str, list[Path]]) -> Path:
    candidates = _candidate_paths(value, manifest_path, data_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if str(value).endswith((".tar", ".tar.gz", ".tgz")):
        tar_match = _match_tar_by_basename(value, tar_by_name)
        if tar_match is not None:
            return tar_match

    return candidates[0] if candidates else Path(_norm(value))


def _resolve_path(path_str: str | None, manifest_path: Path, data_root: Path, tar_by_name: dict[str, list[Path]]) -> str | None:
    if path_str is None:
        return None
    normalized = _norm(path_str)
    if "::" in normalized:
        archive, member = normalized.split("::", 1)
        archive_path = _resolve_file(archive, manifest_path, data_root, tar_by_name).as_posix()
        member_path = _norm(member)
        return f"{archive_path}::{member_path}"
    return _resolve_file(normalized, manifest_path, data_root, tar_by_name).as_posix()


def _sample_files_exist(sample) -> bool:
    image_path = str(sample.image_path)
    if "::" in image_path:
        archive_path, _ = image_path.split("::", 1)
        if not Path(archive_path).exists():
            return False
    else:
        if not Path(image_path).exists():
            return False

    if sample.mask_path is not None and not Path(sample.mask_path).exists():
        return False
    if sample.high_pass_path is not None and not Path(sample.high_pass_path).exists():
        return False
    return True


def find_or_resolve_manifest(data_root: Path, manifest_names: Tuple[str, ...] = ("manifest.parquet", "manifest.json")) -> Path:
    data_root = Path(data_root)
    resolved_manifest_path = data_root / "manifest_resolved.json"

    manifest_candidates = [
        resolved_manifest_path,
        data_root / "manifest.parquet",
        data_root / "manifest.json",
        data_root / "prepared" / "manifest.parquet",
        data_root / "prepared" / "manifest.json",
        data_root / "ngiml" / "manifest.parquet",
        data_root / "ngiml" / "manifest.json",
    ]

    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    if manifest_path is None:
        discovered = sorted(
            p
            for p in data_root.rglob("manifest.*")
            if p.name in manifest_names or p.name == "manifest_resolved.json"
        )
        if discovered:
            manifest_path = discovered[0]
        else:
            raise FileNotFoundError(
                f"No manifest.parquet or manifest.json found under {data_root}. "
                "Check dataset download path, or set DATA_DIR to the folder containing the manifest file."
            )

    if resolved_manifest_path.exists() and resolved_manifest_path.stat().st_size > 0:
        print(f"Using cached resolved manifest: {resolved_manifest_path}")
        return resolved_manifest_path

    print("Using manifest:", manifest_path)
    tar_files, tar_by_name = _build_tar_index(data_root)
    print(f"Indexed tar files under {data_root}: {len(tar_files)}")

    manifest_obj = load_manifest(manifest_path)
    rewritten = 0
    for sample in manifest_obj.samples:
        image_new = _resolve_path(sample.image_path, manifest_path, data_root, tar_by_name)
        mask_new = _resolve_path(sample.mask_path, manifest_path, data_root, tar_by_name) if sample.mask_path else None
        hp_new = _resolve_path(sample.high_pass_path, manifest_path, data_root, tar_by_name) if sample.high_pass_path else None

        if image_new != sample.image_path:
            sample.image_path = image_new
            rewritten += 1
        if mask_new != sample.mask_path:
            sample.mask_path = mask_new
            rewritten += 1
        if hp_new != sample.high_pass_path:
            sample.high_pass_path = hp_new
            rewritten += 1

    original_count = len(manifest_obj.samples)
    manifest_obj.samples = [s for s in manifest_obj.samples if _sample_files_exist(s)]
    filtered_out = original_count - len(manifest_obj.samples)

    if not manifest_obj.samples:
        raise FileNotFoundError(
            "No valid samples remain after path resolution. "
            f"Indexed tar files: {len(tar_files)} under {data_root}. "
            "Likely the downloaded dataset does not contain prepared shards referenced by the manifest."
        )

    with open(resolved_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_obj.to_dict(), handle)

    print(
        f"Wrote resolved manifest to {resolved_manifest_path} "
        f"(updated fields: {rewritten}, removed missing samples: {filtered_out})"
    )
    return resolved_manifest_path


def build_default_components():
    # Import model config classes lazily to avoid importing heavy ML libraries at module import time
    from src.model.backbones.efficientnet_backbone import EfficientNetBackboneConfig
    from src.model.backbones.residual_noise_branch import ResidualNoiseConfig
    from src.model.backbones.swin_backbone import SwinBackboneConfig
    from src.model.feature_fusion import FeatureFusionConfig
    from src.model.hybrid_ngiml import HybridNGIMLConfig, HybridNGIMLOptimizerConfig, OptimizerGroupConfig
    from src.model.losses import MultiStageLossConfig
    from src.model.unet_decoder import UNetDecoderConfig

    model_cfg = HybridNGIMLConfig(
        efficientnet=EfficientNetBackboneConfig(pretrained=True),
        swin=SwinBackboneConfig(model_name="swin_tiny_patch4_window7_224", pretrained=True, input_size=320),
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
        use_boundary_loss=False,
        boundary_weight=0.05,
    )

    default_aug = AugmentationConfig(
        enable=True,
        views_per_sample=2,
        enable_flips=True,
        enable_rotations=True,
        max_rotation_degrees=5.0,
        enable_random_crop=True,
        crop_scale_range=(0.75, 1.0),
        object_crop_bias_prob=0.85,
        min_fg_pixels_for_object_crop=8,
        multiscale_training=False,
        multiscale_short_side_range=(384, 640),
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

    per_dataset_aug = {
        "IMD2020": AugmentationConfig(
            enable=True,
            views_per_sample=3,
            enable_flips=True,
            enable_rotations=True,
            max_rotation_degrees=8.0,
            enable_random_crop=True,
            crop_scale_range=(0.75, 1.0),
            object_crop_bias_prob=0.9,
            min_fg_pixels_for_object_crop=4,
            multiscale_training=False,
            multiscale_short_side_range=(384, 640),
            enable_elastic=False,
            elastic_prob=0.0,
            elastic_alpha=10.0,
            elastic_sigma=5.0,
            enable_color_jitter=True,
            brightness_jitter_factors=(0.85, 1.15),
            contrast_jitter_factors=(0.85, 1.15),
            enable_noise=True,
            noise_std_range=(0.0, 0.02),
        )
    }

    return model_cfg, loss_cfg, default_aug, per_dataset_aug


def build_training_config(
    manifest_path: Path,
    output_dir: str,
    model_cfg: HybridNGIMLConfig,
    loss_cfg: MultiStageLossConfig,
    default_aug: AugmentationConfig,
    per_dataset_aug: dict[str, AugmentationConfig],
) -> dict:
    effective_hybrid_mode = str(getattr(loss_cfg, "hybrid_mode", "dice_bce"))
    effective_dice_weight = float(getattr(loss_cfg, "dice_weight", 1.0))
    effective_bce_weight = float(getattr(loss_cfg, "bce_weight", 1.0))
    effective_focal_gamma = float(getattr(loss_cfg, "focal_gamma", 2.0))
    effective_focal_alpha = float(getattr(loss_cfg, "focal_alpha", 0.25))
    effective_tversky_weight = float(getattr(loss_cfg, "tversky_weight", 0.0))
    effective_tversky_alpha = float(getattr(loss_cfg, "tversky_alpha", 0.3))
    effective_tversky_beta = float(getattr(loss_cfg, "tversky_beta", 0.8))
    effective_lovasz_weight = float(getattr(loss_cfg, "lovasz_weight", 0.0))
    effective_use_boundary_loss = bool(getattr(loss_cfg, "use_boundary_loss", False))
    effective_boundary_weight = float(getattr(loss_cfg, "boundary_weight", 0.05))

    return {
        "manifest": str(manifest_path),
        "output_dir": output_dir,
        "batch_size": 20,
        "grad_accum_steps": 1,
        "epochs": 50,
        "num_workers": 0,
        "amp": True,
        "grad_clip": 1.0,
        "val_every": 1,
        "checkpoint_every": 1,
        "resume": None,
        "auto_resume": True,
        "round_robin_seed": 42,
        "balance_real_fake": True,
        "balanced_positive_ratio": 0.6,
        "prefetch_factor": 2,
        "persistent_workers": False,
        "drop_last": True,
        "views_per_sample": 2,
        "max_rotation_degrees": 5.0,
        "noise_std_max": 0.012,
        "disable_aug": False,
        "max_short_side": 480,
        "device": "cuda",
        "aug_seed": 42,
        "seed": 42,
        "warmup_epochs": 3,
        "early_stopping_patience": 12,
        "early_stopping_min_delta": 1e-4,
        "metric_threshold": 0.5,
        "optimize_threshold": True,
        "threshold_metric": "f1",
        "threshold_start": 0.2,
        "threshold_end": 0.8,
        "threshold_step": 0.02,
        "compute_foreground_ratio": True,
        "foreground_ratio_max_batches": 20,
        "short_side_probe_samples": 0,
        "auto_pos_weight": True,
        "pos_weight_min": 0.5,
        "pos_weight_max": 20.0,
        "balanced_pos_weight_cap": 3.0,
        "loss_hybrid_mode": effective_hybrid_mode,
        "dice_weight": effective_dice_weight,
        "bce_weight": effective_bce_weight,
        "focal_gamma": effective_focal_gamma,
        "focal_alpha": effective_focal_alpha,
        "tversky_weight": effective_tversky_weight,
        "tversky_alpha": effective_tversky_alpha,
        "tversky_beta": effective_tversky_beta,
        "lovasz_weight": effective_lovasz_weight,
        "use_boundary_loss": effective_use_boundary_loss,
        "boundary_weight": effective_boundary_weight,
        "ema_enabled": True,
        "ema_decay": 0.999,
        "hard_mining_enabled": False,
        "hard_mining_start_epoch": 5,
        "hard_mining_weight": 0.03,
        "hard_mining_gamma": 2.0,
        "default_aug": default_aug,
        "per_dataset_aug": per_dataset_aug,
        "model_config": model_cfg,
        "loss_config": loss_cfg,
    }


def apply_phase2_resume_preset(
    training_config: dict,
    resume_checkpoint: str,
    lr_scale: float = 0.33,
    tversky_weight: float = 0.1,
    monitor_metric: str = "iou",
) -> dict:
    """Apply phase-2 fine-tuning settings after a phase-1 plateau.

    Intended usage in notebooks:
      1) Build base config.
      2) Apply runtime/throughput settings.
      3) Call this function to switch into resume + lower-LR overlap-focused tuning.
    """

    if not resume_checkpoint:
        raise ValueError("resume_checkpoint must be a non-empty checkpoint path")

    if lr_scale <= 0.0:
        raise ValueError("lr_scale must be > 0")

    metric = str(monitor_metric).strip().lower()
    if metric not in {"iou", "f1", "dice"}:
        raise ValueError("monitor_metric must be one of: iou, f1, dice")

    training_config.update(
        {
            "resume": str(resume_checkpoint),
            "auto_resume": False,
            "warmup_epochs": 0,
            "early_stopping_monitor": metric,
            "threshold_metric": metric,
            "tversky_weight": float(tversky_weight),
            "lovasz_weight": 0.0,
            "hard_mining_enabled": False,
        }
    )

    model_cfg = training_config.get("model_config")
    optimizer_cfg = getattr(model_cfg, "optimizer", None) if model_cfg is not None else None
    if optimizer_cfg is not None:
        for group_name in ("efficientnet", "swin", "residual", "fusion", "decoder"):
            group = getattr(optimizer_cfg, group_name, None)
            if group is None:
                continue
            group.lr = float(group.lr) * float(lr_scale)

    return training_config


def apply_colab_runtime_settings(
    training_config: dict,
    balance_sampling: bool = True,
    local_cache_dir: str | None = None,
    tune_for_large_batch: bool = False,
) -> dict:
    def _apply_effective_batch_optimizer_scaling(config: dict, base_effective_batch: int = 12) -> None:
        model_cfg = config.get("model_config")
        if model_cfg is None or not hasattr(model_cfg, "optimizer"):
            return

        optimizer_cfg = model_cfg.optimizer
        batch_size = int(config.get("batch_size", 12))
        grad_accum_steps = int(config.get("grad_accum_steps", 1))
        effective_batch = max(1, batch_size * grad_accum_steps)

        ratio = float(effective_batch) / float(max(1, base_effective_batch))
        lr_scale = float(min(max(math.sqrt(ratio), 0.75), 1.8))
        wd_scale = float(min(max(pow(ratio, 0.25), 0.9), 1.35))

        for group_name in ("efficientnet", "swin", "residual", "fusion", "decoder"):
            group = getattr(optimizer_cfg, group_name, None)
            if group is None:
                continue
            group.lr = float(group.lr) * lr_scale
            group.weight_decay = float(group.weight_decay) * wd_scale

    recommended_workers = max(2, min(6, (os.cpu_count() or 4)))
    cache_dir = local_cache_dir or "/content/cache"
    if tune_for_large_batch:
        runtime_precision = _recommended_cuda_precision(default="fp16")
        training_config.update(
            {
                "batch_size": int(max(20, int(training_config.get("batch_size", 20)))),
                "num_workers": recommended_workers,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "pin_memory": True,
                "auto_local_cache": True,
                "local_cache_dir": cache_dir,
                "reuse_local_cache_manifest": True,
                "compile_model": True,
                "compile_mode": "default",
                "channels_last": True,
                "use_tf32": True,
                "precision": runtime_precision,
                "max_short_side": int(max(480, int(training_config.get("max_short_side", 480)))),
                "foreground_ratio_max_batches": int(training_config.get("foreground_ratio_max_batches", 20)),
                "short_side_probe_samples": int(training_config.get("short_side_probe_samples", 0)),
                "balance_sampling": bool(balance_sampling),
            }
        )
        _apply_effective_batch_optimizer_scaling(training_config, base_effective_batch=12)
    else:
        training_config.update(
            {
                "num_workers": recommended_workers,
                "persistent_workers": False,
                "pin_memory": True,
                "auto_local_cache": True,
                "local_cache_dir": cache_dir,
                "reuse_local_cache_manifest": True,
                "compile_model": True,
                "compile_mode": "default",
                "channels_last": True,
                "use_tf32": True,
                "balance_sampling": bool(balance_sampling),
            }
        )
        _apply_effective_batch_optimizer_scaling(training_config, base_effective_batch=12)

    return training_config


def apply_high_throughput_settings(training_config: dict, target_batch_size: int = 32) -> dict:
    """Adjust a training_config dict for high-throughput GPU runs.

    This function updates worker counts, prefetching, memory format and
    precision defaults to better saturate modern accelerators.
    """
    recommended_workers = max(4, min(16, (os.cpu_count() or 4)))
    training_config.update(
        {
            "batch_size": int(target_batch_size),
            "num_workers": recommended_workers,
            "pin_memory": True,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "compile_model": True,
            "compile_mode": "default",
            "channels_last": True,
            "use_tf32": True,
            "precision": "bf16",
        }
    )

    model_cfg = training_config.get("model_config")
    if model_cfg is not None and hasattr(model_cfg, "optimizer"):
        optimizer_cfg = model_cfg.optimizer
        batch_size = int(training_config.get("batch_size", 12))
        grad_accum_steps = int(training_config.get("grad_accum_steps", 1))
        effective_batch = max(1, batch_size * grad_accum_steps)
        ratio = float(effective_batch) / float(12)
        lr_scale = float(min(max(math.sqrt(ratio), 0.75), 1.8))
        wd_scale = float(min(max(pow(ratio, 0.25), 0.9), 1.35))
        for group_name in ("efficientnet", "swin", "residual", "fusion", "decoder"):
            group = getattr(optimizer_cfg, group_name, None)
            if group is None:
                continue
            group.lr = float(group.lr) * lr_scale
            group.weight_decay = float(group.weight_decay) * wd_scale

    return training_config


def stage_persistent_cache_to_runtime(
    persistent_cache_dir: str | Path,
    runtime_cache_dir: str | Path = "/content/cache",
    force: bool = False,
) -> dict[str, object]:
    persistent = Path(persistent_cache_dir)
    runtime = Path(runtime_cache_dir)
    runtime.mkdir(parents=True, exist_ok=True)

    if not persistent.exists():
        return {
            "staged": False,
            "reason": f"Persistent cache not found: {persistent}",
            "persistent_cache_dir": str(persistent),
            "runtime_cache_dir": str(runtime),
        }

    runtime_has_content = any(runtime.iterdir())
    if runtime_has_content and not force:
        return {
            "staged": False,
            "reason": "Runtime cache already populated; skipping copy",
            "persistent_cache_dir": str(persistent),
            "runtime_cache_dir": str(runtime),
        }

    copied_entries = 0
    for src in persistent.iterdir():
        dst = runtime / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied_entries += 1
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_entries += 1

    return {
        "staged": True,
        "copied_entries": copied_entries,
        "persistent_cache_dir": str(persistent),
        "runtime_cache_dir": str(runtime),
    }

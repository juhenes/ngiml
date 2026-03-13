import os
import shutil
import math
from pathlib import Path
from typing import Sequence, Dict, Optional
from tools.manifest_utils import find_or_resolve_manifest
from src.model.hybrid_ngiml import HybridNGIMLConfig
from src.model.losses import MultiStageLossConfig
from src.data.config import AugmentationConfig

def _recommended_cuda_precision(default="bf16"):
    """Return recommended CUDA precision for Colab runtime. Extend as needed."""
    return default

def apply_colab_runtime_settings(
    training_config: dict,
    balance_sampling: bool = False,
    local_cache_dir: str = None,
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
        lr_scale = float(min(max((ratio) ** 0.5, 0.75), 1.8))
        wd_scale = float(min(max((ratio) ** 0.25, 0.9), 1.35))
        for group_name in ("efficientnet", "swin", "residual", "fusion", "decoder"):
            group = getattr(optimizer_cfg, group_name, None)
            if group is None:
                continue
            group.lr = float(group.lr) * lr_scale
            group.weight_decay = float(group.weight_decay) * wd_scale
    recommended_workers = max(2, min(6, (os.cpu_count() or 4)))
    cache_dir = local_cache_dir or "/content/cache"
    if tune_for_large_batch:
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


def build_default_components():
    return _build_default_components_top_level()


def build_training_config(
    manifest_path: Path,
    output_dir: str,
    model_cfg: HybridNGIMLConfig,
    loss_cfg: MultiStageLossConfig,
    default_aug: AugmentationConfig,
    per_dataset_aug: dict[str, AugmentationConfig],
) -> dict:
    return _build_training_config_top_level(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        default_aug=default_aug,
        per_dataset_aug=per_dataset_aug,
    )


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
            "training_phase": "phase2",
            "auto_phase2_enabled": False,
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
    balance_sampling: bool = False,
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

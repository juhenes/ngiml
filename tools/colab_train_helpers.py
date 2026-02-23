from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Tuple

from src.data.dataloaders import AugmentationConfig, load_manifest
from src.model.backbones.efficientnet_backbone import EfficientNetBackboneConfig
from src.model.backbones.residual_noise_branch import ResidualNoiseConfig
from src.model.backbones.swin_backbone import SwinBackboneConfig
from src.model.feature_fusion import FeatureFusionConfig
from src.model.hybrid_ngiml import HybridNGIMLConfig, HybridNGIMLOptimizerConfig, OptimizerGroupConfig
from src.model.losses import MultiStageLossConfig
from src.model.unet_decoder import UNetDecoderConfig


def _norm(value: str) -> str:
    return str(value).replace("\\", "/")


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
    model_cfg = HybridNGIMLConfig(
        efficientnet=EfficientNetBackboneConfig(pretrained=True),
        swin=SwinBackboneConfig(model_name="swin_tiny_patch4_window7_224", pretrained=True),
        residual=ResidualNoiseConfig(num_kernels=3, base_channels=32, num_stages=4),
        fusion=FeatureFusionConfig(fusion_channels=(64, 128, 192, 256)),
        decoder=UNetDecoderConfig(decoder_channels=None, out_channels=1, per_stage_heads=True),
        optimizer=HybridNGIMLOptimizerConfig(
            efficientnet=OptimizerGroupConfig(lr=3e-5, weight_decay=1e-4),
            swin=OptimizerGroupConfig(lr=1e-5, weight_decay=5e-5),
            residual=OptimizerGroupConfig(lr=2e-4, weight_decay=1e-4),
            fusion=OptimizerGroupConfig(lr=2e-4, weight_decay=1e-4),
            decoder=OptimizerGroupConfig(lr=2e-4, weight_decay=1e-4),
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        use_low_level=True,
        use_context=True,
        use_residual=True,
    )

    loss_cfg = MultiStageLossConfig(
        dice_weight=1.0,
        bce_weight=1.0,
        pos_weight=2.0,
        stage_weights=None,
        smooth=1e-6,
    )

    default_aug = AugmentationConfig(
        enable=True,
        views_per_sample=1,
        enable_flips=True,
        enable_rotations=True,
        max_rotation_degrees=5.0,
        enable_random_crop=True,
        crop_scale_range=(0.9, 1.0),
        object_crop_bias_prob=0.75,
        min_fg_pixels_for_object_crop=8,
        enable_elastic=True,
        elastic_prob=0.25,
        elastic_alpha=8.0,
        elastic_sigma=5.0,
        enable_color_jitter=True,
        brightness_jitter_factors=(0.9, 1.1),
        contrast_jitter_factors=(0.9, 1.1),
        enable_noise=True,
        noise_std_range=(0.0, 0.02),
    )

    per_dataset_aug = {
        "IMD2020": AugmentationConfig(
            enable=True,
            views_per_sample=8,
            enable_flips=True,
            enable_rotations=True,
            max_rotation_degrees=8.0,
            enable_random_crop=True,
            crop_scale_range=(0.75, 1.0),
            object_crop_bias_prob=0.9,
            min_fg_pixels_for_object_crop=4,
            enable_elastic=True,
            elastic_prob=0.35,
            elastic_alpha=10.0,
            elastic_sigma=5.0,
            enable_color_jitter=True,
            brightness_jitter_factors=(0.85, 1.15),
            contrast_jitter_factors=(0.85, 1.15),
            enable_noise=True,
            noise_std_range=(0.0, 0.03),
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
    return {
        "manifest": str(manifest_path),
        "output_dir": output_dir,
        "batch_size": 6,
        "grad_accum_steps": 2,
        "epochs": 50,
        "num_workers": 0,
        "amp": True,
        "grad_clip": 1.0,
        "val_every": 1,
        "checkpoint_every": 1,
        "resume": None,
        "auto_resume": True,
        "round_robin_seed": 42,
        "prefetch_factor": 2,
        "persistent_workers": False,
        "drop_last": True,
        "views_per_sample": 1,
        "max_rotation_degrees": 5.0,
        "noise_std_max": 0.02,
        "disable_aug": False,
        "device": "cuda",
        "aug_seed": 42,
        "seed": 42,
        "warmup_epochs": 3,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
        "threshold_metric": "dice",
        "compute_foreground_ratio": True,
        "auto_pos_weight": True,
        "pos_weight_min": 1.0,
        "pos_weight_max": 8.0,
        "ema_enabled": True,
        "ema_decay": 0.999,
        "hard_mining_enabled": True,
        "hard_mining_start_epoch": 3,
        "hard_mining_weight": 0.1,
        "hard_mining_gamma": 1.0,
        "default_aug": default_aug,
        "per_dataset_aug": per_dataset_aug,
        "model_config": model_cfg,
        "loss_config": loss_cfg,
    }


def apply_colab_runtime_settings(
    training_config: dict,
    balance_sampling: bool = True,
    local_cache_dir: str | None = None,
    tune_for_large_batch: bool = False,
) -> dict:
    recommended_workers = max(2, min(8, (os.cpu_count() or 4) // 2))
    cache_dir = local_cache_dir or "/content/cache"
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

    model_cfg = training_config.get("model_config")
    if model_cfg is not None and getattr(model_cfg, "optimizer", None) is not None:
        model_cfg.optimizer.efficientnet.weight_decay = 1e-4
        model_cfg.optimizer.swin.weight_decay = 5e-5
        model_cfg.optimizer.fusion.weight_decay = 1e-4
        model_cfg.optimizer.decoder.weight_decay = 1e-4

        if tune_for_large_batch:
            training_config["batch_size"] = 12
            training_config["grad_accum_steps"] = 1
            model_cfg.optimizer.efficientnet.lr = 4.5e-5
            model_cfg.optimizer.swin.lr = 1.5e-5
            model_cfg.optimizer.residual.lr = 3e-4
            model_cfg.optimizer.fusion.lr = 3e-4
            model_cfg.optimizer.decoder.lr = 3e-4


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

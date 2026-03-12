from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode

from src.data.dataloaders import (
    _compute_high_pass_fallback,
    _load_from_npz,
    _load_from_tar_npz,
    _load_image,
    _normalize,
    load_manifest,
)
from src.data.config import SampleRecord
from src.model.hybrid_ngiml import HybridNGIML
from tools.colab_train_helpers import build_default_components
from tools.local_train_helpers import build_manifest_from_prepared


def _zero_flop_jit(_inputs, _outputs) -> Counter[str]:
    return Counter()


def _build_flop_analysis(model: torch.nn.Module, sample: torch.Tensor):
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import elementwise_flop_counter, generic_activation_jit

    elementwise = elementwise_flop_counter(1, 0)
    analysis = FlopCountAnalysis(model, sample).unsupported_ops_warnings(False)
    analysis = analysis.set_op_handle(
        "aten::add",
        elementwise,
        "aten::sub",
        elementwise,
        "aten::rsub",
        elementwise,
        "aten::mul",
        elementwise,
        "aten::div",
        elementwise,
        "aten::mean",
        elementwise,
        "aten::ne",
        elementwise,
        "aten::sigmoid",
        generic_activation_jit("sigmoid"),
        "aten::gelu",
        generic_activation_jit("gelu"),
        "aten::silu_",
        generic_activation_jit("silu"),
        "aten::softmax",
        generic_activation_jit("softmax"),
        "aten::pad",
        _zero_flop_jit,
        "aten::fill_",
        _zero_flop_jit,
        "aten::repeat",
        _zero_flop_jit,
        "aten::expand_as",
        _zero_flop_jit,
        "aten::feature_dropout",
        _zero_flop_jit,
    )
    return analysis


def find_latest_checkpoint(runs_root: Path) -> Path:
    runs_root = Path(runs_root)
    candidates = sorted(runs_root.rglob("checkpoints/checkpoint_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {runs_root}/**/checkpoints/checkpoint_epoch_*.pt")
    return candidates[-1]


def ensure_local_manifest(prepared_root: Path, manifest_path: Path | None = None) -> Path:
    prepared_root = Path(prepared_root)
    if manifest_path is not None and Path(manifest_path).exists():
        return Path(manifest_path)

    default_manifest = prepared_root / "manifest_local.json"
    if default_manifest.exists() and default_manifest.stat().st_size > 0:
        return default_manifest

    return build_manifest_from_prepared(prepared_root, manifest_out=default_manifest)


def load_default_threshold(checkpoint_path: Path, fallback: float = 0.5) -> float:
    checkpoint_path = Path(checkpoint_path)
    candidate_files = [
        checkpoint_path.parent / "best_threshold.json",
        checkpoint_path.parent.parent / "best_threshold.json",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            import json

            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            threshold = payload.get("threshold", fallback)
            return float(threshold)
        except Exception:
            continue
    return float(fallback)


def resolve_threshold_for_checkpoint(
    checkpoint_path: Path,
    checkpoint_epoch: int | None = None,
    fallback: float = 0.5,
) -> tuple[float, str]:
    checkpoint_path = Path(checkpoint_path)

    # First prefer explicit threshold metadata when it belongs to this checkpoint.
    candidate_files = [
        checkpoint_path.parent / "best_threshold.json",
        checkpoint_path.parent.parent / "best_threshold.json",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            import json

            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload_ckpt = str(payload.get("checkpoint_path", ""))
            if payload_ckpt and Path(payload_ckpt).name == checkpoint_path.name:
                return float(payload.get("threshold", fallback)), f"{candidate.name}:matching_checkpoint"
            if checkpoint_epoch is not None and int(payload.get("epoch", -1)) == int(checkpoint_epoch):
                return float(payload.get("threshold", fallback)), f"{candidate.name}:matching_epoch"
        except Exception:
            continue

    # Fallback to per-epoch checkpoint metrics when available.
    metrics_candidates = [
        checkpoint_path.parent / "checkpoint_metrics.json",
        checkpoint_path.parent.parent / "checkpoint_metrics.json",
    ]
    for metrics_path in metrics_candidates:
        if not metrics_path.exists():
            continue
        try:
            import json

            with open(metrics_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                continue

            by_path = next(
                (
                    record for record in payload
                    if isinstance(record, dict)
                    and str(record.get("checkpoint_path", "")).endswith(checkpoint_path.name)
                    and record.get("val_threshold") is not None
                ),
                None,
            )
            if by_path is not None:
                return float(by_path["val_threshold"]), f"{metrics_path.name}:matching_path"

            if checkpoint_epoch is not None:
                by_epoch = next(
                    (
                        record for record in reversed(payload)
                        if isinstance(record, dict)
                        and int(record.get("epoch", -1)) == int(checkpoint_epoch)
                        and record.get("val_threshold") is not None
                    ),
                    None,
                )
                if by_epoch is not None:
                    return float(by_epoch["val_threshold"]), f"{metrics_path.name}:matching_epoch"
        except Exception:
            continue

    return float(load_default_threshold(checkpoint_path, fallback=fallback)), "fallback"


def _infer_fusion_channels_from_state_dict(model_state: dict) -> tuple[int, ...] | None:
    stage_channels: dict[int, int] = {}
    pattern = re.compile(r"^fusion\.stages\.(\d+)\.projections\.[^.]+\.weight$")
    for key, tensor in model_state.items():
        match = pattern.match(key)
        if not match or not isinstance(tensor, torch.Tensor):
            continue
        stage_idx = int(match.group(1))
        out_channels = int(tensor.shape[0])
        stage_channels[stage_idx] = out_channels

    if not stage_channels:
        return None

    ordered = [stage_channels[idx] for idx in sorted(stage_channels)]
    return tuple(int(value) for value in ordered)


def _build_model_config_from_checkpoint(checkpoint: dict) -> tuple[object, str]:
    model_cfg, _, _, _ = build_default_components()

    train_config = checkpoint.get("train_config") if isinstance(checkpoint, dict) else None
    model_config = train_config.get("model_config") if isinstance(train_config, dict) else None

    if isinstance(model_config, dict):
        fusion_cfg = model_config.get("fusion")
        if isinstance(fusion_cfg, dict):
            fusion_channels = fusion_cfg.get("fusion_channels")
            if isinstance(fusion_channels, (list, tuple)) and fusion_channels:
                model_cfg.fusion.fusion_channels = tuple(int(value) for value in fusion_channels)
            for attr in ("noise_branch", "noise_skip_stage", "noise_decay", "norm", "activation", "fusion_refinement"):
                if attr in fusion_cfg and hasattr(model_cfg.fusion, attr):
                    setattr(model_cfg.fusion, attr, fusion_cfg[attr])

        decoder_cfg = model_config.get("decoder")
        if isinstance(decoder_cfg, dict):
            for attr in (
                "decoder_channels",
                "out_channels",
                "norm",
                "activation",
                "per_stage_heads",
                "enable_edge_guidance",
                "use_dropout",
                "dropout_p",
            ):
                if attr in decoder_cfg and hasattr(model_cfg.decoder, attr):
                    setattr(model_cfg.decoder, attr, decoder_cfg[attr])

        for attr in (
            "use_low_level",
            "use_context",
            "use_residual",
            "enable_residual_attention",
            "gradient_checkpointing",
            "flash_attention",
            "xformers",
        ):
            if attr in model_config and hasattr(model_cfg, attr):
                setattr(model_cfg, attr, model_config[attr])

        return model_cfg, "train_config.model_config"

    inferred_channels = _infer_fusion_channels_from_state_dict(checkpoint.get("model_state", {}))
    if inferred_channels:
        model_cfg.fusion.fusion_channels = inferred_channels
        return model_cfg, "state_dict.inferred_fusion_channels"

    return model_cfg, "defaults"


def _load_state_dict_with_fallback(model: HybridNGIML, model_state: dict) -> tuple[list[str], list[str], int]:
    try:
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        return list(missing), list(unexpected), 0
    except RuntimeError:
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in model_state.items()
            if key in current_state and hasattr(value, "shape") and current_state[key].shape == value.shape
        }
        skipped = int(len(model_state) - len(compatible_state))
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        return list(missing), list(unexpected), skipped


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device | None = None) -> tuple[HybridNGIML, torch.device, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    model_cfg, config_source = _build_model_config_from_checkpoint(checkpoint)
    model = HybridNGIML(model_cfg).to(device)

    missing, unexpected, skipped_mismatched = _load_state_dict_with_fallback(model, checkpoint["model_state"])
    model.eval()
    resolved_threshold, threshold_source = resolve_threshold_for_checkpoint(
        Path(checkpoint_path),
        checkpoint_epoch=checkpoint_epoch,
        fallback=0.5,
    )

    info = {
        "epoch": checkpoint_epoch,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_mismatched_keys": int(skipped_mismatched),
        "config_source": str(config_source),
        "fusion_channels": tuple(int(value) for value in model.cfg.fusion.fusion_channels),
        "default_threshold": float(resolved_threshold),
        "threshold_source": str(threshold_source),
        "max_short_side": int((checkpoint.get("train_config") or {}).get("max_short_side", 0) or 0),
    }
    setattr(model, "default_threshold", float(info["default_threshold"]))
    return model, device, info


def select_manifest_sample(
    manifest_path: Path,
    split_priority: Sequence[str] = ("test", "val", "train"),
    fake_only: bool = True,
) -> SampleRecord:
    manifest = load_manifest(manifest_path)
    samples = manifest.samples

    if fake_only:
        fake_samples = [s for s in samples if int(getattr(s, "label", 0)) == 1 or s.mask_path is not None]
    else:
        fake_samples = samples

    for split_name in split_priority:
        split_samples = [s for s in fake_samples if s.split == split_name]
        if split_samples:
            return split_samples[0]

    if fake_samples:
        return fake_samples[0]

    raise RuntimeError(f"No samples available in manifest: {manifest_path}")


def _resolve_possible_local_path(path_str: str) -> str:
    path = Path(path_str)
    return path.as_posix()


def resize_for_inference(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
    high_pass: torch.Tensor | None = None,
    max_short_side: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    cap = int(max_short_side or 0)
    if cap <= 0:
        return image, mask, high_pass

    h, w = image.shape[-2:]
    short_side = min(h, w)
    if short_side <= 0 or short_side <= cap:
        return image, mask, high_pass

    scale = float(cap) / float(short_side)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    image = TVF.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    if mask is not None:
        mask = TVF.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
    if high_pass is not None:
        high_pass = TVF.resize(high_pass, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    return image, mask, high_pass


def load_image_mask_from_record(
    record: SampleRecord,
    max_short_side: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    image_path = str(record.image_path)
    if "::" in image_path and image_path.endswith(".npz"):
        loaded = _load_from_tar_npz(image_path)
        image, mask, high_pass = loaded[:3]
    elif image_path.endswith(".npz"):
        loaded = _load_from_npz(_resolve_possible_local_path(image_path))
        image, mask, high_pass = loaded[:3]
    else:
        image = _load_image(_resolve_possible_local_path(image_path))
        high_pass = None
        mask = None
        if record.mask_path is not None:
            loaded = _load_image(_resolve_possible_local_path(record.mask_path))
            mask = loaded[:1] if loaded.shape[0] > 1 else loaded
        if record.high_pass_path is not None:
            loaded_high = _load_image(_resolve_possible_local_path(record.high_pass_path))
            high_pass = loaded_high if loaded_high.shape[0] in (1, 3) else loaded_high[:3]

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0

    if mask is None:
        mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.float32)
    else:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] > 1:
            mask = mask[:1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        if tuple(mask.shape[-2:]) != tuple(image.shape[-2:]):
            mask = F.interpolate(mask.unsqueeze(0), size=image.shape[-2:], mode="nearest").squeeze(0)

    if high_pass is not None:
        high_pass = high_pass.float()
        if high_pass.ndim == 2:
            high_pass = high_pass.unsqueeze(0)
        if high_pass.shape[0] == 1:
            high_pass = high_pass.repeat(3, 1, 1)
        elif high_pass.shape[0] > 3:
            high_pass = high_pass[:3]
        if high_pass.max() > 1.0:
            high_pass = high_pass / 255.0
        if tuple(high_pass.shape[-2:]) != tuple(image.shape[-2:]):
            high_pass = F.interpolate(high_pass.unsqueeze(0), size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
    else:
        high_pass = _compute_high_pass_fallback(image)

    image, mask, high_pass = resize_for_inference(image, mask=mask, high_pass=high_pass, max_short_side=max_short_side)
    return image, mask, high_pass


def normalize_image_for_inference(image: torch.Tensor, normalization_mode: str = "zero_one") -> torch.Tensor:
    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    return _normalize(image, str(normalization_mode).strip().lower())


def predict_probability_map(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    normalization_mode: str = "zero_one",
    high_pass: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = normalize_image_for_inference(image, normalization_mode=normalization_mode)
    x = normalized.unsqueeze(0).to(device)
    hp = None
    if high_pass is not None:
        hp = high_pass.float()
        if hp.max() > 1.0:
            hp = hp / 255.0
        hp = hp.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x, target_size=image.shape[-2:], high_pass=hp)[-1]
        prob = torch.sigmoid(logits)[0, 0].detach().cpu()
    return prob


def predict_binary_map(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    normalization_mode: str = "zero_one",
    high_pass: torch.Tensor | None = None,
) -> torch.Tensor:
    prob = predict_probability_map(model, image, device, normalization_mode=normalization_mode, high_pass=high_pass)
    if threshold is None:
        threshold = float(getattr(model, "default_threshold", 0.5))
    return (prob >= float(threshold)).float()


def infer_from_image_path(
    model: HybridNGIML,
    image_path: Path,
    device: torch.device,
    normalization_mode: str = "zero_one",
    max_short_side: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = _load_image(str(Path(image_path).as_posix())).float()
    if image.max() > 1.0:
        image = image / 255.0
    high_pass = _compute_high_pass_fallback(image)
    image, _, high_pass = resize_for_inference(image, mask=None, high_pass=high_pass, max_short_side=max_short_side)
    pred = predict_probability_map(model, image, device, normalization_mode=normalization_mode, high_pass=high_pass)
    return image, pred


def get_model_complexity_stats(
    model: HybridNGIML,
    input_size: tuple[int, int, int, int] = (1, 3, 384, 384),
) -> dict[str, object]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_params = total_params - trainable_params

    stats: dict[str, object] = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(frozen_params),
        "input_size": tuple(int(v) for v in input_size),
    }

    sample_device = next(model.parameters()).device
    sample = torch.randn(*input_size, device=sample_device)

    was_training = model.training
    model.eval()
    try:
        try:
            with torch.no_grad():
                analysis = _build_flop_analysis(model, sample)
                total_flops = float(analysis.total())
                unsupported_ops = {str(name): int(count) for name, count in analysis.unsupported_ops().items()}
            stats["flops"] = total_flops
            stats["macs"] = total_flops / 2.0
            stats["unsupported_ops"] = unsupported_ops
            stats["flops_source"] = "fvcore+custom_op_handles"
            stats["flops_error"] = (
                None
                if not unsupported_ops
                else "FLOPs include custom op-handle estimates; unsupported ops remain in `unsupported_ops`."
            )
        except Exception as fv_error:
            try:
                from thop import profile as thop_profile

                with torch.no_grad():
                    macs, _ = thop_profile(model, inputs=(sample,), verbose=False)
                macs = float(macs)
                stats["macs"] = macs
                stats["flops"] = macs * 2.0
                stats["unsupported_ops"] = None
                stats["flops_source"] = "thop"
                stats["flops_error"] = f"fvcore unavailable ({fv_error}); used thop fallback"
            except Exception as thop_error:
                stats["flops"] = None
                stats["macs"] = None
                stats["unsupported_ops"] = None
                stats["flops_source"] = None
                stats["flops_error"] = (
                    "FLOPs unavailable. "
                    f"fvcore error: {fv_error}. "
                    f"thop error: {thop_error}. "
                    "Try `%pip install fvcore iopath` (or `%pip install thop`) in the active notebook kernel."
                )
    finally:
        model.train(was_training)

    return stats

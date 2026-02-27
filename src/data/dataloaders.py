from __future__ import annotations


import io
import json
import tarfile
import atexit
import warnings
from collections import OrderedDict
import pandas as pd
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import numpy as np
import torch
import torch.nn.functional as NN_F
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import random
import io as pyio

from .config import AugmentationConfig, Manifest, SampleRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npz"}

# Per-process LRU cache of open tar archives to avoid reopening on every sample.
_TAR_CACHE_LIMIT = 8
_TAR_CACHE: "OrderedDict[str, tarfile.TarFile]" = OrderedDict()
_MISSING_HIGH_PASS_WARNED = False


def _close_all_tars() -> None:
    while _TAR_CACHE:
        _, tar = _TAR_CACHE.popitem(last=False)
        try:
            tar.close()
        except Exception:
            pass


atexit.register(_close_all_tars)


def _get_tarfile(archive_path: str) -> tarfile.TarFile:
    tar = _TAR_CACHE.pop(archive_path, None)
    if tar is None or tar.closed:
        tar = tarfile.open(archive_path, "r:*")
    _TAR_CACHE[archive_path] = tar
    if len(_TAR_CACHE) > _TAR_CACHE_LIMIT:
        _, old_tar = _TAR_CACHE.popitem(last=False)
        try:
            old_tar.close()
        except Exception:
            pass
    return tar


def _load_image(path: str) -> torch.Tensor:
    image = torchvision_load_image(path).float() / 255.0
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


def _load_mask(mask_path: str | None, target_hw: Sequence[int]) -> torch.Tensor:
    if mask_path is None:
        return torch.zeros((1, target_hw[0], target_hw[1]), dtype=torch.float32)
    mask = torchvision_load_image(mask_path, as_mask=True).float()
    if mask.shape[0] > 1:
        mask = mask[:1]
    if mask.max() > 1.0:
        mask = mask / 255.0
    if mask.shape[-2:] != tuple(target_hw):
        mask = F.resize(mask, target_hw, interpolation=InterpolationMode.NEAREST)
    return mask


def _safe_scale_to_unit_float32(tensor: torch.Tensor) -> torch.Tensor:
    source_dtype = tensor.dtype
    tensor = tensor.float()
    if tensor.numel() == 0:
        return tensor.to(dtype=torch.float32)
    if source_dtype == torch.uint8:
        return (tensor / 255.0).to(dtype=torch.float32)
    max_value = float(tensor.max().item())
    if max_value > 1.0:
        tensor = tensor / 255.0
    return tensor.to(dtype=torch.float32)


def _compute_high_pass_fallback(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=0, keepdim=True)
    blurred = NN_F.avg_pool2d(gray.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    high_pass = torch.abs(gray - blurred)
    high_pass = torch.clamp(high_pass * 4.0, 0.0, 1.0)
    return high_pass.repeat(3, 1, 1).to(dtype=torch.float32)


def _load_from_npz(path: str | io.BytesIO) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    global _MISSING_HIGH_PASS_WARNED
    with np.load(path, allow_pickle=False) as data:
        image_np = data["image"]
        image = torch.from_numpy(image_np)
        if image.ndim == 3:
            if image.shape[0] in (1, 3):
                pass
            elif image.shape[-1] in (1, 3):
                image = image.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected image shape in NPZ: {image.shape}")
        else:
            raise ValueError(f"Image array must be 3D, got shape {image.shape}")

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = _safe_scale_to_unit_float32(image)

        mask = None
        if "mask" in data:
            mask_np = data["mask"]
            mask = torch.from_numpy(mask_np)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1)
            elif mask.ndim == 3 and mask.shape[0] == 1:
                pass
            else:
                raise ValueError(f"Unexpected mask shape in NPZ: {mask.shape}")
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = mask.float()

        high_pass = None
        if "high_pass" in data:
            hp_np = data["high_pass"]
            if hp_np.size > 0:
                high_pass = torch.from_numpy(hp_np)
                if high_pass.ndim == 2:
                    high_pass = high_pass.unsqueeze(0)
                elif high_pass.ndim == 3 and high_pass.shape[-1] in (1, 3):
                    high_pass = high_pass.permute(2, 0, 1)
                elif high_pass.ndim == 3 and high_pass.shape[0] in (1, 3):
                    pass
                else:
                    raise ValueError(f"Unexpected high_pass shape in NPZ: {high_pass.shape}")
                if high_pass.shape[0] == 1:
                    high_pass = high_pass.repeat(3, 1, 1)
                high_pass = _safe_scale_to_unit_float32(high_pass)

        if high_pass is None:
            if not _MISSING_HIGH_PASS_WARNED:
                warnings.warn(
                    "NPZ sample missing high_pass; computing lightweight fallback from image.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _MISSING_HIGH_PASS_WARNED = True
            high_pass = _compute_high_pass_fallback(image)

    return image, mask, high_pass


def _load_from_tar_npz(tar_spec: str) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if "::" not in tar_spec:
        raise ValueError(f"Invalid tar npz spec: {tar_spec}")
    archive_path, member_name = tar_spec.split("::", 1)
    tar = _get_tarfile(archive_path)
    member = tar.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(f"Missing member {member_name} in {archive_path}")
    npz_bytes = member.read()
    return _load_from_npz(io.BytesIO(npz_bytes))


def torchvision_load_image(path: str, as_mask: bool = False) -> torch.Tensor:
    # Lazy import to avoid hard dependency at module import time.
    from torchvision.io import read_image

    image = read_image(path)
    if as_mask and image.shape[0] > 1:
        image = image[:1]
    return image


class PerDatasetDataset(Dataset):
    def __init__(self, samples: Sequence[SampleRecord], aug_cfg: AugmentationConfig, training: bool) -> None:
        self.samples = list(samples)
        self.aug_cfg = aug_cfg
        self.training = training
        self.sample_labels = [int(sample.label) for sample in self.samples]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:  # type: ignore[override]
        record = self.samples[index]

        if "::" in record.image_path and record.image_path.endswith(".npz"):
            image, mask, high_pass = _load_from_tar_npz(record.image_path)
        elif record.image_path.endswith(".npz"):
            image, mask, high_pass = _load_from_npz(record.image_path)
        else:
            image = _load_image(record.image_path)
            mask = _load_mask(record.mask_path, image.shape[-2:])
            high_pass = None

        if mask is None:
            mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.float32)

        # --- JPEG compression augmentation ---
        cfg = self.aug_cfg
        if cfg.jpeg_aug_prob > 0 and self.training and random.random() < cfg.jpeg_aug_prob:
            # Convert to PIL, compress, reload as tensor
            img = image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            quality = random.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max)
            buf = pyio.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            pil_img_jpeg = Image.open(buf).convert("RGB")
            image = F.to_tensor(pil_img_jpeg)

        # --- Multi-scale training (random resize before crop) ---
        if cfg.multiscale_training and self.training:
            short_min, short_max = cfg.multiscale_short_side_range
            h, w = image.shape[-2:]
            short_side = min(h, w)
            target_short = random.randint(short_min, short_max)
            scale = target_short / float(short_side)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            image = F.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
            if high_pass is not None:
                high_pass = F.resize(high_pass, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

        label = torch.tensor(record.label, dtype=torch.long)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "dataset": record.dataset,
            "high_pass": high_pass,
        }


class CombinedDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = list(datasets)
        self.offsets: List[int] = []
        total = 0
        for ds in self.datasets:
            self.offsets.append(total)
            total += len(ds)
        self.total_len = total

    def __len__(self) -> int:  # type: ignore[override]
        return self.total_len

    def __getitem__(self, index: int) -> dict[str, object]:  # type: ignore[override]
        ds_idx = bisect_right(self.offsets, index) - 1
        local_index = index - self.offsets[ds_idx]
        return self.datasets[ds_idx][local_index]


def _build_real_fake_balanced_sampler(
    combined_dataset: CombinedDataset,
    pos_ratio: float,
    seed: int | None,
    num_samples: int | None = None,
) -> WeightedRandomSampler | None:
    labels: list[int] = []
    for ds in combined_dataset.datasets:
        ds_labels = getattr(ds, "sample_labels", None)
        if ds_labels is None:
            return None
        labels.extend(int(v) for v in ds_labels)

    if not labels:
        return None

    label_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor, minlength=2).float()
    if class_counts.sum().item() <= 0:
        return None

    pos_ratio = float(min(max(pos_ratio, 0.05), 0.95))
    neg_ratio = 1.0 - pos_ratio

    class_weights = torch.zeros_like(class_counts)
    class_weights[0] = neg_ratio / class_counts[0].clamp_min(1.0)
    class_weights[1] = pos_ratio / class_counts[1].clamp_min(1.0)
    sample_weights = class_weights[label_tensor]

    target_num_samples = int(num_samples) if num_samples is not None else len(labels)
    target_num_samples = max(1, target_num_samples)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=target_num_samples,
        replacement=True,
        generator=generator,
    )


class RoundRobinSampler(Sampler[int]):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        shuffle: bool = True,
        seed: int | None = None,
        balance: bool = True,
    ) -> None:
        self.datasets = list(datasets)
        self.shuffle = shuffle
        self.seed = seed
        self.balance = balance
        self._iteration = 0
        self.lengths = [len(ds) for ds in self.datasets]
        self.offsets: List[int] = []
        total = 0
        for length in self.lengths:
            self.offsets.append(total)
            total += length

    def __iter__(self):  # type: ignore[override]
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed + self._iteration)
        self._iteration += 1
        per_dataset_indices: List[List[int]] = []
        for length in self.lengths:
            indices = list(range(length))
            if self.shuffle:
                perm = torch.randperm(length, generator=generator).tolist()
                indices = [indices[i] for i in perm]
            per_dataset_indices.append(indices)

        max_len = max(self.lengths)
        total_rounds = max_len
        for offset in range(total_rounds):
            for ds_idx, indices in enumerate(per_dataset_indices):
                if not indices:
                    continue
                local_len = len(indices)
                if not self.balance and offset >= local_len:
                    continue
                local_idx = indices[offset % local_len] if self.balance else indices[offset]
                yield self.offsets[ds_idx] + local_idx

    def __len__(self) -> int:  # type: ignore[override]
        if self.balance:
            return max(self.lengths) * len(self.datasets)
        return sum(self.lengths)


class RoundRobinBalancedClassSampler(Sampler[int]):
    """Round-robin across datasets while balancing real/fake sampling within each dataset."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        pos_ratio: float = 0.5,
        shuffle: bool = True,
        seed: int | None = None,
        balance: bool = True,
    ) -> None:
        self.datasets = list(datasets)
        self.pos_ratio = float(min(max(pos_ratio, 0.05), 0.95))
        self.shuffle = shuffle
        self.seed = seed
        self.balance = balance
        self._iteration = 0

        self.lengths = [len(ds) for ds in self.datasets]
        self.offsets: List[int] = []
        total = 0
        for length in self.lengths:
            self.offsets.append(total)
            total += length

    def __iter__(self):  # type: ignore[override]
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed) + self._iteration)
        self._iteration += 1

        per_dataset_pos: List[List[int]] = []
        per_dataset_neg: List[List[int]] = []
        for ds_idx, ds in enumerate(self.datasets):
            labels = getattr(ds, "sample_labels", None)
            if labels is None or len(labels) != self.lengths[ds_idx]:
                labels = [0 for _ in range(self.lengths[ds_idx])]

            pos_indices = [i for i, lbl in enumerate(labels) if int(lbl) == 1]
            neg_indices = [i for i, lbl in enumerate(labels) if int(lbl) == 0]

            if self.shuffle:
                if pos_indices:
                    perm_pos = torch.randperm(len(pos_indices), generator=generator).tolist()
                    pos_indices = [pos_indices[i] for i in perm_pos]
                if neg_indices:
                    perm_neg = torch.randperm(len(neg_indices), generator=generator).tolist()
                    neg_indices = [neg_indices[i] for i in perm_neg]

            per_dataset_pos.append(pos_indices)
            per_dataset_neg.append(neg_indices)

        pos_cursors = [0 for _ in self.datasets]
        neg_cursors = [0 for _ in self.datasets]

        def _next_from(pool: List[int], cursor_list: List[int], ds_idx: int) -> int | None:
            if not pool:
                return None
            cursor = cursor_list[ds_idx]
            idx = pool[cursor % len(pool)]
            cursor_list[ds_idx] = cursor + 1
            return idx

        max_len = max(self.lengths) if self.lengths else 0
        total_rounds = max_len
        for offset in range(total_rounds):
            for ds_idx, ds_len in enumerate(self.lengths):
                if ds_len == 0:
                    continue
                if not self.balance and offset >= ds_len:
                    continue

                draw_pos = bool(torch.rand((), generator=generator).item() < self.pos_ratio)
                pos_pool = per_dataset_pos[ds_idx]
                neg_pool = per_dataset_neg[ds_idx]

                if draw_pos:
                    local_idx = _next_from(pos_pool, pos_cursors, ds_idx)
                    if local_idx is None:
                        local_idx = _next_from(neg_pool, neg_cursors, ds_idx)
                else:
                    local_idx = _next_from(neg_pool, neg_cursors, ds_idx)
                    if local_idx is None:
                        local_idx = _next_from(pos_pool, pos_cursors, ds_idx)

                if local_idx is None:
                    continue
                yield self.offsets[ds_idx] + local_idx

    def __len__(self) -> int:  # type: ignore[override]
        if self.balance:
            return max(self.lengths) * len(self.datasets)
        return sum(self.lengths)


def _normalize(
    image: torch.Tensor,
    mode: str,
    imagenet_mean: torch.Tensor | None = None,
    imagenet_std: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "zero_one":
        return image
    if mode == "imagenet":
        if imagenet_mean is None or imagenet_std is None:
            mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        else:
            mean = imagenet_mean
            std = imagenet_std
        return (image - mean) / std
    return image


def _apply_gpu_augmentations(
    image: torch.Tensor,
    mask: torch.Tensor,
    cfg: AugmentationConfig,
    high_pass: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    def _rand_scalar() -> torch.Tensor:
        return torch.rand((), device=image.device, generator=generator)

    def _rand_int(low: int, high: int) -> int:
        if high <= low:
            return low
        return int(torch.randint(low, high, (), device=image.device, generator=generator).item())

    def _smooth_displacement(field: torch.Tensor, sigma: float) -> torch.Tensor:
        kernel = max(3, int(2 * round(float(max(0.5, sigma))) + 1))
        if kernel % 2 == 0:
            kernel += 1
        smoothed = NN_F.avg_pool2d(field, kernel_size=kernel, stride=1, padding=kernel // 2)
        return NN_F.avg_pool2d(smoothed, kernel_size=kernel, stride=1, padding=kernel // 2)

    def _elastic_deform(
        image_t: torch.Tensor,
        mask_t: torch.Tensor,
        high_pass_t: torch.Tensor | None,
        alpha: float,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        _, h, w = image_t.shape
        if h < 4 or w < 4:
            return image_t, mask_t, high_pass_t

        dx = torch.rand((1, 1, h, w), device=image_t.device, generator=generator) * 2.0 - 1.0
        dy = torch.rand((1, 1, h, w), device=image_t.device, generator=generator) * 2.0 - 1.0
        dx = _smooth_displacement(dx, sigma=sigma) * alpha
        dy = _smooth_displacement(dy, sigma=sigma) * alpha

        y_lin = torch.linspace(-1.0, 1.0, h, device=image_t.device)
        x_lin = torch.linspace(-1.0, 1.0, w, device=image_t.device)
        grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

        disp_x = (dx.squeeze(0).squeeze(0) / max(w - 1, 1)) * 2.0
        disp_y = (dy.squeeze(0).squeeze(0) / max(h - 1, 1)) * 2.0
        disp_grid = torch.stack((disp_x, disp_y), dim=-1).unsqueeze(0)
        grid = torch.clamp(base_grid + disp_grid, -1.0, 1.0)

        image_out = NN_F.grid_sample(
            image_t.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        ).squeeze(0)
        mask_out = NN_F.grid_sample(
            mask_t.unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        high_pass_out = None
        if high_pass_t is not None:
            high_pass_out = NN_F.grid_sample(
                high_pass_t.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            ).squeeze(0)

        return image_out, mask_out, high_pass_out

    if cfg.enable_flips:
        if _rand_scalar() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
            if high_pass is not None:
                high_pass = torch.flip(high_pass, dims=[2])
        if _rand_scalar() < 0.2:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
            if high_pass is not None:
                high_pass = torch.flip(high_pass, dims=[1])

    if cfg.enable_rotations and cfg.max_rotation_degrees > 0:
        angle = float((_rand_scalar() * 2 - 1) * cfg.max_rotation_degrees)
        if abs(angle) > 1e-3:
            image = F.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR)
            mask = F.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST)
            if high_pass is not None:
                high_pass = F.rotate(high_pass, angle=angle, interpolation=InterpolationMode.BILINEAR)

    if cfg.enable_random_crop:
        scale = float(
            cfg.crop_scale_range[0]
            + _rand_scalar() * (cfg.crop_scale_range[1] - cfg.crop_scale_range[0])
        )
        _, h, w = image.shape
        crop_h = max(1, int(h * scale))
        crop_w = max(1, int(w * scale))
        if crop_h < h or crop_w < w:
            top = int(_rand_scalar() * (h - crop_h + 1))
            left = int(_rand_scalar() * (w - crop_w + 1))

            object_crop_prob = float(min(max(getattr(cfg, "object_crop_bias_prob", 0.0), 0.0), 1.0))
            min_fg = int(max(1, getattr(cfg, "min_fg_pixels_for_object_crop", 1)))
            if object_crop_prob > 0 and _rand_scalar() < object_crop_prob:
                fg_coords = torch.nonzero(mask[0] > 0.5, as_tuple=False)
                if fg_coords.shape[0] >= min_fg:
                    coord_idx = _rand_int(0, fg_coords.shape[0])
                    center_y = int(fg_coords[coord_idx, 0].item())
                    center_x = int(fg_coords[coord_idx, 1].item())
                    max_top = h - crop_h
                    max_left = w - crop_w
                    top = max(0, min(center_y - crop_h // 2, max_top))
                    left = max(0, min(center_x - crop_w // 2, max_left))

            image = F.resized_crop(image, top, left, crop_h, crop_w, size=[h, w], interpolation=InterpolationMode.BILINEAR)
            mask = F.resized_crop(mask, top, left, crop_h, crop_w, size=[h, w], interpolation=InterpolationMode.NEAREST)
            if high_pass is not None:
                high_pass = F.resized_crop(high_pass, top, left, crop_h, crop_w, size=[h, w], interpolation=InterpolationMode.BILINEAR)

    if getattr(cfg, "enable_elastic", False):
        elastic_prob = float(min(max(getattr(cfg, "elastic_prob", 0.0), 0.0), 1.0))
        elastic_alpha = float(max(0.0, getattr(cfg, "elastic_alpha", 0.0)))
        elastic_sigma = float(max(0.5, getattr(cfg, "elastic_sigma", 1.0)))
        if elastic_prob > 0 and elastic_alpha > 0 and _rand_scalar() < elastic_prob:
            image, mask, high_pass = _elastic_deform(
                image,
                mask,
                high_pass,
                alpha=elastic_alpha,
                sigma=elastic_sigma,
            )

    if cfg.enable_color_jitter:
        brightness_range = getattr(cfg, "brightness_jitter_factors", getattr(cfg, "color_jitter_factors", (0.9, 1.1)))
        contrast_range = getattr(cfg, "contrast_jitter_factors", (0.9, 1.1))

        brightness = float(brightness_range[0] + _rand_scalar() * (brightness_range[1] - brightness_range[0]))
        contrast = float(contrast_range[0] + _rand_scalar() * (contrast_range[1] - contrast_range[0]))

        mean = image.mean(dim=(1, 2), keepdim=True)
        image = (image - mean) * contrast + mean
        image = image * brightness
        image = torch.clamp(image, 0.0, 1.0)

    if cfg.enable_noise and cfg.noise_std_range[1] > 0:
        std = float(cfg.noise_std_range[0] + _rand_scalar() * (cfg.noise_std_range[1] - cfg.noise_std_range[0]))
        if std > 0:
            noise = torch.randn_like(image) * std
            image = torch.clamp(image + noise, 0.0, 1.0)

    return image, mask, high_pass


def _collate_builder(
    per_dataset_aug: Dict[str, AugmentationConfig],
    normalization_mode: str,
    training: bool,
    aug_seed: int | None = None,
):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    aug_generator: torch.Generator | None = None

    def _collate(batch: List[dict[str, object]]) -> dict[str, object]:
        nonlocal aug_generator
        if aug_seed is not None and aug_generator is None:
            aug_generator = torch.Generator()
            worker_info = torch.utils.data.get_worker_info()
            worker_offset = worker_info.id if worker_info is not None else 0
            aug_generator.manual_seed(int(aug_seed) + int(worker_offset))

        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        datasets: List[str] = []
        high_passes: List[torch.Tensor] = []
        collect_high_pass = True

        for sample in batch:
            image = sample["image"]
            mask = sample["mask"]
            label = sample["label"]
            dataset_name = str(sample["dataset"])
            high_pass = sample.get("high_pass")
            if high_pass is None:
                collect_high_pass = False
            aug_cfg = per_dataset_aug.get(dataset_name, AugmentationConfig(enable=False))
            views = aug_cfg.views_per_sample if aug_cfg.enable else 1
            views = max(1, views)

            base_image = image
            base_mask = mask
            base_high_pass = high_pass

            for _ in range(views):
                view_image = base_image
                view_mask = base_mask
                view_high_pass = base_high_pass

                if training and aug_cfg.enable:
                    view_image, view_mask, view_high_pass = _apply_gpu_augmentations(
                        view_image,
                        view_mask,
                        aug_cfg,
                        high_pass=view_high_pass,
                        generator=aug_generator,
                    )

                view_image = _normalize(
                    view_image,
                    normalization_mode,
                    imagenet_mean=imagenet_mean,
                    imagenet_std=imagenet_std,
                )

                images.append(view_image)
                masks.append(view_mask)
                labels.append(label)
                datasets.append(dataset_name)
                if collect_high_pass and view_high_pass is not None:
                    high_passes.append(view_high_pass)

        # Ensure all images/masks/high_pass tensors have the same H,W before stacking.
        # Pad to the maximum H,W in the batch (pad on right and bottom).
        shapes = [img.shape for img in images]
        need_pad = any(s != shapes[0] for s in shapes)

        if need_pad:
            max_c = max(s[0] for s in shapes)
            max_h = max(s[1] for s in shapes)
            max_w = max(s[2] for s in shapes)

            padded_images: List[torch.Tensor] = []
            padded_masks: List[torch.Tensor] = []
            for img, m in zip(images, masks):
                c, h, w = img.shape
                # pad channels if necessary (unlikely)
                if c < max_c:
                    pad_ch = max_c - c
                    img = torch.cat([img, torch.zeros((pad_ch, h, w), dtype=img.dtype, device=img.device)], dim=0)

                pad_w = max_w - w
                pad_h = max_h - h
                if pad_w or pad_h:
                    img = NN_F.pad(img, (0, pad_w, 0, pad_h), value=0)
                # handle missing masks (create zero mask)
                if m is None:
                    m = torch.zeros((1, h, w), dtype=torch.float32, device=img.device)
                else:
                    mc, mh, mw = m.shape
                    if (mh != max_h) or (mw != max_w):
                        m = NN_F.pad(m, (0, pad_w, 0, pad_h), value=0)

                padded_images.append(img)
                padded_masks.append(m)

            images = padded_images
            masks = padded_masks

            if collect_high_pass and high_passes:
                padded_high: List[torch.Tensor] = []
                for hp in high_passes:
                    hc, hh, hw = hp.shape
                    ph = max_h - hh
                    pw = max_w - hw
                    if ph or pw:
                        hp = NN_F.pad(hp, (0, pw, 0, ph), value=0)
                    padded_high.append(hp)
                high_passes = padded_high

        batch_dict = {
            "images": torch.stack(images, dim=0),
            "masks": torch.stack(masks, dim=0),
            "labels": torch.stack(labels, dim=0),
            "datasets": datasets,
        }

        if collect_high_pass and high_passes:
            batch_dict["high_pass"] = torch.stack(high_passes, dim=0)

        return batch_dict

    return _collate


def _group_by(split: str, samples: Iterable[SampleRecord]) -> Dict[str, list[SampleRecord]]:
    grouped: Dict[str, list[SampleRecord]] = {}
    for record in samples:
        if record.split != split:
            continue
        grouped.setdefault(record.dataset, []).append(record)
    return grouped


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return Manifest.from_dataframe(df)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Manifest.from_dict(data)


def create_dataloaders(
    manifest_path: str | Path,
    per_dataset_augmentations: Dict[str, AugmentationConfig],
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    pin_memory: bool = True,
    round_robin_seed: int | None = 0,
    balance_sampling: bool = False,
    drop_last: bool = True,
    aug_seed: int | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    balance_real_fake: bool = True,
    balanced_positive_ratio: float = 0.5,
    balanced_sampler_seed: int | None = 42,
    balanced_sampler_num_samples: int | None = None,
) -> Dict[str, DataLoader]:
    manifest = load_manifest(manifest_path)
    normalization_mode = manifest.normalization_mode

    splits = {
        split: _group_by(split, manifest.samples)
        for split in ("train", "val", "test")
    }

    loaders: Dict[str, DataLoader] = {}

    for split_name, per_dataset_records in splits.items():
        training = split_name == "train"
        datasets: List[PerDatasetDataset] = []
        for dataset_name, records in per_dataset_records.items():
            aug_cfg = per_dataset_augmentations.get(dataset_name, AugmentationConfig(enable=False))
            datasets.append(PerDatasetDataset(records, aug_cfg=aug_cfg, training=training))

        if not datasets:
            continue

        combined = CombinedDataset(datasets)

        if training:
            if balance_real_fake:
                sampler = RoundRobinBalancedClassSampler(
                    datasets,
                    pos_ratio=balanced_positive_ratio,
                    shuffle=True,
                    seed=balanced_sampler_seed if balanced_sampler_seed is not None else round_robin_seed,
                    balance=balance_sampling,
                )
            else:
                sampler = RoundRobinSampler(
                    datasets,
                    shuffle=True,
                    seed=round_robin_seed,
                    balance=balance_sampling,
                )
        else:
            sampler = None

        collate_fn = _collate_builder(
            per_dataset_augmentations,
            normalization_mode,
            training=training,
            aug_seed=aug_seed,
        )

        pf = prefetch_factor if num_workers > 0 else None
        persistent = persistent_workers if num_workers > 0 else False

        loader = DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=drop_last if training else False,
            prefetch_factor=pf,
            persistent_workers=persistent,
        )
        loaders[split_name] = loader

    return loaders

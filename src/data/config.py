"""Configuration dataclasses for dataset preparation and loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass
class DatasetStructureConfig:
    dataset_root: str
    dataset_name: str
    real_subdir: str
    fake_subdir: str
    mask_subdir: str
    mask_suffix: str
    prepared_root: str
    edge_mask_subdir: str | None = None
    edge_mask_suffix: str | None = None
    sample_limit: int = 0  # 0 means use all

    def root(self) -> Path:
        return Path(self.dataset_root) / self.dataset_name

    def prepared_dir(self) -> Path:
        return Path(self.prepared_root) / self.dataset_name


@dataclass
class SplitConfig:
    train: float
    val: float
    test: float
    seed: int = 0

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


@dataclass
class PreparationConfig:
    target_sizes: Sequence[int] = (384,)
    normalization_mode: str = "imagenet"
    tar_shard_size: int = 0  # 0 disables tar sharding; otherwise samples per shard
    enable_high_pass: bool = True

    def target_size_set(self) -> set[int]:
        return {int(s) for s in self.target_sizes}


@dataclass
class AugmentationConfig:
    """Augmentation config for NGIML dataloader.

    Forensic motivation: Applies lightweight augmentations to improve
    generalization and robustness to manipulation
    scale/quality.
    """
    enable: bool = True
    views_per_sample: int = 1
    enable_flips: bool = True
    enable_rotations: bool = True
    max_rotation_degrees: float = 5.0
    enable_random_crop: bool = True
    crop_scale_range: Sequence[float] = (0.9, 1.0)
    object_crop_bias_prob: float = 0.7
    min_fg_pixels_for_object_crop: int = 16
    enable_elastic: bool = True
    elastic_prob: float = 0.3
    elastic_alpha: float = 8.0
    elastic_sigma: float = 5.0
    enable_color_jitter: bool = True
    color_jitter_factors: Sequence[float] = (0.9, 1.1)
    brightness_jitter_factors: Sequence[float] = (0.9, 1.1)
    contrast_jitter_factors: Sequence[float] = (0.9, 1.1)
    enable_noise: bool = True
    noise_std_range: Sequence[float] = (0.0, 0.0)


@dataclass
class SampleRecord:
    dataset: str
    split: str
    image_path: str
    mask_path: str | None
    label: int
    high_pass_path: str | None = None
    edge_mask_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = {
            "dataset": self.dataset,
            "split": self.split,
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "label": self.label,
        }
        if self.high_pass_path:
            data["high_pass_path"] = self.high_pass_path
        if self.edge_mask_path:
            data["edge_mask_path"] = self.edge_mask_path
        return data

    @staticmethod
    def from_dict(data: dict[str, object]) -> "SampleRecord":
        return SampleRecord(
            dataset=str(data["dataset"]),
            split=str(data["split"]),
            image_path=str(data["image_path"]),
            mask_path=str(data.get("mask_path")) if data.get("mask_path") is not None else None,
            label=int(data["label"]),
            high_pass_path=str(data["high_pass_path"]) if "high_pass_path" in data and data.get("high_pass_path") else None,
            edge_mask_path=str(data["edge_mask_path"]) if "edge_mask_path" in data and data.get("edge_mask_path") else None,
        )


@dataclass
class Manifest:
    samples: list[SampleRecord]
    normalization_mode: str = "zero_one"

    def to_dict(self) -> dict[str, object]:
        return {
            "normalization_mode": self.normalization_mode,
            "samples": [s.to_dict() for s in self.samples],
        }

    def to_dataframe(self) -> "pd.DataFrame":  # quoted for forward reference
        records = []
        for s in self.samples:
            row = s.to_dict()
            row["normalization_mode"] = self.normalization_mode
            records.append(row)
        return pd.DataFrame(records)

    @staticmethod
    def from_dict(data: dict[str, object]) -> "Manifest":
        samples = [SampleRecord.from_dict(s) for s in data.get("samples", [])]
        normalization_mode = str(data.get("normalization_mode", "zero_one"))
        return Manifest(samples=samples, normalization_mode=normalization_mode)

    @staticmethod
    def from_dataframe(df: "pd.DataFrame") -> "Manifest":
        if df.empty:
            return Manifest(samples=[], normalization_mode="zero_one")
        norm_mode = "zero_one"
        if "normalization_mode" in df.columns:
            norm_mode = str(df["normalization_mode"].iloc[0])
        samples = [
            SampleRecord(
                dataset=str(row.dataset),
                split=str(row.split),
                image_path=str(row.image_path),
                mask_path=str(row.mask_path) if pd.notna(row.mask_path) else None,
                label=int(row.label),
                high_pass_path=str(row.high_pass_path) if hasattr(row, "high_pass_path") and pd.notna(row.high_pass_path) else None,
                edge_mask_path=str(row.edge_mask_path) if hasattr(row, "edge_mask_path") and pd.notna(row.edge_mask_path) else None,
            )
            for row in df.itertuples(index=False)
        ]
        return Manifest(samples=samples, normalization_mode=norm_mode)

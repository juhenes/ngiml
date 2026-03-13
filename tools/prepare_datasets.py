"""Prepare datasets into a common manifest with optional resizing."""
from __future__ import annotations

import argparse
import io
import random
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:  # tqdm makes progress clearer; fall back to no-op if missing
    from tqdm import tqdm
except ImportError:  # pragma: no cover - lightweight fallback
    def tqdm(iterable: Iterable | None = None, total: int | None = None, desc: str | None = None, **_: object):
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow running as a script without installing the package

from src.data.config import DatasetStructureConfig, Manifest, PreparationConfig, SampleRecord, SplitConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _compute_high_pass(image_np: np.ndarray) -> np.ndarray:
    """Compute a deterministic per-channel high-pass map (uint8 HWC)."""
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape {image_np.shape}")

    image_f = image_np.astype(np.float32) / 255.0
    padded = np.pad(image_f, ((1, 1), (1, 1), (0, 0)), mode="reflect")

    center = padded[1:-1, 1:-1, :]
    top = padded[:-2, 1:-1, :]
    bottom = padded[2:, 1:-1, :]
    left = padded[1:-1, :-2, :]
    right = padded[1:-1, 2:, :]

    hp = np.abs(4.0 * center - top - bottom - left - right)
    scale = np.percentile(hp, 99.5)
    if scale <= 1e-6:
        return np.zeros_like(image_np, dtype=np.uint8)

    hp_uint8 = np.clip((hp / scale) * 255.0, 0.0, 255.0).astype(np.uint8)
    return hp_uint8


def _compute_edge_mask(mask_np: np.ndarray) -> np.ndarray:
    """Compute a thin binary edge map from a single-channel mask."""
    if mask_np.ndim != 2:
        raise ValueError(f"Expected grayscale mask HxW, got shape {mask_np.shape}")

    mask_bin = (mask_np > 127).astype(np.uint8, copy=False)
    if not np.any(mask_bin):
        return np.zeros_like(mask_np, dtype=np.uint8)

    padded = np.pad(mask_bin, ((1, 1), (1, 1)), mode="edge")
    center = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[:-2, :-2],
        padded[:-2, 2:],
        padded[2:, :-2],
        padded[2:, 2:],
    )
    eroded = center.copy()
    for neighbor in neighbors:
        np.bitwise_and(eroded, neighbor, out=eroded)
    edge = np.bitwise_and(center, np.uint8(1) - eroded)
    return edge * np.uint8(255)


class TarShardWriter:
    """Utility to write NPZ payloads into sequential tar shards."""

    def __init__(self, out_root: Path, shard_size: int) -> None:
        self.out_root = out_root
        self.shard_size = max(1, shard_size)
        self.shard_idx = 0
        self.current: tarfile.TarFile | None = None
        self.current_path: Path | None = None
        self.count_in_shard = 0

    def _start_new_shard(self) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        tar_path = self.out_root / f"shard_{self.shard_idx:05d}.tar"
        self.shard_idx += 1
        self.count_in_shard = 0
        if self.current is not None:
            self.current.close()
        self.current = tarfile.open(tar_path, mode="w")
        self.current_path = tar_path

    def add(self, payload_bytes: bytes, member_name: str) -> tuple[str, str]:
        if self.current is None or self.count_in_shard >= self.shard_size:
            self._start_new_shard()
        assert self.current is not None and self.current_path is not None
        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload_bytes)
        info.mtime = time.time()
        self.current.addfile(info, io.BytesIO(payload_bytes))
        self.count_in_shard += 1
        return str(self.current_path), member_name

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
            self.current_path = None


def _discover_images(directory: Path) -> List[Path]:
    return sorted(
        [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def _find_mask(fake_path: Path, mask_dir: Path, mask_suffix: str) -> Path | None:
    candidates = []
    stem = fake_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidates.append(mask_dir / f"{stem}{mask_suffix}{ext}")
        candidates.append(mask_dir / f"{stem}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _split_records(records: Sequence[SampleRecord], split_cfg: SplitConfig) -> Dict[str, List[SampleRecord]]:
    split_cfg.validate()
    rng = random.Random(split_cfg.seed)
    per_label: Dict[int, List[SampleRecord]] = {0: [], 1: []}
    for rec in records:
        per_label.setdefault(rec.label, []).append(rec)

    splits = {"train": [], "val": [], "test": []}
    for label, items in per_label.items():
        items = items.copy()
        rng.shuffle(items)
        n = len(items)
        ratios = [split_cfg.train, split_cfg.val, split_cfg.test]
        raw_counts = [n * r for r in ratios]
        counts = [int(c) for c in raw_counts]
        remainder = n - sum(counts)

        if remainder > 0:
            # Distribute leftover samples by largest fractional remainder,
            # but never allocate to a split whose target ratio is zero.
            order = sorted(
                range(3),
                key=lambda idx: (raw_counts[idx] - counts[idx], ratios[idx]),
                reverse=True,
            )
            for idx in order:
                if remainder == 0:
                    break
                if ratios[idx] <= 0:
                    continue
                counts[idx] += 1
                remainder -= 1

        if remainder > 0:
            counts[0] += remainder

        n_train, n_val, _ = counts
        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]
        splits["train"].extend(train_items)
        splits["val"].extend(val_items)
        splits["test"].extend(test_items)
    return splits


def _build_npz_bytes(
    image_path: Path,
    mask_path: Path | None,
    edge_mask_path: Path | None,
    target_size: int,
    include_high_pass: bool = True,
    compute_edge_mask: bool = False,
) -> bytes:
    image = Image.open(image_path).convert("RGB")
    if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
        buf_jpg = io.BytesIO()
        image.save(buf_jpg, format="JPEG", quality=95)
        buf_jpg.seek(0)
        image = Image.open(buf_jpg).convert("RGB")

    mask_img = Image.open(mask_path).convert("L") if mask_path is not None else None
    edge_mask_img = Image.open(edge_mask_path).convert("L") if edge_mask_path is not None else None

    if target_size > 0:
        image = image.resize((target_size, target_size), Image.BILINEAR)
        if mask_img is not None:
            mask_img = mask_img.resize((target_size, target_size), Image.NEAREST)
        if edge_mask_img is not None:
            edge_mask_img = edge_mask_img.resize((target_size, target_size), Image.NEAREST)

    image_np = np.asarray(image, dtype=np.uint8)
    payload = {"image": image_np}
    if mask_img is not None:
        mask_np = np.asarray(mask_img, dtype=np.uint8)
        payload["mask"] = mask_np
    else:
        mask_np = None
    if edge_mask_img is None and compute_edge_mask and mask_np is not None:
        edge_mask_np = _compute_edge_mask(mask_np)
    elif edge_mask_img is not None:
        edge_mask_np = np.asarray(edge_mask_img, dtype=np.uint8)
    else:
        edge_mask_np = None
    if edge_mask_np is not None:
        payload["edge_mask"] = edge_mask_np
    if include_high_pass:
        payload["high_pass"] = _compute_high_pass(image_np)

    buf = io.BytesIO()
    # np.savez (not compressed) to avoid CPU overhead from compression.
    np.savez(buf, **payload)
    return buf.getvalue()


def prepare_single_dataset(
    cfg: DatasetStructureConfig,
    split_cfg: SplitConfig,
    prep_cfg: PreparationConfig,
) -> List[SampleRecord]:
    root = cfg.root()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root missing: {root}")

    real_dir = root / cfg.real_subdir
    fake_dir = root / cfg.fake_subdir
    mask_dir = root / cfg.mask_subdir
    edge_mask_dir = root / cfg.edge_mask_subdir if cfg.edge_mask_subdir else mask_dir

    real_images = _discover_images(real_dir) if real_dir.exists() else []
    fake_images = _discover_images(fake_dir) if fake_dir.exists() else []

    records: List[SampleRecord] = []

    for real_img in tqdm(real_images, desc=f"{cfg.dataset_name} real", leave=False):
        records.append(
            SampleRecord(
                dataset=cfg.dataset_name,
                split="train",  # placeholder, real split decided later
                image_path=str(real_img),
                mask_path=None,
                label=0,
            )
        )

    for fake_img in tqdm(fake_images, desc=f"{cfg.dataset_name} fake", leave=False):
        mask_path = _find_mask(fake_img, mask_dir, cfg.mask_suffix)
        if mask_path is None:
            print(f"Skipping fake image without mask: {fake_img}", file=sys.stderr)
            continue
        edge_mask_path = None
        if cfg.edge_mask_suffix:
            edge_mask_path = _find_mask(fake_img, edge_mask_dir, cfg.edge_mask_suffix)
        records.append(
            SampleRecord(
                dataset=cfg.dataset_name,
                split="train",  # placeholder
                image_path=str(fake_img),
                mask_path=str(mask_path),
                label=1,
                edge_mask_path=str(edge_mask_path) if edge_mask_path is not None else None,
            )
        )

    splits = _split_records(records, split_cfg)

    prepared_records: List[SampleRecord] = []
    target_size = sorted(prep_cfg.target_size_set())[0]
    for split_name, split_records in splits.items():
        tar_writer: TarShardWriter | None = None
        if prep_cfg.tar_shard_size > 0:
            tar_root = cfg.prepared_dir() / split_name
            tar_writer = TarShardWriter(tar_root, prep_cfg.tar_shard_size)

        for idx, rec in enumerate(tqdm(split_records, desc=f"{cfg.dataset_name} {split_name}", leave=False)):
            image_path = Path(rec.image_path)
            mask_path = Path(rec.mask_path) if rec.mask_path is not None else None
            edge_mask_path = Path(rec.edge_mask_path) if rec.edge_mask_path is not None else None
            npz_bytes = _build_npz_bytes(
                image_path=image_path,
                mask_path=mask_path,
                edge_mask_path=edge_mask_path,
                target_size=target_size,
                include_high_pass=prep_cfg.enable_high_pass,
                compute_edge_mask=(split_name == "train"),
            )

            stem = f"{cfg.dataset_name}_{split_name}_{'fake' if rec.label else 'real'}_{idx:06d}"
            if tar_writer is not None:
                tar_path, member_name = tar_writer.add(npz_bytes, member_name=stem + ".npz")
                sample_path = f"{tar_path}::{member_name}"
            else:
                out_root = cfg.prepared_dir() / split_name
                out_npz = out_root / (stem + ".npz")
                out_npz.parent.mkdir(parents=True, exist_ok=True)
                out_npz.write_bytes(npz_bytes)
                sample_path = str(out_npz)

            prepared_records.append(
                SampleRecord(
                    dataset=rec.dataset,
                    split=split_name,
                    image_path=sample_path,
                    mask_path=None,
                    label=rec.label,
                    high_pass_path=None,
                    edge_mask_path=None,
                )
            )

        if tar_writer is not None:
            tar_writer.close()
    return prepared_records


def prepare_all(
    datasets: Sequence[DatasetStructureConfig],
    per_dataset_splits: Dict[str, SplitConfig],
    prep_cfg: PreparationConfig,
    manifest_out: Path,
) -> Manifest:
    all_records: List[SampleRecord] = []
    for cfg in tqdm(datasets, desc="datasets"):
        split_cfg = per_dataset_splits.get(cfg.dataset_name)
        if split_cfg is None:
            raise ValueError(f"Missing split config for dataset {cfg.dataset_name}")
        records = prepare_single_dataset(cfg, split_cfg, prep_cfg)
        all_records.extend(records)

    manifest = Manifest(samples=all_records, normalization_mode=prep_cfg.normalization_mode)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    df = manifest.to_dataframe()
    df.to_parquet(manifest_out, index=False)
    return manifest


def build_default_configs() -> Tuple[List[DatasetStructureConfig], Dict[str, SplitConfig], PreparationConfig]:
    shared_seed = 42
    datasets = [
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="CASIA2",
            real_subdir="Au",
            fake_subdir="Tp",
            mask_subdir="Gt",
            mask_suffix="_gt",
            prepared_root="./prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="CASIA1",
            real_subdir="Au",
            fake_subdir="Tp",
            mask_subdir="Gt",
            mask_suffix="_gt",
            prepared_root="./prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="COVERAGE",
            real_subdir="real",
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="forged",
            prepared_root="./prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="Columbia",
            real_subdir="real",
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="_edgemask",
            prepared_root="./prepared",
        ),
    ]

    per_dataset_splits = {
        "CASIA2": SplitConfig(train=0.8, val=0.2, test=0.0, seed=shared_seed),
        "CASIA1": SplitConfig(train=0.0, val=0.0, test=1.0, seed=shared_seed),
        "COVERAGE": SplitConfig(train=0.0, val=0.0, test=1.0, seed=shared_seed),
        "Columbia": SplitConfig(train=0.0, val=0.0, test=1.0, seed=shared_seed),
    }

    prep_cfg = PreparationConfig(target_sizes=(384,), normalization_mode="imagenet", tar_shard_size=500)

    return datasets, per_dataset_splits, prep_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets and emit manifest.parquet")
    parser.add_argument("--manifest", type=str, default=None, help="Output manifest path. Defaults to <prepared_root>/manifest.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets, per_dataset_splits, prep_cfg = build_default_configs()

    prepared_root = Path(datasets[0].prepared_root)
    manifest_out = Path(args.manifest) if args.manifest else prepared_root / "manifest.parquet"

    manifest = prepare_all(datasets, per_dataset_splits, prep_cfg, manifest_out)
    print(f"Wrote manifest with {len(manifest.samples)} samples to {manifest_out}")


if __name__ == "__main__":
    main()

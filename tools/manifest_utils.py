# manifest_utils.py
"""
Helpers for manifest and path resolution for Colab and local environments.
"""
from pathlib import Path
from typing import Tuple
from src.data.dataloaders import load_manifest
import json

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
        candidates.extend([
            manifest_path.parent / path_value,
            data_root / path_value,
            data_root / "ngiml" / path_value,
            Path("/content") / path_value,
            Path("/content/data") / path_value,
            Path("/content/ngiml") / path_value,
        ])
    if "prepared/" in normalized:
        suffix = normalized.split("prepared/", 1)[1]
        candidates.extend([
            data_root / "prepared" / suffix,
            data_root / "ngiml" / "prepared" / suffix,
            Path("/content") / "prepared" / suffix,
            Path("/content/ngiml") / "prepared" / suffix,
        ])
    if "datasets/" in normalized:
        suffix = normalized.split("datasets/", 1)[1]
        candidates.extend([
            data_root / "datasets" / suffix,
            data_root / "ngiml" / "datasets" / suffix,
            Path("/content") / "datasets" / suffix,
            Path("/content/ngiml") / "datasets" / suffix,
        ])
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
    # Save as JSON (default)
    with open(resolved_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_obj.to_dict(), handle)
    print(
        f"Wrote resolved manifest to {resolved_manifest_path} "
        f"(updated fields: {rewritten}, removed missing samples: {filtered_out})"
    )
    return resolved_manifest_path

import io

import torch
import numpy as np
from PIL import Image

from src.data.config import Manifest, SampleRecord
from src.model.losses import MultiStageLossConfig, MultiStageManipulationLoss
from tools.prepare_datasets import _build_npz_bytes


def test_manifest_roundtrip_preserves_edge_mask_path():
    manifest = Manifest(
        samples=[
            SampleRecord(
                dataset="CASIA2",
                split="train",
                image_path="prepared/CASIA2/train/fake/sample_01.npz",
                mask_path=None,
                label=1,
                edge_mask_path="prepared/CASIA2/train/fake/sample_01_edge.png",
            )
        ],
        normalization_mode="imagenet",
    )

    recovered = Manifest.from_dataframe(manifest.to_dataframe())

    assert recovered.samples[0].edge_mask_path == "prepared/CASIA2/train/fake/sample_01_edge.png"


def test_boundary_loss_uses_explicit_edge_target_when_present():
    loss_fn = MultiStageManipulationLoss(
        MultiStageLossConfig(
            dice_weight=0.0,
            bce_weight=0.0,
            use_boundary_loss=True,
            boundary_weight=1.0,
        )
    )
    preds = [torch.zeros((2, 1, 8, 8), dtype=torch.float32)]
    target = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    edge_target = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    edge_target[0, :, 3:5, 3:5] = 1.0
    edge_present = torch.tensor([True, False])

    explicit_loss = loss_fn(preds, target, edge_target=edge_target, edge_target_present=edge_present)
    fallback_loss = loss_fn(preds, target)

    assert explicit_loss.item() > fallback_loss.item()


def test_prepare_dataset_does_not_serialize_edge_mask_even_when_requested(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 255
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    npz_bytes = _build_npz_bytes(
        image_path=image_path,
        mask_path=mask_path,
        edge_mask_path=None,
        target_size=8,
        include_high_pass=False,
        compute_edge_mask=True,
    )

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        assert "edge_mask" not in data


def test_boundary_loss_matches_symmetric_band_fallback_when_no_explicit_edge():
    loss_fn = MultiStageManipulationLoss(
        MultiStageLossConfig(
            dice_weight=0.0,
            bce_weight=0.0,
            use_boundary_loss=True,
            boundary_weight=1.0,
        )
    )
    target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target[:, :, 2:6, 2:6] = 1.0

    matching_logits = torch.full((1, 1, 8, 8), -8.0, dtype=torch.float32)
    matching_logits[:, :, 2:6, 2:6] = 8.0

    non_matching_logits = torch.full((1, 1, 8, 8), -8.0, dtype=torch.float32)

    matching_loss = loss_fn([matching_logits], target)
    non_matching_loss = loss_fn([non_matching_logits], target)

    assert matching_loss.item() < non_matching_loss.item()


def test_boundary_loss_runs_under_autocast_without_bce_error():
    loss_fn = MultiStageManipulationLoss(
        MultiStageLossConfig(
            dice_weight=0.0,
            bce_weight=0.0,
            use_boundary_loss=True,
            boundary_weight=1.0,
        )
    )
    target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target[:, :, 2:6, 2:6] = 1.0
    logits = torch.zeros((1, 1, 8, 8), dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = loss_fn([logits], target)

    assert torch.isfinite(loss)


def test_prepare_dataset_ignores_explicit_edge_mask_input(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    edge_mask_path = tmp_path / "edge_mask.png"

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    edge_mask = np.zeros((8, 8), dtype=np.uint8)
    edge_mask[1:7, 1:7] = 255

    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(edge_mask).save(edge_mask_path)

    npz_bytes = _build_npz_bytes(
        image_path=image_path,
        mask_path=mask_path,
        edge_mask_path=edge_mask_path,
        target_size=8,
        include_high_pass=False,
        compute_edge_mask=True,
    )

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        assert "edge_mask" not in data


def test_prepare_dataset_roundtrips_non_jpeg_images_through_jpeg(tmp_path):
    image_path = tmp_path / "image.png"

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = np.arange(8, dtype=np.uint8) * 31
    image[..., 1] = np.arange(8, dtype=np.uint8).reshape(8, 1) * 17
    image[..., 2] = 90
    Image.fromarray(image).save(image_path)

    npz_bytes = _build_npz_bytes(
        image_path=image_path,
        mask_path=None,
        edge_mask_path=None,
        target_size=8,
        include_high_pass=False,
        compute_edge_mask=False,
    )

    expected_buffer = io.BytesIO()
    Image.open(image_path).convert("RGB").save(expected_buffer, format="JPEG", quality=95, subsampling=0)
    expected_buffer.seek(0)
    expected_image = np.asarray(Image.open(expected_buffer).convert("RGB"), dtype=np.uint8)

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        assert np.array_equal(data["image"], expected_image)
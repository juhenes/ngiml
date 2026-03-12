import numpy as np
import torch

from src.data.config import SampleRecord
from src.data.dataloaders import _load_from_npz
from tools.infer_helpers import load_image_mask_from_record


def test_npz_uint8_scaled_to_unit_float32(tmp_path):
    npz_path = tmp_path / "sample_uint8.npz"
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)
    high_pass = np.full((8, 8, 3), 128, dtype=np.uint8)
    np.savez(npz_path, image=image, mask=mask, high_pass=high_pass)

    out_image, out_mask, out_high_pass, out_edge_mask = _load_from_npz(str(npz_path))

    assert out_image.dtype == torch.float32
    assert out_high_pass is not None
    assert out_high_pass.dtype == torch.float32
    assert out_mask is not None
    assert out_mask.dtype == torch.float32
    assert out_edge_mask is None
    assert torch.isclose(out_image.max(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(out_high_pass.max(), torch.tensor(128.0 / 255.0), atol=1e-5)


def test_npz_float_unit_range_not_double_scaled(tmp_path):
    npz_path = tmp_path / "sample_float.npz"
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    high_pass = np.full((8, 8, 3), 0.4, dtype=np.float32)
    np.savez(npz_path, image=image, mask=mask, high_pass=high_pass)

    out_image, out_mask, out_high_pass, out_edge_mask = _load_from_npz(str(npz_path))

    assert out_image.dtype == torch.float32
    assert out_high_pass is not None
    assert out_high_pass.dtype == torch.float32
    assert out_mask is not None
    assert out_mask.dtype == torch.float32
    assert out_edge_mask is None
    assert torch.isclose(out_image.mean(), torch.tensor(0.5), atol=1e-6)
    assert torch.isclose(out_high_pass.mean(), torch.tensor(0.4), atol=1e-6)


def test_npz_optional_edge_mask_is_loaded(tmp_path):
    npz_path = tmp_path / "sample_edge.npz"
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)
    edge_mask = np.triu(np.ones((8, 8), dtype=np.uint8)) * 255
    np.savez(npz_path, image=image, mask=mask, edge_mask=edge_mask)

    out_image, out_mask, out_high_pass, out_edge_mask = _load_from_npz(str(npz_path))

    assert out_image.dtype == torch.float32
    assert out_mask is not None
    assert out_high_pass is not None
    assert out_edge_mask is not None
    assert out_edge_mask.dtype == torch.float32
    assert out_edge_mask.shape == (1, 8, 8)
    assert torch.isclose(out_edge_mask.max(), torch.tensor(1.0), atol=1e-6)


def test_infer_helper_handles_npz_records_with_edge_mask(tmp_path):
    npz_path = tmp_path / "sample_edge_record.npz"
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)
    high_pass = np.full((8, 8, 3), 64, dtype=np.uint8)
    edge_mask = np.triu(np.ones((8, 8), dtype=np.uint8)) * 255
    np.savez(npz_path, image=image, mask=mask, high_pass=high_pass, edge_mask=edge_mask)

    record = SampleRecord(
        dataset="CASIA2",
        split="test",
        image_path=str(npz_path),
        mask_path=None,
        label=1,
    )

    out_image, out_mask, out_high_pass = load_image_mask_from_record(record)

    assert out_image.shape == (3, 8, 8)
    assert out_mask.shape == (1, 8, 8)
    assert out_high_pass is not None
    assert out_high_pass.shape == (3, 8, 8)
    assert torch.isclose(out_mask.max(), torch.tensor(1.0), atol=1e-6)

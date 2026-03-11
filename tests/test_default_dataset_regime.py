import torch

from src.data.config import AugmentationConfig
from src.data.dataloaders import _apply_gpu_augmentations_batch
from tools.colab_train_helpers import build_default_components, build_training_config
from tools.prepare_datasets import build_default_configs
from tools.train_ngiml import _stack_padded_tensors


def test_prepare_default_configs_use_casia2_for_training_and_casia1_coverage_columbia_for_test():
    datasets, per_dataset_splits, prep_cfg = build_default_configs()

    assert [dataset.dataset_name for dataset in datasets] == ["CASIA2", "CASIA1", "COVERAGE", "Columbia"]
    assert prep_cfg.target_sizes == (384,)

    casia2_split = per_dataset_splits["CASIA2"]
    assert casia2_split.train == 0.8
    assert casia2_split.val == 0.2
    assert casia2_split.test == 0.0

    casia1_split = per_dataset_splits["CASIA1"]
    assert casia1_split.train == 0.0
    assert casia1_split.val == 0.0
    assert casia1_split.test == 1.0

    coverage_split = per_dataset_splits["COVERAGE"]
    assert coverage_split.train == 0.0
    assert coverage_split.val == 0.0
    assert coverage_split.test == 1.0

    columbia_split = per_dataset_splits["Columbia"]
    assert columbia_split.train == 0.0
    assert columbia_split.val == 0.0
    assert columbia_split.test == 1.0


def test_default_components_use_shared_augmentation_defaults():
    model_cfg, _loss_cfg, default_aug, per_dataset_aug = build_default_components()

    assert default_aug.enable is True
    assert default_aug.views_per_sample == 2
    assert default_aug.max_rotation_degrees == 10.0
    assert default_aug.crop_scale_range == (0.75, 1.0)
    assert default_aug.noise_std_range == (0.0, 0.012)
    assert default_aug.multiscale_training is True
    assert default_aug.multiscale_short_side_range == (384, 576)
    assert default_aug.enable_blur is True
    assert default_aug.blur_prob == 0.2
    assert default_aug.enable_rescale is True
    assert default_aug.rescale_prob == 0.35
    assert default_aug.enable_compression is True
    assert default_aug.compression_quality_range == (35, 85)
    assert model_cfg.residual.base_channels == 24
    assert tuple(model_cfg.fusion.fusion_channels) == (48, 96, 144, 192)
    assert per_dataset_aug == {}


def test_batched_gpu_augmentations_apply_multiscale_resize():
    images = torch.rand(2, 3, 384, 512)
    masks = torch.zeros(2, 1, 384, 512)
    cfg = AugmentationConfig(
        enable=True,
        enable_flips=False,
        enable_rotations=False,
        enable_random_crop=False,
        enable_elastic=False,
        enable_color_jitter=False,
        enable_noise=False,
        multiscale_training=True,
        multiscale_short_side_range=(448, 448),
    )
    generator = torch.Generator().manual_seed(123)

    out_images, out_masks, out_high_pass, out_edge_masks = _apply_gpu_augmentations_batch(
        images,
        masks,
        cfg,
        generator=generator,
    )

    assert out_images.shape == (2, 3, 448, 597)
    assert out_masks.shape == (2, 1, 448, 597)
    assert out_high_pass is None
    assert out_edge_masks is None


def test_colab_training_config_defaults_batch_size_to_20(tmp_path):
    model_cfg, loss_cfg, default_aug, per_dataset_aug = build_default_components()
    training_config = build_training_config(
        manifest_path=tmp_path / "manifest.json",
        output_dir=str(tmp_path / "runs"),
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        default_aug=default_aug,
        per_dataset_aug=per_dataset_aug,
    )

    assert training_config["batch_size"] == 20
    assert training_config["auto_phase2_enabled"] is False
    assert training_config["hard_mining_enabled"] is True
    assert training_config["hard_mining_start_epoch"] == 2
    assert training_config["hard_mining_weight"] == 0.05
    assert training_config["hard_mining_gamma"] == 2.5


def test_batched_gpu_augmentations_apply_forensic_degradations():
    images = torch.rand(2, 3, 64, 64)
    original = images.clone()
    masks = torch.zeros(2, 1, 64, 64)
    cfg = AugmentationConfig(
        enable=True,
        enable_flips=False,
        enable_rotations=False,
        enable_random_crop=False,
        enable_elastic=False,
        enable_color_jitter=False,
        enable_noise=False,
        enable_blur=True,
        blur_prob=1.0,
        blur_kernel_range=(5, 5),
        enable_rescale=True,
        rescale_prob=1.0,
        rescale_factor_range=(0.65, 0.65),
        enable_compression=True,
        compression_prob=1.0,
        compression_quality_range=(35, 35),
    )

    out_images, out_masks, out_high_pass, out_edge_masks = _apply_gpu_augmentations_batch(
        images,
        masks,
        cfg,
        generator=torch.Generator().manual_seed(321),
    )

    assert out_images.shape == original.shape
    assert out_masks.shape == masks.shape
    assert out_high_pass is None
    assert out_edge_masks is None
    assert not torch.allclose(out_images, original)


def test_stack_padded_tensors_aligns_spatial_shapes():
    first = torch.ones(3, 384, 384)
    second = torch.zeros(3, 472, 472)

    stacked = _stack_padded_tensors([first, second])

    assert stacked.shape == (2, 3, 472, 472)
    assert torch.allclose(stacked[1], second)
    assert torch.allclose(stacked[0, :, :384, :384], first)
    assert torch.count_nonzero(stacked[0, :, 384:, :]) == 0
    assert torch.count_nonzero(stacked[0, :, :, 384:]) == 0
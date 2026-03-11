from tools.colab_train_helpers import build_default_components
from tools.prepare_datasets import build_default_configs


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
    _model_cfg, _loss_cfg, default_aug, per_dataset_aug = build_default_components()

    assert default_aug.enable is True
    assert default_aug.views_per_sample == 2
    assert default_aug.max_rotation_degrees == 6.0
    assert default_aug.crop_scale_range == (0.75, 1.0)
    assert default_aug.noise_std_range == (0.0, 0.012)
    assert default_aug.multiscale_training is False
    assert default_aug.multiscale_short_side_range == (384, 576)
    assert per_dataset_aug == {}
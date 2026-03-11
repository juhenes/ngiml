import sys

from tools.train_ngiml import TrainConfig, parse_args


def test_balance_real_fake_cli_default_false(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ngiml.py", "--manifest", "dummy_manifest.json"])
    cfg = parse_args()
    assert cfg.balance_real_fake is True


def test_batch_size_default_matches_train_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ngiml.py", "--manifest", "dummy_manifest.json"])
    cfg = parse_args()
    assert cfg.batch_size == 20
    assert TrainConfig(manifest="dummy_manifest.json").batch_size == 20


def test_scheduler_type_cli_default_and_mapping(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ngiml.py", "--manifest", "dummy_manifest.json"])
    cfg = parse_args()
    assert cfg.scheduler_type == "cosine"

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_ngiml.py", "--manifest", "dummy_manifest.json", "--scheduler-type", "step"],
    )
    cfg_step = parse_args()
    assert cfg_step.scheduler_type == "step"


def test_balance_real_fake_defaults_consistent():
    cfg = TrainConfig(manifest="dummy_manifest.json")
    assert cfg.balance_real_fake is True
    assert cfg.balanced_positive_ratio == 0.6


def test_loss_defaults_are_stable_overlap_core(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ngiml.py", "--manifest", "dummy_manifest.json"])
    cfg = parse_args()
    assert cfg.tversky_weight == 0.0
    assert cfg.tversky_beta == 0.8
    assert cfg.lovasz_weight == 0.0
    assert cfg.use_boundary_loss is False
    assert cfg.boundary_weight == 0.05


def test_overlap_focused_threshold_and_mining_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ngiml.py", "--manifest", "dummy_manifest.json"])
    cfg = parse_args()
    assert cfg.training_phase == "phase1"
    assert cfg.auto_phase2_enabled is False
    assert cfg.auto_phase2_patience == 5
    assert cfg.auto_phase2_lr_scale == 0.33
    assert cfg.auto_phase2_tversky_weight == 0.1
    assert cfg.auto_phase2_monitor == "iou"
    assert cfg.threshold_metric == "f1"
    assert cfg.threshold_start == 0.2
    assert cfg.threshold_end == 0.8
    assert cfg.threshold_step == 0.02
    assert cfg.pos_weight_max == 10.0
    assert cfg.hard_mining_enabled is False
    assert cfg.hard_mining_start_epoch == 5
    assert cfg.hard_mining_weight == 0.03

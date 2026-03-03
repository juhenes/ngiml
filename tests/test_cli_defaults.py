import sys

from tools.train_ngiml import parse_args


def test_balance_real_fake_cli_default_false(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ngiml.py", "--manifest", "dummy_manifest.json"])
    cfg = parse_args()
    assert cfg.balance_real_fake is False

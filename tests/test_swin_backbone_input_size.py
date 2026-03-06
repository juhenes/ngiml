from types import SimpleNamespace

from src.model.backbones.swin_backbone import SwinBackbone, SwinBackboneConfig


def test_expected_input_size_prefers_configured_model_size_over_default_cfg():
    backbone = object.__new__(SwinBackbone)
    backbone.config = SwinBackboneConfig(input_size=320)
    backbone.model = SimpleNamespace(
        img_size=(320, 320),
        default_cfg={"input_size": (3, 224, 224)},
    )
    backbone.patch_embed = SimpleNamespace(img_size=(320, 320), strict_img_size=True)

    assert backbone._expected_input_size() == (320, 320)


def test_expected_input_size_falls_back_to_default_cfg_when_needed():
    backbone = object.__new__(SwinBackbone)
    backbone.config = SwinBackboneConfig(input_size=None)
    backbone.model = SimpleNamespace(
        img_size=None,
        default_cfg={"input_size": (3, 224, 224)},
    )
    backbone.patch_embed = SimpleNamespace(img_size=None, strict_img_size=False)

    assert backbone._expected_input_size() == (224, 224)
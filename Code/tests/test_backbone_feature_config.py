"""Pure feature resolution supports SimAM and rejects retired or ignored options."""
import copy
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve(config=None, **kwargs):
    from detector.core.backbone_config import resolve_backbone_features
    return resolve_backbone_features(config, **kwargs)


@pytest.mark.parametrize("stage,stride,channels", [("c3", 4, 48), ("c4", 8, 96), ("c5", 16, 128)])
def test_simam_stage_defaults_serialization_and_input_immutability(stage, stride, channels):
    config = {"local_attention": "simam", "local_attention_stage": stage.upper()}
    before = copy.deepcopy(config)
    result = resolve(config, grid_shape=(800, 704))
    assert config == before
    assert result.local_attention_stage == stage
    assert result.local_attention_simam_lambda == .0001
    assert result.local_attention_layer_scale_init == .001
    assert result.effective_placement() == {"local_attention": {
        "stage": stage, "stride": stride, "channels": channels}}
    assert result.has_legacy_placement == (stage == "c3")
    assert resolve(result.to_dict()) == result
    assert len(result.semantic_hash) == 64
    json.dumps(result.semantic_metadata(), allow_nan=False)


def test_missing_settings_are_immutable_and_explicit_defaults_are_canonical():
    result = resolve({})
    assert result.c4_attention == result.c4_attention_qk_norm == result.local_attention == "none"
    assert result.c4_attention_scales == ()
    assert result.neck_fusion_channels == 24 and result.detail_path is False
    with pytest.raises(FrozenInstanceError):
        result.detail_path = True
    implicit = resolve({"local_attention": "simam"})
    explicit = resolve({"local_attention": "simam", "local_attention_stage": "c3"})
    assert implicit.semantic_hash == explicit.semantic_hash


def test_resolver_does_not_import_numeric_frameworks():
    subprocess.run([sys.executable, "-c", "from detector.core.backbone_config import "
                    "resolve_backbone_features; import sys; resolve_backbone_features({}); "
                    "assert not {'torch', 'numpy', 'numba'} & set(sys.modules)"], cwd=ROOT, check=True)


@pytest.mark.parametrize("config", [
    {"c4_context": "other"}, {"c4_context": None},
    {"local_attention": ["simam"]}, {"local_attention": "other"},
    {"local_attention_stage": "c4"}, {"local_attention_layer_scale_init": .001},
    {"local_attention": "simam", "local_attention_stage": False},
    {"local_attention": "simam", "local_attention_stage": "stem"},
    {"c4_attention": "litemla"}, {"c4_attention_scales": [5]},
    {"c4_attention_scales": None}, {"c4_attention_qk_norm": "rmsnorm"},
    {"c4_context_typo": 1}, {"local_attention_typo": 1}, {"focal_stage_typo": "c4"},
    {"attention_orde": "focal_then_local"},
    {"neck_fusion_channels": True}, {"neck_fusion_channels": 24.0},
    {"neck_fusion_channels": 16}, {"detail_path": 1}, {"detail_path": "false"},
    {"backbone": "mobilepixor", "local_attention": "simam"},
    {"backbone": "rpn", "detail_path": False},
    {"neck_type": "rc_sgfpn", "neck_fusion_channels": 32},
    {"neck_type": "rc_bisgfpn", "detail_path": True}, {"neck_type": "typo"},
])
def test_invalid_inactive_and_wrong_backbone_options_reject(config):
    with pytest.raises(ValueError):
        resolve(config)


@pytest.mark.parametrize("field,values", [
    ("local_attention_simam_lambda", [0, -1, True, "0.001", float("inf"), float("nan"), 10**400]),
    ("local_attention_layer_scale_init", [-1, True, float("nan"), 10**400]),
])
def test_local_numeric_validation(field, values):
    for value in values:
        with pytest.raises(ValueError, match=field):
            resolve({"local_attention": "simam", field: value})


def test_variant_settings_change_semantic_identity():
    variants = [{}, {"local_attention": "simam"},
                {"local_attention": "simam", "local_attention_stage": "c4"},
                {"local_attention": "simam", "local_attention_simam_lambda": .002},
                {"neck_fusion_channels": 32}, {"detail_path": True}]
    assert len({resolve(config).semantic_hash for config in variants}) == len(variants)


@pytest.mark.parametrize("shape", [(800, 703), (15, 32), (0, 32), (True, 32), (32.0, 32), (32,)])
def test_candidate_requires_divisible_by_16_grid(shape):
    with pytest.raises(ValueError, match="grid_shape"):
        resolve({"local_attention": "simam"}, grid_shape=shape)


def test_geometry_alignment_and_legacy_backbones():
    geometry = {"x_min": 0, "x_max": 70.3, "x_res": .1,
                "y_min": -40, "y_max": 40, "y_res": .1}
    resolve({}, geometry=geometry)
    with pytest.raises(ValueError, match="grid_shape"):
        resolve({"local_attention": "simam"}, geometry=geometry)
    resolve({"local_attention": "simam"}, geometry={**geometry, "x_max": 70.4})
    with pytest.raises(ValueError, match="grid_shape"):
        resolve({"stage_depths": [3, 4, 2]}, grid_shape=(33, 32))
    assert resolve({"backbone": "pixor"}).local_attention == "none"
    assert resolve({"neck_type": "rc_sgfpn"}).neck_fusion_channels == 24


def test_default_config_uses_only_disabled_attention_settings():
    config = json.loads((ROOT / "configs/config.json").read_text())["model"]
    result = resolve(config)
    assert result.c4_attention == "none" and result.c4_attention_scales == ()


@pytest.mark.parametrize("config", [
    {"c4_context": "focal"}, {"local_attention": "eca"},
    {"c4_context_version": 1}, {"c4_context_bottleneck": 64},
    {"c4_context_dilations": [1, 2, 3]}, {"c4_context_layer_scale_init": .001},
    {"focal_stage": "c4"}, {"attention_order": "focal_then_local"},
    {"local_attention_eca_kernel_size": 3},
    {"local_attention": "simam", "local_attention_eca_kernel_size": 3},
])
def test_retired_feature_settings_are_rejected(config):
    with pytest.raises(ValueError, match="removed"):
        resolve(config)


def test_only_disabled_legacy_context_is_readable_and_not_reserialized():
    result = resolve({"c4_context": "none"})
    assert result == resolve({})
    assert not any("context" in key or "focal" in key or "eca" in key or
                   key == "attention_order" for key in result.to_dict())
    assert not hasattr(result, "c4_context")
    assert result.semantic_metadata()["resolver_version"] == 4
    assert result.effective_placement() == {"local_attention": None}

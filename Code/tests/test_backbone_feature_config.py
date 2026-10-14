"""Pure feature resolution rejects options that would otherwise be ignored."""

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


FOCAL = {"c4_attention": "none", "c4_context": "focal"}


@pytest.mark.parametrize('focal,local,order', [
    ('c3','c3','focal_then_local'), ('c4','c4','local_then_focal'),
    ('c5','c3',None), ('c3','c5',None), ('c4','c3',None),
])
def test_selectable_attention_placement_resolves_and_serializes(focal, local, order):
    cfg = {**FOCAL, 'local_attention':'simam', 'focal_stage':focal.upper(),
           'local_attention_stage':local.upper()}
    if order is not None:
        cfg['attention_order'] = order
    before = copy.deepcopy(cfg)
    result = resolve(cfg)
    assert cfg == before
    assert result.focal_stage == focal and result.local_attention_stage == local
    assert result.attention_order == order
    placement = result.effective_placement()
    assert placement['focal']['channels'] == {'c3':48,'c4':96,'c5':128}[focal]
    assert placement['local_attention']['stride'] == {'c3':4,'c4':8,'c5':16}[local]
    assert result.to_dict()['focal_stage'] == focal


def test_placement_defaults_and_legacy_identity_canonicalize():
    base = resolve({**FOCAL, 'local_attention':'simam'})
    explicit = resolve({**FOCAL, 'local_attention':'simam', 'focal_stage':'c4',
                        'local_attention_stage':'c3'})
    assert base.focal_stage == 'c4' and base.local_attention_stage == 'c3'
    assert base.semantic_metadata() == explicit.semantic_metadata()
    assert base.semantic_metadata()['resolver_version'] == 1
    shared = {**FOCAL, 'local_attention':'simam', 'local_attention_stage':'c4'}
    default = resolve(shared)
    assert default.attention_order == 'focal_then_local'
    assert default.semantic_metadata()['resolver_version'] == 2
    assert default.semantic_hash == resolve({**shared, 'attention_order':'focal_then_local'}).semantic_hash
    assert default.semantic_hash != resolve({**shared, 'attention_order':'local_then_focal'}).semantic_hash
    assert resolve({}).effective_placement() == {'focal':None, 'local_attention':None, 'attention_order':None}


@pytest.mark.parametrize('patch', [
    {'focal_stage':None}, {'focal_stage':True}, {'focal_stage':3}, {'focal_stage':['c3']},
    {'focal_stage':'stem'}, {'focal_stage_typo':'c3'}, {'attention_orde':'focal_then_local'},
    {'local_attention_stage':'c4'}, {'attention_order':'focal_then_local'},
    {'c4_context':'none', 'focal_stage':'c4'},
    {'local_attention':'simam','attention_order':'focal_then_local'},
    {'local_attention':'simam','local_attention_stage':'c4','attention_order':None},
    {'local_attention':'simam','local_attention_stage':'c4','attention_order':'parallel'},
    {'local_attention':'simam','local_attention_stage':False},
])
def test_invalid_placement_options_are_never_ignored(patch):
    with pytest.raises(ValueError):
        resolve({**FOCAL, **patch})


@pytest.mark.parametrize('kind', ['eca','simam'])
def test_local_adapter_stage_is_active_only_and_other_backbones_reject_it(kind):
    result = resolve({'local_attention':kind, 'local_attention_stage':'c5'})
    assert result.local_attention_stage == 'c5' and result.focal_stage is None
    assert 'focal_stage' not in result.to_dict()
    with pytest.raises(ValueError):
        resolve({'backbone':'mobilepixor', 'focal_stage':'c4'})


def test_resolver_exists_without_importing_numeric_frameworks():
    assert (ROOT / "detector/core/backbone_config.py").exists()
    subprocess.run([sys.executable, "-c", "from detector.core.backbone_config import "
                    "resolve_backbone_features; import sys; resolve_backbone_features({}); "
                    "assert not {'torch', 'numpy', 'numba'} & set(sys.modules)"],
                   cwd=ROOT, check=True)


def test_missing_fields_use_convolutional_topology_and_are_immutable():
    result = resolve({})
    assert result.c4_attention == "none"
    assert result.c4_attention_scales == ()
    assert result.c4_attention_qk_norm == "none"
    assert result.c4_context == result.local_attention == "none"
    assert result.neck_fusion_channels == 24 and result.detail_path is False
    assert result.c4_context_bottleneck is None
    with pytest.raises(FrozenInstanceError):
        result.detail_path = True


@pytest.mark.parametrize("local,extra", [
    ("none", {}), ("eca", {"local_attention_eca_kernel_size": 5}),
    ("simam", {"local_attention_simam_lambda": 0.002}),
])
def test_resolved_identity_is_complete_json_and_does_not_mutate(local, extra):
    cfg = {**FOCAL, "local_attention": local, **extra}
    before = copy.deepcopy(cfg)
    result = resolve(cfg, grid_shape=(800, 704))
    assert cfg == before
    assert result.c4_context_version == 1
    assert result.c4_context_dilations == (1, 2, 3)
    assert result.c4_context_bottleneck == 64
    assert result.c4_context_layer_scale_init == 0.001
    saved = result.to_dict()
    assert resolve(saved) == result
    metadata = result.semantic_metadata()
    assert metadata["placement"] == {"context_stride": 8, "local_attention_stride": 4}
    json.dumps(metadata, allow_nan=False)
    assert len(result.semantic_hash) == 64
    metadata["config"]["c4_context_dilations"][0] = 99
    assert result.c4_context_dilations == (1, 2, 3)


def test_invalid_focal_and_common_settings():
    invalid_settings = [
        ("c4_context", "other"), ("local_attention", ["eca", "simam"]),
        ("c4_context_version", 2), ("c4_context_version", True),
        ("c4_context_version", 1.0), ("c4_context_bottleneck", 0),
        ("c4_context_bottleneck", True), ("c4_context_bottleneck", 64.0),
        ("c4_context_dilations", []), ("c4_context_dilations", [1, 1, 3]),
        ("c4_context_dilations", [3, 2, 1]), ("c4_context_dilations", [0, 1]),
        ("c4_context_dilations", [True, 2]), ("c4_context_dilations", [1.0, 2]),
        ("c4_context_layer_scale_init", -1), ("c4_context_layer_scale_init", True),
        ("c4_context_layer_scale_init", "0.001"),
        ("c4_context_layer_scale_init", float("nan")),
        ("c4_context_layer_scale_init", float("inf")),
        ("neck_fusion_channels", 16), ("neck_fusion_channels", True),
        ("neck_fusion_channels", 24.0), ("detail_path", 1), ("detail_path", "false"),
    ]
    for field, value in invalid_settings:
        with pytest.raises(ValueError, match=field):
            resolve({**FOCAL, field: value})


def test_conflicting_or_ignored_module_fields_are_rejected():
    conflicting_configs = [
        {**FOCAL, "c4_attention_scales": [5]},
        {**FOCAL, "c4_attention_qk_norm": "rmsnorm"},
        {"c4_context_bottleneck": 64}, {"c4_context_version": 1},
        {"local_attention_eca_kernel_size": 3},
        {"local_attention_layer_scale_init": 0.001},
        {"local_attention": "eca", "local_attention_simam_lambda": 0.0001},
        {"local_attention": "simam", "local_attention_eca_kernel_size": 3},
        {"c4_attention": "other"}, {"c4_attention_scales": None},
        {"c4_attention_scales": [True]}, {"c4_attention_scales": [4]},
        {"c4_attention_scales": [5, 5]}, {"c4_attention_qk_norm": "other"},
        {"c4_context_typo": 1}, {"local_attention_typo": 1},
    ]
    for config in conflicting_configs:
        with pytest.raises(ValueError):
            resolve(config)


def test_local_numeric_validation():
    cases = [
        ("eca", "local_attention_eca_kernel_size", [0, 2, True, 3.0, "3"]),
        ("simam", "local_attention_simam_lambda", [0, -1, True, "0.001", float("inf"), float("nan")]),
        ("eca", "local_attention_layer_scale_init", [-1, True, float("nan")]),
    ]
    for local, field, values in cases:
        for value in values:
            with pytest.raises(ValueError, match=field):
                resolve({"local_attention": local, field: value})


def test_variant_settings_change_semantic_identity():
    base = resolve(FOCAL)
    for extra in [{"c4_context_bottleneck": 32}, {"c4_context_dilations": [1, 3]},
                  {"c4_context_layer_scale_init": 0}, {"local_attention": "eca"},
                  {"neck_fusion_channels": 32}, {"detail_path": True}]:
        assert resolve({**FOCAL, **extra}).semantic_hash != base.semantic_hash


def test_wrong_backbone_and_optional_neck_selection():
    wrong_configs = [
        {"backbone": "pixor", **FOCAL},
        {"backbone": "mobilepixor", "local_attention": "eca"},
        {"backbone": "rpn", "detail_path": False},
        {"neck_type": "rc_sgfpn", "neck_fusion_channels": 32},
        {"neck_type": "rc_bisgfpn", "detail_path": True},
        {"neck_type": "typo", **FOCAL},
    ]
    for config in wrong_configs:
        with pytest.raises(ValueError):
            resolve(config)


def test_candidate_requires_divisible_by_16_grid():
    for shape in [(800, 703), (15, 32), (0, 32), (True, 32), (32.0, 32), (32,)]:
        with pytest.raises(ValueError, match="grid_shape"):
            resolve(FOCAL, grid_shape=shape)


def test_geometry_alignment_and_legacy_behavior():
    geometry = {"x_min": 0, "x_max": 70.3, "x_res": 0.1,
                "y_min": -40, "y_max": 40, "y_res": 0.1}
    resolve({}, geometry=geometry)  # historical shape behavior is untouched
    with pytest.raises(ValueError, match="grid_shape"):
        resolve(FOCAL, geometry=geometry)
    resolve(FOCAL, geometry={**geometry, "x_max": 70.4})
    with pytest.raises(ValueError, match="grid_shape"):
        resolve({"stage_depths": [3, 4, 2]}, grid_shape=(33, 32))


def test_resolver_accepts_legacy_non_next_config_without_new_options():
    assert resolve({"backbone": "pixor"}).c4_context == "none"
    assert resolve({"neck_type": "rc_sgfpn"}).neck_fusion_channels == 24
    assert resolve({"c4_attention": "none", "c4_attention_scales": [],
                    "c4_attention_qk_norm": "none"}).c4_attention_scales == ()


def test_unrepresentable_scalar_fails_with_configuration_error():
    with pytest.raises(ValueError, match="c4_context_layer_scale_init"):
        resolve({**FOCAL, "c4_context_layer_scale_init": 10**400})


def test_conv_diagnostic_has_only_disabled_attention_settings():
    cfg = json.loads((ROOT / "configs/experiments/attention_diagnostic/e0_conv.json").read_text())["model"]
    result = resolve(cfg)
    assert result.c4_attention == "none" and result.c4_attention_scales == ()
    assert resolve({**cfg, "c4_context": "focal"}).c4_context == "focal"

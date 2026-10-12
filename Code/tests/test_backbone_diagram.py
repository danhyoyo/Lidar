"""Configuration/topology regression checks for the NN-SVG backbone exporter."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.visualization.export_backbone_diagram import build_diagram, export_diagram


def config():
    return json.loads((ROOT / "configs/config.json").read_text())


def nodes(graph):
    return {node["id"]: node for node in graph["nodes"]}


def test_default_shapes_and_two_input_fpn_follow_current_forward():
    cfg = config()
    before = copy.deepcopy(cfg)
    result = build_diagram(cfg)
    overview = result["views"]["overview"]
    by_id = nodes(overview)
    assert cfg == before
    assert by_id["input"]["shape"] == [8, 800, 704]
    assert by_id["stage2"]["shape"] == [48, 200, 176]
    assert by_id["stage3"]["shape"] == [96, 100, 88]
    assert by_id["stage4"]["shape"] == [128, 50, 44]
    assert by_id["output"]["shape"] == [16, 200, 176]
    assert [by_id[f"stage{i}"]["repeat"] for i in (2, 3, 4)] == [2, 4, 2]
    assert {edge["source"] for edge in overview["edges"] if edge["target"] == "p4"} == {"l4", "u4"}
    assert not any("LiteMLA" in n["label"] for n in by_id.values())
    gate = result["views"]["gate4"]
    assert {e["source"] for e in gate["edges"] if e["target"] == "sum"} == {"lateral", "up"}
    assert nodes(gate)["dw"]["groups"] == 48
    assert nodes(result["views"]["block2"])["expand"]["shape"][0] == 120


def test_experiment_width_depth_context_and_detail_branch_are_visible():
    cfg = config()
    cfg["data"]["bev_encoding"] = {"name": "hist14"}
    cfg["model"].update(stage_depths=[3, 4, 2], backbone_out_dim=32,
                        neck_fusion_channels=32, c4_context="focal",
                        local_attention="eca", detail_path=True)
    result = build_diagram(cfg)
    by_id = nodes(result["views"]["overview"])
    assert by_id["input"]["shape"][0] == 14
    assert by_id["stage2"]["repeat"] == 3
    assert by_id["output"]["shape"] == [32, 200, 176]
    assert by_id["l3"]["shape"][0] == 32
    assert by_id["context"]["detail_view"] == "context"
    assert by_id["local"]["detail_view"] == "local"
    assert by_id["detail"]["shape"] == [32, 200, 176]
    assert nodes(result["views"]["context"])["project"]["shape"][0] == 132
    assert nodes(result["views"]["local"])["conv"]["kernel"] == 3


def test_disabled_gate_and_reparameterized_training_deploy_topologies():
    cfg = config()
    cfg["model"].update(scale_gated_fpn=False, use_reparam=True)
    result = build_diagram(cfg)
    assert nodes(result["views"]["overview"])["p4"]["formula"] == "P4 = U4 + L4"
    assert "dw" not in nodes(result["views"]["gate4"])
    block = nodes(result["views"]["block2"])
    assert {"rep7", "rep3", "rep_identity", "rep_sum"} <= block.keys()
    cfg["model"]["deploy"] = True
    deployed = nodes(build_diagram(cfg)["views"]["block2"])
    assert deployed["dw"]["bias"] is True
    assert "rep_sum" not in deployed and "norm1" not in deployed


def test_simam_statistics_broadcast_and_variance_formula():
    cfg = config()
    cfg["model"].update(local_attention="simam", local_attention_simam_lambda=0.002)
    graph = build_diagram(cfg)["views"]["local"]
    by_id = nodes(graph)
    assert by_id["statistics"]["shape"] == [48, 1, 1]
    assert by_id["sigmoid"]["shape"] == [48, 200, 176]
    assert "0.002" in by_id["energy"]["formula"]
    assert {e["source"] for e in graph["edges"] if e["target"] == "energy"} == {"statistics", "input"}


def test_legacy_encoding_and_custom_expansion_use_repository_rounding():
    cfg = config()
    cfg["data"]["bev_encoding"] = {"name": "binary_slices"}
    cfg["model"].update(expansion=2.51, neck_type="sgfpn")
    result = build_diagram(cfg)
    assert nodes(result["views"]["overview"])["input"]["shape"] == [35, 800, 704]
    assert nodes(result["views"]["block3"])["expand"]["shape"][0] == round(96 * 2.51)


@pytest.mark.parametrize("updates,match", [
    ({"backbone": "pixor"}, "mobilepixornext"),
    ({"neck_type": "rc_bisgfpn"}, "neck_type"),
    ({"stage_depths": [2, 0, 2]}, "stage_depths"),
    ({"expansion": float("nan")}, "expansion"),
    ({"scale_gated_fpn": "false"}, "scale_gated_fpn"),
    ({"c4_attention": "litemla"}, "c4_attention"),
])
def test_unsupported_or_invalid_configs_fail_clearly(updates, match):
    cfg = config()
    cfg["model"].update(updates)
    with pytest.raises(ValueError, match=match):
        build_diagram(cfg)


def test_non_aligned_grid_rejected_before_drawing_invalid_fusion():
    cfg = config()
    cfg["data"]["kitti"]["geometry"]["x_max"] = 70.3
    with pytest.raises(ValueError, match="divisible by 16"):
        build_diagram(cfg)


def test_export_is_portable_and_does_not_import_torch(tmp_path):
    result = build_diagram(config(), source="configs/config.json")
    export_diagram(result, tmp_path)
    assert json.loads((tmp_path / "mobilepixornext.json").read_text()) == result
    assert (tmp_path / "mobilepixornext.js").read_text().startswith("window.BACKBONE_DIAGRAM = ")
    subprocess.run([sys.executable, "-c", "from tools.visualization.export_backbone_diagram "
                    "import build_diagram; import sys; assert 'torch' not in sys.modules"],
                   cwd=ROOT, check=True)


def test_inferred_shapes_match_actual_backbone_when_torch_available():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "detector"))
    from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone
    cfg = config()
    result = build_diagram(cfg)
    with torch.device("meta"):
        model = MobilePixorNeXtBackbone()
        output = model(torch.empty(1, 8, 800, 704))
    assert list(output.shape[1:]) == nodes(result["views"]["overview"])["output"]["shape"]

"""Selectable per-group shared features and historical checkpoint compatibility."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]

from common import build_model, checkpoint_identity, generate_run_name, validate_evaluation_checkpoint
from detector.core.models.heads.cnn import Header
from detector.core.models.heads.grouped import GroupedHeader
from detector.core.task_groups import resolve_task_groups


def config(architecture=None):
    result = json.loads((ROOT / "configs/experiments/under1m/hist14_local3_focal_simam_grouped_oga_iqa.json").read_text())
    if architecture is not None:
        result["model"]["header_architecture"] = architecture
    return result


def grouped_header(architecture="shared", **kwargs):
    cfg = config()
    groups = resolve_task_groups(cfg["data"]["head_groups"], cfg["data"]["kitti"]["objects"])
    return GroupedHeader(groups, 32, use_bn=True, act="silu",
                         header_architecture=architecture, **kwargs)


@pytest.mark.parametrize("architecture", ["independent", "shared"])
@pytest.mark.parametrize("classification,widths", [("gaussian", [1, 2]), ("binary", [2, 3])])
@pytest.mark.parametrize("box_mode,iqa", [("bev", False), ("bev", True), ("3d", True)])
def test_output_contract_and_group_parameter_isolation(architecture, classification, widths, box_mode, iqa):
    head = grouped_header(architecture, cls_encoding=classification, use_iou=iqa, box_mode=box_mode).eval()
    output = head(torch.randn(2, 32, 7, 11))["groups"]
    assert list(output) == ["car", "ped_cyc"]
    for name, width in zip(output, widths):
        expected = {"cls": width, "offset": 2, "size": 2, "yaw": 2}
        if iqa:
            expected["iou"] = 1
        if box_mode == "3d":
            expected["vertical"] = 2
        assert output[name].keys() == expected.keys()
        for key, channels in expected.items():
            assert output[name][key].shape == (2, channels, 7, 11)
            assert torch.isfinite(output[name][key]).all()
    pointers = {p.data_ptr() for p in head.heads["car"].parameters()}
    assert pointers.isdisjoint(p.data_ptr() for p in head.heads["ped_cyc"].parameters())


@pytest.mark.parametrize("iqa,expected", [(False, 93295), (True, 111921)])
def test_shared_head_parameter_budget(iqa, expected):
    head = grouped_header(use_iou=iqa)
    assert sum(p.numel() for p in head.parameters()) == expected


def test_shared_block_runs_once_per_group_and_single_task_gradient_is_isolated():
    head = grouped_header(use_iou=True, box_mode="3d")
    calls = {name: 0 for name in head.heads}
    def record(name):
        def hook(*_):
            calls[name] += 1
        return hook
    handles = [group.shared.register_forward_hook(record(name)) for name, group in head.heads.items()]
    x = torch.randn(2, 32, 8, 12, requires_grad=True)
    output = head(x)["groups"]
    for handle in handles:
        handle.remove()
    assert calls == {"car": 1, "ped_cyc": 1}
    output["car"]["iou"].square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    car = head.heads["car"]
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for module in (car.shared, car.iou) for p in module.parameters())
    assert all(p.grad is None for key in ("cls", "offset", "size", "yaw", "vertical")
               for p in getattr(car, key).parameters())
    assert all(p.grad is None for p in head.heads["ped_cyc"].parameters())


@pytest.mark.parametrize("act", [None, "none", "relu", "silu"])
@pytest.mark.parametrize("use_bn", [False, True])
def test_shared_honors_normalization_and_activation(act, use_bn):
    head = Header(3, 16, use_bn=use_bn, act=act, header_architecture="shared").eval()
    expected_act = {None: torch.nn.Identity, "none": torch.nn.Identity,
                    "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}[act]
    assert isinstance(head.shared[1], torch.nn.BatchNorm2d if use_bn else torch.nn.Identity)
    assert isinstance(head.shared[2], expected_act)
    assert isinstance(head.cls.bn1, torch.nn.BatchNorm2d if use_bn else torch.nn.Identity)
    assert isinstance(head.cls.act1, expected_act)
    assert sum(isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3) for m in head.modules()) == 5
    assert torch.isfinite(head(torch.randn(1, 16, 5, 7))["cls"]).all()


@pytest.mark.parametrize("grouped", [False, True])
def test_omitted_and_explicit_independent_preserve_state_and_seeded_predictions(grouped):
    cfg = config()
    if not grouped:
        cfg["model"]["head_mode"] = "legacy_single"
        del cfg["data"]["head_groups"]
        cfg["loss"].pop("group_weights", None)
    torch.manual_seed(23)
    old = build_model(cfg).eval()
    explicit = copy.deepcopy(cfg)
    explicit["model"]["header_architecture"] = "independent"
    torch.manual_seed(23)
    new = build_model(explicit).eval()
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, tensor in old.state_dict().items():
        torch.testing.assert_close(tensor, new.state_dict()[key], rtol=0, atol=0)
    assert not any(".shared." in key for key in old.state_dict())
    new.load_state_dict(old.state_dict(), strict=True)
    assert checkpoint_identity(cfg) == checkpoint_identity(explicit)
    assert generate_run_name(cfg) == generate_run_name(explicit)
    features = torch.randn(1, 14, 32, 48)
    with torch.no_grad():
        expected, actual = old(features), new(features)
    expected = expected["groups"] if grouped else {"single": expected}
    actual = actual["groups"] if grouped else {"single": actual}
    for name in expected:
        for key in expected[name]:
            torch.testing.assert_close(expected[name][key], actual[name][key], rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["independent", "shared"])
def test_pipeline_full_resolution_contract(architecture):
    cfg = config(architecture)
    before = copy.deepcopy(cfg)
    with torch.device("meta"):
        model = build_model(cfg).eval()
        output = model(torch.zeros(1, 14, 800, 704))["groups"]
    assert cfg == before
    assert model.header_architecture == architecture
    assert output["car"]["cls"].shape == (1, 1, 200, 176)
    assert output["ped_cyc"]["cls"].shape == (1, 2, 200, 176)
    assert output["ped_cyc"]["iou"].shape == (1, 1, 200, 176)


@pytest.mark.parametrize("bad", [None, True, 1, "proposed", "", []])
def test_invalid_architecture_rejected_at_config_and_direct_header_boundaries(bad):
    from common import detection_spec
    cfg = config()
    cfg["model"]["header_architecture"] = bad
    with pytest.raises(ValueError, match="header_architecture"):
        detection_spec(cfg)
    with pytest.raises(ValueError, match="header_architecture"):
        Header(3, 32, header_architecture=bad)
    with pytest.raises(ValueError, match="header_architecture"):
        build_model(cfg)


def test_shared_identity_is_canonical_and_run_name_is_distinct():
    original, proposed = config(), config("shared")
    assert checkpoint_identity(original) != checkpoint_identity(proposed)
    assert checkpoint_identity(proposed)["model"]["header_architecture"] == "shared"
    assert "header_architecture" not in checkpoint_identity(original)["model"]
    assert "head_shared" in generate_run_name(proposed)
    assert generate_run_name(original) != generate_run_name(proposed)
    uppercase = config("SHARED")
    assert checkpoint_identity(uppercase) == checkpoint_identity(proposed)
    assert generate_run_name(uppercase) == generate_run_name(proposed)


def test_shared_checkpoint_roundtrip_and_reject_cross_architecture_before_restore():
    from test_grouped_checkpoint import components, payload, restore
    source_cfg = config("shared")
    source = components(source_cfg)
    saved = payload(source_cfg, source)
    destination = components(source_cfg)
    assert restore(saved, source_cfg, destination)["mode"] == "resume"
    for key, value in source[0].state_dict().items():
        torch.testing.assert_close(value, destination[0].state_dict()[key], rtol=0, atol=0)
    validate_evaluation_checkpoint(saved, source_cfg)
    for cfg in (config(), config("independent")):
        target = components(cfg)
        before = {key: value.clone() for key, value in target[0].state_dict().items()}
        with pytest.raises(ValueError, match="identity"):
            restore(saved, cfg, target)
        with pytest.raises(ValueError, match="identity"):
            validate_evaluation_checkpoint(saved, cfg)
        for key, value in before.items():
            torch.testing.assert_close(value, target[0].state_dict()[key], rtol=0, atol=0)
    # A historical checkpoint with no architecture key also resumes when the
    # old architecture is explicitly selected by a newer notebook.
    old_cfg = config()
    old = payload(old_cfg, components(old_cfg))
    assert restore(old, config("independent"), components(config("independent")))["mode"] == "resume"


@pytest.mark.parametrize("classification,strategy,iqa", [
    ("gaussian", "baseline", True), ("gaussian", "oga", True),
    ("binary", "oga", True), ("gaussian", "uwag", False),
    ("gaussian", "exact_q_oga", False)])
def test_shared_real_training_loss_backward_and_optimizer_step(classification, strategy, iqa):
    from test_context_grouped_compatibility import fixture
    import train
    cfg, target = fixture(classification, strategy, iqa, local="simam")
    cfg["model"]["header_architecture"] = "shared"
    model = build_model(cfg).train()
    criterion = train.build_training_criterion(cfg, torch.device("cpu"))
    train.set_loss_epoch(criterion, 2)
    optimizer = train.build_optimizer(model, criterion, cfg)
    result = criterion(model(target["voxel"]), target)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    shared_weights = [group.shared[0].weight for group in model.grouped_header.heads.values()]
    before = [weight.detach().clone() for weight in shared_weights]
    assert all(weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
               for weight in shared_weights)
    optimizer.step()
    assert all(not torch.equal(weight, original) for weight, original in zip(shared_weights, before))


@pytest.mark.parametrize("preset", ["custom", "UNDER1M_FOCAL_OGA_IQA"])
def test_standard_notebook_architecture_control_reaches_saved_config(tmp_path, preset):
    from test_under1m_notebook_controls import execute_cell
    namespace = execute_cell(tmp_path, {"PRESET": preset, "HEADER_ARCHITECTURE": "shared"})
    cfg = namespace["config_dict"]
    assert cfg["model"]["header_architecture"] == "shared"
    assert json.loads(namespace["CONFIG"].read_text())["model"]["header_architecture"] == "shared"
    assert build_model(cfg).grouped_header.heads["car"].header_architecture == "shared"


def test_old_notebook_snapshot_accepts_explicit_default_without_overwrite(tmp_path):
    from notebook_config import save_notebook_config
    old = config()
    path = save_notebook_config(tmp_path, old)
    original = path.read_bytes()
    (tmp_path / "run.json").write_text("{}")
    assert save_notebook_config(tmp_path, config("independent")) == path
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="config"):
        save_notebook_config(tmp_path, config("shared"))


def test_shared_resume_after_optimizer_step_reproduces_the_next_step(tmp_path):
    from test_context_grouped_compatibility import fixture
    from test_grouped_checkpoint import components, payload, restore, stochastic_step
    import train
    cfg, target = fixture(local="simam")
    cfg["model"]["header_architecture"] = "shared"
    train.seed_everything(91)
    parts = components(cfg)
    stochastic_step(cfg, parts, target, 0)
    saved = payload(cfg, parts)
    path = tmp_path / "shared.pt"
    torch.save(saved, path)
    expected_loss = stochastic_step(cfg, parts, target, 1)
    destination = components(cfg)
    restore(torch.load(path, weights_only=True), cfg, destination)
    actual_loss = stochastic_step(cfg, destination, target, 1)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for source, restored in zip(parts[:2], destination[:2]):
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    expected_optimizer, actual_optimizer = parts[2].state_dict(), destination[2].state_dict()
    assert expected_optimizer["param_groups"] == actual_optimizer["param_groups"]
    for key, fields in expected_optimizer["state"].items():
        for name, value in fields.items():
            torch.testing.assert_close(value, actual_optimizer["state"][key][name], rtol=0, atol=0)
    assert parts[3].state_dict() == destination[3].state_dict()


def test_switching_architecture_warm_starts_only_backbone():
    from common import warm_start_backbone
    source = build_model(config())
    target = build_model(config("shared"))
    with torch.no_grad():
        for value in source.backbone.state_dict().values():
            value.fill_(.25 if value.is_floating_point() else 4)
    before = {key: value.clone() for key, value in target.grouped_header.state_dict().items()}
    report = warm_start_backbone(target, source.state_dict())
    assert report["loaded"] and all(key.startswith("backbone.") for key in report["loaded"])
    assert any(key.startswith("grouped_header.") for key in report["skipped"])
    for key, value in target.backbone.state_dict().items():
        torch.testing.assert_close(value, source.backbone.state_dict()[key], rtol=0, atol=0)
    for key, value in before.items():
        torch.testing.assert_close(value, target.grouped_header.state_dict()[key], rtol=0, atol=0)


def test_shared_checkpoint_actual_pytorch_runner_and_grouped_decode(tmp_path):
    from test_grouped_checkpoint import components, payload
    from evaluate_kitti_bev import PyTorchRunner
    from detector.postprocess import filter_pred
    cfg = config("shared")
    geometry = cfg["data"]["kitti"]["geometry"]
    geometry["x_max"] = geometry["x_min"] + 48 * geometry["x_res"]
    geometry["y_max"] = geometry["y_min"] + 32 * geometry["y_res"]
    parts = components(cfg)
    path = tmp_path / "shared.pt"
    torch.save(payload(cfg, parts), path)
    runner = PyTorchRunner(path, cfg, "cpu")
    features = torch.randn(1, 14, 32, 48)
    output, _ = runner.infer(features)
    with torch.no_grad():
        expected = parts[0].eval()(features)
    for group in output["groups"]:
        for key, value in output["groups"][group].items():
            torch.testing.assert_close(value, expected["groups"][group][key], rtol=0, atol=0)
    assert runner.metadata()["parameter_counts"]["heads"] == 111921
    decoded = filter_pred(output, cfg["data"]["kitti"], 4, 1.,
                          task_groups=runner.model.task_groups, cls_encoding="gaussian")
    assert decoded.shape == (0, 7)


@pytest.mark.parametrize("box_mode", ["bev", "3d"])
def test_legacy_single_model_can_select_shared_architecture(box_mode):
    cfg = config("shared")
    cfg["model"]["head_mode"] = "legacy_single"
    cfg["model"]["cls_encoding"] = "binary"
    cfg["data"]["box_mode"] = box_mode
    del cfg["data"]["head_groups"]
    cfg["loss"].pop("group_weights", None)
    if box_mode == "3d":
        cfg["loss"]["vertical_loss_weight"] = 1.
    model = build_model(cfg).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 14, 32, 48))
    assert model.header.header_architecture == "shared"
    assert output["cls"].shape == (1, 4, 8, 12)
    assert ("vertical" in output) == (box_mode == "3d")


def test_shared_example_preserves_simam_f32_architecture_and_expected_budget():
    cfg = json.loads((ROOT / "configs/experiments/under1m/hist14_local3_focal_simam_fusion32_shared_grouped_oga_iqa.json").read_text())
    model = build_model(cfg)
    assert model.header_architecture == "shared"
    assert cfg["model"]["local_attention"] == "simam"
    assert cfg["model"]["neck_fusion_channels"] == 32
    assert sum(p.numel() for p in model.parameters()) == 778690

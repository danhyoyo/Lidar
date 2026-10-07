"""Semantic preflight and deterministic epoch-boundary training restoration."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from test_context_grouped_compatibility import fixture
from test_grouped_loss_state import snapshot, assert_state_equal
from test_grouped_training import configured_model
from common import build_model
import common
import train


def components(config):
    model = build_model(config)
    criterion = train.build_training_criterion(config, torch.device("cpu"))
    optimizer = train.build_optimizer(model, criterion, config)
    scheduler = train.build_scheduler(optimizer, config, 4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(47)
    return model, criterion, optimizer, scheduler, scaler, generator


def payload(config, parts, epoch=1):
    model, criterion, optimizer, scheduler, scaler, generator = parts
    return copy.deepcopy(train.checkpoint_payload(model, criterion, optimizer, scheduler, scaler,
                                                epoch, {"loss": 2.}, 2., config,
                                                loader_generator=generator))


def restore(saved, config, parts, **kwargs):
    model, criterion, optimizer, scheduler, scaler, generator = parts
    return train.restore_checkpoint(saved, config, model, criterion, optimizer, scheduler,
                                    scaler, loader_generator=generator, **kwargs)


def test_identity_is_pure_canonical_and_backend_independent():
    config, _ = fixture()
    before = copy.deepcopy(config)
    identity = common.checkpoint_identity(config)
    assert config == before and identity["version"] == 1
    assert identity["encoding"]["metadata"]["channels"] == 14
    assert identity["groups"][1]["classes"] == ["Pedestrian", "Cyclist"]
    explicit = copy.deepcopy(config)
    explicit["model"].update(c4_context_version=1, c4_context_bottleneck=64,
                             c4_context_dilations=[1, 2, 3], c4_context_layer_scale_init=.001,
                             local_attention_eca_kernel_size=3, local_attention_layer_scale_init=.001,
                             neck_fusion_channels=24, detail_path=False)
    explicit["loss"]["group_weights"] = {"car": 2., "ped_cyc": 2.}
    explicit["data"]["bev_encoding"]["backend"] = "numba"
    assert common.checkpoint_identity(explicit) == identity


CHANGES = [
    ("model", "stage_depths", [2, 4, 2]), ("model", "backbone_out_dim", 16),
    ("model", "c4_context", "none"), ("model", "c4_context_dilations", [1, 3, 5]),
    ("model", "c4_context_bottleneck", 48), ("model", "c4_context_layer_scale_init", .01),
    ("model", "local_attention_eca_kernel_size", 5), ("model", "local_attention_layer_scale_init", .02),
    ("model", "neck_fusion_channels", 32), ("model", "detail_path", True),
    ("loss", "temperature", 3.), ("loss", "iou_target_type", "rotated_iou"),
    ("loss", "iou_loss_weight", 2.), ("loss", "group_weights", {"car": 1., "ped_cyc": 2.}),
    ("data", "head_groups", [{"name": "car", "classes": ["Car"]},
                               {"name": "ped_cyc", "classes": ["Cyclist", "Pedestrian"]}]),
    ("data", "head_groups", [{"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]},
                               {"name": "car", "classes": ["Car"]}]),
    ("data", "bev_encoding", {"name": "rich8"}),
]


@pytest.mark.parametrize("section,key,value", CHANGES)
def test_mismatched_identity_rejects_before_any_model_criterion_or_optimizer_restore(section, key, value):
    config, _ = fixture()
    saved = payload(config, components(config))
    changed = copy.deepcopy(config)
    changed[section][key] = value
    destination = components(changed)
    model_before, criterion_before = snapshot(destination[0]), snapshot(destination[1])
    with pytest.raises(ValueError, match="identity"):
        restore(saved, changed, destination)
    assert_state_equal(destination[0], model_before)
    assert_state_equal(destination[1], criterion_before)
    assert not destination[2].state


@pytest.mark.parametrize("local,field,value", [("simam", "local_attention_simam_lambda", .002),
                                              ("eca", "local_attention", "none")])
def test_shape_compatible_local_semantics_cannot_silently_resume(local, field, value):
    config, _ = fixture(local=local)
    saved = payload(config, components(config))
    changed = copy.deepcopy(config)
    changed["model"][field] = value
    with pytest.raises(ValueError, match="identity"):
        restore(saved, changed, components(changed))


@pytest.mark.parametrize("local", ["none", "eca", "simam"])
def test_matching_serialized_checkpoint_roundtrips_all_bn_gamma_beta_and_iqa(local, tmp_path):
    config, target = fixture(local=local)
    parts = components(config)
    train.set_loss_epoch(parts[1], 0)
    result = parts[1](parts[0](target["voxel"]), target)
    result["loss"].backward()
    parts[2].step()
    parts[3].step()
    saved = payload(config, parts)
    path = tmp_path / "checkpoint.pt"
    torch.save(saved, path)
    # RNG metadata must remain compatible with modern weights-only loading.
    saved = torch.load(path, weights_only=True)
    destination = components(config)
    result = restore(saved, config, destination)
    assert result["mode"] == "resume" and result["epoch"] == 1 and result["best_val"] == 2.
    assert_state_equal(destination[0], snapshot(parts[0]))
    assert_state_equal(destination[1], snapshot(parts[1]))
    assert destination[3].state_dict() == parts[3].state_dict()
    assert any("running_mean" in k for k in saved["model_state_dict"])
    assert any(k.endswith("gamma") for k in saved["model_state_dict"])
    if local != "none":
        assert any(k.endswith("beta") for k in saved["model_state_dict"])


@pytest.mark.parametrize("missing", ["criterion_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                                    "scaler_state_dict", "rng_state", "checkpoint_identity", "ema", "model_tensor"])
def test_incomplete_full_resume_rejects_before_mutation_even_for_legacy_ema_fallback(missing):
    config, _ = fixture()
    saved = payload(config, components(config))
    if missing == "ema":
        key = next(k for k in saved["criterion_state_dict"] if "running_loss_means" in k)
        del saved["criterion_state_dict"][key]
    elif missing == "model_tensor":
        key = next(iter(saved["model_state_dict"]))
        saved["model_state_dict"][key] = torch.zeros(2)
    else:
        del saved[missing]
    destination = components(config)
    model_before, criterion_before = snapshot(destination[0]), snapshot(destination[1])
    with pytest.raises(ValueError):
        restore(saved, config, destination)
    assert_state_equal(destination[0], model_before)
    assert_state_equal(destination[1], criterion_before)


def test_matching_pure_legacy_weights_load_explicitly_without_claiming_training_resume():
    config = configured_model(grouped=False)
    source, destination = components(config), components(config)
    result = restore(copy.deepcopy(source[0].state_dict()), config, destination)
    assert result == {"mode": "legacy_weights", "epoch": 0, "best_val": float("inf"), "backend_changes": {}}
    assert_state_equal(destination[0], snapshot(source[0]))
    assert not destination[2].state
    grouped = configured_model()
    with pytest.raises(ValueError, match="identity|legacy"):
        restore(source[0].state_dict(), grouped, components(grouped))


def test_backend_change_requires_explicit_parity_assertion_and_is_reported():
    config, _ = fixture()
    saved = payload(config, components(config))
    changed = copy.deepcopy(config)
    changed["data"]["bev_encoding"]["backend"] = "numba"
    destination = components(changed)
    with pytest.raises(ValueError, match="parity"):
        restore(saved, changed, destination)
    result = restore(saved, changed, destination, backend_parity_verified=True)
    assert result["backend_changes"] == {"bev": {"saved": "numpy", "current": "numba"}}


def stochastic_step(config, parts, target, epoch):
    model, criterion, optimizer, scheduler, _, generator = parts
    model.train()
    criterion.train()
    train.set_loss_epoch(criterion, epoch)
    optimizer.zero_grad(set_to_none=True)
    order = torch.randperm(8, generator=generator)
    voxel = torch.randn_like(target["voxel"]) * (.1 + random.random() + float(np.random.random()))
    voxel = voxel + order.float().mean() / 100
    result = criterion(model(voxel), target)
    result["loss"].backward()
    optimizer.step()
    scheduler.step()
    return result["loss"].detach()


@pytest.mark.parametrize("strategy,iqa", [("baseline", True), ("oga", True), ("uwag", False),
                                         ("gw_qal", False), ("q_oga", False), ("exact_q_oga", False)])
@pytest.mark.parametrize("empty", [False, True])
def test_uninterrupted_and_restored_next_optimizer_step_match_rng_ema_curriculum_and_scheduler(strategy, iqa, empty):
    train.seed_everything(91)
    config, target = fixture(strategy=strategy, iqa=iqa, empty=empty)
    parts = components(config)
    stochastic_step(config, parts, target, 0)
    saved = payload(config, parts)
    expected_loss = stochastic_step(config, parts, target, 1)
    model_after, criterion_after = snapshot(parts[0]), snapshot(parts[1])
    optimizer_after = copy.deepcopy(parts[2].state_dict())
    draws = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=parts[-1]))
    destination = components(config)
    restore(saved, config, destination)
    actual_loss = stochastic_step(config, destination, target, 1)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert_state_equal(destination[0], model_after)
    assert_state_equal(destination[1], criterion_after)
    assert destination[3].state_dict() == parts[3].state_dict()
    for key, value in optimizer_after["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(destination[2].state_dict()["state"][key][name], tensor, rtol=0, atol=0)
    assert random.random() == draws[0] and np.random.random() == draws[1]
    torch.testing.assert_close(torch.rand(3), draws[2], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(3, generator=destination[-1]), draws[3], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["python", "numpy", "torch_cpu", "loader_generator"])
def test_missing_or_invalid_rng_is_rejected_before_any_state_restore(field):
    config, _ = fixture()
    saved = payload(config, components(config))
    saved["rng_state"][field] = None
    destination = components(config)
    before = snapshot(destination[0])
    with pytest.raises(ValueError, match="RNG"):
        restore(saved, config, destination)
    assert_state_equal(destination[0], before)


@pytest.mark.parametrize("change", ["iqa", "head_mode", "global_classes", "group_name", "density_norm",
                                   "quality_target", "curriculum", "optimizer_mode", "schedule", "precision"])
def test_additional_semantic_mismatches_fail_before_restore(change):
    config, _ = fixture(strategy="exact_q_oga" if change in ("quality_target", "curriculum") else "oga",
                        iqa=change not in ("quality_target", "curriculum"))
    saved = payload(config, components(config))
    changed = copy.deepcopy(config)
    if change == "iqa":
        changed["model"]["header_use_iou"] = changed["loss"]["use_iou"] = False
    elif change == "head_mode":
        changed["model"]["head_mode"] = "legacy_single"
        del changed["data"]["head_groups"]
    elif change == "global_classes":
        changed["data"]["kitti"]["objects"].update(Pedestrian=2, Cyclist=1)
    elif change == "group_name":
        changed["data"]["head_groups"][1]["name"] = "vru"
    elif change == "density_norm":
        changed["data"]["bev_encoding"]["density_norm"] = 64.
    elif change == "quality_target":
        changed["loss"].update(quality_target="mgiou", quality_warmup_epochs=0)
    elif change == "curriculum":
        changed["loss"]["quality_warmup_epochs"] = 8
    elif change == "optimizer_mode":
        changed["train"]["optimizer"] = "adam"
    elif change == "schedule":
        changed["train"]["scheduler"] = "multistep"
    else:
        changed["train"]["precision"] = "fp16"
    destination = components(changed)
    before = snapshot(destination[0])
    with pytest.raises(ValueError, match="identity"):
        restore(saved, changed, destination)
    assert_state_equal(destination[0], before)


@pytest.mark.parametrize("broken", ["optimizer_shape", "optimizer_moment", "optimizer_order", "optimizer_entry", "scheduler_keys", "scaler",
                                   "state_types", "epoch", "metadata", "backends"])
def test_corrupt_training_state_is_preflighted_before_loading_any_weights(broken):
    config, target = fixture()
    source = components(config)
    stochastic_step(config, source, target, 0)
    saved = payload(config, source)
    if broken.startswith("optimizer"):
        opt = saved["optimizer_state_dict"]
        if broken == "optimizer_shape":
            opt["state"][next(iter(opt["state"]))]["exp_avg"] = torch.zeros(2)
        elif broken == "optimizer_moment":
            del opt["state"][next(iter(opt["state"]))]["exp_avg_sq"]
        elif broken == "optimizer_entry":
            del opt["state"][next(iter(opt["state"]))]
        else:
            opt["param_groups"][0]["params"].reverse()
    elif broken == "scheduler_keys":
        saved["scheduler_state_dict"].clear()
    elif broken == "scaler":
        saved["scaler_state_dict"] = {"scale": 12.}
    elif broken == "state_types":
        saved["training_state_types"]["optimizer"] = "SGD"
    elif broken == "epoch":
        saved["epoch"] = -1
    elif broken == "metadata":
        saved["config"]["model"]["c4_context_dilations"] = [1, 2, 4]
    else:
        del saved["backends"]["bev"]
    destination = components(config)
    before = snapshot(destination[0])
    with pytest.raises(ValueError):
        restore(saved, config, destination)
    assert_state_equal(destination[0], before)


def test_worker_replay_limit_is_recorded_and_invalid_curriculum_epoch_fails_preflight():
    config, target = fixture(strategy="exact_q_oga", iqa=False)
    config["train"]["num_workers"] = 2
    source = components(config)
    stochastic_step(config, source, target, 0)
    saved = payload(config, source)
    assert saved["replay"] == {"boundary": "epoch_end", "num_workers": 2, "deterministic_worker_replay": False}
    key = next(k for k in saved["criterion_state_dict"] if k.endswith("quality_epoch"))
    saved["criterion_state_dict"][key].fill_(4)
    destination = components(config)
    before = snapshot(destination[0])
    with pytest.raises(ValueError, match="curriculum"):
        restore(saved, config, destination)
    assert_state_equal(destination[0], before)


@pytest.mark.parametrize("strategy,iqa,grouped", [("oga", True, True), ("q_oga", False, True), ("oga", True, False)])
def test_real_cli_epoch_one_resume_matches_uninterrupted_epoch_two_with_workers_zero(tmp_path, strategy, iqa, grouped):
    from test_grouped_targets import make_dataset
    dataset = make_dataset(tmp_path, grouped=grouped)
    config = configured_model(strategy=strategy, iqa=iqa, grouped=grouped)
    config["model"]["local_attention"] = "simam"
    config["data"] = dataset.config
    config["augmentation"] = {
        "p": .9, "rotation": {"use": True, "limit_angle": .1, "p": 1.},
        "scaling": {"use": False}, "translation": {"use": False}}
    root = Path(config["data"]["kitti"]["location"])
    (root / "pointcloud/000001.bin").write_bytes((root / "pointcloud/000000.bin").read_bytes())
    (root / "label/000001.txt").write_text((root / "label/000000.txt").read_text())
    validation = tmp_path / "validation.txt"
    validation.write_text("000001;kitti\n")
    config["train"].update(data=str(dataset.data_file), epochs=2, physical_batch_size=1,
                           accumulation_steps=1, save_every=1)
    config["val"] = {"data": str(validation), "physical_batch_size": 1}
    if strategy == "q_oga":
        config["loss"].update(quality_target="rotated_iou", quality_warmup_epochs=2)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = ["--config", str(path), "--detector-root", str(Path(__file__).resolve().parents[1] / "detector"),
            "--output-root", str(tmp_path / "runs"), "--device", "cpu", "--num-workers", "0", "--precision", "fp32"]
    train.main(args + ["--run-name", "uninterrupted"])
    first = tmp_path / "runs/uninterrupted/checkpoints/1epoch.pt"
    train.main(args + ["--run-name", "resumed", "--resume", str(first)])
    expected = torch.load(tmp_path / "runs/uninterrupted/checkpoints/2epoch.pt", weights_only=True)
    actual = torch.load(tmp_path / "runs/resumed/checkpoints/2epoch.pt", weights_only=True)
    assert actual["replay"]["deterministic_worker_replay"]
    for key in ("model_state_dict", "criterion_state_dict"):
        assert actual[key].keys() == expected[key].keys()
        for name, value in expected[key].items():
            torch.testing.assert_close(actual[key][name], value, rtol=0, atol=0)
    assert {key: value for key, value in actual["validation"].items() if key != "seconds"} == {
        key: value for key, value in expected["validation"].items() if key != "seconds"}
    assert actual["scheduler_state_dict"] == expected["scheduler_state_dict"]
    for key in ("torch_cpu", "loader_generator"):
        torch.testing.assert_close(actual["rng_state"][key], expected["rng_state"][key], rtol=0, atol=0)


@pytest.mark.parametrize("wrapped", [False, True])
def test_backbone_only_warm_start_preserves_new_input_stage_context_heads_and_training_state(wrapped):
    source_config = configured_model(grouped=False)
    source_config["data"]["bev_encoding"] = {"name": "rich8"}
    source_config["model"].update(stage_depths=[2, 4, 2], c4_context="none", c4_attention="litemla",
                                   c4_attention_scales=[5])
    source = components(source_config)
    with torch.no_grad():
        for value in source[0].state_dict().values():
            value.fill_(.25 if value.is_floating_point() else 4)
    config, _ = fixture(local="eca")
    destination = components(config)
    model_before, criterion_before = snapshot(destination[0]), snapshot(destination[1])
    checkpoint = payload(source_config, source) if wrapped else source[0].state_dict()
    result = common.warm_start_backbone(destination[0], checkpoint)
    assert result["loaded"] and result["skipped"] and result["missing"]
    assert "backbone.stem.0.weight" in result["skipped"]
    assert any(name.startswith("header.") for name in result["skipped"])
    assert any(name.startswith("backbone.c4_attention.") for name in result["skipped"])
    assert any(name.startswith("backbone.stage2.2.") for name in result["missing"])
    assert any(name.startswith("backbone.c4_context.") for name in result["missing"])
    source_state = source[0].state_dict()
    for name, value in destination[0].state_dict().items():
        expected = source_state[name] if name in result["loaded"] else model_before[name]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert not any(name.startswith("grouped_header.") for name in result["loaded"])
    assert_state_equal(destination[1], criterion_before)
    assert not destination[2].state
    assert destination[3].last_epoch == 0


@pytest.mark.parametrize("bad", ["head_only", "wrong_backbone", "invalid_tensor"])
def test_unsupported_warm_start_fails_without_mutating_the_model(bad):
    config, _ = fixture()
    model = components(config)[0]
    before = snapshot(model)
    if bad == "head_only":
        state = {k: v for k, v in model.state_dict().items() if k.startswith("grouped_header.")}
    elif bad == "wrong_backbone":
        state = {"backbone.unknown.weight": torch.zeros(4)}
    else:
        state = {"backbone.stem.0.weight": "bad"}
    with pytest.raises(ValueError, match="warm.start|backbone|tensor"):
        common.warm_start_backbone(model, state)
    assert_state_equal(model, before)


def test_resume_and_warm_start_are_mutually_exclusive_before_loading_files():
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--config", "missing", "--detector-root", "missing", "--output-root", "missing",
                                        "--resume", "old.pt", "--warm-start", "old.pt"])
    args = train.build_parser().parse_args(["--config", "missing", "--detector-root", "missing", "--output-root", "missing",
                                           "--warm-start", "old.pt"])
    assert args.warm_start == Path("old.pt") and args.resume is None


def test_actual_cli_warm_start_reports_keys_and_starts_fresh_exact_quality_training(tmp_path):
    from test_grouped_targets import make_dataset
    dataset = make_dataset(tmp_path)
    config = configured_model(strategy="q_oga", iqa=False)
    config["data"] = dataset.config
    config["loss"].update(quality_target="rotated_iou", quality_warmup_epochs=4)
    config["augmentation"] = {"p": 0., "rotation": {"use": False},
                              "scaling": {"use": False}, "translation": {"use": False}}
    data_root = Path(config["data"]["kitti"]["location"])
    (data_root / "pointcloud/000001.bin").write_bytes((data_root / "pointcloud/000000.bin").read_bytes())
    (data_root / "label/000001.txt").write_text((data_root / "label/000000.txt").read_text())
    validation = tmp_path / "val.txt"
    validation.write_text("000001;kitti\n")
    config["train"].update(data=str(dataset.data_file), epochs=1, physical_batch_size=1, save_every=1)
    config["val"] = {"data": str(validation), "physical_batch_size": 1}
    source_config = configured_model(grouped=False)
    source_config["data"]["bev_encoding"] = {"name": "rich8"}
    source_config["model"].update(stage_depths=[2, 4, 2], c4_context="none", c4_attention="litemla",
                                  c4_attention_scales=[5])
    source = components(source_config)
    source[-1].manual_seed(99)
    for value in source[1].state_dict().values():
        value.fill_(99)
    source_file = tmp_path / "source.pt"
    torch.save(payload(source_config, source, epoch=99), source_file)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    train.main(["--config", str(config_file), "--detector-root", str(Path(__file__).resolve().parents[1] / "detector"),
                "--output-root", str(tmp_path / "runs"), "--run-name", "warm", "--warm-start", str(source_file),
                "--device", "cpu", "--precision", "fp32", "--num-workers", "0"])
    run = tmp_path / "runs/warm"
    report = json.loads((run / "warm_start.json").read_text())
    assert report["mode"] == "backbone_warm_start" and report["sha256"] == common.sha256(source_file)
    assert "backbone.stem.0.weight" in report["skipped"] and report["loaded"]
    saved = torch.load(run / "checkpoints/last.pt", weights_only=True)
    assert saved["epoch"] == 1 and saved["scheduler_state_dict"]["last_epoch"] == 1
    assert saved["config"]["initialization"]["mode"] == "backbone_warm_start"
    for key, value in saved["criterion_state_dict"].items():
        if key.endswith("quality_epoch"):
            assert value == 0
    assert all(value["step"] == 1 for value in saved["optimizer_state_dict"]["state"].values())


@pytest.mark.parametrize("wrapper", ["model_state_dict", "state_dict", "model"])
def test_ddp_wrapped_matching_grouped_weights_still_do_not_initialize_heads(wrapper):
    config, _ = fixture(local="simam")
    source, destination = components(config), components(config)
    before = snapshot(destination[0])
    checkpoint = {wrapper: {"module." + key: value for key, value in source[0].state_dict().items()}}
    report = common.warm_start_backbone(destination[0], checkpoint)
    assert all(key.startswith("backbone.") for key in report["loaded"])
    for key, value in destination[0].state_dict().items():
        expected = source[0].state_dict()[key] if key.startswith("backbone.") else before[key]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)

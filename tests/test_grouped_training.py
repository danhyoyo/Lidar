"""Recursive tensor transfer without converting heterogeneous metadata."""

import sys
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
from train import move_tensor_batch


def test_recursive_transfer_reaches_groups_and_nested_lists_and_tuples():
    points = np.array([[1., 2., 3., .5]], np.float32)
    batch = {"voxel": torch.zeros(1, 14, 16, 32),
             "groups": OrderedDict([
                 ("car", {"cls": torch.zeros(1, 1, 4, 8), "reg_mask": torch.ones(1, 4, 8)}),
                 ("ped_cyc", {"cls": torch.zeros(1, 2, 4, 8), "reg_mask": torch.zeros(1, 4, 8)}),
             ]),
             "metadata": ["kitti", None, (torch.ones(2), [torch.zeros(3), points])],
             "heterogeneous_points": [points, np.empty((0, 4), np.float32)],
             "frame_id": "000000", "batch_number": 2}
    original_car = batch["groups"]["car"]["cls"]
    moved = move_tensor_batch(batch, torch.device("meta"))
    assert moved is not batch and batch["groups"]["car"]["cls"] is original_car
    assert batch["voxel"].device.type == "cpu"
    assert moved["voxel"].device.type == "meta"
    assert isinstance(moved["groups"], OrderedDict)
    assert tuple(moved["groups"]) == ("car", "ped_cyc")
    for group in moved["groups"].values():
        assert all(tensor.device.type == "meta" for tensor in group.values())
    assert isinstance(moved["metadata"], list) and isinstance(moved["metadata"][2], tuple)
    assert moved["metadata"][2][0].device.type == "meta"
    assert moved["metadata"][2][1][0].device.type == "meta"
    assert moved["metadata"][2][1][1] is points
    assert moved["heterogeneous_points"][0] is points
    assert moved["heterogeneous_points"][1] is batch["heterogeneous_points"][1]
    assert moved["metadata"][:2] == ["kitti", None]
    assert moved["frame_id"] == "000000" and moved["batch_number"] == 2


def test_cpu_legacy_transfer_preserves_values_dtypes_and_flat_keys():
    batch = {"voxel": torch.randn(1, 8, 16, 16), "cls": torch.ones(1, 4, 4, dtype=torch.int64),
             "reg_mask": torch.zeros(1, 4, 4), "dtype": ["kitti"]}
    moved = move_tensor_batch(batch, torch.device("cpu"))
    assert moved.keys() == batch.keys()
    for key in ("voxel", "cls", "reg_mask"):
        torch.testing.assert_close(moved[key], batch[key], rtol=0, atol=0)
        assert moved[key].dtype == batch[key].dtype
    assert moved["dtype"] == ["kitti"]


def test_namedtuple_and_autograd_edges_survive_nested_transfer():
    Pair = namedtuple("Pair", "features metadata")
    leaf = torch.randn(2, requires_grad=True)
    pair = Pair(leaf, "unchanged")
    batch = {"nested": pair, "empty": ({}, [], ())}
    moved = move_tensor_batch(batch, torch.device("cpu"))
    assert isinstance(moved["nested"], Pair)
    assert moved["nested"].metadata == "unchanged"
    assert moved["empty"] == ({}, [], ())
    moved["nested"].features.square().sum().backward()
    torch.testing.assert_close(leaf.grad, 2 * leaf.detach())


def test_all_tensor_transfers_keep_nonblocking_behavior():
    class RecordedTensor(torch.Tensor):
        def to(self, *args, **kwargs):
            self.transfer_kwargs = kwargs
            return super().to(*args, **kwargs)

    leaf = torch.ones(1).as_subclass(RecordedTensor)
    move_tensor_batch({"groups": {"car": {"cls": leaf}}}, torch.device("cpu"))
    assert leaf.transfer_kwargs == {"non_blocking": True}


def configured_model(classification="gaussian", strategy="oga", iqa=True, grouped=True):
    from test_grouped_header import pipeline_config
    config = pipeline_config()
    # These fixtures cover historical loss-only, batch-limited training without
    # original KITTI evaluation assets. Actual AP selection has its own suite.
    config['train'].pop('checkpoint_selection', None)
    config["model"].update(cls_encoding=classification, header_use_iou=iqa)
    config["loss"] = {"name": strategy, "use_iou": iqa}
    config["train"].update(learning_rate=.001, optimizer="adamw", scheduler="cosine", warmup_epochs=0)
    if not grouped:
        config["model"]["head_mode"] = "legacy_single"
        del config["data"]["head_groups"]
    return config


@pytest.mark.parametrize("classification,strategy,iqa", [
    ("gaussian", "baseline", True), ("gaussian", "oga", True), ("gaussian", "uwag", False),
    ("gaussian", "gw_qal", False), ("gaussian", "q_oga", False),
    ("binary", "baseline", True), ("binary", "oga", True), ("binary", "uwag", False),
])
@pytest.mark.parametrize("empty_vru", [False, True])
def test_training_factory_optimizer_step_and_validation_preserve_grouped_contract(classification, strategy, iqa, empty_vru):
    import train
    from common import build_model
    from test_grouped_losses import tensors
    from test_grouped_loss_state import snapshot, assert_state_equal
    config = configured_model(classification, strategy, iqa)
    model = build_model(config).train()
    loss = train.build_training_criterion(config, torch.device("cpu")).train()
    optimizer = train.build_optimizer(model, loss, config)
    expected = {id(p) for module in (model, loss) for p in module.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert set(actual) == expected and len(actual) == len(expected)
    assert all(p.device.type == "cpu" for p in loss.parameters())
    _, target = tensors(classification, iqa, empty_vru)
    target["groups"] = {name: {key: value.repeat(*([1] * (value.ndim - 2)), 2, 2)
                                for key, value in item.items()} for name, item in target["groups"].items()}
    target["voxel"] = torch.randn(1, 14, 32, 48)
    initial = model.grouped_header.heads["car"].cls.head.weight.detach().clone()
    result = loss(model(target["voxel"]), target)
    result["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in loss.parameters())
    parameters = [p for module in (model, loss) for p in module.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(parameters, 1.)
    optimizer.step()
    assert not torch.equal(initial, model.grouped_header.heads["car"].cls.head.weight)
    criterion_state = snapshot(loss)
    metrics = train.validate(model, loss, [target], torch.device("cpu"), "fp32")
    assert np.isfinite(metrics["loss"])
    assert "group/car/loss" in metrics
    assert_state_equal(loss, criterion_state)


@pytest.mark.parametrize("grouped", [False, True])
def test_training_criterion_factory_moves_registered_state_and_preserves_legacy_facade(grouped):
    import train
    from core.losses.loss_fn import LossFunction
    config = configured_model(grouped=grouped)
    loss = train.build_training_criterion(config, torch.device("meta"))
    assert all(tensor.device.type == "meta" for tensor in loss.state_dict().values())
    if not grouped:
        assert type(loss) is LossFunction


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_warmup_visits_every_active_head_preserves_bn_and_clears_gradients(grouped, compiled):
    import train
    from common import build_model
    from test_grouped_loss_state import snapshot, assert_state_equal
    config = configured_model(grouped=grouped)
    model = build_model(config).train()
    loss = train.build_training_criterion(config, torch.device("cpu"))
    optimizer = train.build_optimizer(model, loss, config)
    before, loss_before = snapshot(model), snapshot(loss)
    dummy = torch.randn(1, 14, 32, 48)
    if compiled:
        model = torch.compile(model, backend="eager")
    observed = []
    handles = [p.register_hook(lambda grad: observed.append(grad.detach().clone()))
               for p in model.parameters()]
    try:
        train.warmup_model(model, dummy, optimizer, torch.device("cpu"), "fp32")
    finally:
        for handle in handles:
            handle.remove()
    assert len(observed) == len(list(model.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in observed)
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    assert_state_equal(getattr(model, "_orig_mod", model), before)
    assert_state_equal(loss, loss_before)


@pytest.mark.parametrize("classification,strategy,iqa,grouped", [
    ("gaussian", "oga", True, True),
    ("gaussian", "q_oga", False, True),
    ("binary", "oga", True, True),
    ("gaussian", "baseline", True, True),
    ("gaussian", "oga", True, False),
])
def test_real_cli_trains_validates_and_saves_grouped_or_legacy_criterion(tmp_path, classification, strategy, iqa, grouped):
    import json
    import train
    from test_grouped_targets import make_dataset
    dataset = make_dataset(tmp_path, classification=classification, grouped=grouped)
    config = configured_model(classification, strategy, iqa, grouped)
    config["data"] = dataset.config
    config["augmentation"] = {
        "p": 0., "rotation": {"use": False}, "scaling": {"use": False}, "translation": {"use": False}}
    root = Path(config["data"]["kitti"]["location"])
    (root / "pointcloud/000001.bin").write_bytes((root / "pointcloud/000000.bin").read_bytes())
    (root / "label/000001.txt").write_text((root / "label/000000.txt").read_text())
    val = tmp_path / "validation.txt"
    val.write_text("000001;kitti\n")
    config["train"].update(data=str(dataset.data_file), epochs=2, physical_batch_size=1, accumulation_steps=1, save_every=1)
    config["val"] = {"data": str(val), "physical_batch_size": 1}
    if strategy == "q_oga":
        config["loss"].update(quality_target="rotated_iou", quality_warmup_epochs=2)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    train.main(["--config", str(path), "--detector-root", str(ROOT / "detector"),
                "--output-root", str(tmp_path / "runs"), "--run-name", "synthetic", "--device", "cpu",
                "--num-workers", "0", "--precision", "fp32", "--max-train-batches", "1", "--max-val-batches", "1"])
    run = tmp_path / "runs/synthetic"
    payload = torch.load(run / "checkpoints/2epoch.pt", weights_only=False, map_location="cpu")
    assert payload["epoch"] == 2 and np.isfinite(payload["validation"]["loss"])
    keys = list(payload["criterion_state_dict"])
    if grouped:
        assert "group/car/loss" in payload["validation"]
        if strategy != "baseline":
            assert all(any(key.startswith(f"criteria.{name}.") for key in keys) for name in ("car", "ped_cyc"))
        if strategy == "q_oga":
            assert payload["criterion_state_dict"]["criteria.car.strategy.quality_epoch"] == 1
            assert payload["validation"]["quality_iou_mix"] == .5
    else:
        assert all(key.startswith("strategy.") for key in keys)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2


def test_cli_rejects_missing_group_definitions_before_loading_manifests(tmp_path):
    import json
    import train
    config = configured_model()
    del config["data"]["head_groups"]
    config["train"]["data"] = str(tmp_path / "missing_manifest.txt")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="nonempty data.head_groups"):
        train.main(["--config", str(path), "--detector-root", str(ROOT / "detector"),
                    "--output-root", str(tmp_path / "runs"), "--device", "cpu", "--precision", "fp32"])

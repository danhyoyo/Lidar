"""Actual focal/local hooks across supported grouped objectives."""

import pytest
import torch

from test_grouped_training import configured_model
from test_grouped_losses import tensors
from common import build_model
import train
from core.losses.iou_targets import compute_iou_targets


CASES = [
    ("gaussian", "baseline", True), ("gaussian", "oga", True),
    ("gaussian", "uwag", False), ("gaussian", "gw_qal", False),
    ("gaussian", "q_oga", False), ("gaussian", "exact_q_oga", False),
    ("binary", "baseline", True), ("binary", "oga", True),
    ("binary", "uwag", False),
]


def fixture(classification="gaussian", strategy="oga", iqa=True, local="eca", empty=False):
    config = configured_model(classification, "q_oga" if strategy == "exact_q_oga" else strategy, iqa)
    config["model"]["local_attention"] = local
    if strategy == "exact_q_oga":
        config["loss"].update(quality_target="rotated_iou", quality_warmup_epochs=4)
    _, target = tensors(classification, iqa, empty)
    target["groups"] = {name: {key: value.repeat(*([1] * (value.ndim - 2)), 2, 2)
                                for key, value in group.items()}
                        for name, group in target["groups"].items()}
    # Both groups retain ownership at the same cells, with different boxes.
    target["groups"]["ped_cyc"]["offset"][:, 0].add_(1.25)
    target["voxel"] = torch.randn(1, 14, 32, 48)
    return config, target


@pytest.mark.parametrize("local", ["none", "eca", "simam"])
@pytest.mark.parametrize("classification,strategy,iqa", CASES)
@pytest.mark.parametrize("empty", [False, True])
def test_real_shared_adapters_receive_finite_gradients_for_all_supported_objectives(local, classification, strategy, iqa, empty):
    torch.manual_seed(83)
    config, target = fixture(classification, strategy, iqa, local, empty)
    model = build_model(config).train()
    criterion = train.build_training_criterion(config, torch.device("cpu")).train()
    train.set_loss_epoch(criterion, 2)
    optimizer = train.build_optimizer(model, criterion, config)
    expected = {id(p) for module in (model, criterion) for p in module.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(actual) == len(expected) and set(actual) == expected
    prediction = model(target["voxel"])
    for name, pred in prediction["groups"].items():
        quality = compute_iou_targets(pred, target["groups"][name], method="rotated_iou")
        assert not quality.requires_grad and quality.grad_fn is None
    result = criterion(prediction, target)
    result["loss"].backward()
    hooks = {name: p for name, p in model.backbone.named_parameters()
             if "c4_context" in name or "c3_light_attention" in name}
    assert hooks and any(name.endswith("gamma") for name in hooks)
    for name, parameter in hooks.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
        if name.endswith(("gamma", "beta")):
            assert parameter.detach().abs().min() > 0
    if local != "none":
        assert any(name.endswith("beta") for name in hooks)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in criterion.parameters())
    optimizer.step()


@pytest.mark.parametrize("local", ["none", "eca", "simam"])
@pytest.mark.parametrize("strategy", ["baseline", "oga"])
def test_car_iqa_only_reaches_shared_adapters_and_its_own_quality_head(local, strategy):
    torch.manual_seed(84)
    config, target = fixture(strategy=strategy, local=local)
    model = build_model(config).eval()
    criterion = train.build_training_criterion(config, torch.device("cpu")).eval()
    prediction = model(target["voxel"])["groups"]["car"]
    selected = {key: value if key == "iou" else value.detach() for key, value in prediction.items()}
    criterion.criteria["car"](selected, target["groups"]["car"])["loss"].backward()
    hooks = [p for name, p in model.backbone.named_parameters()
             if "c4_context" in name or "c3_light_attention" in name]
    assert hooks and all(p.grad is not None and torch.isfinite(p.grad).all() for p in hooks)
    assert any(p.grad.abs().sum() > 0 for p in hooks)
    for name, parameter in model.grouped_header.heads["car"].named_parameters():
        assert (parameter.grad is not None) == name.startswith("iou.")
    assert all(p.grad is None for p in model.grouped_header.heads["ped_cyc"].parameters())


@pytest.mark.parametrize("local", ["none", "eca", "simam"])
def test_exact_qoga_with_separate_iqa_is_rejected_before_model_construction(local):
    config, _ = fixture(strategy="exact_q_oga", iqa=True, local=local)
    with pytest.raises(ValueError, match="IQA"):
        build_model(config)

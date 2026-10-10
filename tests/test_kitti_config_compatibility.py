"""Exercise KITTI configuration variants through dataset creation and a training step."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "tools" / "kitti_training_pipeline"),
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
]

from common import build_model, input_shape, create_experiment_config
from core.datasets.dataset import Dataset
from core.losses.loss_fn import LossFunction
from train import build_optimizer, build_scheduler, seed_everything, set_loss_epoch

BASE_CONFIG = json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8"))

TEST_VARIANTS = {
    "default_config": {},
    "oga_reparam_iqa": {
        "model": {"scale_gated_fpn": True, "c4_attention": "none", "c4_attention_scales": [],
                  "c4_attention_qk_norm": "none", "use_reparam": True, "header_use_iou": True},
        "loss": {"name": "oga", "use_iou": True},
    },
    "qoga_reparam": {
        "model": {"scale_gated_fpn": True, "c4_attention": "none", "c4_attention_scales": [],
                  "c4_attention_qk_norm": "none", "use_reparam": True, "header_use_iou": False},
        "loss": {"name": "q_oga", "use_iou": False},
    },
    "mobilepixor_legacy35": {
        "model": {"backbone": "mobilepixor", "scale_gated_fpn": False, "c4_attention": "none", "header_use_bn": False, "header_act": "relu"},
        "loss": {"name": "baseline"},
        "data": {"bev_encoding": {"name": "binary_slices"}},
    },
}


@pytest.mark.parametrize("variant_name", list(TEST_VARIANTS.keys()))
@pytest.mark.parametrize("target_backend", ["python", "numba"])
def test_config_supports_dataset_and_training_step(variant_name, target_backend, tmp_path):
    if target_backend == "numba":
        pytest.importorskip("numba")
    seed_everything(42)
    overrides = TEST_VARIANTS[variant_name]
    config = create_experiment_config(BASE_CONFIG, overrides)

    # Use a small rectangular crop with the configured resolution and Z bins.
    # Only paths and XY extents change; model, loss and augmentation stay intact.
    data = config["data"]
    geometry = data["kitti"]["geometry"]
    geometry.update(
        x_min=0.0,
        x_max=96 * geometry["x_res"],
        y_min=-32 * geometry["y_res"],
        y_max=32 * geometry["y_res"],
    )
    data["kitti"]["location"] = str(tmp_path)
    (tmp_path / "label").mkdir()
    (tmp_path / "pointcloud").mkdir()
    center_x = geometry["x_max"] / 2
    (tmp_path / "label" / "000000.txt").write_text(
        f"Car 1.5 1.6 3.7 {center_x} 0 -1 0.2\n", encoding="utf-8"
    )
    np.array([[center_x, 0.0, -1.0, 0.5]], dtype=np.float32).tofile(
        tmp_path / "pointcloud" / "000000.bin"
    )
    manifest = tmp_path / "frames.txt"
    manifest.write_text("000000;kitti\n", encoding="utf-8")

    samples = []
    for task in ("train", "validation"):
        dataset = Dataset(
            str(manifest), data, config["augmentation"],
            config["model"]["cls_encoding"], task, target_backend,
        )
        sample = dataset[0]
        assert tuple(sample["voxel"].shape) == input_shape(config)[1:]
        assert sample["reg_mask"].sum() > 0
        samples.append(sample)
    if config["augmentation"]["p"] == 0:
        for name in samples[0]:
            torch.testing.assert_close(samples[0][name], samples[1][name], rtol=0, atol=0)
    batch = default_collate(samples)

    model = build_model(config)
    criterion = LossFunction(config["model"]["cls_encoding"], config.get("loss"))
    set_loss_epoch(criterion, 0)
    optimizer = build_optimizer(model, criterion, config)
    scheduler = build_scheduler(optimizer, config, config["train"]["epochs"])
    outputs = model(batch["voxel"])
    for name in ("cls", "offset", "size", "yaw"):
        assert outputs[name].shape == batch[name].shape
    if config.get("loss", {}).get("use_iou", False):
        assert "iou" in outputs

    losses = criterion(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)
    optimizer.step()
    scheduler.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())

    model.eval()
    criterion.eval()
    with torch.no_grad():
        assert torch.isfinite(criterion(model(batch["voxel"]), batch)["loss"])

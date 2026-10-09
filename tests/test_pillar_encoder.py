"""Behavioral gates for a learned point-to-BEV encoder and its pipeline."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"),
               str(ROOT / "detector/core/datasets"),
               str(ROOT / "tools/kitti_training_pipeline")]

from core.bev_encoding import resolve_bev_encoding, resolve_input_channels

GEOMETRY = {"x_min": 0, "x_max": 16, "x_res": .5,
            "y_min": -8, "y_max": 8, "y_res": .5,
            "z_min": -2.5, "z_max": 1, "z_res": .1}


def packed(points, geometry=GEOMETRY, encoding="pillar32"):
    from core.datasets.utils_1.pillar_backend import prepare_pillars
    result = prepare_pillars(np.asarray(points, dtype=np.float32).reshape(-1, 4),
                             geometry, {"name": encoding})
    return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in result.items()}


@pytest.fixture(params=["pillar32", "pillar_rich", "pillar_rich_eca"])
def encoder_name(request):
    return request.param


def test_pillar_schema_identifies_learned_features_and_fixed_width():
    schema = resolve_bev_encoding({"name": "pillar32"}, GEOMETRY)
    assert schema.channels == resolve_input_channels({"name": "pillar32"}) == 32
    assert schema.input_shape == (1, 32, 32, 32)
    assert schema.backend == "torch"
    metadata = schema.semantic_metadata()
    assert metadata["learned_encoder"]["point_features"] == 10
    assert metadata["learned_encoder"]["point_limit"] is None
    assert metadata["layout"]["model"] == "packed_pillars"
    changed = resolve_bev_encoding({"name": "pillar32", "intensity_scale": 2}, GEOMETRY)
    assert changed.semantic_hash != schema.semantic_hash
    for options in ({"out_channels": 8}, {"out_channels": True}, {"backend": "numpy"},
                    {"max_points": 32}, {"version": 2}):
        with pytest.raises(ValueError):
            resolve_bev_encoding({"name": "pillar32", **options}, GEOMETRY)


def test_point_preparation_retains_subcell_offsets_and_filters_roi():
    cloud = [[.1, .1, -.5, .2], [.4, .1, .5, .8], [0, .1, 0, .4],
             [16, .1, 0, .4], [.2, .1, 1, .4], [.2, .1, 0, np.nan]]
    sample = packed(cloud)
    assert sample["features"].shape == (2, 10)
    torch.testing.assert_close(sample["features"][:, :4], torch.tensor(cloud[:2]))
    torch.testing.assert_close(sample["features"][:, 4:7],
                               torch.tensor([[-.15, 0, -.5], [.15, 0, .5]]))
    torch.testing.assert_close(sample["features"][:, 7:10],
                               torch.tensor([[-.15, -.15, .25], [.15, -.15, 1.25]]))
    assert sample["coords"].tolist() == [[0, 16, 0]]
    assert sample["pillar_indices"].tolist() == [0, 0]


def test_pillar_preparation_does_not_cap_points():
    points = np.tile([.1, .1, -.5, .2], (100, 1))
    sample = packed(points)
    assert sample["features"].shape[0] == 100
    assert sample["pillar_indices"].numel() == 100


def test_pillar_cells_match_rich8_at_float_grid_boundaries():
    from core.datasets.utils_1.preprocess import encode_bev
    geometry = dict(GEOMETRY, x_res=.1, y_res=.1)
    cloud = np.array([[.5, .5, -.5, .4], [.6, .6, -.5, .4], [1., 1., -.5, .4]],
                     dtype=np.float32)
    sample = packed(cloud, geometry)
    rich8 = encode_bev(cloud, geometry, {"name": "rich8"})
    expected = np.argwhere(rich8[:, :, :3].any(axis=2))
    np.testing.assert_array_equal(sample["coords"][:, 1:].numpy(), expected)


def test_collation_isolates_frames_and_supports_empty_samples():
    from core.datasets.dataset import collate_detector_batch
    samples = [{"voxel": packed([[.1, .1, -.5, .2]]), "cls": torch.zeros(3, 8, 8)},
               {"voxel": packed([]), "cls": torch.zeros(3, 8, 8)},
               {"voxel": packed([[.1, .1, -.5, .8]]), "cls": torch.zeros(3, 8, 8)}]
    batch = collate_detector_batch(samples)
    assert batch["voxel"]["batch_size"] == 3
    assert batch["voxel"]["coords"].tolist() == [[0, 16, 0], [2, 16, 0]]
    assert batch["voxel"]["pillar_indices"].tolist() == [0, 1]
    assert batch["cls"].shape == (3, 3, 8, 8)


def test_encoder_is_permutation_invariant_batch_isolated_and_differentiable():
    from core.datasets.dataset import collate_detector_batch
    from core.models.encoders.pillar import PillarEncoder
    cloud = [[.1, .1, -.5, .2], [.4, .1, .5, .8], [1.1, -.1, -.2, .6]]
    encoder = PillarEncoder(GEOMETRY).eval()
    a = encoder(packed(cloud))
    b = encoder(packed(cloud[::-1]))
    torch.testing.assert_close(a, b)
    assert a.shape == (1, 32, 32, 32)
    assert torch.count_nonzero(a[:, :, 0, 0]) == 0
    batch = collate_detector_batch([{"voxel": packed(cloud)}, {"voxel": packed([])}])
    output = encoder(batch["voxel"])
    torch.testing.assert_close(output[:1], a)
    assert torch.count_nonzero(output[1]) == 0
    output.square().sum().backward()
    assert encoder.linear.weight.grad is not None
    assert torch.count_nonzero(encoder.linear.weight.grad) > 0
    assert sum(p.numel() for p in encoder.parameters()) == 384


@pytest.mark.parametrize("cloud", [[], [[.1, .1, -.5, .2]]])
def test_empty_and_single_point_training_are_finite(cloud):
    from core.models.encoders.pillar import PillarEncoder
    encoder = PillarEncoder(GEOMETRY).train()
    output = encoder(packed(cloud))
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert encoder.linear.weight.grad is not None
    assert encoder.norm.num_batches_tracked == 0


@pytest.mark.parametrize("device,precision", [
    ("cpu", "fp32"),
    pytest.param("cuda", "fp32", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    pytest.param("cuda", "bf16", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_detector_loss_updates_encoder_and_preserves_prediction_shapes(device, precision, encoder_name):
    from core.datasets.dataset import collate_detector_batch
    from common import build_model, model_parameter_report
    from train import autocast_context, build_training_criterion, move_tensor_batch
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["data"]["kitti"]["geometry"] = GEOMETRY
    config["data"]["bev_encoding"] = {"name": encoder_name}
    config["model"].update(c4_attention="none", c4_attention_scales=[])
    device = torch.device(device)
    model = build_model(config).to(device).train()
    criterion = build_training_criterion(config, device)
    from core.datasets.dataset import Dataset
    dataset = Dataset.__new__(Dataset)
    dataset.config = config["data"]
    dataset.output_shape = [8, 8]
    dataset.cls_encoding = "gaussian"
    dataset.num_classes = 3
    dataset.out_size_factor = 4
    labels = dataset.get_label(torch.empty(0, 8), GEOMETRY)
    cloud = [[.1, .1, -.5, .2], [.4, .1, .5, .8], [1.1, -.1, -.2, .6]]
    batch = collate_detector_batch([{**labels, "voxel": packed(cloud, encoding=encoder_name)},
                                    {**labels, "voxel": packed(cloud[::-1], encoding=encoder_name)}])
    batch = move_tensor_batch(batch, device)
    initial = model.point_encoder.linear.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    with autocast_context(device, precision):
        predictions = model(batch["voxel"])
        loss = criterion(predictions, batch)["loss"]
    assert predictions["cls"].shape == (2, 3, 8, 8)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.point_encoder.linear.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(initial, model.point_encoder.linear.weight)
    assert model_parameter_report(model)["parameter_counts"]["point_encoder"] == {
        "pillar32": 384, "pillar_rich": 288, "pillar_rich_eca": 291}[encoder_name]


def test_packed_compile_warmup_preserves_bn_and_training_state(encoder_name):
    from common import build_model, dummy_model_input
    from train import warmup_model
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["data"]["kitti"]["geometry"] = GEOMETRY
    config["data"]["bev_encoding"] = {"name": encoder_name}
    config["model"].update(c4_attention="none", c4_attention_scales=[])
    model = build_model(config).train()
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters())
    device = torch.device("cpu")
    warmup_model(model, dummy_model_input(config, 2, device), optimizer, device, "fp32")
    assert model.training and model.point_encoder.training
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, saved[name]), name


def test_dense_only_deployment_rejects_packed_encoder(encoder_name):
    from common import validate_deployment_config
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["data"]["bev_encoding"] = {"name": encoder_name}
    with pytest.raises(ValueError, match="pillar32|packed"):
        validate_deployment_config(config, "ONNX export")


def test_actual_cli_training_checkpoint_and_ap_evaluation(tmp_path, encoder_name):
    from benchmark_fixtures import asset_fixture
    from common import write_json
    import train
    config, path, _, raw = asset_fixture(tmp_path)
    config["data"]["bev_encoding"] = {"name": encoder_name}
    config["train"].update(epochs=1, checkpoint_selection={"primary": "ap", "ap_every": 1,
                                                         "metric_mode": "local_bev"})
    config["evaluation"] = {"kitti_root": str(raw), "score_threshold": .05,
                             "nms_threshold": .1, "max_detections": 500}
    write_json(path, config)
    train.main(["--config", str(path), "--detector-root", str(ROOT / "detector"),
                "--output-root", str(tmp_path / "runs"), "--run-name", "pillar",
                "--device", "cpu", "--precision", "fp32", "--num-workers", "0"])
    run = tmp_path / "runs/pillar"
    checkpoint = torch.load(run / "checkpoints/last.pt", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert "point_encoder.linear.weight" in checkpoint["model_state_dict"]
    assert checkpoint["ap_selection"]["best_score"] is not None
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["validation"]["samples"] == 1
    from tools.kitti_training_pipeline.evaluate_kitti_bev import run_evaluation
    result = run_evaluation(name=encoder_name, backend="pytorch",
        model_path=run / "selected/best_ap.pt", config_path=run / "config.resolved.json",
        detector_root=ROOT / "detector", kitti_root=raw.parent,
        split_path=Path(config["val"]["data"]), output_path=run / "evaluation.json",
        device="cpu", warmup_frames=0, progress_every=0)
    assert result["status"] == "ok"
    assert result["data"]["input_bytes_fp32"] is None
    assert result["data"]["mean_input_bytes"] > 0


def test_comparison_configs_differ_only_by_encoding_and_have_expected_counts():
    from common import build_model
    from tools.kitti_training_pipeline.notebook_config import resolve_notebook_config
    configs = []
    for name, preset, expected in (("rich8", "ENCODER_RICH8", 648313),
                                    ("pillar32", "ENCODER_PILLAR32", 655609)):
        path = ROOT / f"configs/experiments/encoders/{name}.json"
        config = json.loads(path.read_text())
        resolved = resolve_notebook_config(ROOT, preset=preset, augmentation="config")
        assert config == resolved
        model = build_model(config)
        assert sum(p.numel() for p in model.parameters()) == expected
        config["data"].pop("bev_encoding")
        configs.append(config)
    assert configs[0] == configs[1]


def test_dense_shape_profiler_rejects_packed_input(encoder_name):
    from tools.benchmarks.profile_detector import profile_detector
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["data"]["bev_encoding"] = {"name": encoder_name}
    with pytest.raises(ValueError, match="packed|pillar32"):
        profile_detector(config, device="cpu", shape=(1, 32, 32, 32))


def test_bfloat16_pooling_backward_is_finite(encoder_name):
    from core.models.encoders.pillar import PillarEncoder
    encoder = PillarEncoder(GEOMETRY, {"name": encoder_name}).train()
    cloud = [[.1, .1, -.5, .2], [.4, .1, .5, .8], [1.1, -.1, -.2, .6]]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = encoder(packed(cloud, encoding=encoder_name))
        loss = output.float().square().sum()
    loss.backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(encoder.linear.weight.grad).all()


def test_packed_checkpoint_resume_reproduces_uninterrupted_training(tmp_path, monkeypatch, encoder_name):
    from benchmark_fixtures import asset_fixture
    from common import write_json
    import train
    config, path, _, _ = asset_fixture(tmp_path)
    config["data"]["bev_encoding"] = {"name": encoder_name}
    config["train"].update(epochs=2, checkpoint_selection={"primary": "loss"})
    write_json(path, config)
    args = ["--config", str(path), "--detector-root", str(ROOT / "detector"),
            "--output-root", str(tmp_path / "runs"), "--device", "cpu", "--precision", "fp32",
            "--num-workers", "0"]
    train.main([*args, "--run-name", "full"])
    original_validate = train.validate
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("stop after the retained first epoch")
        return original_validate(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(train, "validate", interrupt_second_epoch)
        with pytest.raises(KeyboardInterrupt):
            train.main([*args, "--run-name", "resumed"])
    retained = tmp_path / "runs/resumed/checkpoints/last.pt"
    assert torch.load(retained, weights_only=True)["epoch"] == 1
    train.main([*args, "--run-name", "resumed", "--resume", str(retained)])
    full = torch.load(tmp_path / "runs/full/checkpoints/last.pt", weights_only=True)
    resumed = torch.load(retained, weights_only=True)
    for key, value in full["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][key]), key
    for key in full["validation"]:
        if key != "seconds":
            assert full["validation"][key] == resumed["validation"][key], key


def test_variable_length_pillars_work_with_spawn_dataloader_workers(tmp_path, encoder_name):
    from benchmark_fixtures import asset_fixture
    from core.datasets.dataset import Dataset, collate_detector_batch
    config, _, _, _ = asset_fixture(tmp_path)
    config["data"]["bev_encoding"] = {"name": encoder_name}
    dataset = Dataset(config["train"]["data"], config["data"], config["augmentation"],
                      "gaussian", "validation", "python")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2,
                                        multiprocessing_context="spawn", timeout=20,
                                        collate_fn=collate_detector_batch)
    batch = next(iter(loader))
    assert batch["voxel"]["batch_size"] == 2
    assert batch["voxel"]["coords"][:, 0].unique().tolist() == [0, 1]
    assert batch["voxel"]["features"].shape == (48, 10)

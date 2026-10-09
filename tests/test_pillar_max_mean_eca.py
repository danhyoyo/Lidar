"""Local max/mean gating must retain statistics, pooling precision and identity."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"),
               str(ROOT / "detector/core/datasets"),
               str(ROOT / "tools/kitti_training_pipeline")]
from core.bev_encoding import resolve_bev_encoding
from core.datasets.dataset import collate_detector_batch
from core.datasets.utils_1.pillar_backend import prepare_pillars
from core.models.encoders.pillar import PillarEncoder

GEOMETRY = {"x_min": 0, "x_max": 16, "x_res": .5,
            "y_min": -8, "y_max": 8, "y_res": .5,
            "z_min": -2.5, "z_max": 1, "z_res": .1}
OPTIONS = {"name": "pillar_rich", "pooling": "max_mean_eca", "eca_kernel_size": 3}
CLOUD = [[.1, .1, -.5, .2], [.4, .1, .5, .8], [1.1, -.1, -.2, .6]]


def packed(cloud, options=OPTIONS):
    arrays = prepare_pillars(np.asarray(cloud, dtype=np.float32).reshape(-1, 4), GEOMETRY, options)
    return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in arrays.items()}


def reference(encoder, value):
    h = F.relu(encoder.norm(encoder.linear(value["features"])))
    maximum, means = [], []
    for index in range(len(value["coords"])):
        local = h[value["pillar_indices"] == index].float()
        maximum.append(local.max(dim=0).values)
        means.append(local.mean(dim=0))
    maximum, means = torch.stack(maximum), torch.stack(means)
    with torch.autocast(device_type=means.device.type, enabled=False):
        alpha = F.conv1d(means[:, None], encoder.eca.weight.float(),
                         padding=encoder.eca.kernel_size[0] // 2).squeeze(1).sigmoid()
    learned = (alpha * maximum + (1 - alpha) * means).to(h.dtype)
    pooled = torch.cat((value["rich_features"].to(h.dtype), learned), dim=1)
    output = pooled.new_zeros((value["batch_size"], 32, encoder.height, encoder.width))
    for index, (batch, y, x) in enumerate(value["coords"].tolist()):
        output[batch, :, y, x] = pooled[index]
    return output


def test_pooling_identity_keeps_historical_default_and_describes_local_gate():
    old = resolve_bev_encoding({"name": "pillar_rich"}, GEOMETRY)
    explicit = resolve_bev_encoding({"name": "pillar_rich", "pooling": "max"}, GEOMETRY)
    assert old.semantic_hash == explicit.semantic_hash
    assert old.semantic_metadata()["learned_encoder"]["pooling"] == "max"
    new = resolve_bev_encoding(OPTIONS, GEOMETRY)
    assert new.is_packed and new.channels == 32
    assert new.semantic_hash != old.semantic_hash
    assert new.semantic_metadata()["learned_encoder"]["pooling"] == "max_mean_eca"
    assert new.semantic_hash == resolve_bev_encoding({"name": "pillar_rich", "pooling": "max_mean_eca"}, GEOMETRY).semantic_hash
    assert new.semantic_hash != resolve_bev_encoding({**OPTIONS, "eca_kernel_size": 5}, GEOMETRY).semantic_hash


@pytest.mark.parametrize("options", [
    {"pooling": "other"}, {"pooling": []}, {"pooling": "max", "eca_kernel_size": 3},
    {"pooling": "max_mean_eca", "eca_kernel_size": 2},
    {"pooling": "max_mean_eca", "eca_kernel_size": True},
    {"pooling": "max_mean_eca", "eca_kernel_size": 3.0},
    {"pooling": "max_mean_eca", "eca_kernel_size": 0},
])
def test_invalid_pooling_and_gate_options_are_rejected(options):
    with pytest.raises(ValueError):
        resolve_bev_encoding({"name": "pillar_rich", **options}, GEOMETRY)


def test_pillar32_does_not_silently_accept_hybrid_gate():
    with pytest.raises(ValueError):
        resolve_bev_encoding({"name": "pillar32", "pooling": "max_mean_eca"}, GEOMETRY)


@pytest.mark.parametrize("kernel", [1, 3, 5])
def test_local_gate_matches_reference_output_and_parameter_gradients(kernel):
    options = {**OPTIONS, "eca_kernel_size": kernel}
    encoder = PillarEncoder(GEOMETRY, options).eval()
    torch.manual_seed(9)
    with torch.no_grad():
        encoder.eca.weight.normal_(0, .3)
    other = copy.deepcopy(encoder)
    value = packed(CLOUD, options)
    actual = encoder(value)
    expected = reference(other, value)
    torch.testing.assert_close(actual, expected)
    weights = torch.linspace(.1, 1, actual.numel()).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for (name, parameter), (_, reference_parameter) in zip(encoder.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, msg=name)
    assert actual.is_contiguous()
    assert sum(p.numel() for p in encoder.parameters()) == 288 + kernel


def test_gating_is_permutation_invariant_and_independent_of_other_frames():
    encoder = PillarEncoder(GEOMETRY, OPTIONS).eval()
    with torch.no_grad():
        encoder.eca.weight.fill_(.4)
    single = encoder(packed(CLOUD))
    batch = collate_detector_batch([{"voxel": packed(CLOUD[::-1])}, {"voxel": packed([])},
                                   {"voxel": packed([[8.1, 2.1, -.4, .9]])}])["voxel"]
    output = encoder(batch)
    torch.testing.assert_close(output[0], single[0])
    assert not output[1].count_nonzero()
    assert output.is_contiguous()
    value = packed(CLOUD)
    torch.testing.assert_close(single[0, :8, value["coords"][:, 1], value["coords"][:, 2]].T,
                               value["rich_features"])


@pytest.mark.parametrize("cloud", [[], [[.1, .1, -.5, .2]]])
def test_empty_and_single_point_backward_connects_gate_and_does_not_update_bn(cloud):
    encoder = PillarEncoder(GEOMETRY, OPTIONS).train()
    output = encoder(packed(cloud))
    assert torch.isfinite(output).all()
    output.sum().backward()
    for parameter in encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert encoder.norm.num_batches_tracked == 0
    if cloud:
        old = PillarEncoder(GEOMETRY, {"name": "pillar_rich"}).eval()
        old.load_state_dict({k: v for k, v in encoder.state_dict().items() if not k.startswith("eca.")})
        torch.testing.assert_close(output, old(packed(cloud)))


def test_gate_can_distinguish_equal_max_with_different_learned_means():
    z_a = [-2, -1.75, -1.5, -.75, -.25, .25]
    z_b = [-2, -1.75, -1.25, -1, -.25, .25]
    encoder = PillarEncoder(GEOMETRY, OPTIONS).eval()
    with torch.no_grad():
        encoder.linear.weight.zero_()
        encoder.linear.weight[:, 2] = 1
        encoder.norm.bias.fill_(1.3)
        encoder.eca.weight.zero_()
    a = encoder(packed([[.1, .1, z, .5] for z in z_a]))
    b = encoder(packed([[.1, .1, z, .5] for z in z_b]))
    torch.testing.assert_close(a[:, :8], b[:, :8])
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_bfloat16_many_points_mean_matches_fp32_reference_and_backward_is_finite():
    torch.manual_seed(4)
    encoder = PillarEncoder(GEOMETRY, OPTIONS).train()
    cloud = np.tile(np.asarray(CLOUD[:2], dtype=np.float32), (1024, 1))
    value = packed(cloud)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = encoder(value)
        # Compute the independent reference with the same point BN batch stats.
        expected = reference(encoder, value)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, expected)
    output.float().square().mean().backward()
    assert encoder.eca.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_cuda_gate_matches_reference_and_receives_finite_gradients(precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    encoder = PillarEncoder(GEOMETRY, OPTIONS).cuda().train()
    value = {k: v.cuda() if torch.is_tensor(v) else v for k, v in packed(CLOUD).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        output = encoder(value)
        expected = reference(encoder, value)
    torch.testing.assert_close(output, expected)
    output.float().square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_preset_config_model_counts_and_checkpoint_identity_are_consistent():
    from common import build_model, checkpoint_identity, generate_run_name, validate_evaluation_checkpoint
    from notebook_config import resolve_notebook_config
    new = resolve_notebook_config(ROOT, preset="ENCODER_PILLAR_RICH_MAX_MEAN_ECA", augmentation="config")
    saved = json.loads((ROOT / "configs/experiments/encoders/pillar_rich_max_mean_eca.json").read_text())
    assert new == saved
    model = build_model(new)
    assert sum(p.numel() for p in model.parameters()) == 655516
    old = resolve_notebook_config(ROOT, preset="ENCODER_PILLAR_RICH", augmentation="config")
    assert generate_run_name(old) != generate_run_name(new)
    with pytest.raises(ValueError, match="identity"):
        validate_evaluation_checkpoint({"checkpoint_identity": checkpoint_identity(old)}, new)
    left, right = copy.deepcopy(old), copy.deepcopy(new)
    left["data"].pop("bev_encoding"); right["data"].pop("bev_encoding")
    assert left == right


def test_pooling_variant_names_stay_unique_with_default_backbone():
    from common import generate_run_name
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["data"]["bev_encoding"] = {"name": "pillar_rich"}
    old_name = generate_run_name(config)
    config["data"]["bev_encoding"] = OPTIONS.copy()
    assert generate_run_name(config) != old_name
    first = generate_run_name(config)
    config["data"]["bev_encoding"]["eca_kernel_size"] = 5
    assert generate_run_name(config) != first


@pytest.mark.parametrize("bad", ["negative_point", "large_point", "negative_coord", "large_batch", "large_y", "large_x"])
def test_combined_bounds_check_rejects_invalid_packed_input(bad):
    encoder = PillarEncoder(GEOMETRY, OPTIONS).eval()
    value = packed(CLOUD)
    if bad == "negative_point":
        value["pillar_indices"][0] = -1
    elif bad == "large_point":
        value["pillar_indices"][0] = len(value["coords"])
    else:
        axis, number = {"negative_coord": (1, -1), "large_batch": (0, 1),
                        "large_y": (1, encoder.height), "large_x": (2, encoder.width)}[bad]
        value["coords"][0, axis] = number
    with pytest.raises(ValueError, match="outside"):
        encoder(value)


def test_random_weight_benchmark_reports_boundaries_and_keeps_config_unchanged():
    from tools.benchmarks.benchmark_pillar_encoders import benchmark_models
    config = json.loads((ROOT / "configs/experiments/encoders/rich8.json").read_text())
    config["data"]["kitti"]["geometry"] = GEOMETRY.copy()
    before = copy.deepcopy(config)
    report = benchmark_models(config, np.asarray(CLOUD, dtype=np.float32),
                              device="cpu", warmup=0, iterations=1)
    assert config == before
    assert "Excludes file I/O, decode/NMS, AP" in report["scope"]
    assert report["results"]["pillar_rich_max_mean_eca"]["parameter_counts"]["total_detector"] == 655516
    assert report["results"]["rich8"]["model_ratio_to_rich8"] == 1
    for result in report["results"].values():
        assert result["model_only"]["samples"] == 1
        assert result["latency"]["input_to_outputs"]["mean_ms"] > 0
    json.dumps(report, allow_nan=False)


def test_gate_survives_cli_train_resume_and_evaluation(tmp_path, monkeypatch):
    from benchmark_fixtures import asset_fixture
    from common import write_json
    from tools.kitti_training_pipeline.evaluate_kitti_bev import run_evaluation
    import train
    config, path, _, raw = asset_fixture(tmp_path)
    config["data"]["bev_encoding"] = OPTIONS.copy()
    config["train"].update(epochs=2, checkpoint_selection={"primary": "loss"})
    write_json(path, config)
    args = ["--config", str(path), "--detector-root", str(ROOT / "detector"),
            "--output-root", str(tmp_path / "runs"), "--run-name", "gated",
            "--device", "cpu", "--precision", "fp32", "--num-workers", "0"]
    original_validate = train.validate
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("exercise resume from the first saved epoch")
        return original_validate(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(train, "validate", interrupt_second_epoch)
        with pytest.raises(KeyboardInterrupt):
            train.main(args)
    run = tmp_path / "runs/gated"
    last = run / "checkpoints/last.pt"
    checkpoint = torch.load(last, weights_only=True)
    assert "point_encoder.eca.weight" in checkpoint["model_state_dict"]
    assert checkpoint["epoch"] == 1
    train.main([*args, "--resume", str(last)])
    resumed = torch.load(last, weights_only=True)
    assert resumed["epoch"] == 2
    assert torch.isfinite(resumed["model_state_dict"]["point_encoder.eca.weight"]).all()
    result = run_evaluation(name="gated", backend="pytorch", model_path=last,
                            config_path=run / "config.resolved.json", detector_root=ROOT / "detector",
                            kitti_root=raw.parent, split_path=Path(config["val"]["data"]),
                            device="cpu", warmup_frames=0, progress_every=0)
    assert result["status"] == "ok"
    assert result["model"]["parameter_counts"]["point_encoder"] == 291
    assert result["data"]["checkpoint_identity"]["encoding"]["metadata"]["learned_encoder"]["pooling"] == "max_mean_eca"

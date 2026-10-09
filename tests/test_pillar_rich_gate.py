"""Rich-conditioned gates preserve the baseline at initialization and learn locally."""

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
from core.bev_encoding import minimum_channels, resolve_bev_encoding, resolve_input_channels
from core.datasets.dataset import collate_detector_batch
from core.datasets.utils_1.pillar_backend import prepare_pillars
from core.models.encoders.pillar import PillarEncoder

GEOMETRY = {"x_min": 0, "x_max": 16, "x_res": .5,
            "y_min": -8, "y_max": 8, "y_res": .5,
            "z_min": -2.5, "z_max": 1, "z_res": .1}
OPTIONS = {"name": "pillar_rich_gate"}
CLOUD = [[.1, .1, -.5, .2], [.4, .1, .5, .8], [1.1, -.1, -.2, .6]]


def packed(cloud, options=OPTIONS):
    arrays = prepare_pillars(np.asarray(cloud, dtype=np.float32).reshape(-1, 4), GEOMETRY, options)
    return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in arrays.items()}


def reference(encoder, value):
    embedded = F.relu(encoder.norm(encoder.linear(value["features"])))
    output = embedded.new_zeros((value["batch_size"], 32, encoder.height, encoder.width))
    for index, (batch, y, x) in enumerate(value["coords"].tolist()):
        maximum = embedded[value["pillar_indices"] == index].max(dim=0).values
        rich = value["rich_features"][index].to(embedded.dtype)
        residual = encoder.gate(torch.cat((maximum, rich)))
        learned = (maximum.float() + residual.float()).to(embedded.dtype)
        output[batch, :, y, x] = torch.cat((rich, learned))
    return output


def activate_gate(encoder):
    with torch.no_grad():
        encoder.gate[0].weight.fill_(.02)
        encoder.gate[0].bias.fill_(.1)
        encoder.gate[2].weight.copy_(torch.linspace(-.3, .3, 384).reshape(24, 16))
        encoder.gate[2].bias.copy_(torch.linspace(-.1, .1, 24))


def test_schema_preserves_baseline_identity_and_describes_fixed_gate():
    baseline = resolve_bev_encoding({"name": "pillar_rich"}, GEOMETRY)
    assert baseline.semantic_hash == "e05ae9866d3e51a4ead83821ea5a1144e6f240af860071e08e50655cc7814d60"
    schema = resolve_bev_encoding(OPTIONS, GEOMETRY)
    assert schema.is_packed and schema.is_rich_pillar and schema.pooling == "max"
    assert minimum_channels(schema.name) == resolve_input_channels(OPTIONS) == 32
    assert schema.semantic_hash != baseline.semantic_hash
    assert schema.semantic_hash == resolve_bev_encoding({**OPTIONS, "pooling": "max"}, GEOMETRY).semantic_hash
    learned = schema.semantic_metadata()["learned_encoder"]
    assert learned["pooling"] == "max" and "mean_accumulation" not in learned
    gate = learned["gate"]
    assert gate["type"] == "rich_conditioned_channel_gate"
    assert gate["descriptor"] == "concat(learned_max24, rich8)"
    assert gate["hidden_channels"] == 16 and gate["initial_scale"] == 1


@pytest.mark.parametrize("options", [{"pooling": "max_mean_eca"}, {"pooling": []},
                                      {"eca_kernel_size": 3}, {"gate_hidden_channels": 8}])
@pytest.mark.parametrize("name", ["pillar32", "pillar_rich", "pillar_rich_gate"])
def test_removed_or_unimplemented_options_are_rejected(name, options):
    with pytest.raises(ValueError):
        resolve_bev_encoding({"name": name, **options}, GEOMETRY)


def test_retired_encoder_and_presets_are_not_silently_reinterpreted():
    from notebook_config import resolve_notebook_config
    with pytest.raises(ValueError, match="unsupported BEV encoding"):
        resolve_bev_encoding({"name": "pillar_rich_eca"}, GEOMETRY)
    for preset in ("ENCODER_PILLAR_RICH_ECA", "ENCODER_PILLAR_RICH_MAX_MEAN_ECA"):
        with pytest.raises(ValueError):
            resolve_notebook_config(ROOT, preset=preset)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_initial_gate_matches_baseline_exactly_and_preserves_full_model_rng(precision):
    from common import build_model
    from notebook_config import resolve_notebook_config
    torch.manual_seed(42)
    baseline = build_model(resolve_notebook_config(ROOT, preset="ENCODER_PILLAR_RICH", augmentation="config")).eval()
    rng_after_baseline = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    gated = build_model(resolve_notebook_config(ROOT, preset="ENCODER_PILLAR_RICH_GATE", augmentation="config")).eval()
    assert torch.equal(rng_after_baseline, torch.random.get_rng_state())
    for key, parameter in baseline.state_dict().items():
        torch.testing.assert_close(parameter, gated.state_dict()[key], rtol=0, atol=0, msg=key)
    # Small geometry is sufficient to check point encoder equality in AMP too.
    torch.manual_seed(42)
    left = PillarEncoder(GEOMETRY, {"name": "pillar_rich"}).eval()
    torch.manual_seed(42)
    right = PillarEncoder(GEOMETRY, OPTIONS).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
        torch.testing.assert_close(right(packed(CLOUD)), left(packed(CLOUD)), rtol=0, atol=0)
    assert sum(p.numel() for p in right.gate.parameters()) == 936
    assert sum(p.numel() for p in right.parameters()) == 1224


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_nonzero_gate_matches_per_pillar_reference_output_and_gradients(precision):
    encoder = PillarEncoder(GEOMETRY, OPTIONS).eval()
    activate_gate(encoder)
    other = copy.deepcopy(encoder)
    value = packed(CLOUD)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
        actual = encoder(value)
        expected = reference(other, value)
    torch.testing.assert_close(actual, expected, rtol=.02 if precision == "bf16" else 1e-5,
                               atol=.01 if precision == "bf16" else 1e-6)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    for (name, parameter), (_, ref_parameter) in zip(encoder.named_parameters(), other.named_parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, ref_parameter.grad,
                                   rtol=.03 if precision == "bf16" else 1e-5,
                                   atol=.02 if precision == "bf16" else 1e-5, msg=name)
    assert encoder.gate[0].weight.grad.abs().sum() > 0
    assert encoder.gate[2].weight.grad.abs().sum() > 0
    assert actual.is_contiguous()


def test_gate_is_point_permutation_invariant_frame_independent_and_keeps_rich8():
    encoder = PillarEncoder(GEOMETRY, OPTIONS).eval()
    activate_gate(encoder)
    single = encoder(packed(CLOUD))
    batch = collate_detector_batch([{"voxel": packed(CLOUD[::-1])}, {"voxel": packed([])},
                                   {"voxel": packed([[8.1, 2.1, -.4, .9]])}])["voxel"]
    output = encoder(batch)
    torch.testing.assert_close(output[0], single[0])
    assert not output[1].count_nonzero()
    value = packed(CLOUD)
    torch.testing.assert_close(single[0, :8, value["coords"][:, 1], value["coords"][:, 2]].T,
                               value["rich_features"], rtol=0, atol=0)


def test_rich_density_changes_gate_when_learned_max_is_identical():
    encoder = PillarEncoder(GEOMETRY, OPTIONS).eval()
    with torch.no_grad():
        encoder.linear.weight.zero_()
        encoder.norm.bias.fill_(1)
        encoder.gate[0].weight.zero_()
        encoder.gate[0].weight[:, 31] = 1  # rich8 log-density, after max24
        encoder.gate[0].bias.zero_()
        encoder.gate[2].weight.fill_(1)
    cloud = np.asarray(CLOUD[:2], dtype=np.float32)
    a = encoder(packed(cloud))
    b = encoder(packed(np.repeat(cloud, 2, axis=0)))
    torch.testing.assert_close(a[:, :7], b[:, :7])
    assert not torch.equal(a[:, 7], b[:, 7])
    assert not torch.equal(a[:, 8:], b[:, 8:])


@pytest.mark.parametrize("cloud", [[], [[.1, .1, -.5, .2]]])
def test_empty_and_singleton_backward_connects_every_parameter_without_updating_bn(cloud):
    encoder = PillarEncoder(GEOMETRY, OPTIONS).train()
    output = encoder(packed(cloud))
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert encoder.norm.num_batches_tracked == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_cuda_gate_backward_is_finite(precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    encoder = PillarEncoder(GEOMETRY, OPTIONS).cuda().train()
    value = {k: v.cuda() if torch.is_tensor(v) else v for k, v in packed(CLOUD).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        output = encoder(value)
    output.float().square().sum().backward()
    assert encoder.gate[2].bias.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_preset_matches_file_parameter_count_and_rejects_other_checkpoint_identity():
    from common import build_model, checkpoint_identity, generate_run_name, validate_evaluation_checkpoint
    from notebook_config import resolve_notebook_config
    new = resolve_notebook_config(ROOT, preset="ENCODER_PILLAR_RICH_GATE", augmentation="config")
    saved = json.loads((ROOT / "configs/experiments/encoders/pillar_rich_gate.json").read_text())
    assert new == saved
    assert sum(p.numel() for p in build_model(new).parameters()) == 656449
    old = resolve_notebook_config(ROOT, preset="ENCODER_PILLAR_RICH", augmentation="config")
    assert generate_run_name(old) != generate_run_name(new)
    assert "-pillar_rich_gate-" in generate_run_name(new)
    with pytest.raises(ValueError, match="identity"):
        validate_evaluation_checkpoint({"checkpoint_identity": checkpoint_identity(old)}, new)
    left, right = copy.deepcopy(old), copy.deepcopy(new)
    left["data"].pop("bev_encoding"); right["data"].pop("bev_encoding")
    assert left == right


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
    assert report["results"]["pillar_rich_gate"]["parameter_counts"]["total_detector"] == 656449
    assert report["results"]["pillar_rich_gate"]["encoding"]["name"] == "pillar_rich_gate"
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
    assert "point_encoder.gate.2.weight" in checkpoint["model_state_dict"]
    assert checkpoint["epoch"] == 1
    train.main([*args, "--resume", str(last)])
    resumed = torch.load(last, weights_only=True)
    assert resumed["epoch"] == 2
    assert torch.isfinite(resumed["model_state_dict"]["point_encoder.gate.2.weight"]).all()
    result = run_evaluation(name="gated", backend="pytorch", model_path=last,
                            config_path=run / "config.resolved.json", detector_root=ROOT / "detector",
                            kitti_root=raw.parent, split_path=Path(config["val"]["data"]),
                            device="cpu", warmup_frames=0, progress_every=0)
    assert result["status"] == "ok"
    assert result["model"]["parameter_counts"]["point_encoder"] == 1224
    assert result["data"]["checkpoint_identity"]["encoding"]["metadata"]["learned_encoder"]["gate"]["type"] == "rich_conditioned_channel_gate"

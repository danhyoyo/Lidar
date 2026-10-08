"""Hybrid encoding must preserve rich8 while learning complementary features."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets"),
               str(ROOT / "tools/kitti_training_pipeline")]
from core.bev_encoding import resolve_bev_encoding
from core.datasets.utils_1.pillar_backend import prepare_pillars
from core.datasets.utils_1.preprocess import encode_bev
from core.datasets.dataset import collate_detector_batch
from core.models.encoders.pillar import PillarEncoder

GEOMETRY = {"x_min": 0, "x_max": 16, "x_res": .5,
            "y_min": -8, "y_max": 8, "y_res": .5,
            "z_min": -2.5, "z_max": 1, "z_res": .1}


def packed(points, geometry=GEOMETRY, **options):
    result = prepare_pillars(np.asarray(points, dtype=np.float32).reshape(-1, 4),
                             geometry, {"name": "pillar_rich", **options})
    return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in result.items()}


def test_hybrid_schema_describes_channel_order_and_density_identity():
    schema = resolve_bev_encoding({"name": "pillar_rich"}, GEOMETRY)
    rich = resolve_bev_encoding({"name": "rich8"}, GEOMETRY)
    assert schema.channels == 32 and schema.is_packed
    assert schema.channel_names[:8] == rich.channel_names
    assert schema.channel_names[8:] == tuple(f"learned_feature_{i}" for i in range(24))
    metadata = schema.semantic_metadata()
    assert metadata["learned_encoder"]["out_channels"] == 24
    assert metadata["handcrafted_encoder"] == rich.semantic_metadata()
    changed = resolve_bev_encoding({"name": "pillar_rich", "density_norm": 64}, GEOMETRY)
    assert changed.semantic_hash != schema.semantic_hash
    assert schema.semantic_hash != resolve_bev_encoding({"name": "pillar32"}, GEOMETRY).semantic_hash
    for options in ({"out_channels": 24}, {"out_channels": True}, {"backend": "numpy"}):
        with pytest.raises(ValueError, match="pillar_rich"):
            resolve_bev_encoding({"name": "pillar_rich", **options}, GEOMETRY)


@pytest.mark.parametrize("scale,density", [(1, 32), (2, 8), (.5, 64)])
def test_compact_statistics_match_rich8_exactly_on_random_points_and_boundaries(scale, density):
    rng = np.random.default_rng(7)
    cloud = rng.uniform([0, -8, -2.5, -.2], [16, 8, 1, 1.2], (2000, 4)).astype(np.float32)
    boundaries = np.array([[.5, .5, -2.5 + 3.5/3, .3], [.6, .6, -2.5 + 7/3, .8],
                           [0, 0, 0, .5], [16, 0, 0, .5], [1, 0, 0, np.nan]], dtype=np.float32)
    cloud = np.concatenate((cloud, boundaries))
    geometry = dict(GEOMETRY, x_res=.1, y_res=.1)
    sample = packed(cloud, geometry, intensity_scale=scale, density_norm=density)
    rich = encode_bev(cloud, geometry, {"name": "rich8", "intensity_scale": scale, "density_norm": density})
    coords = sample["coords"].numpy()
    np.testing.assert_array_equal(sample["rich_features"].numpy(), rich[coords[:, 1], coords[:, 2]])
    torch.testing.assert_close(sample["features"][:, :4],
        torch.from_numpy(prepare_pillars(cloud, geometry, {"name": "pillar32", "intensity_scale": scale})["features"][:, :4]))


def test_fusion_preserves_rich8_and_is_batch_isolated_permutation_invariant():
    cloud = [[.1, .1, -.5, .2], [.4, .1, .5, .8], [1.1, -.1, -.2, .6]]
    encoder = PillarEncoder(GEOMETRY, {"name": "pillar_rich"}).eval()
    samples = [packed(cloud), packed([]), packed(cloud[::-1])]
    batch = collate_detector_batch([{"voxel": s} for s in samples])
    assert batch["voxel"]["rich_features"].shape == (4, 8)
    output = encoder(batch["voxel"])
    assert output.shape == (3, 32, 32, 32)
    rich = torch.from_numpy(encode_bev(np.array(cloud, dtype=np.float32), GEOMETRY,
                                     {"name": "rich8"})).permute(2, 0, 1)
    torch.testing.assert_close(output[0, :8], rich, rtol=0, atol=0)
    torch.testing.assert_close(output[0], output[2])
    assert torch.count_nonzero(output[1]) == 0
    output[:, 8:].square().sum().backward()
    assert encoder.linear.weight.grad.abs().sum() > 0
    assert sum(p.numel() for p in encoder.parameters()) == 288


def test_hybrid_retains_density_when_max_pooled_features_cannot_count_duplicates():
    cloud = np.array([[.1, .1, -.5, .2], [.4, .1, .5, .8]], dtype=np.float32)
    encoder = PillarEncoder(GEOMETRY, {"name": "pillar_rich"}).eval()
    a, b = encoder(packed(cloud)), encoder(packed(np.repeat(cloud, 2, axis=0)))
    torch.testing.assert_close(a[:, 8:], b[:, 8:])
    assert not torch.equal(a[:, 7], b[:, 7])


def test_hybrid_bfloat16_preserves_rich8_at_expected_precision():
    cloud = np.array([[.1, .1, -.5, .2], [.4, .1, .5, .8]], dtype=np.float32)
    encoder = PillarEncoder(GEOMETRY, {"name": "pillar_rich"}).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = encoder(packed(cloud))
    rich = torch.from_numpy(encode_bev(cloud, GEOMETRY, {"name": "rich8"})).permute(2, 0, 1)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output[0, :8], rich.to(torch.bfloat16), rtol=0, atol=0)


def test_hybrid_statistics_follow_training_augmentation(tmp_path):
    from benchmark_fixtures import asset_fixture
    from core.datasets.dataset import Dataset
    config, _, processed, _ = asset_fixture(tmp_path)
    config["data"]["bev_encoding"] = {"name": "pillar_rich"}
    dataset = Dataset(config["train"]["data"], config["data"], config["augmentation"],
                      "gaussian", "train", "python")
    delta = np.array([.75, .25, .25], dtype=np.float32)

    def translated(points, boxes):
        points, boxes = points.copy(), boxes.copy()
        points[:, :3] += delta
        boxes[:, 3:6] += delta
        return points, boxes

    dataset.augmentation_mode = "one_of"
    dataset.augment = translated
    value = dataset[0]["voxel"]
    points = np.fromfile(processed / "pointcloud/000000.bin", dtype=np.float32).reshape(-1, 4)
    points[:, :3] += delta
    geometry = config["data"]["kitti"]["geometry"]
    expected = encode_bev(points, geometry, {"name": "rich8"})
    coords = value["coords"].numpy()
    np.testing.assert_array_equal(value["rich_features"].numpy(), expected[coords[:, 1], coords[:, 2]])
    torch.testing.assert_close(value["features"], packed(points, geometry)["features"])


@pytest.mark.parametrize("cloud", [[], [[.1, .1, -.5, .2]]])
def test_hybrid_empty_and_singleton_backward_keep_bn_unchanged(cloud):
    encoder = PillarEncoder(GEOMETRY, {"name": "pillar_rich"}).train()
    output = encoder(packed(cloud))
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert encoder.linear.weight.grad is not None
    assert encoder.norm.num_batches_tracked == 0


def test_hybrid_rejects_missing_or_misaligned_stats_and_mixed_encoder_batches():
    encoder = PillarEncoder(GEOMETRY, {"name": "pillar_rich"})
    sample = packed([[.1, .1, -.5, .2]])
    for stats in (None, torch.zeros(2, 8), torch.zeros(1, 7)):
        value = dict(sample)
        if stats is None:
            value.pop("rich_features")
        else:
            value["rich_features"] = stats
        with pytest.raises(ValueError, match="rich_features"):
            encoder(value)
    plain = dict(sample)
    plain.pop("rich_features")
    with pytest.raises(ValueError, match="mix"):
        collate_detector_batch([{"voxel": plain}, {"voxel": sample}])


def test_hybrid_config_is_encoder_only_comparator_and_rejects_other_checkpoint_identity():
    from common import build_model, checkpoint_identity, validate_evaluation_checkpoint
    from notebook_config import resolve_notebook_config
    hybrid = resolve_notebook_config(ROOT, preset="ENCODER_PILLAR_RICH", augmentation="config")
    saved = json.loads((ROOT / "configs/experiments/encoders/pillar_rich.json").read_text())
    assert hybrid == saved
    model = build_model(hybrid)
    assert sum(p.numel() for p in model.parameters()) == 655513
    other = resolve_notebook_config(ROOT, preset="ENCODER_PILLAR32", augmentation="config")
    with pytest.raises(ValueError, match="identity"):
        validate_evaluation_checkpoint({"checkpoint_identity": checkpoint_identity(other)}, hybrid)
    left, right = copy.deepcopy(hybrid), copy.deepcopy(other)
    left["data"].pop("bev_encoding"); right["data"].pop("bev_encoding")
    assert left == right

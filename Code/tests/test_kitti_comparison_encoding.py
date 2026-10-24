"""Comparison inputs must be identical to the training dataset encoders."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("encoding", [
    {"name": "binary_slices"}, {"name": "hist14", "backend": "numpy"},
])
def test_comparison_encoder_matches_dataset_with_xy_orientation_and_open_bounds(encoding):
    from tools.visualization.predict_kitti_overview import encode_model_points, ROOT
    from tools.kitti_training_pipeline.common import configure_detector_imports, bev_encoding_spec
    configure_detector_imports(ROOT / "detector")
    from core.datasets.dataset import Dataset

    geometry = {"x_min": 0., "x_max": 4., "x_res": .1,
                "y_min": -3., "y_max": 3., "y_res": .1,
                "z_min": -2.5, "z_max": 1., "z_res": .1}
    config = {"data": {"bev_encoding": encoding, "kitti": {"geometry": geometry}}}
    points = np.array([[1.25, -2.25, -.45, .8], [2.35, 1.45, .35, .4],
                       [0, 0, 0, 1], [4, 0, 0, 1], [1, 0, -2.5, 1],
                       [1, 0, 1, 1]], dtype=np.float32)
    dataset = Dataset.__new__(Dataset)
    dataset.bev_encoding = encoding
    expected = dataset.encode_input(points, geometry)
    original = points.copy()
    actual = encode_model_points(points, bev_encoding_spec(config))
    np.testing.assert_array_equal(actual, expected.numpy())
    np.testing.assert_array_equal(points, original)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert actual.shape == (35 if encoding["name"] == "binary_slices" else 14, 60, 40)
    assert torch.count_nonzero(expected) > 0


def comparison_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Demo/code"))
    import export_kitti_model_comparison
    return export_kitti_model_comparison


def test_heatmap_places_positive_y_on_left_and_forward_x_up_without_rescaling():
    module = comparison_module()
    plt = module.setup_matplotlib()
    fig, ax = plt.subplots()
    maps = np.zeros((1, 6, 4), dtype=np.float32)
    maps[0, 4, 2] = .75
    result = {"config": {"data": {"kitti": {"geometry": {
        "x_min": 0, "x_max": 4, "y_min": -3, "y_max": 3}}}},
        "heatmaps": {"probabilities": maps}}
    artist = module.draw_heatmap(ax, result, [0, 4, -3, 3, -2.8, 4])
    expected = np.zeros((4, 6), dtype=np.float32)
    expected[2, 1] = .75
    np.testing.assert_array_equal(artist.get_array(), expected)
    assert artist.get_clim() == (0., 1.)
    assert artist.get_interpolation() == "nearest"
    assert artist.origin == "lower"
    assert list(artist.get_extent()) == [-3, 3, 0, 4]
    plt.close(fig)


def test_backup_with_mismatched_original_scan_hash_is_rejected(tmp_path, monkeypatch):
    import json
    module = comparison_module()
    archive, metadata = tmp_path / "input.npz", tmp_path / "metadata.json"
    np.savez(archive, raw=np.array([[1, 2, 3, .5]], dtype=np.float32))
    metadata.write_text(json.dumps({"source": {"bin_sha256": "wrong original hash"}}))
    monkeypatch.setitem(module.BACKUPS, "000134", (str(archive), "raw", str(metadata)))
    with pytest.raises(ValueError, match="differ from recorded original"):
        module.load_cloud("000134")

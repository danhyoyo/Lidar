"""Encoding identity and input shapes must agree without importing PyTorch."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.core.bev_encoding import resolve_bev_encoding, resolve_input_channels

GEOMETRY = {
    "x_min": 0, "x_max": 16, "x_res": 0.5,
    "y_min": -6, "y_max": 6, "y_res": 0.25,
    "z_min": -2.5, "z_max": 1, "z_res": 0.1,
}


def test_hist14_schema_layout_normalization_edges_and_precision():
    schema = resolve_bev_encoding({"name": "hist14"}, GEOMETRY)
    assert schema.channels == 14
    assert schema.version == 1
    assert schema.input_shape == (1, 14, 48, 32)
    assert schema.output_shape == (48, 32, 14)
    assert schema.bin_edges == (-2.5, -1.625, -0.75, 0.125, 1.0)
    assert schema.channel_names == (
        "height_count_b0", "height_count_b1", "height_count_b2", "height_count_b3",
        "z_max", "z_mean", "z_span", "z_std", "intensity_max", "intensity_mean",
        "log_density", "occupancy", "point_offset_x", "point_offset_y",
    )
    metadata = schema.semantic_metadata()
    assert metadata["density_norm"] == 32.0
    assert metadata["intensity_scale"] == 1.0
    assert metadata["layout"] == {"encoding": "YXC", "model": "BCYX"}
    assert metadata["precision"] == {"coordinates": "float64", "index": "int64",
                                      "accumulation": "float64", "output": "float32"}
    assert metadata["boundary"]["epsilon"] == 0.001
    assert metadata["boundary"]["interval"] == "open"
    assert metadata["height_bin_ties"] == "upper"
    assert metadata["finite_filter"] == "first_four_columns"
    assert len(schema.semantic_hash) == 64
    json.dumps(metadata, allow_nan=False)


def test_semantic_identity_excludes_backend_but_includes_effective_values():
    numpy = resolve_bev_encoding({"name": "hist14"}, GEOMETRY)
    explicit = resolve_bev_encoding({"name": "hist14", "version": 1,
                                    "out_channels": 14, "backend": "numba",
                                    "density_norm": 32, "intensity_scale": 1}, GEOMETRY)
    assert numpy.backend == "numpy" and explicit.backend == "numba"
    assert numpy.semantic_hash == explicit.semantic_hash
    assert numpy.semantic_metadata() == explicit.semantic_metadata()
    for overrides in [{"density_norm": 16}, {"intensity_scale": 2}]:
        changed = resolve_bev_encoding({"name": "hist14", **overrides}, GEOMETRY)
        assert changed.semantic_hash != numpy.semantic_hash
    changed_geometry = dict(GEOMETRY, x_min=1, x_max=17)
    assert resolve_bev_encoding({"name": "hist14"}, changed_geometry).semantic_hash != numpy.semantic_hash


def test_invalid_encoding_options_fail():
    invalid_encodings = [
        {"name": "unknown"}, {"name": "hist14", "version": 2},
        {"name": "hist14", "version": True}, {"name": "hist14", "version": 1.0},
        {"name": "hist14", "out_channels": 15}, {"name": "hist14", "out_channels": 8},
        {"name": "hist14", "out_channels": True}, {"name": "hist14", "out_channels": 14.0},
        {"name": "hist14", "backend": "cuda"}, {"name": "hist14", "num_bins": 8},
        {"name": "hist14", "bin_edges": [0, 1]}, {"name": "hist14", "epsilon": 0},
        {"name": "hist14", "density_norm": 1}, {"name": "hist14", "density_norm": float("nan")},
        {"name": "hist14", "intensity_scale": 0}, {"name": "hist14", "intensity_scale": float("inf")},
    ]
    for encoding in invalid_encodings:
        with pytest.raises(ValueError):
            resolve_bev_encoding(encoding, GEOMETRY)


def test_invalid_geometry_fails_before_shape_resolution():
    invalid_changes = [
        {"x_res": 0}, {"y_res": -1}, {"z_max": -3}, {"x_max": float("inf")},
        {"y_min": float("nan")}, {"x_res": 0.3}, {"z_res": 0.4},
    ]
    for change in invalid_changes:
        with pytest.raises(ValueError):
            resolve_bev_encoding({"name": "hist14"}, dict(GEOMETRY, **change))


@pytest.mark.parametrize("name,configured,expected", [
    ("rich8", None, 8), ("rich10", 8, 10), ("rich11", None, 11),
    ("rich12", None, 12), ("rich8", 9, 9), ("rich8", 12, 12),
    ("rich10", 16, 16), ("rich8", "10", 10),
])
def test_rich_legacy_channel_padding_agrees_with_actual_encoder(name, configured, expected):
    from detector.core.datasets.utils_1.preprocess import encode_bev
    options = {"name": name, "out_channels": configured}
    schema = resolve_bev_encoding(options, GEOMETRY)
    points = np.array([[1.1, -1.1, -1.0, 0.2], [1.1, -1.1, 0.5, 0.8]], dtype=np.float32)
    actual = encode_bev(points, GEOMETRY, options)
    assert actual.shape == schema.output_shape == (48, 32, expected)
    assert resolve_input_channels(options) == expected
    if expected == 9:
        assert schema.channel_names[8] == "reserved_zero_8"
        assert not actual[..., 8].any()
    if expected >= 10:
        assert schema.channel_names[8:10] == ("z_span", "z_std")
        assert actual[..., 8].max() > 0
    if expected > 12:
        assert not actual[..., 12:].any()
        assert schema.channel_names[12] == "reserved_zero_12"


def test_binary_slices_preserve_geometry_channels_and_ignore_out_channels():
    from detector.core.datasets.utils_1.preprocess import encode_bev
    options = {"name": "binary_slices", "out_channels": 99}
    schema = resolve_bev_encoding(options, GEOMETRY)
    actual = encode_bev(np.empty((0, 4), dtype=np.float32), GEOMETRY, options)
    assert schema.channels == 35
    assert actual.shape == schema.output_shape == (48, 32, 35)
    assert resolve_input_channels(options, GEOMETRY) == 35
    assert resolve_input_channels({}, default_channels=8) == 8


def test_binary_shapes_keep_legacy_truncation_at_float_roundoff():
    from detector.core.datasets.utils_1.preprocess import encode_bev
    geometry = dict(GEOMETRY, x_max=4.8, x_res=0.1)
    schema = resolve_bev_encoding(None, geometry)
    assert schema.output_shape == encode_bev(np.empty((0, 4)), geometry).shape


def test_schema_is_detached_from_input_and_returned_metadata():
    options, geometry = {"name": "hist14"}, copy.deepcopy(GEOMETRY)
    schema = resolve_bev_encoding(options, geometry)
    identity = schema.semantic_hash
    options["name"] = "rich8"
    geometry["z_max"] = 10
    metadata = schema.semantic_metadata()
    metadata["geometry"]["z_max"] = 10
    metadata["channel_names"].clear()
    assert schema.semantic_hash == identity


def test_pure_schema_import_and_legacy_core_import_context():
    script = (
        "import sys; from detector.core.bev_encoding import resolve_bev_encoding; "
        "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
        "sys.path.insert(0, 'detector'); from core.bev_encoding import resolve_input_channels; "
        "assert resolve_input_channels({'name': 'hist14'}) == 14"
    )
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_pipeline_shape_uses_schema_and_rejects_invalid_geometry():
    from tools.kitti_training_pipeline.common import input_shape
    config = {"data": {"bev_encoding": {"name": "hist14"}, "kitti": {"geometry": GEOMETRY}}}
    assert input_shape(config) == resolve_bev_encoding({"name": "hist14"}, GEOMETRY).input_shape
    config["data"]["kitti"]["geometry"] = dict(GEOMETRY, x_res=0.3)
    with pytest.raises(ValueError, match="divisible"):
        input_shape(config)


def test_pipeline_notebook_imports_work_outside_repository_without_torch(tmp_path):
    script = f"""
import sys
sys.path.insert(0, {str(ROOT / 'tools/kitti_training_pipeline')!r})
from common import input_shape
from notebook_config import resolve_notebook_config
assert input_shape({{'data': {{'bev_encoding': {{'name': 'hist14'}}, 'kitti': {{'geometry': {GEOMETRY!r}}}}}}}) == (1, 14, 48, 32)
cfg = resolve_notebook_config({str(ROOT)!r}, preset='RICH10_SGFPN', augmentation='none')
assert input_shape(cfg)[1] == 10
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)

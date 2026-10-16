"""Unit tests for GT sampler comparison and visualization tool."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.visualization.compare_gt_samplers import (
    ALL_FLAGS,
    define_comparison_suite,
    load_scene_data,
    run_sampler_variant,
    render_bev_figure,
    render_composite_matrix_grid,
)

KITTI_DIR = ROOT / "Kitti_Raw"
DONOR_DIR = ROOT.parent / "Demo/outputs/pipeline/005663/donor_database"
FRAME_ID = "005663"


def test_define_comparison_suite():
    suite = define_comparison_suite()
    assert len(suite) == 12
    keys = [item["key"] for item in suite]
    assert "raw_original" in keys
    assert "openpcdet" in keys
    assert "hybrid_default" in keys
    assert "hybrid_all_off" in keys
    assert "hybrid_all_on" in keys
    for flag in ALL_FLAGS:
        assert f"hybrid_toggle_{flag}" in keys


def test_load_scene_data():
    points, boxes, class_ids = load_scene_data(FRAME_ID, KITTI_DIR)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2 and points.shape[1] == 4
    assert points.shape[0] > 10000

    assert isinstance(boxes, np.ndarray)
    assert boxes.ndim == 2 and boxes.shape[1] == 8
    assert boxes.shape[0] > 0
    assert set(class_ids.keys()) == {"Car", "Pedestrian", "Cyclist"}


def test_run_raw_variant():
    points, boxes, class_ids = load_scene_data(FRAME_ID, KITTI_DIR)
    res = run_sampler_variant(
        "raw_original", "Raw", "Raw scene", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    assert res["sampler_type"] == "none"
    assert res["metadata"]["num_inserted"] == 0
    assert len(res["boxes"]) == len(boxes)
    assert not res["box_is_sampled"].any()
    assert not res["point_is_sampled"].any()


def test_run_openpcdet_variant():
    points, boxes, class_ids = load_scene_data(FRAME_ID, KITTI_DIR)
    res = run_sampler_variant(
        "openpcdet", "OpenPCDet", "OpenPCDet test", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    assert res["sampler_type"] == "openpcdet"
    assert len(res["boxes"]) >= len(boxes)
    assert res["metadata"]["num_inserted"] == len(res["boxes"]) - len(boxes)
    assert res["box_is_sampled"].sum() == res["metadata"]["num_inserted"]


def test_run_hybrid_all_off_and_default():
    points, boxes, class_ids = load_scene_data(FRAME_ID, KITTI_DIR)
    # Test All OFF
    res_off = run_sampler_variant(
        "hybrid_all_off", "Hybrid All OFF", "Test all off", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    assert res_off["sampler_type"] == "hybrid"
    for f in ALL_FLAGS:
        assert res_off["flags"][f] is False
    assert res_off["metadata"]["num_inserted"] > 0

    # Test Default
    res_def = run_sampler_variant(
        "hybrid_default", "Hybrid Default", "Test default", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    assert res_def["sampler_type"] == "hybrid"
    assert res_def["flags"]["ENABLE_GROUND_VALIDATION"] is True
    assert res_def["flags"]["ENABLE_STATIC_COLLISION"] is True
    assert res_def["flags"]["ENABLE_LINE_OF_SIGHT"] is False
    assert res_def["flags"]["ENABLE_SHADOW_MASKING"] is False
    assert res_def["flags"]["ENABLE_DENSITY_SUBSAMPLE"] is True
    assert res_def["flags"]["ENABLE_VISIBILITY_PROTECTION"] is True


def test_render_bev_and_grid_images(tmp_path):
    points, boxes, class_ids = load_scene_data(FRAME_ID, KITTI_DIR)
    geometry = {"x_min": 0.0, "x_max": 70.4, "y_min": -40.0, "y_max": 40.0, "z_min": -2.5, "z_max": 1.0}

    res_raw = run_sampler_variant(
        "raw_original", "Raw", "Raw scene", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    out_img = tmp_path / "raw.png"
    render_bev_figure(
        res_raw, FRAME_ID, geometry, class_ids, seed=42, output_path=out_img, max_bg_points=1000, dpi=80
    )
    assert out_img.is_file()
    assert out_img.stat().st_size > 5000

    # Test grid rendering with mock map
    res_open = run_sampler_variant(
        "openpcdet", "OpenPCDet", "OpenPCDet", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    res_off = run_sampler_variant(
        "hybrid_all_off", "Hybrid OFF", "All OFF", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    res_def = run_sampler_variant(
        "hybrid_default", "Hybrid Def", "Default", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    res_stat = run_sampler_variant(
        "hybrid_toggle_ENABLE_STATIC_COLLISION", "Static", "Static", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    res_los = run_sampler_variant(
        "hybrid_toggle_ENABLE_LINE_OF_SIGHT", "LOS", "LOS", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    res_shd = run_sampler_variant(
        "hybrid_toggle_ENABLE_SHADOW_MASKING", "Shadow", "Shadow", points, boxes, DONOR_DIR, class_ids, seed=42
    )
    res_on = run_sampler_variant(
        "hybrid_all_on", "Hybrid ON", "All ON", points, boxes, DONOR_DIR, class_ids, seed=42
    )

    results_map = {
        "raw_original": res_raw,
        "openpcdet": res_open,
        "hybrid_all_off": res_off,
        "hybrid_default": res_def,
        "hybrid_toggle_ENABLE_STATIC_COLLISION": res_stat,
        "hybrid_toggle_ENABLE_LINE_OF_SIGHT": res_los,
        "hybrid_toggle_ENABLE_SHADOW_MASKING": res_shd,
        "hybrid_all_on": res_on,
    }

    grid_img = tmp_path / "grid.png"
    render_composite_matrix_grid(
        results_map, FRAME_ID, geometry, class_ids, seed=42, output_path=grid_img, dpi=80
    )
    assert grid_img.is_file()
    assert grid_img.stat().st_size > 20000

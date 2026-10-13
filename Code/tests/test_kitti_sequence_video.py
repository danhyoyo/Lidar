"""Sequence order must follow official KITTI mapping, never sorted Object IDs."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def api():
    return importlib.import_module("tools.visualization.export_kitti_sequence_video")


def fixture_mapping(tmp_path):
    mapping = tmp_path / "mapping"
    mapping.mkdir()
    (mapping / "train_rand.txt").write_text("3,1,2,4")
    (mapping / "train_mapping.txt").write_text(
        "date drive_a 0000000007\ndate drive_a 0000000008\n"
        "date drive_a 0000000009\ndate drive_b 0000000010\n")
    points = tmp_path / "velodyne"
    points.mkdir()
    for identifier in range(4):
        np.array([[identifier, 1, -1, .5]], dtype="<f4").tofile(points / f"{identifier:06d}.bin")
    ids = tmp_path / "ids.txt"
    ids.write_text("000001\n000002\n000000\n")
    return ids, mapping, points


def test_official_mapping_uses_one_based_permutation_and_preserves_chosen_order(tmp_path):
    ids, mapping, points = fixture_mapping(tmp_path)
    entries = api().load_sequence(ids, mapping, points)
    assert [row["object_id"] for row in entries] == ["000001", "000002", "000000"]
    assert [row["raw_frame"] for row in entries] == [7, 8, 9]
    assert all(row["drive"] == "drive_a" for row in entries)


@pytest.mark.parametrize("text", ["000000\n000001\n", "000001\n000000\n",
                                  "000000\n000003\n", "000001\n000001\n"])
def test_reversed_gapped_mixed_drive_or_duplicate_sequence_is_rejected(tmp_path, text):
    ids, mapping, points = fixture_mapping(tmp_path)
    ids.write_text(text)
    with pytest.raises(ValueError):
        api().load_sequence(ids, mapping, points)


def test_missing_bin_is_not_skipped_or_replaced(tmp_path):
    ids, mapping, points = fixture_mapping(tmp_path)
    (points / "000002.bin").unlink()
    with pytest.raises(FileNotFoundError):
        api().load_sequence(ids, mapping, points)


def test_invalid_official_permutation_is_rejected(tmp_path):
    ids, mapping, points = fixture_mapping(tmp_path)
    (mapping / "train_rand.txt").write_text("3,1,1,4")
    with pytest.raises(ValueError, match="permutation"):
        api().load_sequence(ids, mapping, points)


def test_renderer_handles_empty_then_rotated_detections_without_changing_geometry(tmp_path):
    renderer = api().SequenceRenderer(roi=[0, 8, -4, 4, -2.5, 3], width=960, height=480,
        elevation=28, azimuth=-145, heatmap_extent=[0, 8, -4, 4],
        title="Fixture sequence", score_threshold=.3)
    points = np.array([[1, 1, -1, .5], [5, -2, 1, .7]], dtype=np.float32)
    probabilities = np.zeros((3, 20, 20), dtype=np.float32)
    probabilities[2, 10, 5] = .8
    renderer.render(points, np.empty((0, 7)), probabilities,
        class_ids={"Car": 0, "Pedestrian": 1, "Cyclist": 2}, index=0, total=2,
        entry={"object_id": "000001", "raw_frame": 7}, fps=10, output=tmp_path / "empty.png")
    rows = np.array([[2, .8, 5, -2, 2, .6, np.pi / 2],
                     [0, .9, 100, 0, 4, 2, 0]], dtype=np.float32)
    visible_counts = renderer.render(points, rows, probabilities,
        class_ids={"Car": 0, "Pedestrian": 1, "Cyclist": 2}, index=1, total=2,
        entry={"object_id": "000002", "raw_frame": 8}, fps=10, output=tmp_path / "box.png")
    assert visible_counts == {"Cyclist": 1}
    np.testing.assert_array_equal(renderer.heatmap_artist.get_array(), probabilities.max(axis=0))
    np.testing.assert_array_equal(renderer.bev_points.get_offsets(), points[:, :2])
    assert renderer.heatmap_artist.get_clim() == (0, 1)
    from PIL import Image
    with Image.open(tmp_path / "box.png") as image:
        assert image.size == (960, 480)
        image.verify()
    renderer.close()

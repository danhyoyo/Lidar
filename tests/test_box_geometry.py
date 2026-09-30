import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'detector'))
from core.datasets.utils_1.box_geometry import ray_box_intervals


@pytest.mark.parametrize('columns', [7, 8])
def test_ray_hand_computed_intervals(columns):
    box = np.array([2, 2, 4, 10, 0, -1, 0], dtype=float)
    if columns == 8:
        box = np.r_[0, box]
    points = np.array([[20, 0, 0], [10, 0, 0], [5, 0, 0], [-20, 0, 0],
                       [0, 0, 0], [20, 3, 0], [20, 0, 4], [20, 2.5, 0]])
    hit, enter, leave = ray_box_intervals(points, box)
    np.testing.assert_array_equal(hit, [True, True, True, False, False, False, False, True])
    np.testing.assert_allclose(enter[hit], [.4, .8, 1.6, .4])
    np.testing.assert_allclose(leave[hit], [.6, 1.2, 2.4, .4])
    assert np.isinf(enter[~hit]).all()


def test_ray_rotation_invariance_and_sensor_inside():
    box = np.array([2, 2, 4, 10, 0, -1, 0], dtype=float)
    points = np.array([[20, 0, 0], [20, 3, 0], [20, 0, 2]], dtype=float)
    expected = ray_box_intervals(points, box)
    for angle in [np.pi / 2, np.pi, -.83]:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        rotated = points.copy()
        rotated[:, :2] = points[:, :2] @ rotation.T
        rotated_box = box.copy()
        rotated_box[3:5] = box[3:5] @ rotation.T
        rotated_box[-1] = angle
        for actual, want in zip(ray_box_intervals(rotated, rotated_box), expected):
            np.testing.assert_allclose(actual, want)
    hit, enter, leave = ray_box_intervals([[10, 0, 0], [0, 0, 0]], [2, 2, 4, 0, 0, -1, 0])
    np.testing.assert_array_equal(hit, [True, False])
    assert enter[0] == 0
    assert leave[0] == pytest.approx(.2)


def test_ray_parallel_axes_and_empty():
    hit, enter, leave = ray_box_intervals([[0, 20, 0], [1e-14, 20, 0]], [2, 4, 2, 0, 10, -1, 0])
    assert hit.all()
    np.testing.assert_allclose(enter, [.4, .4])
    np.testing.assert_allclose(leave, [.6, .6])
    assert all(a.shape == (0,) for a in ray_box_intervals(np.empty((0, 4)), [2, 2, 4, 10, 0, -1, 0]))


@pytest.mark.parametrize('box', [[0, 2, 4, 10, 0, -1, 0], [2, 2, -4, 10, 0, -1, 0], [2, 2, 4, np.nan, 0, -1, 0], [1, 2]])
def test_ray_invalid_box(box):
    with pytest.raises(ValueError):
        ray_box_intervals([[20, 0, 0]], box)


def test_ray_invalid_points():
    with pytest.raises(ValueError):
        ray_box_intervals([[np.inf, 0, 0]], [2, 2, 4, 10, 0, -1, 0])

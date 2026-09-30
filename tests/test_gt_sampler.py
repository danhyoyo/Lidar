import sys
from pathlib import Path
import pickle
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
]

from core.datasets.utils_1.gt_sampler import GTSampler


def test_gt_sampler_with_8col_boxes(tmp_path, monkeypatch):
    np.random.seed(42)
    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.0, 0.0],
                    dtype=np.float32,
                ),
                "points": np.column_stack((np.linspace(-.5,.5,50), np.zeros(50), np.linspace(.1,1.4,50), np.full(50,.9))).astype(np.float32),
                "r_origin": 15.0,
                "num_points": 50,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    sampler = GTSampler(str(db_file), sample_counts={"Car": 1}, p=1.0)
    monkeypatch.setattr(sampler, "_propose_pose", lambda cls, attempt: (30, 0, 0))
    init_points = flat_ground()
    # Existing scene with 1 pedestrian (cls=1)
    init_boxes = np.array(
        [[1.0, 1.7, 0.6, 0.8, 10.0, 5.0, -1.0, 0.0]], dtype=np.float32
    )

    aug_points, aug_boxes = sampler(init_points, init_boxes)
    assert len(aug_boxes) == 2
    assert aug_boxes.shape[1] == 8
    assert aug_boxes[1, 0] == 0.0  # Inserted car has class id 0
    assert len(aug_points) > 0
    assert np.any(aug_points[:, 2] > -1.6)


def test_gt_sampler_probability_zero(tmp_path):
    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.0, 0.0],
                    dtype=np.float32,
                ),
                "points": np.zeros((10, 4), dtype=np.float32),
                "r_origin": 15.0,
                "num_points": 10,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    sampler = GTSampler(str(db_file), sample_counts={"Car": 1}, p=0.0)
    init_points = np.zeros((50, 4), dtype=np.float32)
    init_boxes = np.zeros((0, 8), dtype=np.float32)

    aug_points, aug_boxes = sampler(init_points, init_boxes)
    assert len(aug_boxes) == 0
    assert len(aug_points) == 50


def test_gt_sampler_removes_interior_background_points(tmp_path):
    np.random.seed(42)
    db_file = tmp_path / "mock_db.pkl"
    # Canonical car points (centered at 0, bottom at 0)
    car_points = np.zeros((20, 4), dtype=np.float32)
    car_points[:, 0] = np.linspace(-1.0, 1.0, 20)  # within l=4.5
    car_points[:, 1] = np.linspace(-0.5, 0.5, 20)  # within w=1.8
    car_points[:, 2] = np.linspace(0.1, 1.4, 20)  # within h=1.5
    car_points[:, 3] = 0.9

    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.6, 0.0],
                    dtype=np.float32,
                ),
                "points": car_points,
                "r_origin": 15.0,
                "num_points": 20,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    sampler = GTSampler(
        str(db_file), sample_counts={"Car": 1}, p=1.0, enable_physics=True
    )
    # Background scene with ground points and a cluster of noise points
    bg_pts = []
    for x in np.linspace(5, 40, 30):
        for y in np.linspace(-10, 10, 20):
            bg_pts.append([x, y, -1.6, 0.5])
    init_points = np.array(bg_pts, dtype=np.float32)
    init_boxes = np.zeros((0, 8), dtype=np.float32)

    aug_points, aug_boxes = sampler(init_points, init_boxes)
    assert len(aug_boxes) == 1
    # Verify placed car is within BEV bounds
    bx = aug_boxes[0, 4]
    by = aug_boxes[0, 5]
    assert 2.0 <= bx <= 65.0
    assert -35.0 <= by <= 35.0
    assert len(aug_points) > 0


def test_gt_sampler_rejects_placement_inside_wall(tmp_path):
    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.6, 0.0],
                    dtype=np.float32,
                ),
                "points": np.ones((50, 4), dtype=np.float32),
                "r_origin": 15.0,
                "num_points": 50,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    sampler = GTSampler(
        str(db_file), sample_counts={"Car": 1}, p=1.0, enable_physics=True
    )

    # Empty scene with only a massive wall at x in [5, 65], y in [-35, 35], z in [-0.5, 3.0]
    # No valid flat ground anywhere
    xs = np.linspace(5, 65, 30)
    ys = np.linspace(-30, 30, 30)
    xx, yy = np.meshgrid(xs, ys)
    wall_pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(900, 1.0), np.ones(900)]
    ).astype(np.float32)

    aug_pts, aug_boxes = sampler(wall_pts, np.zeros((0, 8), dtype=np.float32))
    # Since only elevated obstacles exist and no road support exists, placement must be safely rejected
    assert len(aug_boxes) == 0



@pytest.fixture
def sampler_factory(tmp_path, monkeypatch):
    def make(counts=None, poses=None, samples=None, **kwargs):
        canonical = np.column_stack((np.linspace(-1, 1, 20), np.zeros(20),
                                     np.full(20, .6), np.full(20, .9))).astype(np.float32)
        sample = {'box': np.array([0, 2, 2, 4, 10, 0, -1.6, 0], np.float32),
                  'points': canonical, 'num_points': 20, 'r_origin': 10.}
        db = samples if samples is not None else {'Car': [sample]}
        path = tmp_path / 'fixture.pkl'
        with path.open('wb') as stream:
            pickle.dump(db, stream)
        sampler = GTSampler(str(path), sample_counts={'Car': 1} if counts is None else counts,
                            **kwargs)
        if poses is not None:
            monkeypatch.setattr(sampler, '_propose_pose', lambda cls, attempt: poses[cls], raising=False)
        return sampler
    return make


def flat_ground():
    x, y = np.meshgrid(np.arange(5, 66, .5), np.arange(-20, 21, .5))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.6),
                            np.full(x.size, .1))).astype(np.float32)


def test_rejected_empty_candidate_is_atomic(sampler_factory):
    tiny = {'box': np.array([0, 2, 2, 4, 10, 0, -1.6, 0], np.float32),
            'points': np.array([[0, 0, .5, .9]], np.float32), 'num_points': 1, 'r_origin': .001}
    sampler = sampler_factory(samples={'Car': [tiny]}, poses={'Car': (50, 0, 0)}, enable_physics=True)
    points = flat_ground()
    before = points.copy()
    out, boxes, meta = sampler(points, np.empty((0, 8), np.float32), True)
    np.testing.assert_array_equal(out, before)
    np.testing.assert_array_equal(points, before)
    assert len(boxes) == meta['num_inserted'] == 0
    assert meta['inserted_points'] == []


@pytest.mark.parametrize('visible,total,accepted', [(0, 30, False), (15, 30, True), (14, 30, False),
                                                     (4, 4, True), (3, 4, False), (0, 0, True)])
def test_original_label_visibility_fixed_baseline(sampler_factory, visible, total, accepted):
    sampler = sampler_factory(poses={'Car': (10, 0, 0)}, enable_physics=False,
                              enable_shadow_masking=True)
    points = np.zeros((total, 4), np.float32)
    points[:, 0] = 20
    points[:, 2] = -.5
    points[:, 3] = .7
    points[:visible, 1] = 4
    box = np.array([[0, 2, 10, 4, 20, 0, -1.6, 0]], np.float32)
    out, boxes, meta = sampler(points, box, True)
    assert len(boxes) == 1 + accepted
    np.testing.assert_array_equal(boxes[0], box[0])
    if not accepted:
        np.testing.assert_array_equal(out, points)
        assert meta['num_inserted'] == 0


def test_visibility_tracks_overlapping_original_memberships(sampler_factory):
    sampler = sampler_factory(poses={'Car': (10, 0, 0)}, enable_physics=False, enable_shadow_masking=True)
    points = np.array([[20, 0, -.5, .7]] * 10 + [[20, 4, -.5, .7]] * 10, np.float32)
    boxes = np.array([[0, 2, 10, 4, 20, 0, -1.6, 0],
                      [0, 2, 2, 4, 20, 0, -1.6, 0]], np.float32)
    out, final = sampler(points, boxes)
    # Broad box retains half; overlapping narrow box loses all and vetoes placement.
    np.testing.assert_array_equal(out, points)
    np.testing.assert_array_equal(final, boxes)


def test_inserted_visibility_cumulative_and_metadata_final(sampler_factory):
    db = {}
    for cls, class_id, width, ys in [('Car', 0, 20, np.array([-8,-6,-4,-2,0,2,4,6,8,9])),
                                   ('Pedestrian', 1, 4, np.zeros(10)),
                                   ('Cyclist', 2, 1, np.zeros(10))]:
        points = np.column_stack((np.zeros(10), ys, np.full(10, 1.6), np.full(10, .8))).astype(np.float32)
        db[cls] = [{'box': np.array([class_id, 2, width, 4, 30, 0, -1.6, 0], np.float32),
                    'points': points, 'num_points': 10, 'r_origin': 30.}]
    sampler = sampler_factory(counts={'Car': 1, 'Pedestrian': 1, 'Cyclist': 1}, samples=db,
                              poses={'Car': (30, 0, 0), 'Pedestrian': (20, -3, 0), 'Cyclist': (10, 1, 0)},
                              enable_physics=False, enable_shadow_masking=True)
    out, boxes, meta = sampler(np.empty((0, 4), np.float32), np.empty((0, 8), np.float32), True)
    assert len(boxes) == 2  # Third insertion would cumulatively remove >50% of first.
    assert len(meta['inserted_points'][0]) == 6
    assert sum(map(len, meta['inserted_points'])) == len(out)
    for cloud in meta['inserted_points']:
        assert all(np.any(np.all(out == point, axis=1)) for point in cloud)


@pytest.mark.parametrize('shadow,density,radiometric', [(a,b,c) for a in [False,True] for b in [False,True] for c in [False,True]])
def test_flags_have_independent_observable_effects(sampler_factory, shadow, density, radiometric):
    np.random.seed(142)
    sampler = sampler_factory(poses={'Car': (30, 0, 0)}, enable_physics=False,
                              enable_shadow_masking=shadow, enable_density_subsample=density,
                              enable_radiometric_calibration=radiometric)
    bg = np.array([[30, 0, -.5, .2], [50, 0, 0, .3]], np.float32)
    out, boxes, meta = sampler(bg, np.empty((0, 8), np.float32), True)
    assert len(boxes) == 1
    assert not np.any(out[:, 3] == .2)  # Interior deletion always enabled.
    assert np.any(out[:, 3] == .3) == (not shadow)
    inserted = meta['inserted_points'][0]
    assert len(inserted) == (5 if density else 20)
    assert np.all(inserted[:, 3] < .9) if radiometric else np.all(inserted[:, 3] == .9)


@pytest.mark.parametrize('ground,static,los,accepted', [(False,False,False,True), (True,False,False,False),
                                                       (False,True,False,False), (False,False,True,False)])
def test_flags_placement_checks_independent(sampler_factory, ground, static, los, accepted):
    sampler = sampler_factory(poses={'Car': (30, 0, 0)}, enable_physics=False,
                              enable_ground_validation=ground, enable_static_collision=static,
                              enable_line_of_sight=los)
    points = np.array([[30, 0, -.5, .2]] * 3 + [[15, 0, -.2, .3]] * 5, np.float32)
    _, boxes = sampler(points, np.empty((0, 8), np.float32))
    assert bool(len(boxes)) == accepted


def test_ground_empty_scene_rejected_and_real_ground_accepted(sampler_factory):
    sampler = sampler_factory(poses={'Car': (10, 0, 0)})
    _, boxes = sampler(np.empty((0, 4), np.float32), np.empty((0, 8), np.float32))
    assert len(boxes) == 0
    _, boxes = sampler(flat_ground(), np.empty((0, 8), np.float32))
    assert len(boxes) == 1
    assert boxes[0, 6] == pytest.approx(-1.6)


@pytest.mark.parametrize('columns', [7,8])
@pytest.mark.parametrize('counts,p', [({},1), ({'Car':0},1), ({'Car':1},0)])
def test_noop_metadata_schema_counts_probability(sampler_factory, monkeypatch, columns, counts, p):
    sampler = sampler_factory(counts=counts, p=p)
    monkeypatch.setattr(np.random, 'random', lambda *args: 0.)
    lidar, boxes = np.empty((0,4), np.float32), np.empty((0,columns), np.float32)
    out, final, meta = sampler(lidar, boxes, True)
    assert final.shape == (0,columns)
    assert out.dtype == lidar.dtype and final.dtype == boxes.dtype
    assert meta['num_inserted'] == 0
    np.testing.assert_array_equal(meta['final_boxes'], final)


def test_empty_7col_insertion_and_no_mutation(sampler_factory):
    sampler = sampler_factory(poses={'Car': (10, 0, np.pi/2)}, enable_physics=False)
    before = sampler.database['Car'][0]['points'].copy()
    lidar, boxes = np.empty((0,4), np.float32), np.empty((0,7), np.float32)
    out, final = sampler(lidar, boxes)
    assert final.shape == (1,7)
    np.testing.assert_allclose(out[:, 1], before[:, 0], atol=1e-6)
    np.testing.assert_allclose(out[:, 0], 10)
    np.testing.assert_array_equal(sampler.database['Car'][0]['points'], before)
    assert lidar.shape == (0,4) and boxes.shape == (0,7)


@pytest.mark.parametrize('kwargs', [{'p': -1}, {'p': 2}, {'p': np.nan}, {'p': np.inf}, {'sample_counts': {'Car': -1}},
                                    {'sample_counts': {'Car': 1.5}}, {'sample_counts': {'Car': True}},
                                    {'enable_physics':'false'}, {'enable_shadow_masking':'false'},
                                    {'min_visible_points':-1}, {'min_visible_ratio':1.1}])
def test_invalid_sampler_configuration(tmp_path, kwargs):
    path = tmp_path / 'db.pkl'
    path.write_bytes(pickle.dumps({}))
    with pytest.raises(ValueError):
        GTSampler(str(path), **kwargs)


@pytest.mark.parametrize('lidar,boxes', [(np.zeros((1,3)),np.empty((0,8))), (np.zeros((1,4)),np.empty((0,6))),
                                        (np.full((1,4),np.nan),np.empty((0,8))),
                                        (np.zeros((1,4)),np.array([[0,0,2,4,10,0,-1,0]]))])
def test_invalid_scene_input_early(sampler_factory, lidar, boxes):
    sampler = sampler_factory(counts={})
    with pytest.raises(ValueError):
        sampler(lidar, boxes)


@pytest.mark.parametrize('change', [{'box':np.zeros(7)}, {'box':np.full(8,np.nan)}, {'points':np.zeros((0,4))},
                                   {'points':np.full((20,4),np.inf)}, {'r_origin':0}, {'num_points':3},
                                   {'box':np.array([0,-1,2,4,10,0,-1,0])}])
def test_database_validation_has_class_sample_context(tmp_path, change):
    item = {'box':np.array([0,2,2,4,10,0,-1.6,0]), 'points':np.ones((20,4)), 'r_origin':10., 'num_points':20}
    item.update(change)
    path = tmp_path / 'bad.pkl'
    path.write_bytes(pickle.dumps({'Car':[item]}))
    with pytest.raises(ValueError, match='Car.*sample 0'):
        GTSampler(str(path), sample_counts={'Car':1})


def test_database_requested_class_missing_and_zero_counts_allowed(tmp_path):
    path = tmp_path / 'empty.pkl'
    path.write_bytes(pickle.dumps({}))
    with pytest.raises(ValueError, match='Car'):
        GTSampler(str(path), sample_counts={'Car':1})
    GTSampler(str(path), sample_counts={})
    GTSampler(str(path), sample_counts={'Car':0})

"""Exact rotated NMS without GEOS work for disjoint bounding boxes."""
import numpy as np
import pytest
import postprocess


def legacy_nms(boxes, scores, threshold):
    polygons = postprocess.convert_format(boxes)
    indices = scores.argsort()[::-1]
    picked = []
    while len(indices):
        picked.append(indices[0])
        overlaps = postprocess.compute_iou(polygons[indices[0]], polygons[indices[1:]])
        indices = np.delete(indices, np.where(overlaps > threshold)[0] + 1)
        indices = np.delete(indices, 0)
    return np.asarray(picked, np.int32)


def rectangles(seed, count):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, (count, 2))
    sizes = rng.uniform(.05, 4, (count, 2))
    angles = rng.uniform(-np.pi, np.pi, count)
    local = np.array([[-.5,.5],[-.5,-.5],[.5,-.5],[.5,.5]])[None] * sizes[:,None]
    c, s = np.cos(angles), np.sin(angles)
    rotations = np.stack([c,-s,s,c],-1).reshape(-1,2,2)
    return (local @ rotations.transpose(0,2,1) + centers[:,None]).astype(np.float32)


@pytest.mark.parametrize('threshold', [0., .1, .5, 1.])
@pytest.mark.parametrize('tied_scores', [False, True])
def test_rotated_nms_preserves_legacy_selection_and_order(threshold, tied_scores):
    boxes = rectangles(42, 100)
    boxes[-3:] = boxes[:3]
    scores = np.ones(len(boxes), np.float32) if tied_scores else np.random.default_rng(7).random(len(boxes)).astype(np.float32)
    np.testing.assert_array_equal(postprocess.non_max_suppression(boxes,scores,threshold),
                                  legacy_nms(boxes,scores,threshold))


def test_disjoint_candidates_do_not_invoke_polygon_iou(monkeypatch):
    boxes = rectangles(5, 40)
    boxes += np.arange(40)[:,None,None] * 20
    original = postprocess.compute_iou
    checked = []
    def count_pairs(polygon, others):
        checked.append(len(others))
        return original(polygon, others)
    monkeypatch.setattr(postprocess,'compute_iou',count_pairs)
    scores = np.linspace(.9,.1,len(boxes),dtype=np.float32)
    np.testing.assert_array_equal(postprocess.non_max_suppression(boxes,scores,.1), np.arange(40))
    assert sum(checked) == 0


@pytest.mark.parametrize('threshold', [0., .1, .5, 1.])
def test_touching_and_degenerate_rectangles_preserve_legacy_results(threshold):
    local = np.array([[-.5,.5],[-.5,-.5],[.5,-.5],[.5,.5]],np.float32)
    boxes = np.stack([local,local+[1,0],local+[0,1],local+[.25,.25],local*0,
                      local*np.array([0,1]),local+[(1-threshold)/(1+threshold),0]]).astype(np.float32)
    scores = np.linspace(.9,.1,len(boxes),dtype=np.float32)
    np.testing.assert_array_equal(postprocess.non_max_suppression(boxes,scores,threshold),
                                  legacy_nms(boxes,scores,threshold))

"""Ensure instance visualizations retain the correct raw-frame provenance."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "Code"
sys.path.insert(0, str(ROOT))
from tools.visualization.export_kitti_height_detections import (
    verified_cache, point_owners, crop_indices, instance_color,
)


def cache_fixture(directory):
    points = np.array([[10, 1, -1.6, .5], [10, 1, -.6, .6]], dtype=np.float32)
    scan = directory/'005663.bin'
    points.tofile(scan)
    rows = np.array([[0, .8, 10, 1, 4, 2, 0]], dtype=np.float32)
    np.save(directory/'005663_bev_predictions.npy', rows)
    config = {'data':{'kitti':{'objects':{'Car':0}}}}
    (directory/'config.inference.json').write_text(json.dumps(config))
    source = {'frame_id':'005663','box_source':'fresh checkpoint inference','ground_truth_boxes_used':False,
              'inference':{'box_mode':'bev','config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()},
              'source_bin':{'path':str(scan),'sha256':hashlib.sha256(scan.read_bytes()).hexdigest()},
              'predictions':[{'bev_row':rows[0].tolist()}]}
    (directory/'metadata.json').write_text(json.dumps(source))
    return scan, points, rows


@pytest.mark.parametrize('tamper', ['scan','rows','config','ground_truth'])
def test_rejects_cache_from_wrong_input_or_unverified_predictions(tmp_path,tamper):
    scan, points, rows = cache_fixture(tmp_path)
    _, actual_points, actual_rows, objects, _ = verified_cache(tmp_path)
    np.testing.assert_array_equal(actual_points, points)
    np.testing.assert_array_equal(actual_rows, rows)
    assert objects[0].name=='Car'
    if tamper=='scan':
        altered=points.copy(); altered[0,2]=0; altered.tofile(scan)
    elif tamper=='rows':
        altered=rows.copy(); altered[0,2]=15
        np.save(tmp_path/'005663_bev_predictions.npy',altered)
    elif tamper=='config':
        (tmp_path/'config.inference.json').write_text('{"data":{"kitti":{"objects":{"Pedestrian":0}}}}')
    else:
        path=tmp_path/'metadata.json'; m=json.loads(path.read_text());m['ground_truth_boxes_used']=True
        path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        verified_cache(tmp_path)


def test_foreground_overlap_uses_confidence_and_keeps_source_indices():
    objects = [SimpleNamespace(index=7,score=.7),SimpleNamespace(index=3,score=.9)]
    reports = {7:{'support_indices':[1,4]},3:{'support_indices':[4,6]}}
    owners = point_owners(8,objects,reports)
    np.testing.assert_array_equal(owners,[-1,7,-1,-1,3,-1,3,-1])
    np.testing.assert_array_equal(point_owners(8,objects[::-1],reports),owners)
    assert len({instance_color(index) for index in range(100)})==100
    assert instance_color(7)==instance_color(objects[0].index)


def test_zoom_crop_retains_original_indices_and_excludes_nonfinite_points():
    points = np.array([[0,0,0,.5],[10,2,-1.2,.5],[np.nan,2,0,.5],[11,2,.1,.5],[10,4,0,.5]])
    indices = crop_indices(points,[9,11,1,3,-1.5,.5])
    np.testing.assert_array_equal(indices,[1,3])
    np.testing.assert_array_equal(points[indices],points[[1,3]])

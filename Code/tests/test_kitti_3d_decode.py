"""Complete boxes keep vertical/quality association through NMS and calibration."""

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from test_grouped_decode import GEOMETRY, OBJECTS, groups, grouped_predictions, flat_predictions
import postprocess


def decode(pred, **kwargs):
    assert hasattr(postprocess,"filter_pred_3d"), "Explicit 3D decoder is missing"
    return postprocess.filter_pred_3d(pred,{"geometry":GEOMETRY,"objects":OBJECTS},4,.05,.1,
                                     task_groups=groups() if "groups" in pred else None,**kwargs)


def conversion():
    root=Path(__file__).resolve().parents[1]
    assert (root/"tools/kitti_training_pipeline/kitti_box_conversion.py").is_file(), "Calibration conversion is missing"
    return importlib.import_module("tools.kitti_training_pipeline.kitti_box_conversion")


@pytest.mark.parametrize("cap",[1,2,7])
def test_grouped_vertical_rows_follow_bev_quality_class_mapping_nms_and_global_cap(cap):
    pred=grouped_predictions(iqa=True)
    before=copy.deepcopy(pred)
    for i,h in enumerate(pred["groups"].values()):
        h["vertical"]=torch.zeros_like(h["offset"])
        h["vertical"][:,0]=-(i+1)
        h["vertical"][:,1]=np.log(1.5+i)
    actual=decode(pred,max_detections=cap,use_iou=True)
    expected=postprocess.filter_pred(before,{"geometry":GEOMETRY,"objects":OBJECTS},4,.05,.1,
                                    task_groups=groups(),use_iou=True,max_detections=cap)
    assert actual.shape==(len(expected),9) and actual.dtype==np.float32
    np.testing.assert_allclose(actual[:,[0,1,2,3,5,6,8]],expected,rtol=1e-6,atol=1e-6)
    for row in actual:
        np.testing.assert_allclose(row[[4,7]],[-1.,1.5] if row[0]==0 else [-2.,2.5],atol=1e-6)


def test_empty_3d_detections_keep_nine_columns():
    pred=grouped_predictions(empty=True)
    for h in pred["groups"].values():h["vertical"]=torch.zeros_like(h["offset"])
    assert decode(pred).shape==(0,9)


@pytest.mark.parametrize("problem",["missing","wrong_width","nan","overflow","underflow"])
def test_invalid_vertical_decode_cannot_be_hidden_by_no_candidates(problem):
    pred=grouped_predictions(empty=True)
    for h in pred["groups"].values():h["vertical"]=torch.zeros_like(h["offset"])
    h=pred["groups"]["car"]
    if problem=="missing":del h["vertical"]
    elif problem=="wrong_width":h["vertical"]=h["vertical"][:,:1]
    elif problem=="nan":h["vertical"][:,0]=float("nan")
    elif problem=="overflow":h["vertical"][:,1]=1000.
    else:h["vertical"][:,1]=-1000.
    with pytest.raises((ValueError,KeyError,FloatingPointError),match="vertical|3D|finite|height"):
        decode(pred)


def test_binary_3d_flat_decoder_excludes_background_winners():
    pred=flat_predictions()
    pred["cls"]=torch.full((1,4,3,5),-20.)
    pred["cls"][:,0]=20.
    pred["vertical"]=torch.zeros_like(pred["offset"])
    assert decode(pred,cls_encoding="binary").shape==(0,9)
    pred["cls"][0,3,1,2]=30.
    rows=decode(pred,cls_encoding="binary")
    assert rows.shape==(1,9) and rows[0,0]==2


def calibration():
    module=conversion()
    base=np.array([[0,-1,0],[0,0,-1],[1,0,0]],float)
    c,s=np.cos(.23),np.sin(.23)
    turn=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    c,s=np.cos(.015),np.sin(.015)
    tilt=np.array([[1,0,0],[0,c,-s],[0,s,c]])
    transform=np.eye(4);transform[:3,:3]=tilt@turn@base;transform[:3,3]=[.2,-.1,.3]
    projection=np.array([[500,0,620,0],[0,500,180,0],[0,0,1,0]],float)
    return module.Calibration(transform,projection)


@pytest.mark.parametrize("yaw",[-1.1,0.,.8])
def test_bottom_center_dimensions_and_heading_roundtrip_under_nontrivial_calibration(yaw):
    module=conversion();calib=calibration()
    camera=np.array([1.,1.8,20.,4.2,1.8,1.6,yaw])
    lidar=module.camera_box_to_lidar(camera,calib)
    actual=module.lidar_box_to_camera(lidar,calib)
    np.testing.assert_allclose(actual[:6],camera[:6],atol=1e-9)
    np.testing.assert_allclose([np.cos(actual[6]),np.sin(actual[6])],[np.cos(yaw),np.sin(yaw)],atol=1e-9)


def test_prepare_labels_to_conversion_roundtrip_preserves_existing_bottom_z_convention(tmp_path):
    module=conversion();calib=calibration()
    path=tmp_path/"calib.txt"
    path.write_text("P2: "+" ".join(map(str,calib.p2.ravel()))+"\nR0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: "+" ".join(map(str,calib.velo_to_rect[:3].ravel()))+"\n")
    label=tmp_path/"label.txt";label.write_text("Car 0 0 -10 10 20 100 120 1.6 1.8 4.2 1 1.8 20 0.8\n")
    from tools.kitti_training_pipeline.prepare_kitti import convert_labels
    processed,_=convert_labels(label,path)
    fields=np.array([float(v) for v in processed[0].split()[1:]]) # h,w,l,x,y,z,yaw
    lidar=fields[[3,4,5,2,1,0,6]]
    actual=module.lidar_box_to_camera(lidar,module.read_calibration(path))
    np.testing.assert_allclose(actual,[1,1.8,20,4.2,1.8,1.6,.8],atol=2e-6)


def test_projection_uses_eight_corners_and_explicit_near_plane_image_clipping():
    module=conversion();calib=calibration()
    box=np.array([0.,1.5,20.,4.,1.8,1.5,.3])
    corners=module.camera_box_corners(box)
    assert corners.shape==(8,3)
    assert set(np.round(corners[:,1],8))=={0.,1.5}
    box2d=module.project_camera_box(box,calib,image_size=(1242,375))
    assert box2d.shape==(4,) and np.isfinite(box2d).all() and box2d[3]>box2d[1]
    crossing=box.copy();crossing[2]=.2
    clipped=module.project_camera_box(crossing,calib,image_size=(1242,375),near_plane=.1)
    assert clipped is not None and np.isfinite(clipped).all()
    behind=box.copy();behind[2]=-20
    assert module.project_camera_box(behind,calib,image_size=(1242,375)) is None


def test_prediction_annotation_uses_reference_dimension_order_and_no_aos():
    module=conversion();calib=calibration()
    lidar=module.camera_box_to_lidar([0,1.5,10,4,1.8,1.5,.2],calib)
    rows=np.array([[0,.9,*lidar]],np.float32)
    annotation=module.predictions_to_annotation(rows,calib,OBJECTS,image_size=(1242,375))
    assert annotation["name"].tolist()==["Car"]
    np.testing.assert_allclose(annotation["dimensions"],[ [4,1.5,1.8] ],atol=1e-6) # l,h,w
    assert annotation["alpha"].tolist()==[-10.]
    assert annotation["bbox"][0,3]-annotation["bbox"][0,1]>40
    assert module.predictions_to_annotation(np.empty((0,9)),calib,OBJECTS)["bbox"].shape==(0,4)


@pytest.mark.parametrize("field",["transform","projection"])
def test_invalid_calibration_is_rejected(field):
    module=conversion();calib=calibration()
    transform=calib.velo_to_rect.copy();projection=calib.p2.copy()
    if field=="transform":transform[:3,:3]=0
    else:projection[0,0]=np.nan
    with pytest.raises(ValueError,match="calibration|Calibration"):
        module.Calibration(transform,projection)

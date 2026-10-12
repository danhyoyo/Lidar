"""Pinned reference KITTI semantics, both metrics and both recall samplings."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tools/kitti_training_pipeline")]


def api():
    assert (ROOT/"tools/kitti_training_pipeline/evaluate_kitti_3d.py").is_file(), "Reference evaluator wrapper is missing"
    return importlib.import_module("tools.kitti_training_pipeline.evaluate_kitti_3d")


def gpu_reference():
    module=api()
    from numba import cuda
    if not cuda.is_available():pytest.skip("Reference needs compatible real Numba CUDA; no CPU substitute")
    return module.reference_module()


def annotation():
    return {"name":np.array(["Car","Pedestrian","Cyclist"]),
            "bbox":np.tile([10.,10.,100.,100.],(3,1)),
            "dimensions":np.array([[4.,1.5,1.8],[.8,1.7,.6],[1.2,1.6,.7]]),
            "location":np.array([[0.,1.5,10.],[10.,1.7,10.],[20.,1.6,10.]]),
            "rotation_y":np.zeros(3),"alpha":np.full(3,-10.),"score":np.array([.9,.85,.8]),
            "truncated":np.zeros(3),"occluded":np.zeros(3,dtype=int)}


def scenario(kind):
    gt,dt=annotation(),annotation()
    if kind=="disjoint":dt["location"][:,0]+=100
    elif kind=="vertical_mismatch":dt["location"][:,1]+=5
    elif kind=="duplicates":
        for k in dt:dt[k]=np.concatenate([dt[k],dt[k]])
        dt["score"][3:]=.99
    elif kind=="neighbors":
        for k in gt:gt[k]=np.concatenate([gt[k],gt[k][:2]])
        gt["name"]=np.array(["Car","Pedestrian","Cyclist","Van","Person_sitting"])
        gt["location"][3:,0]+=60
        for k in dt:dt[k]=np.concatenate([dt[k],dt[k][:2]])
        dt["location"][3:,0]+=60
    elif kind=="dontcare":
        for k in gt:gt[k]=np.concatenate([gt[k],gt[k][:1]])
        gt["name"]=np.array(["Car","Pedestrian","Cyclist","DontCare"])
        gt["location"][3,0]=100
        for k in dt:dt[k]=np.concatenate([dt[k],dt[k][:1]])
        dt["location"][3,0]=100;dt["score"][3]=.99
    elif kind=="dt_height_boundary":dt["bbox"][0]=[10,10,100,49.9]
    elif kind=="gt_height_boundary":gt["bbox"][0]=[10,10,100,50.]
    return [copy.deepcopy(gt) for _ in range(41)],[copy.deepcopy(dt) for _ in range(41)]


@pytest.mark.parametrize("mode",["3d","bev"])
@pytest.mark.parametrize("kind",["perfect","disjoint","vertical_mismatch","duplicates","neighbors","dontcare","dt_height_boundary","gt_height_boundary"])
def test_wrapper_matches_pinned_reference_for_ignored_difficulty_overlap_and_sampling(mode,kind):
    reference=gpu_reference();gt,dt=scenario(kind)
    result=api().evaluate_annotations(gt,dt,metric_mode=mode)
    thresholds=np.tile([.7,.5,.5],(1,3,1))
    direct=reference.eval_class(gt,dt,[0,1,2],[0,1,2],2 if mode=="3d" else 1,thresholds,num_parts=1)
    for sampling,function in [("R11",reference.get_mAP),("R40",reference.get_mAP_R40)]:
        expected=function(direct["precision"])[...,0]
        for i,name in enumerate(["Car","Pedestrian","Cyclist"]):
            for j,difficulty in enumerate(["Easy","Moderate","Hard"]):
                assert result[sampling]["per_class"][name][difficulty]==pytest.approx(expected[i,j],abs=1e-9)
        if kind=="perfect" or (kind=="vertical_mismatch" and mode=="bev"):
            assert result[sampling]["map_moderate_percent"]==pytest.approx(100.)
        if kind=="disjoint" or (kind=="vertical_mismatch" and mode=="3d"):
            assert result[sampling]["map_moderate_percent"]==0
        if kind=="dontcare":assert result[sampling]["per_class"]["Car"]["Moderate"]<100
        if "height_boundary" in kind:
            assert result[sampling]["per_class"]["Car"]["Easy"]==0
            assert result[sampling]["per_class"]["Car"]["Moderate"]==pytest.approx(100.)


def test_pinned_source_hashes_and_revision_are_checked_before_import():
    provenance=api().reference_provenance()
    assert provenance["revision"]=="233f849829b6ac19afb8af8837a0246890908755"
    assert provenance["source_hashes_verified"] is True
    assert {"eval.py","rotate_iou.py","LICENSE"}<=set(provenance["sha256"])


def test_label_parser_preserves_neighbors_dontcare_and_reference_dimension_order(tmp_path):
    path=tmp_path/"labels.txt"
    path.write_text("Car 0 0 -10 10 20 100 120 1.6 1.8 4.2 1 1.8 20 0.8\nVan 0 1 -10 10 20 100 120 2 2 5 3 2 20 0\nDontCare -1 -1 -10 2 3 40 60 -1 -1 -1 -1000 -1000 -1000 -10\n")
    result=api().read_annotation(path)
    assert result["name"].tolist()==["Car","Van","DontCare"]
    np.testing.assert_array_equal(result["dimensions"][0],[4.2,1.6,1.8])
    assert result["occluded"].dtype.kind=="i"
    path.write_text("");assert api().read_annotation(path)["bbox"].shape==(0,4)


@pytest.mark.parametrize("line",["Car 0\n","Car 0 0 -10 1 2 3 4 1 1 1 0 1 10 0 nan\n"])
def test_malformed_prediction_labels_are_rejected(line,tmp_path):
    path=tmp_path/"pred.txt";path.write_text(line)
    with pytest.raises(ValueError):api().read_annotation(path,prediction=True)


@pytest.mark.parametrize("mode",[None,"AP3D","local_bev"])
def test_unknown_reference_mode_is_rejected_before_cuda(mode):
    with pytest.raises(ValueError,match="metric_mode"):
        api().evaluate_annotations([],[],metric_mode=mode)


def test_reference_cli_saved_predictions_and_modes_report_both_samplings(tmp_path):
    gpu_reference()
    root=tmp_path/"training";(root/"label_2").mkdir(parents=True)
    predictions=tmp_path/"predictions";predictions.mkdir()
    ids=[]
    for i in range(41):
        identifier=f"{i:06d}";ids.append(identifier)
        text="Car 0 0 -10 10 10 100 100 1.5 1.8 4 0 1.5 10 0"
        (root/"label_2"/f"{identifier}.txt").write_text(text+"\n")
        (predictions/f"{identifier}.txt").write_text(text+" 0.9\n")
    split=tmp_path/"split.txt";split.write_text("\n".join(ids)+"\n")
    for mode in ("3d","bev"):
        output=tmp_path/f"{mode}.json"
        result=subprocess.run([sys.executable,str(ROOT/"tools/kitti_training_pipeline/evaluate_kitti_3d.py"),
            "--predictions",str(predictions),"--kitti-root",str(root),"--split",str(split),
            "--metric-mode",mode,"--output",str(output)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        report=json.loads(output.read_text());assert report["protocol"]["metric_mode"]==mode
        assert set(report["accuracy"])=={"R11","R40"}
        assert report["accuracy"]["R40"]["per_class"]["Car"]["Moderate"]==pytest.approx(100.)
        assert report["protocol"]["domain"]=="full benchmark GT"
        assert report["protocol"]["aos"] is False


def test_comparison_exports_separate_reference_mode_and_recall_sampling_fields():
    module=api()
    from compare_models import flatten,markdown_table,parser
    assert parser().parse_args(["--model","a=pytorch:a.pt","--config","c.json","--detector-root","detector",
                               "--kitti-root","k","--split","s","--output-dir","o","--metric-mode","3d"]).metric_mode=="3d"
    values={s:{"map_moderate_percent":v,"mean_ap_9_percent":v,"per_class":{}} for s,v in [("R11",70.),("R40",65.)]}
    result={"name":"reference","status":"ok","protocol":{"metric_mode":"3d"},"accuracy":values}
    row=flatten(result)
    assert row["metric_mode"]=="3d" and row["3d_R11_map_moderate_percent"]==70 and row["3d_R40_map_moderate_percent"]==65
    assert "AP3D" in markdown_table([result]) and "R11" in markdown_table([result])


@pytest.mark.parametrize("mode", ["3d", "bev"])
def test_checkpoint_dataset_calibration_and_comparison_cli_use_reference_metrics(tmp_path, mode):
    gpu_reference()
    import math
    import struct
    import zlib
    import torch
    from test_vertical_head import config_3d
    from test_grouped_targets import make_dataset
    from test_grouped_checkpoint import components, payload
    from tools.kitti_training_pipeline.kitti_box_conversion import Calibration, predictions_to_annotation
    from compare_models import main
    from common import sha256

    dataset = make_dataset(tmp_path)
    config = config_3d()
    config["data"] = copy.deepcopy(dataset.config)
    config["data"]["box_mode"] = "3d"
    config["augmentation"] = {"p":0.,"rotation":{"use":False},"scaling":{"use":False},"translation":{"use":False}}
    parts = components(config)
    with torch.no_grad():
        for name, head in parts[0].grouped_header.heads.items():
            for parameter in head.parameters():
                parameter.zero_()
            head.cls.head.bias.copy_(torch.logit(torch.tensor([.9] if name == "car" else [.8,.7])))
            head.offset.head.bias.copy_(torch.tensor([8.,3.2]))
            head.size.head.bias.fill_(math.log(20.))
            head.yaw.head.bias[0] = 1.
            head.vertical.head.bias.copy_(torch.tensor([-1.5,math.log(1.5)]))
    checkpoint = tmp_path/"checkpoint.pt"
    torch.save(payload(config,parts,epoch=0),checkpoint)
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(config))
    root = tmp_path/"training"
    for folder in ("label_2","calib","image_2"):
        (root/folder).mkdir(parents=True)
    transform = np.array([[0.,-1.,0.,0.],[0.,0.,-1.,0.],[1.,0.,0.,0.],[0.,0.,0.,1.]])
    p2 = np.array([[700.,0.,620.,0.],[0.,700.,180.,0.],[0.,0.,1.,0.]])
    # Legacy NMS resolves equal-score plateaus from the final flattened cell.
    # This fixture's stride-4 map is 16x12, with 0.4m/0.8m Y/X spacing.
    rows = np.array([[i,.9,8.+11*.8,15*.4,-1.5,20.,20.,1.5,0.] for i in range(3)],np.float32)
    expected = predictions_to_annotation(rows,Calibration(transform,p2),config["data"]["kitti"]["objects"],image_size=(1242,375))
    labels = []
    for i, name in enumerate(expected["name"]):
        length,height,width = expected["dimensions"][i]
        values = [0,0,-10,*expected["bbox"][i],height,width,length,*expected["location"][i],expected["rotation_y"][i]]
        labels.append(name+" "+" ".join(str(v) for v in values))
    def png_chunk(kind,data):
        return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data))
    png = (b"\x89PNG\r\n\x1a\n"+png_chunk(b"IHDR",struct.pack(">IIBBBBB",1242,375,8,0,0,0,0))+
           png_chunk(b"IDAT",zlib.compress(bytes(1243*375)))+png_chunk(b"IEND",b""))
    ids = [f"{i:06d}" for i in range(41)]
    processed = Path(config["data"]["kitti"]["location"])
    pointcloud = (processed/"pointcloud/000000.bin").read_bytes()
    processed_label = (processed/"label/000000.txt").read_text()
    for identifier in ids:
        (processed/"pointcloud"/f"{identifier}.bin").write_bytes(pointcloud)
        (processed/"label"/f"{identifier}.txt").write_text(processed_label)
        (root/"label_2"/f"{identifier}.txt").write_text("\n".join(labels)+"\n")
        (root/"calib"/f"{identifier}.txt").write_text(
            "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\nP2: 700 0 620 0 0 700 180 0 0 0 1 0\n")
        (root/"image_2"/f"{identifier}.png").write_bytes(png)
    split = tmp_path/"split.txt"
    split.write_text("\n".join(ids)+"\n")
    output = tmp_path/"comparison"
    main(["--model",f"missing=pytorch:{tmp_path/'missing.pt'}","--model",f"synthetic=pytorch:{checkpoint}","--config",str(config_path),"--detector-root",str(ROOT/"detector"),
          "--kitti-root",str(root),"--split",str(split),"--metric-mode",mode,"--output-dir",str(output),
          "--device","cpu","--score-threshold",".1","--max-detections","3","--warmup-frames","0","--progress-every","0"])
    result = json.loads((output/"synthetic.json").read_text())
    assert result["protocol"]["metric_mode"] == mode
    assert result["protocol"]["quality_target"] == "BEV IQA"
    assert result["counts"]["detections"] == 123
    assert result["model"]["parameter_counts"]["backbone_including_neck"] == 660528
    assert result["model"]["parameter_counts"]["total_detector"] == 883941
    assert result["model"]["parameter_counts"]["criterion_train_only"] == 12
    assert result["model"]["checkpoint_epoch"] == 0
    assert result["data"]["resolved_config_sha256"] == sha256(config_path)
    assert len(result["data"]["calibration_sha256"]) == 41
    for sampling in ("R11","R40"):
        assert result["accuracy"][sampling]["map_moderate_percent"] == pytest.approx(100.)
    assert f"{mode}_R40_map_moderate_percent" in (output/"comparison.csv").read_text()
    comparison = json.loads((output/"comparison.json").read_text())
    assert comparison["status"] == "partial" and comparison["models_succeeded"] == 1
    assert comparison["models"][0]["protocol"]["metric_mode"] == mode

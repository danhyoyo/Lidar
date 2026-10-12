#!/usr/bin/env python3
"""Pinned KITTI reference AP3D/APBEV, each reporting R11 and R40."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    from .common import (configure_detector_imports, detection_spec, read_json, sha256, write_json,
                         checkpoint_identity, git_metadata, evaluation_asset_hashes)
    from .kitti_box_conversion import empty_annotation, predictions_to_annotation, read_calibration
    from .prepare_kitti import read_ids
except ImportError:
    from common import (configure_detector_imports, detection_spec, read_json, sha256, write_json,
                        checkpoint_identity, git_metadata, evaluation_asset_hashes)
    from kitti_box_conversion import empty_annotation, predictions_to_annotation, read_calibration
    from prepare_kitti import read_ids

CLASSES = ("Car", "Pedestrian", "Cyclist")
DIFFICULTIES = ("Easy", "Moderate", "Hard")


def reference_provenance():
    directory = Path(__file__).with_name("kitti_reference")
    record = read_json(directory / "provenance.json")
    for name, expected in record["sha256"].items():
        if sha256(directory / name) != expected:
            raise RuntimeError(f"Pinned reference source hash mismatch: {name}")
    return {**record, "source_hashes_verified": True}


def reference_module():
    reference_provenance()
    from numba import cuda
    if not cuda.is_available():
        raise RuntimeError("Reference KITTI evaluation requires compatible real Numba CUDA and libNVVM/libdevice; no CPU/local BEV fallback")
    package = f"{__package__}.kitti_reference.eval" if __package__ else "kitti_reference.eval"
    return importlib.import_module(package)


def read_annotation(path, *, prediction=False):
    """Read original KITTI fields, retaining neighbors/DontCare and full GT domain."""
    annotation = empty_annotation()
    rows = {key: [] for key in annotation}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != (16 if prediction else 15):
            raise ValueError(f"Malformed KITTI annotation {path}:{number}")
        values = np.array([float(v) for v in fields[1:]])
        if not np.isfinite(values).all() or values[1] != int(values[1]):
            raise ValueError(f"Non-finite/invalid KITTI annotation {path}:{number}")
        if prediction and (not 0 <= values[-1] <= 1 or (values[7:10] <= 0).any()):
            raise ValueError(f"Invalid prediction dimensions/score {path}:{number}")
        row = {"name": fields[0], "truncated": values[0], "occluded": int(values[1]),
               "alpha": values[2], "bbox": values[3:7], "dimensions": values[[9,7,8]],
               "location": values[10:13], "rotation_y": values[13],
               "score": values[14] if prediction else 0.}
        for key in rows:
            rows[key].append(row[key])
    if rows["name"]:
        annotation = {key: np.asarray(value, dtype=str if key == "name" else annotation[key].dtype)
                      .reshape((-1, *annotation[key].shape[1:])) for key, value in rows.items()}
    return annotation


def evaluate_annotations(ground_truth, detections, *, metric_mode="3d"):
    if metric_mode not in {"3d", "bev"}:
        raise ValueError("metric_mode must be 3d or bev")
    if not ground_truth or len(ground_truth) != len(detections):
        raise ValueError("Reference evaluation requires nonempty aligned GT/detection frame lists")
    reference = reference_module()
    overlaps = np.tile([.7, .5, .5], (1, 3, 1))
    curves = reference.eval_class(ground_truth, detections, [0, 1, 2], [0, 1, 2],
                                  2 if metric_mode == "3d" else 1, overlaps,
                                  compute_aos=False, num_parts=50)
    result = {}
    for sampling, function in (("R11", reference.get_mAP), ("R40", reference.get_mAP_R40)):
        values = function(curves["precision"])[..., 0]
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite reference AP")
        result[sampling] = {
            "per_class": {name: {difficulty: float(values[i,j]) for j,difficulty in enumerate(DIFFICULTIES)}
                          for i,name in enumerate(CLASSES)},
            "map_moderate_percent": float(values[:,1].mean()),
            "mean_ap_9_percent": float(values.mean()),
            "valid_gt_counts": {name: {difficulty: sum(reference.clean_data(g,d,i,j)[0]
                    for g,d in zip(ground_truth,detections)) for j,difficulty in enumerate(DIFFICULTIES)}
                    for i,name in enumerate(CLASSES)},
            "macro_classes": list(CLASSES), "missing_class_policy": "all three classes retained; valid GT counts recorded"}
    return result


def png_image_size(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Expected KITTI PNG image with IHDR: {path}")
    return struct.unpack(">II", header[16:24])


def _report(ids, ground_truth, detections, *, metric_mode, root, split_path, source):
    import numba
    from numba import cuda
    result = {"status": "ok", "accuracy": evaluate_annotations(ground_truth, detections, metric_mode=metric_mode),
        "protocol": {"metric_mode": metric_mode, "ap_samplings": ["R11", "R40"],
            "domain": "full benchmark GT", "classes": list(CLASSES), "difficulties": list(DIFFICULTIES),
            "iou_thresholds": dict(zip(CLASSES,[.7,.5,.5])), "aos": False,
            "ignored_neighbor_dontcare": "unchanged pinned reference, metric-specific",
            "projection": "eight upright camera box corners, edge-clipped z>=0.1m then actual image rectangle",
            "orientation": "pi-symmetric box yaw; no AOS",
            "reference": reference_provenance()},
        "data": {"frames": len(ids), "frame_ids": ids, "split_sha256": sha256(split_path),
            "kitti_root": str(Path(root).resolve()),
            "label_sha256": {identifier: sha256(Path(root)/"label_2"/f"{identifier}.txt") for identifier in ids}},
        "counts": {"detections": sum(len(a["name"]) for a in detections),
            "all_gt_labels": sum(len(a["name"]) for a in ground_truth)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
            "numba_cuda_module": cuda.__file__, "cuda_runtime": list(cuda.runtime.get_version()),
            "CUDA_HOME": os.environ.get("CUDA_HOME"), "reference_source": source}}
    return result


def evaluate_saved_predictions(*, predictions, kitti_root, split_path, metric_mode="3d", output_path=None):
    ids = read_ids(Path(split_path))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Split must contain unique nonempty frame IDs")
    root, predictions = Path(kitti_root), Path(predictions)
    gt = [read_annotation(root/"label_2"/f"{identifier}.txt") for identifier in ids]
    dt = [read_annotation(predictions/f"{identifier}.txt",prediction=True) for identifier in ids]
    result = _report(ids,gt,dt,metric_mode=metric_mode,root=root,split_path=split_path,source="saved KITTI prediction files")
    result["data"]["prediction_sha256"] = {identifier:sha256(predictions/f"{identifier}.txt") for identifier in ids}
    if output_path is not None:
        write_json(Path(output_path),result)
    return result


def run_evaluation(*, name, backend, model_path, config_path, detector_root, kitti_root,
                   split_path, output_path=None, device="cuda", metric_mode="3d", score_threshold=.05,
                   nms_threshold=.10, max_detections=500, warmup_frames=0, max_frames=None,
                   progress_every=50):
    """Infer a complete 3D checkpoint; reference mode may be AP3D or APBEV."""
    started = time.time()
    if backend != "pytorch":
        raise ValueError("Reference 3D checkpoint evaluation currently supports PyTorch only")
    if metric_mode not in {"3d","bev"}:
        raise ValueError("metric_mode must be 3d or bev")
    if not 0 <= score_threshold <= 1 or not 0 <= nms_threshold <= 1 or type(max_detections) is not int or max_detections < 1:
        raise ValueError("Invalid score/NMS threshold or detection cap")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be positive")
    if type(warmup_frames) is not int or warmup_frames < 0:
        raise ValueError("warmup_frames must be nonnegative")
    reference_module()  # Dependency failure must occur before expensive inference.
    configure_detector_imports(detector_root)
    import torch
    from core.datasets.dataset import Dataset
    from postprocess import filter_pred_3d
    try:
        from .evaluate_kitti_bev import PyTorchRunner
    except ImportError:
        from evaluate_kitti_bev import PyTorchRunner
    config = read_json(Path(config_path))
    detection = detection_spec(config)
    if detection.box_mode != "3d":
        raise ValueError("Reference checkpoint mode requires box_mode=3d and real vertical predictions")
    ids = read_ids(Path(split_path))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Split must contain unique nonempty frame IDs")
    if max_frames is not None:
        ids = ids[:max_frames]
    root = Path(kitti_root)
    input_asset_sha256 = evaluation_asset_hashes(config, ids, root, reference=True)
    runner = PyTorchRunner(Path(model_path),config,device,reference_3d=True)
    annotations, gt, calibration_hashes = [], [], {}
    with tempfile.TemporaryDirectory(prefix="kitti-reference-") as directory:
        manifest = Path(directory)/"frames.txt"
        manifest.write_text("".join(f"{identifier};kitti\n" for identifier in ids))
        dataset = Dataset(str(manifest),config["data"],config["augmentation"],detection.cls_encoding,"validation",
                          config.get("train",{}).get("target_backend","python"),
                          task_groups=detection.groups if detection.head_mode == "grouped" else None)
        if warmup_frames:
            tensor,_ = runner.transfer(dataset[0]["voxel"])
            for _ in range(warmup_frames):runner.infer(tensor)
        for i,identifier in enumerate(ids):
            tensor,_ = runner.transfer(dataset[i]["voxel"])
            prediction,_ = runner.infer(tensor)
            rows = filter_pred_3d(prediction,config["data"]["kitti"],4,score_threshold,nms_threshold,
                task_groups=detection.groups if detection.head_mode == "grouped" else None,
                cls_encoding=detection.cls_encoding,use_iou=detection.use_iou,max_detections=max_detections)
            path = root/"calib"/f"{identifier}.txt"
            calibration_hashes[identifier] = sha256(path)
            annotations.append(predictions_to_annotation(rows,read_calibration(path),config["data"]["kitti"]["objects"],
                                                        image_size=png_image_size(root/"image_2"/f"{identifier}.png")))
            gt.append(read_annotation(root/"label_2"/f"{identifier}.txt"))
            if progress_every and (i+1)%progress_every == 0:
                print(f"Reference inference {i+1}/{len(ids)}",flush=True)
    result = _report(ids,gt,annotations,metric_mode=metric_mode,root=root,split_path=split_path,source="3D PyTorch checkpoint")
    result.update(name=name,model={"backend":backend,"path":str(Path(model_path).resolve()),
                                  "sha256":sha256(Path(model_path)), **runner.metadata()})
    result["protocol"].update(head_mode=detection.head_mode,cls_encoding=detection.cls_encoding,
        quality_target="BEV IQA" if detection.use_iou else None,nms_alpha=config["data"]["kitti"].get("nms_alpha",.5),
        score_threshold=score_threshold,nms_threshold=nms_threshold,max_detections=max_detections,
        nms_metric="classwise BEV footprint; one global cap",checkpoint_identity=checkpoint_identity(config))
    result["data"].update(resolved_config_sha256=sha256(Path(config_path)),calibration_sha256=calibration_hashes,
                          input_asset_sha256=input_asset_sha256)
    result["runtime"]["elapsed_seconds"] = time.time()-started
    result["runtime"]["source"] = git_metadata(Path(__file__).resolve().parents[2])
    if output_path is not None:
        write_json(Path(output_path),result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions",type=Path,help="Directory of original KITTI result text files")
    source.add_argument("--checkpoint",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--name",help="Run label for checkpoint comparison reports")
    parser.add_argument("--detector-root",type=Path,default=Path(__file__).resolve().parents[2]/"detector")
    parser.add_argument("--kitti-root",required=True,type=Path,help="Original KITTI training directory")
    parser.add_argument("--split",required=True,type=Path)
    parser.add_argument("--metric-mode",choices=("3d","bev"),default="3d")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--score-threshold",type=float,default=.05)
    parser.add_argument("--nms-threshold",type=float,default=.10)
    parser.add_argument("--max-detections",type=int,default=500)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args(argv)
    if args.predictions:
        result = evaluate_saved_predictions(predictions=args.predictions,kitti_root=args.kitti_root,
            split_path=args.split,metric_mode=args.metric_mode,output_path=args.output)
    else:
        if args.config is None:parser.error("--checkpoint requires --config")
        result = run_evaluation(name=args.name or args.checkpoint.stem,backend="pytorch",model_path=args.checkpoint,
            config_path=args.config,detector_root=args.detector_root,kitti_root=args.kitti_root,split_path=args.split,
            metric_mode=args.metric_mode,device=args.device,score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,max_detections=args.max_detections,output_path=args.output)
    print(json.dumps({"status":result["status"],"metric_mode":args.metric_mode,"AP":result["accuracy"]},indent=2))
    return result


if __name__ == "__main__":
    main()

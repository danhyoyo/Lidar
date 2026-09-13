#!/usr/bin/env python3
"""Evaluate a PyTorch checkpoint or TensorRT plan on a fixed KITTI BEV split.

Reports local loader-aligned KITTI-style rotated BEV AP R40 and offline latency.
This is not a submission to KITTI's hidden official test server.
"""
from __future__ import annotations

import argparse
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from shapely.geometry import Polygon

from common import build_model, configure_detector_imports, normalize_state_dict
from common import read_json, sha256, write_json
from prepare_kitti import normalize_yaw, parse_calibration

CLASSES = ("Car", "Pedestrian", "Cyclist")
CLASS_IDS = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
IOU_THRESHOLDS = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}
NEIGHBOURS = {"Car": {"Van"}, "Pedestrian": {"Person_sitting"}, "Cyclist": set()}
DIFFICULTIES = {
    "Easy": {"min_height": 40.0, "max_occlusion": 0, "max_truncation": 0.15},
    "Moderate": {"min_height": 25.0, "max_occlusion": 1, "max_truncation": 0.30},
    "Hard": {"min_height": 25.0, "max_occlusion": 2, "max_truncation": 0.50},
}


@dataclass(frozen=True)
class GroundTruth:
    name: str
    truncation: float
    occlusion: int
    bbox_height: float
    x: float
    y: float
    length: float
    width: float
    yaw: float

    def passes(self, difficulty: str) -> bool:
        rule = DIFFICULTIES[difficulty]
        return (self.bbox_height >= rule["min_height"]
                and self.occlusion <= rule["max_occlusion"]
                and self.truncation <= rule["max_truncation"])


def read_ids(path: Path) -> List[str]:
    values = [line.strip().split(";", 1)[0]
              for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate frame IDs in {path}")
    return values


def inside_roi(x: float, y: float, geom: Mapping[str, float]) -> bool:
    return geom["x_min"] < x < geom["x_max"] and geom["y_min"] < y < geom["y_max"]


def load_ground_truth(frame_id: str, root: Path,
                      geom: Mapping[str, float]) -> List[GroundTruth]:
    label_path = root / "training" / "label_2" / f"{frame_id}.txt"
    calib_path = root / "training" / "calib" / f"{frame_id}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    if not calib_path.is_file():
        raise FileNotFoundError(calib_path)
    rect_to_velo = np.linalg.inv(parse_calibration(calib_path))
    rotation = rect_to_velo[:3, :3]
    accepted = set(CLASSES).union(*NEIGHBOURS.values())
    result = []
    for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 15:
            raise ValueError(f"Malformed label {label_path}:{line_number}")
        name = fields[0]
        if name not in accepted:
            continue
        truncation, occlusion = float(fields[1]), int(fields[2])
        bbox_height = max(0.0, float(fields[7]) - float(fields[5]))
        _, width, length = map(float, fields[8:11])
        cx, cy, cz = map(float, fields[11:14])
        rotation_y = float(fields[14])
        center = rect_to_velo @ np.array([cx, cy, cz, 1.0])
        heading = rotation @ np.array(
            [math.cos(rotation_y), 0.0, -math.sin(rotation_y)])
        yaw = normalize_yaw(math.atan2(heading[1], heading[0]))
        x, y = float(center[0]), float(center[1])
        if inside_roi(x, y, geom):
            result.append(GroundTruth(name, truncation, occlusion, bbox_height,
                                      x, y, length, width, yaw))
    return result


def polygon(x: float, y: float, length: float, width: float, yaw: float) -> Polygon:
    c, s = math.cos(yaw), math.sin(yaw)
    local = ((-length / 2, width / 2), (-length / 2, -width / 2),
             (length / 2, -width / 2), (length / 2, width / 2))
    return Polygon([(x + dx * c - dy * s, y + dx * s + dy * c)
                    for dx, dy in local])


def iou(first: Polygon, second: Polygon) -> float:
    union = first.union(second).area
    return 0.0 if union <= 0 else float(first.intersection(second).area / union)


def ap_r40(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if not recalls.size:
        return 0.0
    envelope = np.maximum.accumulate(precisions[::-1])[::-1]
    samples = np.arange(1, 41, dtype=np.float64) / 40.0
    return 100.0 * float(np.mean([
        np.max(envelope[recalls >= sample]) if np.any(recalls >= sample) else 0.0
        for sample in samples
    ]))


def evaluate_one(predictions: Mapping[str, np.ndarray],
                 labels: Mapping[str, Sequence[GroundTruth]],
                 class_name: str, difficulty: str) -> Dict[str, Any]:
    valid, ignored, total_gt = {}, {}, 0
    for frame_id, objects in labels.items():
        valid[frame_id], ignored[frame_id] = [], []
        for obj in objects:
            shape = polygon(obj.x, obj.y, obj.length, obj.width, obj.yaw)
            if obj.name == class_name:
                (valid if obj.passes(difficulty) else ignored)[frame_id].append(shape)
            elif obj.name in NEIGHBOURS[class_name]:
                ignored[frame_id].append(shape)
        total_gt += len(valid[frame_id])

    ranked = []
    class_id = CLASS_IDS[class_name]
    for frame_id, boxes in predictions.items():
        for row in boxes:
            if int(row[0]) == class_id:
                ranked.append((float(row[1]), frame_id,
                               polygon(*map(float, row[2:7]))))
    ranked.sort(reverse=True, key=lambda item: item[0])
    used_valid = {frame_id: set() for frame_id in labels}
    used_ignored = {frame_id: set() for frame_id in labels}
    tp, fp, ignored_count = [], [], 0

    for _, frame_id, prediction in ranked:
        candidates = [(iou(prediction, box), index)
                      for index, box in enumerate(valid[frame_id])
                      if index not in used_valid[frame_id]]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= IOU_THRESHOLDS[class_name]:
            used_valid[frame_id].add(best[1])
            tp.append(1.0)
            fp.append(0.0)
            continue
        candidates = [(iou(prediction, box), index)
                      for index, box in enumerate(ignored[frame_id])
                      if index not in used_ignored[frame_id]]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= IOU_THRESHOLDS[class_name]:
            used_ignored[frame_id].add(best[1])
            ignored_count += 1
            continue
        tp.append(0.0)
        fp.append(1.0)

    cumulative_tp = np.cumsum(np.asarray(tp))
    cumulative_fp = np.cumsum(np.asarray(fp))
    recalls = cumulative_tp / total_gt if total_gt else np.zeros_like(cumulative_tp)
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    return {
        "ap_r40_percent": ap_r40(recalls, precisions) if total_gt else None,
        "iou_threshold": IOU_THRESHOLDS[class_name], "ground_truth": total_gt,
        "ranked_predictions": len(ranked), "ignored_predictions": ignored_count,
        "true_positives": int(cumulative_tp[-1]) if cumulative_tp.size else 0,
        "false_positives": int(cumulative_fp[-1]) if cumulative_fp.size else 0,
        "max_recall": float(recalls[-1]) if recalls.size else 0.0,
    }


def evaluate_accuracy(predictions, labels) -> Dict[str, Any]:
    per_class, all_values, moderate_values = {}, [], []
    for class_name in CLASSES:
        difficulty_results = {}
        for difficulty in DIFFICULTIES:
            value = evaluate_one(predictions, labels, class_name, difficulty)
            difficulty_results[difficulty] = value
            if value["ap_r40_percent"] is not None:
                all_values.append(value["ap_r40_percent"])
                if difficulty == "Moderate":
                    moderate_values.append(value["ap_r40_percent"])
        values = [x["ap_r40_percent"] for x in difficulty_results.values()
                  if x["ap_r40_percent"] is not None]
        per_class[class_name] = {
            "difficulties": difficulty_results,
            "mean_ap_r40_percent": float(np.mean(values)) if values else None,
        }
    return {
        "per_class": per_class,
        "mean_ap_9_percent": float(np.mean(all_values)) if all_values else None,
        "map_moderate_percent": (float(np.mean(moderate_values))
                                 if moderate_values else None),
    }


def timing_summary(values) -> Dict[str, Any]:
    if not values:
        return dict(samples=0, mean_ms=None, p50_ms=None, p95_ms=None,
                    p99_ms=None, fps=None)
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    return {"samples": len(values), "mean_ms": mean,
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "fps": 1000.0 / mean if mean else None}


def cuda_timed(function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = function()
    end.record()
    end.synchronize()
    return value, float(start.elapsed_time(end))


class PyTorchRunner:
    def __init__(self, path: Path, config, device: str):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA is required for comparable timings")
        self.model = build_model(config)
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(normalize_state_dict(checkpoint), strict=True)
        self.model = self.model.to(self.device).eval()

    def transfer(self, voxel):
        return cuda_timed(lambda: voxel.unsqueeze(0).contiguous().to(self.device))

    def infer(self, tensor):
        with torch.inference_mode():
            return cuda_timed(lambda: self.model(tensor))

    def metadata(self):
        return {"framework": f"PyTorch {torch.__version__}",
                "device": str(self.device), "precision": "fp32"}


class TensorRTRunner:
    def __init__(self, path: Path, config, device: str):
        if not device.startswith("cuda"):
            raise ValueError("TensorRT requires CUDA")
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError("TensorRT Python package is not installed") from error
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize this GPU/version-specific plan")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16,
                  trt.int32: torch.int32, trt.int8: torch.int8}
        if hasattr(trt, "bool"):
            dtypes[trt.bool] = torch.bool
        self.buffers, self.inputs, self.outputs = {}, [], []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(size < 0 for size in shape):
                raise RuntimeError(f"Dynamic tensor unsupported: {name} {shape}")
            dtype = dtypes.get(self.engine.get_tensor_dtype(name))
            if dtype is None:
                raise RuntimeError(f"Unsupported TensorRT dtype: {name}")
            self.buffers[name] = torch.empty(shape, dtype=dtype, device="cuda")
            mode = self.engine.get_tensor_mode(name)
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(name)
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
        if len(self.inputs) != 1 or set(self.outputs) != {"cls", "offset", "size", "yaw"}:
            raise RuntimeError(f"Unexpected TensorRT IO: {self.inputs}, {self.outputs}")

    def transfer(self, voxel):
        target = self.buffers[self.inputs[0]]
        source = voxel.unsqueeze(0).contiguous()
        if source.shape != target.shape:
            raise ValueError(f"Input {tuple(source.shape)} != engine {tuple(target.shape)}")
        _, elapsed = cuda_timed(lambda: target.copy_(source))
        return target, elapsed

    def infer(self, tensor):
        def execute():
            if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            return {name: self.buffers[name] for name in self.outputs}
        return cuda_timed(execute)

    def metadata(self):
        return {"framework": f"TensorRT {self.trt.__version__}", "device": "cuda",
                "input": self.inputs[0], "outputs": sorted(self.outputs)}


def run_evaluation(*, name: str, backend: str, model_path: Path,
                   config_path: Path, detector_root: Path, kitti_root: Path,
                   split_path: Path, output_path: Path | None = None,
                   device: str = "cuda", score_threshold: float = 0.05,
                   nms_threshold: float = 0.10, max_detections: int = 500,
                   warmup_frames: int = 10, max_frames: int | None = None,
                   progress_every: int = 50) -> Dict[str, Any]:
    started = time.time()
    for path in (model_path, config_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    configure_detector_imports(detector_root)
    from core.datasets.dataset import Dataset
    from postprocess import filter_pred

    config = read_json(config_path)
    geom = config["data"]["kitti"]["geometry"]
    all_ids = read_ids(split_path)
    frame_ids = all_ids[:max_frames] if max_frames else all_ids
    if not frame_ids:
        raise ValueError("Evaluation split is empty")
    dataset = Dataset(str(split_path), config["data"], config["augmentation"],
                      config["model"]["cls_encoding"], task="test")
    runner = (PyTorchRunner(model_path, config, device) if backend == "pytorch"
              else TensorRTRunner(model_path, config, device))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    predictions, labels = {}, {}
    timings = {key: [] for key in
               ("preprocess", "host_to_device", "model", "decode_nms",
                "input_to_detections")}
    detection_count = 0

    for index, frame_id in enumerate(frame_ids):
        total_start = time.perf_counter()
        part_start = time.perf_counter()
        sample = dataset[index]
        preprocess_ms = (time.perf_counter() - part_start) * 1000
        input_tensor, transfer_ms = runner.transfer(sample["voxel"])
        output, model_ms = runner.infer(input_tensor)
        part_start = time.perf_counter()
        boxes = filter_pred(output, config["data"]["kitti"],
                            config["data"]["out_size_factor"],
                            score_threshold, nms_threshold)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - part_start) * 1000
        if boxes.size:
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
            boxes = boxes[np.argsort(boxes[:, 1])[::-1][:max_detections]]
        else:
            boxes = np.empty((0, 7), dtype=np.float32)
        total_ms = (time.perf_counter() - total_start) * 1000
        predictions[frame_id] = boxes
        labels[frame_id] = load_ground_truth(frame_id, kitti_root, geom)
        detection_count += len(boxes)
        if index >= warmup_frames:
            for key, value in (
                ("preprocess", preprocess_ms), ("host_to_device", transfer_ms),
                ("model", model_ms), ("decode_nms", decode_ms),
                ("input_to_detections", total_ms)):
                timings[key].append(value)
        if progress_every and ((index + 1) % progress_every == 0
                               or index + 1 == len(frame_ids)):
            print(f"[{name}] {index + 1}/{len(frame_ids)}, "
                  f"detections={detection_count}", flush=True)

    result = {
        "status": "ok", "name": name,
        "model": {"backend": backend, "path": str(model_path.resolve()),
                  "bytes": model_path.stat().st_size, "sha256": sha256(model_path),
                  **runner.metadata()},
        "data": {"config": str(config_path.resolve()),
                 "detector_root": str(detector_root.resolve()),
                 "kitti_root": str(kitti_root.resolve()),
                 "split": str(split_path.resolve()), "frames": len(frame_ids),
                 "full_split_frames": len(all_ids)},
        "protocol": {
            "name": "local loader-aligned KITTI-style rotated BEV AP R40",
            "official_hidden_test_submission": False,
            "classes": list(CLASSES), "difficulty_rules": DIFFICULTIES,
            "iou_thresholds": IOU_THRESHOLDS,
            "neighbour_class_ignores": {k: sorted(v) for k, v in NEIGHBOURS.items()},
            "roi_rule": "strict LiDAR-frame center inside configured x/y ROI",
            "score_threshold": score_threshold, "nms_threshold": nms_threshold,
            "max_detections_per_frame": max_detections,
            "warmup_frames_excluded_from_latency_only": min(warmup_frames, len(frame_ids)),
            "latency_boundary": "Sequential offline bin load + voxelization + H2D + model + decode/NMS; excludes ROS, tracking and HMI.",
        },
        "accuracy": evaluate_accuracy(predictions, labels),
        "latency": {key: timing_summary(value) for key, value in timings.items()},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                    "torch_peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576.0,
                    "elapsed_seconds": time.time() - started},
        "counts": {"detections": detection_count,
                   "detections_per_frame": detection_count / len(frame_ids)},
    }
    if output_path:
        write_json(output_path, result)
        print(f"Wrote {output_path.resolve()}", flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--name", required=True)
    value.add_argument("--backend", choices=("pytorch", "tensorrt"), required=True)
    value.add_argument("--model", required=True, type=Path)
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--detector-root", required=True, type=Path)
    value.add_argument("--kitti-root", required=True, type=Path)
    value.add_argument("--split", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--score-threshold", type=float, default=0.05)
    value.add_argument("--nms-threshold", type=float, default=0.10)
    value.add_argument("--max-detections", type=int, default=500)
    value.add_argument("--warmup-frames", type=int, default=10)
    value.add_argument("--max-frames", type=int)
    value.add_argument("--progress-every", type=int, default=50)
    return value


def optional(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.nms_threshold <= 1:
        raise ValueError("score/NMS thresholds must be between 0 and 1")
    if args.max_detections <= 0 or args.warmup_frames < 0:
        raise ValueError("invalid max detections or warmup")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    result = run_evaluation(
        name=args.name, backend=args.backend, model_path=args.model,
        config_path=args.config, detector_root=args.detector_root,
        kitti_root=args.kitti_root, split_path=args.split,
        output_path=args.output, device=args.device,
        score_threshold=args.score_threshold, nms_threshold=args.nms_threshold,
        max_detections=args.max_detections, warmup_frames=args.warmup_frames,
        max_frames=args.max_frames, progress_every=args.progress_every)
    print(f"mAP Moderate={optional(result['accuracy']['map_moderate_percent'])}%, "
          f"Mean AP-9={optional(result['accuracy']['mean_ap_9_percent'])}%, "
          f"model FPS={optional(result['latency']['model']['fps'])}")


if __name__ == "__main__":
    main()

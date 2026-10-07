#!/usr/bin/env python3
"""Short synthetic dataset/optimizer/resume/decode gates; never an AP benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets"),
               str(ROOT / "tools/kitti_training_pipeline")]
from common import (bev_encoding_spec, build_model, checkpoint_identity, detection_spec,
                    git_metadata, model_parameter_report, read_json, write_json)
from core.datasets.dataset import Dataset
from postprocess import filter_pred, filter_pred_3d
import postprocess
import train


def synthetic_config(config, directory, precision, workers):
    """Keep model/objective settings; use disjoint IDs and deterministic tiny scenes."""
    config = copy.deepcopy(config)
    root = Path(directory)
    for folder in ("pointcloud", "label"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    labels = ["Car 1.5 1.6 3.5 2.15 .05 -1.7 .2",
              "Pedestrian 1.7 .6 .8 2.35 .08 -1.7 -.3",
              "Cyclist 1.7 .6 1.4 3.35 .85 -1.7 .4"]
    for index in range(6):
        selected = labels if index % 3 == 0 else labels[:1] if index % 3 == 1 else []
        clouds = []
        for label in selected:
            center = np.array(list(map(float, label.split()[4:7])))
            xyz = center + rng.uniform([-.15, -.15, .1], [.15, .15, .9], (16, 3))
            clouds.append(np.column_stack((xyz, np.full(16, .6))))
        points = np.concatenate(clouds).astype(np.float32) if clouds else np.empty((0, 4), np.float32)
        points.tofile(root / "pointcloud" / f"{index:06d}.bin")
        (root / "label" / f"{index:06d}.txt").write_text("\n".join(selected))
    for name, ids in (("train", range(3)), ("val", range(3, 6))):
        (root / f"{name}.txt").write_text("".join(f"{index:06d};kitti\n" for index in ids))
    config["data"]["kitti"].update(location=str(root))
    geometry = config["data"]["kitti"]["geometry"]
    geometry.update(x_min=0., x_max=48 * geometry["x_res"],
                    y_min=-32 * geometry["y_res"], y_max=32 * geometry["y_res"])
    config["augmentation"] = {"p": 0., "rotation": {"use": False},
                              "scaling": {"use": False}, "translation": {"use": False}}
    config["train"].update(data=str(root / "train.txt"), num_workers=workers, precision=precision,
                           physical_batch_size=2, accumulation_steps=1, warmup_epochs=0, epochs=4)
    if 'checkpoint_selection' in config['train']:
        # This gate checks training-state restoration on synthetic scenes. It
        # has no original KITTI annotations and must not invent an AP score.
        config['train']['checkpoint_selection']['primary'] = 'loss'
    config["val"].update(data=str(root / "val.txt"), physical_batch_size=2)
    return config


def shapes(value):
    return {key: shapes(item) for key, item in value.items()} if isinstance(value, dict) else list(value.shape)


def modules(config, device):
    model = build_model(config).to(device)
    criterion = train.build_training_criterion(config, device)
    optimizer = train.build_optimizer(model, criterion, config)
    scheduler = train.build_scheduler(optimizer, config, config["train"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["train"]["precision"] == "fp16")
    return model, criterion, optimizer, scheduler, scaler


def assert_nested_equal(actual, expected):
    """Check restored optimizer/schedule/scaler/RNG values, including tensors."""
    if torch.is_tensor(expected):
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


def verify_vertical_ownership(dataset, sample, detection):
    """Match every supervised synthetic cell to its independent label geometry."""
    boxes = torch.as_tensor(dataset.get_boxes(0))
    geom = dataset.config["kitti"]["geometry"]
    factor = dataset.config["out_size_factor"]
    for group in detection.groups:
        target = sample["groups"][group.name]
        candidates = boxes[torch.isin(boxes[:, 0], torch.tensor(group.global_ids))]
        ys, xs = target["reg_mask"].bool().nonzero(as_tuple=True)
        expected = torch.stack((candidates[:, 4], candidates[:, 5], candidates[:, 2].log(),
                                candidates[:, 3].log(), (2*candidates[:, 7]).cos(),
                                (2*candidates[:, 7]).sin()), dim=1)
        origins = torch.stack((xs*geom["x_res"]*factor+geom["x_min"],
                               ys*geom["y_res"]*factor+geom["y_min"]), dim=1)
        values = torch.cat((target["offset"][:, ys, xs].T+origins,
                            target["size"][:, ys, xs].T, target["yaw"][:, ys, xs].T), dim=1)
        for i, value in enumerate(values):
            matches = torch.isclose(expected, value[None], atol=1e-5, rtol=1e-5).all(dim=1)
            assert matches.any(), "Supervised BEV cell has no matching label owner"
            owner = candidates[matches]
            vertical = torch.stack((owner[:, 6], owner[:, 1].log()), dim=1)
            assert torch.isclose(vertical, target["vertical"][:, ys[i], xs[i]][None],
                                 atol=1e-5, rtol=1e-5).all(dim=1).any()
    return True


def precision_run(source, device, precision, steps, workers, full_resolution):
    with tempfile.TemporaryDirectory(prefix="lidar-smoke-") as directory:
        train.seed_everything(42)
        config = synthetic_config(source, directory, precision, workers)
        config["train"]["epochs"] = max(steps + 1, 4)
        detection = detection_spec(config)
        dataset = Dataset(config["train"]["data"], config["data"], config["augmentation"],
                          detection.cls_encoding, "train", config["train"].get("target_backend", "python"))
        preview = dataset[0]  # Parent preview/Numba warmup before worker creation.
        ownership_verified = (verify_vertical_ownership(dataset, preview, detection)
                              if detection.box_mode == "3d" else None)
        kwargs = train.loader_kwargs(workers, device.type == "cuda")
        if workers:
            kwargs["multiprocessing_context"] = "spawn"
            kwargs["timeout"] = 60
        generator = torch.Generator().manual_seed(42)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, generator=generator, **kwargs)
        model, criterion, optimizer, scheduler, scaler = modules(config, device)
        parameter_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        required = {id(p) for module in (model, criterion) for p in module.parameters() if p.requires_grad}
        assert set(parameter_ids) == required and len(parameter_ids) == len(required)
        assert all(t.device == device for module in (model, criterion) for t in module.state_dict().values())
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        head = model.grouped_header.heads[detection.groups[0].name].cls.head.weight
        initial = head.detach().clone()
        vertical_initial = {name: branch.vertical.head.weight.detach().clone()
                            for name, branch in model.grouped_header.heads.items()
                            if detection.box_mode == "3d"}
        vertical_gradients = {name: False for name in vertical_initial}
        vertical_losses = []
        overflow_diagnostics = []
        modulation_dtypes = set()
        handle = model.backbone.c4_context.out_projection.register_forward_pre_hook(
            lambda module, inputs: modulation_dtypes.add(str(inputs[0].dtype)))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        iterator = iter(loader)
        next_sample_index = 0
        updates, skipped, attempts = 0, 0, 0
        retry_batch = False
        negative_group, empty_scene = False, False
        loss_values = []
        try:
            while updates < steps:
                if attempts >= steps + (32 if detection.box_mode == "3d" else 16):
                    raise FloatingPointError(f"AMP did not produce enough finite optimizer updates: precision={precision}, diagnostics={overflow_diagnostics}")
                if not retry_batch:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        next_sample_index = 0
                        batch = next(iterator)
                    decoded_sample_index = next_sample_index
                    next_sample_index += batch["voxel"].shape[0]
                    batch = train.move_tensor_batch(batch, device)
                retry_batch = False
                if attempts == 0:
                    input_dimensions = list(batch["voxel"].shape)
                empty_scene |= bool((~batch["voxel"].flatten(1).bool().any(dim=1)).any())
                negative_group |= any(bool((~group["reg_mask"].flatten(1).bool().any(dim=1)).any())
                                      for group in batch["groups"].values())
                train.set_loss_epoch(criterion, updates)
                optimizer.zero_grad(set_to_none=True)
                with train.autocast_context(device, precision):
                    prediction = model(batch["voxel"])
                    objective = criterion(prediction, batch)
                    loss = objective["loss"]
                if detection.box_mode == "3d":
                    assert objective["vertical"].dtype == torch.float32
                if not torch.isfinite(loss) or loss.dtype != torch.float32:
                    raise FloatingPointError("Smoke objective must be finite FP32")
                scaler.scale(loss).backward()
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                gradients = [p.grad for p in parameters if p.grad is not None]
                finite = gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
                attempts += 1
                if not finite:
                    if not scaler.is_enabled():
                        raise FloatingPointError("Non-finite FP32/BF16 gradients")
                    overflow_diagnostics.append({"scale": scaler.get_scale(), "loss": float(loss.detach()),
                        "nonfinite_parameters": [name for name, p in model.named_parameters()
                            if p.grad is not None and not torch.isfinite(p.grad).all()]})
                    scaler.step(optimizer)
                    scaler.update()
                    skipped += 1
                    # Retry the same synthetic supervised 3D scene: otherwise
                    # negative-only successes can exhaust this very short gate
                    # before any vertical branch sees a finite gradient.
                    retry_batch = detection.box_mode == "3d"
                    continue
                torch.nn.utils.clip_grad_norm_(parameters, config["train"].get("grad_clip_norm", 10.))
                for name in vertical_gradients:
                    gradients_for_head = [p.grad for p in model.grouped_header.heads[name].vertical.parameters()]
                    vertical_gradients[name] |= any(g is not None and bool(g.ne(0).any()) for g in gradients_for_head)
                if detection.box_mode == "3d":
                    vertical_losses.append(float(objective["vertical"].detach()))
                scaler.step(optimizer)
                scaler.update()
                loss_values.append(float(loss.detach()))
                updates += 1
                scheduler.step()
                if not all(bool(torch.isfinite(p).all()) for p in parameters):
                    raise FloatingPointError("Non-finite parameters after optimizer step")
        finally:
            handle.remove()
        state = {key: tensor.detach().clone() for key, tensor in criterion.state_dict().items()}
        validation = train.validate(model, criterion, [batch], device, precision)
        assert all(torch.equal(state[key], tensor) for key, tensor in criterion.state_dict().items())
        checkpoint = Path(directory) / "smoke.pt"
        torch.save(train.checkpoint_payload(model, criterion, optimizer, scheduler, scaler, steps,
                   validation, validation["loss"], config, loader_generator=generator), checkpoint)
        restored = modules(config, device)
        restored_generator = torch.Generator().manual_seed(7)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        outcome = train.restore_checkpoint(saved,
            config, *restored, loader_generator=restored_generator)
        assert outcome["mode"] == "resume"
        for source_module, restored_module in zip((model, criterion), restored[:2]):
            assert all(torch.equal(tensor, restored_module.state_dict()[key])
                       for key, tensor in source_module.state_dict().items())
        for key, module in zip(("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"), restored[2:]):
            assert_nested_equal(module.state_dict(), saved[key])
        assert_nested_equal(train.capture_rng_state(restored_generator), saved["rng_state"])
        model.eval()
        restored[0].eval()
        with torch.inference_mode(), train.autocast_context(device, precision):
            actual = restored[0](batch["voxel"])
            expected = model(batch["voxel"])
            for name, heads in actual["groups"].items():
                for key, tensor in heads.items():
                    torch.testing.assert_close(tensor, expected["groups"][name][key], rtol=0, atol=0)
            local = model.backbone.c3_light_attention
            local_dtype = None
            if hasattr(local, "core"):
                values = torch.randn(2, 48, 4, 6, device=device, dtype=next(iter(actual["groups"].values()))["cls"].dtype)
                weights = (local.core.channel_weights(values) if hasattr(local.core, "channel_weights")
                           else local.core.spatial_weights(values))
                assert weights.dtype == torch.float32 and torch.isfinite(weights).all()
                local_dtype = str(weights.dtype)
        one_frame = {"groups": {name: {key: value[:1] for key, value in heads.items()}
                                for name, heads in actual["groups"].items()}}
        decoder = filter_pred_3d if detection.box_mode == "3d" else filter_pred
        boxes = decoder(one_frame, config["data"]["kitti"], 4, .05, .1,
                            task_groups=detection.groups, cls_encoding=detection.cls_encoding,
                            use_iou=detection.use_iou, max_detections=7)
        assert boxes.shape[1] == (9 if detection.box_mode == "3d" else 7)
        assert boxes.dtype == np.float32 and np.isfinite(boxes).all()
        assert modulation_dtypes == {"torch.float32"}
        result = {"status": "passed", "optimizer_updates": updates, "amp_skipped_updates": skipped,
            "losses": loss_values, "finite_loss": True, "finite_gradients": True,
            "loss_dtype": str(loss.dtype), "head_output_dtype": str(next(iter(actual["groups"].values()))["cls"].dtype),
            "focal_modulation_input_dtype": next(iter(modulation_dtypes)), "local_gate_dtype": local_dtype,
            "head_weight_changed": not torch.equal(initial, head), "optimizer_membership_exact": True,
            "model_and_criterion_on_device": True, "checkpoint_restore_verified": True,
            "criterion_state_unchanged_in_validation": True, "input_shape": input_dimensions,
            "output_shapes": shapes(actual), "validation_groups": [g.name for g in detection.groups],
            "validation_loss": validation["loss"], "negative_only_group_seen": negative_group,
            "empty_scene_seen": empty_scene, "detection_shape": list(boxes.shape),
            "rotated_nms_backend": ("torchvision_cuda" if device.type == "cuda" and postprocess._torchvision_nms_rotated
                                    else "cpu_polygon_fallback"),
            "effective_config_identity": checkpoint_identity(config),
            **model_parameter_report(model, criterion)}
        if detection.box_mode == "3d":
            result.update(vertical_loss_dtype="torch.float32", vertical_losses=vertical_losses,
                          saved_predictions_3d=boxes.tolist(),
                          saved_ground_truth_3d=dataset.get_boxes(decoded_sample_index).tolist(),
                          saved_frame_id=f"{decoded_sample_index:06d}",
                          amp_final_scale=scaler.get_scale(), amp_overflow_diagnostics=overflow_diagnostics,
                          amp_retry_policy="same synthetic batch after overflow; production trainer unchanged",
                          vertical_gradient_verified=all(vertical_gradients.values()),
                          vertical_target_ownership_verified=ownership_verified,
                          vertical_head_weight_changed={name: not torch.equal(value, model.grouped_header.heads[name].vertical.head.weight)
                              for name, value in vertical_initial.items()})
            assert result["vertical_gradient_verified"] and all(result["vertical_head_weight_changed"].values()), {
                "precision": precision, "vertical_gradients": vertical_gradients,
                "vertical_head_weight_changed": result["vertical_head_weight_changed"],
                "vertical_losses": vertical_losses, "successful_updates": updates,
                "skipped_updates": skipped}
        result.update(optimizer_state_restore_verified=True, scheduler_scaler_rng_restore_verified=True)
        assert result["head_weight_changed"]
        if full_resolution:
            with torch.inference_mode(), train.autocast_context(device, precision):
                full = model(torch.zeros(bev_encoding_spec(source).input_shape, device=device))
            assert all(bool(torch.isfinite(tensor).all()) for group in full["groups"].values() for tensor in group.values())
            result.update(full_resolution_input_shape=list(bev_encoding_spec(source).input_shape),
                          full_resolution_output_shapes=shapes(full), full_resolution_finite=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        del iterator, loader
        return result


def run_smoke(config, *, device="cuda", precisions=("fp32", "fp16", "bf16"), steps=3,
              num_workers=0, full_resolution=False):
    """Run actual training helpers on synthetic scenes; unsupported AMP stays explicit."""
    if type(steps) is not int or steps < 1 or type(num_workers) is not int or num_workers < 0:
        raise ValueError("steps must be positive and num_workers nonnegative integers")
    if (not isinstance(precisions, (list, tuple)) or "fp32" not in precisions or
            len(set(precisions)) != len(precisions) or any(p not in {"fp32", "fp16", "bf16"} for p in precisions)):
        raise ValueError("Require FP32 and distinct supported precision names")
    requested = torch.device(device)
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("Smoke supports only CPU or CUDA")
    detection = detection_spec(config)
    if detection.head_mode != "grouped" or config["model"].get("c4_context") != "focal":
        raise ValueError("Smoke requires a grouped focal config")
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU/meta is not a substitute for GPU readiness")
        requested = torch.device("cuda", torch.cuda.current_device() if requested.index is None else requested.index)
        torch.cuda.set_device(requested)
    report = {"status": "passed", "synthetic": True, "accuracy_measured": False,
        "box_mode": detection.box_mode,
        "compile_model": False, "device": str(requested), "worker_count": num_workers,
        "cuda_device_name": torch.cuda.get_device_name(requested) if requested.type == "cuda" else None,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "source_config_identity": checkpoint_identity(config),
        "source_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source": git_metadata(ROOT), "precisions": {},
        "synthetic_overrides": "tiny geometry, disjoint temporary IDs, batch2, no augmentation, warmup0, loss-only selection without AP; optimizer/objective/model preserved",
        "limits": "No AP/latency benchmark. Hybrid paste is covered separately by CPU assembled gates. No Inductor or deterministic persistent-worker replay claim."}
    for precision in precisions:
        if precision != "fp32" and (requested.type != "cuda" or
                (precision == "bf16" and not torch.cuda.is_bf16_supported())):
            report["precisions"][precision] = {"status": "unsupported", "reason": "CUDA precision requires supported hardware"}
            report["status"] = "partial"
            continue
        report["precisions"][precision] = precision_run(config, requested, precision, steps, num_workers, full_resolution)
        if requested.type == "cuda":
            torch.cuda.empty_cache()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "bf16"])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--full-resolution", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_smoke(read_json(args.config), device=args.device, precisions=args.precisions,
        steps=args.steps, num_workers=args.num_workers, full_resolution=args.full_resolution)
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "device": report["device"],
                      "precisions": {key: value["status"] for key, value in report["precisions"].items()}}, indent=2))
    return report


if __name__ == "__main__":
    main()

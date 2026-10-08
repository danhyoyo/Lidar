#!/usr/bin/env python3
"""Instantiated parameter and shape-based Conv2d MAC audit; no latency/AP claim."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"),
               str(ROOT / "detector/core/datasets"), str(ROOT / "tools/kitti_training_pipeline")]
from core.losses.loss_fn import build_loss_function
from core.models.backbones.mobilepixornext_blocks import LiteMLARefinement
from tools.kitti_training_pipeline.common import (backbone_feature_spec, build_model, bev_encoding_spec,
                                                 detection_spec, model_parameter_report)

VARIANT_OVERRIDES = {
    "reference": {}, "focal": {"c4_context": "focal"},
    "focal_eca": {"c4_context": "focal", "local_attention": "eca"},
    "focal_simam": {"c4_context": "focal", "local_attention": "simam"},
    "focal_fusion32": {"c4_context": "focal", "neck_fusion_channels": 32},
    "focal_detail": {"c4_context": "focal", "detail_path": True},
    "focal_fusion32_detail_eca": {"c4_context": "focal", "local_attention": "eca",
                                   "neck_fusion_channels": 32, "detail_path": True},
}


def reference_config():
    """Return a profiling-only no-context reference with the current single head.

    This does not create/enable the future grouped training recipe.
    """
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["data"]["bev_encoding"] = {"name": "hist14", "version": 1,
                                      "backend": "numpy", "density_norm": 32, "intensity_scale": 1}
    config["model"].update(backbone="mobilepixornext", stage_depths=[3, 4, 2],
                           backbone_out_dim=32, c4_attention="none", c4_attention_scales=[],
                           c4_attention_qk_norm="none", scale_gated_fpn=True,
                           neck_type="scale_gated_fpn", expansion=2.5, use_reparam=False,
                           deploy=False, header_use_iou=True, head_mode="legacy_single")
    config["loss"].update(name="oga", use_iou=True)
    return config


def variant_config(name="reference"):
    """Profiling-only variants retain the implemented single head and OGA/IQA."""
    if name not in VARIANT_OVERRIDES:
        raise ValueError(f"unsupported profiling variant: {name!r}")
    config = reference_config()
    config["model"].update(copy.deepcopy(VARIANT_OVERRIDES[name]))
    return config


def profile_detector(config, *, device="meta", shape=None, max_backbone_parameters=1_000_000):
    """Profile actual modules; criterion parameters are separate from inference."""
    config = copy.deepcopy(config)
    if str(config["model"]["backbone"]).lower() != "mobilepixornext":
        raise ValueError("body/neck profiling currently requires mobilepixornext")
    detection = detection_spec(config)
    schema = bev_encoding_spec(config)
    if schema.is_packed:
        raise ValueError(f"{schema.name} requires packed points; this dense-shape profiler is unsupported. "
                         "Use model_parameter_report(build_model(config)) for parameter counts.")
    features = backbone_feature_spec(config["model"], geometry=config["data"]["kitti"]["geometry"])
    config["model"].update(features.to_dict())
    input_dims = tuple(schema.input_shape if shape is None else shape)
    if (len(input_dims) != 4 or any(type(v) is not int or v <= 0 for v in input_dims) or
            input_dims[1] != schema.channels or any(size % 16 for size in input_dims[2:])):
        raise ValueError("profile shape requires positive BCHW dimensions, schema channels and XY divisible by 16")
    with torch.device(device):
        model = build_model(config).eval()
        criterion = build_loss_function(detection.cls_encoding, config.get("loss"), head_mode=detection.head_mode,
                                        task_groups=detection.groups if detection.head_mode == "grouped" else None,
                                        box_mode=detection.box_mode).eval()
    parts = {name: sum(p.numel() for p in module.parameters())
             for name, module in model.backbone.named_children()}
    detail_scale = model.backbone.detail_gamma
    detail_scale_count = 0 if detail_scale is None else detail_scale.numel()
    if detail_scale_count:
        parts["detail_gamma"] = detail_scale_count
    backbone = sum(p.numel() for p in model.backbone.parameters())
    local = model.backbone.c3_light_attention
    local_total = sum(p.numel() for p in local.parameters())
    local_core = sum(p.numel() for p in local.core.parameters()) if hasattr(local, "core") else 0
    context_total = sum(p.numel() for p in model.backbone.c4_context.parameters())
    context_scale = (model.backbone.c4_context.gamma.numel()
                     if hasattr(model.backbone.c4_context, "gamma") else 0)
    feature_counts = {
        "context": {"kind": features.c4_context, "total": context_total,
                    "core": context_total - context_scale, "residual_scale": context_scale},
        "local": {"kind": features.local_attention, "total": local_total,
                  "core": local_core, "residual_scale": local_total - local_core},
        "detail": {"total": parts["detail_branch"] + detail_scale_count,
                   "core": parts["detail_branch"], "residual_scale": detail_scale_count},
    }
    parameters = model_parameter_report(model, criterion)
    counts = parameters["parameter_counts"]
    if backbone >= max_backbone_parameters:
        raise ValueError(f"backbone including neck has {backbone} parameters; must be strictly less than {max_backbone_parameters}")
    if torch.device(device).type == "meta" and any(isinstance(m, LiteMLARefinement) for m in model.modules()):
        raise ValueError("LiteMLA autocast cannot run on meta; select --device cpu for this legacy variant")

    conv_macs, feature_shape, handles = {}, [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            def hook(module, inputs, output, name=name):
                macs = (output.numel() * (module.in_channels // module.groups) *
                        module.kernel_size[0] * module.kernel_size[1])
                conv_macs[name] = conv_macs.get(name, 0) + macs
            handles.append(module.register_forward_hook(hook))
    handles.append(model.backbone.register_forward_hook(
        lambda module, inputs, output: feature_shape.extend(output.shape)))
    try:
        with torch.no_grad():
            outputs = model(torch.zeros(input_dims, device=device))
    finally:
        for handle in handles:
            handle.remove()
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    def output_shapes(value):
        if isinstance(value, dict):
            return {name: output_shapes(item) for name, item in value.items()}
        return list(value.shape)

    return {
        "method": "Instantiated PyTorch parameters and Conv2d output-shape MAC hooks",
        "device": str(device), "torch_version": torch.__version__, "head_mode": detection.head_mode,
        "cls_encoding": detection.cls_encoding, "task_groups": [group.to_dict() for group in detection.groups],
        "box_mode": detection.box_mode, "criterion_box_mode": criterion.box_mode,
        "head_parameter_counts": parameters["head_parameter_counts"],
        "resolved_config": config, "config_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "encoding_semantic_hash": schema.semantic_hash, "encoding_backend": schema.backend,
        "feature_metadata": features.semantic_metadata(), "feature_semantic_hash": features.semantic_hash,
        "feature_parameter_counts": feature_counts,
        "parameter_counts": counts, "backbone_parts": parts,
        "budget": {"scope": "backbone including neck", "strict_upper_limit": max_backbone_parameters,
                   "passes": backbone < max_backbone_parameters},
        "input_shape": list(input_dims), "backbone_output_shape": list(feature_shape),
        "output_shapes": output_shapes(outputs),
        "conv2d_mac_estimate": {"total": sum(conv_macs.values()), "by_module": conv_macs,
                                "excluded_operations": ["attention matmuls", "Conv1d channel correlation", "normalization", "activations",
                                                        "interpolation", "elementwise operations", "preprocessing"]},
        "limitations": "Shape/count audit only; not measured latency, peak memory, accuracy or 3D readiness",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--config", type=Path, help="Profile an explicit legacy or grouped detector config")
    selection.add_argument("--variant", choices=tuple(VARIANT_OVERRIDES), help="Profiling-only variant; default reference")
    parser.add_argument("--device", default="meta")
    parser.add_argument("--shape", type=int, nargs=4, metavar=("B", "C", "H", "W"))
    parser.add_argument("--max-backbone-parameters", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text()) if args.config else variant_config(args.variant or "reference")
    report = profile_detector(config, device=args.device, shape=args.shape,
                              max_backbone_parameters=args.max_backbone_parameters)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                     ("device", "head_mode", "parameter_counts", "budget", "input_shape",
                      "backbone_output_shape", "output_shapes", "limitations")}, indent=2))
    return report


if __name__ == "__main__":
    main()

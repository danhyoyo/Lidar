#!/usr/bin/env python3
"""Export a selected PyTorch KITTI detector checkpoint to fixed-shape ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from common import (build_model, configure_detector_imports, input_shape,
                    normalize_state_dict, read_json, sha256, write_json)


class RawHeadWrapper(nn.Module):
    OUTPUT_NAMES = ("cls", "offset", "size", "yaw")

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.output_names = self.OUTPUT_NAMES + (
            ("log_var",) if model.header.predict_log_variance else ()
        )

    def forward(self, voxel):
        outputs = self.model(voxel)
        return tuple(outputs[name] for name in self.output_names)


def prediction_columns(box_encoding: str, has_log_var: bool):
    columns = (
        ["class", "score", "x", "y", "z", "w", "l", "h", "yaw"]
        if box_encoding == "center3d"
        else ["class", "score", "x", "y", "l", "w", "yaw"]
    )
    if has_log_var:
        columns += [
            "log_var_dx", "log_var_dy", "log_var_z", "log_var_logw",
            "log_var_logl", "log_var_logh",
        ]
    return columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--detector-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.opset < 1:
        raise ValueError("opset must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    configure_detector_imports(args.detector_root)
    config = read_json(args.config)
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(normalize_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()
    wrapper = RawHeadWrapper(model).eval()
    sample_shape = input_shape(config, "kitti")
    sample = torch.zeros(sample_shape, dtype=torch.float32, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reference = wrapper(sample)
    torch.onnx.export(wrapper, sample, str(args.output), input_names=["voxel"],
        output_names=list(wrapper.output_names), opset_version=args.opset,
        do_constant_folding=True, dynamo=False)
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install the 'onnx' Python package to validate the export") from error
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    graph_outputs = [output.name for output in graph.graph.output]
    if graph_outputs != list(wrapper.output_names):
        raise RuntimeError(f"Unexpected ONNX outputs: {graph_outputs}")
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "bev_encoding": config["data"].get("bev_encoding", {"name": "binary_slices"}),
        "box_encoding": config["model"].get("box_encoding", "bev"),
        "scale_gated_fpn": config["model"].get("scale_gated_fpn", False),
        "input_name": "voxel",
        "input_shape": list(sample_shape), "input_dtype": "float32",
        "outputs": {name: list(tensor.shape) for name, tensor in
                    zip(wrapper.output_names, reference)},
        "prediction_schema": {
            "version": 2 if "log_var" in wrapper.output_names else 1,
            "decoded_columns": prediction_columns(
                config["model"].get("box_encoding", "bev"),
                "log_var" in wrapper.output_names,
            ),
        },
        "opset": args.opset, "onnx_sha256": sha256(args.output)
    }
    write_json(args.output.with_suffix(args.output.suffix + ".json"), metadata)
    print(f"Validated ONNX: {args.output.resolve()}")
    print(f"Input voxel shape: {sample_shape}")
    print(f"Outputs: {metadata['outputs']}")


if __name__ == "__main__":
    main()

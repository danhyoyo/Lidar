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

    def forward(self, voxel):
        outputs = self.model(voxel)
        return tuple(outputs[name] for name in self.OUTPUT_NAMES)


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
    configure_detector_imports(args.detector_root)
    config = read_json(args.config)
    device = torch.device(args.device)
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
        output_names=list(RawHeadWrapper.OUTPUT_NAMES), opset_version=args.opset,
        do_constant_folding=True, dynamo=False)
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install the 'onnx' Python package to validate the export") from error
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    graph_outputs = [output.name for output in graph.graph.output]
    if graph_outputs != list(RawHeadWrapper.OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected ONNX outputs: {graph_outputs}")
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()), "input_name": "voxel",
        "input_shape": list(sample_shape), "input_dtype": "float32",
        "outputs": {name: list(tensor.shape) for name, tensor in
                    zip(RawHeadWrapper.OUTPUT_NAMES, reference)},
        "opset": args.opset, "onnx_sha256": sha256(args.output)
    }
    write_json(args.output.with_suffix(args.output.suffix + ".json"), metadata)
    print(f"Validated ONNX: {args.output.resolve()}")
    print(f"Input voxel shape: {sample_shape}")
    print(f"Outputs: {metadata['outputs']}")


if __name__ == "__main__":
    main()

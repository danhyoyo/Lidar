#!/usr/bin/env python3
"""Build and smoke-test a fixed-shape TensorRT engine with trtexec."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import read_json, sha256, write_json


def find_trtexec(explicit: Path | None) -> str | None:
    import shutil

    candidates = [str(explicit)] if explicit else []
    discovered = shutil.which("trtexec")
    if discovered:
        candidates.append(discovered)
    candidates.extend(("/usr/src/tensorrt/bin/trtexec", "/opt/tensorrt/bin/trtexec"))
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def run(command) -> None:
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def build_with_python(onnx_path: Path, engine_path: Path, precision: str,
                      workspace_mib: int, smoke_test: bool) -> str:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "Neither trtexec nor the TensorRT Python package is available"
        ) from error

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 5
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU/TensorRT build does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned no serialized engine")
    engine_path.write_bytes(bytes(serialized))

    if smoke_test:
        import torch

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("TensorRT could not deserialize the engine it just built")
        context = engine.create_execution_context()
        buffers = {}
        torch_dtypes = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
        }
        if hasattr(trt, "bool"):
            torch_dtypes[trt.bool] = torch.bool
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            shape = tuple(engine.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Dynamic tensor is not supported by this node: {name} {shape}")
            dtype = torch_dtypes.get(engine.get_tensor_dtype(name))
            if dtype is None:
                raise RuntimeError(f"Unsupported TensorRT dtype for {name}")
            buffers[name] = torch.zeros(shape, dtype=dtype, device="cuda")
            context.set_tensor_address(name, buffers[name].data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        if not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT runtime smoke test returned false")
        torch.cuda.synchronize()
        output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        if set(output_names) != {"cls", "offset", "size", "yaw"}:
            raise RuntimeError(f"Unexpected TensorRT outputs: {output_names}")
    return f"TensorRT Python {trt.__version__}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--trtexec", type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--skip-runtime-smoke-test", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.workspace_mib < 1:
        raise ValueError("workspace_mib must be positive")
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    trtexec = find_trtexec(args.trtexec)
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    if trtexec:
        help_result = subprocess.run([trtexec, "--help"], text=True,
                                     capture_output=True, check=True)
        help_text = help_result.stdout + help_result.stderr
        command = [trtexec, f"--onnx={args.onnx.resolve()}",
                   f"--saveEngine={args.engine.resolve()}", "--skipInference"]
        if args.precision == "fp16":
            command.append("--fp16")
        if "--memPoolSize" in help_text:
            command.append(f"--memPoolSize=workspace:{args.workspace_mib}")
        elif "--workspace" in help_text:
            command.append(f"--workspace={args.workspace_mib}")
        if "--builderOptimizationLevel" in help_text:
            command.append("--builderOptimizationLevel=5")
        run(command)
        if not args.skip_runtime_smoke_test:
            run([trtexec, f"--loadEngine={args.engine.resolve()}",
                 "--warmUp=200", "--duration=3"])
        version = subprocess.run([trtexec, "--version"], text=True,
                                 capture_output=True, check=False)
        builder_description = (version.stdout + version.stderr).strip()
    else:
        print("trtexec not found; using TensorRT Python builder", flush=True)
        builder_description = build_with_python(
            args.onnx.resolve(), args.engine.resolve(), args.precision,
            args.workspace_mib, not args.skip_runtime_smoke_test)
    if not args.engine.is_file() or args.engine.stat().st_size == 0:
        raise RuntimeError(f"TensorRT did not create a valid engine: {args.engine}")
    onnx_metadata_path = args.onnx.with_suffix(args.onnx.suffix + ".json")
    onnx_metadata = read_json(onnx_metadata_path) if onnx_metadata_path.is_file() else None
    write_json(args.engine.with_suffix(args.engine.suffix + ".json"), {
        "onnx": str(args.onnx.resolve()), "onnx_sha256": sha256(args.onnx),
        "engine": str(args.engine.resolve()), "engine_sha256": sha256(args.engine),
        "precision": args.precision, "trtexec": trtexec,
        "builder": builder_description,
        "io_contract": onnx_metadata,
        "note": "TensorRT plans are specific to GPU architecture and TensorRT/CUDA versions."
    })
    print(f"TensorRT engine: {args.engine.resolve()}")


if __name__ == "__main__":
    main()

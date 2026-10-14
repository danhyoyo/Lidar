#!/usr/bin/env python3
"""Export MobilePixorNeXt topology for the NN-SVG viewer, without PyTorch.

This is a curated architecture adapter, not an arbitrary model tracer. Shared
repository resolvers supply input geometry and active options. --verify-model
additionally audits tensor shapes against a real PyTorch model on the meta device.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.core.backbone_config import ATTENTION_STAGES, finite_scalar, positive_integer, resolve_backbone_features
from detector.core.bev_encoding import resolve_bev_encoding

VIEWER = ROOT / "tools/visualization/nn_svg"
SOURCE_FILES = (
    "detector/core/models/backbones/mobilepixornext.py",
    "detector/core/models/backbones/mobilepixornext_blocks.py",
    "detector/core/models/backbones/rep_blocks.py",
    "detector/core/models/backbones/focal_context.py",
    "detector/core/models/backbones/light_attention.py",
)


class Graph:
    """Small JSON graph builder; positions are logical columns and rows."""

    def __init__(self, title, description=""):
        self.data = dict(title=title, description=description, nodes=[], edges=[])

    def node(self, id_, label, shape, col, row=1, kind="conv", **details):
        self.data["nodes"].append(dict(id=id_, label=label, shape=list(shape),
                                       col=col, row=row, kind=kind, **details))
        return id_

    def edge(self, source, target, label="", kind="flow", **routing):
        self.data["edges"].append(dict(source=source, target=target, label=label, kind=kind, **routing))

    def chain(self, *ids):
        for source, target in zip(ids, ids[1:]):
            self.edge(source, target)


def _block_view(channels, h, w, expansion, reparam, deploy, stage):
    hidden = int(round(channels * expansion))
    shape = (channels, h, w)
    g = Graph(f"MobilePixorNeXtBlock · Stage {stage}",
              f"C={channels}; expansion={expansion:g}; hidden=round(C × expansion)={hidden}. "
              "Residual path preserves BCHW; gamma is learned per channel.")
    g.node("input", "X", shape, 0, kind="input")
    if reparam and not deploy:
        for id_, label, row, kernel in (("rep7", "DW 7×7 + BN", 0, 7),
                                        ("rep3", "DW 3×3 + BN", 1, 3),
                                        ("rep_identity", "Identity + BN", 2, 1)):
            settings = dict(kernel=kernel, groups=channels, bias=False) if id_ != "rep_identity" else {}
            g.node(id_, label, shape, 1, row,
                   kind="depthwise" if id_ != "rep_identity" else "norm", **settings)
            g.edge("input", id_)
        g.node("rep_sum", "RepConv branch sum", shape, 2, kind="add", formula="Y7 + Y3 + BN(X)")
        for id_ in ("rep7", "rep3", "rep_identity"):
            g.edge(id_, "rep_sum")
        previous, start = "rep_sum", 3
    else:
        g.node("dw", "DW 7×7" + (" · fused" if reparam else ""), shape, 1,
               kind="depthwise", kernel=7, groups=channels, padding=3, bias=reparam,
               source="dw_block.rbr_reparam" if reparam else "dwconv")
        g.edge("input", "dw")
        previous, start = "dw", 2
        if not reparam:
            g.node("norm1", "BatchNorm", shape, 2, kind="norm")
            g.edge(previous, "norm1")
            previous, start = "norm1", 3
    for offset, (id_, label, c, kind) in enumerate((
            ("expand", "PW 1×1 · expand", hidden, "conv"),
            ("silu", "SiLU", hidden, "activation"),
            ("project", "PW 1×1 · project", channels, "conv"),
            ("norm2", "BatchNorm", channels, "norm"),
            ("scale", "LayerScale γ", channels, "scale"),
            ("add", "Residual add", channels, "add"))):
        g.node(id_, label, (c, h, w), start + offset, kind=kind,
               **({"initial_value": 1e-5, "formula": "γ ⊙ F(X)"} if id_ == "scale" else {}))
        g.edge(previous, id_)
        previous = id_
    g.edge("input", "add", "identity", "skip")
    return g.data


def _gate_view(channels, h, w, enabled, level):
    shape = (channels, h, w)
    g = Graph(f"{'Scale-gated' if enabled else 'Additive'} FPN fusion · P{level}",
              "L = lateral projection; U = resized and refined top-down feature. " +
              ("Depthwise gate weights and bias initialize to zero, giving gate=1."
               if enabled else "Gating is disabled; fusion is the direct sum L + U."))
    g.node("lateral", "L · lateral", shape, 0, 0, kind="input")
    g.node("up", "U · top-down", shape, 0, 2, kind="input")
    g.node("sum", "L + U" if enabled else f"P{level} = L + U", shape, 1, kind="add")
    g.edge("lateral", "sum")
    g.edge("up", "sum")
    if enabled:
        g.node("dw", "DW 3×3 · gate logits", shape, 2, kind="depthwise",
               kernel=3, padding=1, groups=channels, bias=True, initial_value=0)
        g.node("sigmoid", "2 × sigmoid", shape, 3, kind="activation", formula="G = 2σ(DW(L + U))")
        g.node("multiply", "G ⊙ L", shape, 4, kind="multiply")
        g.node("output", f"P{level} = U + G ⊙ L", shape, 5, kind="add")
        g.chain("sum", "dw", "sigmoid", "multiply", "output")
        g.edge("lateral", "multiply", "lateral", "skip")
        g.edge("up", "output", "top-down", "skip")
    return g.data


def _context_view(features, h, w):
    spec = ATTENTION_STAGES[features.focal_stage]
    channels = spec.channels
    d = features.c4_context_bottleneck
    dilations = features.c4_context_dilations
    levels = len(dilations)
    shape, small = (channels, h, w), (d, h, w)
    g = Graph(f"FocalContext · {spec.name.upper()} · stride {spec.stride}",
              "Sequential dilated local levels plus FP32 global mean; per-pixel softmax gates. "
              "Q is a modulation feature, not a dot-product attention query.")
    g.node("input", "X", shape, 0, kind="input")
    g.node("norm", "Pre BatchNorm", shape, 1, kind="norm")
    g.node("project", "PW 1×1 · split Q / S / G", (2*d + levels + 1, h, w), 2,
           formula=f"{channels} → {2*d + levels + 1}; split ({d}, {d}, {levels+1})")
    g.chain("input", "norm", "project")
    g.node("query", "Q · modulation", small, 3, 0, kind="feature")
    g.node("seed", "S · context seed", small, 3, 1, kind="feature")
    g.node("gates", "FP32 softmax(G)", (levels+1, h, w), 3, 3, kind="activation")
    for id_ in ("query", "seed", "gates"):
        g.edge("project", id_)
    previous = "seed"
    for i, dilation in enumerate(dilations):
        id_ = f"level{i}"
        g.node(id_, f"DW 3×3 · d={dilation} + SiLU", small, 4+i, kind="depthwise",
               kernel=3, dilation=dilation, groups=d)
        g.edge(previous, id_)
        previous = id_
    col = 4 + levels
    g.node("global", "FP32 spatial mean", (d, 1, 1), col, 2, kind="pool")
    g.edge(previous, "global")
    g.node("weighted", "Σ gates × context levels", small, col+1, kind="add",
           formula="Σᵢ wᵢ ⊙ Sᵢ + w_global ⊙ mean(S_last)")
    for i in range(levels):
        g.edge(f"level{i}", "weighted", f"w{i+1} ⊙ S{i+1}", route_row=0.32+i*0.13/levels*3)
    g.edge("global", "weighted", "broadcast")
    g.edge("gates", "weighted", "weights", route_row=2.5)
    g.node("context_proj", "PW 1×1 · D → D", small, col+2)
    g.node("multiply", "FP32 Q ⊙ context", small, col+3, kind="multiply")
    g.node("out_proj", f"PW 1×1 · D → {channels} + BN", shape, col+4)
    g.node("scale", "LayerScale γ", shape, col+5, kind="scale",
           initial_value=features.c4_context_layer_scale_init)
    g.node("output", "X + γ ⊙ F(X)", shape, col+6, kind="add")
    g.chain("weighted", "context_proj", "multiply", "out_proj", "scale", "output")
    g.edge("query", "multiply", "Q", route_row=0)
    g.edge("input", "output", "identity", "skip")
    return g.data


def _local_view(features, h, w):
    spec = ATTENTION_STAGES[features.local_attention_stage]
    channels = spec.channels
    g = Graph(f"Residual {features.local_attention.upper()} · {spec.name.upper()} · stride {spec.stride}",
              "X + β × (T(X) − X); FP32 statistics; β is a learned scalar.")
    shape = (channels, h, w)
    g.node("input", "X", shape, 0, kind="input")
    if features.local_attention == "eca":
        g.node("statistics", "FP32 spatial mean", (channels, 1, 1), 1, kind="pool")
        g.node("conv", "Conv1d across channels", (channels, 1, 1), 2,
               kernel=features.local_attention_eca_kernel_size, initial_value=0,
               formula="Conv1d(1 → 1) on B × 1 × C")
        previous = "conv"
        g.chain("input", "statistics", "conv")
    else:
        g.node("statistics", "FP32 mean / variance", (channels, 1, 1), 1, kind="norm")
        g.node("energy", "Spatial energy", shape, 2, kind="feature",
               formula=f"(X − μ)² / [4(var + {features.local_attention_simam_lambda:g})] + 0.5")
        g.chain("input", "statistics", "energy")
        previous = "energy"
    gate_shape = (channels, 1, 1) if features.local_attention == "eca" else shape
    g.node("sigmoid", "Sigmoid gates", gate_shape, 3, kind="activation")
    g.node("attended", "T(X) = X ⊙ gates", shape, 4, kind="multiply")
    g.node("delta", "T(X) − X", shape, 5, kind="feature")
    g.node("scale", "Residual scale β", shape, 6, kind="scale",
           initial_value=features.local_attention_layer_scale_init, symbol="β")
    g.node("output", "X + β × (T(X) − X)", shape, 7, kind="add")
    g.chain(previous, "sigmoid", "attended", "delta", "scale", "output")
    if features.local_attention == "simam":
        g.edge("input", "energy", "X", "skip")
    for target in ("attended", "delta", "output"):
        g.edge("input", target, "X", "skip")
    return g.data


def build_diagram(config, *, source="configuration"):
    """Resolve a full pipeline config into a portable, deterministic graph bundle."""
    cfg = dict(config.get("model", {}))
    if str(cfg.get("backbone", "mobilepixor")).lower() != "mobilepixornext":
        raise ValueError("diagram exporter supports backbone=mobilepixornext only")
    neck = cfg.get("neck_type") or "scale_gated_fpn"
    if str(neck).lower() not in {"scale_gated_fpn", "sgfpn"}:
        raise ValueError("diagram exporter supports neck_type=scale_gated_fpn only; RC necks need a separate adapter")
    data = config.get("data", {})
    geometry = data.get("kitti", {}).get("geometry") or cfg.get("geometry")
    if geometry is None:
        raise ValueError("data.kitti.geometry is required for exact tensor dimensions")
    encoding = data.get("bev_encoding", cfg.get("bev_encoding"))
    bev = resolve_bev_encoding(encoding, geometry)
    _, _, h, w = bev.input_shape
    if h % 16 or w % 16:
        raise ValueError("diagram input grid height and width must be divisible by 16 for exact FPN fusion")
    cfg.update(geometry=geometry, bev_encoding=encoding, neck_type="scale_gated_fpn")
    features = resolve_backbone_features(cfg, geometry=geometry)
    depths = cfg.get("stage_depths", [2, 4, 2])
    if not isinstance(depths, (list, tuple)) or len(depths) != 3:
        raise ValueError("stage_depths must contain three positive integers")
    for depth in depths:
        positive_integer(depth, "stage_depths")
    expansion = finite_scalar(cfg.get("expansion", 2.5), "expansion", positive=True)
    if any(int(round(c * expansion)) < 1 for c in (48, 96, 128)):
        raise ValueError("expansion produces zero hidden channels")
    out = positive_integer(cfg.get("backbone_out_dim", 16), "backbone_out_dim")
    for option in ("use_reparam", "deploy", "scale_gated_fpn"):
        if option in cfg and type(cfg[option]) is not bool:
            raise ValueError(f"{option} must be a boolean")
    reparam, deploy = cfg.get("use_reparam", False), cfg.get("deploy", False)
    gated = cfg.get("scale_gated_fpn", True)
    fusion = features.neck_fusion_channels
    shapes = {"stem": (32, h//2, w//2), "stage2": (48, h//4, w//4),
              "stage3": (96, h//8, w//8), "stage4": (128, h//16, w//16)}
    g = Graph("MobilePixorNeXt · backbone + Scale-Gated FPN",
              "Bottom-up encoder → top-down FPN → stride-4 feature for the detection header. "
              "Refined C3/C4/C5 feed laterals and the next stage. Tensor labels are C × H × W (batch omitted).")
    g.node("input", f"BEV · {bev.name}", (bev.channels, h, w), 0, kind="input",
           source="core.bev_encoding", channel_names=list(bev.channel_names))
    g.node("stem", "Stem · stride 2", shapes["stem"], 1, detail_view="stem", source="stem")
    g.edge("input", "stem")
    views = {}
    previous = "stem"
    refined_outputs = {}
    for stage, channels, col, in_channels in ((2, 48, 2, 32), (3, 96, 4, 48), (4, 128, 6, 96)):
        id_ = f"stage{stage}"
        shape = shapes[id_]
        g.node(id_, f"Stage {stage} · stride {2**stage}", shape, col,
               kind="block", repeat=depths[stage-2], detail_view=id_, source=f"down{stage} → {id_}",
               operations=[f"Conv 3×3 s2 · {in_channels} → {channels} + BN + SiLU",
                           f"MobilePixorNeXtBlock × {depths[stage-2]}"],
               footprint=f"7×7 kernel span: {7 * 2**stage * geometry['x_res']:g} m × "
                         f"{7 * 2**stage * geometry['y_res']:g} m")
        g.edge(previous, id_)
        stage_graph = Graph(f"Stage {stage} · each residual block", "Downsample first; each numbered block has its own weights.")
        input_shape = shapes["stem" if stage == 2 else f"stage{stage-1}"]
        stage_graph.node("input", "Stage input", input_shape, 0, kind="input")
        stage_graph.node("down", "Conv 3×3 · s2 + BN + SiLU", shape, 1, source=f"down{stage}",
                         kernel=3, stride=2, padding=1)
        stage_graph.edge("input", "down")
        last = "down"
        for i in range(depths[stage-2]):
            block_id = f"block{i}"
            stage_graph.node(block_id, f"NeXt block {i+1}", shape, i+2, kind="block",
                             source=f"stage{stage}.{i}", detail_view=f"block{stage}")
            stage_graph.edge(last, block_id)
            last = block_id
        views[id_] = stage_graph.data
        views[f"block{stage}"] = _block_view(channels, shape[1], shape[2], expansion, reparam, deploy, stage)
        previous = id_
        level = f"c{stage+1}"
        operations = []
        if features.focal_stage == level:
            operations.append(("context", "FocalContext", "c4_context", _context_view))
        if features.local_attention_stage == level:
            operations.append(("local", "Residual " + features.local_attention.upper(), "c3_light_attention", _local_view))
        if features.attention_order == "local_then_focal":
            operations.reverse()
        for index, (attention_id, label, source_module, view_builder) in enumerate(operations):
            attention_col = col+1 if len(operations) == 1 else col+.65*(index+1)
            g.node(attention_id, f"{label} · {level.upper()}", shape, attention_col, kind="attention",
                   source=source_module, detail_view=attention_id)
            g.edge(previous, attention_id)
            previous = attention_id
            views[attention_id] = view_builder(features, shape[1], shape[2])
        refined_outputs[id_] = previous
    c2 = refined_outputs["stage2"]
    c4 = refined_outputs["stage3"]
    for id_, label, shape, col, row, src in (
            ("l5", "L5 · lateral 1×1", (48, h//16, w//16), 6, 3, refined_outputs["stage4"]),
            ("l4", "L4 · lateral 1×1", (48, h//8, w//8), 4, 2, c4),
            ("l3", "L3 · lateral 1×1", (fusion, h//4, w//4), 2, 2, c2)):
        g.node(id_, label, shape, col, row, source=f"lat_c{id_[-1]}", kernel=1)
        g.edge(src, id_, "lateral", "lateral")
    for level, channels, col, src in ((4, 48, 5, "l5"), (3, fusion, 3, "p4")):
        shape = (channels, h//(2**(level-1)), w//(2**(level-1)))
        up, p = f"u{level}", f"p{level}"
        refinement = "DW 3×3 + BN + SiLU" if level == 4 else f"Conv 3×3 · 48 → {fusion} + BN + SiLU"
        g.node(up, f"U{level} · bilinear ↑2", shape, col, 3, kind="upsample",
               operations=["Bilinear ×2, align_corners=False", refinement], detail_view=up,
               source="refine_u4" if level == 4 else "proj_u3")
        g.node(p, f"P{level} · {'scale gate' if gated else 'add'}", shape, col-1, 3, kind="gate" if gated else "add",
               detail_view=f"gate{level}", formula=f"P{level} = U{level} + " +
               (f"2σ(DW(L{level} + U{level})) ⊙ L{level}" if gated else f"L{level}"))
        g.chain(src, up, p)
        g.edge(f"l{level}", p, "lateral", "lateral")
        views[f"gate{level}"] = _gate_view(channels, shape[1], shape[2], gated, level)
        u = Graph(f"Top-down upsampling · U{level}", "Interpolation creates no parameters; convolution refines the resized tensor.")
        input_shape = (48, shape[1]//2, shape[2]//2)
        u.node("input", "Top-down input", input_shape, 0, kind="input")
        u.node("resize", "Bilinear ↑2", (48, shape[1], shape[2]), 1, kind="upsample", align_corners=False)
        u.node("conv", "DW 3×3" if level == 4 else "Conv 3×3 · projection", shape, 2,
               kind="depthwise" if level == 4 else "conv", kernel=3, groups=48 if level == 4 else 1)
        u.node("norm", "BatchNorm", shape, 3, kind="norm")
        u.node("act", "SiLU", shape, 4, kind="activation")
        u.chain("input", "resize", "conv", "norm", "act")
        views[up] = u.data
    output_row, last = 3, "p3"
    if features.detail_path:
        detail_shape = (fusion, h//4, w//4)
        g.node("detail", "Stem detail path", detail_shape, 1, 2, kind="block", detail_view="detail",
               source="detail_branch", operations=["DW 3×3 s2 + BN + SiLU", f"PW 1×1 · 32 → {fusion} + BN + SiLU"])
        g.edge("stem", "detail", "detail", "skip")
        g.node("detail_add", "P3 + γ_detail ⊙ detail", detail_shape, 2, 4, kind="add", initial_value=0.001)
        g.edge("detail", "detail_add", "γ_detail", "skip")
        g.edge("p3", "detail_add")
        output_row, last = 4, "detail_add"
        detail = Graph("Stem detail path · stride 2 → stride 4", "FP32 residual accumulation; gamma initializes to 0.001.")
        detail.node("input", "Stem feature", shapes["stem"], 0, kind="input")
        detail.node("dw", "DW 3×3 · s2 + BN + SiLU", (32, h//4, w//4), 1, kind="depthwise", groups=32)
        detail.node("pw", f"PW 1×1 · 32 → {fusion} + BN + SiLU", detail_shape, 2)
        detail.node("scale", "γ_detail ⊙ detail", detail_shape, 3, kind="scale", initial_value=0.001)
        detail.node("p3", "FPN P3", detail_shape, 3, 0, kind="input")
        detail.node("add", "FP32 P3 + γ_detail ⊙ detail", detail_shape, 4, kind="add")
        detail.chain("input", "dw", "pw", "scale", "add")
        detail.edge("p3", "add")
        views["detail"] = detail.data
    output_shape = (out, h//4, w//4)
    g.node("out_conv", "Output 3×3 + BN + SiLU", output_shape, 1, output_row,
           source="out_conv", detail_view="output_projection", kernel=3)
    g.node("output", "To detection header · stride 4", output_shape, 0, output_row, kind="output")
    g.chain(last, "out_conv", "output")
    stem = Graph("Stem · two spatial convolutions", "First convolution downsamples by 2; second keeps the spatial resolution.")
    stem.node("input", "BEV input", (bev.channels, h, w), 0, kind="input")
    previous = "input"
    for i, (label, kind) in enumerate((("Conv 3×3 · s2", "conv"), ("BatchNorm", "norm"),
                                       ("SiLU", "activation"), ("Conv 3×3 · s1", "conv"),
                                       ("BatchNorm", "norm"), ("SiLU", "activation"))):
        id_ = f"op{i}"
        stem.node(id_, label, shapes["stem"], i+1, kind=kind, source=f"stem.{i}")
        stem.edge(previous, id_)
        previous = id_
    views["stem"] = stem.data
    projection = Graph("Output projection · detector interface", "Backbone output only; detection heads are outside this figure.")
    projection.node("input", "Fused P3", (fusion, h//4, w//4), 0, kind="input")
    for i, (id_, label, kind) in enumerate((("conv", "Conv 3×3", "conv"), ("bn", "BatchNorm", "norm"),
                                           ("silu", "SiLU", "activation"))):
        projection.node(id_, label, output_shape, i+1, kind=kind)
    projection.chain("input", "conv", "bn", "silu")
    views["output_projection"] = projection.data
    return dict(schema_version=1, name="MobilePixorNeXt", source=source,
                shape_mode="inferred from repository architecture and shared configuration contracts",
                source_files=list(SOURCE_FILES), input_encoding=bev.name,
                effective_attention_placement=features.effective_placement(),
                options=dict(stage_depths=list(depths), expansion=expansion, use_reparam=reparam,
                             deploy=deploy, scale_gated_fpn=gated, backbone_out_dim=out, **features.to_dict()),
                views={"overview": g.data, **views})


def export_diagram(diagram, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(diagram, ensure_ascii=False, indent=2, allow_nan=False)
    (output_dir / "mobilepixornext.json").write_text(payload + "\n", encoding="utf-8")
    (output_dir / "mobilepixornext.js").write_text("window.BACKBONE_DIAGRAM = " + payload + ";\n", encoding="utf-8")


def verify_model(config, diagram):
    """Run the actual backbone on meta tensors without allocating BEV activations."""
    import torch
    sys.path.insert(0, str(ROOT / "detector"))
    from core.models.backbones.registry import build_backbone
    model_config = dict(config["model"], geometry=config["data"]["kitti"]["geometry"],
                        bev_encoding=config["data"].get("bev_encoding"))
    expected = {node["id"]: node["shape"] for node in diagram["views"]["overview"]["nodes"]}
    measured = {}
    handles = []
    with torch.device("meta"):
        model = build_backbone("mobilepixornext", model_config, input_channels=expected["input"][0]).eval()
        for name in ("stem", "stage2", "stage3", "stage4"):
            handles.append(getattr(model, name).register_forward_hook(
                lambda module, args, output, name=name: measured.update({name: list(output.shape[1:])})))
        for node, module_name in (("context", "c4_context"), ("local", "c3_light_attention")):
            if node in expected:
                handles.append(getattr(model, module_name).register_forward_hook(
                    lambda module, args, output, node=node: measured.update({node:list(output.shape[1:])})))
        for level, target in (("c3","l3"),("c4","l4"),("c5","l5")):
            refined = next(e["source"] for e in diagram["views"]["overview"]["edges"] if e["target"] == target)
            handles.append(getattr(model, f"attention_{level}_output").register_forward_hook(
                lambda module,args,output,refined=refined: measured.update({refined:list(output.shape[1:])})))
        try:
            output = model(torch.empty(1, *expected["input"]))
        finally:
            for handle in handles:
                handle.remove()
    measured["output"] = list(output.shape[1:])
    for name, shape in measured.items():
        if shape != expected[name]:
            raise ValueError(f"model shape mismatch at {name}: {shape} != {expected[name]}")
    diagram["shape_mode"] = "verified against PyTorch meta-device forward"
    diagram["model_parameters"] = sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.json")
    parser.add_argument("--output-dir", type=Path, default=VIEWER,
                        help="Directory for JSON and file:// compatible JavaScript preset")
    parser.add_argument("--svg-dir", type=Path, help="Also render every view to standalone SVG using Node.js")
    parser.add_argument("--svg-style", choices=("3d", "paper", "reference", "mono"), default="3d",
                        help="SVG presentation; paper overview includes block and fusion panels")
    parser.add_argument("--verify-model", action="store_true", help="Audit using PyTorch on the meta device")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        source = str(args.config.resolve().relative_to(ROOT)) if args.config.resolve().is_relative_to(ROOT) else args.config.name
        diagram = build_diagram(config, source=source)
        if args.verify_model:
            verify_model(config, diagram)
        export_diagram(diagram, args.output_dir)
        if args.svg_dir:
            subprocess.run(["node", str(VIEWER / "export-svg.cjs"),
                            str(args.output_dir / "mobilepixornext.json"), str(args.svg_dir),
                            "--style", args.svg_style], check=True)
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Cannot export backbone: {exc}\n")
    print(f"Exported {len(diagram['views'])} views to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Reproducible audit witnesses, not a PyTorch model execution or benchmark."""
import ast
import hashlib
import json
import math
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PAPER = ROOT / "paper"


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def static_architecture(depths, output_width, fusion_width=32, iou=True):
    """Count the source's Conv2d, affine BN and LayerScale parameters by formula.

    Source modules: mobilepixornext.py / mobilepixornext_blocks.py / heads/cnn.py.
    The counter does not count activations, interpolation or SimAM arithmetic.
    """
    records = []
    def conv(name, group, cin, cout, k, h, w, groups=1, bias=False, bn=True):
        params = cout * (cin // groups) * k * k + (cout if bias else 0)
        records.append({"name": name, "group": group, "parameters": params + (2 * cout if bn else 0),
                        "conv_macs": h * w * cout * (cin // groups) * k * k,
                        "output_shape_chw": [cout, h, w]})
    conv("stem.0", "body", 14, 32, 3, 400, 352)
    conv("stem.1", "body", 32, 32, 3, 400, 352)
    previous = 32
    for index, (depth, channels, h, w) in enumerate(zip(depths, (48, 96, 128), (200, 100, 50), (176, 88, 44)), 2):
        conv(f"down{index}", "body", previous, channels, 3, h, w)
        hidden = round(2.5 * channels)
        for block in range(depth):
            prefix = f"stage{index}.{block}"
            conv(prefix + ".dw", "body", channels, channels, 7, h, w, groups=channels)
            conv(prefix + ".expand", "body", channels, hidden, 1, h, w, bn=False)
            conv(prefix + ".project", "body", hidden, channels, 1, h, w)
            records.append({"name": prefix + ".layer_scale", "group": "body", "parameters": channels, "conv_macs": 0})
        previous = channels
    conv("lat_c5", "neck", 128, 48, 1, 50, 44, bn=False)
    conv("lat_c4", "neck", 96, 48, 1, 100, 88, bn=False)
    conv("lat_c3", "neck", 48, fusion_width, 1, 200, 176, bn=False)
    conv("refine_u4", "neck", 48, 48, 3, 100, 88, groups=48)
    conv("proj_u3", "neck", 48, fusion_width, 3, 200, 176)
    conv("gate_c4", "neck", 48, 48, 3, 100, 88, groups=48, bias=True, bn=False)
    conv("gate_c3", "neck", fusion_width, fusion_width, 3, 200, 176, groups=fusion_width, bias=True, bn=False)
    conv("out_conv", "neck", fusion_width, output_width, 3, 200, 176)
    for group_name, classes in [("car", 1), ("ped_cyc", 2)]:
        group = "head_" + group_name
        conv(group_name + ".shared", group, output_width, output_width, 3, 200, 176)
        branches = {"cls": classes, "offset": 2, "size": 2, "yaw": 2}
        if iou:
            branches["iou"] = 1
        for branch, outputs in branches.items():
            conv(group_name + "." + branch + ".private", group, output_width, output_width, 3, 200, 176)
            conv(group_name + "." + branch + ".out", group, output_width, outputs, 1, 200, 176, bias=True, bn=False)
    totals = {group: {key: sum(item[key] for item in records if item["group"] == group)
                      for key in ("parameters", "conv_macs")}
              for group in ("body", "neck", "head_car", "head_ped_cyc")}
    return {"depths": depths, "output_width": output_width, "fusion_width": fusion_width,
            "iou": iou, "groups": totals,
            "parameters": sum(item["parameters"] for item in records),
            "conv_macs": sum(item["conv_macs"] for item in records), "records": records}


config = json.loads((ROOT / "Result/iqa_comparison_20261019/config.inference.json").read_text())
log = json.loads((ROOT / "Result/benchmark_T4_20261018/sources/ours_best_ap.json").read_text())
final = static_architecture(config["model"]["stage_depths"], config["model"]["backbone_out_dim"], config["model"]["neck_fusion_channels"])
old = static_architecture([2, 4, 2], 16)
no_attention_same_shape = static_architecture([3, 4, 2], 32)
no_iou = static_architecture([3, 4, 2], 32, iou=False)
assert final["parameters"] == log["model"]["parameters"] == 753569
assert final["groups"]["body"]["parameters"] == 604864
assert final["groups"]["neck"]["parameters"] == 36784
assert final["groups"]["head_car"]["parameters"] == 55944
assert final["groups"]["head_ped_cyc"]["parameters"] == 55977
assert final["parameters"] == no_attention_same_shape["parameters"]
parameter_audit = {"scope": "static source-based arithmetic; no PyTorch forward, no latency measurement",
                   "full_network": final, "context_pipeline_2_4_2_H16": old,
                   "same_architecture_without_attention_parameters": no_attention_same_shape["parameters"],
                   "without_iou_parameters": no_iou["parameters"],
                   "iou_heads_delta": final["parameters"] - no_iou["parameters"],
                   "training_only_logits": 12,
                   "flop_convention_2_per_mac_excluding_nonconv": 2 * final["conv_macs"],
                   "excluded_ops": ["BN", "SiLU", "SimAM", "interpolation", "gates elementwise ops", "rasterization", "decode", "NMS"]}
save("parameter_and_mac_audit.json", parameter_audit)

cls = [0.9, 0.8]
quality = [0.04, 0.81]
score = [math.sqrt(p * q) for p, q in zip(cls, quality)]
peak_witness = {"adjacent_cells_cls": cls, "quality": quality, "joint_score": score,
                "threshold": 0.05, "classification_peak_index": cls.index(max(cls)),
                "joint_peak_index": score.index(max(score)),
                "result": "peak-selection changes, even though both cells pass both score thresholds"}
assert peak_witness["classification_peak_index"] != peak_witness["joint_peak_index"]

def corners(length, width, theta, order="code"):
    c, s = math.cos(theta), math.sin(theta)
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    if order == "cyclic":
        signs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    return [(a * length / 2 * c - b * width / 2 * s,
             a * length / 2 * s + b * width / 2 * c) for a, b in signs]


def error(a, b):
    return max(abs(x - y) for p, q in zip(a, b) for x, y in zip(p, q))


permutation = {}
for order in ["code", "cyclic"]:
    original = corners(4, 2, 0, order)
    rotated = corners(4, 2, math.pi, order)
    permutation[order] = {"reverse_order_pi_error": error(rotated, original[::-1])}
assert permutation["code"]["reverse_order_pi_error"] < 1e-12
assert permutation["cyclic"]["reverse_order_pi_error"] > 1

z = [0.0, 1.0]
mean = sum(z) / len(z)
population_std = math.sqrt(sum((x - mean)**2 for x in z) / len(z))
sample_std = math.sqrt(sum((x - mean)**2 for x in z) / (len(z) - 1))
outer_side, inner_side = 4.0, 2.0
interval_giou = inner_side / outer_side  # coincident centers, nested intervals, no hull penalty
area_iou = inner_side**2 / outer_side**2
stride_x = (log['data']['checkpoint_identity']['encoding']['metadata']['geometry']['x_res']
            * log['data']['checkpoint_identity']['output_stride'])
aug_sampler = config['augmentation']['AUG_CONFIG_LIST'][0]
protect_guard = bool(aug_sampler['ENABLE_VISIBILITY_PROTECTION'] and (
    aug_sampler['ENABLE_SHADOW_MASKING']
    or any(width > 0 for width in aug_sampler['REMOVE_EXTRA_WIDTH'])
    or aug_sampler['COLLISION_MARGIN'] <= 1e-8))
assert not protect_guard
scalar_witnesses = {"peak_selection": peak_witness, "corner_order": permutation,
                    "height_std_example": {"normalized_heights": z, "formula_population_std": population_std, "sample_std": sample_std},
                    "projection_proxy": {"gt_side": inner_side, "pred_side": outer_side,
                                         "mean_1d_giou": interval_giou, "area_iou": area_iou},
                    "rda_grid_shift": {"output_stride_m": stride_x, "code_added_half_cell_per_axis_m": stride_x/2},
                    "visibility_guard_final": {"configured": aug_sampler['ENABLE_VISIBILITY_PROTECTION'],
                                               "explicit_protect_check_active": protect_guard,
                                               "scope": "evaluated guard copied from hybrid_sampler.py; no data sampling run"},
                    "tsbuw_weight_limits_J6_b3_T2": {"lower": 6/(1+5*math.exp(3)), "upper": 6/(1+5*math.exp(-3))}}
save("mathematical_witnesses.json", scalar_witnesses)

train_ids = (ROOT / "Code/splits/kitti/train.txt").read_text().split()
val_ids = (ROOT / "Code/splits/kitti/val.txt").read_text().split()
assert len(train_ids) == 3712 and len(val_ids) == 3769
assert not set(train_ids) & set(val_ids)
save("dataset_split_check.json", {"train": len(train_ids), "val": len(val_ids), "overlap": 0,
                                  "train_sha256": hashlib.sha256((ROOT / "Code/splits/kitti/train.txt").read_bytes()).hexdigest(),
                                  "val_sha256": hashlib.sha256((ROOT / "Code/splits/kitti/val.txt").read_bytes()).hexdigest()})

save("resolved_evidence.json", {"model": config["model"], "augmentation": config["augmentation"],
                                "loss": config["loss"], "training": config["train"],
                                "checkpoint_identity": log["data"]["checkpoint_identity"],
                                "model_log": log["model"], "runtime_log": log["runtime"],
                                "latency_log": log["latency"], "protocol_log": log["protocol"]})
before = json.loads((OUT / "source_hashes_before.json").read_text())
changed = [p for p, h in before.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
assert not changed, changed
save("source_integrity.json", {"checked_files": len(before), "changed_files": changed})
print(json.dumps({"model_parameters": final["parameters"], "body": final["groups"]["body"]["parameters"],
                  "neck": final["groups"]["neck"]["parameters"], "iou_delta": parameter_audit["iou_heads_delta"],
                  "context_pipeline_parameters": old["parameters"],
                  "conv_GMACs_analytic_only": final["conv_macs"]/1e9, "peak_witness": peak_witness,
                  "dataset_overlap": 0, "source_files_unchanged": len(before)}, indent=2))

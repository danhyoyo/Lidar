#!/usr/bin/env python3
"""Preview original / sampled / world-transformed scenes with point provenance."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]
from prepare_kitti import read_ids
from core.datasets.augmentor.data_augmentor import DataAugmentor
from core.datasets.augmentor.geometry import bev_polygon, points_in_box


def preview_sampling(config, repo_dir=ROOT, *, frame_index=0, seed=42, force_sampling=False):
    repo = Path(repo_dir)
    root = repo / config["data"]["kitti"]["location"]
    data = config["data"]["kitti"]
    train_ids = read_ids(repo / config["train"]["data"])
    if not 0 <= frame_index < len(train_ids):
        raise ValueError(f"frame_index must be in [0,{len(train_ids)-1}]")
    frame = train_ids[frame_index]
    aug = copy.deepcopy(config["augmentation"])
    disabled = set(aug.get("DISABLE_AUG_LIST", []))
    ops = [op for op in aug.get("AUG_CONFIG_LIST", [])
           if op["NAME"] not in disabled and op.get("PROBABILITY", 1) > 0]
    gt_names = ("gt_sampling", "hybrid_gt_sampling")
    if (aug.get("mode") != "openpcdet" or not ops or ops[0]["NAME"] not in gt_names
            or sum(op["NAME"] in gt_names for op in ops) != 1):
        raise ValueError('Select AUGMENTATION="openpcdet_gt" or "hybrid_gt" and prepare its train database')
    op = ops[0]
    for path in op["DB_INFO_PATH"]:
        if not (root / path).is_file():
            raise FileNotFoundError(f"Prepare GT database first: {root / path}")
    points = np.fromfile(root / "pointcloud" / f"{frame}.bin", np.float32).reshape(-1, 4)
    boxes = []
    for line in (root / "label" / f"{frame}.txt").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in data["objects"]:
            if len(fields) != 8:
                raise ValueError("Expected detector labels: CLASS H W L X Y Z_BOTTOM YAW")
            boxes.append([data["objects"][fields[0]], *map(float, fields[1:])])
    boxes = np.asarray(boxes, np.float32).reshape(-1, 8)
    rng = np.random.default_rng(seed)
    kwargs = dict(root_path=root, class_names=data["objects"], rng=rng,
                  geometry=data["geometry"], allowed_frame_ids=train_ids)
    sampler = DataAugmentor(augmentor_configs={"AUG_CONFIG_LIST": [op]}, **kwargs).queue[0][2]
    probability = op.get("PROBABILITY", 1)
    skip = not force_sampling and probability < 1 and rng.random() >= probability
    if skip:
        p1, b1 = points.copy(), boxes.copy()
        meta = {"num_inserted": 0, "probability_skipped": True,
                "point_is_sampled": np.zeros(len(p1), bool), "box_is_sampled": np.zeros(len(b1), bool)}
    elif op["NAME"] == "hybrid_gt_sampling":
        p1, b1, meta = sampler(points, boxes, rng, return_metadata=True)
    else:
        p1, b1 = sampler(points, boxes, rng)
        keep = np.ones(len(points), bool)
        for box in b1[len(boxes):]:
            keep &= ~points_in_box(points, box, op.get("REMOVE_EXTRA_WIDTH", [0, 0, 0]))
        meta = {"num_inserted": len(b1)-len(boxes), "removed_interior_points": int((~keep).sum()),
                "point_is_sampled": np.arange(len(p1)) >= int(keep.sum()),
                "box_is_sampled": np.arange(len(b1)) >= len(boxes)}
    global_aug = DataAugmentor(augmentor_configs={"AUG_CONFIG_LIST": ops[1:]}, **kwargs)
    p2, b2 = global_aug(p1, b1)
    def stage(title, p, b, pm, bm):
        return dict(title=title, points=p, boxes=b, point_is_sampled=pm, box_is_sampled=bm)
    return {"frame_id": frame, "seed": seed, "forced": force_sampling, "sampler": op["NAME"],
        "metadata": meta, "stages": [
            stage("1. Original", points, boxes, np.zeros(len(points), bool), np.zeros(len(boxes), bool)),
            stage("2. GT sampling", p1, b1, meta["point_is_sampled"], meta["box_is_sampled"]),
            stage("3. GT + world transforms", p2, b2, meta["point_is_sampled"].copy(), meta["box_is_sampled"].copy())]}


def render_preview(preview, config, *, max_points=25000):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    geom = config["data"]["kitti"]["geometry"]
    names = {value: name for name, value in config["data"]["kitti"]["objects"].items()}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(19, 9), sharex=True, sharey=True)
    for ax, stage in zip(axes, preview["stages"]):
        points, boxes = stage["points"], stage["boxes"]
        ax.set_facecolor("#101820")
        visible = ((points[:, 0] > geom["x_min"]) & (points[:, 0] < geom["x_max"])
                   & (points[:, 1] > geom["y_min"]) & (points[:, 1] < geom["y_max"])
                   & (points[:, 2] >= geom["z_min"]) & (points[:, 2] < geom["z_max"]))
        for sampled, color, size in ((False, "#99a7b5", .3), (True, "#66ff66", 1.2)):
            idx = np.flatnonzero(visible & (stage["point_is_sampled"] == sampled))
            if len(idx) > max_points:
                idx = rng.choice(idx, max_points, replace=False)
            ax.scatter(points[idx, 1], points[idx, 0], s=size, c=color, alpha=.8 if sampled else .4, linewidths=0)
        for box, sampled in zip(boxes, stage["box_is_sampled"]):
            if not (geom["x_min"] < box[4] < geom["x_max"] and geom["y_min"] < box[5] < geom["y_max"]):
                continue
            color = "#66ff66" if sampled else "#4db8ff"
            corners = np.asarray(bev_polygon(box).exterior.coords)
            ax.plot(corners[:, 1], corners[:, 0], color=color, lw=1.2)
            ax.text(box[5], box[4], names[int(box[0])], color=color, fontsize=6)
        ax.set(xlim=(geom["y_max"], geom["y_min"]), ylim=(geom["x_min"], geom["x_max"]),
               xlabel="Y lateral (m)", title=stage["title"])
        ax.set_aspect("equal")
        ax.grid(alpha=.12)
    axes[0].set_ylabel("X forward (m)")
    fig.legend(handles=[Line2D([0], [0], color="#4db8ff", label="Original boxes"),
                        Line2D([0], [0], color="#66ff66", label="Inserted boxes / points")],
               loc="lower center", ncol=2)
    forced = " | forced preview; training probability unchanged" if preview["forced"] else ""
    fig.suptitle(f"{preview['sampler']} | Frame {preview['frame_id']} | seed={preview['seed']} | "
                 f"added={preview['metadata']['num_inserted']}{forced}")
    fig.tight_layout(rect=(0, .04, 1, .96))
    return fig


def print_summary(preview):
    meta = preview["metadata"]
    print(f"Frame={preview['frame_id']}; sampler={preview['sampler']}; seed={preview['seed']}; forced={preview['forced']}")
    for key in ("num_inserted", "probability_skipped", "accepted_by_class", "rejected_by_reason",
                "removed_interior_points", "removed_shadow_points", "geometry_backend", "cache", "inserted_sources"):
        if key in meta:
            print(f"{key}: {meta[key]}")
    if not meta["num_inserted"]:
        print("No insertion: inspect rejection counters or try a different frame/seed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-sampling", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    preview = preview_sampling(cfg, frame_index=args.frame_index, seed=args.seed, force_sampling=args.force_sampling)
    print_summary(preview)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_preview(preview, cfg).savefig(args.output, dpi=140)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze and visualize actual 8/14/18/22-channel KITTI BEV inputs.

Uses the completed CANUPO feasibility split/cores for paired weak-label probes.
Experimental moment-window features are computed from points, never GT boxes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "tools/analysis")]
from bev_geometry_maps import encode_maps, channel_names, SCALES
from analyze_canupo_kitti import digest, write_json, binary_metrics, operating_threshold
from tools.visualization.export_kitti_overview import dataset_paths, read_points, read_objects, CLASS_COLORS

VARIANTS = ("rich8", "hist14", "hist18", "hist22")
LABELS = {"rich8": "Rich8", "hist14": "Hist14 (historical Rich14)",
          "hist18": "Hist14 + neighborhood", "hist22": "Hist14 + neighborhood + geometry"}
SHOWCASE = ("000804", "004425", "004567", "001801")


def sample_features(bundle, xyz, geometry):
    # Hist14 promotes coordinates before floor; legacy Rich8 uses float32 //.
    # Preserve each rasterizer's actual indexing, especially at bin boundaries.
    promoted = xyz.astype(np.float64)
    ix = np.floor((promoted[:, 0] - geometry["x_min"]) / geometry["x_res"]).astype(int)
    iy = np.floor((promoted[:, 1] - geometry["y_min"]) / geometry["y_res"]).astype(int)
    result = {key: bundle[key][iy, ix].copy() for key in ["hist14", "hist18", "hist22"]}
    rich_x = ((xyz[:, 0] - geometry["x_min"]) // geometry["x_res"]).astype(int)
    rich_y = ((xyz[:, 1] - geometry["y_min"]) // geometry["y_res"]).astype(int)
    result["rich8"] = bundle["rich8"][rich_y, rich_x].copy()
    return result


def extract(args, metadata, paths, geometry):
    pieces = {key: [] for key in (*VARIANTS, "labels", "uniform", "frame", "partition", "selection")}
    timing = []
    snapshots = {}
    for j, row in enumerate(metadata):
        frame = row["frame"]
        points = read_points(paths.velodyne / f"{frame}.bin")
        if digest(paths.velodyne / f"{frame}.bin") != row["hashes"]["scan"]:
            raise ValueError("Raw scan changed since CANUPO study")
        bundle = encode_maps(points, geometry)
        with np.load(args.study / "frames" / frame / "features.npz") as previous:
            values = sample_features(bundle, previous["xyz"], geometry)
            # All new variants retain the original Hist14 channels exactly.
            np.testing.assert_array_equal(bundle["hist14"], bundle["hist22"][..., :14])
            np.testing.assert_array_equal(values["hist14"][:, 11], np.ones(len(previous["xyz"])))
            if not np.all(values["rich8"][:, 7] > 0):
                raise ValueError("Rich8 core-to-pillar indexing mismatch")
            for key, value in values.items():
                pieces[key].append(value)
            n = len(previous["labels"])
            for key in ("labels", "uniform"):
                pieces[key].append(previous[key].copy())
            for key in ("frame", "partition", "selection"):
                pieces[key].append(np.full(n, row[key]))
        occupied = bundle["occupied"]
        timing.append({"frame": frame, "roi_points": bundle["roi_points"],
                       "occupied_pillars": int(occupied.sum()), **bundle["timing"],
                       "valid_1m": float(bundle["hist22"][..., 15][occupied].mean()),
                       "valid_2m": float(bundle["hist22"][..., 17][occupied].mean())})
        if frame in args.frames:
            folder = args.output / frame
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "bev_tensors.npz", **{key: bundle[key] for key in VARIANTS},
                                counts=bundle["counts"], occupied=bundle["occupied"],
                                barycentric=bundle["barycentric"])
            snapshots[frame] = {"windows": bundle["windows"], "occupied_pillars": int(occupied.sum()),
                                "timing": bundle["timing"], "roi_points": bundle["roi_points"]}
        print(f"MAP {j + 1}/{len(metadata)} {frame} occupied={occupied.sum()} extra={bundle['timing']['geometry_seconds'] * 1000:.1f}ms", flush=True)
    packed = {key: np.concatenate(value) for key, value in pieces.items()}
    np.savez_compressed(args.output / "sampled_bev_features.npz", **packed)
    write_json(args.output / "encoding_timing.json", timing)
    write_json(args.output / "showcase_metadata.json", snapshots)
    return packed, timing, snapshots


def evaluate(packed, output, reuse_summary=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.utils.class_weight import compute_sample_weight
    fit = packed["partition"] == "fit"
    tune = (packed["partition"] == "tune") & packed["uniform"]
    primary = ((packed["partition"] == "validation") & packed["uniform"]
               & (packed["selection"] == "random"))
    labels = packed["labels"] > 0
    models, scores, result = {}, {}, {}
    for name in VARIANTS:
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=.08,
                                               l2_regularization=1., early_stopping=False, random_state=42)
        model.fit(packed[name][fit], labels[fit], sample_weight=compute_sample_weight("balanced", labels[fit]))
        prediction = model.predict_proba(packed[name])[:, 1]
        threshold = operating_threshold(labels[tune], prediction[tune])
        metric = binary_metrics(labels[primary], prediction[primary], threshold)
        metric["channels"] = packed[name].shape[1]
        result[name] = metric
        models[name], scores[name] = model, prediction
        print("PROBE", name, json.dumps(metric), flush=True)
    previous = json.loads((output / "metrics.json").read_text()) if reuse_summary and (output / "metrics.json").exists() else None
    if previous:
        for name in VARIANTS:
            if abs(previous["metrics"][name]["average_precision"] - result[name]["average_precision"]) > 1e-12:
                raise ValueError("Render-only refit differs from archived metrics; use a fresh full run")
    rng = np.random.default_rng(43)
    groups = [np.flatnonzero(primary & (packed["frame"] == f)) for f in np.unique(packed["frame"][primary])]
    deltas = {"hist18_minus_hist14": [], "hist22_minus_hist18": [], "hist22_minus_hist14": []}
    for _ in range(0 if previous else 200):
        ids = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        ap = {key: average_precision_score(labels[ids], scores[key][ids]) for key in ["hist14", "hist18", "hist22"]}
        deltas["hist18_minus_hist14"].append(ap["hist18"] - ap["hist14"])
        deltas["hist22_minus_hist18"].append(ap["hist22"] - ap["hist18"])
        deltas["hist22_minus_hist14"].append(ap["hist22"] - ap["hist14"])
    summary = {"metrics": result, "primary_frames": len(groups), "primary_points": int(primary.sum()),
               "bootstrap": previous["bootstrap"] if previous else {key: np.quantile(values, [.025, .975]).tolist() for key, values in deltas.items()},
               "bootstrap_replicates": 200, "bootstrap_seed": 43,
               "protocol": "Same 144 frames/cores and weak labels as CANUPO study; actual BEV channel features only, no extra per-point z/intensity/range."}
    write_json(output / "metrics.json", summary)
    np.savez_compressed(output / "probe_predictions.npz", **scores, labels=packed["labels"], primary=primary,
                        frame=packed["frame"], uniform=packed["uniform"])
    with (output / "metrics.csv").open("w") as stream:
        fields = ["variant", "channels", "average_precision", "auroc", "recall", "outside_box_rejection"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name, m in result.items():
            writer.writerow({"variant": name, **{key: m[key] for key in fields[1:]}})
    # Display scales and feature statistics use only fit samples, without labels.
    names = channel_names()
    display_max = {str(j): float(np.quantile(packed["hist22"][fit, j], .995)) for j in [14, 16]}
    feature_stats = []
    for j, name in enumerate(names["hist22"]):
        values = packed["hist22"][fit, j]
        correlations = []
        if values.std() > 1e-9:
            for k in range(14):
                base = packed["hist14"][fit, k]
                corr = 0. if base.std() < 1e-9 else float(np.corrcoef(values, base)[0, 1])
                correlations.append((abs(corr), corr, names["hist14"][k]))
        strongest = max(correlations) if correlations else (0., 0., "constant")
        feature_stats.append({"index": j, "name": name, "min": float(values.min()), "max": float(values.max()),
                              "mean": float(values.mean()), "std": float(values.std()),
                              "quantiles": np.quantile(values, [.01, .5, .99]).tolist(),
                              "strongest_hist14_correlation": {"channel": strongest[2], "pearson": strongest[1]}})
    write_json(output / "feature_statistics.json", {"fit_only": True, "display_count_log_max": display_max,
                                                    "channels": feature_stats})
    return models, summary, display_max


def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#0b1020", "axes.facecolor": "#0b1020",
                        "text.color": "#e5e7eb", "axes.labelcolor": "#cbd5e1",
                        "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
                        "axes.edgecolor": "#334155", "font.size": 9})
    return plt


def draw_gt(ax, objects, limits=None):
    for obj in objects:
        center = obj.center
        if limits and not (limits[0] <= center[1] <= limits[1] and limits[2] <= center[0] <= limits[3]):
            continue
        corners = obj.corners[:4]
        ax.plot(np.r_[corners[:, 1], corners[0, 1]], np.r_[corners[:, 0], corners[0, 0]],
                color=CLASS_COLORS[obj.name], linewidth=.8, alpha=.9)


def map_image(ax, values, geometry, objects, title, *, limits=None, cmap="magma", vmin=0., vmax=1.):
    image = values.transpose(1, 0, 2) if values.ndim == 3 else values.T
    extent = [geometry["y_min"], geometry["y_max"], geometry["x_min"], geometry["x_max"]]
    handle = ax.imshow(image, origin="lower", extent=extent, interpolation="nearest", cmap=cmap,
                       vmin=vmin if values.ndim == 2 else None, vmax=vmax if values.ndim == 2 else None)
    draw_gt(ax, objects, limits)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("LiDAR y (m)"); ax.set_ylabel("LiDAR x (m)")
    if limits:
        ax.set_xlim(*limits[:2]); ax.set_ylim(*limits[2:])
    return handle


def scalar_rgba(values, occupied, *, lo=0., hi=1., cmap="magma"):
    import matplotlib
    normalized = np.clip((values - lo) / max(hi - lo, 1e-12), 0, 1)
    rgba = matplotlib.colormaps[cmap](normalized)
    rgba[~occupied] = [11 / 255, 16 / 255, 32 / 255, 1.]
    return rgba


def export_scene(frame, args, geometry, paths, models, display_max):
    plt = setup_plotting()
    folder = args.output / frame
    with np.load(folder / "bev_tensors.npz") as z:
        bundle = {key: z[key] for key in z.files}
    objects = read_objects(paths.labels / f"{frame}.txt", paths.calib / f"{frame}.txt")
    occupied = bundle["occupied"]
    h = bundle["hist22"]
    rgb_hist = np.stack([h[..., 3], h[..., 1] + h[..., 2], h[..., 0]], axis=-1).clip(0, 1)
    rgb_hist[~occupied] = [11 / 255, 16 / 255, 32 / 255]
    rich = bundle["rich8"]
    rich_occupied = rich[..., 7] > 0
    rgb_rich = np.stack([rich[..., 2], rich[..., 1], rich[..., 0]], axis=-1)
    rgb_rich[~rich_occupied] = [11 / 255, 16 / 255, 32 / 255]
    # R/G/B = 1D/2D/3D dimensionality. Invalid/empty is dark, not fake 3D.
    shape_rgb = []
    for j in range(2):
        rgb = bundle["barycentric"][..., j, :].copy()
        rgb[h[..., 15 + 2 * j] == 0] = [11 / 255, 16 / 255, 32 / 255]
        shape_rgb.append(rgb)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = [(rgb_rich, "Rich8: high / middle / low occupancy RGB", None),
              (rgb_hist, "Hist14: high / middle / low count RGB", None),
              (h[..., 4], "Hist14: normalized maximum height", 1.),
              (h[..., 14], "Added neighborhood: log(1+N), 1.1m XY", display_max["14"]),
              (shape_rgb[0], "Added geometry: 1.1m XY window", None),
              (shape_rgb[1], "Added geometry: 2.1m XY window", None)]
    for ax, (values, title, vmax) in zip(axes.flat, panels):
        if values.ndim == 2:
            values = scalar_rgba(values, occupied, hi=vmax)
        map_image(ax, values, geometry, objects, title)
    fig.suptitle(f"KITTI {frame} — real BEV layers, GT outlines only\nGeometry RGB: red=1D, green=2D, blue=3D; dark=empty/invalid", fontsize=14)
    fig.subplots_adjust(left=.05, right=.99, bottom=.06, top=.85, hspace=.4, wspace=.24)
    fig.savefig(folder / "overview.png", dpi=170); plt.close(fig)
    # Full channel sheets; scales fixed across frames and variants.
    names = channel_names()
    for key, indices, shape in [("hist14", list(range(14)), (4, 4)),
                                ("added8", list(range(14, 22)), (2, 4)),
                                ("rich8", list(range(8)), (2, 4))]:
        tensor = rich if key == "rich8" else h
        labels = names["rich8"] if key == "rich8" else names["hist22"]
        fig, axes = plt.subplots(*shape, figsize=(17, 4.4 * shape[0]))
        for ax, index in zip(axes.flat, indices):
            lo, hi = (-.5, .5) if index in (12, 13) and key != "rich8" else (0., 1.)
            if key != "rich8" and index in (14, 16): hi = display_max[str(index)]
            mask = rich_occupied if key == "rich8" else occupied
            rgba = scalar_rgba(tensor[..., index], mask, lo=lo, hi=hi, cmap="coolwarm" if lo < 0 else "magma")
            map_image(ax, rgba, geometry, objects, f"ch{index}: {labels[index]}\nrange [{lo:g}, {hi:.3g}]")
        for ax in list(axes.flat)[len(indices):]: ax.axis("off")
        fig.suptitle(f"KITTI {frame} | {key} | individual tensor channels | GT outlines", fontsize=13)
        fig.subplots_adjust(left=.045, right=.99, bottom=.045, top=.93, hspace=.52, wspace=.3)
        fig.savefig(folder / f"channels_{key}.png", dpi=150); plt.close(fig)
    # Direct comparison uses a separate, clearly named weak-label classifier map.
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.3))
    for ax, variant in zip(axes, VARIANTS):
        score = np.zeros(occupied.shape, np.float32)
        mask = rich_occupied if variant == "rich8" else occupied
        score[mask] = models[variant].predict_proba(bundle[variant][mask])[:, 1]
        rgba = scalar_rgba(score, mask)
        map_image(ax, rgba, geometry, objects, f"{LABELS[variant]}\nweak-label foreground score")
        bundle["score_" + variant] = score
    fig.suptitle(f"KITTI {frame} — paired probe scores, not detector outputs\nIdentical color scale 0–1; GT boxes are overlays, not encoder inputs", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(folder / "foreground_comparison.png", dpi=170); plt.close(fig)
    # Select one supported instance per class for paired close-up inspection.
    counts_by_obj = {}
    old_meta = json.loads((args.study / "frames" / frame / "metadata.json").read_text())
    for obj in old_meta["objects"]:
        counts_by_obj[(obj["class"], obj["id"])] = obj["roi_points"]
    selected = []
    for name in ("Car", "Pedestrian", "Cyclist"):
        candidates = [(j, obj) for j, obj in enumerate(objects) if obj.name == name and obj.center[0] > 0]
        if candidates:
            selected.append(max(candidates, key=lambda pair: counts_by_obj.get((pair[1].name, pair[0]), 0))[1])
    if selected:
        fig, axes = plt.subplots(len(selected), 5, figsize=(18, 3.7 * len(selected)), squeeze=False)
        for row, obj in enumerate(selected):
            center = obj.center
            half = max(2., float(np.ptp(obj.corners[:, :2], axis=0).max()) / 2 + 1.2)
            limits = (center[1] - half, center[1] + half, center[0] - half, center[0] + half)
            panels = [(rgb_rich, "Rich8 occupancy"), (rgb_hist, "Hist14 height counts"),
                      (scalar_rgba(h[..., 14], occupied, hi=display_max["14"]), "Neighborhood 1.1m"),
                      (shape_rgb[0], "Geometry 1.1m"), (shape_rgb[1], "Geometry 2.1m")]
            for ax, (values, title) in zip(axes[row], panels):
                map_image(ax, values, geometry, objects, f"{obj.name}, {np.linalg.norm(center[:2]):.1f}m\n{title}", limits=limits)
        fig.suptitle(f"KITTI {frame} — object crops of real feature maps; same coordinates in each row", fontsize=13)
        fig.subplots_adjust(left=.045, right=.99, bottom=.06, top=.88, hspace=.55, wspace=.35)
        fig.savefig(folder / "object_zooms.png", dpi=190); plt.close(fig)
    # Axis-free, correctly oriented channel tiles for the local side-by-side viewer.
    tiles = folder / "tiles"; tiles.mkdir(exist_ok=True)
    for variant, tensor, labels in [("hist22", h, names["hist22"]), ("rich8", rich, names["rich8"])]:
        for index, label in enumerate(labels):
            lo, hi = (-.5, .5) if variant == "hist22" and index in (12, 13) else (0., 1.)
            if variant == "hist22" and index in (14, 16): hi = display_max[str(index)]
            mask = rich_occupied if variant == "rich8" else occupied
            rgba = scalar_rgba(tensor[..., index], mask, lo=lo, hi=hi, cmap="coolwarm" if lo < 0 else "magma")
            plt.imsave(tiles / f"{variant}_{index}.png", np.flipud(rgba.transpose(1, 0, 2)))
    for j, rgb in enumerate(shape_rgb):
        plt.imsave(tiles / f"geometry_rgb_{j}.png", np.flipud(rgb.transpose(1, 0, 2)))
    for variant in VARIANTS:
        mask = rich_occupied if variant == "rich8" else occupied
        rgba = scalar_rgba(bundle["score_" + variant], mask)
        plt.imsave(tiles / f"score_{variant}.png", np.flipud(rgba.transpose(1, 0, 2)))
    return {"frame": frame, "objects": [{"class": o.name, "corners": o.corners[:4, :2].tolist()} for o in objects]}


def write_viewer(output, frames, geometry, display_max):
    payload = {"frames": frames, "geometry": geometry, "channels": channel_names(), "display": display_max}
    html = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BEV: Hist14, lân cận và hình học</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#e2e8f0;font:15px system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 12px}p{line-height:1.6;color:#b9c6da}select,button{background:#172238;color:#e2e8f0;border:1px solid #40516e;border-radius:7px;padding:9px;margin:4px}button{cursor:pointer}a{color:#70d6ff}header{display:flex;align-items:center;flex-wrap:wrap;gap:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pane{background:#111b2e;border:1px solid #293854;border-radius:12px;padding:12px}.stage{position:relative;overflow:hidden;background:#0b1020;aspect-ratio:800/704;touch-action:none;cursor:grab}.layer{position:absolute;width:100%;height:100%;transform-origin:0 0}.layer img,.layer svg{position:absolute;left:0;top:0;width:100%;height:100%;image-rendering:pixelated}.info{min-height:45px;color:#95adc9;font-size:13px;margin-top:10px}.links{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}.legend{padding:12px;background:#152036;border-radius:8px}.gradient{display:inline-block;width:140px;height:12px;background:linear-gradient(90deg,#000004,#51127c,#b73779,#fc8961,#fcfdbf);margin:0 8px}.note{font-size:13px} @media(max-width:850px){.grid{grid-template-columns:1fr}}
</style><main><h1>BEV: Hist14 + lân cận + hình học</h1>
<p>Tensor thật trên cùng point cloud KITTI. Hist14 là biểu diễn 14 kênh từng được ghi là “Rich14”; Rich8 là baseline riêng. Các kênh Hist14 được giữ nguyên ở bản 18/22 kênh.</p>
<header><label>Frame <select id="frame"></select></label><label><input id="gt" type="checkbox" checked> Hộp GT</label><button id="reset">Đặt lại zoom</button><span>Kéo để dịch, lăn chuột để zoom — hai map đồng bộ.</span></header>
<div class="grid"><section class="pane"><label>Trái <select class="variant" id="v0"></select></label><select id="c0"></select><div class="stage" id="s0"><div class="layer" id="l0"><img id="i0"><svg id="g0" viewBox="0 0 800 704"></svg></div></div><div class="info" id="t0"></div></section>
<section class="pane"><label>Phải <select class="variant" id="v1"></select></label><select id="c1"></select><div class="stage" id="s1"><div class="layer" id="l1"><img id="i1"><svg id="g1" viewBox="0 0 800 704"></svg></div></div><div class="info" id="t1"></div></section></div>
<p class="legend"><span class="gradient"></span> Kênh scalar: thấp → cao. Geometry RGB: <span style="color:#ff6464">đỏ = 1D</span>, <span style="color:#6ee788">xanh lá = 2D</span>, <span style="color:#79aaff">xanh dương = 3D</span>. Ô trống/geometry invalid có nền tối.</p>
<div class="links" id="links"></div><p class="note">Lân cận là cửa sổ XY 11×11 / 21×21 ô tại độ phân giải 0,1 m, có chiều cao của toàn ROI; khác quả cầu 3D của CANUPO. Đủ 6 lân cận và covariance không suy biến mới có geometry hợp lệ. Hộp GT chỉ dùng để đối chiếu, không tham gia tạo feature. Map foreground là classifier thử với nhãn yếu, không phải detector.</p>
<p><a href="metrics.json">Metrics</a> · <a href="feature_statistics.json">Thống kê kênh</a> · <a href="manifest.json">Protocol</a></p></main>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);let zoom=1,tx=0,ty=0,drag=null;
const labels={rich8:'Rich8 · 8 kênh',hist14:'Hist14 / Rich14 cũ · 14 kênh',hist18:'Hist14 + lân cận · 18 kênh',hist22:'Hist14 + lân cận + hình học · 22 kênh'};
DATA.frames.forEach(f=>$('frame').add(new Option(f.frame,f.frame)));
for(let k=0;k<2;k++){Object.entries(labels).forEach(([v,t])=>$('v'+k).add(new Option(t,v)));$('v'+k).value=k?'hist22':'hist14';$('v'+k).onchange=()=>channels(k);$('c'+k).onchange=()=>render(k);}
function channels(k){let v=$('v'+k).value,c=$('c'+k);c.replaceChildren();DATA.channels[v].forEach((t,j)=>c.add(new Option('ch'+j+' · '+t,String(j))));if(v==='hist22'){c.add(new Option('Geometry RGB · 1,1 m','rgb0'));c.add(new Option('Geometry RGB · 2,1 m','rgb1'));}c.add(new Option('Foreground score · probe','score'));c.value=(k===1&&v==='hist22')?'rgb0':'4';render(k)}
function render(k){let v=$('v'+k).value,c=$('c'+k).value,f=$('frame').value;let tile=c==='score'?'score_'+v:c.startsWith('rgb')?'geometry_rgb_'+c.slice(3):(v==='rich8'?'rich8':'hist22')+'_'+c;$('i'+k).src=f+'/tiles/'+tile+'.png';let range=c==='score'?'0–1, weak-label probe':c.startsWith('rgb')?'1D / 2D / 3D':v!=='rich8'&&[14,16].includes(+c)?'log(1+N), display max '+DATA.display[c].toFixed(2):v!=='rich8'&&[12,13].includes(+c)?'−0,5 … +0,5':'0 … 1';$('t'+k).textContent=labels[v]+' | '+range;overlay(k)}
function overlay(k){let f=DATA.frames.find(f=>f.frame===$('frame').value),g=DATA.geometry;let colors={Car:'#ff6257',Pedestrian:'#88eb93',Cyclist:'#ffca55'};$('g'+k).innerHTML=$('gt').checked?f.objects.map(o=>'<polygon points="'+o.corners.map(p=>[(p[1]-g.y_min)/(g.y_max-g.y_min)*800,(g.x_max-p[0])/(g.x_max-g.x_min)*704].join(',')).join(' ')+'" fill="none" stroke="'+colors[o.class]+'" stroke-width="1.1"/>').join(''):''}
function update(){for(let k=0;k<2;k++)$('l'+k).style.transform=`translate(${tx}px,${ty}px) scale(${zoom})`}
$('frame').onchange=()=>{render(0);render(1);links()};$('gt').onchange=()=>{overlay(0);overlay(1)};$('reset').onclick=()=>{zoom=1;tx=ty=0;update()};
function links(){let f=$('frame').value;$('links').innerHTML=[['overview.png','Toàn cảnh lớp thông tin'],['channels_hist14.png','14 kênh Hist14'],['channels_added8.png','8 kênh bổ sung'],['object_zooms.png','Phóng to đối tượng'],['foreground_comparison.png','So sánh foreground probe'],['bev_tensors.npz','Tensor NPZ']].map(([p,t])=>'<a target="_blank" href="'+f+'/'+p+'">'+t+'</a>').join('')}
for(let k=0;k<2;k++){let el=$('s'+k);el.addEventListener('wheel',e=>{e.preventDefault();let r=el.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,old=zoom;zoom=Math.min(12,Math.max(1,zoom*Math.exp(-e.deltaY*.001)));tx=x-(x-tx)*zoom/old;ty=y-(y-ty)*zoom/old;update()},{passive:false});el.onpointerdown=e=>{drag=[e.clientX,e.clientY,tx,ty];el.setPointerCapture(e.pointerId)};el.onpointermove=e=>{if(drag){tx=drag[2]+e.clientX-drag[0];ty=drag[3]+e.clientY-drag[1];update()}};el.onpointerup=()=>drag=null;el.onpointercancel=()=>drag=null;}
channels(0);channels(1);links();</script></html>'''
    (output / "index.html").write_text(html.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT.parent / "artifacts/canupo_feasibility_20261016/main")
    parser.add_argument("--kitti-root", type=Path, default=ROOT / "Kitti_Raw")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "artifacts/bev_geometry_maps_20261016")
    parser.add_argument("--frames", nargs="+", default=list(SHOWCASE))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(); args.output = args.output.resolve(); args.study = args.study.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = json.loads((ROOT / "configs/config.json").read_text())["data"]["kitti"]["geometry"]
    metadata = json.loads((args.study / "frames.json").read_text())
    if not set(args.frames) <= {m["frame"] for m in metadata if m["partition"] == "validation"}:
        raise ValueError("Showcase frames must belong to study validation")
    paths = dataset_paths(args.kitti_root)
    manifest = {"script_sha256": digest(__file__), "encoder_sha256": digest(ROOT / "tools/analysis/bev_geometry_maps.py"),
                "sampling_version": "hist14_float64_floor_rich8_native_float32_divmod_v2",
                "study_manifest_sha256": digest(args.study / "manifest.json"), "geometry": geometry,
                "channels": channel_names(), "showcase_frames": args.frames,
                "method": "Float64 pooled xyz moments in centered XY column windows; ROI-clipped; not original CANUPO spheres.",
                "min_neighbors": 6, "scales_m": list(SCALES),
                "preserves_hist14": True, "labels_used_in_encoding": False,
                "rich14_resolution": "Historical name for Hist14 in project figures; no independent rich14 encoder.",
                "command": sys.argv}
    manifest_path = args.output / "manifest.json"
    if args.render_only and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("sampling_version", "encoder_sha256", "study_manifest_sha256", "geometry", "showcase_frames"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Cached visualization protocol changed: {key}")
    write_json(manifest_path, manifest)
    if args.render_only:
        with np.load(args.output / "sampled_bev_features.npz") as z:
            packed = {k: z[k] for k in z.files}
        snapshots = json.loads((args.output / "showcase_metadata.json").read_text())
    else:
        packed, timing, snapshots = extract(args, metadata, paths, geometry)
    models, metrics, display_max = evaluate(packed, args.output, reuse_summary=args.render_only)
    views = [export_scene(frame, args, geometry, paths, models, display_max) for frame in args.frames]
    write_viewer(args.output, views, geometry, display_max)
    print("COMPLETE", args.output / "index.html", flush=True)


if __name__ == "__main__":
    main()

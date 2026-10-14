#!/usr/bin/env python3
"""Visualize attention placement interventions on one strictly loaded checkpoint.

C4 Focal and SimAM use trained weights. Focal at C3/C5 is newly initialized.
These images are not trained-placement ablations or detection-quality evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import html
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/lidar-attention-matplotlib')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'detector')]
import numpy as np

STAGES = ('off', 'C3', 'C4', 'C5')
MODULES = {'C3': 'stage2', 'C4': 'stage3', 'C5': 'stage4'}
COLORS = {'Car': '#ff6767', 'Pedestrian': '#62e38e', 'Cyclist': '#ffda62'}


def checkpoint_forward(model, tensor):
    """Capture the trained production topology without inserting or replacing modules."""
    from tools.visualization.export_kitti_pipeline import backbone_capture_modules, capture_forward
    return capture_forward(model, tensor, modules=backbone_capture_modules(model))


def export_checkpoint_view(model, tensor, *, output, geometry, objects, spec, config,
                           frame, points, checkpoint_path, checkpoint_hash):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from tools.visualization.predict_kitti_overview import classification_heatmaps
    prediction, captured = checkpoint_forward(model, tensor)
    classes = sorted(config['data']['kitti']['objects'], key=config['data']['kitti']['objects'].get)
    logits, probabilities = classification_heatmaps(prediction, groups=spec.groups, num_classes=len(classes))
    np.savez_compressed(output/'tensors/trained_features.npz', **captured,
                        logits=logits, probabilities=probabilities)
    np.savez_compressed(output/'tensors/input.npz', hist14=tensor[0].numpy(), points_xyzi=points)
    extent = [geometry['x_min'],geometry['x_max'],geometry['y_min'],geometry['y_max']]
    panels = [(stage.upper()+' refined features',energy(captured[f'backbone.attention_{stage}_output']),None)
              for stage in ('c3','c4','c5')]
    panels += [('Final features',energy(captured['backbone']),None)]
    panels += [(name+' probability',probabilities[i],1.) for i,name in enumerate(classes)]
    fig,axes = plt.subplots(2,4,figsize=(16,10),constrained_layout=True)
    for ax,(title,values,bound) in zip(axes.ravel(),panels):
        bound = bound if bound is not None else max(float(np.percentile(values,99.5)),1e-12)
        im = ax.imshow(values,origin='lower',extent=extent,vmin=0,vmax=bound,cmap='magma')
        for obj in objects:
            ax.add_patch(Polygon(obj.corners[:4,:2],fill=False,edgecolor=COLORS[obj.name],lw=.6))
        ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    placement = model.backbone.feature_options.effective_placement()
    fig.suptitle(f'Trained checkpoint | KITTI {frame}\n{placement}',fontsize=11)
    fig.savefig(output/'trained_feature_maps.png',dpi=170,bbox_inches='tight')
    plt.close(fig)
    metadata = {'mode':'checkpoint','frame':frame,'checkpoint':str(checkpoint_path),
        'checkpoint_sha256':checkpoint_hash,'effective_attention_placement':placement,
        'config':config,'point_count':len(points),'strict_checkpoint_load':True,
        'modules_replaced':False,'new_attention_weights_initialized':False,
        'limitations':['Single-frame activations, no AP comparison.',
                       'Feature RMS panels each use their own 99.5th-percentile scale.',
                       'Classification probabilities before IQA/NMS; outlines are GT.']}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Trained checkpoint features</title>'
        '<h1>Trained checkpoint feature maps</h1><p>Actual saved topology; no attention interventions. '
        'GT outlines; no AP comparison.</p><a href="metadata.json">Metadata</a>'
        '<img src="trained_feature_maps.png" style="width:100%">')
    return metadata


def placement_forward(model, tensor, focals, simam, focal='off', sim='off', order='FS'):
    """Temporarily intervene after complete stages; restore even on failure."""
    import torch
    if model.training or focal not in STAGES or sim not in STAGES or order not in ('FS', 'SF'):
        raise ValueError('Expected eval model and valid placement/order')
    if not model.backbone.feature_options.has_legacy_placement:
        raise ValueError('Interventions require a native C4/C3 checkpoint; use --mode checkpoint for trained placements')
    backbone = model.backbone
    originals = backbone.c3_light_attention, backbone.c4_context
    handles, captured = [], {}

    def stage_hook(stage):
        def hook(module, inputs, output):
            value = output
            captured[stage + '_before'] = value.detach().float().cpu().numpy().copy()
            for kind in order:
                if kind == 'F' and focal == stage:
                    _, _, gates = focals[stage].prepare_context(value)
                    captured[stage + '_focal_gates'] = gates.detach().float().cpu().numpy().copy()
                    value = focals[stage](value)
                elif kind == 'S' and sim == stage:
                    gates = simam.core.spatial_weights(value)
                    captured[stage + '_simam_gate'] = gates.detach().float().cpu().numpy().copy()
                    value = simam(value)
            captured[stage] = value.detach().float().cpu().numpy().copy()
            return value
        return hook

    def final_hook(module, inputs, output):
        captured['final'] = output.detach().float().cpu().numpy().copy()

    try:
        backbone.c3_light_attention = torch.nn.Identity()
        backbone.c4_context = torch.nn.Identity()
        for stage, name in MODULES.items():
            handles.append(getattr(backbone, name).register_forward_hook(stage_hook(stage)))
        handles.append(backbone.register_forward_hook(final_hook))
        with torch.inference_mode():
            prediction = model(tensor)
        if not all(np.isfinite(value).all() for value in captured.values()):
            raise ValueError('Non-finite captured activations')
        return prediction, captured
    finally:
        for handle in handles:
            handle.remove()
        backbone.c3_light_attention, backbone.c4_context = originals


def energy(feature):
    return np.sqrt(np.mean(np.square(feature[0], dtype=np.float64), axis=0)).astype(np.float32)


def split_ids(path):
    return {line.split(';')[0].strip() for line in path.read_text().splitlines() if line.strip()}


def choose_frame(paths, geometry, split):
    from tools.visualization.export_kitti_overview import read_objects
    candidates = []
    allowed = split_ids(split) if split and split.is_file() else None
    for label in sorted(paths.labels.glob('*.txt')):
        if allowed is not None and label.stem not in allowed:
            continue
        if not (paths.velodyne / (label.stem + '.bin')).is_file():
            continue
        if not (paths.calib / label.name).is_file():
            continue
        cheap = Counter()
        for line in label.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in COLORS and 0 < float(fields[13]) < 72:
                cheap[fields[0]] += 1
        candidates.append((sum(cheap.values()), label.stem))
    if not candidates:
        raise ValueError('No labeled pointcloud available')
    ranked = []
    for _, frame in sorted(candidates, reverse=True)[:60]:
        objects = read_objects(paths.labels / (frame + '.txt'), paths.calib / (frame + '.txt'))
        inside = [obj for obj in objects if in_range(obj.corners.mean(axis=0), geometry)]
        counts = Counter(obj.name for obj in inside)
        # Prefer scenes containing all classes, then the greatest object count.
        ranked.append((len(counts), len(inside), min(counts.values(), default=0), frame, inside))
    best = max(ranked, key=lambda row: (row[0], row[1], row[2], -int(row[3])))
    return best[3], best[4], [{'frame': r[3], 'in_range_objects': r[1],
                             'counts': dict(Counter(obj.name for obj in r[4]))}
                            for r in sorted(ranked, key=lambda r: r[:3], reverse=True)]


def in_range(center, geometry):
    return all(geometry[a + '_min'] < center[i] < geometry[a + '_max']
               for i, a in enumerate('xy'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--kitti-root', type=Path, default=ROOT / 'Kitti_Raw')
    parser.add_argument('--split', type=Path, default=ROOT / 'splits/kitti/val.txt')
    parser.add_argument('--frame')
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent / 'Demo/outputs/attention_placements')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=('intervention','checkpoint'), default='intervention',
                        help='checkpoint captures saved topology; intervention inserts modules without training')
    args = parser.parse_args(argv)
    import torch
    from tools.kitti_training_pipeline.common import (configure_detector_imports, build_model,
        normalize_state_dict, validate_evaluation_checkpoint, bev_encoding_spec, detection_spec)
    configure_detector_imports(ROOT / 'detector')
    from core.bev_encoding import resolve_bev_encoding
    from core.datasets.utils_1.bev_backend import encode_hist14_chw
    from core.models.backbones.focal_context import FocalContext
    from tools.visualization.export_kitti_overview import dataset_paths, read_points, read_objects, sha256
    from tools.visualization.predict_kitti_overview import classification_heatmaps
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = copy.deepcopy(checkpoint['config'])
    validate_evaluation_checkpoint(checkpoint, config)
    model = build_model(config).eval()
    model.load_state_dict(normalize_state_dict(checkpoint), strict=True)
    if args.mode == 'intervention' and (config['model'].get('c4_context') != 'focal' or
            config['model'].get('local_attention') != 'simam' or not model.backbone.feature_options.has_legacy_placement):
        raise ValueError('Use a trained focal C4 + SimAM C3 checkpoint')
    geometry = config['data']['kitti']['geometry']
    schema = bev_encoding_spec(config)
    if schema.name != 'hist14':
        raise ValueError('Expected Hist14')
    # NumPy encoding has the same semantic contract and needs no cache writes.
    schema = resolve_bev_encoding({**dict(config['data']['bev_encoding']), 'backend': 'numpy'}, geometry)
    spec = detection_spec(config)
    if spec.cls_encoding != 'gaussian':
        raise ValueError('This heatmap viewer requires Gaussian classification heads')
    paths = dataset_paths(args.kitti_root)
    if args.frame:
        frame = args.frame
        objects = [obj for obj in read_objects(paths.labels / (frame + '.txt'),
                   paths.calib / (frame + '.txt')) if in_range(obj.corners.mean(axis=0), geometry)]
        ranking = []
    else:
        frame, objects, ranking = choose_frame(paths, geometry, args.split)
    points = read_points(paths.velodyne / (frame + '.bin'))
    checkpoint_hash = sha256(args.checkpoint)
    output = args.output_dir / frame
    if args.mode == 'checkpoint':
        output = output / ('trained_'+checkpoint_hash[:12])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'tensors').mkdir(exist_ok=True)
    print(f'Frame {frame}: {len(points):,} points, {dict(Counter(o.name for o in objects))}', flush=True)
    encoded = encode_hist14_chw(points, schema)
    tensor = torch.from_numpy(encoded).unsqueeze(0)
    if args.mode == 'checkpoint':
        metadata = export_checkpoint_view(model,tensor,output=output,geometry=geometry,objects=objects,
            spec=spec,config=config,frame=frame,points=points,checkpoint_path=args.checkpoint,
            checkpoint_hash=checkpoint_hash)
        assert sha256(args.checkpoint) == checkpoint_hash
        print(f'Trained checkpoint gallery: {output / "index.html"}',flush=True)
        return metadata
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    original_features = []
    handle = model.backbone.register_forward_hook(
        lambda module, inputs, value: original_features.append(value.detach().cpu().numpy().copy()))
    try:
        with torch.inference_mode():
            original_prediction = model(tensor)
    finally:
        handle.remove()
    original_logits, original_probabilities = classification_heatmaps(original_prediction, groups=spec.groups,
                                                      num_classes=config['data']['num_classes'])
    focals = {'C4': copy.deepcopy(model.backbone.c4_context).eval()}
    for stage, channels in [('C3', 48), ('C5', 128)]:
        with torch.random.fork_rng():
            torch.manual_seed(args.seed + channels)
            focals[stage] = FocalContext(channels=channels,
                bottleneck=config['model']['c4_context_bottleneck'],
                dilations=tuple(config['model']['c4_context_dilations']),
                layer_scale_init=config['model']['c4_context_layer_scale_init']).eval()
    simam = copy.deepcopy(model.backbone.c3_light_attention).eval()
    variants = [(f, s, 'FS') for f in STAGES for s in STAGES]
    variants += [(stage, stage, 'SF') for stage in STAGES[1:]]
    results, rows = {}, []
    class_order = sorted(config['data']['kitti']['objects'], key=config['data']['kitti']['objects'].get)
    for index, (focal, sim, order) in enumerate(variants):
        key = f'F_{focal}_S_{sim}_{order}'
        prediction, captured = placement_forward(model, tensor, focals, simam, focal, sim, order)
        logits, probabilities = classification_heatmaps(prediction, groups=spec.groups,
                                                        num_classes=len(class_order))
        maps = {stage: energy(captured[stage]) for stage in [*MODULES, 'final']}
        before_maps = {stage + '_before': energy(captured[stage + '_before']) for stage in MODULES}
        gates = {name: value[0] if 'focal' in name else value[0].mean(axis=0)
                 for name, value in captured.items() if '_gate' in name}
        np.savez_compressed(output / 'tensors' / (key + '.npz'), final_features=captured['final'][0],
            probabilities=probabilities, logits=logits, **maps, **before_maps, **gates)
        results[key] = dict(focal=focal, sim=sim, order=order, maps=maps,
            probabilities=probabilities, before=before_maps, gates=gates,
            final=captured['final'][0])
        if focal == 'C4' and sim == 'C3':
            np.testing.assert_allclose(captured['final'], original_features[0], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(logits, original_logits, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(probabilities, original_probabilities, rtol=1e-5, atol=1e-6)
        baseline = results['F_off_S_off_FS']['final']
        delta = captured['final'][0] - baseline
        rows.append(dict(variant=key, focal=focal, simam=sim, order=order,
            focal_weights='new_untrained' if focal in ('C3', 'C5') else 'trained' if focal == 'C4' else 'off',
            feature_delta_rms=float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
            feature_relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(baseline), 1e-12)),
            max_probability_delta=float(np.max(np.abs(probabilities - results['F_off_S_off_FS']['probabilities'])))))
        print(f'[{index+1}/{len(variants)}] {key}: feature delta RMS {rows[-1]["feature_delta_rms"]:.5g}', flush=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)
    assert sha256(args.checkpoint) == checkpoint_hash
    np.savez_compressed(output / 'tensors/input.npz', points_xyzi=points, hist14=encoded)
    with (output / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    extent = [geometry['x_min'], geometry['x_max'], geometry['y_min'], geometry['y_max']]
    centers = np.array([o.corners.mean(axis=0) for o in objects])
    # Crop only the display; the network always receives the complete scan/BEV.
    roi = [max(0., float(centers[:, 0].min())-5), min(70.4, float(centers[:, 0].max())+5),
           max(-40., float(centers[:, 1].min())-5), min(40., float(centers[:, 1].max())+5)]
    def overlay(ax, labels=False):
        for obj in objects:
            polygon = obj.corners[:4, :2]
            ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor=COLORS[obj.name], lw=.65, alpha=.85))
            if labels:
                c = polygon.mean(axis=0)
                ax.text(*c, obj.name[:3], color=COLORS[obj.name], fontsize=6)
        ax.set_xlim(roi[:2]); ax.set_ylim(roi[2:]); ax.set_aspect('equal')
    def save(fig, name):
        fig.savefig(output / (name + '.png'), dpi=170, bbox_inches='tight')
        plt.close(fig)
    def matrix(name, title, getter, vmin, vmax, cmap):
        fig, axes = plt.subplots(4, 4, figsize=(17, 15), constrained_layout=True)
        for i, f in enumerate(STAGES):
            for j, s in enumerate(STAGES):
                result = results[f'F_{f}_S_{s}_FS']
                ax = axes[i, j]
                image = ax.imshow(getter(result), origin='lower', extent=extent,
                                  vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
                overlay(ax)
                tag = ' [NEW F]' if f in ('C3', 'C5') else ''
                tag += ' [TRAINED ORIGINAL]' if (f, s) == ('C4', 'C3') else ''
                ax.set_title(f'Focal {f} / SimAM {s}{tag}', fontsize=10)
                ax.tick_params(labelsize=7)
        fig.colorbar(image, ax=axes, shrink=.65, label=title)
        fig.suptitle(f'{title} | KITTI {frame}\nCheckpoint interventions; NEW F = untrained Focal; shared stage: Focal then SimAM', fontsize=14)
        save(fig, name)

    valid = np.isfinite(points).all(axis=1)
    valid &= (points[:, 0] >= extent[0]) & (points[:, 0] <= extent[1])
    valid &= (points[:, 1] >= extent[2]) & (points[:, 1] <= extent[3])
    cloud = points[valid]
    zmin, zmax = np.percentile(cloud[:, 2], [2, 98])
    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    ax = fig.add_subplot(121)
    ax.set_facecolor('#080b12')
    ax.scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], s=.25, cmap='turbo', vmin=zmin, vmax=zmax, rasterized=True)
    overlay(ax, labels=True)
    ax.set_title(f'Raw BEV + ground truth | {frame} | {len(objects)} objects')
    ax.set_xlabel('LiDAR X (m)'); ax.set_ylabel('LiDAR Y (m)')
    ax3 = fig.add_subplot(122, projection='3d')
    display = cloud[(cloud[:, 0] >= roi[0]) & (cloud[:, 0] <= roi[1]) &
                    (cloud[:, 1] >= roi[2]) & (cloud[:, 1] <= roi[3])]
    ax3.scatter(*display[:, :3].T, c=display[:, 2], s=.15, cmap='turbo', vmin=zmin, vmax=zmax, rasterized=True)
    from tools.visualization.export_kitti_overview import EDGES
    for obj in objects:
        for a, b in EDGES:
            ax3.plot(*obj.corners[[a, b]].T, color=COLORS[obj.name], lw=.7)
    ax3.set_zlim(-3, 3); ax3.set_box_aspect((roi[1]-roi[0], roi[3]-roi[2], 7))
    ax3.view_init(32, -120); ax3.set_title('Measured pointcloud + calibrated GT boxes')
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)'); ax3.set_zlabel('Z (m)')
    save(fig, '00_scene')
    common_max = float(np.percentile(np.concatenate([r['maps']['final'].ravel() for r in results.values()]), 99.5))
    matrix('01_feature_rms_matrix', 'Final feature RMS across 32 channels (shared scale)',
           lambda r: r['maps']['final'], 0, common_max, 'magma')
    baseline = results['F_off_S_off_FS']
    for result in results.values():
        delta = result['final'] - baseline['final']
        result['difference'] = np.sqrt(np.mean(delta.astype(np.float64)**2, axis=0))
    difference_max = float(np.percentile(np.concatenate([r['difference'].ravel() for r in results.values()]), 99.5))
    matrix('02_feature_difference_matrix', 'RMS feature change vs both modules off (shared scale)',
           lambda r: r['difference'], 0, max(difference_max, 1e-12), 'inferno')
    for i, name in enumerate(class_order):
        matrix('03_heatmap_' + name.lower(), name + ' Gaussian classification probability',
               lambda r, i=i: r['probabilities'][i], 0, 1, 'magma')
        delta_max = max(float(np.max(np.abs(r['probabilities'][i] - baseline['probabilities'][i])))
                        for r in results.values())
        matrix('04_heatmap_delta_' + name.lower(), name + ' probability change vs both modules off',
               lambda r, i=i: r['probabilities'][i] - baseline['probabilities'][i],
               -max(delta_max, 1e-12), max(delta_max, 1e-12), 'RdBu_r')
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for i, stage in enumerate(STAGES[1:]):
        fs, sf = results[f'F_{stage}_S_{stage}_FS'], results[f'F_{stage}_S_{stage}_SF']
        diff = np.sqrt(np.mean((fs['final']-sf['final']).astype(np.float64)**2, axis=0))
        for j, (value, vmax, label) in enumerate([(fs['maps']['final'], common_max, 'Focal then SimAM'),
            (sf['maps']['final'], common_max, 'SimAM then Focal'), (diff, max(float(diff.max()), 1e-12), 'Order difference RMS')]):
            im = axes[i,j].imshow(value, origin='lower', extent=extent, vmin=0, vmax=vmax, cmap='magma')
            overlay(axes[i,j]); axes[i,j].set_title(f'{stage}: {label}', fontsize=10)
            fig.colorbar(im, ax=axes[i,j], shrink=.7)
    fig.suptitle('Same-stage order comparison | C3/C5 Focal is untrained', fontsize=14)
    save(fig, '05_shared_stage_order')
    # All stage outputs, before and after insertion, at their native resolutions.
    for stage in STAGES[1:]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
        local_results = [baseline, results[f'F_off_S_{stage}_FS'],
                         results[f'F_{stage}_S_off_FS'], results[f'F_{stage}_S_{stage}_FS']]
        local_max = float(np.percentile(np.concatenate([r['maps'][stage].ravel() for r in local_results]), 99.5))
        for j, (label, result) in enumerate(zip(['Both off', 'SimAM only', 'Focal only', 'Focal then SimAM'], local_results)):
            before = result['before'][stage + '_before']
            after = result['maps'][stage]
            axes[0,j].imshow(after, origin='lower', extent=extent, vmin=0, vmax=local_max, cmap='magma')
            change = after - before
            bound = max(float(np.max(np.abs(change))), 1e-12)
            im = axes[1,j].imshow(change, origin='lower', extent=extent, vmin=-bound, vmax=bound, cmap='RdBu_r')
            fig.colorbar(im, ax=axes[1,j], shrink=.7)
            axes[0,j].set_title(label); axes[1,j].set_title('After RMS - before RMS (individual delta scale)', fontsize=9)
            overlay(axes[0,j]); overlay(axes[1,j])
        fig.suptitle(f'{stage} native feature maps | Focal {"trained" if stage == "C4" else "NEW / UNTRAINED"}', fontsize=14)
        save(fig, '06_stage_' + stage.lower())
    fig, axes = plt.subplots(3, 5, figsize=(17, 13), constrained_layout=True)
    for row, stage in enumerate(STAGES[1:]):
        sim_result = results[f'F_off_S_{stage}_FS']
        focal_result = results[f'F_{stage}_S_off_FS']
        maps = [sim_result['gates'][stage + '_simam_gate'],
                *focal_result['gates'][stage + '_focal_gates']]
        labels = ['SimAM mean gate across channels', 'Focal level 1', 'Focal level 2',
                  'Focal level 3', 'Focal global level']
        for column, (value, label) in enumerate(zip(maps, labels)):
            im = axes[row,column].imshow(value, origin='lower', extent=extent, vmin=0, vmax=1, cmap='viridis')
            overlay(axes[row,column])
            axes[row,column].set_title(f'{stage}: {label}', fontsize=9)
    fig.colorbar(im, ax=axes, shrink=.7, label='Gate weight (0 to 1)')
    fig.suptitle('Attention gates with only that module enabled | C3/C5 Focal untrained', fontsize=14)
    save(fig, '07_attention_gates')
    selected = ['F_off_S_off_FS', 'F_off_S_C3_FS', 'F_C4_S_off_FS',
                'F_C4_S_C3_FS', 'F_C4_S_C4_FS', 'F_C3_S_C4_FS']
    channels = np.argsort(np.var(baseline['final'], axis=(1,2)))[-4:][::-1]
    fig, axes = plt.subplots(4, 6, figsize=(19, 15), constrained_layout=True)
    for row, channel in enumerate(channels):
        bound = max(float(np.percentile(np.abs(np.concatenate(
            [results[key]['final'][channel].ravel() for key in selected])), 99.5)), 1e-12)
        for column, key in enumerate(selected):
            im = axes[row,column].imshow(results[key]['final'][channel], origin='lower', extent=extent,
                                         vmin=-bound, vmax=bound, cmap='RdBu_r')
            overlay(axes[row,column])
            axes[row,column].set_title(f'Channel {channel} | {key[2:-3]}', fontsize=9)
        fig.colorbar(im, ax=axes[row,:], shrink=.8, label=f'Channel {channel} activation')
    fig.suptitle('Four highest-variance final channels in both-off baseline | shared scale within each row', fontsize=14)
    save(fig, '08_individual_feature_channels')
    pedestrian = class_order.index('Pedestrian') if 'Pedestrian' in class_order else 0
    ped_centers = np.array([o.center for o in objects if o.name == class_order[pedestrian]])
    if len(ped_centers):
        zoom = [max(extent[0], float(ped_centers[:,0].min())-3),
                min(extent[1], float(ped_centers[:,0].max())+3),
                max(extent[2], float(ped_centers[:,1].min())-3),
                min(extent[3], float(ped_centers[:,1].max())+3)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for ax, key in zip(axes.ravel(), selected):
            im = ax.imshow(results[key]['probabilities'][pedestrian], origin='lower', extent=extent,
                           vmin=0, vmax=1, cmap='magma', interpolation='nearest')
            overlay(ax); ax.set_xlim(zoom[:2]); ax.set_ylim(zoom[2:])
            ax.set_title(key, fontsize=10)
        fig.colorbar(im, ax=axes, shrink=.75, label='Pedestrian probability')
        fig.suptitle('Pedestrian region zoom | identical probability scale; C3 Focal untrained', fontsize=14)
        save(fig, '09_pedestrian_zoom')
    # Fixed-scale panels permit switching two placements in the local HTML viewer.
    interactive = output / 'interactive'
    interactive.mkdir(exist_ok=True)
    for key, result in results.items():
        fig, axes = plt.subplots(1, 3, figsize=(11, 7), constrained_layout=True)
        data = [(result['maps']['final'], 0, common_max, 'magma', 'Final feature RMS'),
                (result['difference'], 0, max(difference_max, 1e-12), 'inferno', 'Feature change vs both off'),
                (result['probabilities'][pedestrian], 0, 1, 'magma', 'Pedestrian probability')]
        for ax, (value, lower, upper, cmap, label) in zip(axes, data):
            im = ax.imshow(value, origin='lower', extent=extent, vmin=lower, vmax=upper,
                           cmap=cmap, interpolation='nearest')
            overlay(ax); ax.set_title(label, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=.7)
        fig.suptitle(key + (' | NEW / UNTRAINED Focal' if result['focal'] in ('C3','C5') else ''), fontsize=12)
        fig.savefig(interactive / (key + '.png'), dpi=140, bbox_inches='tight')
        plt.close(fig)
    metadata = dict(frame=frame, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=checkpoint.get('epoch'), source_cloud=str(paths.velodyne / (frame + '.bin')),
        cloud_sha256=sha256(paths.velodyne / (frame + '.bin')), point_count=len(points),
        objects_in_range=dict(Counter(o.name for o in objects)), selection_candidates=ranking,
        selected_from_validation=frame in split_ids(args.split) if args.split.is_file() else None,
        interventions=rows, seed=args.seed, class_order=class_order, geometry=geometry, display_roi=roi,
        focal_new_weights=['C3', 'C5'], focal_trained_weights=['C4'], simam_beta=float(simam.beta.detach()),
        displayed_feature_channels=[int(c) for c in channels],
        focal_trained_gamma_summary={'min': float(focals['C4'].gamma.detach().min()),
                                    'max': float(focals['C4'].gamma.detach().max())},
        verification={'strict_checkpoint_load': True, 'native_probability_replay': True,
                      'native_features_and_logits_replay': True,
                      'original_state_unchanged': True, 'checkpoint_file_unchanged': True,
                      'all_captured_tensors_finite': True},
        limitations=['Single-frame checkpoint interventions, no retraining, no AP comparisons.',
            'C3/C5 Focal modules fully newly initialized with seed+channels; reused identically across combinations.',
            'Both-off baseline removes modules from a trained P16, not an independently trained baseline.',
            'Raw classification sigmoid probabilities, before IQA calibration and NMS.',
            'Main figure scales shared; per-panel direct stage delta scales explicitly labeled.',
            'Ground truth selects the frame and draws overlays; it is never input to the network.'])
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'config.original.json').write_text(json.dumps(config, indent=2) + '\n')
    descriptions = {'00_scene': 'Pointcloud thật và GT: Car đỏ, Pedestrian xanh, Cyclist vàng.',
        '01_feature_rms_matrix': 'RMS của 32 kênh đầu ra FPN; cùng thang màu cho cả 16 tổ hợp.',
        '02_feature_difference_matrix': 'Chênh lệch RMS trên tensor 32 kênh so với bỏ cả hai module.',
        '05_shared_stage_order': 'Cùng tầng: so sánh Focal → SimAM và SimAM → Focal.'}
    figures = []
    for path in sorted(output.glob('*.png')):
        label = descriptions.get(path.stem, path.stem.replace('_', ' '))
        figures.append(f'<article><h2>{html.escape(label)}</h2><a href="{path.name}"><img src="{path.name}" loading="lazy"></a></article>')
    document = '<!doctype html><meta charset="utf-8"><title>Attention placement comparison</title>'
    document += '<style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:16px/1.6 sans-serif;background:#f4f5f7}article{background:white;padding:20px;margin:24px 0}img{width:100%} .note{background:#fff0cc;padding:18px}</style>'
    document += f'<h1>SimAM / Focal tại các tầng · KITTI {frame}</h1><p>{len(objects)} GT objects · {len(points):,} points · 19 configurations</p>'
    document += '<div class="note">Đây là can thiệp trên cùng checkpoint P16, không huấn luyện lại. Focal C4 và SimAM dùng trọng số đã train; Focal C3/C5 mới khởi tạo (NEW F). Both off là bỏ module từ P16. Không dùng hình để kết luận vị trí nào có AP tốt hơn. Các đường viền là GT, không phải detection. Heatmap là xác suất classification trước IQA/NMS.</div>'
    options = ''.join(f'<option value="{key}">{key}' + (' · NEW F' if result['focal'] in ('C3','C5') else '') + '</option>'
                      for key, result in results.items())
    document += '<h2>Chọn hai cấu hình để so sánh</h2><p>FS = Focal rồi SimAM; SF = SimAM rồi Focal. Thang màu giữ nguyên giữa các cấu hình.</p>'
    document += '<div style="display:flex;gap:12px;flex-wrap:wrap">'
    for letter, key in [('a', 'F_off_S_off_FS'), ('b', 'F_C4_S_C3_FS')]:
        document += f'<div style="flex:1;min-width:400px"><select id="select_{letter}">{options}</select><a id="link_{letter}" href="interactive/{key}.png"><img id="image_{letter}" src="interactive/{key}.png"></a></div>'
    document += '</div><script>for(const [id,key] of [["a","F_off_S_off_FS"],["b","F_C4_S_C3_FS"]]){const select=document.getElementById("select_"+id);select.value=key;select.addEventListener("change",()=>{const url="interactive/"+select.value+".png";document.getElementById("image_"+id).src=url;document.getElementById("link_"+id).href=url;});}</script>'
    document += '<p><a href="metrics.csv">Metrics CSV</a> · <a href="metadata.json">Metadata và kiểm chứng</a></p>' + ''.join(figures)
    (output / 'index.html').write_text(document)
    (output / 'README.md').write_text(f'# Attention placement interventions: {frame}\n\n'
        + '\n'.join('- ' + item for item in metadata['limitations'])
        + '\n\nOpen index.html for all figures. Metrics measure activation changes, not accuracy.\n'
        + '\nNumeric tensors include final 32-channel features, class logits/probabilities, stage RMS before/after attention, and gates.\n')
    print(f'Gallery: {output / "index.html"}', flush=True)
    return metadata


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Export real KITTI points, training augmentation and actual checkpoint tensors.

hist14 directly groups measured points into XY pillars and four height bands.
The occupied-cell picture illustrates those groups; it is not a separate sparse
3D network. Augmented inference illustrates a training data path, not KITTI AP.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "Code"
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/kitti_training_pipeline')]
from tools.visualization.export_kitti_overview import (
    CLASS_COLORS, EDGES, camera_basis, canvas_limits, dataset_paths, projected,
    read_points, render_panel, roi_vertices, setup_matplotlib, sha256,
)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def hist14_cells(points, schema):
    """Use the encoder's float64 arithmetic, epsilon and four-band grouping."""
    if schema.name != 'hist14':
        raise ValueError('Occupied height-band cells require hist14')
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError('Expected XYZI points')
    geometry = dict(schema.geometry)
    valid = np.isfinite(points[:, :4]).all(axis=1)
    for column, axis in enumerate('xyz'):
        valid &= points[:, column] > geometry[f'{axis}_min'] + .001
        valid &= points[:, column] < geometry[f'{axis}_max'] - .001
    selected = points[valid]
    indices = np.empty((len(selected), 3), dtype=np.int64)
    for column, axis in enumerate('xy'):
        indices[:, column] = np.floor((selected[:, column] - geometry[f'{axis}_min'])
                                     / geometry[f'{axis}_res']).astype(np.int64)
    height = (selected[:, 2] - geometry['z_min']) / (geometry['z_max'] - geometry['z_min'])
    indices[:, 2] = np.minimum(np.floor(height * 4).astype(np.int64), 3)
    unique, counts = np.unique(indices, axis=0, return_counts=True)
    return unique, counts, valid


def height_count_channels(indices, counts, schema):
    x_size, y_size, _ = schema.grid_shape
    channels = np.zeros((4, y_size, x_size), np.float32)
    if len(indices):
        channels[indices[:, 2], indices[:, 1], indices[:, 0]] = np.minimum(
            1., np.log1p(counts) / np.log1p(schema.density_norm))
    return channels


def backbone_capture_modules(model):
    """Capture raw stages and refined taps; old backbones retain legacy captures."""
    available = dict(model.named_modules())
    taps = tuple(f'backbone.attention_{stage}_output' for stage in ('c3','c4','c5'))
    optional = taps if all(name in available for name in taps) else ('backbone.stage3',)
    names = ('backbone.stem','backbone.stage2','backbone.stage3','backbone.stage4',
             *optional,'backbone.refine_u4','backbone.proj_u3','backbone')
    return [name for name in names if name in available]


def backbone_placement(model):
    features = getattr(model.backbone, 'feature_options', None)
    return features.effective_placement() if features is not None else None


def capture_forward(model, tensor, *, modules):
    """Capture real BCHW outputs in one eval forward, with unconditional cleanup."""
    import torch
    if model.training:
        raise ValueError('Capture requires model.eval() to preserve BatchNorm state')
    available = dict(model.named_modules())
    missing = set(modules) - set(available)
    if missing:
        raise ValueError(f'Unknown capture modules: {sorted(missing)}')
    captured, handles = {}, []

    def hook(name):
        def capture(module, inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4 or output.shape[0] != 1:
                raise ValueError(f'Expected one BCHW tensor at {name}')
            if not torch.isfinite(output).all():
                raise ValueError(f'Non-finite feature tensor at {name}')
            captured[name] = output.detach().float().cpu().numpy().copy()
        return capture

    try:
        for name in modules:
            handles.append(available[name].register_forward_hook(hook(name)))
        with torch.inference_mode():
            prediction = model(tensor)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(modules):
        raise ValueError('Some selected modules did not execute')
    return prediction, captured


def feature_energy(features):
    features = np.asarray(features)
    if (features.ndim != 4 or features.shape[0] != 1 or not np.isfinite(features).all()
            or features.shape[1] == 0):
        raise ValueError('Expected finite BCHW activations for one frame')
    return np.sqrt(np.mean(np.square(features[0], dtype=np.float64), axis=0)).astype(np.float32)


def flattened_heads(prediction, prefix=''):
    import torch
    result = {}
    for name, value in prediction.items():
        key = f'{prefix}{name}'
        if isinstance(value, dict):
            result.update(flattened_heads(value, key + '.'))
        elif isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError(f'Non-finite head tensor: {key}')
            result[key] = value.detach().float().cpu().numpy().copy()
    return result


def detector_boxes(label_path, calibration_path, class_ids):
    from tools.kitti_training_pipeline.prepare_kitti import convert_labels
    lines, _ = convert_labels(label_path, calibration_path)
    rows = [[class_ids[fields[0]], *map(float, fields[1:])]
            for line in lines if (fields := line.split())[0] in class_ids]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 8), lines


def prepare_donors(paths, train_ids, val_ids, scene, destination, class_ids, *, donor_count=128):
    """Prepare a documented train-only subset; link scans without copying them."""
    from tools.kitti_training_pipeline.build_gt_database import build_database
    if set(train_ids) & set(val_ids):
        raise ValueError('Train/validation donor manifests overlap')
    if scene not in set(train_ids):
        raise ValueError('Choose a train frame to illustrate the training augmentation path')
    candidates = sorted(set(train_ids) - {scene})
    counts = {}
    for identifier in candidates:
        label = paths.labels / f'{identifier}.txt'
        if all(path.is_file() for path in (label, paths.velodyne / f'{identifier}.bin',
                                          paths.calib / f'{identifier}.txt')):
            counts[identifier] = Counter(line.split()[0] for line in label.read_text().splitlines() if line.split())
    # Include diverse ordinary frames and frames containing the rare cyclist class.
    ordinary = sorted(counts)[:donor_count // 2]
    rare = sorted(counts, key=lambda identifier: (-counts[identifier]['Cyclist'], identifier))
    donors = list(dict.fromkeys(ordinary + rare))[:donor_count]
    if len(donors) < donor_count:
        raise ValueError(f'Only {len(donors)} train donors available, requested {donor_count}')
    destination.mkdir(parents=True, exist_ok=True)
    for folder in ('pointcloud', 'label'):
        (destination / folder).mkdir(exist_ok=True)
    sources = {}
    for identifier in donors:
        scan = paths.velodyne / f'{identifier}.bin'
        target = destination / 'pointcloud' / scan.name
        if not target.exists():
            target.symlink_to(scan.resolve())
        if target.resolve() != scan.resolve():
            raise ValueError(f'Donor scan conflicts with KITTI source: {target}')
        label, calibration = paths.labels / f'{identifier}.txt', paths.calib / f'{identifier}.txt'
        _, lines = detector_boxes(label, calibration, class_ids)
        (destination / 'label' / label.name).write_text('\n'.join(lines) + '\n')
        sources[identifier] = {'bin_sha256': sha256(scan), 'label_sha256': sha256(label),
                               'calibration_sha256': sha256(calibration)}
    manifest = destination / 'donor_train.txt'
    manifest.write_text(''.join(f'{identifier};kitti\n' for identifier in donors))
    validation = destination / 'validation.txt'
    validation.write_text(''.join(f'{identifier};kitti\n' for identifier in val_ids))
    info = build_database(destination, manifest, destination / 'gt_database', class_names=class_ids,
                          val_manifest=validation, min_points=5)
    data = json.loads(info.read_text())
    return {'scope': 'real KITTI train-only donor subset for visualization; not the original full training DB',
            'source_frame_ids': donors, 'scene_excluded': True, 'validation_excluded': True,
            'counts': data['counts'], 'source_hashes': sources, 'manifest_sha256': sha256(manifest),
            'database_info_sha256': sha256(info), 'path': str(info.resolve())}


def augment_scene(points, boxes, config, donor_root, allowed_ids, *, seed=2, force_sampling=False):
    from core.datasets.augmentor.data_augmentor import DataAugmentor
    settings = copy.deepcopy(config['augmentation'])
    if settings.get('mode') != 'openpcdet':
        raise ValueError('This training-path exporter requires the openpcdet augmentation queue')
    disabled = set(settings.get('DISABLE_AUG_LIST', []))
    operations = [op for op in settings['AUG_CONFIG_LIST'] if op['NAME'] not in disabled]
    if not operations or operations[0]['NAME'] != 'hybrid_gt_sampling':
        raise ValueError('Select a checkpoint with hybrid_gt_sampling as its first augmentation')
    rng = np.random.default_rng(seed)
    kwargs = dict(root_path=donor_root, class_names=config['data']['kitti']['objects'], rng=rng,
                  geometry=config['data']['kitti']['geometry'], allowed_frame_ids=allowed_ids)
    sampler = DataAugmentor(augmentor_configs={'AUG_CONFIG_LIST': operations[:1]}, **kwargs).queue[0][2]
    probability = operations[0].get('PROBABILITY', 1.)
    skipped = not force_sampling and (probability == 0 or (probability < 1 and rng.random() >= probability))
    if skipped:
        sampled, sampled_boxes = points.copy(), boxes.copy()
        metadata = {'num_inserted': 0, 'point_is_sampled': np.zeros(len(points), bool),
                    'box_is_sampled': np.zeros(len(boxes), bool)}
    else:
        sampled, sampled_boxes, metadata = sampler(points.copy(), boxes.copy(), rng, return_metadata=True)
    augmented, augmented_boxes = DataAugmentor(
        augmentor_configs={'AUG_CONFIG_LIST': operations[1:]}, **kwargs)(sampled.copy(), sampled_boxes.copy())
    return sampled, augmented, augmented_boxes, metadata, {
        'seed': seed, 'forced_sampling_preview': force_sampling, 'probability_skipped': bool(skipped),
        'original_sampling_probability': probability, 'operations': operations,
        'label_usage': 'ground-truth boxes only for augmentation; model forward receives encoded points only'}


def save_plot(fig, output, *, dpi=160):
    for suffix in ('png', 'svg'):
        fig.savefig(output.with_suffix('.' + suffix), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
    setup_matplotlib().close(fig)


def map_axis(ax, values, geometry, *, title='', cmap='viridis', vmin=0, vmax=1):
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite spatial map')
    ax.set_facecolor('#070b12')
    image = ax.imshow(values.T, origin='lower', interpolation='nearest', cmap=cmap,
                      vmin=vmin, vmax=vmax, extent=(geometry['y_min'], geometry['y_max'],
                                                  geometry['x_min'], geometry['x_max']))
    ax.set_xlim(geometry['y_max'], geometry['y_min'])
    ax.set_xlabel('Y lateral (m)')
    ax.set_ylabel('X forward (m)')
    ax.set_title(title, fontsize=10)
    return image


def map_gallery(maps, names, geometry, output, *, columns=4, feature=False):
    plt = setup_matplotlib()
    rows = (len(maps) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.1, rows * 4.2), squeeze=False)
    ranges = []
    for ax, value, name in zip(axes.flat, maps, names):
        offset = name.startswith('point_offset')
        signed_feature = feature and float(value.min()) < 0
        signed = offset or signed_feature
        high = max(float(np.quantile(np.abs(value) if signed_feature else value, .99)), 1e-8) if feature else 1.
        low = -high if signed_feature else 0.
        if offset:
            low, high = -.5, .5
        image = map_axis(ax, value, geometry, title=name, cmap='coolwarm' if signed else 'viridis',
                         vmin=low, vmax=high)
        fig.colorbar(image, ax=ax, shrink=.7, pad=.02)
        ranges.append({'name': name, 'display_min': low, 'display_max': high,
                       'actual_min': float(value.min()), 'actual_max': float(value.max())})
    for ax in list(axes.flat)[len(maps):]:
        ax.axis('off')
    fig.tight_layout()
    save_plot(fig, output)
    return ranges


def render_cells(indices, counts, geometry, roi, basis, limits, output, *, max_cells=16000):
    from matplotlib.collections import PolyCollection
    plt = setup_matplotlib()
    step = np.array([geometry['x_res'], geometry['y_res'],
                     (geometry['z_max'] - geometry['z_min']) / 4])
    centers = np.array([geometry['x_min'], geometry['y_min'], geometry['z_min']]) + (indices + .5) * step
    visible = np.ones(len(centers), bool)
    for axis in range(3):
        visible &= centers[:, axis] >= roi[axis * 2]
        visible &= centers[:, axis] <= roi[axis * 2 + 1]
    chosen = np.flatnonzero(visible)
    if len(chosen) > max_cells:
        chosen = chosen[np.linspace(0, len(chosen) - 1, max_cells, dtype=int)]
    corners = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                        [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
    xyz = centers[chosen, None, :] + corners[None, :, :] * step
    uvz = projected(xyz.reshape(-1, 3), basis).reshape(-1, 8, 3)
    faces = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    vertices = np.stack([uvz[:, face, :2] for face in faces], axis=1).reshape(-1, 4, 2)
    depth = np.stack([uvz[:, face, 2].mean(axis=1) for face in faces], axis=1).ravel()
    colors = np.repeat(plt.get_cmap('viridis')(np.clip(np.log1p(counts[chosen]) / np.log1p(32), 0, 1)), 6, axis=0)
    order = np.argsort(depth, kind='stable')
    fig = plt.figure(figsize=(9, 6), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
    ax.add_collection(PolyCollection(vertices[order], facecolors=colors[order], edgecolors='none',
                                     rasterized=True, alpha=.9))
    ax.set(xlim=limits[:2], ylim=limits[2:])
    ax.set_aspect('equal')
    ax.axis('off')
    save_plot(fig, output)
    return {'occupied_cells_full_grid': len(indices), 'occupied_cells_in_display_roi': int(visible.sum()),
            'rendered_cells': len(chosen), 'render_sampling': 'evenly spaced occupied-cell indices',
            'cell_size_xyz': step.tolist(), 'height_bands': 4,
            'purpose': 'diagnostic view of hist14 grouping, not an additional 3D voxel network'}


def render_bev(points, objects, geometry, output):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor('black')
    valid = ((points[:, 0] > geometry['x_min']) & (points[:, 0] < geometry['x_max'])
             & (points[:, 1] > geometry['y_min']) & (points[:, 1] < geometry['y_max']))
    ax.scatter(points[valid, 1], points[valid, 0], s=.15, c='#52d9e8', linewidths=0, rasterized=True)
    for obj in objects:
        footprint = np.r_[obj.footprint, obj.footprint[:1]]
        color = CLASS_COLORS[obj.name]
        ax.plot(footprint[:, 1], footprint[:, 0], color=color, lw=1)
        ax.text(obj.row[3], obj.row[2], f'{obj.name} {obj.score:.2f}', color='white', fontsize=6,
                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': .6, 'pad': .8})
    ax.set(xlim=(geometry['y_max'], geometry['y_min']), ylim=(geometry['x_min'], geometry['x_max']),
           xlabel='Y lateral (m)', ylabel='X forward (m)', title='Checkpoint detections on augmented input')
    ax.set_aspect('equal')
    save_plot(fig, output)


def overview(output, entries, frame, seed):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(3, 3, figsize=(21, 17))
    for ax, (filename, title, subtitle) in zip(axes.flat, entries):
        ax.imshow(plt.imread(output / filename))
        ax.set_title(title + '\n' + subtitle, fontsize=11)
        ax.axis('off')
    fig.suptitle(f'KITTI {frame} | real data through the checkpoint | augmentation seed {seed}', fontsize=18)
    fig.text(.5, .013, 'Augmented training-path demo. 3D panels show a fixed ROI; full tensors are saved. Feature colors are activations.',
             ha='center', fontsize=11)
    fig.tight_layout(rect=(0, .035, 1, .96))
    save_plot(fig, output / 'pipeline_overview', dpi=180)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--kitti-root', type=Path, default=ROOT / 'Kitti_Raw')
    parser.add_argument('--frame', default='005663')
    parser.add_argument('--train-split', type=Path, default=ROOT / 'splits/kitti/train.txt')
    parser.add_argument('--val-split', type=Path, default=ROOT / 'splits/kitti/val.txt')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--donor-frames', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--force-sampling', action='store_true', help='Force GT sampling for a labeled preview only')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--score-threshold', type=float, default=.3)
    parser.add_argument('--nms-threshold', type=float, default=.1)
    parser.add_argument('--roi', nargs=6, type=float, default=[-2, 45, -24, 24, -2.8, 4])
    args = parser.parse_args(argv)
    if len(args.frame) != 6 or not args.frame.isascii() or not args.frame.isdigit():
        parser.error('Frame must be a six-digit KITTI ID')
    if min(args.donor_frames, args.threads) < 1 or args.seed < 0:
        parser.error('Positive donor/thread counts and nonnegative seed required')
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.nms_threshold <= 1:
        parser.error('Score/NMS thresholds must be in [0,1]')
    if not np.isfinite(args.roi).all() or any(args.roi[i] >= args.roi[i + 1] for i in (0, 2, 4)):
        parser.error('ROI must contain finite increasing XYZ bounds')
    started = time.perf_counter()
    import torch
    from tools.kitti_training_pipeline.common import (
        configure_detector_imports, build_model, validate_evaluation_checkpoint,
        normalize_state_dict, bev_encoding_spec, detection_spec,
    )
    configure_detector_imports(ROOT / 'detector')
    from core.datasets.utils_1.bev_backend import encode_hist14_chw
    from postprocess import filter_pred
    from tools.kitti_training_pipeline.prepare_kitti import read_ids
    from tools.visualization.predict_kitti_overview import classification_heatmaps, bev_detections, points_in_bev
    from tools.visualization.lift_kitti_bev_to_3d import local_ground, lift_object
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/lidar-kitti-overview-numba')
    torch.set_num_threads(args.threads)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / 'best.pt'
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = copy.deepcopy(checkpoint['config'])
    schema, spec = bev_encoding_spec(config), detection_spec(config)
    if schema.name != 'hist14' or spec.cls_encoding != 'gaussian':
        parser.error('This exporter requires hist14 input and Gaussian classification heads')
    validate_evaluation_checkpoint(checkpoint, config)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    model = build_model(config)
    model.load_state_dict(normalize_state_dict(checkpoint), strict=True)
    model.to(device).eval()
    paths = dataset_paths(args.kitti_root)
    frame = args.frame
    raw_path = paths.velodyne / f'{frame}.bin'
    points = read_points(raw_path)
    boxes, _ = detector_boxes(paths.labels / f'{frame}.txt', paths.calib / f'{frame}.txt',
                              config['data']['kitti']['objects'])
    train_ids, val_ids = read_ids(args.train_split), read_ids(args.val_split)
    output = (args.output_dir or ROOT.parent / f'Demo/outputs/pipeline/{frame}').expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f'Preparing {args.donor_frames} real train-only KITTI donor frames...', flush=True)
    donor_root = output / 'donor_database'
    donor_report = prepare_donors(paths, train_ids, val_ids, frame, donor_root,
                                  config['data']['kitti']['objects'], donor_count=args.donor_frames)
    sampled, augmented, augmented_boxes, sample_meta, aug_meta = augment_scene(
        points, boxes, config, donor_root, train_ids, seed=args.seed, force_sampling=args.force_sampling)
    print(f'Raw {len(points)} -> augmented {len(augmented)} points; inserted {sample_meta["num_inserted"]} objects', flush=True)
    tensor = torch.from_numpy(encode_hist14_chw(augmented, schema)).unsqueeze(0).to(device)
    modules = backbone_capture_modules(model)
    print('Capturing actual checkpoint input, backbone features and heads...', flush=True)
    prediction, features = capture_forward(model, tensor, modules=modules)
    rows = filter_pred(prediction, config['data']['kitti'], config['data']['out_size_factor'],
        args.score_threshold, args.nms_threshold, task_groups=spec.groups if spec.head_mode == 'grouped' else None,
        cls_encoding=spec.cls_encoding, use_iou=spec.use_iou, max_detections=500, box_mode=spec.box_mode)
    class_order = sorted(config['data']['kitti']['objects'], key=config['data']['kitti']['objects'].get)
    logits, probabilities = classification_heatmaps(prediction,
        groups=spec.groups if spec.head_mode == 'grouped' else (), num_classes=len(class_order))
    heads = flattened_heads(prediction)
    encoded = tensor[0].detach().float().cpu().numpy()
    indices, cell_counts, valid = hist14_cells(augmented, schema)
    np.testing.assert_allclose(height_count_channels(indices, cell_counts, schema), encoded[:4], atol=2e-6)
    np.testing.assert_array_equal((encoded[11] > 0).sum(), len(np.unique(indices[:, :2], axis=0)))
    numeric = output / 'tensors'
    numeric.mkdir(exist_ok=True)
    np.savez_compressed(numeric / 'pointcloud_stages.npz', raw=points, after_sampling=sampled,
        augmented=augmented, original_boxes=boxes, augmented_boxes=augmented_boxes,
        point_is_sampled=sample_meta['point_is_sampled'], box_is_sampled=sample_meta['box_is_sampled'])
    np.savez_compressed(numeric / 'hist14_and_cells.npz', hist14_cyx=encoded, cell_indices_xyz=indices,
        cell_counts=cell_counts, valid_point_mask=valid, channel_names=np.array(schema.channel_names))
    np.savez_compressed(numeric / 'backbone_features.npz', **features)
    np.savez_compressed(numeric / 'head_outputs.npz', **heads)
    np.savez_compressed(numeric / 'classification_heatmaps.npz', logits=logits, probabilities=probabilities,
                        class_order=np.array(class_order))
    np.save(numeric / 'detections.npy', rows)
    geometry, roi = config['data']['kitti']['geometry'], args.roi
    basis = camera_basis(28., -145.)
    limits = canvas_limits(roi_vertices(roi), basis, aspect=1.5, margin=.03)

    def visible(cloud):
        keep = np.ones(len(cloud), bool)
        for axis in range(3):
            keep &= (cloud[:, axis] >= roi[2 * axis]) & (cloud[:, axis] <= roi[2 * axis + 1])
        return cloud[keep], keep

    print('Rendering raw/augmented/cell/encoding panels...', flush=True)
    raw_visible, _ = visible(points)
    aug_visible, visible_mask = visible(augmented)
    for filename, cloud in (('01_raw_pointcloud', raw_visible), ('02_augmented_pointcloud', aug_visible)):
        render_panel(cloud, [], basis=basis, limits=limits, output=output / filename,
                     size=(1400, 930), dpi=160, point_size=.28, raw=True)
    # Green dots identify transformed KITTI crop points; they are not generated points.
    plt = setup_matplotlib()
    fig = plt.figure(figsize=(9, 6), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
    uv = projected(aug_visible, basis)
    mask = np.asarray(sample_meta['point_is_sampled'])[visible_mask]
    ax.scatter(uv[:, 0], uv[:, 1], s=.28, c='#52d9e8', linewidths=0, alpha=.65, rasterized=True)
    ax.scatter(uv[mask, 0], uv[mask, 1], s=.6, c='#79ff95', linewidths=0, rasterized=True)
    ax.set(xlim=limits[:2], ylim=limits[2:]); ax.set_aspect('equal'); ax.axis('off')
    save_plot(fig, output / '02_augmented_point_provenance')
    cell_report = render_cells(indices, cell_counts, geometry, roi, basis, limits, output / '03_occupied_cells')
    hist_ranges = map_gallery(encoded, schema.channel_names, geometry, output / '04_hist14_channels', columns=7)
    fig, ax = plt.subplots(figsize=(8, 8))
    rgb = np.stack([encoded[4], encoded[9], encoded[10]], axis=-1)
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower', interpolation='nearest',
              extent=(geometry['y_min'], geometry['y_max'], geometry['x_min'], geometry['x_max']))
    ax.set(xlim=(geometry['y_max'], geometry['y_min']), xlabel='Y lateral (m)', ylabel='X forward (m)',
           title='Hist14 preview: R=z_max, G=intensity_mean, B=log_density')
    save_plot(fig, output / '04_hist14_preview')
    feature_dir = output / 'feature_maps'
    feature_dir.mkdir(exist_ok=True)
    feature_names = [f'{name}\n{list(value.shape)}' for name, value in features.items()]
    feature_ranges = map_gallery([feature_energy(value) for value in features.values()], feature_names,
                                 geometry, output / '05_backbone_features', feature=True)
    for name, value in features.items():
        channels = np.unique(np.linspace(0, value.shape[1] - 1, min(8, value.shape[1]), dtype=int))
        map_gallery([value[0, channel] for channel in channels],
                    [f'{name} channel {channel}' for channel in channels], geometry,
                    feature_dir / (name.replace('.', '_') + '_channels'), feature=True)
        map_gallery([feature_energy(value)], [f'{name} RMS | {list(value.shape)}'], geometry,
                    feature_dir / (name.replace('.', '_') + '_energy'), columns=1, feature=True)
    map_gallery(probabilities, class_order, geometry, output / '06_classification_heads', columns=3)
    map_gallery([probabilities.max(axis=0)], ['Max class probability (before decode/NMS)'],
                 geometry, output / '06_heatmap_preview', columns=1)
    if spec.box_mode == '3d':
        bev_rows = rows[:, [0, 1, 2, 3, 5, 6, 8]]
    else:
        bev_rows = rows
    objects = bev_detections(bev_rows, config['data']['kitti']['objects'])
    render_bev(augmented, objects, geometry, output / '07_bev_detections')
    print(f'Rendering {len(rows)} decoded detections...', flush=True)
    cuboids, vertical_reports = [], []
    if spec.box_mode == 'bev':
        occupied = np.zeros(len(augmented), bool)
        for obj in objects:
            occupied |= points_in_bev(augmented, obj)
        for obj in objects:
            try:
                plane, ground = local_ground(augmented, obj, occupied, z_bounds=(roi[4], roi[5]))
                lifted, report = lift_object(augmented, obj, plane, z_bounds=(roi[4], roi[5]))
                if lifted is not None:
                    cuboids.append(lifted)
                report['ground'] = ground
            except ValueError as error:
                report = {'prediction_index': obj.index, 'status': 'unsupported', 'reason': str(error)}
            vertical_reports.append(report)
        vertical_source = 'BEV checkpoint + existing measured-LiDAR Z/height postprocess; not learned 3D'
    else:
        from types import SimpleNamespace
        for obj, row in zip(objects, rows):
            base = np.c_[obj.footprint, np.full(4, row[4])]
            cuboids.append(SimpleNamespace(name=obj.name, label_index=obj.index, score=obj.score,
                                          corners=np.r_[base, base + [0, 0, row[7]]]))
        vertical_source = 'native checkpoint z_bottom and height predictions'
    render_panel(aug_visible, cuboids, basis=basis, limits=limits, output=output / '08_3d_detections',
                 size=(1400, 930), dpi=160, point_size=.28, labels=True, class_colors=True)
    stage2_name = 'backbone_stage2' if 'backbone.stage2' in features else 'backbone'
    summary = [
        ('01_raw_pointcloud.png', '1. Raw KITTI point cloud', f'{len(points):,} measured XYZI points'),
        ('02_augmented_point_provenance.png', '2. Hybrid GT + world transforms',
         f'{len(augmented):,} points; {sample_meta["num_inserted"]} real objects inserted'),
        ('03_occupied_cells.png', '3. Occupied XY / height-band cells',
         f'{geometry["x_res"]:g} x {geometry["y_res"]:g} m XY; 4 height bands; grouping diagnostic'),
        ('04_hist14_preview.png', '4. Hist14 input tensor', str(list(tensor.shape)) + ' | all 14 channels exported'),
        (f'feature_maps/{stage2_name}_energy.png', '5. Backbone intermediate', 'RMS of actual feature channels'),
        ('feature_maps/backbone_energy.png', '6. Backbone / neck output', str(list(features['backbone'].shape))),
        ('06_heatmap_preview.png', '7. Gaussian classification heads', str(list(probabilities.shape)) + ' | sigmoid'),
        ('07_bev_detections.png', '8. Decode + NMS', f'{len(rows)} detections | score >= {args.score_threshold:g}'),
        ('08_3d_detections.png', '9. 3D view', 'Native 3D head' if spec.box_mode == '3d' else 'BEV boxes + LiDAR-estimated Z / height'),
    ]
    overview(output, summary, frame, args.seed)
    assert sha256(checkpoint_path) == checkpoint_hash
    metadata = {
        'frame_id': frame, 'pipeline': 'training augmentation -> hist14 -> model.eval forward -> decode/NMS',
        'metric_claim': 'visualization only; no validation/test AP computed',
        'source': {'bin': str(raw_path), 'bin_sha256': sha256(raw_path),
                   'label_sha256': sha256(paths.labels / f'{frame}.txt'),
                   'calibration_sha256': sha256(paths.calib / f'{frame}.txt')},
        'checkpoint': {'path': str(checkpoint_path), 'sha256': checkpoint_hash, 'epoch': checkpoint.get('epoch'),
                       'strict_state_dict': True, 'embedded_config_used': True, 'box_mode': spec.box_mode},
        'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        'donor_database': donor_report, 'augmentation': aug_meta,
        'sampling': {key: value for key, value in sample_meta.items()
                     if key not in ('point_is_sampled', 'box_is_sampled')},
        'point_counts': {'raw': len(points), 'after_sampling': len(sampled), 'augmented': len(augmented),
                         'encoder_valid': int(valid.sum()), 'raw_display_roi': len(raw_visible),
                         'augmented_display_roi': len(aug_visible)},
        'occupied_cells': cell_report, 'encoding': schema.semantic_metadata(),
        'effective_attention_placement': backbone_placement(model),
        'input_shape_bcyx': list(tensor.shape), 'hist14_display_ranges': hist_ranges,
        'feature_shapes_bcyx': {name: list(value.shape) for name, value in features.items()},
        'feature_visualization': {'reduction': 'sqrt(mean(channel**2))',
            'normalization': 'individual map 99th percentile, display only; raw values saved unchanged',
            'color_is_detection_confidence': False, 'ranges': feature_ranges},
        'head_shapes': {name: list(value.shape) for name, value in heads.items()},
        'class_order': class_order, 'detections': {'count': len(rows), 'rows': rows,
            'score_threshold': args.score_threshold, 'nms_threshold': args.nms_threshold,
            'per_class': dict(Counter(obj.name for obj in objects))},
        'vertical_source': vertical_source, 'cuboids_rendered': len(cuboids), 'vertical_reports': vertical_reports,
        'roi_xyz': roi, 'device': device, 'elapsed_seconds': time.perf_counter() - started,
    }
    (output / 'config.inference.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'metadata.json').write_text(json.dumps(jsonable(metadata), indent=2, allow_nan=False) + '\n')
    (output / 'README.md').write_text(
        f'# KITTI {frame}: actual pipeline visualization\n\n'
        'Open `pipeline_overview.png`, then the numbered PNG/SVG files. '
        '`04_hist14_channels.png` shows every input channel; `05_backbone_features.png` '
        'shows spatial RMS activations. `feature_maps/` also contains eight actual channels per layer.\n\n'
        'Raw XYZI, augmented XYZI, occupied-cell indices/counts, full hist14/features/heads and predictions '
        'are in `tensors/`. No feature maps or detections are hand-painted.\n\n'
        'This is the augmentation/training data path followed by a checkpoint eval forward, not an AP evaluation. '
        'The donor DB is a documented real train-only subset; original points and checkpoint are preserved. '
        'The voxel picture depicts hist14 grouping, not an additional 3D sparse network. '
        'The 3D panels show the fixed ROI recorded in metadata; full scans are saved. '
        'Feature maps use their individual 99th percentile for display; signed channels use a diverging scale.\n\n'
        f'Vertical geometry: {vertical_source}.\n\n'
        '| Captured tensor | Shape [B,C,Y,X] |\n|---|---|\n'
        f'| hist14 input | {list(tensor.shape)} |\n' +
        ''.join(f'| {name} | {list(value.shape)} |\n' for name, value in features.items()))
    print(f'Complete: {output / "pipeline_overview.png"}', flush=True)
    return metadata


if __name__ == '__main__':
    main()

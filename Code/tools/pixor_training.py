#!/usr/bin/env python3
"""Standalone, from-scratch PIXOR Car/BEV reproduction on canonical KITTI val.

This is a documented reproduction recipe, not the original author's code/weights.
Torch is imported only by GPU/model commands so geometry can be checked on CPU.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np

import openpcdet_benchmark as common


GEOMETRY = dict(x_min=0., x_max=70., y_min=-40., y_max=40., z_min=-2.5,
                z_max=1., resolution=.1, stride=4)
# Raw target order: cos(yaw), sin(yaw), dx, dy, log(width), log(length).
# Fixed constants, independent of both train and validation label statistics.
REG_MEAN = np.array([0., 0., 0., 0., math.log(1.6), math.log(3.9)], dtype=np.float32)
REG_SCALE = np.array([1., 1., 3., 3., 1., 1.], dtype=np.float32)
RECIPE = 'pixor_car_bev_reproduction_v1'


def default_settings():
    return dict(epochs=50, batch_size=2, accumulation_steps=4, workers=2, seed=42,
                learning_rate=.01, momentum=.9, weight_decay=.0001, amp=True,
                augmentation=True, eval_every=5, save_every=5, warmup_frames=10,
                score_threshold=.05, nms_iou=.1, pre_nms_topk=1000, max_detections=100,
                copy_data_to_local=True, resume='auto', eval_checkpoint='best',
                kitti_root='', output_root='', work_dir='', run_name='pixor_50epochs', image_sizes_json=None)


def grid_shape(geometry):
    g = geometry
    return (round((g['y_max'] - g['y_min']) / g['resolution']),
            round((g['x_max'] - g['x_min']) / g['resolution']),
            round((g['z_max'] - g['z_min']) / g['resolution']))


def output_grid(geometry):
    height, width, _ = grid_shape(geometry)
    step = geometry['resolution'] * geometry['stride']
    yy, xx = np.meshgrid(geometry['y_min'] + (np.arange(math.ceil(height / 4)) + .5) * step,
                         geometry['x_min'] + (np.arange(math.ceil(width / 4)) + .5) * step,
                         indexing='ij')
    return xx.astype(np.float32), yy.astype(np.float32)


def encode_bev(points, geometry=GEOMETRY):
    """35 height occupancy slices and max reflectance; rows=y, columns=x."""
    g = geometry
    height, width, depth = grid_shape(g)
    bev = np.zeros((depth + 1, height, width), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 4)
    mask = np.isfinite(points).all(axis=1)
    for axis, name in enumerate(('x', 'y', 'z')):
        mask &= (points[:, axis] >= g[name + '_min']) & (points[:, axis] < g[name + '_max'])
    points = points[mask]
    if not len(points):
        return bev
    indices = np.floor((points[:, :3] - [g['x_min'], g['y_min'], g['z_min']]) / g['resolution']).astype(np.int64)
    x, y, z = indices.T
    # Guard floating-point values infinitesimally inside the upper boundary.
    x, y, z = np.minimum(x, width - 1), np.minimum(y, height - 1), np.minimum(z, depth - 1)
    bev[z, y, x] = 1.
    np.maximum.at(bev[-1], (y, x), points[:, 3].clip(0, 1))
    return bev


def read_calibration(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('P2', 'R0_rect', 'R_rect', 'Tr_velo_to_cam', 'Tr_velo_cam'):
            values[key] = np.asarray([float(x) for x in value.split()], dtype=np.float64)
    rotation = np.eye(4)
    rotation[:3, :3] = values.get('R0_rect', values.get('R_rect')).reshape(3, 3)
    velo = np.eye(4)
    velo[:3] = values.get('Tr_velo_to_cam', values.get('Tr_velo_cam')).reshape(3, 4)
    result = dict(transform=rotation @ velo, projection=values['P2'].reshape(3, 4))
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError(f'Nonfinite calibration: {path}')
    return result


def read_objects(path, calibration):
    annotation = common.read_annotation(path)
    inverse = np.linalg.inv(calibration['transform'])
    boxes = []
    for i, name in enumerate(annotation['name']):
        if name not in ('Car', 'Van'):
            continue
        length, height, width = annotation['dimensions'][i]
        if min(length, height, width) <= 0:
            continue
        camera = np.asarray([*annotation['location'][i], 1.])
        bottom = inverse @ camera
        # KITTI camera heading is (cos(ry), 0, -sin(ry)).
        ry = annotation['rotation_y'][i]
        direction = inverse[:3, :3] @ [math.cos(ry), 0, -math.sin(ry)]
        boxes.append(dict(name=name, x=float(bottom[0]), y=float(bottom[1]), bottom_z=float(bottom[2]),
                          height=height, width=width, length=length,
                          yaw=math.atan2(float(direction[1]), float(direction[0]))))
    return boxes


def transform_scene(points, boxes, matrix):
    points = points.copy()
    points[:, :2] = points[:, :2] @ matrix[:2, :2].T + matrix[:2, 2]
    result = copy.deepcopy(boxes)
    scale = float(np.linalg.norm(matrix[:2, 0]))
    for box in result:
        xy = matrix @ [box['x'], box['y'], 1.]
        direction = matrix[:2, :2] @ [math.cos(box['yaw']), math.sin(box['yaw'])]
        box.update(x=float(xy[0]), y=float(xy[1]), yaw=math.atan2(float(direction[1]), float(direction[0])),
                   length=box['length'] * scale, width=box['width'] * scale)
    return points, result


def augmentation_matrix():
    angle = float(np.random.uniform(-math.pi / 36, math.pi / 36))
    scale = float(np.random.uniform(.95, 1.05))
    matrix = np.eye(3)
    matrix[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    matrix[:2, :2] *= scale
    if np.random.random() < .5:
        matrix[:2, 1] *= -1
    matrix[:2, 2] = np.random.uniform(-.25, .25, size=2)
    return matrix


def projection_mask(calibration, image_shape, geometry=GEOMETRY, matrix=None, dontcare=()):
    xx, yy = output_grid(geometry)
    xy = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    if matrix is not None:
        xy = np.linalg.inv(matrix) @ xy
    points = np.column_stack([xy[0], xy[1], np.full(xx.size, -1.73), np.ones(xx.size)])
    projected = points @ calibration['transform'].T @ calibration['projection'].T
    depth = projected[:, 2]
    uv = projected[:, :2] / np.maximum(depth[:, None], 1e-6)
    height, width = image_shape
    mask = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    # Ground-plane projection is an explicit approximation for FOV/ignore masks.
    for x1, y1, x2, y2 in dontcare:
        mask &= ~((uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2))
    return mask.reshape(xx.shape)


def make_targets(boxes, geometry=GEOMETRY, valid_mask=None):
    xx, yy = output_grid(geometry)
    cls = np.zeros((1, *xx.shape), dtype=np.float32)
    reg = np.zeros((6, *xx.shape), dtype=np.float32)
    valid = np.ones((1, *xx.shape), dtype=np.float32)
    if valid_mask is not None:
        valid[0] = valid_mask
    # Small boxes are assigned last when footprints overlap.
    for box in sorted(boxes, key=lambda b: b['length'] * b['width'], reverse=True):
        dx, dy = xx - box['x'], yy - box['y']
        c, s = math.cos(box['yaw']), math.sin(box['yaw'])
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        outer = (np.abs(local_x) <= .6 * box['length']) & (np.abs(local_y) <= .6 * box['width'])
        valid[0, outer] = 0
        inside_roi = geometry['x_min'] <= box['x'] < geometry['x_max'] and geometry['y_min'] <= box['y'] < geometry['y_max']
        if box['name'] != 'Car' or not inside_roi:
            continue
        positive = (np.abs(local_x) <= .35 * box['length']) & (np.abs(local_y) <= .35 * box['width'])
        if valid_mask is not None:
            positive &= valid_mask
        cls[0, positive] = 1
        valid[0, positive] = 1
        raw = np.stack([np.full_like(xx, c), np.full_like(xx, s), -dx, -dy,
                        np.full_like(xx, math.log(box['width'])), np.full_like(xx, math.log(box['length']))])
        normalized = (raw - REG_MEAN[:, None, None]) / REG_SCALE[:, None, None]
        reg[:, positive] = normalized[:, positive]
    # Existing positives have priority over subsequent overlapping ignore bands.
    valid[cls > 0] = 1
    return cls, reg, valid


def decode_arrays(logits, regression, geometry=GEOMETRY, threshold=.05, topk=1000, valid_mask=None):
    logits = np.asarray(logits, dtype=np.float32)
    scores = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    mask = (scores >= threshold) & np.isfinite(scores)
    if valid_mask is not None:
        mask &= valid_mask.astype(bool)
    rows, cols = np.where(mask)
    if not len(rows):
        return np.empty((0, 6), dtype=np.float32)
    order = np.argsort(-scores[rows, cols], kind='stable')[:topk]
    rows, cols = rows[order], cols[order]
    raw = regression[:, rows, cols].T * REG_SCALE + REG_MEAN
    xx, yy = output_grid(geometry)
    boxes = np.column_stack([xx[rows, cols] + raw[:, 2], yy[rows, cols] + raw[:, 3],
        np.exp(np.clip(raw[:, 5], -4, 4)), np.exp(np.clip(raw[:, 4], -4, 4)),
        np.arctan2(raw[:, 1], raw[:, 0]), scores[rows, cols]])
    return boxes[np.isfinite(boxes).all(axis=1)].astype(np.float32)


def rotated_nms(boxes, threshold=.1, maximum=100):
    import cv2
    if not len(boxes):
        return boxes
    rectangles = [((float(x), float(y)), (float(length), float(width)), float(yaw * 180 / math.pi))
                  for x, y, length, width, yaw, _ in boxes]
    indices = cv2.dnn.NMSBoxesRotated(rectangles, boxes[:, 5].tolist(), 0., threshold)
    return boxes[np.asarray(indices, dtype=np.int64).reshape(-1)[:maximum]]


def kitti_prediction_row(box, calibration, image_shape, prior):
    x, y, length, width, yaw, score = map(float, box)
    height, bottom_z = float(prior['height']), float(prior['bottom_z'])
    transform, projection = calibration['transform'], calibration['projection']
    location = (transform @ [x, y, bottom_z, 1.])[:3]
    direction = transform[:3, :3] @ [math.cos(yaw), math.sin(yaw), 0]
    ry = math.atan2(-float(direction[2]), float(direction[0]))
    alpha = (ry - math.atan2(float(location[0]), float(location[2])) + math.pi) % (2 * math.pi) - math.pi
    corners = []
    for z in (bottom_z, bottom_z + height):
        for a, b in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            dx, dy = a * length / 2, b * width / 2
            corners.append([x + math.cos(yaw) * dx - math.sin(yaw) * dy,
                            y + math.sin(yaw) * dx + math.cos(yaw) * dy, z, 1.])
    projected = np.asarray(corners) @ transform.T @ projection.T
    projected = projected[projected[:, 2] > .1]
    if not len(projected):
        return None
    uv = projected[:, :2] / projected[:, 2:3]
    image_height, image_width = image_shape
    bbox = np.r_[uv.min(axis=0), uv.max(axis=0)]
    bbox[[0, 2]] = bbox[[0, 2]].clip(0, image_width - 1)
    bbox[[1, 3]] = bbox[[1, 3]].clip(0, image_height - 1)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    values = [alpha, *bbox, height, width, length, *location, ry, score]
    return 'Car -1 -1 ' + ' '.join(f'{value:.4f}' for value in values)


def build_model():
    import torch
    from torch import nn

    def conv(in_channels, out_channels, kernel=3, stride=1):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, bias=False),
                             nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    class Bottleneck(nn.Module):
        def __init__(self, incoming, planes, stride):
            super().__init__()
            self.layers = nn.Sequential(conv(incoming, planes, 1), conv(planes, planes, 3, stride),
                nn.Conv2d(planes, 4 * planes, 1, bias=False), nn.BatchNorm2d(4 * planes))
            self.skip = (nn.Sequential(nn.Conv2d(incoming, 4 * planes, 1, stride, bias=False),
                         nn.BatchNorm2d(4 * planes)) if incoming != 4 * planes or stride != 1 else nn.Identity())
        def forward(self, inputs):
            return torch.relu(self.layers(inputs) + self.skip(inputs))

    class PIXOR(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(conv(36, 32), conv(32, 32))
            self.blocks = nn.ModuleList()
            incoming = 32
            for planes, count in ((24, 3), (48, 6), (64, 6), (96, 3)):
                layers = [Bottleneck(incoming, planes, 2)]
                layers += [Bottleneck(4 * planes, planes, 1) for _ in range(count - 1)]
                self.blocks.append(nn.Sequential(*layers))
                incoming = 4 * planes
            self.lateral5 = nn.Conv2d(384, 196, 1)
            self.lateral4 = nn.Conv2d(256, 128, 1)
            self.lateral3 = nn.Conv2d(192, 96, 1)
            self.up5 = nn.ConvTranspose2d(196, 128, 3, 2, 1)
            self.up4 = nn.ConvTranspose2d(128, 96, 3, 2, 1)
            self.header = nn.Sequential(*[conv(96, 96) for _ in range(4)], nn.Conv2d(96, 7, 3, padding=1))
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.normal_(self.header[-1].weight, std=.001)
            with torch.no_grad():
                self.header[-1].bias[0] = math.log(.01 / .99)
        def forward(self, inputs):
            features = self.stem(inputs)
            levels = []
            for block in self.blocks:
                features = block(features)
                levels.append(features)
            p4 = self.lateral4(levels[2])
            p4 = p4 + self.up5(self.lateral5(levels[3]), output_size=p4.shape)
            p3 = self.lateral3(levels[1])
            return self.header(p3 + self.up4(p4, output_size=p3.shape))

    return PIXOR()


def compute_loss(prediction, classification, regression, valid):
    import torch
    from torch.nn import functional as functional
    logits, values = prediction[:, :1].float(), prediction[:, 1:].float()
    classification, regression, valid = classification.float(), regression.float(), valid.float()
    probability = logits.sigmoid()
    pt = probability * classification + (1 - probability) * (1 - classification)
    alpha = .25 * classification + .75 * (1 - classification)
    denominator = classification.sum().clamp(min=1)
    cls = (functional.binary_cross_entropy_with_logits(logits, classification, reduction='none')
           * alpha * (1 - pt).square() * valid).sum() / denominator
    reg = (functional.smooth_l1_loss(values, regression, reduction='none') * classification).sum() / denominator
    return cls + reg, cls.detach(), reg.detach()


def read_settings(path):
    settings = {**default_settings(), **json.loads(Path(path).read_text())}
    for name in ('epochs', 'batch_size', 'accumulation_steps', 'eval_every', 'save_every',
                 'warmup_frames', 'pre_nms_topk', 'max_detections'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(settings['workers']) is not int or settings['workers'] < 0:
        raise ValueError('workers must be a nonnegative integer')
    for name in ('score_threshold', 'nms_iou'):
        if not 0 < settings[name] < 1:
            raise ValueError(f'{name} must be between 0 and 1')
    if not settings['run_name'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. ' for c in settings['run_name']) or settings['run_name'] in ('.', '..'):
        raise ValueError('run_name must be a simple directory name')
    for name in ('kitti_root', 'output_root', 'work_dir'):
        if not settings[name]:
            raise ValueError(f'Set {name}')
        settings[name] = str(Path(settings[name]).expanduser().resolve())
    raw = Path(settings['kitti_root'])
    for name in ('output_root', 'work_dir'):
        destination = Path(settings[name])
        if destination == raw or raw in destination.parents:
            raise ValueError(f'{name} must be outside the source dataset')
    return settings


def paths(settings):
    return Path(settings['work_dir']), Path(settings['output_root']) / settings['run_name']


def audit_partition(work):
    result = {}
    for name in ('train', 'val'):
        result[name] = common.read_ids(Path(work) / 'splits' / (name + '.txt'))
        digest = hashlib.sha256(('\n'.join(result[name]) + '\n').encode()).hexdigest()
        if digest != common.SPLIT_HASHES[name]:
            raise ValueError(f'{name}: expected the canonical KITTI 3712/3769 partition')
    common.check_partition(result['train'], result['val'])
    return result


def verify_reference(work):
    for name, digest in common.EVALUATOR_HASHES.items():
        if common.sha256(Path(work) / 'kitti_reference' / name) != digest:
            raise ValueError('Reference evaluator source changed: ' + name)


def prepare(settings):
    work, output = paths(settings)
    splits = audit_partition(work)
    verify_reference(work)
    raw = common.normalize_training_root(settings['kitti_root'], settings.get('image_sizes_json'))
    identifiers = splits['train'] + splits['val']
    shapes, image_sources, image_metadata = common.load_image_dimensions(raw, identifiers, settings.get('image_sizes_json'))
    root = work / 'kitti_local/training' if settings['copy_data_to_local'] else raw
    records, pending, train_heights, train_bottoms = {}, [], [], []
    signature = hashlib.sha256()
    for path in image_sources:
        if path.suffix.lower() == '.json':
            signature.update(path.read_bytes())
        else:
            with path.open('rb') as stream:
                signature.update(stream.read(24))
    for split, identifiers in splits.items():
        for index, identifier in enumerate(identifiers):
            for folder, suffix in (('velodyne', '.bin'), ('calib', '.txt'), ('label_2', '.txt')):
                original = raw / folder / (identifier + suffix)
                stat = original.stat()
                if folder == 'velodyne' and (stat.st_size == 0 or stat.st_size % 16):
                    raise ValueError(f'Invalid XYZI point file: {original}')
                signature.update(f'{identifier}/{folder}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
                if folder in ('label_2', 'calib'):
                    signature.update(original.read_bytes())
                # PNG pixels are never model input; cache only dimensions.
                if settings['copy_data_to_local'] and folder != 'image_2':
                    target = root / folder / original.name
                    if not target.exists() or (target.stat().st_size, target.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                        pending.append((original, target, stat.st_size))
            calibration = read_calibration(raw / 'calib' / (identifier + '.txt'))
            label = raw / 'label_2' / (identifier + '.txt')
            boxes = read_objects(label, calibration)
            annotation = common.read_annotation(label)
            dontcare = [annotation['bbox'][i] for i, name in enumerate(annotation['name']) if name == 'DontCare']
            record = dict(image_shape=shapes[identifier])
            if split == 'train':
                record.update(boxes=boxes, dontcare=dontcare)
                for box in boxes:
                    if box['name'] == 'Car':
                        train_heights.append(box['height'])
                        train_bottoms.append(box['bottom_z'])
            records[identifier] = record
            if (index + 1) % 500 == 0:
                print(f'Audit {split}: {index + 1}/{len(identifiers)}', flush=True)
    if not train_heights:
        raise ValueError('No Car training labels found')
    required = sum(n for _, _, n in pending)
    if pending and shutil.disk_usage(work).free < required + 1024 ** 3:
        raise RuntimeError(f'Need {required / 1e9:.1f} GB for local staging; disable COPY_DATA_TO_LOCAL or choose a larger disk.')
    for i, (original, target, _) in enumerate(pending):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.copying')
        shutil.copy2(original, temporary)
        temporary.replace(target)
        if (i + 1) % 500 == 0:
            print(f'Copy local: {i + 1}/{len(pending)}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    assets = dict(training_root=str(root), source_training_root=str(raw), splits=splits, frames=records,
                  asset_signature=signature.hexdigest(), image_sizes_json=image_metadata,
                  projection_prior=dict(height=float(np.median(train_heights)), bottom_z=float(np.median(train_bottoms))),
                  prior_scope='Median Car height/bottom-z from train IDs only, used only to project BEV boxes into KITTI format. No AP3D is computed.')
    common.write_json(work / 'assets.json', assets)
    common.write_json(output / 'data_audit.json', {key: value for key, value in assets.items() if key != 'frames'})
    common.write_json(output / 'config.resolved.json', {**settings, 'recipe': RECIPE, 'geometry': GEOMETRY,
        'regression_mean': REG_MEAN.tolist(), 'regression_scale': REG_SCALE.tolist(),
        'architecture_blocks': [3, 6, 6, 3], 'classes': ['Car'], 'metric': 'APBEV_R40_IoU0.7',
        'selection': 'Best full validation Car Moderate R40, evaluated every eval_every epochs and at the final epoch',
        'changes_from_original': '50-epoch SGD recipe; fixed regression normalization; max reflectance; BN/ReLU header; inner/outer target masks; BEV-plane augmentation and projected FOV/DontCare masks. A reproduction, not an exact replay of original author experiments.'})
    for name in splits:
        shutil.copyfile(work / 'splits' / (name + '.txt'), output / (name + '.txt'))
    print('Prepared 3712 train / 3769 val; no random repartition; no val labels used for training/prior.', flush=True)


def training_identity(settings, asset_signature):
    keys = ('epochs', 'batch_size', 'accumulation_steps', 'seed', 'learning_rate', 'momentum', 'weight_decay',
            'amp', 'augmentation', 'eval_every', 'score_threshold', 'nms_iou', 'pre_nms_topk', 'max_detections')
    return common.fingerprint(dict(recipe=RECIPE, settings={key: settings[key] for key in keys},
        geometry=GEOMETRY, regression_mean=REG_MEAN.tolist(), regression_scale=REG_SCALE.tolist(),
        splits=common.SPLIT_HASHES, asset_signature=asset_signature,
        runner_sha256=common.sha256(__file__)))


def load_assets(settings):
    work, _ = paths(settings)
    splits = audit_partition(work)
    assets = json.loads((work / 'assets.json').read_text())
    if assets['splits'] != splits or assets['source_training_root'] != str(common.normalize_training_root(settings['kitti_root'], settings.get('image_sizes_json'))):
        raise ValueError('Dataset settings changed; rerun prepare')
    return assets


def setup(settings):
    import torch
    work, output = paths(settings)
    if platform.system() != 'Linux' or not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError('Use Linux Python 3.10–3.13 with CUDA PyTorch; e.g. Colab GPU')
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU runtime before setup')
    environment = work / 'env'
    python = common.ensure_python_environment(environment)
    common.run_command([python, '-m', 'pip', 'install', 'numpy>=1.26,<2.3',
                        'numba>=0.61.2,<0.62', 'opencv-python-headless>=4.8,<4.12'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'pip_freeze.txt').open('w') as stream:
        subprocess.run([str(python), '-m', 'pip', 'freeze'], stdout=stream, check=True)
    print('PIXOR is pure 2D PyTorch: no spconv/OpenPCDet/C++ build required.', flush=True)


def configure_runtime(seed):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for notebook smoke/train/evaluate')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')


def worker_seed(worker_id):
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class TrainingDataset:
    def __init__(self, assets, settings, identifiers=None):
        self.assets, self.settings = assets, settings
        self.identifiers = identifiers if identifiers is not None else assets['splits']['train']
        self.root = Path(assets['training_root'])
    def __len__(self):
        return len(self.identifiers)
    def __getitem__(self, index):
        identifier = self.identifiers[index]
        points = np.fromfile(self.root / 'velodyne' / (identifier + '.bin'), dtype='<f4').reshape(-1, 4)
        record = self.assets['frames'][identifier]
        calibration = read_calibration(self.root / 'calib' / (identifier + '.txt'))
        matrix = augmentation_matrix() if self.settings['augmentation'] else np.eye(3)
        points, boxes = transform_scene(points, record['boxes'], matrix)
        mask = projection_mask(calibration, record['image_shape'], matrix=matrix, dontcare=record['dontcare'])
        cls, reg, valid = make_targets(boxes, valid_mask=mask)
        return encode_bev(points), cls, reg, valid


def data_loader(assets, settings, epoch, smoke=False):
    import torch
    generator = torch.Generator().manual_seed(settings['seed'] + epoch)
    # Epoch seeding makes a resumed epoch reproduce shuffle and augmentations.
    random.seed(settings['seed'] + epoch)
    np.random.seed(settings['seed'] + epoch)
    dataset = TrainingDataset(assets, settings, assets['splits']['train'][:2] if smoke else None)
    return torch.utils.data.DataLoader(dataset, batch_size=min(settings['batch_size'], len(dataset)), shuffle=True,
        num_workers=0 if smoke else settings['workers'], pin_memory=True, drop_last=False,
        worker_init_fn=worker_seed, generator=generator,
        **({'prefetch_factor': 1} if settings['workers'] and not smoke else {}))


def reference_check(settings):
    import torch
    work, output = paths(settings)
    verify_reference(work)
    sys.path.insert(0, str(work))
    from kitti_reference.rotate_iou import rotate_iou_gpu_eval
    box = np.array([[0, 0, 2, 2, 0]], dtype=np.float32)
    overlap = float(rotate_iou_gpu_eval(box, box)[0, 0])
    if abs(overlap - 1) > 1e-5:
        raise RuntimeError('Reference rotated IoU GPU kernel failed')
    common.write_json(output / 'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
        torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        gpu_capability=list(torch.cuda.get_device_capability(0)),
        nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
        evaluator_revision=common.REVISION, evaluator_sha256=common.EVALUATOR_HASHES,
        train_precision='amp-fp16' if settings['amp'] else 'fp32', inference_precision='fp32', allow_tf32=False))


def smoke(settings):
    import torch
    configure_runtime(settings['seed'])
    reference_check(settings)
    assets = load_assets(settings)
    model = build_model().cuda().train()
    inputs, cls, reg, valid = [value.cuda(non_blocking=True) for value in next(iter(data_loader(assets, settings, 0, smoke=True)))]
    scaler = torch.amp.GradScaler('cuda', enabled=settings['amp'])
    optimizer = torch.optim.SGD(model.parameters(), lr=settings['learning_rate'])
    max_attempts = 16 if settings['amp'] else 1
    for attempt in range(1, max_attempts + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=settings['amp']):
            prediction = model(inputs)
            loss, cls_loss, reg_loss = compute_loss(prediction, cls, reg, valid)
        if not torch.isfinite(loss) or prediction.shape[-2:] != cls.shape[-2:]:
            raise RuntimeError('Smoke produced invalid loss/shape')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # FP16 backward can overflow while the initial loss scale calibrates.
        # Unscale records these gradients so GradScaler can safely skip the step.
        # Only clip finite gradients; retain the strict check for FP32 failures.
        gradients_finite = torch.stack([torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None]).all()
        if not settings['amp'] or gradients_finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            print(f'Smoke AMP overflow {attempt}/{max_attempts}: optimizer step skipped; '
                  f'loss_scale={previous_scale:g} -> {scaler.get_scale():g}; retrying batch.', flush=True)
            continue
        print(f'Smoke OK: input={tuple(inputs.shape)}, output={tuple(prediction.shape)}, '
              f'positive_pixels={int(cls.sum())}, loss={float(loss.detach()):.5f}, '
              f'attempts={attempt}, loss_scale={scaler.get_scale():g}. No training checkpoint saved.', flush=True)
        return
    raise FloatingPointError(f'Smoke AMP gradients remain nonfinite after {max_attempts} attempts; '
                             'set AMP=False in notebook settings to diagnose in FP32.')


def save_checkpoint(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_ap, identity, history):
    import torch
    state = np.random.get_state()
    return dict(model_state=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(), epoch=epoch, best_ap=best_ap, identity=identity, recipe=RECIPE, history=history,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
        numpy_rng=[state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])])


def restore_random_states(state):
    import torch
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    value = state['numpy_rng']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))


def finalize_epoch(model, optimizer, scheduler, scaler, settings, assets, history, best_ap, identity):
    """Also used on resume if validation was interrupted after the epoch save."""
    _, output = paths(settings)
    row = history[-1]
    epoch = row['epoch']
    is_best = False
    if epoch % settings['eval_every'] == 0 or epoch == settings['epochs']:
        report = evaluate_model(model, settings, assets, output / 'validation' / f'epoch_{epoch:03d}')
        ap = report['accuracy']['R40']['Moderate']
        row['car_bev_r40_moderate'] = ap
        if ap > best_ap:
            best_ap, is_best = ap, True
    payload = checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_ap, identity, history)
    payload['validation_pending'] = False
    checkpoint_dir = output / 'checkpoints'
    if is_best:
        save_checkpoint(checkpoint_dir / 'best.pth', payload)
    if epoch % settings['save_every'] == 0 or epoch == settings['epochs']:
        save_checkpoint(checkpoint_dir / f'epoch_{epoch:03d}.pth', payload)
    common.write_json(output / 'history.json', history)
    with (output / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(history)
    # Commit last only after best/history/periodic files are complete. A failure
    # before this point leaves the pre-validation checkpoint pending for retry.
    save_checkpoint(checkpoint_dir / 'last.pth', payload)
    print(json.dumps(row), flush=True)
    return best_ap


def score_predictions(assets, prediction_dir, identifiers, work):
    sys.path.insert(0, str(work))
    verify_reference(work)
    from kitti_reference import eval as reference
    root = Path(assets['training_root'])
    gt = [common.array_annotation(common.read_annotation(root / 'label_2' / (identifier + '.txt'))) for identifier in identifiers]
    dt = [common.array_annotation(common.read_annotation(prediction_dir / (identifier + '.txt'), prediction=True)) for identifier in identifiers]
    curves = reference.eval_class(gt, dt, [0], [0, 1, 2], 1, np.full((1, 3, 1), .7),
                                  compute_aos=False, num_parts=50)
    result = {}
    for name, function in [('R11', reference.get_mAP), ('R40', reference.get_mAP_R40)]:
        values = function(curves['precision'])[0, :, 0]
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite reference AP')
        result[name] = {difficulty: float(values[i]) for i, difficulty in enumerate(common.DIFFICULTIES)}
    return result


def evaluate_model(model, settings, assets, destination, *, profile=False):
    import torch
    work, _ = paths(settings)
    destination = Path(destination)
    predictions = destination / 'kitti_predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    root = Path(assets['training_root'])
    model = model.float().eval()
    identifiers = assets['splits']['val']
    timings = {stage: [] for stage in common.STAGES}
    rows = []
    raw_predictions = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, identifier in enumerate(identifiers):
            if profile:
                torch.cuda.synchronize()
            start = time.perf_counter()
            calibration = read_calibration(root / 'calib' / (identifier + '.txt'))
            image_shape = assets['frames'][identifier]['image_shape']
            points = np.fromfile(root / 'velodyne' / (identifier + '.bin'), dtype='<f4').reshape(-1, 4)
            inputs = torch.from_numpy(encode_bev(points)).unsqueeze(0)
            mask = projection_mask(calibration, image_shape)
            preprocessed = time.perf_counter()
            if profile:
                inputs, h2d = common.timed_cuda(lambda: inputs.cuda())
                prediction, network = common.timed_cuda(lambda: model(inputs))
            else:
                prediction = model(inputs.cuda())
            decoded_start = time.perf_counter()
            array = prediction[0].float().cpu().numpy()
            boxes = rotated_nms(decode_arrays(array[0], array[1:], threshold=settings['score_threshold'],
                topk=settings['pre_nms_topk'], valid_mask=mask), settings['nms_iou'], settings['max_detections'])
            if profile:
                torch.cuda.synchronize()
            end = time.perf_counter()
            if profile:
                row = dict(frame_id=identifier, preprocess=(preprocessed - start) * 1000,
                           host_to_device=h2d, network=network, decode_nms=(end - decoded_start) * 1000,
                           input_to_detections=(end - start) * 1000, included=index >= settings['warmup_frames'])
                rows.append(row)
                if row['included']:
                    for stage in timings:
                        timings[stage].append(row[stage])
            # Projection/export/AP and GT access are outside the latency boundary.
            lines = [kitti_prediction_row(box, calibration, image_shape, assets['projection_prior']) for box in boxes]
            lines = [line for line in lines if line is not None]
            (predictions / (identifier + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
            raw_predictions[identifier] = boxes.tolist()
            if (index + 1) % 250 == 0:
                print(f'Validation: {index + 1}/{len(identifiers)}', flush=True)
    accuracy = score_predictions(assets, predictions, identifiers, work)
    report = dict(status='ok', recipe=RECIPE, classes=['Car'], metric='BEV', iou=.7,
        frames=len(identifiers), benchmark_complete=len(identifiers) == 3769,
        accuracy=accuracy, parameters=sum(p.numel() for p in model.parameters()),
        projection_prior=assets['projection_prior'], projection_scope=assets['prior_scope'],
        evaluator_revision=common.REVISION, evaluator_sha256=common.EVALUATOR_HASHES,
        latency={stage: common.timing_summary(values) for stage, values in timings.items()} if profile else None,
        peak_memory_bytes=torch.cuda.max_memory_allocated(), precision='fp32',
        prediction_sha256={identifier: common.sha256(predictions / (identifier + '.txt')) for identifier in identifiers})
    common.write_json(destination / 'report.json', report)
    common.write_json(destination / 'bev_predictions_lidar.json', raw_predictions)
    if profile:
        with (destination / 'latency_per_frame.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('Car APBEV@0.7 R40:', accuracy['R40'], flush=True)
    return report


def train(settings):
    import torch
    configure_runtime(settings['seed'])
    reference_check(settings)
    work, output = paths(settings)
    assets = load_assets(settings)
    identity = training_identity(settings, assets['asset_signature'])
    model = build_model().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=settings['learning_rate'], momentum=settings['momentum'],
                                weight_decay=settings['weight_decay'])
    milestones = sorted(set([max(1, round(settings['epochs'] * .6)), max(1, round(settings['epochs'] * .8))]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=.1)
    scaler = torch.amp.GradScaler('cuda', enabled=settings['amp'])
    checkpoint_dir = output / 'checkpoints'
    latest = checkpoint_dir / 'last.pth'
    resume = settings['resume']
    resume_path = latest if resume == 'auto' else (Path(resume).expanduser() if resume else None)
    start_epoch, best_ap, history = 0, -1., []
    if latest.exists() and not resume:
        raise ValueError('Existing run: enable RESUME or select another RUN_NAME')
    if resume_path and resume_path.exists():
        state = torch.load(resume_path, map_location='cpu', weights_only=True)
        if state.get('identity') != identity:
            raise ValueError('Resume checkpoint has different data, split, code or training settings. Choose a new RUN_NAME.')
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, best_ap, history = state['epoch'], state['best_ap'], state['history']
        restore_random_states(state)
        if state.get('validation_pending'):
            print(f'Completing pending validation/checkpoint finalization for epoch {start_epoch}', flush=True)
            best_ap = finalize_epoch(model, optimizer, scheduler, scaler, settings, assets, history, best_ap, identity)
        del state
        print(f'Resume epoch {start_epoch}; target TOTAL epochs={settings["epochs"]}', flush=True)
    elif resume not in ('auto', '', None, False):
        raise FileNotFoundError(resume_path)
    for epoch in range(start_epoch, settings['epochs']):
        model.train()
        loader = data_loader(assets, settings, epoch)
        optimizer.zero_grad(set_to_none=True)
        total, examples = np.zeros(3), 0
        started = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        accumulation = settings['accumulation_steps']
        for index, batch in enumerate(loader):
            inputs, cls, reg, valid = [value.cuda(non_blocking=True) for value in batch]
            # The final, smaller accumulation group receives the correct divisor.
            group_size = min(accumulation, len(loader) - (index // accumulation) * accumulation)
            with torch.autocast('cuda', dtype=torch.float16, enabled=settings['amp']):
                prediction = model(inputs)
                loss, cls_loss, reg_loss = compute_loss(prediction, cls, reg, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch + 1}, batch {index}')
            scaler.scale(loss / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=not settings['amp'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            count = len(inputs)
            total += np.array([float(loss.detach()), float(cls_loss), float(reg_loss)]) * count
            examples += count
            if (index + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{settings["epochs"]}, batch {index + 1}/{len(loader)}, loss={total[0] / examples:.4f}', flush=True)
        scheduler.step()
        row = dict(epoch=epoch + 1, learning_rate=lr, train_loss=float(total[0] / examples),
                   classification_loss=float(total[1] / examples), regression_loss=float(total[2] / examples),
                   train_frames=examples, train_seconds=time.perf_counter() - started, car_bev_r40_moderate=None)
        # Save immediately after the training epoch, before potentially long validation.
        pending = checkpoint_payload(model, optimizer, scheduler, scaler, epoch + 1, best_ap, identity, history + [row])
        pending['validation_pending'] = True
        save_checkpoint(latest, pending)
        del pending
        history.append(row)
        best_ap = finalize_epoch(model, optimizer, scheduler, scaler, settings, assets, history, best_ap, identity)
    print(f'Training complete: {settings["epochs"]} TOTAL epochs; best validation Car Moderate R40={best_ap:.3f}', flush=True)


def evaluate(settings):
    import torch
    configure_runtime(settings['seed'])
    reference_check(settings)
    _, output = paths(settings)
    assets = load_assets(settings)
    choice = settings['eval_checkpoint']
    checkpoint = output / 'checkpoints' / (choice + '.pth') if choice in ('best', 'last') else Path(choice).expanduser()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('identity') != training_identity(settings, assets['asset_signature']):
        raise ValueError('Evaluation checkpoint differs from this run identity')
    model = build_model().cuda()
    model.load_state_dict(state['model_state'], strict=True)
    epoch, best_ap = state['epoch'], state['best_ap']
    del state
    destination = output / 'evaluation' / checkpoint.stem
    report = evaluate_model(model, settings, assets, destination, profile=True)
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=common.sha256(checkpoint), epoch=epoch,
                  selection_best_car_moderate_r40=best_ap, split_sha256=common.SPLIT_HASHES)
    common.write_json(destination / 'report.json', report)
    with (destination / 'comparison_car_bev.csv').open('w', newline='') as stream:
        row = dict(model='PIXOR reproduction', epoch=epoch, parameters=report['parameters'],
            **{'car_bev_r40_' + k.lower(): v for k, v in report['accuracy']['R40'].items()},
            network_ms=report['latency']['network']['mean_ms'], network_fps=report['latency']['network']['fps'],
            pipeline_ms=report['latency']['input_to_detections']['mean_ms'], pipeline_fps=report['latency']['input_to_detections']['fps'])
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('setup', 'prepare', 'smoke', 'train', 'evaluate'))
    parser.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    settings = read_settings(args.settings)
    globals()[args.command](settings)


if __name__ == '__main__':
    main()

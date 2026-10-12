#!/usr/bin/env python3
"""Read-only configured KITTI/GT database audit for Colab benchmark provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/kitti_training_pipeline')]
from common import read_json, sha256, write_json
from prepare_kitti import parse_calibration, read_ids
from tools.kitti_training_pipeline.kitti_box_conversion import read_calibration
from tools.kitti_training_pipeline.evaluate_kitti_3d import read_annotation, png_image_size


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def repo_path(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def checked_ids(path):
    ids = read_ids(path)
    if not ids or len(ids) != len(set(ids)) or any(not i.isdigit() for i in ids):
        raise ValueError(f'Split must contain unique nonempty numeric IDs: {path}')
    return ids


def audit_assets(config_path, repo_root=ROOT, *, kitti_root, metric_mode='local_bev', database_source_split=None):
    if metric_mode not in {'local_bev', 'bev', '3d'}:
        raise ValueError('metric_mode must be local_bev, bev or 3d')
    config_path = repo_path(config_path, repo_root)
    config = read_json(config_path)
    processed = repo_path(config['data']['kitti']['location'], repo_root)
    raw = repo_path(kitti_root, repo_root)
    if (raw / 'training').is_dir():
        raw = raw / 'training'
    splits = {key: repo_path(config[key]['data'], repo_root) for key in ('train', 'val')}
    ids = {key: checked_ids(path) for key, path in splits.items()}
    train_ids = set(ids['train'])
    if train_ids & set(ids['val']):
        raise ValueError('Train/validation IDs must be disjoint')
    classes = config['data']['kitti']['objects']
    files, labels = {}, {}

    def record(path):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        files[str(path)] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}

    def points(path, count=None):
        path = path.resolve()
        size = path.stat().st_size
        if size % 16 or (count is not None and size != count * 16):
            raise ValueError(f'Malformed float32 point/crop file size: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                if not np.isfinite(np.frombuffer(block, dtype=np.float32)).all():
                    raise ValueError(f'Non-finite point/crop data: {path}')
                digest.update(block)
        files[str(path)] = {'path': str(path), 'bytes': size, 'sha256': digest.hexdigest()}

    for identifier in ids['train'] + ids['val']:
        points(processed / 'pointcloud' / f'{identifier}.bin')
        path = processed / 'label' / f'{identifier}.txt'
        boxes = []
        for line in path.read_text().splitlines():
            fields = line.split()
            if not fields:
                boxes.append(None)
                continue
            if len(fields) != 8 or fields[0] not in classes:
                raise ValueError(f'Malformed/unknown processed label: {path}')
            box = np.array(list(map(float, fields[1:])), dtype=np.float32)
            if not np.isfinite(box).all() or (box[:3] <= 0).any():
                raise ValueError(f'Invalid processed label dimensions: {path}')
            boxes.append((fields[0], box))
        labels[identifier] = boxes
        record(path)
    for identifier in ids['val']:
        path = raw / 'label_2' / f'{identifier}.txt'
        read_annotation(path)
        record(path)
        path = raw / 'calib' / f'{identifier}.txt'
        transform = parse_calibration(path)
        if not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-8:
            raise ValueError(f'Non-finite/singular calibration: {path}')
        if metric_mode != 'local_bev':
            read_calibration(path)
        record(path)
        if metric_mode != 'local_bev':
            path = raw / 'image_2' / f'{identifier}.png'
            if min(png_image_size(path)) <= 0:
                raise ValueError(f'Invalid evaluation image: {path}')
            record(path)
    source = repo_path(database_source_split, repo_root) if database_source_split else splits['train']
    if set(checked_ids(source)) != train_ids:
        raise ValueError('Database source manifest must contain the same training IDs')
    source_hash = sha256(source)
    record(source)
    for path in splits.values():
        record(path)
    databases = []
    for operation in config.get('augmentation', {}).get('AUG_CONFIG_LIST', []):
        if operation.get('NAME') not in {'gt_sampling', 'hybrid_gt_sampling'}:
            continue
        for relative in operation.get('DB_INFO_PATH', []):
            path = (processed / relative).resolve()
            db = read_json(path)
            expected = {'format': 'lidar_gt_database_v1', 'num_point_features': 4,
                'box_format': 'x_y_z_center_length_width_height_yaw',
                'point_coordinates': 'relative_to_box_center_without_yaw_rotation'}
            if any(db.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Unsupported database convention: {path}')
            source_ids = db.get('source_frame_ids', [])
            if len(source_ids) != len(set(source_ids)) or set(source_ids) != train_ids:
                raise ValueError('Database source frames must be training-only and cover the training split')
            if db.get('source_manifest_sha256') != source_hash:
                raise ValueError('Database source manifest hash does not match; supply its original source manifest explicitly')
            infos = db.get('db_infos', {})
            if set(infos) != set(classes) or set(db.get('counts', {})) != set(classes):
                raise ValueError('Database class/count metadata does not match configured classes')
            for name, entries in infos.items():
                if type(db['counts'][name]) is not int or db['counts'][name] != len(entries):
                    raise ValueError('Database class count does not match entries')
                for entry in entries:
                    identifier, index = entry.get('image_idx'), entry.get('gt_idx')
                    if entry.get('name') != name or identifier not in train_ids:
                        raise ValueError('Database entry must match its class and training source ID')
                    if type(index) is not int or index < 0 or index >= len(labels[identifier]):
                        raise ValueError('Database source label index is invalid')
                    source_label = labels[identifier][index]
                    if source_label is None or source_label[0] != name:
                        raise ValueError('Database source label class does not match')
                    h, w, length, x, y, z, yaw = source_label[1]
                    expected_box = np.array([x, y, z + h / 2, length, w, h, yaw])
                    box = np.asarray(entry.get('box3d_lidar'), dtype=float)
                    if box.shape != (7,) or not np.isfinite(box).all() or not np.allclose(box, expected_box, atol=1e-5, rtol=1e-5):
                        raise ValueError('Database geometry does not match its processed source label')
                    count = entry.get('num_points_in_gt')
                    if type(count) is not int or count < 1:
                        raise ValueError('Database crop count must be a positive integer')
                    points(processed / entry['path'], count)
            record(path)
            databases.append({'path': str(path), 'sha256': sha256(path), 'counts': db['counts'],
                              'source_manifest_sha256': source_hash})
        if not operation.get('DB_INFO_PATH'):
            raise ValueError('Configured GT sampler requires DB_INFO_PATH')
    inventory = [files[key] for key in sorted(files)]
    return {'version': 1, 'type': 'kitti_assets_audit', 'status': 'passed', 'metric_mode': metric_mode,
        'config_path': str(config_path), 'resolved_config_sha256': canonical_hash(config),
        'processed_root': str(processed), 'kitti_root': str(raw),
        'database_source_split': str(source),
        'splits': {key: {'path': str(path), 'count': len(ids[key]), 'sha256': sha256(path)} for key, path in splits.items()},
        'checks': {'all_points_finite': True, 'processed_labels_valid': True,
            'original_validation_labels_calibration': True, 'reference_projection_inputs': metric_mode != 'local_bev',
            'database_source_labels_match': True, 'train_only_database': True},
        'databases': databases, 'asset_files': inventory, 'inventory_sha256': canonical_hash(inventory),
        'limits': 'Checks finite point/crop contents and database source-label geometry; does not match every crop point to its original scene or authenticate KITTI origin.'}


def verify_asset_audit(report, config_path, repo_root=ROOT, *, metric_mode):
    if report.get('version') != 1 or report.get('type') != 'kitti_assets_audit' or report.get('status') != 'passed':
        raise ValueError('Incomplete asset audit report')
    current = audit_assets(config_path, repo_root, kitti_root=report['kitti_root'], metric_mode=metric_mode,
                           database_source_split=report.get('database_source_split'))
    if report != current:
        raise ValueError('Asset audit is stale or does not match current config/files/mode')
    return current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--kitti-root', required=True, type=Path)
    parser.add_argument('--metric-mode', choices=('local_bev', 'bev', '3d'), default='local_bev')
    parser.add_argument('--database-source-split', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit_assets(args.config, args.repo_root, kitti_root=args.kitti_root,
                          metric_mode=args.metric_mode, database_source_split=args.database_source_split)
    write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'splits': result['splits'], 'databases': result['databases']}, indent=2))
    return result


if __name__ == '__main__':
    main()

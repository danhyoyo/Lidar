#!/usr/bin/env python3
"""Verify full evaluator output and loss/AP checkpoint selection provenance."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/kitti_training_pipeline')]
from common import (checkpoint_identity, detection_spec, read_json, sha256, write_json,
                    evaluation_asset_hashes, verify_retained_checkpoint)
from tools.benchmarks.audit_kitti_assets import canonical_hash, checked_ids, repo_path

CLASSES = ('Car', 'Pedestrian', 'Cyclist')
DIFFICULTIES = ('Easy', 'Moderate', 'Hard')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, *, percent=False):
    require(type(value) in (int, float) and math.isfinite(value), 'Metric/loss must be finite numeric data')
    require(not percent or 0 <= value <= 100, 'AP must be a percentage between 0 and 100')
    return float(value)


def normalized_metrics(report, mode):
    accuracy = report['accuracy']
    output = {}
    for sampling in (['R40'] if mode == 'local_bev' else ['R11', 'R40']):
        source = accuracy if mode == 'local_bev' else accuracy[sampling]
        require(set(source['per_class']) == set(CLASSES), 'Every benchmark class must be present')
        values = {}
        for name in CLASSES:
            cells = source['per_class'][name]
            if mode == 'local_bev':
                cells = cells['difficulties']
            require(set(cells) == set(DIFFICULTIES), 'Every difficulty must be present')
            values[name] = {}
            for difficulty in DIFFICULTIES:
                if mode == 'local_bev':
                    cell = cells[difficulty]
                    count, value = cell['ground_truth'], cell['ap_r40_percent']
                else:
                    count, value = source['valid_gt_counts'][name][difficulty], cells[difficulty]
                require(type(count) is int and count > 0, 'Missing valid GT classes/difficulties cannot establish a full benchmark baseline')
                values[name][difficulty] = finite_number(value, percent=True)
        moderate = sum(values[name]['Moderate'] for name in CLASSES) / 3
        ap9 = sum(value for row in values.values() for value in row.values()) / 9
        require(math.isclose(finite_number(source['map_moderate_percent'], percent=True), moderate, abs_tol=1e-8),
                'Moderate macro does not include all three classes correctly')
        require(math.isclose(finite_number(source['mean_ap_9_percent'], percent=True), ap9, abs_tol=1e-8),
                'AP9 macro does not include all nine cells correctly')
        output[sampling] = {'per_class': values, 'map_moderate_percent': moderate, 'mean_ap_9_percent': ap9}
    return output


def record_evidence(protocol_path, evaluation_path, selection_path, history_path, *, synthetic=False):
    """Validate trusted execution records against their actual files and config."""
    from tools.kitti_training_pipeline.evaluate_kitti_bev import DIFFICULTIES as local_difficulties, NEIGHBOURS
    import torch
    paths = [Path(p).expanduser().resolve() for p in (protocol_path, evaluation_path, selection_path, history_path)]
    protocol, report, selection = map(read_json, paths[:3])
    require(report.get('status') == 'ok', 'Evaluation must have succeeded')
    require(type(synthetic) is bool, 'synthetic must be a boolean')
    config_path = Path(protocol['source_config']).resolve()
    config = read_json(config_path)
    require(sha256(config_path) == protocol['source_config_sha256'] and canonical_hash(config) == protocol['resolved_config_sha256'],
            'Protocol config is stale or does not match its source')
    require(checkpoint_identity(config) == protocol['checkpoint_identity'], 'Protocol checkpoint identity does not match config')
    require(sha256(repo_path(config['train']['data'], ROOT)) == protocol['splits']['train']['sha256'], 'Training split changed')
    mode = protocol['evaluation']['metric_mode']
    require(mode in {'local_bev', 'bev', '3d'}, 'Unknown evaluation mode')
    data, actual = report['data'], report['protocol']
    frame_ids = checked_ids(repo_path(config['val']['data'], ROOT))
    require(data['frames'] == len(frame_ids) and data['split_sha256'] == protocol['splits']['val']['sha256'],
            'Evaluation must cover the exact full validation split')
    require(sha256(repo_path(config['val']['data'], ROOT)) == data['split_sha256'], 'Validation split changed')
    expected_hash = canonical_hash(config) if mode == 'local_bev' else sha256(config_path)
    require(data['resolved_config_sha256'] == expected_hash, 'Evaluator config hash does not match baseline config')
    input_assets = evaluation_asset_hashes(config, frame_ids, data['kitti_root'], reference=mode != 'local_bev')
    require(data.get('input_asset_sha256') == input_assets, 'Evaluator input assets changed or input provenance is missing')
    require(actual['classes'] == list(CLASSES) and actual['iou_thresholds'] == protocol['evaluation']['iou_thresholds'],
            'Evaluator classes/IoU thresholds do not match protocol')
    decode = protocol['decode']
    for key in ('score_threshold', 'nms_threshold', 'nms_alpha'):
        require(actual[key] == decode[key], f'Evaluator {key} does not match protocol')
    if mode == 'local_bev':
        require(actual.get('name') == 'local loader-aligned KITTI-style rotated BEV AP R40' and actual.get('box_mode') == 'bev',
                'Local BEV evaluator/mode does not match protocol')
        require(actual['roi_rule'] == 'strict LiDAR-frame center inside configured x/y ROI' and
                protocol['evaluation']['domain'] == 'strict configured XY-center ROI', 'Local BEV ROI domain does not match')
        require(actual['difficulty_rules'] == local_difficulties, 'Local difficulty rules do not match')
        require(actual['neighbour_class_ignores'] == {key: sorted(value) for key, value in NEIGHBOURS.items()}, 'Local neighbor ignore rules do not match')
        require(actual['peak_mode'] == decode['peak_mode'] and actual['max_detections_per_frame'] == decode['max_detections'] and
                actual['max_detections_scope'] == decode['cap_scope'], 'Evaluator peak mode/detection cap does not match')
        require(data['checkpoint_identity'] == checkpoint_identity(config), 'Evaluator checkpoint identity does not match')
        detection = detection_spec(config)
        quality = config['loss'].get('iou_target_type', 'mgiou') if detection.use_iou else detection.quality_target
        require(actual['quality_target'] == quality, 'Evaluator quality target does not match config')
        require(data['full_split_frames'] == len(frame_ids), 'Partial local evaluation cannot establish baseline evidence')
        reference = None
    else:
        from tools.kitti_training_pipeline.evaluate_kitti_3d import reference_provenance
        reference = reference_provenance()
        require(actual['metric_mode'] == mode and actual['ap_samplings'] == ['R11', 'R40'] and actual['domain'] == 'full benchmark GT',
                'Reference mode/sampling/domain does not match protocol')
        require(actual['reference']['revision'] == reference['revision'] and actual['reference']['sha256'] == reference['sha256'],
                'Reference evaluator revision/hashes do not match')
        require(actual['difficulties'] == list(DIFFICULTIES) and actual['aos'] is False and
                actual['ignored_neighbor_dontcare'] == 'unchanged pinned reference, metric-specific', 'Reference difficulty/ignore/orientation policy does not match')
        require(actual['max_detections'] == decode['max_detections'] and actual['nms_metric'] == 'classwise BEV footprint; one global cap',
                'Reference detection cap/NMS does not match')
        require(data['frame_ids'] == frame_ids and actual['checkpoint_identity'] == checkpoint_identity(config),
                'Reference frame IDs/checkpoint identity do not match')
    metrics = normalized_metrics(report, mode)
    checkpoint = Path(report['model']['path']).resolve()
    require(report['model']['backend'] == 'pytorch', 'Evidence currently requires an actual PyTorch detector checkpoint')
    require(sha256(checkpoint) == report['model']['sha256'], 'Evaluation checkpoint hash changed or does not match')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(saved.get('checkpoint_identity') == checkpoint_identity(config) and
            checkpoint_identity(saved['config']) == checkpoint_identity(config), 'Selected checkpoint identity does not match baseline config')
    from tools.kitti_training_pipeline.checkpoint_selection import selection_protocol, verify_selection
    expected_selection = selection_protocol(config)
    require(all(protocol['checkpoint_selection'].get(key) == value for key, value in expected_selection.items()),
            'Checkpoint selection protocol does not match the resolved config')
    rows = [json.loads(line) for line in paths[3].read_text().splitlines() if line.strip()]
    epochs = config['train']['epochs']
    verify_selection(config, selection, rows, saved)
    if expected_selection['policy'] == 'maximum validation AP' and expected_selection['metric_mode'] == mode:
        require(math.isclose(metrics['R40']['map_moderate_percent'], selection['score'], abs_tol=1e-8),
                'Re-evaluated checkpoint AP differs from its training selection score')
    retained = Path(selection['checkpoint']).resolve()
    verify_retained_checkpoint(checkpoint, retained, selected_state=saved)
    sources = list(dict.fromkeys(paths + [config_path, checkpoint, retained, repo_path(config['train']['data'], ROOT),
                                         repo_path(config['val']['data'], ROOT)] + [Path(path) for path in input_assets]))
    raw = Path(data['kitti_root']).resolve()
    if (raw / 'training').is_dir():
        raw = raw / 'training'
    return {'version': 1, 'type': 'benchmark_evaluation_evidence', 'status': 'passed', 'synthetic': synthetic,
        'metric_mode': mode, 'ap_samplings': protocol['evaluation']['ap_samplings'], 'name': report.get('name'),
        'protocol_path': str(paths[0]), 'evaluation_path': str(paths[1]), 'selection_path': str(paths[2]), 'history_path': str(paths[3]),
        'protocol_sha256': sha256(paths[0]), 'resolved_config_sha256': canonical_hash(config),
        'checkpoint_sha256': sha256(checkpoint), 'checkpoint_selection_verified': True,
        'selection_policy': expected_selection['policy'], 'training_epochs': epochs, 'full_split_verified': True,
        'validation_split_sha256': data['split_sha256'], 'processed_root': str(repo_path(config['data']['kitti']['location'], ROOT)),
        'kitti_root': str(raw), 'metrics': metrics, 'reference': reference,
        'source_files': [{'path': str(path), 'sha256': sha256(path)} for path in sources],
        'limits': 'Trusted evaluator/training records with file freshness and consistency checks; not authenticated execution, genuine-KITTI certification, or SOTA evidence.'}


def verify_evidence(report):
    require(report.get('version') == 1 and report.get('type') == 'benchmark_evaluation_evidence' and
            report.get('status') == 'passed', 'Incomplete reproduced evaluation evidence')
    for source in report['source_files']:
        require(sha256(Path(source['path'])) == source['sha256'], 'Evidence source file is stale or changed')
    current = record_evidence(report['protocol_path'], report['evaluation_path'], report['selection_path'], report['history_path'],
                              synthetic=report['synthetic'])
    require(current == report, 'Evidence record is stale or does not match its source files')
    return current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--history', required=True, type=Path)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = record_evidence(args.protocol, args.evaluation, args.selection, args.history, synthetic=args.synthetic)
    write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'mode': result['metric_mode'], 'metrics': result['metrics'],
                      'synthetic': result['synthetic']}, indent=2))
    return result


if __name__ == '__main__':
    main()

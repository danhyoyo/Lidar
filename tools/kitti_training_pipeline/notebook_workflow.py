"""Checked building blocks for the one-master-config Colab workflow."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from .common import checkpoint_identity, read_json, sha256, write_json, verify_retained_checkpoint
except ImportError:
    from common import checkpoint_identity, read_json, sha256, write_json, verify_retained_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def source_preflight(repo_root):
    """Fail early when an old remote branch lacks the uploaded implementation."""
    root = Path(repo_root)
    required = ['configs/config.json', 'tools/kitti_training_pipeline/notebook_config.py',
        'tools/kitti_training_pipeline/notebook_workflow.py', 'tools/benchmarks/audit_kitti_assets.py',
        'tools/benchmarks/benchmark_evidence.py', 'tools/benchmarks/benchmark_protocol.py',
        'tools/benchmarks/smoke_detector.py', 'tools/kitti_training_pipeline/evaluate_kitti_3d.py',
        'detector/core/models/heads/grouped.py', 'detector/core/models/backbones/focal_context.py']
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError('Selected checkout is missing current tools. Publish the intended branch or '
                                'upload the complete checkout and use SOURCE_MODE="existing". Missing: '+', '.join(missing))
    return True


def validate_modes(config, modes):
    if not isinstance(modes, (list, tuple)) or not modes or len(set(modes)) != len(modes):
        raise ValueError('EVALUATION_MODES must be a nonempty list of unique modes')
    box = config['data'].get('box_mode', 'bev')
    allowed = {'local_bev'} if box == 'bev' else {'3d', 'bev'} if box == '3d' else set()
    if not set(modes) <= allowed:
        raise ValueError('BEV checkpoints use local_bev; final 3d/bev reference metrics require a trained 3D config')
    return tuple(modes)


def selected_run(run_dir):
    """Use this run's resolved config and complete minimum-loss selection only."""
    import torch
    root = Path(run_dir).expanduser().resolve()
    config_path, checkpoint = root/'config.resolved.json', root/'selected/best.pt'
    selection_path, history_path = root/'selected/selection.json', root/'metrics.jsonl'
    for path in (config_path, checkpoint, selection_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required selected-run artifact is missing: {path}; no last.pt fallback')
    config, selection = read_json(config_path), read_json(selection_path)
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if [row['epoch'] for row in rows] != list(range(1, config['train']['epochs']+1)):
        raise ValueError('Selected-run evaluation requires complete ordered configured training history')
    running_best = math.inf
    for row in rows:
        loss = row['validation']['loss']
        if (type(loss) not in (int, float) or not math.isfinite(loss) or
                type(row['optimizer_updates']) is not int or row['optimizer_updates'] < 1 or
                row['retained'] is not (loss < running_best) or
                row['precision'] != config['train']['precision'] or row['loss'] != config['loss']['name']):
            raise ValueError('Training history must have finite loss, successful updates and consistent retention/runtime')
        running_best = min(running_best, loss)
    best = min(rows, key=lambda row: row['validation']['loss'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (selection['criterion'] != 'minimum mean validation loss' or selection['epoch'] != best['epoch'] or
            saved['epoch'] != best['epoch'] or saved.get('checkpoint_identity') != checkpoint_identity(config) or
            checkpoint_identity(saved['config']) != checkpoint_identity(config)):
        raise ValueError('Selected checkpoint/config identity or minimum-loss epoch does not match this run')
    for loss in (selection['validation_objective'], saved['validation']['loss'], saved['best_validation_objective']):
        if type(loss) not in (int, float) or not math.isfinite(loss) or not math.isclose(loss, best['validation']['loss'], abs_tol=1e-8):
            raise ValueError('Selected checkpoint loss differs from complete training history')
    retained = Path(selection['checkpoint']).expanduser().resolve()
    verify_retained_checkpoint(checkpoint, retained, selected_state=saved)
    return {'run_dir': root, 'config_path': config_path, 'config': config, 'checkpoint': checkpoint,
            'selection_path': selection_path, 'history_path': history_path, 'completed_epochs': len(rows)}


def evaluation_command(run, mode, raw_root, output, *, repo_root=ROOT, device='cuda'):
    config = run['config']; validate_modes(config, [mode])
    settings = config.get('evaluation', {})
    root = Path(repo_root).resolve()
    raw = Path(raw_root).expanduser().resolve()
    if mode == 'local_bev' and (raw / 'label_2').is_dir():
        raw = raw.parent  # Historical local loader appends training/ itself.
    split = Path(config['val']['data']).expanduser()
    if not split.is_absolute(): split = root / split
    script = 'evaluate_kitti_bev.py' if mode == 'local_bev' else 'evaluate_kitti_3d.py'
    command = [sys.executable, str(root/'tools/kitti_training_pipeline'/script),
        '--name', run['run_dir'].name, '--config', str(run['config_path']),
        '--detector-root', str(root/'detector'), '--kitti-root', str(raw), '--split', str(split),
        '--device', device, '--output', str(output),
        '--score-threshold', str(settings.get('score_threshold', .05)),
        '--nms-threshold', str(settings.get('nms_threshold', .10)),
        '--max-detections', str(settings.get('max_detections', 500))]
    if mode == 'local_bev': command.extend(['--backend', 'pytorch', '--model', str(run['checkpoint'])])
    else: command.extend(['--checkpoint', str(run['checkpoint']), '--metric-mode', mode])
    return command


def training_command(config_path, output_root, run_name, *, repo_root=ROOT, resume=None, warm_start=None, device='cuda'):
    if resume is not None and warm_start is not None:
        raise ValueError('Resume and backbone warm-start are mutually exclusive')
    config = read_json(Path(config_path)); training = config['train']
    command = [sys.executable, str(Path(repo_root)/'tools/kitti_training_pipeline/train.py'),
        '--config', str(config_path), '--detector-root', str(Path(repo_root)/'detector'),
        '--output-root', str(output_root), '--run-name', run_name, '--device', device,
        '--num-workers', str(training['num_workers']), '--target-backend', training['target_backend'],
        '--precision', training['precision']]
    if training.get('compile_model', False): command.append('--compile-model')
    if resume is not None: command.extend(['--resume', str(resume)])
    if warm_start is not None: command.extend(['--warm-start', str(warm_start)])
    return command


def evaluation_rows(report):
    """Render exact reported metrics; missing/failed results never become zero."""
    if report.get('status') != 'ok': raise ValueError('Evaluation failed; no accuracy row is available')
    mode = report['protocol'].get('metric_mode', 'local_bev')
    rows = []
    for sampling in (['R40'] if mode == 'local_bev' else ['R11', 'R40']):
        accuracy = report['accuracy'] if mode == 'local_bev' else report['accuracy'][sampling]
        row = {'name': report.get('name'), 'metric': mode, 'sampling': sampling,
               'map_moderate_percent': accuracy['map_moderate_percent'], 'mean_ap_9_percent': accuracy['mean_ap_9_percent']}
        for name in ('Car', 'Pedestrian', 'Cyclist'):
            values = accuracy['per_class'][name]
            for difficulty in ('Easy', 'Moderate', 'Hard'):
                row[f'{name}_{difficulty}'] = (values['difficulties'][difficulty]['ap_r40_percent']
                                               if mode == 'local_bev' else values[difficulty])
        rows.append(row)
    return rows


def require_training_ready(purpose, protocols):
    if purpose not in {'development', 'benchmark'}: raise ValueError('RUN_PURPOSE must be development or benchmark')
    if purpose == 'benchmark' and (not protocols or any(
            record.get('long_training_allowed') is not True or record.get('pending_gates') != [] for record in protocols)):
        raise ValueError('Benchmark training requires all selected protocols frozen with actual data/GPU/baseline evidence')
    return True


def validate_comparator(candidate, baseline, *, kind='focal_ablation'):
    """A focal ablation changes context alone; general protocol comparisons are explicit."""
    if kind not in {'focal_ablation', 'protocol'}:
        raise ValueError('COMPARISON_KIND must be focal_ablation or protocol')
    if kind == 'protocol':
        return True
    if candidate['model'].get('c4_context', 'none') != 'focal' or baseline['model'].get('c4_context', 'none') != 'none':
        raise ValueError('Focal ablation requires a focal candidate and a no-context baseline')
    identities = [checkpoint_identity(config) for config in (candidate, baseline)]
    for identity in identities:
        features = identity['model'].get('features', {}).get('config', {})
        for key in list(features):
            if key.startswith('c4_context'):
                del features[key]
    mismatches = [key for key in identities[0] if identities[0][key] != identities[1][key]]
    for key in ('seed', 'augmentation'):
        if candidate.get(key) != baseline.get(key):
            mismatches.append(key)
    for key, default in (('num_workers', 2), ('target_backend', 'python'), ('compile_model', False)):
        if candidate['train'].get(key, default) != baseline['train'].get(key, default):
            mismatches.append('train.'+key)
    if candidate['data']['bev_encoding'].get('backend', 'numpy') != baseline['data']['bev_encoding'].get('backend', 'numpy'):
        mismatches.append('encoder_backend')
    if mismatches:
        raise ValueError(f'Focal ablation conditions must match outside context: {mismatches}')
    return True


def verify_reference_runtime(repo_root, output):
    """Run all sixteen pinned wrapper/direct-CUDA parity scenarios on this runtime."""
    try:
        from .evaluate_kitti_3d import reference_provenance
    except ImportError:
        from evaluate_kitti_3d import reference_provenance
    root, output = Path(repo_root).resolve(), Path(output).resolve()
    suite = root/'tests/test_kitti_3d_evaluation.py'
    if not suite.is_file(): raise FileNotFoundError(suite)
    output.parent.mkdir(parents=True, exist_ok=True)
    xml = output.with_suffix('.xml'); log = output.with_suffix('.log')
    command = [sys.executable, '-m', 'pytest', str(suite), '-q', '-k',
               'wrapper_matches_pinned_reference_for_ignored_difficulty_overlap_and_sampling', f'--junitxml={xml}']
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=180)
    log.write_text(result.stdout+result.stderr)
    cases = list(ET.parse(xml).getroot().iter('testcase')) if xml.is_file() else []
    skipped = sum(case.find('skipped') is not None for case in cases)
    errors = sum(case.find('error') is not None or case.find('failure') is not None for case in cases)
    if result.returncode != 0 or len(cases) != 16 or skipped or errors:
        raise RuntimeError(f'Actual reference CUDA parity failed or skipped; see {log}. No local BEV substitute.')
    receipt = {'status': 'passed', 'scope': '16 selected wrapper/direct pinned reference CUDA parity cases',
        'reference_provenance': reference_provenance(), 'gpu_reference_cases': len(cases),
        'full': {'exit_code': result.returncode, 'skips': skipped, 'passed': len(cases)},
        'command': command, 'log': str(log), 'log_sha256': sha256(log), 'junit_sha256': sha256(xml)}
    write_json(output, receipt)
    return receipt

"""Actual CPU KITTI-format training, periodic AP, resume and notebook evidence."""
import copy
import json
from pathlib import Path

import pytest
import torch

from benchmark_fixtures import ROOT, asset_fixture
import train
from common import checkpoint_identity, write_json
from tools.kitti_training_pipeline import notebook_workflow
from tools.benchmarks.benchmark_protocol import freeze_protocol, compare_protocols
from tools.benchmarks.benchmark_evidence import record_evidence


def setup(directory, *, primary='ap', epochs=3, interval=2):
    config, path, _, raw = asset_fixture(directory)
    config['train'].update(epochs=epochs, save_every=2,
        checkpoint_selection={'primary': primary, 'ap_every': interval, 'metric_mode': 'auto'})
    config['evaluation'] = {'kitti_root': str(raw), 'score_threshold': .05,
        'nms_threshold': .10, 'max_detections': 500}
    write_json(path, config)
    args = ['--config', str(path), '--detector-root', str(ROOT/'detector'),
        '--output-root', str(directory/'runs'), '--run-name', 'test', '--device', 'cpu',
        '--num-workers', '0', '--target-backend', 'python', '--precision', 'fp32']
    return config, path, raw, directory/'runs/test', args


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def history(run):
    return [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]


def test_actual_training_saves_dual_winners_and_full_split_ap_evidence(tmp_path, capsys):
    config, path, raw, run, args = setup(tmp_path)
    train.main(args)
    rows = history(run)
    assert [r['epoch'] for r in rows if r.get('ap_validation') is not None] == [2, 3], 'Periodic AP was not run'
    assert all(r['validation']['samples'] == 1 for r in rows)
    ap_rows = [r for r in rows if r['ap_validation'] is not None]
    best_ap = max(ap_rows, key=lambda r: r['ap_validation']['score'])
    best_loss = min(rows, key=lambda r: r['validation']['loss'])
    assert load(run/'selected/best_ap.pt')['epoch'] == best_ap['epoch']
    assert load(run/'selected/best_loss.pt')['epoch'] == best_loss['epoch']
    chosen = notebook_workflow.selected_run(run)
    assert chosen['checkpoint'] == run/'selected/best_ap.pt'
    assert load(run/'checkpoints/last.pt')['ap_selection']['best_epoch'] == best_ap['epoch']
    output = capsys.readouterr().out
    assert 'AP validation' in output and 'BEST LOSS' in output and 'BEST AP' in output

    from tools.kitti_training_pipeline.evaluate_kitti_bev import run_evaluation
    evaluation = run/'evaluation.json'
    run_evaluation(name='actual_ap', backend='pytorch', model_path=chosen['checkpoint'],
        config_path=chosen['config_path'], detector_root=ROOT/'detector', kitti_root=raw.parent,
        split_path=Path(config['val']['data']), output_path=evaluation, device='cpu', warmup_frames=0, progress_every=0)
    report = json.loads(evaluation.read_text())
    assert report['accuracy']['map_moderate_percent'] == best_ap['ap_validation']['score']
    protocol = run/'protocol.json'
    write_json(protocol, freeze_protocol(chosen['config_path'], metric_mode='local_bev'))
    evidence = record_evidence(protocol, evaluation, chosen['selection_path'], chosen['history_path'], synthetic=True)
    assert evidence['selection_policy'] == 'maximum validation AP' and evidence['checkpoint_selection_verified']
    corrupted = copy.deepcopy(rows)
    for row in corrupted:
        if row['ap_validation'] is not None and row['epoch'] != best_ap['epoch']:
            row['ap_validation']['frames'] = 99
    chosen['history_path'].write_text(''.join(json.dumps(r)+'\n' for r in corrupted))
    with pytest.raises(ValueError, match='split|frames'):
        notebook_workflow.selected_run(run)


def test_ap_evaluation_does_not_change_training_rng_or_model_updates(tmp_path):
    _, _, _, ap_run, ap_args = setup(tmp_path/'ap', epochs=2, interval=1)
    _, _, _, loss_run, loss_args = setup(tmp_path/'loss', primary='loss', epochs=2, interval=1)
    train.main(ap_args); train.main(loss_args)
    a, b = load(ap_run/'checkpoints/last.pt'), load(loss_run/'checkpoints/last.pt')
    assert a.get('ap_selection', {}).get('best_score') is not None, 'AP evaluator did not run'
    for key in a['model_state_dict']:
        assert torch.equal(a['model_state_dict'][key], b['model_state_dict'][key]), key
    assert a['validation']['loss'] == b['validation']['loss']
    assert [r['train_objective'] for r in history(ap_run)] == [r['train_objective'] for r in history(loss_run)]


def test_resume_keeps_ap_winner_schedule_and_exact_cpu_training_state(tmp_path, monkeypatch):
    _, _, _, full, full_args = setup(tmp_path/'full', epochs=4, interval=2)
    _, _, _, resumed, resume_args = setup(tmp_path/'resume', epochs=4, interval=2)
    train.main(full_args)
    original = train.validate
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt('synthetic interruption before epoch 4 save')
        return original(*args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(train, 'validate', interrupt)
        with pytest.raises(KeyboardInterrupt): train.main(resume_args)
    boundary = load(resumed/'checkpoints/last.pt')
    assert boundary['epoch'] == 3 and boundary.get('ap_selection', {}).get('best_epoch') == 2, 'AP resume state missing'
    train.main(resume_args + ['--resume', str(resumed/'checkpoints/last.pt')])
    a, b = load(full/'checkpoints/last.pt'), load(resumed/'checkpoints/last.pt')
    assert a['ap_selection'] == b['ap_selection']
    for key in a['model_state_dict']:
        assert torch.equal(a['model_state_dict'][key], b['model_state_dict'][key]), key
    assert [r['epoch'] for r in history(resumed) if r['ap_validation'] is not None] == [2, 4]
    notebook_workflow.selected_run(resumed)


def test_partial_smoke_cannot_claim_best_ap(tmp_path):
    _, _, _, _, args = setup(tmp_path)
    with pytest.raises(ValueError, match='AP.*full|full.*AP'):
        train.main(args + ['--max-val-batches', '1'])


def test_ap_policy_and_decode_changes_are_resume_incompatible(tmp_path):
    config, _, _, _, _ = setup(tmp_path)
    base = checkpoint_identity(config)
    for change in ('interval', 'decode'):
        other = copy.deepcopy(config)
        if change == 'interval': other['train']['checkpoint_selection']['ap_every'] = 1
        else: other['evaluation']['nms_threshold'] = .2
        assert checkpoint_identity(other) != base, 'AP selection policy must be part of resume identity'


def test_comparator_requires_identical_ap_selection_interval(tmp_path):
    config, path, _, _, _ = setup(tmp_path)
    a = freeze_protocol(path, metric_mode='local_bev')
    other = copy.deepcopy(config); other['train']['checkpoint_selection']['ap_every'] = 1
    other_path = tmp_path/'other.json'; write_json(other_path, other)
    b = freeze_protocol(other_path, metric_mode='local_bev')
    assert not compare_protocols(a, b)['matched'], 'AP cadence mismatch was accepted'


def test_failed_ap_evaluation_restores_training_rng_and_removes_temporary_checkpoint(tmp_path, monkeypatch):
    import random
    import numpy as np
    import evaluate_kitti_bev
    config, path, _, run, _ = setup(tmp_path)
    run.mkdir(parents=True)
    generator = torch.Generator().manual_seed(42)
    before = train.capture_rng_state(generator)
    def fail(**kwargs):
        random.random(); np.random.random(); torch.rand(3)
        assert Path(kwargs['model_path']).is_file()
        raise RuntimeError('synthetic AP failure')
    monkeypatch.setattr(evaluate_kitti_bev, 'run_evaluation', fail)
    with pytest.raises(RuntimeError, match='synthetic AP failure'):
        train.evaluate_training_ap({'config': config}, path, ROOT/'detector', torch.device('cpu'), generator, run)
    after = train.capture_rng_state(generator)
    assert before['python'] == after['python']
    assert torch.equal(before['numpy']['keys'], after['numpy']['keys'])
    assert torch.equal(before['torch_cpu'], after['torch_cpu'])
    assert torch.equal(before['loader_generator'], after['loader_generator'])
    assert not list(run.glob('ap-validation-*'))


def test_synthetic_gpu_gate_accepts_ap_source_without_computing_ap(tmp_path):
    from tools.benchmarks.smoke_detector import run_smoke
    config, _, _, _, _ = setup(tmp_path)
    before = copy.deepcopy(config)
    report = run_smoke(config, device='cpu', precisions=['fp32'], steps=1)
    assert report['status'] == 'passed' and report['precisions']['fp32']['checkpoint_restore_verified']
    assert config == before and config['train']['checkpoint_selection']['primary'] == 'ap'

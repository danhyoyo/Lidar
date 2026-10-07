"""Both checkpoint winners are evaluated with their actual selection policy."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmark_fixtures import ROOT
from common import checkpoint_identity, sha256, write_json
from tools.kitti_training_pipeline import notebook_workflow as workflow
from tools.kitti_training_pipeline.checkpoint_selection import selection_settings, advance_ap_state, save_selections


def distinct_run(root, primary='ap'):
    root.mkdir(parents=True)
    config = json.loads((ROOT/'configs/config.json').read_text())
    config['train'].update(epochs=3, precision='fp32', checkpoint_selection={'primary': primary, 'ap_every': 1})
    config['loss'] = {'name': 'baseline'}
    split = root/'val.txt'; split.write_text('000001;kitti\n000002;kitti\n000003;kitti\n')
    config['val']['data'] = str(split)
    write_json(root/'config.resolved.json', config)
    settings = selection_settings(config); state = None; rows = []; best_loss = float('inf')
    for epoch, loss, score in [(1, 4., 10.), (2, 2., 30.), (3, 1., 20.)]:
        measurement = {'score': score, 'metric_mode': 'local_bev', 'sampling': 'R40',
            'metric': 'map_moderate_percent', 'frames': 3, 'split_sha256': sha256(split),
            'per_class_moderate_percent': dict(Car=score, Pedestrian=score, Cyclist=score)} if primary == 'ap' else None
        state, retained_ap = advance_ap_state(state, measurement, epoch, settings)
        retained_loss = loss < best_loss; best_loss = min(best_loss, loss)
        row = {'epoch': epoch, 'validation': {'loss': loss}, 'retained': retained_loss,
            'ap_validation': measurement, 'retained_ap': retained_ap, 'optimizer_updates': 1,
            'precision': 'fp32', 'loss': 'baseline'}
        rows.append(row)
        payload = {'config': config, 'checkpoint_identity': checkpoint_identity(config),
            'epoch': epoch, 'validation': row['validation'], 'best_validation_objective': best_loss,
            'ap_selection': state, 'ap_validation': measurement,
            'model_state_dict': {'weight': torch.tensor([float(epoch)])}}
        save_selections(root, payload, retained_loss=retained_loss, retained_ap=retained_ap)
    (root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return root


def test_both_winners_resolve_without_changing_config_or_primary_alias(tmp_path):
    run = distinct_run(tmp_path/'run'); before = (run/'config.resolved.json').read_bytes()
    assert hasattr(workflow, 'selected_runs'), 'Dual checkpoint evaluation is missing'
    winners = workflow.selected_runs(run)
    assert set(winners) == {'ap', 'loss'}
    assert winners['ap']['checkpoint'] == run/'selected/best_ap.pt'
    assert winners['loss']['checkpoint'] == run/'selected/best_loss.pt'
    assert winners['ap']['epoch'] == 2 and winners['loss']['epoch'] == 3
    assert winners['ap']['selection_path'] == run/'selected/selection_ap.json'
    assert winners['loss']['selection_path'] == run/'selected/selection_loss.json'
    assert workflow.selected_run(run)['checkpoint'] == winners['ap']['checkpoint']
    assert (run/'config.resolved.json').read_bytes() == before


@pytest.mark.parametrize('kind', ['ap', 'loss'])
def test_requested_winner_cannot_be_substituted_with_other_checkpoint(tmp_path, kind):
    run = distinct_run(tmp_path/'run')
    other = 'loss' if kind == 'ap' else 'ap'
    shutil.copyfile(run/f'selected/best_{other}.pt', run/f'selected/best_{kind}.pt')
    with pytest.raises(ValueError, match='winner|epoch'):
        workflow.selected_run(run, kind=kind)


def test_missing_new_ap_winner_is_an_error_and_loss_selection_still_checks_ap_history(tmp_path):
    run = distinct_run(tmp_path/'run')
    assert hasattr(workflow, 'selected_runs'), 'Dual checkpoint evaluation is missing'
    (run/'selected/best_ap.pt').unlink()
    with pytest.raises(FileNotFoundError): workflow.selected_runs(run)
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    rows[-1]['retained_ap'] = True
    (run/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='AP retention'):
        workflow.selected_run(run, kind='loss')


def test_legacy_run_exposes_only_its_genuine_loss_winner(tmp_path):
    run = distinct_run(tmp_path/'run', primary='loss')
    config = json.loads((run/'config.resolved.json').read_text()); del config['train']['checkpoint_selection']
    write_json(run/'config.resolved.json', config)
    retained = run/'best_checkpoints/3epoch.pt'
    saved = torch.load(retained, weights_only=True); saved['config'] = config
    saved['checkpoint_identity'] = checkpoint_identity(config)
    torch.save(saved, retained); shutil.copyfile(retained, run/'selected/best.pt')
    (run/'selected/best_loss.pt').unlink(); (run/'selected/selection_loss.json').unlink()
    assert hasattr(workflow, 'selected_runs'), 'Dual checkpoint evaluation is missing'
    winners = workflow.selected_runs(run)
    assert set(winners) == {'loss'} and winners['loss']['checkpoint'].name == 'best.pt'


def test_notebook_evaluation_reloads_helpers_cached_before_a_colab_pull():
    notebook = json.loads((ROOT/'3D_Lidar_Object_Detection_Notebook_standard.ipynb').read_text())
    cell = next(''.join(c['source']) for c in notebook['cells']
        if c['cell_type'] == 'code' and 'comparison.to_csv' in ''.join(c['source']))
    prefix = cell.split('if RUN_EVALUATION:')[0]
    script = """from tools.kitti_training_pipeline import notebook_workflow
from tools.benchmarks import benchmark_protocol
del notebook_workflow.selected_runs
benchmark_protocol.freeze_protocol = None
""" + prefix + "\nassert callable(selected_runs) and callable(freeze_protocol)\n"
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('with_baseline', [False, True])
def test_actual_notebook_cell_evaluates_both_winners_and_records_independent_evidence(tmp_path, with_baseline):
    import train
    from test_ap_training import setup
    from tools.benchmarks.benchmark_protocol import freeze_protocol
    from tools.benchmarks.benchmark_evidence import record_evidence, verify_evidence
    from common import read_json
    config, _, raw, run, args = setup(tmp_path)
    train.main(args)
    benchmark = run/'benchmark'; benchmark.mkdir()
    commands = []
    def command(values, **options):
        values = list(map(str, values)); values[values.index('--device')+1] = 'cpu'
        commands.append(values)
        result = subprocess.run(values, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
    namespace = dict(RUN_EVALUATION=True, RUN_DIR=run, REPO_DIR=ROOT, RAW_TRAINING=raw,
        BENCHMARK_DIR=benchmark, EVALUATION_MODES=['local_bev'], BASELINE_REPORTS={},
        BASELINE_EVIDENCE={}, COMPARISON_KIND='protocol', selected_run=workflow.selected_run,
        evaluation_command=workflow.evaluation_command,
        run_command=command, read_json=read_json, write_json=write_json,
        freeze_protocol=freeze_protocol, record_evidence=record_evidence)
    if with_baseline:
        from benchmark_fixtures import evaluation_fixture
        baseline = tmp_path/'baseline'
        files = evaluation_fixture(baseline)
        shutil.copyfile(files[4], baseline/'config.resolved.json')
        # A Colab kernel may still hold selected_run's older return dictionary.
        namespace['baseline_run'] = {'run_dir': baseline}
        namespace['BASELINE_REPORTS'] = {'local_bev': read_json(files[1])}
    notebook = json.loads((ROOT/'3D_Lidar_Object_Detection_Notebook_standard.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells']
        if c['cell_type'] == 'code' and 'comparison.to_csv' in ''.join(c['source']))
    exec(compile(source, 'notebook_dual_evaluation', 'exec'), namespace)
    assert len(commands) == 2, 'Notebook did not evaluate both best AP and best loss'
    comparison = read_json(run/'comparison.json')
    assert {r['checkpoint_selection'] for r in comparison['rows']} == {'ap', 'loss'}
    baseline_rows = [r for r in comparison['rows'] if r['role'] == 'baseline']
    assert len(baseline_rows) == int(with_baseline)
    if with_baseline:
        assert baseline_rows[0]['checkpoint_selection'] == 'loss'
        assert baseline_rows[0]['validation_loss'] is not None
    assert all(r['checkpoint_epoch'] >= 1 and r['checkpoint_path'] for r in comparison['rows'])
    assert set(comparison['checkpoint_evidence']) == {'ap', 'loss'}
    for kind, policy in [('ap', 'maximum validation AP'), ('loss', 'minimum validation loss')]:
        report = read_json(run/f'evaluation_{kind}_local_bev.json')
        evidence = read_json(comparison['checkpoint_evidence'][kind]['local_bev'])
        assert evidence['selection_policy'] == policy and evidence['checkpoint_selection_verified']
        assert report['model']['path'] == str(run/f'selected/best_{kind}.pt')
        verify_evidence(evidence)
    assert read_json(comparison['candidate_evidence']['local_bev'])['selection_policy'] == 'maximum validation AP'
    with pytest.raises(ValueError, match='selection protocol'):
        record_evidence(benchmark/'evaluated_protocol_ap_local_bev.json',
            run/'evaluation_loss_local_bev.json', run/'selected/selection_loss.json', run/'metrics.jsonl')

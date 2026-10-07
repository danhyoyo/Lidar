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
del notebook_workflow.selected_runs
""" + prefix + "\nassert callable(selected_runs)\n"
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('primary', ['ap', 'loss'])
def test_actual_notebook_cell_evaluates_independent_winners_with_own_config(tmp_path, primary):
    import shlex
    import train
    from test_ap_training import setup
    from common import read_json
    from notebook_test_utils import notebook_cell, execute_shell_cell

    config, _, raw, run, args = setup(tmp_path, primary=primary)
    train.main(args)
    namespace = dict(RUN_EVALUATION=True, RUN_DIR=run, REPO_DIR=ROOT, RAW_KITTI_ROOT=raw.parent,
        EVALUATION_MODES=['local_bev'], RUN_NAME=run.name, DEVICE='cpu',
        q=shlex.quote, sys=sys, Path=Path, read_json=read_json, write_json=write_json,
        validate_modes=workflow.validate_modes)
    execute_shell_cell(notebook_cell('comparison.to_csv'), namespace)
    comparison = read_json(run/'comparison.json')
    expected = {'ap', 'loss'} if primary == 'ap' else {'loss'}
    assert {r['checkpoint_selection'] for r in comparison['rows']} == expected
    assert (run/'comparison.csv').is_file()
    for kind in expected:
        selected = workflow.selected_run(run, kind=kind)
        report = read_json(run/f'evaluation_{kind}_local_bev.json')
        assert report['model']['path'] == str(selected['checkpoint'])
        assert report['model']['checkpoint_epoch'] == selected['epoch']
        assert report['data']['frames'] == report['data']['full_split_frames']
        assert all(r['validation_loss'] is not None for r in comparison['rows'])

"""Independent loss/AP winners, strict ties and auditable full-split selection."""
import copy
import importlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'tools/kitti_training_pipeline/checkpoint_selection.py'
    assert path.is_file(), 'Dual checkpoint selection is not implemented'
    return importlib.import_module('tools.kitti_training_pipeline.checkpoint_selection')


def config(primary='ap', interval=1, box='bev'):
    return {'data': {'box_mode': box}, 'train': {'epochs': 6, 'precision': 'fp32',
        'checkpoint_selection': {'primary': primary, 'ap_every': interval, 'metric_mode': 'auto'}},
        'loss': {'name': 'baseline'}}


def measurement(score=30., mode='local_bev'):
    return {'score': score, 'metric_mode': mode, 'sampling': 'R40',
        'metric': 'map_moderate_percent', 'frames': 3, 'split_sha256': 'val-hash',
        'per_class_moderate_percent': dict(Car=score, Pedestrian=score, Cyclist=score)}


def test_schedule_includes_final_epoch_without_reducing_loss_validation():
    m = module(); settings = m.selection_settings(config(interval=5))
    assert [e for e in range(1, 7) if m.ap_due(e, 6, settings)] == [5, 6]
    assert [e for e in range(1, 7) if m.ap_due(e, 6, m.selection_settings(config()))] == list(range(1, 7))
    assert not m.ap_due(6, 6, m.selection_settings(config(primary='loss')))


def test_legacy_configs_keep_loss_policy_and_reference_mode_is_explicit():
    m = module()
    legacy = config(); del legacy['train']['checkpoint_selection']
    assert m.selection_settings(legacy)['primary'] == 'loss'
    assert m.selection_settings(config(box='3d'))['metric_mode'] == '3d'


@pytest.mark.parametrize('key,value', [('primary', 'best'), ('ap_every', 0), ('ap_every', True),
    ('ap_every', 1.5), ('metric_mode', 'bev'), ('metric_mode', '3d')])
def test_invalid_selection_settings_are_rejected(key, value):
    c = config(); c['train']['checkpoint_selection'][key] = value
    with pytest.raises(ValueError): module().selection_settings(c)


def test_ap_state_handles_zero_strict_ties_and_resume_without_resetting_winner():
    m = module(); settings = m.selection_settings(config())
    state, retained = m.advance_ap_state(None, measurement(0.), 1, settings)
    assert retained and state['best_score'] == 0.
    state, retained = m.advance_ap_state(state, measurement(30.), 2, settings)
    assert retained and state['best_epoch'] == 2
    resumed = json.loads(json.dumps(state))
    for epoch, score in [(3, 30.), (4, 10.)]:
        resumed, retained = m.advance_ap_state(resumed, measurement(score), epoch, settings)
        assert not retained and resumed['best_epoch'] == 2 and resumed['best_score'] == 30.
    with pytest.raises(ValueError):
        m.advance_ap_state(resumed, measurement(), 5, m.selection_settings(config(interval=5)))


def report(score=30.):
    cells = {name: {'difficulties': {'Moderate': {'ap_r40_percent': score, 'ground_truth': 1}}}
             for name in ('Car', 'Pedestrian', 'Cyclist')}
    return {'status': 'ok', 'protocol': {'metric_mode': 'local_bev'},
        'accuracy': {'map_moderate_percent': score, 'per_class': cells},
        'data': {'frames': 3, 'full_split_frames': 3, 'split_sha256': 'val-hash'}}


def test_ap_measurement_uses_equal_three_class_moderate_r40():
    m = module(); settings = m.selection_settings(config())
    assert m.ap_measurement(report(30.), settings, full_frames=3, split_sha256='val-hash') == measurement()


@pytest.mark.parametrize('mode', ['3d', 'bev'])
def test_reference_selection_reads_r40_instead_of_r11(mode):
    m = module(); c = config(box='3d'); c['train']['checkpoint_selection']['metric_mode'] = mode
    settings = m.selection_settings(c)
    cells = {name: {'Moderate': 30.} for name in ('Car', 'Pedestrian', 'Cyclist')}
    accuracy = {'per_class': cells, 'map_moderate_percent': 30.,
        'valid_gt_counts': {name: {'Moderate': 1} for name in cells}}
    r = {'status': 'ok', 'protocol': {'metric_mode': mode},
        'accuracy': {'R11': {'map_moderate_percent': 99.}, 'R40': accuracy},
        'data': {'frames': 3, 'split_sha256': 'val-hash'}}
    assert m.ap_measurement(r, settings, full_frames=3, split_sha256='val-hash') == measurement(mode=mode)


@pytest.mark.parametrize('problem', ['partial', 'missing_class', 'no_gt', 'nan', 'macro', 'mode', 'split'])
def test_partial_missing_or_inconsistent_ap_is_rejected(problem):
    r = report()
    if problem == 'partial': r['data']['frames'] = 2
    elif problem == 'missing_class': del r['accuracy']['per_class']['Cyclist']
    elif problem == 'no_gt': r['accuracy']['per_class']['Car']['difficulties']['Moderate']['ground_truth'] = 0
    elif problem == 'nan': r['accuracy']['map_moderate_percent'] = math.nan
    elif problem == 'macro': r['accuracy']['map_moderate_percent'] = 90.
    elif problem == 'mode': r['protocol']['metric_mode'] = 'bev'
    else: r['data']['split_sha256'] = 'other'
    with pytest.raises(ValueError):
        module().ap_measurement(r, module().selection_settings(config()), full_frames=3, split_sha256='val-hash')


def test_independent_checkpoint_files_keep_different_epoch_winners(tmp_path):
    import torch
    m = module(); c = config(); settings = m.selection_settings(c)
    state = None; best_loss = math.inf
    for epoch, loss, score in [(1, 4., 10.), (2, 2., 30.), (3, 1., 20.), (4, 1., 30.)]:
        retained_loss = loss < best_loss; best_loss = min(best_loss, loss)
        state, retained_ap = m.advance_ap_state(state, measurement(score), epoch, settings)
        payload = {'config': c, 'epoch': epoch, 'validation': {'loss': loss},
            'best_validation_objective': best_loss, 'ap_selection': state,
            'model_state_dict': {'weight': torch.tensor([float(epoch)])}}
        m.save_selections(tmp_path, payload, retained_loss=retained_loss, retained_ap=retained_ap)
    selected = tmp_path / 'selected'
    for name, epoch in [('best_ap.pt', 2), ('best_loss.pt', 3), ('best.pt', 2)]:
        assert torch.load(selected / name, weights_only=True)['epoch'] == epoch
    assert json.loads((selected / 'selection.json').read_text())['criterion'] == 'maximum validation AP'
    assert json.loads((selected / 'selection_loss.json').read_text())['epoch'] == 3


def test_verified_history_uses_ap_winner_even_when_loss_winner_differs():
    m = module(); c = config(); c['train']['epochs'] = 3
    settings = m.selection_settings(c); state = None; rows = []
    for e, loss, score in [(1, 4., 10.), (2, 2., 30.), (3, 1., 20.)]:
        state, keep = m.advance_ap_state(state, measurement(score), e, settings)
        rows.append({'epoch': e, 'validation': {'loss': loss}, 'retained': True,
            'ap_validation': measurement(score), 'retained_ap': keep,
            'optimizer_updates': 2, 'precision': 'fp32', 'loss': 'baseline'})
    saved = {'epoch': 2, 'validation': {'loss': 2.}, 'best_validation_objective': 2.,
        'ap_selection': {**state, 'best_epoch': 2}}
    selection = {'criterion': 'maximum validation AP', 'epoch': 2, 'score': 30., 'settings': settings}
    assert m.verify_selection(c, selection, rows, saved)['epoch'] == 2
    bad = copy.deepcopy(rows); bad[2]['retained_ap'] = True
    with pytest.raises(ValueError): m.verify_selection(c, selection, bad, saved)
    bad = copy.deepcopy(selection); bad['epoch'] = 3
    with pytest.raises(ValueError): m.verify_selection(c, bad, rows, saved)

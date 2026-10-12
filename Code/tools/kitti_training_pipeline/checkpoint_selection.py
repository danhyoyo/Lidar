"""Shared loss/AP checkpoint policy, retention and history verification."""
from __future__ import annotations

import math
from pathlib import Path

CLASSES = ('Car', 'Pedestrian', 'Cyclist')
LOSS_CRITERION = 'minimum mean validation loss'
AP_CRITERION = 'maximum validation AP'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, *, percent=False):
    _require(type(value) in (int, float) and math.isfinite(value), 'Selection metric must be finite numeric data')
    _require(not percent or 0 <= value <= 100, 'AP must be a percentage between 0 and 100')
    return float(value)


def selection_settings(config):
    """Missing policy preserves historical loss-selected runs."""
    source = config.get('train', {}).get('checkpoint_selection', {})
    _require(isinstance(source, dict), 'train.checkpoint_selection must be a dictionary')
    _require(set(source) <= {'primary', 'ap_every', 'metric_mode'}, 'Unknown checkpoint selection option')
    primary, interval = source.get('primary', 'loss'), source.get('ap_every', 1)
    _require(primary in {'loss', 'ap'}, 'checkpoint_selection.primary must be loss or ap')
    _require(type(interval) is int and interval > 0, 'checkpoint_selection.ap_every must be a positive integer')
    box = config.get('data', {}).get('box_mode', 'bev')
    mode = source.get('metric_mode', 'auto')
    if mode == 'auto':
        mode = 'local_bev' if box == 'bev' else '3d'
    _require(mode in ({'local_bev'} if box == 'bev' else {'3d', 'bev'} if box == '3d' else set()),
             'Checkpoint AP mode must match the trained BEV/3D box mode')
    return {'primary': primary, 'ap_every': interval, 'metric_mode': mode,
            'sampling': 'R40', 'metric': 'map_moderate_percent',
            'difficulty': 'Moderate', 'classes': list(CLASSES)}


def selection_protocol(config, *, kind=None):
    settings = selection_settings(config)
    kind = settings['primary'] if kind is None else kind
    _require(kind in {'ap', 'loss'}, 'Checkpoint selection kind must be ap or loss')
    if kind == 'loss':
        return {'policy': 'minimum validation loss'}
    _require(settings['primary'] == 'ap', 'This training policy has no validated best AP checkpoint')
    return {'policy': 'maximum validation AP', **settings}


def ap_due(epoch, epochs, settings):
    return settings['primary'] == 'ap' and (epoch % settings['ap_every'] == 0 or epoch == epochs)


def _check_measurement(value, settings):
    _require(isinstance(value, dict), 'AP validation measurement is missing')
    for key in ('metric_mode', 'sampling', 'metric'):
        _require(value.get(key) == settings[key], f'AP selection {key} differs from configured policy')
    _require(type(value.get('frames')) is int and value['frames'] > 0 and
             isinstance(value.get('split_sha256'), str) and bool(value['split_sha256']), 'AP validation split provenance is missing')
    cells = value.get('per_class_moderate_percent', {})
    _require(set(cells) == set(CLASSES), 'AP selection requires all three classes')
    score = _number(value.get('score'), percent=True)
    macro = sum(_number(cells[name], percent=True) for name in CLASSES) / 3
    _require(math.isclose(score, macro, abs_tol=1e-8), 'AP macro must equally average all three classes')
    return score


def ap_measurement(report, settings, *, full_frames, split_sha256):
    """Extract a complete validation Moderate R40 macro without reduced classes."""
    _require(report.get('status') == 'ok', 'AP evaluation did not succeed')
    mode = settings['metric_mode']
    actual_mode = report['protocol'].get('metric_mode', 'local_bev')
    _require(actual_mode == mode, 'AP evaluator metric mode differs from selection policy')
    data = report['data']
    _require(data['frames'] == full_frames and data['split_sha256'] == split_sha256,
             'AP selection requires the exact full validation split')
    if mode == 'local_bev':
        _require(data['full_split_frames'] == full_frames, 'Partial AP validation is forbidden')
    accuracy = report['accuracy'] if mode == 'local_bev' else report['accuracy']['R40']
    _require(set(accuracy['per_class']) == set(CLASSES), 'AP selection requires all three classes')
    cells = {}
    for name in CLASSES:
        if mode == 'local_bev':
            cell = accuracy['per_class'][name]['difficulties']['Moderate']
            count, score = cell['ground_truth'], cell['ap_r40_percent']
        else:
            count = accuracy['valid_gt_counts'][name]['Moderate']
            score = accuracy['per_class'][name]['Moderate']
        _require(type(count) is int and count > 0, f'AP selection requires valid Moderate GT for {name}')
        cells[name] = _number(score, percent=True)
    value = {'score': accuracy['map_moderate_percent'], 'metric_mode': mode,
             'sampling': 'R40', 'metric': settings['metric'], 'frames': full_frames,
             'split_sha256': split_sha256, 'per_class_moderate_percent': cells}
    _check_measurement(value, settings)
    return value


def advance_ap_state(state, measurement, epoch, settings):
    """Keep the first maximum, including a legitimate AP of zero."""
    if state is None:
        state = {'settings': settings, 'best_score': None, 'best_epoch': None, 'best_measurement': None}
    else:
        _require(state['settings'] == settings, 'Checkpoint AP selection policy changed on resume')
        state = dict(state)
        if state['best_score'] is not None:
            _number(state['best_score'], percent=True)
    retained = False
    if measurement is not None:
        score = _check_measurement(measurement, settings)
        retained = state['best_score'] is None or score > state['best_score']
        if retained:
            state.update(best_score=score, best_epoch=epoch, best_measurement=measurement)
    return state, retained


def save_selections(run_dir, payload, *, retained_loss, retained_ap):
    """Retain independent winners plus the primary best.pt compatibility alias."""
    try:
        from .common import atomic_torch_save, write_json
    except ImportError:
        from common import atomic_torch_save, write_json
    root = Path(run_dir); selected = root / 'selected'; selected.mkdir(parents=True, exist_ok=True)
    retained = root / 'best_checkpoints' / f"{payload['epoch']}epoch.pt"
    if retained_loss or retained_ap:
        atomic_torch_save(payload, retained)
    settings = selection_settings(payload['config'])
    for kind, keep in (('loss', retained_loss), ('ap', retained_ap)):
        if not keep:
            continue
        record = {'epoch': payload['epoch'], 'checkpoint': str(retained.resolve()),
                  'criterion': LOSS_CRITERION if kind == 'loss' else AP_CRITERION}
        if kind == 'loss':
            record['validation_objective'] = payload['validation']['loss']
        else:
            record.update(score=payload['ap_selection']['best_score'], settings=settings)
        atomic_torch_save(payload, selected / f'best_{kind}.pt')
        write_json(selected / f'selection_{kind}.json', record)
        if settings['primary'] == kind:
            atomic_torch_save(payload, selected / 'best.pt')
            write_json(selected / 'selection.json', record)


def verify_selection(config, selection, rows, saved, *, kind=None):
    """Verify the first policy winner against a complete successful history."""
    epochs = config['train']['epochs']; settings = selection_settings(config)
    kind = settings['primary'] if kind is None else kind
    selection_protocol(config, kind=kind)
    _require([row['epoch'] for row in rows] == list(range(1, epochs + 1)),
             'Selected-run evaluation requires complete ordered configured training history')
    running_loss, running_ap = math.inf, None
    best_loss, best_ap = None, None
    split_provenance = None
    expected_split = None
    if settings['primary'] == 'ap' and config.get('val', {}).get('data'):
        try:
            from .common import sha256
        except ImportError:
            from common import sha256
        path = Path(config['val']['data']).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        ids = [line.strip().split(';')[0] for line in path.read_text().splitlines() if line.strip()]
        _require(bool(ids) and len(ids) == len(set(ids)), 'AP validation split must contain unique nonempty IDs')
        expected_split = (len(ids), sha256(path))
    for row in rows:
        loss = _number(row['validation']['loss'])
        _require(type(row['optimizer_updates']) is int and row['optimizer_updates'] > 0,
                 'Training history requires actual optimizer updates')
        _require(row['precision'] == config['train']['precision'] and row['loss'] == config['loss']['name'],
                 'Training history precision/objective does not match config')
        _require(type(row['retained']) is bool and row['retained'] == (loss < running_loss),
                 'Training retention history does not match minimum-loss policy')
        if loss < running_loss:
            running_loss, best_loss = loss, row
        if settings['primary'] == 'ap':
            measurement = row.get('ap_validation')
            _require((measurement is not None) == ap_due(row['epoch'], epochs, settings),
                     'AP validation history does not match configured evaluation schedule')
            keep = False
            if measurement is not None:
                score = _check_measurement(measurement, settings)
                current_split = (measurement['frames'], measurement['split_sha256'])
                _require(expected_split is None or current_split == expected_split,
                         'AP validation frames/hash differ from the configured full split')
                _require(split_provenance is None or split_provenance == current_split, 'AP validation split changed during training')
                split_provenance = current_split
                keep = running_ap is None or score > running_ap
                if keep:
                    running_ap, best_ap = score, row
            _require(type(row.get('retained_ap')) is bool and row['retained_ap'] == keep,
                     'AP retention history does not match maximum-AP policy')
    # Always audit the original training history, then select the requested
    # winner. Evaluating best loss does not disable this run's AP history checks.
    select_ap = kind == 'ap'
    best = best_ap if select_ap else best_loss
    _require(best is not None and selection['epoch'] == best['epoch'] and saved['epoch'] == best['epoch'] and
             selection['criterion'] == (AP_CRITERION if select_ap else LOSS_CRITERION),
             'Selected checkpoint epoch does not match the configured loss/AP winner')
    _require(math.isclose(_number(saved['validation']['loss']), best['validation']['loss'], abs_tol=1e-8),
             'Selected checkpoint loss differs from complete training history')
    if select_ap:
        state = saved.get('ap_selection', {})
        _require(selection.get('settings') == settings and state.get('settings') == settings and
                 state.get('best_epoch') == best['epoch'] and state.get('best_measurement') == best['ap_validation'],
                 'Selected checkpoint AP state differs from its selection history')
        for value in (selection.get('score'), state.get('best_score')):
            _require(math.isclose(_number(value, percent=True), running_ap, abs_tol=1e-8), 'Selected AP differs from training history')
        loss_at_epoch = min(row['validation']['loss'] for row in rows[:best['epoch']])
        _require(math.isclose(_number(saved['best_validation_objective']), loss_at_epoch, abs_tol=1e-8),
                 'AP-selected checkpoint best-loss state differs from its epoch history')
    else:
        for value in (selection['validation_objective'], saved['best_validation_objective']):
            _require(math.isclose(_number(value), running_loss, abs_tol=1e-8), 'Selected checkpoint loss differs from complete training history')
    return best

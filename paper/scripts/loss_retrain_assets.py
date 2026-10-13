"""Read the supplied 100-epoch OGA/Q-OGA pair; never run a detector."""
from __future__ import annotations

import hashlib
import json
import pickletools
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt

CLASSES = ('Car', 'Pedestrian', 'Cyclist')
DIFFICULTIES = ('Easy', 'Moderate', 'Hard')
SOURCE_ROOT = Path('/home/duyennh/Downloads/drive-download-20261013T095105Z-1-001')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            field = f'{prefix}.{key}' if prefix else key
            if key not in left or key not in right:
                result.append({'field': field, 'oga': left.get(key), 'q_oga': right.get(key)})
            else:
                result.extend(differences(left[key], right[key], field))
        return result
    return [] if left == right else [{'field': prefix, 'oga': left, 'q_oga': right}]


def prepare_pair(paper, copy):
    research = paper / 'research'
    pair = {'runs': {}, 'evaluations': [], 'configuration_audit': [], 'metrics': {}}
    configs, reports = {}, {}
    for directory in sorted(SOURCE_ROOT.iterdir()):
        if not (directory / 'config.resolved.json').exists():
            continue
        config_path = directory / 'config.resolved.json'
        config = json.loads(config_path.read_text())
        objective = config['loss']['name']
        if objective not in ('oga', 'q_oga'):
            continue
        label = 'O100' if objective == 'oga' else 'Q100'
        if label in pair['runs']:
            raise ValueError(f'Duplicate retraining run: {label}')
        configs[label] = config
        destination = research / 'loss_retrain' / label
        for name in ('config.resolved.json', 'run.json', 'metrics.jsonl',
                     'selected/selection.json', 'selected/selection_ap.json',
                     'selected/selection_loss.json'):
            copy(directory / name, destination / name, 'Supplied matched 100-epoch loss retraining evidence')
        copy(config_path, research / 'configs' / (label + '.json'), 'Unmodified retraining configuration matching the recorded training hash')
        copy(config_path, research / 'configs' / 'as_provided' / (label + '.json'), 'Unmodified supplied retraining configuration')
        copy(directory / 'run.json', research / 'configs' / (label + '_run.json'), 'Actual retraining run settings')
        run_record = json.loads((directory / 'run.json').read_text())
        if digest(config_path) != run_record['config_sha256']:
            raise ValueError(f'Retraining config digest mismatch: {label}')
        metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines() if line.strip()]
        if [m['epoch'] for m in metrics] != list(range(1, 101)) or run_record['epochs'] != 100:
            raise ValueError(f'Incomplete 100-epoch retraining: {label}')
        if config['train']['epochs'] != 100 or run_record['seed'] != 42:
            raise ValueError(f'Unexpected retraining budget or seed: {label}')
        selection = json.loads((directory / 'selected/selection.json').read_text())
        loss_selection = json.loads((directory / 'selected/selection_loss.json').read_text())
        curve = [{'epoch': m['epoch'], 'map_moderate': m['ap_validation']['score'],
                  'per_class_moderate': m['ap_validation']['per_class_moderate_percent']}
                 for m in metrics if m.get('ap_validation')]
        if [m['epoch'] for m in curve] != list(range(10, 101, 10)):
            raise ValueError(f'Unexpected AP evaluation schedule: {label}')
        best = max(curve, key=lambda m: m['map_moderate'])
        loss_best = min(metrics, key=lambda m: m['validation']['loss'])
        if best['epoch'] != selection['epoch'] or abs(best['map_moderate'] - selection['score']) > 1e-9:
            raise ValueError(f'AP selection mismatch: {label}')
        if loss_best['epoch'] != loss_selection['epoch'] or abs(loss_best['validation']['loss'] - loss_selection['validation_objective']) > 1e-9:
            raise ValueError(f'Loss selection mismatch: {label}')
        reports[label] = {}
        for mode in ('ap', 'loss'):
            path = directory / f'evaluation_{mode}_local_bev.json'
            raw = json.loads(path.read_text())
            reports[label][mode] = raw
            copy(path, destination / path.name, 'Raw saved local BEV evaluation, including per-frame input hashes')
            checkpoint = directory / 'selected' / f'best_{mode}.pt'
            if digest(checkpoint) != raw['model']['sha256']:
                raise ValueError(f'Checkpoint digest mismatch: {label}/{mode}')
            if raw['data']['config_sha256'] != run_record['config_sha256']:
                raise ValueError(f'Evaluation config digest mismatch: {label}/{mode}')
            if raw['data']['frames'] != 3769 or raw['model']['parameters'] != 815744:
                raise ValueError(f'Unexpected evaluation scope or model size: {label}/{mode}')
            data = {key: value for key, value in raw['data'].items() if not isinstance(value, (dict, list))}
            pair['evaluations'].append({'paper_label': label, 'source': str(path), 'sha256': digest(path),
                                       'accuracy': raw['accuracy'], 'model': raw['model'],
                                       'protocol': raw['protocol'], 'data': data,
                                       'latency': raw['latency'], 'counts': raw['counts'], 'runtime': raw['runtime']})
        raw = reports[label]['ap']
        loss_raw = reports[label]['loss']
        if raw['model']['checkpoint_epoch'] != selection['epoch'] or loss_raw['model']['checkpoint_epoch'] != loss_selection['epoch']:
            raise ValueError(f'Evaluated checkpoint epoch mismatch: {label}')
        if abs(raw['accuracy']['map_moderate_percent'] - selection['score']) > 1e-9:
            raise ValueError(f'Evaluated selection score mismatch: {label}')
        with zipfile.ZipFile(directory / 'selected/best_ap.pt') as archive:
            member = next(name for name in archive.namelist() if name.endswith('data.pkl'))
            strings = [arg for op, arg, _ in pickletools.genops(archive.read(member))
                       if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE')]
            head_keys = [name for name in strings if 'header' in name and name.endswith('weight')][:12]
        if not any('.cls.conv1.weight' in name for name in head_keys):
            raise ValueError(f'Independent head evidence missing: {label}')
        pair['configuration_audit'].append({'label': label, 'supplied_sha256': digest(config_path),
                                            'recorded_training_sha256': run_record['config_sha256'],
                                            'recovered_sha256': digest(config_path),
                                            'removed_post_run_header_architecture': False,
                                            'checkpoint_head_parameter_names': head_keys})
        moderate = [raw['accuracy']['per_class'][c]['difficulties']['Moderate'] for c in CLASSES]
        gt = sum(row['ground_truth'] for row in moderate)
        tp = sum(row['true_positives'] for row in moderate)
        fp = sum(row['false_positives'] for row in moderate)
        row = {'run_id': label, 'paper_label': label, 'evidence_series': 'matched_loss_retraining_100_epochs',
               'name': directory.name, 'directory': str(directory), 'local_directory': str(directory),
               'config': str(config_path), 'checkpoint': str(directory / 'selected/best_ap.pt'),
               'evaluation': str(directory / 'evaluation_ap_local_bev.json'), 'selection': 'ap',
               'seed': 42, 'epochs_completed': 100, 'epochs_configured': 100, 'complete': True,
               'selected_epoch': selection['epoch'], 'loss': objective, 'focal': 'focal', 'iqa': False,
               'detail': False, 'local_attention': 'simam', 'backbone': 'mobilepixornext',
               'encoding': 'hist14', 'head_mode': 'grouped', 'header_width': 32, 'fusion_width': 32,
               'neck': 'scale_gated_fpn', 'scale_gated_fpn': True, 'params': raw['model']['parameters'],
               'map_moderate': raw['accuracy']['map_moderate_percent'], 'mean_ap9': raw['accuracy']['mean_ap_9_percent'],
               'class_ap': {c: {d: raw['accuracy']['per_class'][c]['difficulties'][d]['ap_r40_percent'] for d in DIFFICULTIES} for c in CLASSES},
               'distance_ap': {c: raw['accuracy']['per_class'][c].get('moderate_distance_bands', {}) for c in CLASSES},
               'loss_checkpoint_map': loss_raw['accuracy']['map_moderate_percent'],
               'loss_checkpoint_epoch': loss_selection['epoch'],
               'loss_checkpoint_validation_objective': loss_selection['validation_objective'],
               'latency_ms': {stage: value['mean_ms'] for stage, value in raw['latency'].items()},
               'model_hz': raw['latency']['model']['fps'], 'end_to_end_hz': raw['latency']['input_to_detections']['fps'],
               'runtime': raw['runtime'], 'source': raw['data']['source'],
               'detections_per_frame': raw['counts']['detections_per_frame'], 'ap_curve': curve,
               'operating_point': {'true_positives': tp, 'false_positives': fp, 'ground_truth': gt,
                                   'micro_recall_percent': 100 * tp / gt,
                                   'false_positives_per_frame': fp / raw['data']['frames']}}
        pair['runs'][label] = row
        pair['metrics'][label] = metrics
    if set(pair['runs']) != {'O100', 'Q100'}:
        raise ValueError('Expected exactly one OGA and one Q-OGA retraining run')
    config_diff = differences(configs['O100'], configs['Q100'])
    if config_diff != [{'field': 'loss.name', 'oga': 'oga', 'q_oga': 'q_oga'}]:
        raise ValueError(f'Additional retraining config differences: {config_diff}')
    reference = reports['O100']['ap']
    for label in ('O100', 'Q100'):
        for raw in reports[label].values():
            for key in ('input_asset_sha256', 'split_sha256', 'encoding_semantic_hash', 'source'):
                if raw['data'][key] != reference['data'][key]:
                    raise ValueError(f'Retraining evidence mismatch: {label}/{key}')
            protocol_diff = differences(reference['protocol'], raw['protocol'])
            if any(item['field'] != 'quality_target' for item in protocol_diff):
                raise ValueError(f'Unmatched retraining evaluation protocol: {label}')
            if raw['data']['source']['dirty'] or raw['model']['parameter_counts'] != reference['model']['parameter_counts']:
                raise ValueError(f'Unmatched source or parameter structure: {label}')
            if any(value['samples'] != 3759 for value in raw['latency'].values()):
                raise ValueError(f'Unexpected timing sample count: {label}')
    if digest(research / 'val_ids.txt') != reference['data']['split_sha256']:
        raise ValueError('Retraining split differs from the manuscript split')
    oga, qoga = pair['runs']['O100'], pair['runs']['Q100']
    comparison = {'date': '2026-10-13', 'source_root': str(SOURCE_ROOT),
                  'configuration_differences': config_diff, 'completed_epochs_each': 100, 'seed': 42,
                  'AP_selection': 'maximum local BEV R40 Moderate mAP among epochs 10,20,...,100',
                  'selected_epochs': {label: row['selected_epoch'] for label, row in pair['runs'].items()},
                  'source': reference['data']['source'], 'split_sha256': reference['data']['split_sha256'],
                  'encoding_semantic_hash': reference['data']['encoding_semantic_hash'],
                  'all_four_reports_match_input_asset_hashes': True,
                  'detector_parameters_each': reference['model']['parameters'],
                  'delta_Q_minus_O_pp': {'map_moderate': qoga['map_moderate'] - oga['map_moderate'],
                                        **{c: qoga['class_ap'][c]['Moderate'] - oga['class_ap'][c]['Moderate'] for c in CLASSES}},
                  'epoch_100_delta_pp': qoga['ap_curve'][-1]['map_moderate'] - oga['ap_curve'][-1]['map_moderate'],
                  'Q_higher_scheduled_AP_evaluations': sum(q['map_moderate'] > o['map_moderate'] for o, q in zip(oga['ap_curve'], qoga['ap_curve'])),
                  'scheduled_AP_evaluations_each': 10,
                  'loss_selected_delta_pp': qoga['loss_checkpoint_map'] - oga['loss_checkpoint_map'],
                  'false_positive_reduction_percent': 100 * (1 - qoga['operating_point']['false_positives'] / oga['operating_point']['false_positives']),
                  'OGA_AP_minus_loss_selected_pp': oga['map_moderate'] - oga['loss_checkpoint_map'],
                  'raw_objective_values_are_not_comparable_performance_metrics': True,
                  'new_detector_training_or_inference_performed': False}
    pair['comparison'] = comparison
    (research / 'loss_retrain_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    (research / 'loss_retrain' / 'run_inventory.json').write_text(json.dumps(list(pair['runs'].values()), indent=2) + '\n')
    return pair


def render_pair(pair, table, save, paper, all_runs):
    runs = pair['runs']
    table('loss_retrain.tex', 'llrrrr', ['Run', 'Objective', 'Car', 'Ped.', 'Cyc.', 'mAP'],
          [[label, 'OGA' if label == 'O100' else 'Q-OGA'] +
           [f'{runs[label]["class_ap"][c]["Moderate"]:.2f}' for c in CLASSES] + [f'{runs[label]["map_moderate"]:.2f}']
           for label in ('O100', 'Q100')])
    table('loss_retrain_curve.tex', 'rrr', ['Epoch', 'OGA mAP', 'Q-OGA mAP'],
          [[o['epoch'], f'{o["map_moderate"]:.2f}', f'{q["map_moderate"]:.2f}']
           for o, q in zip(runs['O100']['ap_curve'], runs['Q100']['ap_curve'])])
    operating_rows = []
    for label in ('P2', 'P3', 'P4', 'P11', 'P16', 'O100', 'Q100'):
        row = all_runs[label]
        if label in runs:
            op = runs[label]['operating_point']
        else:
            raw = json.loads((Path(row['local_directory']) / Path(row['evaluation']).name).read_text())
            moderate = [raw['accuracy']['per_class'][c]['difficulties']['Moderate'] for c in CLASSES]
            tp = sum(m['true_positives'] for m in moderate)
            fp = sum(m['false_positives'] for m in moderate)
            gt = sum(m['ground_truth'] for m in moderate)
            op = {'true_positives': tp, 'false_positives': fp, 'micro_recall_percent': 100 * tp / gt,
                  'false_positives_per_frame': fp / raw['data']['frames']}
        values = [label, f'{op["true_positives"]:,}', f'{op["false_positives"]:,}',
                  f'{op["micro_recall_percent"]:.2f}', f'{op["false_positives_per_frame"]:.2f}', f'{row["end_to_end_hz"]:.2f}']
        operating_rows.append(values)
    table('operating.tex', 'lrrrrr', ['Run', 'TP', 'FP', 'Recall (\\%)', 'FP/frame', 'Pipeline Hz'], operating_rows)
    colors = {'O100': '#0072B2', 'Q100': '#D55E00'}
    fig, ax = plt.subplots(figsize=(3.5, 2.4), layout='constrained')
    for label in ('O100', 'Q100'):
        curve = runs[label]['ap_curve']
        ax.plot([m['epoch'] for m in curve], [m['map_moderate'] for m in curve],
                color=colors[label], marker='o' if label == 'O100' else 's', markersize=3,
                label='OGA' if label == 'O100' else 'Q-OGA', linewidth=1.1)
    ax.axvline(70, color='#666666', ls=':', lw=.8)
    ax.text(70, 51, 'Selected epoch 70', fontsize=6.5, rotation=90, va='bottom', ha='right')
    ax.set(xlabel='Training epoch', ylabel='Local BEV mAP Moderate (%)',
           xticks=[10, 30, 50, 70, 100], ylim=(46, 73))
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    ax.grid(alpha=.2)
    save(fig, 'loss_retrain')
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), layout='constrained')
    for label in ('O100', 'Q100'):
        metrics = pair['metrics'][label]
        for ax, values in zip(axes, ([m['train_objective'] for m in metrics],
                                     [m['validation']['loss'] for m in metrics])):
            ax.plot([m['epoch'] for m in metrics], values, color=colors[label],
                    label='OGA' if label == 'O100' else 'Q-OGA', linewidth=1)
    for ax, title in zip(axes, ('(a) Training objective', '(b) Validation objective')):
        ax.set(xlabel='Training epoch', ylabel='Recorded objective (log scale)', yscale='log', title=title)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=7)
    save(fig, 'loss_retrain_objectives')

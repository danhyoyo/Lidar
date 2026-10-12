"""Actual small optimizer/checkpoint/evaluator evidence, including negative gates."""
import importlib
import json

import pytest
import torch

from benchmark_fixtures import ROOT, evaluation_fixture


def evidence_module():
    assert (ROOT/'tools/benchmarks/benchmark_evidence.py').is_file(), 'Reproduced benchmark evidence recorder is missing'
    return importlib.import_module('tools.benchmarks.benchmark_evidence')


def test_actual_full_local_bev_evaluation_and_minimum_loss_selection_are_recorded(tmp_path):
    module=evidence_module()
    protocol,evaluation,selection,history,_,_,_=evaluation_fixture(tmp_path)
    result=module.record_evidence(protocol,evaluation,selection,history)
    assert result['status']=='passed' and result['metric_mode']=='local_bev'
    assert result['ap_samplings']==['R40'] and result['full_split_verified']
    assert result['checkpoint_selection_verified'] and result['training_epochs']==2
    assert set(result['metrics']['R40']['per_class'])=={'Car','Pedestrian','Cyclist'}
    assert result['source_files'] and len(result['checkpoint_sha256'])==64
    assert result['selection_policy']=='minimum validation loss'
    assert module.verify_evidence(result)==result


@pytest.mark.parametrize('problem',['failed','partial','split','config','checkpoint_hash','score','nms','alpha','cap',
    'peak','iou','difficulty','macro','missing_class','nan_ap','out_of_range','selection_epoch','selection_loss',
    'history_missing_epoch','history_no_updates','history_better_loss','selected_checkpoint_changed'])
def test_invalid_or_stale_evaluation_selection_and_training_history_is_rejected(tmp_path,problem):
    module=evidence_module()
    protocol,evaluation,selection,history,_,_,_=evaluation_fixture(tmp_path)
    report=json.loads(evaluation.read_text());choice=json.loads(selection.read_text())
    rows=[json.loads(line) for line in history.read_text().splitlines()]
    if problem=='failed':report['status']='error'
    elif problem=='partial':report['data']['frames']=0
    elif problem=='split':report['data']['split_sha256']='0'*64
    elif problem=='config':report['data']['resolved_config_sha256']='0'*64
    elif problem=='checkpoint_hash':report['model']['sha256']='0'*64
    elif problem=='score':report['protocol']['score_threshold']=.6
    elif problem=='nms':report['protocol']['nms_threshold']=.8
    elif problem=='alpha':report['protocol']['nms_alpha']=0.
    elif problem=='cap':report['protocol']['max_detections_per_frame']=3
    elif problem=='peak':report['protocol']['peak_mode']='legacy'
    elif problem=='iou':report['protocol']['iou_thresholds']['Car']=.5
    elif problem=='difficulty':report['protocol']['difficulty_rules']['Easy']['min_height']=0.
    elif problem=='macro':report['accuracy']['mean_ap_9_percent']+=1.
    elif problem=='missing_class':report['accuracy']['per_class'].pop('Cyclist')
    elif problem=='nan_ap':report['accuracy']['per_class']['Car']['difficulties']['Easy']['ap_r40_percent']=float('nan')
    elif problem=='out_of_range':report['accuracy']['per_class']['Car']['difficulties']['Easy']['ap_r40_percent']=101.
    elif problem=='selection_epoch':choice['epoch']=99
    elif problem=='selection_loss':choice['validation_objective']+=1.
    elif problem=='history_missing_epoch':rows=rows[:1]
    elif problem=='history_no_updates':rows[0]['optimizer_updates']=0
    elif problem=='history_better_loss':rows[0]['validation']['loss']=-1.
    else:
        with open(report['model']['path'],'ab') as stream:stream.write(b'changed')
    evaluation.write_text(json.dumps(report));selection.write_text(json.dumps(choice))
    history.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError):module.record_evidence(protocol,evaluation,selection,history)


def test_reverification_rejects_edited_source_report(tmp_path):
    module=evidence_module()
    args=evaluation_fixture(tmp_path)
    result=module.record_evidence(*args[:4])
    report=json.loads(args[1].read_text());report['name']='changed';args[1].write_text(json.dumps(report))
    with pytest.raises(ValueError,match='stale|changed|match'):module.verify_evidence(result)


def test_synthetic_flag_is_preserved_for_freeze_rejection(tmp_path):
    module=evidence_module();args=evaluation_fixture(tmp_path)
    result=module.record_evidence(*args[:4],synthetic=True)
    assert result['synthetic'] is True


@pytest.mark.parametrize('mode',['bev','3d'])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Reference overlaps require actual compatible CUDA')
def test_actual_reference_checkpoint_evidence_retains_mode_both_samplings_and_pin(tmp_path,mode):
    module=evidence_module();args=evaluation_fixture(tmp_path,metric_mode=mode)
    result=module.record_evidence(*args[:4],synthetic=True)
    assert result['metric_mode']==mode and result['ap_samplings']==['R11','R40']
    assert result['reference']['revision']=='233f849829b6ac19afb8af8837a0246890908755'
    assert module.verify_evidence(result)==result


@pytest.mark.parametrize('asset',['processed_point','raw_label','raw_calibration'])
def test_actual_evaluator_records_asset_hashes_and_evidence_rejects_changed_inputs(tmp_path,asset):
    module=evidence_module();args=evaluation_fixture(tmp_path)
    report=json.loads(args[1].read_text())
    assert report['data'].get('input_asset_sha256'), 'Evaluator input asset provenance is missing'
    paths={'processed_point':args[5]/'pointcloud/000002.bin',
           'raw_label':args[6]/'label_2/000002.txt','raw_calibration':args[6]/'calib/000002.txt'}
    path=paths[asset]
    with path.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='asset|input|changed'):
        module.record_evidence(*args[:4])


@pytest.mark.parametrize('problem',['train_split','quality_target'])
def test_training_split_freshness_and_quality_target_are_verified(tmp_path,problem):
    module=evidence_module();args=evaluation_fixture(tmp_path)
    if problem=='train_split':
        (tmp_path/'train.txt').write_text('000000;kitti\n')
    else:
        report=json.loads(args[1].read_text());report['protocol']['quality_target']='different'
        args[1].write_text(json.dumps(report))
    with pytest.raises(ValueError):module.record_evidence(*args[:4])


def test_actual_iqa_off_checkpoint_evidence_has_matching_quality_semantics(tmp_path):
    import inspect
    assert 'iqa' in inspect.signature(evaluation_fixture).parameters, 'IQA-off evidence integration fixture is missing'
    args=evaluation_fixture(tmp_path,iqa=False)
    result=evidence_module().record_evidence(*args[:4],synthetic=True)
    protocol=json.loads(args[0].read_text())
    assert protocol['decode']['nms_alpha']==0.
    assert protocol['decode']['quality_semantics']=='no IoU quality ranking'
    assert result['status']=='passed' and result['full_split_verified']

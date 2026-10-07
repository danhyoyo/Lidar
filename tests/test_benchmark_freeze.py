"""Task51 consumes actual asset/selection/evaluation evidence, never declarations alone."""
import copy
import inspect
import json

import pytest
import torch

from benchmark_fixtures import ROOT, evaluation_fixture
from tools.benchmarks.audit_kitti_assets import audit_assets
from tools.benchmarks.benchmark_evidence import record_evidence
from tools.benchmarks.benchmark_protocol import freeze_protocol, compare_protocols


def required_api():
    assert 'asset_audit' in inspect.signature(freeze_protocol).parameters, 'Asset audit consumption is missing'
    assert 'comparator_evidence' in inspect.signature(freeze_protocol).parameters, 'Reproduced comparator evidence consumption is missing'


def files(tmp_path, *, synthetic=False):
    args=evaluation_fixture(tmp_path/'baseline')
    protocol,evaluation,selection,history,config_path,processed,raw=args
    config=json.loads(config_path.read_text());config['model']['c4_context']='focal'
    config['experiment']['name']='candidate_focal'
    candidate=tmp_path/'candidate.json';candidate.write_text(json.dumps(config))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(audit_assets(candidate,kitti_root=raw)))
    evidence=tmp_path/'evidence.json';evidence.write_text(json.dumps(record_evidence(protocol,evaluation,selection,history,synthetic=synthetic)))
    return candidate,audit,evidence,args


def test_asset_and_actual_baseline_clear_only_their_gates_while_gpu_stays_pending(tmp_path):
    required_api();candidate,audit,evidence,_=files(tmp_path)
    result=freeze_protocol(candidate,metric_mode='local_bev',asset_audit=audit,comparator_evidence=[evidence])
    assert result['pending_gates']==['final_BEV_GPU_recipe']
    assert result['configured_data']['audit_complete'] and result['baseline_accuracy_verified']
    assert result['comparison_status']=='matched protocols and reproduced evaluation evidence'
    assert not result['long_training_allowed'] and not result['accuracy_measured']
    assert len(result['comparators'])==1 and result['comparators'][0]['matched']


def test_supplied_synthetic_evidence_cannot_enable_benchmark_training(tmp_path):
    required_api();candidate,audit,evidence,_=files(tmp_path,synthetic=True)
    with pytest.raises(ValueError,match='[Ss]ynthetic'):
        freeze_protocol(candidate,metric_mode='local_bev',asset_audit=audit,comparator_evidence=[evidence])


def test_stale_asset_report_cannot_clear_data_gate(tmp_path):
    required_api();candidate,audit,_,args=files(tmp_path)
    with (args[5]/'pointcloud/000001.bin').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError):freeze_protocol(candidate,metric_mode='local_bev',asset_audit=audit)


def test_different_candidate_roi_or_quality_cannot_match_a_comparator(tmp_path):
    required_api();candidate,audit,evidence,_=files(tmp_path)
    result=freeze_protocol(candidate,metric_mode='local_bev')
    other=copy.deepcopy(result);other['evaluation']['roi_geometry']['x_max']+=1
    assert not compare_protocols(result,other)['matched']
    other=copy.deepcopy(result);other['decode']['nms_alpha']=0.
    assert 'decode.nms_alpha' in compare_protocols(result,other)['mismatches']
    other=copy.deepcopy(result);other['decode']['peak_mode']='legacy'
    assert 'decode.peak_mode' in compare_protocols(result,other)['mismatches']


def test_comparator_protocol_or_evidence_alone_cannot_clear_baseline_gate(tmp_path):
    required_api();candidate,audit,evidence,args=files(tmp_path)
    result=freeze_protocol(candidate,metric_mode='local_bev',asset_audit=audit,comparators=[args[0]])
    assert 'comparator_protocols_and_reproduced_baseline' in result['pending_gates']
    assert result['comparators'][0]['matched'] and not result['long_training_allowed']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA smoke required for freeze integration')
def test_actual_cuda_report_asset_audit_and_reproduced_baseline_freeze_local_protocol(tmp_path):
    from tools.benchmarks.smoke_detector import run_smoke
    required_api();candidate,audit,evidence,_=files(tmp_path)
    report=run_smoke(json.loads(candidate.read_text()),device='cuda',steps=3,num_workers=0,full_resolution=True)
    smoke=tmp_path/'smoke.json';smoke.write_text(json.dumps(report))
    result=freeze_protocol(candidate,metric_mode='local_bev',asset_audit=audit,comparator_evidence=[evidence],smoke_reports=[smoke])
    assert result['pending_gates']==[] and result['long_training_allowed']
    assert result['status']=='protocol frozen for supplied verified evidence'
    assert result['current_runtime']['gpu_recipe_verified'] and result['baseline_accuracy_verified']
    # This is a branch-control test on synthetic assets, not a real KITTI freeze or accuracy claim.

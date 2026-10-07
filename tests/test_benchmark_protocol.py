"""Record honest 3D/BEV protocol boundaries and reject unmatched comparisons."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/kitti_training_pipeline")]
CONFIG = ROOT / "configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json"


def protocol():
    assert (ROOT / "tools/benchmarks/benchmark_protocol.py").is_file(), "Protocol recorder is missing"
    return importlib.import_module("tools.benchmarks.benchmark_protocol")


@pytest.mark.parametrize("mode", ["3d", "bev"])
def test_protocol_records_both_recall_samplings_without_claiming_reference_readiness(mode):
    result = protocol().freeze_protocol(CONFIG, ROOT, metric_mode=mode)
    assert result["evaluation"]["metric_mode"] == mode
    assert result["evaluation"]["ap_samplings"] == ["R11", "R40"]
    assert result["evaluation"]["classes"] == ["Car", "Pedestrian", "Cyclist"]
    assert result["evaluation"]["iou_thresholds"] == {"Car": .7, "Pedestrian": .5, "Cyclist": .5}
    assert result["evaluation"]["domain"] == "full benchmark GT; ROI metrics separate"
    assert result["candidate_config"]["data"]["box_mode"] == "bev"
    assert result["candidate_config"]["model"]["local_attention"] == "none"
    assert result["current_runtime"]["metric_mode"] == "bev"
    assert result["current_runtime"]["ap_samplings"] == ["R40"]
    assert result["checkpoint_selection"]["policy"] == "minimum validation loss"
    assert result["long_training_allowed"] is False
    assert result["comparison_status"] == "pending comparator protocols"
    assert result["splits"]["overlap"] == []
    assert result["splits"]["train"]["count"] + result["splits"]["val"]["count"] == 7481
    assert result["aggregation"]["moderate_macro_mean_classes"] == ["Car", "Pedestrian", "Cyclist"]
    assert len(result["source_config_sha256"]) == 64
    assert "reference_evaluator_revision" in result["pending_gates"]


@pytest.mark.parametrize("field", ["metric_mode", "ap_samplings", "classes", "iou_thresholds", "domain"])
def test_comparator_protocol_mismatch_is_explicit(field):
    expected = protocol().freeze_protocol(CONFIG, ROOT)
    changed = copy.deepcopy(expected)
    changed["evaluation"][field] = "different"
    comparison = protocol().compare_protocols(expected, changed)
    assert comparison["matched"] is False and f"evaluation.{field}" in comparison["mismatches"]


def test_comparator_split_or_selection_mismatch_cannot_be_called_matched():
    expected = protocol().freeze_protocol(CONFIG, ROOT)
    same = copy.deepcopy(expected)
    assert protocol().compare_protocols(expected, same)["matched"]
    same["splits"]["val"]["sha256"] = "changed"
    same["checkpoint_selection"]["policy"] = "best validation AP"
    mismatches = protocol().compare_protocols(expected, same)["mismatches"]
    assert "splits.val.sha256" in mismatches and "checkpoint_selection.policy" in mismatches
    assert not protocol().compare_protocols(expected, {})["matched"]


@pytest.mark.parametrize("mode", ["AP3D", "typo", None])
def test_unknown_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="metric_mode"):
        protocol().freeze_protocol(CONFIG, ROOT, metric_mode=mode)


@pytest.mark.parametrize("field", ["score_threshold", "nms_threshold", "max_detections"])
def test_decode_threshold_and_limit_mismatch_is_recorded(field):
    expected = protocol().freeze_protocol(CONFIG, ROOT)
    changed = copy.deepcopy(expected)
    changed["decode"][field] = 999
    assert f"decode.{field}" in protocol().compare_protocols(expected, changed)["mismatches"]


def test_protocol_cli_records_metric_mode_and_comparator_mismatch(tmp_path):
    reference = protocol().freeze_protocol(CONFIG, ROOT)
    mismatch = copy.deepcopy(reference)
    mismatch["evaluation"]["metric_mode"] = "bev"
    comparator = tmp_path / "comparator.json"
    comparator.write_text(json.dumps(mismatch))
    output = tmp_path / "protocol.json"
    completed = subprocess.run([sys.executable, str(ROOT / "tools/benchmarks/benchmark_protocol.py"),
        "--config", str(CONFIG), "--metric-mode", "3d", "--comparator-protocol", str(comparator),
        "--output", str(output)], cwd=tmp_path, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["comparison_status"] == "mismatched comparator protocols"
    assert not result["long_training_allowed"]
    assert "evaluation.metric_mode" in result["comparators"][0]["mismatches"]


@pytest.mark.parametrize("invalid", ["overlap", "duplicates"])
def test_invalid_split_cannot_be_recorded_as_valid_protocol(tmp_path, invalid):
    config = json.loads(CONFIG.read_text())
    (tmp_path / "train.txt").write_text("000000;kitti\n000000;kitti\n" if invalid == "duplicates" else "000000;kitti\n")
    (tmp_path / "val.txt").write_text("000000;kitti\n" if invalid == "overlap" else "000001;kitti\n")
    config["train"]["data"] = "train.txt"
    config["val"]["data"] = "val.txt"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="disjoint|unique"):
        protocol().freeze_protocol(path, tmp_path)


@pytest.mark.parametrize("mode", ["3d","bev"])
def test_complete_3d_protocol_records_reference_runtime_and_retains_external_gates(mode):
    config = ROOT/"configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa_3d.json"
    result = protocol().freeze_protocol(config,ROOT,metric_mode=mode)
    assert result["current_runtime"]["metric_mode"] == mode
    assert result["current_runtime"]["ap_samplings"] == ["R11","R40"]
    assert result["current_runtime"]["box_mode"] == "3d"
    assert result["evaluation"]["reference_revision"] == "233f849829b6ac19afb8af8837a0246890908755"
    assert result["current_runtime"]["reference_equivalence_verified"] is False
    assert "reference_R11_R40_parity" in result["pending_gates"]
    assert "reference_evaluator_revision" not in result["pending_gates"]
    assert result["long_training_allowed"] is False
    assert "configured_data_and_train_only_gt_database_audit" in result["pending_gates"]


def test_cpu_smoke_cannot_be_used_as_gpu_protocol_evidence(tmp_path):
    from tools.benchmarks.smoke_detector import run_smoke
    config = ROOT/"configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa_3d.json"
    report = run_smoke(json.loads(config.read_text()),device="cpu",precisions=["fp32"],steps=1)
    path = tmp_path/"cpu.json";path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="CUDA|GPU"):
        protocol().freeze_protocol(config,ROOT,smoke_reports=[path])


@pytest.mark.parametrize("problem",["config","precision","failed","vertical","workers"])
def test_invalid_declared_smoke_evidence_cannot_clear_gpu_gate(tmp_path,problem):
    import hashlib
    config_path = ROOT/"configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa_3d.json"
    config = json.loads(config_path.read_text())
    # Malformed records are tested only for rejection; these are not GPU evidence.
    report = {"status":"passed","device":"cuda:0","cuda_device_name":"declared",
              "box_mode":"3d","source_config_sha256":hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
              "worker_count":0,"precisions":{p:{"status":"passed","vertical_gradient_verified":True,
                  "checkpoint_restore_verified":True,"vertical_head_weight_changed":{"car":True,"ped_cyc":True}}
                  for p in ("fp32","fp16","bf16")}}
    if problem=="config":report["source_config_sha256"]="stale"
    elif problem=="precision":del report["precisions"]["bf16"]
    elif problem=="failed":report["status"]="partial"
    elif problem=="vertical":report["precisions"]["fp32"]["vertical_gradient_verified"]=False
    else:report["worker_count"]=-1
    path=tmp_path/"invalid.json";path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="smoke|config|precision|vertical|worker"):
        protocol().freeze_protocol(config_path,ROOT,smoke_reports=[path])


def test_invalid_reference_revision_cannot_clear_parity_gate(tmp_path):
    config = ROOT/"configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa_3d.json"
    record = {"reference_provenance":{"revision":"moving-master","sha256":{}},
              "gpu_reference_cases":19,"full":{"exit_code":0,"skips":0}}
    path=tmp_path/"reference.json";path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="reference"):
        protocol().freeze_protocol(config,ROOT,reference_verification=path)


def test_explicit_local_bev_policy_records_only_actual_r40_and_roi():
    result=protocol().freeze_protocol(CONFIG,ROOT,metric_mode='local_bev')
    assert result['evaluation']['metric_mode']=='local_bev'
    assert result['evaluation']['ap_samplings']==['R40']
    assert result['evaluation']['domain']=='strict configured XY-center ROI'
    assert result['evaluation']['roi_geometry']['x_max']==70.4
    assert result['current_runtime']['ap_samplings']==['R40']
    assert 'reference_R11_R40_parity' not in result['pending_gates']
    assert 'final_BEV_GPU_recipe' in result['pending_gates']
    assert result['training']['required_3d_extension']=='not required for local BEV development'


def test_protocol_cli_defaults_to_current_local_bev_workflow(tmp_path):
    output=tmp_path/'protocol.json'
    completed=subprocess.run([sys.executable,str(ROOT/'tools/benchmarks/benchmark_protocol.py'),
        '--config',str(CONFIG),'--output',str(output)],text=True,capture_output=True)
    assert completed.returncode==0,completed.stderr
    assert json.loads(output.read_text())['evaluation']['metric_mode']=='local_bev'


def test_local_bev_policy_rejects_a_3d_runtime_config():
    path=ROOT/'configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa_3d.json'
    with pytest.raises(ValueError,match='BEV|bev|3D'):
        protocol().freeze_protocol(path,ROOT,metric_mode='local_bev')


@pytest.mark.parametrize('problem',['partial','hash','precision','restore','updates','shape','budget'])
def test_invalid_local_bev_gpu_evidence_is_rejected(tmp_path,problem):
    from tools.benchmarks.audit_kitti_assets import canonical_hash
    from common import input_shape
    config=json.loads(CONFIG.read_text())
    flags=('finite_loss','finite_gradients','head_weight_changed','optimizer_membership_exact','checkpoint_restore_verified',
        'optimizer_state_restore_verified','scheduler_scaler_rng_restore_verified','criterion_state_unchanged_in_validation','full_resolution_finite')
    run={'status':'passed',**dict.fromkeys(flags,True),'optimizer_updates':3,
         'full_resolution_input_shape':list(input_shape(config)), 'parameter_counts':{'backbone_including_neck':660528}}
    # Deliberately malformed declared records test rejection, not actual GPU readiness.
    report={'status':'passed','device':'cuda:0','cuda_device_name':'declared','box_mode':'bev',
        'source_config_sha256':canonical_hash(config),'worker_count':0,
        'precisions':{key:copy.deepcopy(run) for key in ('fp32','fp16','bf16')}}
    if problem=='partial':report['status']='partial'
    elif problem=='hash':report['source_config_sha256']='changed'
    elif problem=='precision':report['precisions'].pop('bf16')
    elif problem=='restore':report['precisions']['fp32']['optimizer_state_restore_verified']=False
    elif problem=='updates':report['precisions']['fp32']['optimizer_updates']=0
    elif problem=='shape':report['precisions']['fp32']['full_resolution_input_shape']=[1,14,32,32]
    else:report['precisions']['fp32']['parameter_counts']['backbone_including_neck']=1_000_000
    path=tmp_path/'gpu.json';path.write_text(json.dumps(report))
    with pytest.raises(ValueError):protocol().freeze_protocol(CONFIG,metric_mode='local_bev',smoke_reports=[path])


def test_no_context_reference_protocol_records_its_actual_attention():
    path=ROOT/'configs/experiments/under1m/hist14_local3_grouped_oga_iqa.json'
    result=protocol().freeze_protocol(path,metric_mode='local_bev')
    assert result['training']['main_attention'].startswith('none, local none')


@pytest.mark.parametrize('iqa',[False,True])
def test_local_bev_protocol_records_effective_quality_and_peak_settings(tmp_path,iqa):
    config=json.loads(CONFIG.read_text())
    config['model']['header_use_iou']=iqa;config['loss']['use_iou']=iqa
    config['nms_alpha']=.8;config['data']['kitti']['nms_alpha']=.7
    config['peak_mode']='legacy'
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    result=protocol().freeze_protocol(path,metric_mode='local_bev')
    assert result['decode']['nms_alpha']==(.8 if iqa else 0.)
    assert result['decode']['peak_mode']=='legacy'

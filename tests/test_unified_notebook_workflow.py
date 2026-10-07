"""One-master-file resolution and actual selected-run Colab workflow contracts."""
import copy
import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from benchmark_fixtures import ROOT, evaluation_fixture
from tools.kitti_training_pipeline import notebook_config


def workflow():
    path = ROOT / 'tools/kitti_training_pipeline/notebook_workflow.py'
    assert path.is_file(), 'Checked notebook workflow helpers are missing'
    return importlib.import_module('tools.kitti_training_pipeline.notebook_workflow')


@pytest.mark.parametrize('preset', list(notebook_config.UNDER1M_PRESETS))
def test_named_presets_need_only_master_json(tmp_path, preset):
    (tmp_path / 'configs').mkdir()
    shutil.copyfile(ROOT / 'configs/config.json', tmp_path / 'configs/config.json')
    actual = notebook_config.resolve_notebook_config(tmp_path, preset=preset, augmentation='config')
    expected = notebook_config.resolve_notebook_config(ROOT, preset=preset, augmentation='config')
    assert actual == expected


@pytest.mark.parametrize('profile', ['hybrid_gt', 'openpcdet_gt', 'openpcdet_global'])
def test_augmentation_needs_only_master_json_and_matches_existing_recipe(tmp_path, profile):
    (tmp_path / 'configs').mkdir()
    shutil.copyfile(ROOT / 'configs/config.json', tmp_path / 'configs/config.json')
    actual = notebook_config.resolve_notebook_config(tmp_path, preset='custom', augmentation=profile)
    expected = json.loads((ROOT / f'configs/augmentation/{profile}.json').read_text())['augmentation']
    assert actual['augmentation'] == expected


def actual_run(tmp_path, **options):
    args = evaluation_fixture(tmp_path, **options)
    shutil.copyfile(args[4], tmp_path / 'config.resolved.json')
    return args


def test_selected_run_uses_own_resolved_config_and_minimum_loss_checkpoint(tmp_path):
    module = workflow(); actual_run(tmp_path)
    result = module.selected_run(tmp_path)
    assert result['config_path'] == tmp_path / 'config.resolved.json'
    assert result['checkpoint'] == tmp_path / 'selected/best.pt'
    assert result['config']['model']['c4_context'] == 'none'
    assert result['completed_epochs'] == 2


@pytest.mark.parametrize('problem', ['missing_best', 'missing_resolved', 'partial_history', 'last_only', 'bad_selection'])
def test_selected_run_does_not_substitute_last_checkpoint_or_incomplete_training(tmp_path, problem):
    module = workflow(); actual_run(tmp_path)
    if problem in {'missing_best', 'last_only'}:
        (tmp_path / 'checkpoints').mkdir()
        shutil.copyfile(tmp_path / 'selected/best.pt', tmp_path / 'checkpoints/last.pt')
        (tmp_path / 'selected/best.pt').unlink()
    elif problem == 'missing_resolved': (tmp_path / 'config.resolved.json').unlink()
    elif problem == 'partial_history':
        path = tmp_path / 'metrics.jsonl'; path.write_text(path.read_text().splitlines()[0]+'\n')
    else:
        path = tmp_path / 'selected/selection.json'; value = json.loads(path.read_text())
        value['epoch'] = 99; path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)): module.selected_run(tmp_path)


@pytest.mark.parametrize('box,modes', [('bev',['3d']), ('bev',['bev']), ('3d',['local_bev']),
    ('bev',[]), ('bev',['APBEV']), ('bev',['local_bev','local_bev'])])
def test_invalid_metric_modes_are_rejected(box, modes):
    module = workflow()
    with pytest.raises(ValueError): module.validate_modes({'data':{'box_mode':box}}, modes)


def test_explicit_3d_accepts_both_final_reference_metrics():
    assert workflow().validate_modes({'data':{'box_mode':'3d'}}, ['3d','bev']) == ('3d','bev')


def test_local_evaluation_command_uses_own_config_split_and_configured_decode(tmp_path):
    module = workflow(); args = actual_run(tmp_path)
    run = module.selected_run(tmp_path)
    run['config']['evaluation'] = {'score_threshold':.07,'nms_threshold':.2,'max_detections':123}
    command = module.evaluation_command(run, 'local_bev', args[6], tmp_path/'out.json', repo_root=ROOT, device='cpu')
    assert command[0] == sys.executable
    assert command[1].endswith('evaluate_kitti_bev.py')
    assert command[command.index('--config')+1] == str(tmp_path/'config.resolved.json')
    assert command[command.index('--split')+1] == run['config']['val']['data']
    assert command[command.index('--model')+1] == str(tmp_path/'selected/best.pt')
    assert command[command.index('--kitti-root')+1] == str(args[6].parent)
    assert command[command.index('--score-threshold')+1] == '0.07'
    assert command[command.index('--max-detections')+1] == '123'
    assert not any('deploy' in value for value in command)


def test_reference_command_routes_own_3d_config_without_changing_local_config(tmp_path):
    module = workflow(); args = actual_run(tmp_path)
    run = module.selected_run(tmp_path)
    run['config']['data']['box_mode'] = '3d'
    command = module.evaluation_command(run, 'bev', args[6], tmp_path/'out.json', repo_root=ROOT)
    assert command[1].endswith('evaluate_kitti_3d.py')
    assert command[command.index('--metric-mode')+1] == 'bev'
    assert '--checkpoint' in command and '--model' not in command


def test_actual_evaluator_metrics_render_from_accuracy_not_missing_top_level_keys(tmp_path):
    module = workflow(); args = actual_run(tmp_path)
    report = json.loads(args[1].read_text())
    report['accuracy']['map_moderate_percent'] = 71.25
    report['accuracy']['mean_ap_9_percent'] = 69.75
    rows = module.evaluation_rows(report)
    assert rows[0]['sampling'] == 'R40' and rows[0]['metric'] == 'local_bev'
    assert rows[0]['map_moderate_percent'] == 71.25
    assert rows[0]['mean_ap_9_percent'] == 69.75
    assert 'Car_Moderate' in rows[0] and 'Cyclist_Hard' in rows[0]


def test_reference_rows_keep_both_samplings_and_true_zero_results():
    module = workflow()
    accuracy = {sampling:{'map_moderate_percent':value,'mean_ap_9_percent':value,
        'per_class':{name:{difficulty:value for difficulty in ['Easy','Moderate','Hard']}
                     for name in ['Car','Pedestrian','Cyclist']}} for sampling,value in [('R11',75.),('R40',0.)]}
    rows = module.evaluation_rows({'status':'ok','name':'final','protocol':{'metric_mode':'3d'},'accuracy':accuracy})
    assert [row['sampling'] for row in rows] == ['R11','R40']
    assert rows[1]['map_moderate_percent'] == 0.


def test_missing_or_failed_metrics_raise_instead_of_printing_zero():
    for report in ({'status':'error'}, {'status':'ok','accuracy':{},'protocol':{}}):
        with pytest.raises((ValueError,KeyError)): workflow().evaluation_rows(report)


def test_training_command_preserves_config_runtime_and_excludes_shell_interpolation(tmp_path):
    module = workflow()
    config = json.loads((ROOT/'configs/config.json').read_text())
    config['train'].update(num_workers=6,target_backend='numba',compile_model=True)
    path=tmp_path/'config with spaces.json';path.write_text(json.dumps(config))
    command=module.training_command(path,tmp_path,'run_main',repo_root=ROOT,resume=tmp_path/'last.pt')
    from tools.kitti_training_pipeline.train import build_parser
    parsed=build_parser().parse_args(command[2:])
    assert parsed.config==path and parsed.num_workers==6 and parsed.target_backend=='numba'
    assert parsed.compile_model and parsed.resume==tmp_path/'last.pt'
    assert parsed.precision==config['train']['precision']


@pytest.mark.parametrize('purpose,records,allowed', [('development',[],True),('benchmark',[],False),
    ('benchmark',[{'long_training_allowed':False,'pending_gates':['baseline']}],False),
    ('benchmark',[{'long_training_allowed':True,'pending_gates':[]}],True),
    ('benchmark',[{'long_training_allowed':True,'pending_gates':['data']}],False)])
def test_benchmark_train_gate_cannot_be_bypassed_by_missing_or_partial_records(purpose,records,allowed):
    module=workflow()
    if allowed:assert module.require_training_ready(purpose,records) is True
    else:
        with pytest.raises(ValueError):module.require_training_ready(purpose,records)


def test_preflight_explains_missing_uploaded_or_remote_source(tmp_path):
    with pytest.raises(FileNotFoundError,match='checkout|upload|branch'):
        workflow().source_preflight(tmp_path)


@pytest.mark.parametrize('consumer',['selected_run','evidence'])
def test_independently_serialized_selected_and_retained_checkpoints_are_compatible(tmp_path,consumer):
    args=actual_run(tmp_path)
    chosen=tmp_path/'selected/best.pt'
    saved=torch.load(chosen,map_location='cpu',weights_only=True)
    torch.save(saved,chosen)  # Actual trainer serializes each destination independently.
    retained=Path(json.loads(args[2].read_text())['checkpoint'])
    from tools.kitti_training_pipeline.common import sha256
    assert sha256(chosen)!=sha256(retained)
    if consumer=='selected_run':assert workflow().selected_run(tmp_path)['completed_epochs']==2
    else:
        from tools.kitti_training_pipeline.evaluate_kitti_bev import run_evaluation
        from tools.benchmarks.benchmark_evidence import record_evidence
        run_evaluation(name='internal_no_context',backend='pytorch',model_path=chosen,config_path=args[4],
            detector_root=ROOT/'detector',kitti_root=args[6].parent,split_path=tmp_path/'val.txt',
            output_path=args[1],device='cpu',warmup_frames=0,progress_every=0)
        assert record_evidence(*args[:4],synthetic=True)['status']=='passed'


@pytest.mark.parametrize('state',['model_state_dict','criterion_state_dict','optimizer_state_dict','rng_state','config'])
def test_retained_checkpoint_requires_all_training_state_not_only_model_identity(tmp_path,state):
    args=actual_run(tmp_path)
    chosen=tmp_path/'selected/best.pt'
    saved=torch.load(chosen,map_location='cpu',weights_only=True)
    if state in {'model_state_dict','criterion_state_dict'}:
        key=next(iter(saved[state]));saved[state][key]=saved[state][key]+1
    elif state=='optimizer_state_dict':saved[state]['param_groups'][0]['lr']+=.001
    elif state=='rng_state':saved[state]['torch_cpu'][0]^=1
    else:saved['config']['seed']+=1
    torch.save(saved,chosen)
    with pytest.raises(ValueError,match='retained|identity|state'):workflow().selected_run(tmp_path)


def test_configurable_decode_is_shared_by_protocol_and_notebook(tmp_path):
    from tools.benchmarks.benchmark_protocol import freeze_protocol
    args=actual_run(tmp_path)
    config=json.loads(args[4].read_text())
    config['evaluation']={'score_threshold':.07,'nms_threshold':.2,'max_detections':123}
    args[4].write_text(json.dumps(config))
    result=freeze_protocol(args[4],metric_mode='local_bev')
    assert result['decode']['score_threshold']==.07
    assert result['decode']['nms_threshold']==.2
    assert result['decode']['max_detections']==123


@pytest.mark.parametrize('settings', [{'score_threshold':float('nan')},{'nms_threshold':2.},
    {'max_detections':0},{'max_detections':True}])
def test_invalid_notebook_decode_settings_fail_before_training(settings):
    config=notebook_config.resolve_notebook_config(ROOT,preset='UNDER1M_FOCAL_OGA_IQA',augmentation='config')
    config['evaluation']=settings
    with pytest.raises(ValueError):notebook_config.validate_notebook_config(config)


@pytest.mark.parametrize('field,value',[('nms_alpha',2.),('nms_alpha',float('nan')),
    ('nms_alpha',True),('peak_mode','invalid')])
def test_notebook_quality_and_peak_controls_fail_before_training(field,value):
    config=notebook_config.resolve_notebook_config(ROOT,preset='UNDER1M_FOCAL_OGA_IQA',augmentation='config')
    config['data']['kitti'][field]=value
    with pytest.raises(ValueError):notebook_config.validate_notebook_config(config)


@pytest.mark.parametrize('problem',['seed','schedule','augmentation','loss','encoding','heads','local_attention'])
def test_focal_ablation_rejects_changes_to_other_controlled_conditions(problem):
    module=workflow()
    assert hasattr(module,'validate_comparator'), 'Controlled focal ablation validation is missing'
    candidate=notebook_config.resolve_notebook_config(ROOT,preset='UNDER1M_FOCAL_OGA_IQA',augmentation='config')
    baseline=notebook_config.resolve_notebook_config(ROOT,preset='UNDER1M_GROUPED_OGA_IQA',augmentation='config')
    if problem=='seed':baseline['seed']+=1
    elif problem=='schedule':baseline['train']['epochs']+=1
    elif problem=='augmentation':baseline['augmentation']['AUG_CONFIG_LIST'][0]['PROBABILITY']=.3
    elif problem=='loss':baseline['loss']['name']='baseline'
    elif problem=='encoding':baseline['data']['bev_encoding']['backend']='numba'
    elif problem=='heads':baseline['model']['backbone_out_dim']=24
    else:baseline['model']['local_attention']='eca'
    with pytest.raises(ValueError,match='ablation|match'):module.validate_comparator(candidate,baseline)


def test_matching_focal_ablation_accepts_normalized_objective_defaults():
    module=workflow()
    assert hasattr(module,'validate_comparator'), 'Controlled focal ablation validation is missing'
    candidate=notebook_config.resolve_notebook_config(ROOT,preset='UNDER1M_FOCAL_OGA_IQA',augmentation='config')
    baseline=notebook_config.resolve_notebook_config(ROOT,preset='UNDER1M_GROUPED_OGA_IQA',augmentation='config')
    candidate['loss']['iou_loss_weight']=1.
    assert module.validate_comparator(candidate,baseline) is True
    baseline['train']['epochs']+=1
    assert module.validate_comparator(candidate,baseline,kind='protocol') is True


def test_notebook_default_custom_controls_reach_the_main_recipe(tmp_path):
    from test_under1m_notebook_controls import execute_cell
    ns=execute_cell(tmp_path)
    assert ns['PRESET']=='custom', 'Notebook controls must apply without first switching preset'
    assert ns['AUGMENTATION']=='hybrid_gt'
    assert ns['config_dict']['data']['box_mode']=='bev'
    assert ns['config_dict']['evaluation']['modes']==['local_bev']
    changed=execute_cell(tmp_path/'other',{'C4_CONTEXT':'none','BACKBONE_OUT_DIM':24})
    assert changed['config_dict']['model']['c4_context']=='none'
    assert changed['config_dict']['model']['backbone_out_dim']==24


def test_notebook_3d_controls_enable_owned_vertical_supervision(tmp_path):
    from test_under1m_notebook_controls import execute_cell
    ns=execute_cell(tmp_path,{'BOX_MODE':'3d','EVALUATION_MODES':['3d','bev'],'VERTICAL_LOSS_WEIGHT':.5})
    assert ns['config_dict']['data']['box_mode']=='3d'
    assert ns['config_dict']['loss']['vertical_loss_weight']==.5
    assert ns['config_dict']['evaluation']['modes']==['3d','bev']


def test_all_notebook_code_cells_transform_and_compile():
    from IPython.core.interactiveshell import InteractiveShell
    notebook=json.loads((ROOT/'3D_Lidar_Object_Detection_Notebook_standard.ipynb').read_text())
    shell=InteractiveShell.instance()
    for index,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':compile(shell.input_transformer_manager.transform_cell(''.join(cell['source'])),f'cell{index}','exec')
    source='\n'.join(''.join(c['source']) for c in notebook['cells'])
    for term in ['research/mobilepixornext-under1m', 'evaluation_rows', 'selected_runs',
                 'SMOKE_ROOT', 'RESUME_ARGUMENT', '!{q(sys.executable)} -u']:
        assert term in source,f'Missing notebook workflow: {term}'
    assert 'CONFIG_OVERRIDE =' not in source


@pytest.mark.parametrize('box_mode', ['bev', '3d'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual GPU trainer/evaluator required')
def test_actual_notebook_cells_complete_a_small_run_with_direct_commands(tmp_path, box_mode):
    from test_simple_notebook_workflow import run_notebook_training
    run = run_notebook_training(tmp_path, box_mode=box_mode, device='cuda')
    assert workflow().selected_run(run)['completed_epochs'] == 2
    assert (run/'selected/best_ap.pt').is_file() and (run/'comparison.csv').is_file()
    comparison = json.loads((run/'comparison.json').read_text())
    assert len(comparison['rows']) == (2 if box_mode == 'bev' else 8)
    assert {row['checkpoint_selection'] for row in comparison['rows']} == {'ap', 'loss'}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual compatible reference CUDA required')
def test_reference_runtime_receipt_requires_actual_unskipped_pinned_cuda_parity(tmp_path):
    result=workflow().verify_reference_runtime(ROOT,tmp_path/'reference.json')
    assert result['status']=='passed' and result['gpu_reference_cases']==16
    assert result['full']['exit_code']==0 and result['full']['skips']==0
    assert result['reference_provenance']['source_hashes_verified']

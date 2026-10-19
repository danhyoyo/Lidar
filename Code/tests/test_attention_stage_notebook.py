"""Users edit notebook controls; saved configs must reflect actual topology."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from attention_stage_test_utils import CASES
from test_under1m_notebook_controls import execute_cell
from tools.kitti_training_pipeline.notebook_config import resolve_notebook_config, save_notebook_config, validate_notebook_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('notebook_name', [
    '3D_Lidar_Object_Detection_Notebook_standard.ipynb',
    '3D_Lidar_Detection_Pipeline_Robust.ipynb',
])
def test_config_cell_uses_reloaded_helpers_when_package_alias_is_stale(notebook_name, tmp_path, monkeypatch):
    import importlib
    import test_under1m_notebook_controls as controls

    # A running Colab kernel may retain the package alias loaded before a Drive sync.
    stale = importlib.import_module('tools.kitti_training_pipeline.notebook_config')
    monkeypatch.delattr(stale, 'attention_placement_overrides')
    monkeypatch.delattr(stale, 'attention_placement_summary')
    notebook = json.loads((ROOT / notebook_name).read_text())
    source = next(''.join(c['source']) for c in notebook['cells']
                  if 'config_dict = resolve_notebook_config(' in ''.join(c['source']))
    monkeypatch.setattr(controls, 'config_cell', lambda: source)
    namespace = {'REPO_DIR': ROOT, 'ARTIFACT_ROOT': tmp_path,
                 'PROCESSED_DATASET_DIR': tmp_path / 'processed', 'RAW_KITTI_ROOT': ROOT / 'Kitti_Raw',
                 'SEED': 42, 'EPOCHS': 8, 'WARMUP_EPOCHS': 1, 'LEARNING_RATE': .0007,
                 'NUM_WORKERS': 0, 'PHYSICAL_BATCH_SIZE': 2, 'ACCUMULATION_STEPS': 1,
                 'PRECISION': 'fp32', 'TARGET_BACKEND': 'python', 'COMPILE_MODEL': False,
                 'VAL_BATCH_SIZE': 3}
    result = controls.execute_cell(tmp_path, {'PRESET': 'custom', 'LOCAL_ATTENTION': 'simam',
        'LOCAL_ATTENTION_STAGE': 'c5', 'AUGMENTATION': 'none'}, namespace)
    model = result['config_dict']['model']
    assert model['local_attention_stage'] == 'c5'
    assert 'SIMAM: C5' in result['attention_placement_summary'](result['config_dict'])


@pytest.mark.parametrize('preset', ['config','UNDER1M_SIMAM_OGA_IQA'])
def test_noncustom_modes_ignore_invalid_unused_stage_controls(preset, tmp_path):
    ns = execute_cell(tmp_path, {'PRESET':preset, 'LOCAL_ATTENTION_STAGE':True})
    assert ns['config_dict']['model'].get('local_attention_stage') in (None,'c3')





def test_generated_robust_controls_preserve_existing_backend_and_head_options(tmp_path):
    subprocess.run([sys.executable,str(ROOT/'tools/generate_robust_notebook.py')],cwd=tmp_path,check=True,capture_output=True)
    notebook = json.loads((tmp_path/'3D_Lidar_Detection_Pipeline_Robust.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells'] if 'config_dict = resolve_notebook_config(' in ''.join(c['source']))
    assert 'BEV_BACKEND = "numba"' in source
    assert 'HEADER_ARCHITECTURE = "independent"' in source
    assert '"header_architecture": HEADER_ARCHITECTURE' in source
    assert 'RUN_REFERENCE_EVALUATION = True' in source
    tracked = json.loads((ROOT/'3D_Lidar_Detection_Pipeline_Robust.ipynb').read_text())
    tracked_source = next(''.join(c['source']) for c in tracked['cells'] if 'config_dict = resolve_notebook_config(' in ''.join(c['source']))
    assert source == tracked_source


@pytest.mark.parametrize('case', CASES)
def test_robust_config_cell_executes_supported_placement_controls(case, tmp_path, monkeypatch):
    import test_under1m_notebook_controls as controls
    notebook = json.loads((ROOT/'3D_Lidar_Detection_Pipeline_Robust.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells'] if 'config_dict = resolve_notebook_config(' in ''.join(c['source']))
    monkeypatch.setattr(controls,'config_cell',lambda:source)
    local = case[1]
    ns = {'REPO_DIR':ROOT,'ARTIFACT_ROOT':tmp_path,'PROCESSED_DATASET_DIR':tmp_path/'processed',
          'RAW_KITTI_ROOT':ROOT/'Kitti_Raw'}
    controls.execute_cell(tmp_path,{'PRESET':'custom','LOCAL_ATTENTION':'simam' if local else 'none',
        'LOCAL_ATTENTION_STAGE':local or 'c3','AUGMENTATION':'none'},ns)
    assert ns['config_dict']['model'].get('local_attention_stage') == local


def test_legacy_trained_snapshot_accepts_materialized_stages_without_rewriting(tmp_path):
    from attention_stage_test_utils import configured
    config,_ = configured(('c4','c3',None))
    validate_notebook_config(config)
    old = copy.deepcopy(config)
    old['model'].pop('local_attention_stage')
    path = save_notebook_config(tmp_path,old)
    original = path.read_bytes()
    (tmp_path/'run.json').write_text('{}')
    assert save_notebook_config(tmp_path,config) == path
    assert path.read_bytes() == original
    changed = copy.deepcopy(config)
    changed['model']['local_attention_stage'] = 'c5'
    with pytest.raises(ValueError):
        save_notebook_config(tmp_path,changed)
    assert path.read_bytes() == original





@pytest.mark.parametrize('case', CASES)
def test_notebook_controls_save_selected_stage(case, tmp_path):
    local = case[1]
    ns = execute_cell(tmp_path, {'PRESET':'custom','LOCAL_ATTENTION':'simam' if local else 'none',
        'LOCAL_ATTENTION_STAGE':local or 'c3','AUGMENTATION':'none'})
    model = ns['config_dict']['model']
    assert model.get('local_attention_stage') == local
    assert json.loads(ns['CONFIG'].read_text())['model'] == model
    from common import build_model
    assert build_model(ns['config_dict']).backbone.local_attention_stage == local


def base_with_shared_attention(tmp_path):
    config = json.loads((ROOT/'configs/config.json').read_text())
    config['model'].update(backbone='mobilepixornext',local_attention='simam',local_attention_stage='c4',
        local_attention_simam_lambda=.0001)
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/config.json').write_text(json.dumps(config))
    return config


@pytest.mark.parametrize('patch', [{'local_attention_stage':'c3'}, {'local_attention':'none'}])
def test_custom_merge_removes_only_inherited_inactive_fields(patch, tmp_path):
    base = base_with_shared_attention(tmp_path)
    original = copy.deepcopy(patch)
    config = resolve_notebook_config(tmp_path,preset='custom',augmentation='config',custom_overrides={'model':patch})
    assert patch == original
    assert json.loads((tmp_path/'configs/config.json').read_text()) == base
    if patch.get('local_attention') == 'none':
        assert 'local_attention_simam_lambda' not in config['model']
        assert 'local_attention_stage' not in config['model']


@pytest.mark.parametrize('patch', [
    {'c4_context':'none','focal_stage':'c4'},
    {'local_attention_stage':'c3','attention_order':'focal_then_local'},
    {'local_attention':'eca','local_attention_simam_lambda':.001},
])
def test_custom_merge_does_not_hide_explicit_invalid_fields(patch, tmp_path):
    base_with_shared_attention(tmp_path)
    with pytest.raises(ValueError):
        resolve_notebook_config(tmp_path,preset='custom',augmentation='config',custom_overrides={'model':patch})





def test_reexecuting_stage_controls_and_presets_leaves_no_stale_placement(tmp_path):
    ns = execute_cell(tmp_path,{'LOCAL_ATTENTION':'simam','LOCAL_ATTENTION_STAGE':'c4'})
    assert ns['config_dict']['model']['local_attention_stage'] == 'c4'
    ns = execute_cell(tmp_path,{'LOCAL_ATTENTION':'simam','LOCAL_ATTENTION_STAGE':'c3'},ns)
    assert ns['config_dict']['model']['local_attention_stage'] == 'c3'
    ns = execute_cell(tmp_path,{'LOCAL_ATTENTION':'none'},ns)
    assert 'local_attention_stage' not in ns['config_dict']['model']
    ns = execute_cell(tmp_path,{'PRESET':'UNDER1M_SIMAM_OGA_IQA','LOCAL_ATTENTION_STAGE':'c5'},ns)
    assert ns['config_dict']['model']['local_attention_stage'] == 'c3'

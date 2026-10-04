"""Controls must isolate augmentation while preserving model/training settings."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/kitti_training_pipeline'))
from common import generate_run_name

DIRECTORY = ROOT / 'configs/kitti/augmentation'
FILES = [
    'a_standard.json', 'b_light_sampling.json', 'c_sampling_density.json',
    'd_sampling_shadow.json', 'e_sampling_intensity.json',
]


def configs():
    return [json.loads((DIRECTORY / name).read_text()) for name in FILES]


def test_only_augmentation_experiments_remain():
    assert {p.relative_to(ROOT / 'configs/kitti').as_posix() for p in (ROOT / 'configs/kitti').rglob('*.json')} == {
        'augmentation/' + name for name in FILES
    }


def test_all_runs_share_model_loss_data_and_training_budget():
    values = configs()
    for config in values:
        for key in ['model', 'loss', 'data', 'train', 'val', 'seed']:
            assert config[key] == values[0][key], key
        assert config['train']['epochs'] == 50
        assert 'stop_after_epoch' not in config['train']
        assert config['train']['warmup_epochs'] == 4
        assert config['train']['grad_clip_norm'] == 10
        assert config['train']['physical_batch_size'] == 16
        assert config['train']['learning_rate'] == .0007
        assert config['train']['weight_decay'] == .001
    assert len({generate_run_name(config, seed=42) for config in values}) == 5


def test_sampling_does_not_change_global_augmentation():
    values = configs()
    def global_aug(config):
        return {k: v for k, v in config['augmentation'].items() if k not in {'use_pcu_aug', 'pcu_aug'}}
    assert all(global_aug(c) == global_aug(values[0]) for c in values)
    aug = values[0]['augmentation']
    assert aug['mode'] == 'one_of' and aug['p'] == .5
    assert aug['translation']['scale_z'] == .4
    assert not aug.get('flip_y', {}).get('use', False)
    assert not aug.get('point_dropout', {}).get('use', False)


@pytest.mark.parametrize('index, enabled', [(1, None), (2, 'enable_density_subsample'),
                                         (3, 'enable_shadow_masking'), (4, 'enable_radiometric_calibration')])
def test_physics_trials_change_only_one_flag_from_light_sampling(index, enabled):
    values = configs()
    baseline = values[1]['augmentation']['pcu_aug']
    trial = values[index]['augmentation']['pcu_aug']
    assert baseline['p'] == .2
    assert baseline['sample_counts'] == {'Pedestrian': 1, 'Cyclist': 1}
    assert baseline['placement_mode'] == 'source_relative'
    assert baseline['range_scale'] == [.8, 1.2]
    expected = dict(baseline)
    if enabled:
        expected[enabled] = True
    assert trial == expected
    assert values[index]['augmentation']['use_pcu_aug'] is True
    assert values[0]['augmentation']['use_pcu_aug'] is False

"""Exercise real CPU training, checkpointing and resume with a fixed LR horizon."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/kitti_training_pipeline'))
from common import generate_run_name
from train import main
from config_helpers import augmentation_config


def test_screening_checkpoints_and_resume_keep_the_original_schedule(tmp_path):
    config = augmentation_config()
    data = tmp_path / 'data'
    (data / 'pointcloud').mkdir(parents=True)
    (data / 'label').mkdir()
    for frame in ['000001', '000002']:
        points = np.array([[3 + i * .02, 0, -1.5 + i * .03, .5] for i in range(30)], np.float32)
        points.tofile(data / 'pointcloud' / f'{frame}.bin')
        (data / 'label' / f'{frame}.txt').write_text('Car 1.5 1.8 4.5 3.2 0 -1.6 0\n')
    manifest = tmp_path / 'frames.txt'
    manifest.write_text('000001;kitti\n000002;kitti\n')
    config['data']['kitti']['location'] = str(data)
    config['data']['kitti']['geometry'].update(x_max=6.4, y_min=-3.2, y_max=3.2)
    config['train']['data'] = config['val']['data'] = str(manifest)
    config['train']['physical_batch_size'] = config['val']['physical_batch_size'] = 2
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    arguments = ['--config', str(path), '--detector-root', str(ROOT / 'detector'),
                 '--output-root', str(tmp_path / 'artifacts'), '--device', 'cpu',
                 '--precision', 'fp32', '--num-workers', '0']
    main(arguments + ['--stop-after-epoch', '2'])
    directory = tmp_path / 'artifacts' / generate_run_name(config, seed=42)
    checkpoint = directory / 'checkpoints/last.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['epoch'] == 2
    assert state['config']['train']['epochs'] == 50
    assert state['scheduler_state_dict']['_schedulers'][1]['T_max'] == 46
    assert (directory / 'checkpoints/2epoch.pt').is_file()

    main(arguments + ['--resume', str(checkpoint), '--stop-after-epoch', '3'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['epoch'] == 3
    assert state['scheduler_state_dict']['last_epoch'] == 3
    assert state['scheduler_state_dict']['_schedulers'][1]['T_max'] == 46
    assert state['optimizer_state_dict']['param_groups'][0]['lr'] > 1e-6
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows] == [1, 2, 3]
    assert all(row['optimizer_updates'] == 1 and np.isfinite(row['validation']['loss']) for row in rows)
    with pytest.raises(ValueError, match='schedule horizon'):
        main(arguments + ['--resume', str(checkpoint), '--epochs', '100', '--stop-after-epoch', '4'])
    assert json.loads((directory / 'config.resolved.json').read_text())['train']['epochs'] == 50

    main(arguments + ['--resume', str(checkpoint)])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['epoch'] == 50
    assert state['scheduler_state_dict']['last_epoch'] == 50
    assert state['optimizer_state_dict']['param_groups'][0]['lr'] == pytest.approx(1e-6)
    assert (directory / 'checkpoints/50epoch.pt').is_file()

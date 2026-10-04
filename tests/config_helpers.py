"""Build model-feature test fixtures independently of branch experiment configs."""
import json
from pathlib import Path


def augmentation_config(name='a_standard.json'):
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / 'configs/kitti/augmentation' / name).read_text())


def oga_config(*, iqa=False, reparam=False, multiscale=False, qk_norm='none'):
    config = augmentation_config()
    config['model'].update(use_reparam=reparam, header_use_iou=iqa,
                           c4_attention_scales=[3, 5] if multiscale else [5],
                           c4_attention_qk_norm=qk_norm)
    config['loss'] = {
        'name': 'oga', 'temperature': 2., 'clamp_bound': 3., 'corner_beta': 1.,
        'ema_momentum': .99, 'epsilon': 1e-6, 'max_abs_log_size': 10.,
    }
    if iqa:
        config['loss'].update(use_iou=True, iou_target_type='mgiou', iou_loss_weight=1.)
        config['data']['kitti']['nms_alpha'] = .5
        config['nms_alpha'] = .5
    return config

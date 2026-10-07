"""Read-only architecture count and shape-based Conv MAC audit; no latency claim."""
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'detector'))
from core.models.model import CustomModel
from core.models.backbones.mobilepixornext_blocks import LiteMLARefinement

cfg = json.loads((ROOT / 'configs/config.json').read_text())
base = dict(cfg['model'], bev_encoding=cfg['data']['bev_encoding'],
            geometry=cfg['data']['kitti']['geometry'])
variants = [
    ('base', {}),
    ('no_attention', {'c4_attention': 'none'}),
    ('head32', {'backbone_out_dim': 32}),
    ('expansion3', {'expansion': 3.0}),
    ('expansion4', {'expansion': 4.0}),
    ('ms_attention', {'c4_attention_scales': [3, 5], 'c4_attention_qk_norm': 'rmsnorm'}),
    ('rc_sgfpn', {'neck_type': 'rc_sgfpn'}),
    ('rc_bisgfpn', {'neck_type': 'rc_bisgfpn'}),
    ('rep_training', {'use_reparam': True}),
    ('rep_deploy', {'use_reparam': True, 'deploy': True}),
]
rows = []
for name, override in variants:
    with torch.device('meta'):
        model = CustomModel(dict(base, **override), num_classes=3)
    parts = {key: sum(p.numel() for p in value.parameters())
             for key, value in model.backbone.named_children()}
    body_names = ['stem', 'down2', 'stage2', 'down3', 'stage3',
                  'c4_attention', 'down4', 'stage4']
    body = sum(parts.get(key, 0) for key in body_names)
    backbone = sum(p.numel() for p in model.backbone.parameters())
    rows.append(dict(name=name, backbone_body=body, neck=backbone-body,
                     backbone_including_neck=backbone,
                     head=sum(p.numel() for p in model.header.parameters()),
                     total=sum(p.numel() for p in model.parameters()), parts=parts))

with torch.device('meta'):
    model = CustomModel(base, num_classes=3).eval()

# Meta device does not support the autocast context. Bypass only the autocast
# wrapper for a shape audit, invoking the actual attention-core implementation.
# This does not measure numerical correctness, accuracy, allocation or latency.
for module in model.modules():
    if isinstance(module, LiteMLARefinement):
        module.linear_attention = types.MethodType(
            lambda self, qkv: self._linear_attention_core(qkv), module)

macs = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        def hook(module, inputs, output, name=name):
            macs[name] = (output.numel() * (module.in_channels // module.groups)
                          * module.kernel_size[0] * module.kernel_size[1])
        module.register_forward_hook(hook)
with torch.no_grad():
    outputs = model(torch.empty((1, 8, 800, 704), device='meta'))

def grouped_gmac(prefixes):
    return sum(value for key, value in macs.items() if key.startswith(prefixes)) / 1e9

result = {
    'method': 'PyTorch meta-device instantiated parameters; Conv2d output-shape MAC hooks',
    'torch_version': torch.__version__,
    'input_shape': [1, 8, 800, 704],
    'limitations': 'No measured accuracy, latency, peak memory or numerical forward. Conv MACs exclude attention matmuls, BN, activations, interpolation, elementwise and preprocessing.',
    'variants': rows,
    'output_shapes': {key: list(value.shape) for key, value in outputs.items()},
    'conv_macs': macs,
    'conv_gmac': {
        'total': sum(macs.values()) / 1e9,
        'stem': grouped_gmac(('backbone.stem',)),
        'neck': grouped_gmac(('backbone.lat', 'backbone.refine', 'backbone.proj',
                              'backbone.gate', 'backbone.out')),
        'head': grouped_gmac(('header',)),
        'attention_convs': grouped_gmac(('backbone.c4_attention',)),
    },
}
assert rows[0]['backbone_body'] == 647200
assert rows[0]['backbone_including_neck'] == 674256
assert rows[0]['total'] == 693097
assert result['output_shapes']['cls'] == [1, 3, 200, 176]
Path('/tmp/lidar_backbone_research_params.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'parameters': [{key: row[key] for key in ['name', 'backbone_body', 'neck', 'backbone_including_neck', 'head', 'total']} for row in rows],
                  'conv_gmac': result['conv_gmac'], 'output_shapes': result['output_shapes']}, indent=2))

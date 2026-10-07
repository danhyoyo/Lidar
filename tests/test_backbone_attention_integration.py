"""Attention routing, registry options and notebook validation share one contract."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'detector')]
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone
from core.models.backbones.registry import build_backbone
from tools.kitti_training_pipeline.notebook_config import validate_notebook_config
from tools.kitti_training_pipeline.common import generate_run_name

BASE = dict(backbone='mobilepixornext', backbone_out_dim=32, stage_depths=[3,4,2],
            c4_attention='none', c4_attention_scales=[], c4_attention_qk_norm='none')


@pytest.mark.parametrize('local', ['none','eca','simam'])
def test_hooks_route_refined_features_to_both_downstream_and_laterals(local):
    cfg = {**BASE, 'c4_context':'focal', 'local_attention':local}
    m = build_backbone('mobilepixornext', cfg, input_channels=14).train()
    assert m.c4_context.__class__.__name__ == 'FocalContext'
    if local == 'none':
        assert isinstance(m.c3_light_attention, nn.Identity)
    seen = {}
    handles = []
    for name in ['c3_light_attention', 'c4_context']:
        handles.append(getattr(m,name).register_forward_hook(lambda _m,_i,y,name=name: seen.update({name:y})))
    for name in ['down3','lat_c3','down4','lat_c4']:
        handles.append(getattr(m,name).register_forward_pre_hook(lambda _m,args,name=name: seen.update({name:args[0]})))
    try:
        x = torch.randn(2,14,32,48,requires_grad=True)
        y = m(x)
        assert seen['down3'] is seen['lat_c3'] is seen['c3_light_attention']
        assert seen['down4'] is seen['lat_c4'] is seen['c4_context']
        assert y.shape == (2,32,8,12)
        y.square().mean().backward()
        for p in m.c4_context.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
        for p in m.c3_light_attention.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    finally:
        for h in handles: h.remove()


def test_registry_threads_every_feature_option_without_mutating_config():
    cfg={**BASE,'c4_context':'focal','c4_context_bottleneck':32,'c4_context_dilations':[1,3],
         'c4_context_layer_scale_init':0.002,'local_attention':'eca','local_attention_eca_kernel_size':5,
         'local_attention_layer_scale_init':0.003,'neck_fusion_channels':32,'detail_path':True}
    before=copy.deepcopy(cfg)
    m=build_backbone('mobilepixornext',cfg,14)
    assert cfg==before and m.c4_context.bottleneck==32 and m.c4_context.dilations==(1,3)
    assert m.c3_light_attention.core.kernel_size==5
    torch.testing.assert_close(m.c3_light_attention.beta,torch.tensor(0.003))
    assert m.neck_fusion_channels==32 and m.detail_path


@pytest.mark.parametrize('cfg', [
    {'c4_context':'focal'}, {**BASE,'c4_context_version':1},
    {**BASE,'local_attention':'eca','local_attention_simam_lambda':0.001},
    {**BASE,'detail_path':True,'neck_type':'rc_sgfpn'},
    {**BASE,'c4_context':'focal','c4_attention_scales':[5]},
    {**BASE,'c4_context_typo':1},
])
def test_registry_rejects_conflicting_and_ignored_fields(cfg):
    with pytest.raises(ValueError): build_backbone('mobilepixornext',cfg,14)


def test_other_backbone_cannot_ignore_new_context_fields():
    with pytest.raises(ValueError):
        build_backbone('mobilepixor',{'c4_context':'focal'},8)


def test_direct_constructor_and_registry_have_same_resolved_weights():
    torch.manual_seed(721)
    direct=MobilePixorNeXtBackbone(input_channels=14,**{k:v for k,v in BASE.items() if k!='backbone'},c4_context='focal')
    torch.manual_seed(721)
    built=build_backbone('mobilepixornext',{**BASE,'c4_context':'focal'},14)
    assert set(direct.state_dict())==set(built.state_dict())
    for k,v in direct.state_dict().items(): assert torch.equal(v,built.state_dict()[k]),k


def test_notebook_rejects_conflicts_before_sanitizing_and_resolves_new_defaults():
    config=json.loads((ROOT/'configs/config.json').read_text())
    config['model'].update(**BASE,c4_context='focal')
    validate_notebook_config(config)
    assert config['model']['c4_context_bottleneck']==64
    assert config['model']['c4_context_dilations']==[1,2,3]
    conflict=copy.deepcopy(config)
    conflict['model']['c4_attention_scales']=[5]
    with pytest.raises(ValueError): validate_notebook_config(conflict)


def test_candidate_grid_check_occurs_before_hooks():
    m=build_backbone('mobilepixornext',{**BASE,'c4_context':'focal'},14)
    with pytest.raises(ValueError,match='divisible by 16'): m(torch.zeros(1,14,31,48))


def test_new_architectures_get_distinct_reproducible_run_names():
    config = json.loads((ROOT / 'configs/config.json').read_text())
    names = set()
    for extra in [{}, {'c4_context':'focal'}, {'c4_context':'focal','local_attention':'eca'},
                  {'c4_context':'focal','local_attention':'simam'}, {'detail_path':True},
                  {'neck_fusion_channels':32}, {'c4_context':'focal','c4_context_bottleneck':32}]:
        candidate = copy.deepcopy(config)
        candidate['model'].update(BASE, **extra)
        name = generate_run_name(candidate)
        assert name == generate_run_name(copy.deepcopy(candidate))
        names.add(name)
    assert len(names) == 7

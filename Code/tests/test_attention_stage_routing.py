"""Actual stage composition, shared consumers and construction compatibility."""
import pytest
import torch
from torch import nn

from attention_stage_test_utils import CASES, placement_patch
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone
from core.models.backbones.registry import build_backbone

BASE = {'backbone':'mobilepixornext','backbone_out_dim':32,'stage_depths':[1,1,1],
        'neck_fusion_channels':32}


@pytest.mark.parametrize('case', CASES)
def test_refined_stage_outputs_feed_both_downstream_and_lateral(case):
    torch.set_num_threads(2)
    model = build_backbone('mobilepixornext', {**BASE, **placement_patch(case)}, 14).eval()
    seen, handles = {}, []
    for stage in ('c3','c4','c5'):
        handles.append(getattr(model, f'attention_{stage}_output').register_forward_hook(
            lambda m, args, y, stage=stage: seen.update({stage:y})))
    for name in ('down3','down4','lat_c3','lat_c4','lat_c5'):
        handles.append(getattr(model, name).register_forward_pre_hook(
            lambda m, args, name=name: seen.update({name:args[0]})))
    try:
        with torch.inference_mode():
            output = model(torch.randn(1,14,32,48))
        assert output.shape == (1,32,8,12)
        assert seen['c3'] is seen['down3'] is seen['lat_c3']
        assert seen['c4'] is seen['down4'] is seen['lat_c4']
        assert seen['c5'] is seen['lat_c5']
    finally:
        for handle in handles:
            handle.remove()


class Affine(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.scale, self.offset, self.calls = scale, offset, 0
    def forward(self, value):
        self.calls += 1
        return value * self.scale + self.offset


@pytest.mark.parametrize("stage", ["c3", "c4", "c5"])
def test_selected_stage_applies_one_adapter_once(stage):
    model = build_backbone("mobilepixornext", {
        **BASE, "local_attention": "simam", "local_attention_stage": stage}, 14).eval()
    model.c3_light_attention = Affine(3, -4)
    x = torch.randn(1, {"c3": 48, "c4": 96, "c5": 128}[stage], 4, 6)
    other = next(name for name in ("c3", "c4", "c5") if name != stage)
    assert model._apply_stage_attention(x, other) is x
    torch.testing.assert_close(model._apply_stage_attention(x, stage), 3*x - 4)
    assert model.c3_light_attention.calls == 1


@pytest.mark.parametrize('stage', ['c3','c4','c5'])
def test_direct_and_registry_construction_initialize_same_parameters(stage):
    cfg = {**BASE, **placement_patch((None,stage,None))}
    torch.manual_seed(721)
    direct = MobilePixorNeXtBackbone(input_channels=14, **{k:v for k,v in cfg.items() if k != 'backbone'})
    torch.manual_seed(721)
    built = build_backbone('mobilepixornext', cfg, 14)
    assert list(direct.state_dict()) == list(built.state_dict())
    for key, value in direct.state_dict().items():
        assert torch.equal(value, built.state_dict()[key])


def test_constructor_rejects_explicit_null_stage():
    with pytest.raises(ValueError, match="local_attention_stage"):
        MobilePixorNeXtBackbone(local_attention="simam", local_attention_stage=None)


@pytest.mark.parametrize('neck', ['rc_sgfpn','rc_bisgfpn'])
def test_rc_neck_receives_all_refined_features(neck):
    geometry = {'x_min':0.,'x_max':4.8,'x_res':.1,'y_min':-1.6,'y_max':1.6,'y_res':.1}
    model = MobilePixorNeXtBackbone(input_channels=14, local_attention='simam',local_attention_stage='c4',neck_type=neck,geometry=geometry).eval()
    seen, handles = {}, []
    for stage in ('c3','c4','c5'):
        handles.append(getattr(model, f'attention_{stage}_output').register_forward_hook(
            lambda m, a, y, stage=stage: seen.update({stage:y})))
    handles.append(model.rc_neck.register_forward_pre_hook(lambda m,a: seen.update(neck=a)))
    try:
        with torch.inference_mode():
            model(torch.randn(1,14,32,48))
        assert all(seen[s] is value for s,value in zip(('c3','c4','c5'),seen['neck']))
    finally:
        for handle in handles:
            handle.remove()

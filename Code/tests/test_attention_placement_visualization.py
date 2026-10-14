"""Check that diagnostic interventions preserve normal inference and cleanup."""
import numpy as np
import pytest
import torch

from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone
from core.models.backbones.focal_context import FocalContext
from tools.visualization.compare_attention_placements import placement_forward, split_ids
from attention_stage_test_utils import configured


def test_checkpoint_capture_preserves_trained_topology_and_exact_predictions():
    from common import build_model
    from tools.visualization.compare_attention_placements import checkpoint_forward
    cfg,_ = configured(('c3','c3','local_then_focal'))
    model = build_model(cfg).eval()
    x = torch.randn(1,14,32,48)
    with torch.inference_mode():
        expected = model(x)
    predicted,captured = checkpoint_forward(model,x)
    torch.testing.assert_close(predicted,expected,rtol=0,atol=0)
    assert captured['backbone.attention_c3_output'].shape == (1,48,8,12)
    assert model.backbone.attention_order == 'local_then_focal'
    assert not any(m._forward_hooks for m in model.modules())


def test_intervention_rejects_nonlegacy_checkpoint_topology():
    from common import build_model
    cfg,_ = configured(('c5','c5','focal_then_local'))
    model = build_model(cfg).eval()
    with pytest.raises(ValueError,match='checkpoint'):
        placement_forward(model,torch.randn(1,14,32,48),{},model.backbone.c3_light_attention)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobilePixorNeXtBackbone(input_channels=14,
            backbone_out_dim=32, stage_depths=(1, 1, 1), c4_context='focal',
            local_attention='simam', neck_fusion_channels=32)

    def forward(self, tensor):
        return self.backbone(tensor)


def test_native_placement_replays_original_and_preserves_all_state():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    model = TinyModel().eval()
    tensor = torch.randn(1, 14, 32, 32)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        expected = model(tensor)
    original_focal, original_simam = model.backbone.c4_context, model.backbone.c3_light_attention
    actual, captured = placement_forward(model, tensor, {'C4': original_focal}, original_simam,
                                         focal='C4', sim='C3')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert captured['C3'].shape == (1, 48, 8, 8)
    assert captured['C4'].shape == (1, 96, 4, 4)
    assert captured['C5'].shape == (1, 128, 2, 2)
    assert model.backbone.c4_context is original_focal
    assert model.backbone.c3_light_attention is original_simam
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize('order', ['FS', 'SF'])
def test_shared_stage_applies_requested_order_to_actual_features(order):
    torch.manual_seed(8)
    model = TinyModel().eval()
    focal = FocalContext(channels=48, layer_scale_init=.2).eval()
    simam = model.backbone.c3_light_attention
    with torch.no_grad():
        simam.beta.fill_(.7)
    _, captured = placement_forward(model, torch.randn(1, 14, 32, 32), {'C3': focal},
                                     simam, focal='C3', sim='C3', order=order)
    before = torch.from_numpy(captured['C3_before'])
    with torch.inference_mode():
        expected = simam(focal(before)) if order == 'FS' else focal(simam(before))
    np.testing.assert_allclose(captured['C3'], expected.numpy(), rtol=1e-6, atol=1e-7)


def test_failed_intervention_restores_hooks_and_modules():
    model = TinyModel().eval()
    focal, simam = model.backbone.c4_context, model.backbone.c3_light_attention
    with pytest.raises(KeyError):
        placement_forward(model, torch.randn(1, 14, 32, 32), {}, simam, focal='C3')
    assert model.backbone.c4_context is focal
    assert model.backbone.c3_light_attention is simam
    assert all(not module._forward_hooks for module in model.modules())


def test_split_reads_dataset_suffix_and_plain_frame_ids(tmp_path):
    path = tmp_path / 'val.txt'
    path.write_text('000001;kitti\n000002\n\n')
    assert split_ids(path) == {'000001', '000002'}

"""Real losses/optimizers plus full-grid shape audits for all placement cases."""
import pytest
import torch

from attention_stage_test_utils import CASES, configured
from common import build_model
import train


@pytest.mark.parametrize('case', CASES)
def test_all_placements_backpropagate_actual_grouped_loss_once_per_parameter(case):
    torch.set_num_threads(2)
    torch.manual_seed(83)
    config, target = configured(case)
    target['voxel'] = target['voxel'].repeat(2,1,1,1)
    target['groups'] = {name:{key:value.repeat(2,*([1]*(value.ndim-1)))
                              for key,value in group.items()} for name,group in target['groups'].items()}
    model = build_model(config).train()
    criterion = train.build_training_criterion(config, torch.device('cpu')).train()
    optimizer = train.build_optimizer(model, criterion, config)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for module in (model,criterion) for p in module.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids)) and set(ids) == expected
    loss = criterion(model(target['voxel']), target)['loss']
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.backbone.c3_light_attention,):
        for parameter in module.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.backbone.stem[0].weight.grad is not None
    optimizer.step()


@pytest.mark.parametrize('case', CASES)
def test_full_kitti_shapes_and_parameter_counts_on_meta(case):
    config, _ = configured(case)
    with torch.device('meta'):
        model = build_model(config).eval()
        prediction = model(torch.zeros(1,14,800,704))
    for group in prediction['groups'].values():
        assert all(value.shape[-2:] == (200,176) for value in group.values())
    assert sum(p.numel() for p in model.backbone.c3_light_attention.parameters()) == int(case[1] is not None)
    assert not any('attention_c' in key for key in model.state_dict())


@pytest.mark.parametrize('stage', ['c4','c5'])
def test_simam_can_move_without_changing_core_parameter_count(stage):
    config,_ = configured((None,stage,None), local_kind='simam')
    model = build_model(config).eval()
    assert sum(p.numel() for p in model.backbone.c3_light_attention.parameters()) == 1
    with torch.inference_mode():
        assert model(torch.randn(1,14,32,48))['groups']['car']['cls'].shape[-2:] == (8,12)


def test_shared_3d_header_contract_is_preserved_with_colocated_attention():
    config,_ = configured((None,'c5',None))
    config['data']['box_mode'] = '3d'
    config['model']['header_architecture'] = 'shared'
    config['loss']['vertical_loss_weight'] = 1.
    model = build_model(config).train()
    prediction = model(torch.randn(2,14,32,48))
    assert all(group['vertical'].shape == (2,2,8,12) for group in prediction['groups'].values())
    loss = sum(value.square().mean() for group in prediction['groups'].values() for value in group.values())
    loss.backward()
    assert model.backbone.c3_light_attention.beta.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable; CPU checks do not establish AMP parity')
@pytest.mark.parametrize('dtype', [torch.float16,torch.bfloat16])
@pytest.mark.parametrize('case', CASES)
def test_cuda_amp_forward_backward_is_finite(case, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA BF16 unsupported')
    config,_ = configured(case)
    model = build_model(config).cuda().train()
    with torch.autocast('cuda', dtype=dtype):
        prediction = model(torch.randn(2,14,32,48,device='cuda'))
        loss = sum(value.float().square().mean() for group in prediction['groups'].values() for value in group.values())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

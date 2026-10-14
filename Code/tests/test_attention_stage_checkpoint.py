"""Relocation/reordering with equal tensor shapes must fail semantic preflight."""
import copy
import pytest

from attention_stage_test_utils import CASES, configured
from common import checkpoint_identity, generate_run_name, validate_evaluation_checkpoint
from test_grouped_checkpoint import components, payload, restore
from test_grouped_loss_state import snapshot, assert_state_equal
from tools.benchmarks.profile_detector import profile_detector


def test_all_19_topologies_have_unique_reproducible_identities_and_readable_names():
    identities, names = set(), set()
    import json
    for case in CASES:
        config,_ = configured(case)
        identities.add(json.dumps(checkpoint_identity(config), sort_keys=True))
        name = generate_run_name(config)
        assert name == generate_run_name(copy.deepcopy(config))
        if not (case[0] in (None,'c4') and case[1] in (None,'c3')):
            if case[0]:
                assert 'at'+case[0] in name
            if case[2]:
                assert ('orderfs' if case[2]=='focal_then_local' else 'ordersf') in name
        names.add(name)
    assert len(identities) == len(names) == 19


@pytest.mark.parametrize('before,after', [
    (('c4','c3',None),('c4','c5',None)),
    (('c4','c4','focal_then_local'),('c4','c4','local_then_focal')),
])
def test_equal_shape_relocation_or_order_cannot_restore_or_evaluate(before, after):
    config,_ = configured(before)
    saved = payload(config, components(config))
    changed,_ = configured(after)
    destination = components(changed)
    state, criterion = snapshot(destination[0]), snapshot(destination[1])
    with pytest.raises(ValueError,match='identity'):
        restore(saved, changed, destination)
    with pytest.raises(ValueError,match='identity'):
        validate_evaluation_checkpoint(saved, changed)
    assert_state_equal(destination[0], state)
    assert_state_equal(destination[1], criterion)
    assert not destination[2].state


@pytest.mark.parametrize('case', [('c4','c3',None),('c3','c5',None),('c4','c4','local_then_focal')])
def test_new_placement_checkpoint_roundtrips_full_resume(case, tmp_path):
    import torch
    config,target = configured(case)
    source = components(config)
    import train
    train.set_loss_epoch(source[1],0)
    loss = source[1](source[0](target['voxel']),target)['loss']
    loss.backward()
    source[2].step()
    source[3].step()
    saved = payload(config, source)
    path = tmp_path/'state.pt'
    torch.save(saved,path)
    saved = torch.load(path,weights_only=True)
    validate_evaluation_checkpoint(saved,config)
    destination = components(config)
    assert restore(saved,config,destination)['mode'] == 'resume'
    assert_state_equal(destination[0],snapshot(source[0]))
    assert_state_equal(destination[1],snapshot(source[1]))
    torch.testing.assert_close(destination[2].state_dict(),source[2].state_dict(),rtol=0,atol=0)
    assert destination[3].state_dict() == source[3].state_dict()


def test_profile_reports_effective_placement_even_for_legacy_metadata():
    config,_ = configured(('c4','c3',None))
    report = profile_detector(config,device='meta')
    assert report['effective_attention_placement']['focal']['stage'] == 'c4'
    assert report['effective_attention_placement']['local_attention']['stage'] == 'c3'

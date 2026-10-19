"""Frozen pre-placement identities and initialization, not regenerated expectations."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from core.backbone_config import resolve_backbone_features
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone

GOLDEN = json.loads((Path(__file__).parent / 'fixtures/attention_placement_legacy_v1.json').read_text())


def test_schema_version_changes_identity_explicitly():
    # Historical records remain frozen evidence; retired topologies are not loadable.
    for record in GOLDEN["feature_cases"]:
        if record["input"].get("c4_context") == "focal":
            with pytest.raises(ValueError, match="removed"):
                resolve_backbone_features(record["input"])
        else:
            features = resolve_backbone_features(record["input"])
            assert features.semantic_metadata()["resolver_version"] == 4
            assert features.semantic_hash != record["sha256"]


@pytest.mark.parametrize('record', [case for case in GOLDEN['feature_cases']
                                    if case['input'].get('c4_context', 'none') == 'none'])
def test_legacy_initialization_state_and_outputs(record):
    if torch.__version__ != GOLDEN['torch_version']:
        pytest.skip('Initialization golden uses the recorded PyTorch version')
    torch.set_num_threads(2)
    torch.manual_seed(721)
    model = MobilePixorNeXtBackbone(input_channels=14, backbone_out_dim=32,
        stage_depths=(3, 4, 2), **{k:v for k,v in record['input'].items() if k != 'backbone'}).eval()
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    assert list(model.state_dict()) == record['state_keys']
    assert digest.hexdigest() == record['state_sha256']
    with torch.inference_mode():
        output = model(torch.linspace(-1, 1, 14*32*48).reshape(1, 14, 32, 48))
    torch.testing.assert_close(output.flatten()[::53], torch.tensor(record['output_samples']),
                               rtol=1e-5, atol=1e-6)

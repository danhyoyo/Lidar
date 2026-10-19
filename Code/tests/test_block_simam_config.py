"""Block SimAM placement is canonical, validated and checkpoint-visible."""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector.core.backbone_config import resolve_backbone_features


def test_block_stages_canonicalize_without_mutating_input():
    config = {"block_simam_stages": ["C5", "c4"]}
    before = copy.deepcopy(config)
    result = resolve_backbone_features(config)
    assert config == before
    assert result.block_simam_stages == ("c4", "c5")
    assert result.block_simam_lambda == .0001
    assert result.to_dict()["block_simam_stages"] == ["c4", "c5"]
    assert result.semantic_metadata()["resolver_version"] == 4
    assert [entry["stride"] for entry in result.effective_placement()["block_simam"]] == [8, 16]
    assert result.semantic_hash == resolve_backbone_features(
        {"block_simam_stages": ["c4", "c5"], "block_simam_lambda": .0001}).semantic_hash


def test_disabled_block_attention_preserves_existing_serialization_and_identity():
    config = {"local_attention": "simam"}
    implicit = resolve_backbone_features(config)
    explicit = resolve_backbone_features({**config, "block_simam_stages": []})
    assert implicit.to_dict() == explicit.to_dict()
    assert not any(key.startswith("block_simam") for key in explicit.to_dict())
    assert implicit.semantic_hash == explicit.semantic_hash
    assert implicit.semantic_metadata()["resolver_version"] == 4


@pytest.mark.parametrize("patch", [
    {"block_simam_stages": None}, {"block_simam_stages": True},
    {"block_simam_stages": "c4"}, {"block_simam_stages": ["stem"]},
    {"block_simam_stages": ["c4", "C4"]}, {"block_simam_stages": [3]},
    {"block_simam_stages": [["c4"]]},
    {"block_simam_stages": ["c4"], "block_simam_lambda": 0},
    {"block_simam_stages": ["c4"], "block_simam_lambda": float("nan")},
    {"block_simam_stages": ["c4"], "block_simam_lambda": True},
    {"block_simam_stages": ["c4"], "block_simam_lambda": None},
    {"block_simam_stages": [], "block_simam_lambda": .001},
    {"block_simam_lambda": .001}, {"block_simam_stage": "c4"},
    {"backbone": "mobilepixor", "block_simam_stages": ["c4"]},
])
def test_invalid_or_inactive_block_options_are_rejected(patch):
    with pytest.raises(ValueError, match="block_simam|MobilePixorNeXt"):
        resolve_backbone_features(patch)


def test_parameter_free_topologies_and_lambda_have_different_identities():
    variants = [{}, {"block_simam_stages": ["c4"]}, {"block_simam_stages": ["c5"]},
                {"block_simam_stages": ["c4", "c5"]},
                {"block_simam_stages": ["c4", "c5"], "block_simam_lambda": .002}]
    assert len({resolve_backbone_features(config).semantic_hash for config in variants}) == len(variants)

"""Retired experiments must fail at configuration boundaries."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.core.backbone_config import resolve_backbone_features
from detector.core.bev_encoding import resolve_input_channels
from detector.core.datasets.utils_1.preprocess import encode_bev
from detector.core.detection_config import resolve_detection_config


@pytest.mark.parametrize("name", ["rich10", "rich11", "rich12"])
def test_removed_encodings_rejected_by_model_and_rasterizer(name):
    with pytest.raises(ValueError, match="unsupported BEV encoding"):
        resolve_input_channels({"name": name})
    with pytest.raises(ValueError, match="unsupported BEV encoding"):
        encode_bev(np.empty((0, 4)), {}, {"name": name})


def test_removed_loss_rejected_before_model_construction():
    with pytest.raises(ValueError, match="name must be one of"):
        resolve_detection_config({"loss": {"name": "gw_qal"}})


@pytest.mark.parametrize("config", [
    {"c4_attention": "litemla"},
    {"c4_attention_scales": [5]},
    {"c4_attention_qk_norm": "rmsnorm"},
])
def test_removed_c4_attention_settings_rejected(config):
    with pytest.raises(ValueError, match="c4_attention"):
        resolve_backbone_features(config)


def test_default_backbone_uses_convolution_and_allows_focal_context():
    assert resolve_backbone_features().c4_attention == "none"
    assert resolve_backbone_features({"c4_context": "focal"}).c4_context == "focal"

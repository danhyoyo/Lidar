import math
import torch
import pytest
from core.models.backbones.rc_sgfpn import FourierRangeEmbedding


def test_fourier_range_embedding_shapes_and_values():
    h, w = 100, 88
    x_bounds = (0.0, 70.4)
    y_bounds = (-40.0, 40.0)
    num_bands = 4

    fre = FourierRangeEmbedding(h, w, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_bands)

    assert hasattr(fre, "embedding")
    assert fre.embedding.shape == (1, 2 * num_bands, h, w)
    assert fre.out_dim == 2 * num_bands

    # Values must be bounded in [-1.0, 1.0]
    assert torch.all(fre.embedding >= -1.0 - 1e-5)
    assert torch.all(fre.embedding <= 1.0 + 1e-5)
    assert torch.isfinite(fre.embedding).all()

    # Exact origin check with odd dimensions: at (x=0, y=0), r=0 => sin(0)=0, cos(0)=1
    fre_odd = FourierRangeEmbedding(101, 89, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_bands)
    origin_y = 50
    origin_x = 0
    origin_feats = fre_odd.embedding[0, :, origin_y, origin_x]
    for b in range(num_bands):
        sin_val = origin_feats[2 * b].item()
        cos_val = origin_feats[2 * b + 1].item()
        assert abs(sin_val) < 1e-5
        assert abs(cos_val - 1.0) < 1e-5

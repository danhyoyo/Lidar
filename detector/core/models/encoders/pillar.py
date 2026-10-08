"""Small learned pillar encoder with uncapped, unpadded point pooling."""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from core.bev_encoding import resolve_bev_encoding


class PillarEncoder(nn.Module):
    """Shared point MLP (10 -> 32), max pooling and dense BEV scatter."""

    def __init__(self, geometry):
        super().__init__()
        schema = resolve_bev_encoding({"name": "pillar32"}, geometry)
        self.width, self.height = schema.grid_shape[:2]
        self.channels = schema.channels
        self.linear = nn.Linear(10, self.channels, bias=False)
        self.norm = nn.BatchNorm1d(self.channels, eps=1e-3, momentum=.01)

    def forward(self, packed):
        if not isinstance(packed, Mapping):
            raise ValueError("pillar32 requires packed point features, not an encoded BEV tensor")
        features = packed["features"]
        indices = packed["pillar_indices"]
        coords = packed["coords"]
        batch_size = packed["batch_size"]
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("pillar batch_size must be a positive integer")
        if features.ndim != 2 or features.shape[1] != 10:
            raise ValueError("pillar features must have shape [N, 10]")
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.dtype != torch.int64:
            raise ValueError("pillar coords must be int64 [K, 3] in batch,y,x order")
        if indices.shape != (len(features),) or indices.dtype != torch.int64:
            raise ValueError("pillar_indices must be int64 [N]")
        if not len(features):
            if len(coords):
                raise ValueError("empty point input must have empty pillar coordinates")
            # Keep all encoder parameters connected for empty-frame backward.
            zero = sum(parameter.sum() * 0 for parameter in self.parameters())
            return features.new_zeros((batch_size, self.channels, self.height, self.width)) + zero
        if not len(coords):
            raise ValueError("nonempty points require occupied pillar coordinates")
        if (torch.any(indices < 0) or torch.any(indices >= len(coords)) or
                torch.any(coords < 0) or torch.any(coords[:, 0] >= batch_size) or
                torch.any(coords[:, 1] >= self.height) or torch.any(coords[:, 2] >= self.width)):
            raise ValueError("pillar indices or coordinates are outside the batch/grid")
        embedded = self.linear(features)
        if self.training and len(features) == 1:
            embedded = F.batch_norm(embedded, self.norm.running_mean, self.norm.running_var,
                                    self.norm.weight, self.norm.bias, training=False, eps=self.norm.eps)
        else:
            embedded = self.norm(embedded)
        embedded = F.relu(embedded)
        pooled = embedded.new_zeros((len(coords), self.channels))
        pooled.scatter_reduce_(0, indices[:, None].expand_as(embedded), embedded,
                               reduce="amax", include_self=True)
        locations = (coords[:, 0] * self.height + coords[:, 1]) * self.width + coords[:, 2]
        dense = pooled.new_zeros((batch_size * self.height * self.width, self.channels))
        dense.index_copy_(0, locations, pooled)
        return dense.view(batch_size, self.height, self.width, self.channels).permute(0, 3, 1, 2).contiguous()

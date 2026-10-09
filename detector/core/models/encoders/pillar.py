"""Small learned pillar encoder with uncapped, unpadded point pooling."""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from core.bev_encoding import resolve_bev_encoding


class PillarEncoder(nn.Module):
    """Point MLP, max pooling, optional rich8 fusion and one dense BEV scatter."""

    def __init__(self, geometry, encoding=None):
        super().__init__()
        schema = resolve_bev_encoding(encoding or {"name": "pillar32"}, geometry)
        if not schema.is_packed:
            raise ValueError("PillarEncoder requires a packed pillar encoding")
        self.name = schema.name
        self.width, self.height = schema.grid_shape[:2]
        self.channels = schema.channels
        self.learned_channels = 24 if self.name == "pillar_rich" else self.channels
        self.linear = nn.Linear(10, self.learned_channels, bias=False)
        self.norm = nn.BatchNorm1d(self.learned_channels, eps=1e-3, momentum=.01)
        self.pooling = schema.pooling
        if self.pooling == "max_mean_eca":
            self.eca = nn.Conv1d(1, 1, schema.eca_kernel_size,
                                 padding=schema.eca_kernel_size // 2, bias=False)
            nn.init.zeros_(self.eca.weight)

    def forward(self, packed):
        if not isinstance(packed, Mapping):
            raise ValueError(f"{self.name} requires packed point features, not an encoded BEV tensor")
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
        rich = packed.get("rich_features")
        if self.name == "pillar_rich" and (
                not torch.is_tensor(rich) or rich.shape != (len(coords), 8) or
                not rich.is_floating_point() or rich.device != features.device):
            raise ValueError("pillar_rich rich_features must be floating [K, 8] on the point device")
        if not len(features):
            if len(coords):
                raise ValueError("empty point input must have empty pillar coordinates")
            # Keep all encoder parameters connected for empty-frame backward.
            zero = sum(parameter.sum() * 0 for parameter in self.parameters())
            return features.new_zeros((batch_size, self.channels, self.height, self.width)) + zero
        if not len(coords):
            raise ValueError("nonempty points require occupied pillar coordinates")
        # A single device predicate avoids five host synchronizations on CUDA.
        invalid_indices = ((indices < 0) | (indices >= len(coords))).any()
        invalid_coords = ((coords < 0).any() | (coords[:, 0] >= batch_size).any() |
                          (coords[:, 1] >= self.height).any() | (coords[:, 2] >= self.width).any())
        if invalid_indices | invalid_coords:
            raise ValueError("pillar indices or coordinates are outside the batch/grid")
        embedded = self.linear(features)
        if self.training and len(features) == 1:
            embedded = F.batch_norm(embedded, self.norm.running_mean, self.norm.running_var,
                                    self.norm.weight, self.norm.bias, training=False, eps=self.norm.eps)
        else:
            embedded = self.norm(embedded)
        embedded = F.relu(embedded)
        pooled = embedded.new_zeros((len(coords), self.learned_channels))
        pooled.scatter_reduce_(0, indices[:, None].expand_as(embedded), embedded,
                               reduce="amax", include_self=True)
        if self.pooling == "max_mean_eca":
            # Keep summation and the tiny gate FP32 under BF16/FP16 autocast.
            # Process occupied pillars only; do not materialize a mean BEV.
            with torch.autocast(device_type=embedded.device.type, enabled=False):
                means = embedded.new_zeros((len(coords), self.learned_channels), dtype=torch.float32)
                means.index_add_(0, indices, embedded.float())
                counts = torch.bincount(indices, minlength=len(coords)).clamp_min(1)
                means = means / counts[:, None]
                # The exact short channel convolution as shifted addcmul:
                # avoids a Conv1d workspace for K tiny, one-channel sequences.
                kernel = self.eca.kernel_size[0]
                padded = F.pad(means, (kernel // 2, kernel // 2)) if kernel > 1 else means
                weights = self.eca.weight.float().flatten()
                logits = padded[:, :self.learned_channels] * weights[0]
                for offset in range(1, kernel):
                    logits = torch.addcmul(logits, padded[:, offset:offset + self.learned_channels],
                                          weights[offset])
                alpha = logits.sigmoid()
                pooled = torch.lerp(means, pooled.float(), alpha).to(embedded.dtype)
        if self.name == "pillar_rich":
            pooled = torch.cat((rich.to(pooled.dtype), pooled), dim=1)
        # Scatter into the final layout, avoiding a second full BEV allocation
        # and the NHWC -> NCHW contiguous copy (68.75 MiB/frame at KITTI FP32).
        dense = pooled.new_zeros((batch_size, self.channels, self.height * self.width))
        dense[coords[:, 0], :, coords[:, 1] * self.width + coords[:, 2]] = pooled
        return dense.view(batch_size, self.channels, self.height, self.width)

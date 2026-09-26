"""Optional C4 refinements for dense LiDAR BEV features.

LSK follows the spatial-selection mechanism of Li et al., ICCV 2023.
LiteMLA follows the multi-scale ReLU linear attention of Cai et al., ICCV 2023.
DAT follows the grouped learned-offset sampling of Xia et al., CVPR 2022.
BRA follows the coarse region routing plus token attention of Zhu et al., CVPR 2023.
These are residual adapters, not complete upstream blocks or backbones.
See docs/backbone_c4_ablation.md and third_party/attention/NOTICE.md for
equations, adaptation details, citations, and upstream license notices.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _layer_scale(channels, value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("layer_scale_init must be finite and nonnegative")
    return nn.Parameter(torch.full((1, channels, 1, 1), float(value)))


class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW feature maps."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class LSKRefinement(nn.Module):
    """Residual large selective kernel attention with a small LayerScale.

    Sequential depthwise 5x5 and dilated 7x7 (dilation=3) branches have
    effective receptive fields 5x5 and 23x23. Two spatial sigmoid gates select
    their projected features. The selected feature modulates the input to
    the spatial gate, followed by an output projection and residual addition.
    We omit LSKNet's separate MLP and stochastic depth for this ablation.
    """

    def __init__(self, channels=64, layer_scale_init=0.01):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("LSK channels must be positive and even")
        self.norm = nn.BatchNorm2d(channels)
        self.input_projection = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.context = nn.Conv2d(
            channels, channels, 7, padding=9, dilation=3, groups=channels
        )
        self.local_projection = nn.Conv2d(channels, channels // 2, 1)
        self.context_projection = nn.Conv2d(channels, channels // 2, 1)
        self.selector = nn.Conv2d(2, 2, 7, padding=3)
        self.gate_projection = nn.Conv2d(channels // 2, channels, 1)
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    def forward(self, x):
        features = self.activation(self.input_projection(self.norm(x)))
        local = self.local(features)
        context = self.context(local)
        local = self.local_projection(local)
        context = self.context_projection(context)
        joined = torch.cat((local, context), dim=1)
        descriptors = torch.cat(
            (joined.mean(dim=1, keepdim=True), joined.amax(dim=1, keepdim=True)),
            dim=1,
        )
        gates = self.selector(descriptors).sigmoid()
        selected = gates[:, :1] * local + gates[:, 1:] * context
        refined = self.output_projection(features * self.gate_projection(selected))
        return x + self.layer_scale.to(refined.dtype) * refined


class LiteMLARefinement(nn.Module):
    """Residual multi-scale linear attention; no spatial NxN score matrix.

    Native and depthwise/grouped-convolution QKV scales become extra heads.
    ReLU(Q), ReLU(K) use normalized kernel attention, not softmax attention.
    Accumulation and normalization use FP32 under FP16/BF16 autocast. The
    existing MobilePIXOR blocks supply local processing; no extra EfficientViT
    MBConv/FFN is added. A per-channel LayerScale initializes near identity.
    """

    def __init__(
        self, channels=64, head_dim=16, scales=(5,), eps=1e-6,
        layer_scale_init=0.01,
    ):
        super().__init__()
        if type(head_dim) is not int or head_dim < 1 or channels % head_dim:
            raise ValueError("head_dim must be a positive integer dividing channels")
        if not isinstance(scales, (list, tuple)) or any(
            type(scale) is not int or scale < 3 or scale % 2 == 0
            for scale in scales
        ) or len(set(scales)) != len(scales):
            raise ValueError("scales must contain distinct odd integers >= 3")
        if isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.head_dim = head_dim
        self.eps = float(eps)
        heads = channels // head_dim
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.aggregations = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    3 * channels, 3 * channels, scale, padding=scale // 2,
                    groups=3 * channels, bias=False,
                ),
                nn.Conv2d(
                    3 * channels, 3 * channels, 1, groups=3 * heads, bias=False,
                ),
            )
            for scale in scales
        ])
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels * (1 + len(scales)), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    def linear_attention(self, qkv):
        """Evaluate V K^T Q / (sum(K)^T Q + eps) in channel-first layout."""
        batch, _, height, width = qkv.shape
        original_dtype = qkv.dtype
        with torch.autocast(device_type=qkv.device.type, enabled=False):
            if original_dtype in (torch.float16, torch.bfloat16):
                qkv = qkv.float()
            packed = qkv.reshape(batch, -1, 3 * self.head_dim, height * width)
            query, key, value = packed.split(self.head_dim, dim=2)
            query, key = query.relu(), key.relu()
            # Only dxd and dxN intermediates; never NxN (N=H*W).
            numerator = (value @ key.transpose(-1, -2)) @ query
            denominator = key.sum(dim=-1, keepdim=True).transpose(-1, -2) @ query
            attended = numerator / (denominator + self.eps)
            attended = attended.reshape(batch, -1, height, width)
        return attended.to(original_dtype)

    def forward(self, x):
        qkv = self.qkv(x)
        packed = torch.cat([qkv] + [op(qkv) for op in self.aggregations], dim=1)
        refined = self.output_projection(self.linear_attention(packed))
        return x + self.layer_scale.to(refined.dtype) * refined


class DeformableAttentionRefinement(nn.Module):
    """Residual DAT-style grouped deformable self-attention.

    One offset grid is learned per channel group from the query features. Keys
    and values are bilinearly sampled at those displaced reference points, so
    every dense C4 query attends to a reduced, data-dependent set of locations.
    A stride of four keeps the C4 attention matrix practical for 100x88 maps.
    """

    def __init__(
        self, channels=64, num_heads=4, num_groups=4, stride=4,
        offset_kernel_size=5, offset_range_factor=2.0,
        layer_scale_init=0.01,
    ):
        super().__init__()
        integer_options = {
            "num_heads": num_heads,
            "num_groups": num_groups,
            "stride": stride,
            "offset_kernel_size": offset_kernel_size,
        }
        if any(type(value) is not int or value < 1 for value in integer_options.values()):
            raise ValueError("DAT integer options must be positive integers")
        if channels % num_heads or channels % num_groups or num_heads % num_groups:
            raise ValueError("DAT channels/heads/groups must divide evenly")
        if offset_kernel_size < 3 or offset_kernel_size % 2 == 0:
            raise ValueError("DAT offset_kernel_size must be an odd integer >= 3")
        if (
            isinstance(offset_range_factor, bool)
            or not math.isfinite(offset_range_factor)
            or offset_range_factor < 0
        ):
            raise ValueError("DAT offset_range_factor must be finite and nonnegative")

        self.channels = channels
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = channels // num_heads
        self.group_channels = channels // num_groups
        self.stride = stride
        self.offset_range_factor = float(offset_range_factor)
        self.scale = self.head_dim ** -0.5

        self.norm = _LayerNorm2d(channels)
        self.query_projection = nn.Conv2d(channels, channels, 1)
        self.offset = nn.Sequential(
            nn.Conv2d(
                self.group_channels,
                self.group_channels,
                offset_kernel_size,
                stride=stride,
                padding=offset_kernel_size // 2,
                groups=self.group_channels,
                bias=False,
            ),
            _LayerNorm2d(self.group_channels),
            nn.GELU(),
            nn.Conv2d(self.group_channels, 2, 1, bias=False),
        )
        self.key_projection = nn.Conv2d(channels, channels, 1)
        self.value_projection = nn.Conv2d(channels, channels, 1)
        self.local_position = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    @staticmethod
    def _reference_grid(height, width, *, dtype, device):
        y, x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device),
            indexing="ij",
        )
        y = y / max(height - 1.0, 1.0) * 2.0 - 1.0
        x = x / max(width - 1.0, 1.0) * 2.0 - 1.0
        return torch.stack((y, x), dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        features = self.norm(x)
        query = self.query_projection(features)
        grouped_query = query.reshape(
            batch * self.num_groups, self.group_channels, height, width
        )
        offset = self.offset(grouped_query)
        sample_height, sample_width = offset.shape[-2:]
        offset_range = offset.new_tensor((
            1.0 / max(sample_height - 1.0, 1.0),
            1.0 / max(sample_width - 1.0, 1.0),
        )).reshape(1, 2, 1, 1)
        offset = (
            offset.tanh() * offset_range * self.offset_range_factor
        ).permute(0, 2, 3, 1)
        reference = self._reference_grid(
            sample_height, sample_width, dtype=offset.dtype, device=offset.device
        ).unsqueeze(0)
        sampling_grid = offset + reference

        grouped_features = features.reshape(
            batch * self.num_groups, self.group_channels, height, width
        )
        # CPU grid_sample does not implement every low-precision dtype. FP32
        # sampling also keeps offset gradients stable under BF16/FP16 autocast.
        sampled = F.grid_sample(
            grouped_features.float(),
            sampling_grid[..., (1, 0)].float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).to(features.dtype)
        sampled = sampled.reshape(batch, channels, sample_height, sample_width)

        query = query.reshape(
            batch, self.num_heads, self.head_dim, height * width
        )
        key = self.key_projection(sampled).reshape(
            batch, self.num_heads, self.head_dim, sample_height * sample_width
        )
        value = self.value_projection(sampled).reshape_as(key)
        scores = torch.einsum(
            "bhdm,bhdn->bhmn", query.float(), key.float()
        ) * self.scale
        weights = scores.softmax(dim=-1)
        attended = torch.einsum(
            "bhmn,bhdn->bhdm", weights, value.float()
        ).to(query.dtype)
        attended = attended.reshape(batch, channels, height, width)
        refined = self.output_projection(attended + self.local_position(query.reshape(
            batch, channels, height, width
        )))
        return x + self.layer_scale.to(refined.dtype) * refined


class BiLevelRoutingAttentionRefinement(nn.Module):
    """Residual BiFormer-style bi-level routing attention for NCHW C4 maps.

    Region descriptors first select top-k relevant regions. Fine token attention
    then operates only over the selected regions. Routing indices are detached,
    matching the official readable PyTorch implementation; gradients still flow
    through the token-level query, key, value, and output projections.
    """

    def __init__(
        self, channels=64, num_heads=4, n_win=8, topk=4,
        side_dwconv=3, layer_scale_init=0.01,
    ):
        super().__init__()
        options = {
            "num_heads": num_heads,
            "n_win": n_win,
            "topk": topk,
            "side_dwconv": side_dwconv,
        }
        if any(type(value) is not int or value < 0 for value in options.values()):
            raise ValueError("BRA integer options must be nonnegative integers")
        if num_heads < 1 or n_win < 2 or topk < 1:
            raise ValueError("BRA requires num_heads>=1, n_win>=2, and topk>=1")
        if channels % num_heads:
            raise ValueError("BRA channels must be divisible by num_heads")
        if topk > n_win * n_win:
            raise ValueError("BRA topk cannot exceed the number of regions")
        if side_dwconv and (side_dwconv < 3 or side_dwconv % 2 == 0):
            raise ValueError("BRA side_dwconv must be zero or an odd integer >= 3")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.n_win = n_win
        self.topk = topk
        # The public NCHW BiFormer implementation uses dim**-0.5 here (rather
        # than the more usual per-head dimension scale). Keep that choice so
        # this adapter does not silently change the referenced mechanism.
        self.token_scale = channels ** -0.5
        self.routing_scale = channels ** -0.5

        self.norm = _LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.local_position = (
            nn.Conv2d(
                channels, channels, side_dwconv,
                padding=side_dwconv // 2, groups=channels,
            )
            if side_dwconv
            else None
        )
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    def _token_windows(self, x, window_height, window_width):
        batch = x.shape[0]
        return (
            x.reshape(
                batch, self.num_heads, self.head_dim,
                self.n_win, window_height, self.n_win, window_width,
            )
            .permute(0, 1, 3, 5, 4, 6, 2)
            .reshape(
                batch, self.num_heads, self.n_win * self.n_win,
                window_height * window_width, self.head_dim,
            )
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        features = self.norm(x)
        query, key, value = self.qkv(features).chunk(3, dim=1)
        window_height = math.ceil(height / self.n_win)
        window_width = math.ceil(width / self.n_win)
        padded_height = window_height * self.n_win
        padded_width = window_width * self.n_win
        padding = (0, padded_width - width, 0, padded_height - height)
        query_padded = F.pad(query, padding)
        key_padded = F.pad(key, padding)
        value_padded = F.pad(value, padding)

        # Coarse routing uses one C-dimensional descriptor per region and one
        # shared route graph for all heads, as in the official NCHW BRA module.
        validity = F.pad(
            query.new_ones((1, 1, height, width)), padding
        ).reshape(
            1, 1, self.n_win, window_height, self.n_win, window_width
        ).sum(dim=(3, 5)).clamp_min_(1)

        def region_descriptors(tensor):
            # Divide by the number of real (not padded) tokens. This matches
            # count_include_pad=False while retaining exactly n_win**2 regions,
            # including on the tiny feature maps used by smoke/export tests.
            region_sum = tensor.detach().reshape(
                batch, channels, self.n_win, window_height,
                self.n_win, window_width,
            ).sum(dim=(3, 5))
            pooled = region_sum / validity
            return pooled.permute(0, 2, 3, 1).reshape(
                batch, self.n_win * self.n_win, channels
            )

        region_query = region_descriptors(query_padded)
        region_key = region_descriptors(key_padded)
        routing_scores = torch.matmul(
            region_query.float(), region_key.float().transpose(-1, -2)
        ) * self.routing_scale
        route_indices = routing_scores.topk(self.topk, dim=-1).indices

        query_windows = self._token_windows(
            query_padded, window_height, window_width
        )
        key_windows = self._token_windows(key_padded, window_height, window_width)
        value_windows = self._token_windows(
            value_padded, window_height, window_width
        )
        regions = self.n_win * self.n_win
        tokens = window_height * window_width
        expanded_routes = route_indices[:, None].expand(
            batch, self.num_heads, regions, self.topk
        )
        bases = (
            torch.arange(batch * self.num_heads, device=x.device)
            .reshape(batch, self.num_heads, 1, 1) * regions
        )
        flat_routes = (expanded_routes + bases).reshape(-1)

        def gather_routed(windows):
            selected = windows.reshape(
                batch * self.num_heads * regions, tokens, self.head_dim
            ).index_select(0, flat_routes)
            return selected.reshape(
                batch, self.num_heads, regions,
                self.topk * tokens, self.head_dim,
            )

        routed_key = gather_routed(key_windows)
        routed_value = gather_routed(value_windows)
        scores = torch.einsum(
            "bhrtd,bhrsd->bhrts", query_windows.float(), routed_key.float()
        ) * self.token_scale
        weights = scores.softmax(dim=-1)
        attended = torch.einsum(
            "bhrts,bhrsd->bhrtd", weights, routed_value.float()
        ).to(query_windows.dtype)
        attended = (
            attended.reshape(
                batch, self.num_heads, self.n_win, self.n_win,
                window_height, window_width, self.head_dim,
            )
            .permute(0, 1, 6, 2, 4, 3, 5)
            .reshape(batch, channels, padded_height, padded_width)
        )
        if self.local_position is not None:
            attended = attended + self.local_position(value_padded)
        refined = self.output_projection(attended)[..., :height, :width]
        return x + self.layer_scale.to(refined.dtype) * refined


def build_c4_attention(name, *, lsk=None, litemla=None, dat=None, bra=None):
    if not isinstance(name, str):
        raise ValueError("c4_attention must be 'none', 'lsk', 'litemla', 'dat', or 'bra'")
    name = name.lower()
    if name == "none":
        return nn.Identity()
    if name == "lsk":
        return LSKRefinement(channels=64, **(lsk or {}))
    if name == "litemla":
        return LiteMLARefinement(channels=64, **(litemla or {}))
    if name == "dat":
        return DeformableAttentionRefinement(channels=64, **(dat or {}))
    if name == "bra":
        return BiLevelRoutingAttentionRefinement(channels=64, **(bra or {}))
    raise ValueError(
        f"Unsupported C4 attention {name!r}; expected "
        "'none', 'lsk', 'litemla', 'dat', or 'bra'"
    )

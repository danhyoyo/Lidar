import torch
import torch.nn as nn
from detector.core.models.encoders.mamba_ops import SelectiveSSM
from detector.core.models.encoders.pillar_ops import group_and_sort_pillars


class RichMambaEncoder(nn.Module):
    """
    Intra-Pillar Height-Causal State Space BEV Encoder.
    Processes Z-sorted points within each pillar using a Selective SSM,
    pools features without zero-padding pollution, and scatters into an (8, H, W) BEV pseudo-image.
    """
    def __init__(self, config: dict, geometry: dict):
        super().__init__()
        self.geometry = geometry
        self.d_model = int(config.get("d_model", 16))
        self.d_state = int(config.get("d_state", 16))
        self.max_points = int(config.get("max_points_per_pillar", 20))
        self.max_pillars = int(config.get("max_pillars", 32000))
        self.out_channels = int(config.get("out_channels", 8))
        self.use_dual_pooling = bool(config.get("use_dual_pooling", True))

        self.x_size = int(round((float(geometry["x_max"]) - float(geometry["x_min"])) / float(geometry["x_res"])))
        self.y_size = int(round((float(geometry["y_max"]) - float(geometry["y_min"])) / float(geometry["y_res"])))

        # Point projection: 8 enriched coordinates -> d_model using LayerNorm (safe on any pillar batch size)
        self.in_proj = nn.Sequential(
            nn.Linear(8, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
        )

        # 1D Selective SSM along the vertical elevation axis
        self.ssm = SelectiveSSM(self.d_model, d_state=self.d_state)

        # Output projection from pooled representation to out_channels
        pooled_dim = self.d_model * 2 if self.use_dual_pooling else self.d_model
        self.out_proj = nn.Sequential(
            nn.Linear(pooled_dim, self.out_channels),
            nn.LayerNorm(self.out_channels),
            nn.SiLU(),
        )

    def forward(self, points):
        """
        Args:
            points: Tensor of (N, 4), or (B, N, 4), or list of B tensors [(N_1, 4), ... (N_B, 4)]
        Returns:
            bev_map: Tensor of (B, out_channels, H, W)
        """
        if isinstance(points, (list, tuple)):
            batch_list = [self._encode_single_scene(p) for p in points]
            return torch.stack(batch_list, dim=0)
        elif isinstance(points, torch.Tensor) and points.ndim == 2:
            return self._encode_single_scene(points).unsqueeze(0)
        elif isinstance(points, torch.Tensor) and points.ndim == 3:
            batch_list = [self._encode_single_scene(points[b]) for b in range(points.shape[0])]
            return torch.stack(batch_list, dim=0)
        else:
            raise ValueError(f"Unsupported points input shape: {points.shape if hasattr(points, 'shape') else type(points)}")

    def _encode_single_scene(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        bev_map = torch.zeros(
            (self.out_channels, self.y_size, self.x_size),
            dtype=torch.float32,
            device=device,
        )

        if points.shape[0] == 0:
            return bev_map

        pillar_feats, pillar_indices, num_pillars, pillar_point_counts = group_and_sort_pillars(
            points,
            self.geometry,
            max_points_per_pillar=self.max_points,
            max_pillars=self.max_pillars,
        )

        if num_pillars == 0:
            return bev_map

        # Direct projection with LayerNorm: (P, K, 8) -> (P, K, d_model)
        projected = self.in_proj(pillar_feats)

        # Process through SSM along elevation: (P, K, d_model)
        ssm_out = self.ssm(projected)

        P, K, D = ssm_out.shape

        # Dual-pooling without zero-padding pollution:
        # 1. Terminal state at the actual top point of each pillar
        terminal_state = ssm_out[torch.arange(P, device=device), pillar_point_counts - 1]

        # 2. Max-pooling strictly over valid points
        mask = torch.arange(K, device=device).unsqueeze(0) < pillar_point_counts.unsqueeze(1)  # (P, K)
        ssm_out_valid = ssm_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_state = torch.max(ssm_out_valid, dim=1).values

        if self.use_dual_pooling:
            fused = torch.cat([terminal_state, max_state], dim=-1)
        else:
            fused = max_state

        # Output projection: (P, out_channels)
        out_feats = self.out_proj(fused)

        # Scatter to 2D BEV map: pillar_indices[:, 0] is y, pillar_indices[:, 1] is x
        y_coords = pillar_indices[:, 0]
        x_coords = pillar_indices[:, 1]
        bev_map[:, y_coords, x_coords] = out_feats.t()

        return bev_map

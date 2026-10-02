# RichMamba Intra-Pillar Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the RichMamba Intra-Pillar Height-Causal State Space Encoder as a learnable drop-in replacement for hand-crafted `rich8`, boosting small-object 3D detection on KITTI.

**Architecture:** A local 1D Selective State Space Model (SSM) processes elevation-sorted LiDAR points within each pillar cell. Features are pooled via dual-pooling (terminal hidden state + max-pooling) and scattered into a 2D BEV map feeding MobilePixorNeXt. Supports dual-engine execution: Native CUDA (`mamba_ssm`) with a Pure PyTorch Fallback.

**Tech Stack:** PyTorch 2.x, NumPy, `mamba_ssm` (optional with pure PyTorch fallback), pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-richmamba-encoder-design.md`

## Global Constraints
- Python 3.10+ and PyTorch 2.0+ compatible.
- Pure PyTorch fallback must require zero external CUDA/C++ libraries beyond PyTorch and NumPy.
- Output BEV feature map must strictly match `(B, C_out, 800, 704)` where `C_out = 8` by default.
- SSM recurrence computation must execute in FP32 accumulation to prevent exponential underflow/overflow.
- Pillar buffer capped at `max_pillars = 32000` to guarantee deterministic GPU VRAM safety.

---

### Task 1: Dual-Engine Selective State Space Operator (`mamba_ops.py`)

**Files:**
- Create: `detector/core/models/encoders/mamba_ops.py`
- Test: `tests/test_mamba_ops.py`

**Interfaces:**
- Produces: `class SelectiveSSM(nn.Module)`
  - `__init__(d_model: int, d_state: int = 16, dt_rank: str | int = "auto")`
  - `forward(x: torch.Tensor) -> torch.Tensor`: Input `(B, L, D)`, output `(B, L, D)`

- [ ] **Step 1: Write the failing test for SelectiveSSM**

Create `tests/test_mamba_ops.py`:
```python
import pytest
import torch
from detector.core.models.encoders.mamba_ops import SelectiveSSM, pure_pytorch_selective_scan

def test_selective_ssm_shape_and_grad():
    torch.manual_seed(42)
    B, L, D = 4, 16, 16
    x = torch.randn(B, L, D, requires_grad=True)
    ssm = SelectiveSSM(d_model=D, d_state=16)
    out = ssm(x)
    assert out.shape == (B, L, D)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_pure_pytorch_scan_numerical_finite():
    torch.manual_seed(42)
    B, L, D, N = 2, 8, 8, 16
    u = torch.randn(B, L, D)
    delta = torch.rand(B, L, D) * 0.1
    A = -torch.rand(D, N)
    B_tensor = torch.randn(B, L, N)
    C_tensor = torch.randn(B, L, N)
    D_tensor = torch.randn(D)
    
    y = pure_pytorch_selective_scan(u, delta, A, B_tensor, C_tensor, D_tensor)
    assert y.shape == (B, L, D)
    assert torch.isfinite(y).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mamba_ops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detector.core.models.encoders'`

- [ ] **Step 3: Implement `SelectiveSSM` and `pure_pytorch_selective_scan`**

Create `detector/core/models/encoders/mamba_ops.py`:
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    selective_scan_fn = None
    HAS_MAMBA_CUDA = False


def pure_pytorch_selective_scan(u, delta, A, B, C, D=None):
    """
    Pure PyTorch sequential fallback for selective scan.
    Args:
        u: (B, L, D)
        delta: (B, L, D)
        A: (D, N)
        B: (B, L, N)
        C: (B, L, N)
        D: (D,) optional
    Returns:
        y: (B, L, D)
    """
    batch_size, seq_len, d_model = u.shape
    d_state = A.shape[1]

    # Compute discretization in FP32
    u_f32 = u.float()
    delta_f32 = delta.float()
    A_f32 = A.float()
    B_f32 = B.float()
    C_f32 = C.float()

    # deltaA = exp(delta * A): (B, L, D, N)
    deltaA = torch.exp(delta_f32.unsqueeze(-1) * A_f32.unsqueeze(0).unsqueeze(0))
    # deltaB_u = (delta * u) * B: (B, L, D, N)
    deltaB_u = (delta_f32 * u_f32).unsqueeze(-1) * B_f32.unsqueeze(2)

    h = torch.zeros(batch_size, d_model, d_state, device=u.device, dtype=torch.float32)
    ys = []

    for t in range(seq_len):
        h = deltaA[:, t] * h + deltaB_u[:, t]
        # y_t = (h_t * C_t).sum(-1): (B, D)
        y_t = torch.einsum("bdn,bn->bd", h, C_f32[:, t])
        ys.append(y_t)

    y = torch.stack(ys, dim=1)
    if D is not None:
        y = y + u_f32 * D.float().unsqueeze(0).unsqueeze(0)

    return y.to(dtype=u.dtype)


class SelectiveSSM(nn.Module):
    """
    Dual-Engine Selective State Space Model block.
    Uses fused CUDA selective_scan_fn if installed; otherwise falls back to pure PyTorch.
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: str | int = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # Projections for selective parameters
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Initialize S4 parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        # dt_proj initialization
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize dt bias to ~0.001 to 0.1
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        batch_size, seq_len, d_model = x.shape
        A = -torch.exp(self.A_log.float())  # (D, N)

        # x_proj outputs dt_rank + B + C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt_part = x_dbl[:, :, : self.dt_rank]
        B_part = x_dbl[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C_part = x_dbl[:, :, self.dt_rank + self.d_state :]

        dt = F.softplus(self.dt_proj(dt_part))  # (B, L, D)

        if HAS_MAMBA_CUDA and x.is_cuda and selective_scan_fn is not None:
            # mamba_ssm expects (B, D, L)
            u_t = x.transpose(1, 2).contiguous()
            delta_t = dt.transpose(1, 2).contiguous()
            B_t = B_part.transpose(1, 2).contiguous()
            C_t = C_part.transpose(1, 2).contiguous()
            out = selective_scan_fn(u_t, delta_t, A, B_t, C_t, self.D.float(), z=None, delta_bias=None, delta_softplus=False)
            return out.transpose(1, 2).contiguous().to(dtype=x.dtype)
        else:
            return pure_pytorch_selective_scan(x, dt, A, B_part, C_part, self.D)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mamba_ops.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detector/core/models/encoders/mamba_ops.py tests/test_mamba_ops.py
git commit -m "feat(mamba): add dual-engine SelectiveSSM operator with pure PyTorch fallback"
```

---

### Task 2: Pillar Grouping & Z-Axis Canonical Sorting (`pillar_ops.py`)

**Files:**
- Create: `detector/core/models/encoders/pillar_ops.py`
- Test: `tests/test_pillar_ops.py`

**Interfaces:**
- Produces: `def group_and_sort_pillars(points, geometry, max_points_per_pillar=20, max_pillars=32000)`
  - Input: `points` tensor `(N, 4)` $[x, y, z, r]$, `geometry` dict
  - Output:
    - `pillar_features`: `(P, K, 8)` enriched 8-dim coordinates sorted by elevation
    - `pillar_indices`: `(P, 2)` 2D pixel coordinates $(u, v)$ on BEV grid
    - `num_pillars`: integer count $P$

- [ ] **Step 1: Write the failing test for pillar grouping & sorting**

Create `tests/test_pillar_ops.py`:
```python
import pytest
import torch
from detector.core.models.encoders.pillar_ops import group_and_sort_pillars

def test_group_and_sort_pillars():
    geometry = {
        "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
        "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
    }
    # Create 5 points: 3 in cell (10, 20), 2 in cell (30, 40)
    # Give cell (10, 20) points with unsorted z
    points = torch.tensor([
        [1.05, -38.05, 0.5, 0.8],    # z = 0.5
        [1.05, -38.05, -1.2, 0.4],   # z = -1.2
        [1.05, -38.05, -0.1, 0.6],   # z = -0.1
        [3.05, -36.05, 0.2, 0.9],
        [3.05, -36.05, -0.8, 0.3],
    ], dtype=torch.float32)

    features, indices, num_pillars = group_and_sort_pillars(
        points, geometry, max_points_per_pillar=20, max_pillars=32000
    )
    assert num_pillars == 2
    assert features.shape == (2, 20, 8)
    assert indices.shape == (2, 2)

    # Check Z-sorting in the first pillar (3 points)
    # The z-coordinates should be strictly ascending: -1.2, -0.1, 0.5
    # Point features format: [dx, dy, dz, x, y, z, r, z - z_min]
    z_vals = features[0, :3, 5]
    assert z_vals[0] < z_vals[1] < z_vals[2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pillar_ops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detector.core.models.encoders.pillar_ops'`

- [ ] **Step 3: Implement `pillar_ops.py`**

Create `detector/core/models/encoders/pillar_ops.py`:
```python
import torch


def filter_roi_points(points: torch.Tensor, geometry: dict) -> torch.Tensor:
    """Filter non-finite points and keep points within ROI geometry bounds."""
    eps = 0.001
    valid = torch.isfinite(points[:, :4]).all(dim=1)
    valid &= points[:, 0] > (geometry["x_min"] + eps)
    valid &= points[:, 0] < (geometry["x_max"] - eps)
    valid &= points[:, 1] > (geometry["y_min"] + eps)
    valid &= points[:, 1] < (geometry["y_max"] - eps)
    valid &= points[:, 2] > (geometry["z_min"] + eps)
    valid &= points[:, 2] < (geometry["z_max"] - eps)
    pts = points[valid].clone()
    if pts.shape[0] > 0:
        pts[:, 3] = torch.clamp(pts[:, 3], 0.0, 1.0)
    return pts


def group_and_sort_pillars(
    points: torch.Tensor,
    geometry: dict,
    max_points_per_pillar: int = 20,
    max_pillars: int = 32000,
):
    """
    Groups points into BEV pillars, enriches coordinates, and sorts points by elevation (Z).
    Returns:
        pillar_features: (P, max_points_per_pillar, 8)
        pillar_indices: (P, 2) [y_idx, x_idx]
        num_pillars: int
    """
    pts = filter_roi_points(points, geometry)
    device = points.device

    if pts.shape[0] == 0:
        empty_feat = torch.zeros(
            (0, max_points_per_pillar, 8), dtype=torch.float32, device=device
        )
        empty_idx = torch.zeros((0, 2), dtype=torch.int64, device=device)
        return empty_feat, empty_idx, 0

    x_res = geometry["x_res"]
    y_res = geometry["y_res"]
    x_min = geometry["x_min"]
    y_min = geometry["y_min"]
    z_min = geometry["z_min"]
    z_max = geometry["z_max"]
    z_c = (z_min + z_max) / 2.0

    # Grid indices
    x_idx = ((pts[:, 0] - x_min) / x_res).long()
    y_idx = ((pts[:, 1] - y_min) / y_res).long()

    # Unique 1D grid cell IDs
    x_size = int(round((geometry["x_max"] - x_min) / x_res))
    flat_ids = y_idx * x_size + x_idx

    unique_flat_ids, inverse_indices = torch.unique(flat_ids, return_inverse=True)
    num_pillars = min(int(unique_flat_ids.shape[0]), max_pillars)

    if num_pillars < unique_flat_ids.shape[0]:
        # Keep top max_pillars
        unique_flat_ids = unique_flat_ids[:num_pillars]
        mask = inverse_indices < num_pillars
        pts = pts[mask]
        inverse_indices = inverse_indices[mask]

    # Pre-allocate output tensors
    pillar_features = torch.zeros(
        (num_pillars, max_points_per_pillar, 8), dtype=torch.float32, device=device
    )
    pillar_indices = torch.zeros((num_pillars, 2), dtype=torch.int64, device=device)

    # Decode 2D indices for active pillars
    pillar_indices[:, 0] = unique_flat_ids // x_size  # y_idx
    pillar_indices[:, 1] = unique_flat_ids % x_size   # x_idx

    # Pillar centers in metric space
    xc = (pillar_indices[:, 1].float() + 0.5) * x_res + x_min
    yc = (pillar_indices[:, 0].float() + 0.5) * y_res + y_min

    # Sort each pillar by elevation z
    for p in range(num_pillars):
        p_mask = inverse_indices == p
        pts_in_p = pts[p_mask]
        if pts_in_p.shape[0] == 0:
            continue

        # Sort ascending by z-coordinate
        z_order = torch.argsort(pts_in_p[:, 2])
        pts_sorted = pts_in_p[z_order]

        k = min(pts_sorted.shape[0], max_points_per_pillar)
        selected_pts = pts_sorted[:k]

        # Point enrichment: [dx, dy, dz, x, y, z, r, z - z_min]
        dx = selected_pts[:, 0] - xc[p]
        dy = selected_pts[:, 1] - yc[p]
        dz = selected_pts[:, 2] - z_c
        z_rel = selected_pts[:, 2] - z_min

        enriched = torch.stack(
            [
                dx,
                dy,
                dz,
                selected_pts[:, 0],
                selected_pts[:, 1],
                selected_pts[:, 2],
                selected_pts[:, 3],
                z_rel,
            ],
            dim=-1,
        )

        pillar_features[p, :k] = enriched

    return pillar_features, pillar_indices, num_pillars
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pillar_ops.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detector/core/models/encoders/pillar_ops.py tests/test_pillar_ops.py
git commit -m "feat(mamba): add pillar grouping, point enrichment, and elevation sorting ops"
```

---

### Task 3: RichMambaEncoder & BEV Scattering (`rich_mamba.py`)

**Files:**
- Create: `detector/core/models/encoders/rich_mamba.py`
- Create: `detector/core/models/encoders/__init__.py`
- Test: `tests/test_rich_mamba.py`

**Interfaces:**
- Produces: `class RichMambaEncoder(nn.Module)`
  - `__init__(config: dict, geometry: dict)`
  - `forward(points_batch: list[torch.Tensor] | torch.Tensor) -> torch.Tensor`: Outputs `(B, 8, H, W)`

- [ ] **Step 1: Write the failing test for RichMambaEncoder**

Create `tests/test_rich_mamba.py`:
```python
import pytest
import torch
from detector.core.models.encoders.rich_mamba import RichMambaEncoder

def test_rich_mamba_encoder_forward_and_backward():
    geometry = {
        "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
        "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
    }
    cfg = {
        "d_model": 16,
        "d_state": 16,
        "max_points_per_pillar": 20,
        "max_pillars": 1000,
        "out_channels": 8,
    }
    encoder = RichMambaEncoder(cfg, geometry)
    
    # 200 random points in ROI
    N = 200
    pts = torch.rand(N, 4)
    pts[:, 0] = pts[:, 0] * 60.0 + 5.0
    pts[:, 1] = pts[:, 1] * 70.0 - 35.0
    pts[:, 2] = pts[:, 2] * 3.0 - 2.0
    pts[:, 3] = torch.rand(N)

    bev = encoder(pts)
    assert bev.shape == (1, 8, 800, 704)
    assert torch.isfinite(bev).all()

    loss = bev.sum()
    loss.backward()

    # Verify gradients flow into encoder weights
    assert encoder.in_proj.weight.grad is not None
    assert torch.isfinite(encoder.in_proj.weight.grad).all()

def test_rich_mamba_empty_scene():
    geometry = {
        "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
        "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
    }
    encoder = RichMambaEncoder({}, geometry)
    empty_pts = torch.zeros((0, 4))
    bev = encoder(empty_pts)
    assert bev.shape == (1, 8, 800, 704)
    assert (bev == 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rich_mamba.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detector.core.models.encoders.rich_mamba'`

- [ ] **Step 3: Implement `rich_mamba.py` and `__init__.py`**

Create `detector/core/models/encoders/rich_mamba.py`:
```python
import torch
import torch.nn as nn
from detector.core.models.encoders.mamba_ops import SelectiveSSM
from detector.core.models.encoders.pillar_ops import group_and_sort_pillars


class RichMambaEncoder(nn.Module):
    """
    Intra-Pillar Height-Causal State Space BEV Encoder.
    Processes Z-sorted points within each pillar using a Selective SSM,
    pools features, and scatters into an (8, H, W) BEV pseudo-image.
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

        self.x_size = int(round((geometry["x_max"] - geometry["x_min"]) / geometry["x_res"]))
        self.y_size = int(round((geometry["y_max"] - geometry["y_min"]) / geometry["y_res"]))

        # Point projection: 8 enriched coordinates -> d_model
        self.in_proj = nn.Sequential(
            nn.Linear(8, self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU(),
        )

        # 1D Selective SSM along the vertical axis
        self.ssm = SelectiveSSM(self.d_model, d_state=self.d_state)

        # Output projection from pooled representation to out_channels
        pooled_dim = self.d_model * 2 if self.use_dual_pooling else self.d_model
        self.out_proj = nn.Sequential(
            nn.Linear(pooled_dim, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.SiLU(),
        )

    def forward(self, points):
        """
        Args:
            points: Tensor of (N, 4) or list of B tensors [(N_1, 4), ... (N_B, 4)]
        Returns:
            bev_map: Tensor of (B, out_channels, H, W)
        """
        if isinstance(points, (list, tuple)):
            batch_list = [self._encode_single_scene(p) for p in points]
            return torch.stack(batch_list, dim=0)
        elif points.ndim == 2:
            return self._encode_single_scene(points).unsqueeze(0)
        elif points.ndim == 3:
            batch_list = [self._encode_single_scene(points[b]) for b in range(points.shape[0])]
            return torch.stack(batch_list, dim=0)
        else:
            raise ValueError(f"Unsupported points input shape: {points.shape}")

    def _encode_single_scene(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        bev_map = torch.zeros(
            (self.out_channels, self.y_size, self.x_size),
            dtype=torch.float32,
            device=device,
        )

        if points.shape[0] == 0:
            return bev_map

        pillar_feats, pillar_indices, num_pillars = group_and_sort_pillars(
            points,
            self.geometry,
            max_points_per_pillar=self.max_points,
            max_pillars=self.max_pillars,
        )

        if num_pillars == 0:
            return bev_map

        # Flatten (P, K, 8) -> (P*K, 8) for BatchNorm1d
        P, K, _ = pillar_feats.shape
        flat_feats = pillar_feats.view(P * K, 8)
        projected = self.in_proj(flat_feats).view(P, K, self.d_model)

        # Process through SSM: (P, K, d_model)
        ssm_out = self.ssm(projected)

        # Dual-pooling: terminal state (P, d_model) + max-pooling (P, d_model)
        if self.use_dual_pooling:
            terminal_state = ssm_out[:, -1, :]
            max_state = torch.max(ssm_out, dim=1).values
            fused = torch.cat([terminal_state, max_state], dim=-1)
        else:
            fused = torch.max(ssm_out, dim=1).values

        # Output projection: (P, out_channels)
        out_feats = self.out_proj(fused)

        # Scatter to 2D BEV map: indices[:, 0] is y, indices[:, 1] is x
        y_coords = pillar_indices[:, 0]
        x_coords = pillar_indices[:, 1]
        bev_map[:, y_coords, x_coords] = out_feats.t()

        return bev_map
```

Create `detector/core/models/encoders/__init__.py`:
```python
from detector.core.models.encoders.rich_mamba import RichMambaEncoder

__all__ = ["RichMambaEncoder"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rich_mamba.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detector/core/models/encoders/rich_mamba.py detector/core/models/encoders/__init__.py tests/test_rich_mamba.py
git commit -m "feat(mamba): implement RichMambaEncoder with dual-pooling and 2D BEV scattering"
```

---

### Task 4: Integrate `RichMambaEncoder` into `CustomModel`

**Files:**
- Modify: `detector/core/models/model.py:7-40`
- Test: `tests/test_model_rich_mamba_e2e.py`

**Interfaces:**
- Consumes: `RichMambaEncoder` from `detector.core.models.encoders`
- Produces: `CustomModel` capable of receiving raw points when `bev_encoding["name"] == "rich_mamba"`.

- [ ] **Step 1: Write the failing e2e test for `CustomModel` with RichMamba**

Create `tests/test_model_rich_mamba_e2e.py`:
```python
import pytest
import torch
from detector.core.models.model import CustomModel

def test_custom_model_with_rich_mamba():
    cfg = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
        "bev_encoding": {
            "name": "rich_mamba",
            "d_model": 16,
            "d_state": 16,
            "max_points_per_pillar": 20,
            "max_pillars": 500,
            "out_channels": 8,
        },
        "geometry": {
            "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
            "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
            "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
        }
    }
    model = CustomModel(cfg, num_classes=4, input_channels=8)

    # Synthetic point cloud
    pts = torch.rand(150, 4)
    pts[:, 0] = pts[:, 0] * 60.0 + 5.0
    pts[:, 1] = pts[:, 1] * 70.0 - 35.0
    pts[:, 2] = pts[:, 2] * 3.0 - 2.0
    pts[:, 3] = torch.rand(150)

    outputs = model(pts)
    assert "cls" in outputs
    assert "offset" in outputs
    assert "size" in outputs
    assert "yaw" in outputs
    assert outputs["cls"].shape == (1, 4, 200, 176)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_rich_mamba_e2e.py -v`
Expected: FAIL with attribute error or dimension mismatch because `CustomModel` does not yet accept raw points.

- [ ] **Step 3: Modify `detector/core/models/model.py`**

Edit [`detector/core/models/model.py`](file:///home/duyennh/AI_projects/research_lidar/Lidar/detector/core/models/model.py):
```python
import torch.nn as nn

from core.models.backbones.registry import build_backbone
from core.models.heads.cnn import Header
from detector.core.models.encoders.rich_mamba import RichMambaEncoder


class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes=4, input_channels=35):
        super(CustomModel, self).__init__()
        bev_cfg = cfg.get("bev_encoding", {})
        bev_name = bev_cfg.get("name", "binary_slices")
        geometry = cfg.get("geometry", {
            "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
            "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
            "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
        })

        if bev_name == "rich_mamba":
            self.encoder = RichMambaEncoder(bev_cfg, geometry)
            input_channels = int(bev_cfg.get("out_channels", 8))
        else:
            self.encoder = None

        backbone_name = str(cfg.get("backbone", "mobilepixor"))
        self.backbone = build_backbone(backbone_name, cfg, input_channels=input_channels)

        self.num_classes = num_classes
        cls_encoding = str(cfg.get("cls_encoding", "gaussian")).lower()
        if cls_encoding == "binary":
            self.num_classes += 1

        is_mobilepixornext = backbone_name.lower() == "mobilepixornext"
        use_bn = cfg.get("header_use_bn", is_mobilepixornext)
        act = cfg.get("header_act", "silu" if is_mobilepixornext else "none")

        backbone_out_dim = cfg.get("backbone_out_dim", 16)

        use_iou = bool(cfg.get("header_use_iou", False))
        self.header = Header(
            self.num_classes,
            backbone_out_dim,
            use_bn=use_bn,
            act=act,
            use_iou=use_iou,
        )

    def forward(self, x):
        if self.encoder is not None and not isinstance(x, torch.Tensor) or (isinstance(x, torch.Tensor) and x.ndim <= 3):
            # Input is raw points (N, 4) or (B, N, 4) or list of points
            x = self.encoder(x)
        features = self.backbone(x)
        pred = self.header(features)
        return pred

    def switch_to_deploy(self):
        """Deploy-time weight fusion for the backbone."""
        if hasattr(self.backbone, "switch_to_deploy"):
            self.backbone.switch_to_deploy()
        return self

    def export_deploy_state_dict(self):
        """Switch to deploy mode and return the fused state_dict."""
        self.switch_to_deploy()
        return self.state_dict()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_rich_mamba_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detector/core/models/model.py tests/test_model_rich_mamba_e2e.py
git commit -m "feat(mamba): integrate RichMambaEncoder into CustomModel forward pipeline"
```

---

### Task 5: Training Configuration & End-to-End Validation

**Files:**
- Create: `configs/kitti/rich_mamba/kitti_mobilepixornext_richmamba.json`
- Test: `tests/test_rich_mamba_config_dryrun.py`

- [ ] **Step 1: Write the configuration dry-run test**

Create `tests/test_rich_mamba_config_dryrun.py`:
```python
import json
import pytest
import torch
from pathlib import Path
from detector.core.models.model import CustomModel

def test_richmamba_config_instantiation():
    config_path = Path("configs/kitti/rich_mamba/kitti_mobilepixornext_richmamba.json")
    assert config_path.exists()

    with open(config_path) as f:
        cfg = json.load(f)

    model = CustomModel(cfg["model"])
    assert model.encoder is not None
    assert type(model.backbone).__name__ == "MobilePixorNeXtBackbone"

    # Forward dummy points
    pts = torch.rand(50, 4)
    pts[:, 0] = pts[:, 0] * 50.0 + 5.0
    pts[:, 1] = pts[:, 1] * 60.0 - 30.0
    pts[:, 2] = pts[:, 2] * 2.5 - 2.0
    pts[:, 3] = torch.rand(50)

    out = model(pts)
    assert out["cls"].shape == (1, 4, 200, 176)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rich_mamba_config_dryrun.py -v`
Expected: FAIL with `AssertionError: assert config_path.exists()`

- [ ] **Step 3: Create `configs/kitti/rich_mamba/kitti_mobilepixornext_richmamba.json`**

Create `configs/kitti/rich_mamba/kitti_mobilepixornext_richmamba.json`:
```json
{
  "dataset": "kitti",
  "num_classes": 4,
  "out_size_factor": 4,
  "train": {
    "epochs": 70,
    "batch_size": 4,
    "lr": 0.001,
    "weight_decay": 0.0001
  },
  "data": {
    "bev_encoding": {
      "name": "rich_mamba",
      "d_model": 16,
      "d_state": 16,
      "max_points_per_pillar": 20,
      "max_pillars": 32000,
      "out_channels": 8,
      "use_dual_pooling": true
    }
  },
  "model": {
    "backbone": "mobilepixornext",
    "backbone_out_dim": 16,
    "cls_encoding": "gaussian",
    "header_use_bn": true,
    "header_act": "silu",
    "header_use_iou": true,
    "bev_encoding": {
      "name": "rich_mamba",
      "d_model": 16,
      "d_state": 16,
      "max_points_per_pillar": 20,
      "max_pillars": 32000,
      "out_channels": 8,
      "use_dual_pooling": true
    },
    "geometry": {
      "x_min": 0.0,
      "x_max": 70.4,
      "x_res": 0.1,
      "y_min": -40.0,
      "y_max": 40.0,
      "y_res": 0.1,
      "z_min": -2.5,
      "z_max": 1.0,
      "z_res": 0.1
    }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rich_mamba_config_dryrun.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to ensure zero regressions**

Run: `pytest tests/test_mamba_ops.py tests/test_pillar_ops.py tests/test_rich_mamba.py tests/test_model_rich_mamba_e2e.py tests/test_rich_mamba_config_dryrun.py -v`
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add configs/kitti/rich_mamba/kitti_mobilepixornext_richmamba.json tests/test_rich_mamba_config_dryrun.py
git commit -m "feat(mamba): add RichMamba training config and end-to-end integration validation"
```

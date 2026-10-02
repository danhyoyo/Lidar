# RichMamba: Intra-Pillar Height-Causal State Space Encoder for 3D LiDAR Object Detection

## 1. Executive Summary & Problem Formulation

### 1.1. Context & Motivation
In 3D LiDAR object detection for autonomous driving and robotics, transforming unstructured, irregular 3D point clouds $\mathcal{P} \in \mathbb{R}^{N \times 4}$ into structured 2D Bird's Eye View (BEV) representations is the critical first stage of the perception pipeline. 

The baseline architecture in this repository relies on **`rich8`** (RichBEV-8), a hand-crafted statistical BEV encoder that projects points into a $0.1\text{m} \times 0.1\text{m}$ grid and compresses each pillar into 8 pre-defined metrics:
- 3 coarse height-band occupancy flags ($[0, 1/3), [1/3, 2/3), [2/3, 1.0]$)
- Height statistics: $z_{\max}$ and $z_{\text{mean}}$
- Reflectance statistics: $i_{\max}$ and $i_{\text{mean}}$
- Logarithmic point density: $\min(1.0, \ln(1 + N) / \ln(1 + 32))$

### 1.2. The Small-Object Bottleneck
While `rich8` achieves a $77.1\%$ reduction in memory bandwidth and payload ($18.02\text{ MB/frame}$ vs $78.85\text{ MB/frame}$ of legacy 35-slice binary occupancy), it introduces severe limitations for small and thin objects—specifically **Pedestrians** ($\sim 0.8\text{m} \times 0.6\text{m} \times 1.7\text{m}$) and **Cyclists** ($\sim 1.7\text{m} \times 0.6\text{m} \times 1.7\text{m}$):
1. **Vertical Discretization Loss:** Spanning a $3.5\text{m}$ elevation range ($[-2.5\text{m}, +1.0\text{m}]$), each of the 3 height bands spans $\sim 1.17\text{m}$. A pedestrian's vertical profile is collapsed into only 1 or 2 bands, obliterating the geometric cues that distinguish a human torso from road furniture, poles, or tree trunks.
2. **Fixed Hand-Crafted Representation (Zero Learnable Parameters):** The statistical metrics ($z_{\max}, z_{\text{mean}}, i_{\max}, i_{\text{mean}}$) are non-differentiable CPU heuristics. Gradients cannot flow back to the point distribution, preventing end-to-end task-driven feature optimization.
3. **PointNet Max-Pooling Limitation:** Standard learnable pillar encoders (such as PointPillars) treat points within a pillar as an *unordered set*, discarding physical geometric ordering along the vertical axis.

### 1.3. The RichMamba Hypothesis
LiDAR pillars possess a natural, physical **causal direction along the elevation axis ($Z$)**:
$$\text{Ground Surface} \longrightarrow \text{Feet / Base} \longrightarrow \text{Torso} \longrightarrow \text{Head}$$
By sorting points within each pillar ascendingly by elevation ($z$) and processing them using a **Selective State Space Model (SSM / Mamba)**, the model can capture the continuous vertical structural transitions of small objects with linear computational complexity $\mathcal{O}(K)$, while retaining end-to-end learnable weights.

---

## 2. Related Work & Research Novelty Positioning

### 2.1. Landscape of Mamba in 3D Point Clouds
Recent advances have integrated Selective SSMs into 3D LiDAR perception:
- **Global BEV / Neck Mamba (Mamba-BEV, PillarMamba):** Apply 2D SSMs across the entire macro BEV feature map (flattened via raster or Hilbert curves) to replace 2D convolutions or Transformers in the backbone.
- **Sparse Voxel Serialization (Voxel Mamba, UniMamba):** Serialize 3D sparse voxels across the entire scene using space-filling curves (Morton/Z-order curves) to replace 3D sparse convolutions.
- **Point-Level Mamba (PointMamba, PointLAM):** Downsample the whole scene via Farthest Point Sampling (FPS) and serialize points globally.

### 2.2. Novelty of RichMamba
**RichMamba operates at a fundamentally different level of abstraction:**
- Rather than acting as a global backbone or scene serializer, RichMamba is a **Local, Intra-Pillar, Height-Causal State Space Encoder**.
- It is the first architecture to formulate the points within each vertical pillar as an elevation-ordered 1D trajectory, utilizing data-dependent selective scan parameters $(\Delta, \mathbf{B}, \mathbf{C})$ to dynamically discard ground clutter (setting step size $\Delta_t \to 0$) and retain anatomic transitions.
- It acts as a lightweight, drop-in neural replacement for `rich8` directly feeding the `< 2.0M` parameter **`MobilePixorNeXt`** backbone.

---

## 3. Mathematical Formulation

```
      Raw Points in Pillar (u, v)
                 │
                 ▼
       [ Elevation Sort: z_(1) <= z_(2) <= ... <= z_(K) ]
                 │
                 ▼
       [ Canonical Point Enrichment: p'_i in R^8 ]
                 │
                 ▼
       [ Input Linear Projection: e_t = W_in * p'_t in R^D ]
                 │
                 ▼
  ┌────────────────────────────────────────────────────────┐
  │         1D Selective State Space Layer (Mamba)         │
  │                                                        │
  │    h_t = exp(Δ_t A) h_{t-1} + (Δ_t A)^(-1)(exp(Δ_t A) - I) Δ_t B_t e_t │
  │    y_t = C_t h_t + D e_t                              │
  └────────────────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   [ Final State h_K ]  [ Max-Pool y_t ]
        └────────┬────────┘
                 ▼
    [ Dual-Pooling Fusion: [h_K || max(y)] ]
                 │
                 ▼
    [ Linear Head: W_out -> R^(C_out) ]  (C_out = 8)
                 │
                 ▼
    [ Scatter to 2D BEV Grid: (B, 8, 800, 704) ]
```

### 3.1. Pillar Canonicalization & Point Enrichment
Let $\mathcal{P}_{u, v} = \{p_i = (x_i, y_i, z_i, r_i)\}_{i=1}^{K}$ be the set of points within pillar $(u, v)$, bounded by spatial resolution $\Delta x = 0.1\text{m}, \Delta y = 0.1\text{m}$.
The pillar center coordinate is:
$$x_c = u \cdot \Delta x + x_{\min} + \frac{\Delta x}{2}, \quad y_c = v \cdot \Delta y + y_{\min} + \frac{\Delta y}{2}, \quad z_c = \frac{z_{\min} + z_{\max}}{2}$$

Each point is enriched into an 8-dimensional feature vector:
$$p'_i = \left[ x_i - x_c, \; y_i - y_c, \; z_i - z_c, \; x_i, \; y_i, \; z_i, \; r_i, \; z_i - z_{\min} \right]^T \in \mathbb{R}^8$$

### 3.2. Height-Causal Ordering
Points within each pillar are sorted in ascending order of their $z$-coordinate:
$$z_{(1)} \le z_{(2)} \le \dots \le z_{(K)}$$
Pillars with $K < K_{\max}$ ($K_{\max} = 20$) are zero-padded; pillars with $K > K_{\max}$ are sub-sampled preserving the minimum and maximum elevation extremes.

### 3.3. Selective State Space Dynamics
The enriched sequence is projected into latent dimension $D$ ($D \in \{16, 32\}$):
$$e_t = \mathbf{W}_{in} p'_{(t)} + b_{in}, \quad e_t \in \mathbb{R}^D$$

The discretized Selective State Space equations govern the hidden state evolution:
$$h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t e_t$$
$$y_t = \mathbf{C}_t h_t + \mathbf{D} e_t$$
where:
$$\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$$
$$\bar{\mathbf{B}}_t = (\Delta_t \mathbf{A})^{-1}(\exp(\Delta_t \mathbf{A}) - \mathbf{I}) \cdot \Delta_t \mathbf{B}_t$$

The selection mechanism parameterizes $\Delta_t, \mathbf{B}_t, \mathbf{C}_t$ as input-dependent projections of $e_t$:
$$\Delta_t = \text{softplus}(\mathbf{W}_\Delta e_t + b_\Delta)$$
$$\mathbf{B}_t = \mathbf{W}_B e_t, \quad \mathbf{C}_t = \mathbf{W}_C e_t$$
- **Ground Suppression:** When a point corresponds to uninformative asphalt reflections, the network predicts $\Delta_t \to 0$, causing $\bar{\mathbf{A}}_t \to \mathbf{I}$ and $\bar{\mathbf{B}}_t \to 0$, effectively passing through without polluting the memory state.
- **Structural Transition:** When laser returns encounter human knees, hips, or torso, $\Delta_t > 0$ triggers rapid state updates that store vertical geometry into $h_t$.

### 3.4. Dual-Pooling Representation
The final representation for pillar $(u, v)$ aggregates both the terminal contextual state $h_K$ and the peak salience across elevations:
$$F_{u, v} = \mathbf{W}_{out} \left( \left[ h_K \; \Vert \; \max_{t=1\dots K}(y_t) \right] \right) + b_{out} \in \mathbb{R}^{C_{out}}$$
where $\Vert$ denotes channel concatenation, and $C_{out} = 8$ by default.

### 3.5. Scatter to 2D BEV Grid
The dense tensor $\mathbf{V} \in \mathbb{R}^{B \times C_{out} \times H \times W}$ is constructed via 2D spatial scattering:
$$\mathbf{V}[:, :, v_k, u_k] = F_{u_k, v_k}$$
Cells with zero points remain all-zero tensors.

---

## 4. Architecture & System Integration

### 4.1. Directory & File Structure
```text
detector/core/models/
├── encoders/
│   ├── __init__.py           # Exports RichMambaEncoder and build_encoder
│   ├── mamba_ops.py          # Dual-Engine SSM: Native CUDA vs Pure PyTorch Fallback
│   └── rich_mamba.py         # RichMambaEncoder: pillar grouping, Z-sorting, SSM, scattering
├── backbones/
│   └── mobilepixornext.py    # Existing MobilePixorNeXt backbone (unchanged)
├── heads/
│   └── cnn.py                # Existing 5-task header (unchanged)
└── model.py                  # CustomModel: routes input points through RichMambaEncoder
```

### 4.2. Dual-Engine Execution Strategy (`mamba_ops.py`)
To ensure high training throughput on Linux CUDA servers while maintaining portability for standard PyTorch/ONNX execution:
1. **Engine 1: Native CUDA Kernel (`mamba_ssm`)**
   - Automatically detected via `import mamba_ssm` and `from mamba_ssm.ops.selective_scan_interface import selective_scan_fn`.
   - Utilizes fused hardware SRAM streaming for linear scan in $\mathcal{O}(K)$ time.
2. **Engine 2: Pure PyTorch Associative Scan Fallback**
   - Fallback activated automatically if `mamba_ssm` is not installed or when running on CPU/unsupported accelerators.
   - Implements parallel prefix associative scan using standard PyTorch tensor primitives.
   - Numerically verified to yield identical gradients and outputs within $\epsilon < 1e-4$.

### 4.3. Pipeline Routing in `CustomModel` ([`model.py`](file:///home/duyennh/AI_projects/research_lidar/Lidar/detector/core/models/model.py))
`CustomModel` inspects `cfg.get("bev_encoding", {}).get("name")`:
- If `rich8` or `binary_slices`: expects pre-voxelized tensor `batch["voxel"]` from CPU DataLoader (legacy mode).
- If `rich_mamba`: instantiates `self.encoder = RichMambaEncoder(cfg["bev_encoding"], geometry)` and takes `batch["points"]` as input in `forward(x)`:
  ```python
  def forward(self, x):
      if self.encoder is not None:
          x = self.encoder(x)  # (B, N, 4) -> (B, 8, 800, 704)
      features = self.backbone(x)
      pred = self.header(features)
      return pred
  ```

### 4.4. Configuration Schema
Any JSON config in `configs/kitti/` can activate `rich_mamba`:
```json
{
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
  }
}
```

---

## 5. Numerical Safety & Edge Case Handling

1. **Empty Point Clouds / Out-of-Bound Scenes:**
   - If a scene contains zero valid LiDAR points within the KITTI ROI bounds ($X \in [0, 70.4]$, $Y \in [-40, 40]$, $Z \in [-2.5, 1.0]$), `RichMambaEncoder` returns a zero tensor of shape $(B, C_{out}, H, W)$ with `requires_grad=True`, preventing empty tensor CUDA runtime errors.
2. **Single-Point Pillars ($K=1$):**
   - For sparse distant returns where a pillar contains only 1 point, the sequence length is 1. The SSM executes a single-step recurrent update with $h_0 = 0$. No division-by-zero or indexing exceptions occur.
3. **Non-Finite Coordinate Cleaning:**
   - All input coordinates are filtered via `torch.isfinite(points[:, :4]).all(dim=-1)` and reflectance values are strictly clamped to $[0.0, 1.0]$ via `torch.clamp`.
4. **FP32 Internal Accumulation for Selective Scan:**
   - Even when training under `torch.amp.autocast("cuda", dtype=torch.bfloat16)` or `torch.float16`, the core recurrence $\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$ is explicitly cast to `torch.float32` to prevent underflow/overflow during exponential discretization.
5. **Pillar Buffer Capping:**
   - To guard against memory spikes, the active pillar capacity is clamped at `max_pillars = 32000`. In standard KITTI frames, active pillars rarely exceed 18,000, ensuring guaranteed deterministic VRAM ceilings.

---

## 6. Verification & Test Plan

A dedicated unit test suite [`tests/test_rich_mamba.py`](file:///home/duyennh/AI_projects/research_lidar/Lidar/tests/test_rich_mamba.py) will be developed covering:

1. **`test_mamba_ops_dual_engine_consistency`:**
   - Generates random input tensors $(P, K, D)$ and runs both the Native CUDA kernel and the Pure PyTorch Fallback. Asserts maximum absolute error $\|y_{\text{cuda}} - y_{\text{fallback}}\|_\infty < 1e-4$.
2. **`test_rich_mamba_output_shape_and_range`:**
   - Passes synthetic KITTI point clouds shaped $(B, N, 4)$ through `RichMambaEncoder`. Asserts output shape is exactly $(B, 8, 800, 704)$, all values are finite, and memory remains within budget.
3. **`test_rich_mamba_gradient_flow`:**
   - Computes a dummy scalar loss $L = \sum \mathbf{V}$ and executes `L.backward()`. Verifies that all parameter gradients ($\nabla \mathbf{W}_{in}, \nabla \mathbf{W}_{\Delta}, \nabla \mathbf{W}_{out}$) exist, are non-zero, and contain no NaNs.
4. **`test_custom_model_e2e_integration`:**
   - Instantiates `CustomModel` with `rich_mamba` configuration, verifies forward pass produces valid `cls`, `offset`, `size`, `yaw` output dictionaries matching the target tensor shapes of `MobilePixorNeXt`.
5. **`test_empty_scene_handling`:**
   - Feeds an all-zero point cloud and asserts graceful zero-tensor return without crashing.

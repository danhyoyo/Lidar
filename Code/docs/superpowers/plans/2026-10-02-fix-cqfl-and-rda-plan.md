# CenterNet Quality Focal Loss (CQFL) & True Metric RDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 14,000+ False Positive explosion and dormant RDA weight in both `gw_qal` and `q_oga` by:
1. Re-engineering `quality_focal_loss.py` to be 100% compatible with CenterNet continuous Gaussian heatmaps (CQFL: Dirac peak negative suppression with $(1 - Y)^4$ attenuation and quality-modulated positive center loss).
2. Reconstructing true 3D metric world coordinates $(x_{\text{world}}, y_{\text{world}})$ from BEV grid meshes for Range-Density Adaptation (RDA) in `gw_qal.py` and `q_oga.py`.

**Architecture:**
- **CQFL:** At positive peak centers ($Y=1.0$), loss modulates confidence target with geometric alignment $y^* = Q^\beta$. At non-peak Gaussian neighbors ($Y < 1.0$) and background ($Y = 0$), negative focal loss with $(1 - Y)^4$ penalty reduction forces predictions $p \to 0$, eliminating Gaussian plateaus and restoring sharp Dirac needle peaks.
- **True Metric RDA:** Reconstructs metric cell centers $(x_{\text{grid}}, y_{\text{grid}})$ from spatial feature map indices $(h, w)$ and cell resolution, adding sub-cell offset $(dx, dy)$ to evaluate true radial sensor distance $r = \sqrt{x_{\text{world}}^2 + y_{\text{world}}^2} \in [0, 70.4\text{m}]$.

**Tech Stack:** Python 3.10+, PyTorch 2.11+, NumPy, PyTest.

**Spec / Diagnostic Reference:** Diagnostic findings from validation evaluations `evaluation_validation (1).json` (GW-QAL), `evaluation_validation (2).json` (OGA), and `evaluation_validation.json` (Q-OGA).

---

## File Structure & Proposed Modifications

- Modify: `detector/core/losses/quality_focal_loss.py`
  - Re-engineer `quality_focal_loss` into CenterNet Quality Focal Loss (CQFL).
- Modify: `detector/core/losses/strategies/gw_qal.py`
  - Reconstruct metric grid coordinates and calculate true LiDAR radial distance for RDA.
- Modify: `detector/core/losses/strategies/q_oga.py`
  - Reconstruct metric grid coordinates and calculate true LiDAR radial distance for RDA.
- Create: `tests/test_quality_focal_loss_cqfl.py`
  - Comprehensive unit tests: Dirac peak suppression on Gaussian disks, positive quality modulation, numerical stability in BF16/FP32, and mathematical identity to CenterNet modified focal loss when $Q=1.0$.
- Create: `tests/test_metric_rda.py`
  - Unit tests verifying metric radial distance computation across grid coordinates and RDA scaling $\omega(r)$ between $1.0$ and $2.1$.

---

### Task 1: CenterNet-Compatible Quality Focal Loss (CQFL)

**Files:**
- Test: `tests/test_quality_focal_loss_cqfl.py`
- Modify: `detector/core/losses/quality_focal_loss.py`

- [ ] **Step 1: Write failing unit tests for CQFL**
  Create `tests/test_quality_focal_loss_cqfl.py` testing:
  1. `test_exact_equivalence_to_centernet_when_quality_is_one`: When $Q = 1.0$, `quality_focal_loss` matches `modified_focal_loss` within $10^{-5}$.
  2. `test_suppression_of_gaussian_neighbor_plateau`: When $0 < Y < 1.0$ and prediction $p > 0$, gradient $\frac{\partial \mathcal{L}}{\partial p} > 0$ (pushes $p \to 0$, suppressing plateaus).
  3. `test_quality_modulation_at_positive_peak`: When $Q = 0.5$ and $\beta = 1.0$, optimal prediction at center pixel is $p = 0.5$.
  4. `test_empty_mask_stability`: When $Y = 0$ everywhere, returns finite positive loss with valid gradients.

- [ ] **Step 2: Run tests to confirm failure**
  Run: `python -m unittest tests/test_quality_focal_loss_cqfl.py`
  Confirm failure due to existing symmetric BCE behavior.

- [ ] **Step 3: Implement CQFL in `quality_focal_loss.py`**
  Update `quality_focal_loss.py`:
  - Separate `pos_mask = target_prob.ge(1.0)` and `neg_mask = target_prob.lt(1.0)`.
  - For positive peak pixels ($Y=1.0$), modulate positive target: $y^* = Q^\beta$, compute focal BCE.
  - For negative/neighbor pixels ($Y < 1.0$), compute CenterNet attenuated negative focal loss: $(1 - Y)^\alpha \cdot p^\gamma \cdot (-\log(1 - p))$.
  - Normalize sum by $N_{\text{pos}} = \text{pos\_mask.sum().clamp\_min(1.0)}$.

- [ ] **Step 4: Run tests to confirm all pass**
  Run: `python -m unittest tests/test_quality_focal_loss_cqfl.py`
  Verify all 4 test cases pass cleanly.

---

### Task 2: Metric Radial Distance Reconstruction for RDA in `gw_qal.py` and `q_oga.py`

**Files:**
- Test: `tests/test_metric_rda.py`
- Modify: `detector/core/losses/strategies/gw_qal.py`
- Modify: `detector/core/losses/strategies/q_oga.py`

- [ ] **Step 1: Write failing unit tests for Metric RDA**
  Create `tests/test_metric_rda.py`:
  1. `test_reconstructed_coordinates_at_known_grid_positions`: Verify cell $(h=100, w=150)$ maps to $(x \approx 60\text{m}, y \approx 0\text{m})$ and produces $\omega(r) \approx 2.0 \sim 2.1$.
  2. `test_near_vs_far_rda_differentiation`: Verify an object at $x=10\text{m}$ receives $\omega \approx 1.03$ while an object at $x=60\text{m}$ receives $\omega \approx 2.09$ (monotonic growth).

- [ ] **Step 2: Run test to confirm failure or missing functionality**
  Run: `python -m unittest tests/test_metric_rda.py`

- [ ] **Step 3: Implement true metric coordinate reconstruction**
  In `gw_qal.py` and `q_oga.py`:
  - Build coordinate grid mesh from feature map dimensions $(H, W)$ and geometry parameters $(x_{\min}, x_{\max}, y_{\min}, y_{\max})$.
  - Extract metric grid coordinates $(x_{\text{grid}}, y_{\text{grid}})$ for positive cells.
  - Compute true world coordinates: $(x_{\text{world}}, y_{\text{world}}) = (x_{\text{grid}} + dx, y_{\text{grid}} + dy)$.
  - Pass $(x_{\text{world}}, y_{\text{world}})$ into `compute_range_weights(world_coords, r_max=70.4, gamma=..., alpha=...)`.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `python -m unittest tests/test_metric_rda.py`

---

### Task 3: Regression & Integration Verification Across Full Test Suite

**Files:**
- Test: `tests/test_loss_strategy_gw_qal.py`
- Test: `tests/test_loss_strategy_q_oga.py`
- Test: `tests/test_gaussian_wasserstein.py`
- Test: `tests/test_smooth_corner_loss.py`
- Test: Existing regression tests in `tests/`

- [ ] **Step 1: Run full test suite**
  Run: `python -m unittest discover -s tests -p "test_*.py"`
  Confirm all unit and integration tests across the entire codebase pass with 0 errors and 0 failures.

- [ ] **Step 2: Verify Backward Compatibility**
  Ensure `baseline`, `uwag`, and `oga` strategies remain 100% unaffected.

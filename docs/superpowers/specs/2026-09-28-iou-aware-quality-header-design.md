# Pillar 4: IoU-Aware Quality Detection Header (IQA-Header) & Joint NMS Design Spec

**Date:** 2026-09-28  
**Author:** AI Pair Programmer & duyennh  
**Target Branch:** `feature/pillar4-iou-aware-header` (branched from `feature/pillar1-structural-reparameterization`)  
**Status:** Approved  

---

## 1. Problem Statement & Motivation

In anchor-free 3D LiDAR object detection in Bird's Eye View (BEV) (such as PIXOR, PointPillars, and MobilePixorNeXt), Non-Maximum Suppression (NMS) during post-processing filters proposals strictly using classification confidence $P_{\text{cls}}$.

### The "Score-IoU Misalignment" Phenomenon
1. **Edge Points vs. Center Geometry**: Anchor cells located near the visible boundary of a vehicle often receive dense LiDAR reflections, causing the classification head to output very high confidence ($P_{\text{cls}} = 0.95$). However, because the anchor is on the boundary, its geometric regression has higher variance, leading to a suboptimal bounding box (actual rotated $\text{IoU} = 0.40$).
2. **High-Quality Center Proposals Suppressed**: Conversely, an anchor near the vehicle's geometric center may produce an exceptionally precise 3D bounding box ($\text{IoU} = 0.88$), but have slightly lower classification confidence ($P_{\text{cls}} = 0.82$).
3. **NMS Failure**: Standard NMS selects the $0.95$ score proposal and suppresses the $0.88$ proposal. This severely degrades detection accuracy at strict evaluation thresholds (e.g. KITTI 3D / BEV moderate & hard benchmarks @ $\text{IoU} \ge 0.70$).

### The Solution: Pillar 4
Add a lightweight 5th prediction branch (`iou_head`) to predict the expected IoU quality $S_{\text{iou}} \in [0, 1]$ of the predicted bounding box, and calibrate candidate ranking during NMS using:
$$S_{\text{final}} = P_{\text{cls}}^{1 - \alpha} \cdot S_{\text{iou}}^{\alpha}$$
- $\alpha = 0.0$: Reverts 100% to standard classification-only NMS.
- $\alpha = 0.5$ (Default): Balanced geometric mean combining semantic class confidence and spatial box quality.

---

## 2. Model Architecture: Head Extension

### 2.1. Class `Header` in `detector/core/models/heads/cnn.py`
Add optional parameter `use_iou: bool = False`:
- When `use_iou=True`:
  - Instantiates `self.iou = Head(in_channels, 1, use_bn=use_bn, act=act)`.
  - Architecture: $\text{Conv}_{3\times 3} \to \text{BN} \to \text{SiLU} \to \text{Conv}_{3\times 3} \to \text{BN} \to \text{SiLU} \to \text{Conv}_{1\times 1}$ (1 channel output).
  - Forward: returns dictionary `{"cls": ..., "offset": ..., "size": ..., "yaw": ..., "iou": ...}` where `iou` has shape $[B, 1, H, W]$.
- When `use_iou=False` (default):
  - No `self.iou` instantiated (zero parameter overhead).
  - Output dictionary contains strictly `{"cls", "offset", "size", "yaw"}` (100% backward compatible).

### 2.2. Class `CustomModel` in `detector/core/models/model.py`
Reads `cfg.get("header_use_iou", False)` and threads it into `Header`.

---

## 3. Training-Time Target Generation: `iou_targets.py`

Create `detector/core/losses/iou_targets.py` to compute dynamic supervision target $y_{\text{iou}} \in [0, 1]$ for all positive anchors (`reg_mask == 1`).

### 3.1. Fundamental Principle: Gradient Isolation via `.detach()`
The predicted bounding box $\widehat{B} = (\text{offset}, \text{size}, \text{yaw})$ **must be detached** before computing $y_{\text{iou}}$:
$$\widehat{B}_{\text{detached}} = \widehat{B}.\text{detach}()$$
*Rationale:* The IoU target evaluates current box fidelity. If gradients flowed from $y_{\text{iou}}$ back into the regression heads, the network would try to manipulate box coordinates to artificially game the target value, leading to severe regression training instability.

### 3.2. Supported Target Generation Methods
Both methods operate entirely as **pure PyTorch GPU tensor operations** with zero external C++/CUDA dependencies:

#### Method 1: Option C – Projection MGIoU (`method="mgiou"`, Recommended Default)
1. Decode corners and local axes:
   - `pred_c, pred_ax = box_corners(pred_offset.detach(), pred_size.detach(), pred_yaw.detach())`
   - `tgt_c, tgt_ax = box_corners(target_offset, target_size, target_yaw)`
2. Project corners onto 4 normal axes (2 from pred, 2 from target) and compute 1D GIoU on each axis using `multiaxis_projection_giou`:
   $$\text{GIoU}_k = \frac{\text{Inter}_k}{\text{Union}_k} - \frac{\text{Hull}_k - \text{Union}_k}{\text{Hull}_k} \in [-1, 1]$$
3. Compute mean projection GIoU and clamp to valid probability range:
   $$y_{\text{iou}} = \text{clamp}\left(\frac{1}{4} \sum_{k=1}^4 \text{GIoU}_k, 0.0, 1.0\right)$$

#### Method 2: Option B – Yaw-Modulated Footprint IoU (`method="yaw_footprint"`, Ablation)
1. Compute axis-aligned bounding box footprint IoU from centers and dimensions:
   $$\text{Inter} = (\min(p_{\max}, t_{\max}) - \max(p_{\min}, t_{\min}))_x^+ \cdot (\min(p_{\max}, t_{\max}) - \max(p_{\min}, t_{\min}))_y^+$$
   $$\text{IoU}_{\text{footprint}} = \frac{\text{Inter}}{\text{Area}_p + \text{Area}_t - \text{Inter} + \epsilon}$$
2. Compute normalized $\pi$-periodic doubled-yaw cosine agreement:
   $$\mathbf{u}_{\text{pred}} = \frac{\text{pred\_yaw}.\text{detach}()}{\|\text{pred\_yaw}.\text{detach}()\|}, \quad \mathbf{u}_{\text{tgt}} = \frac{\text{target\_yaw}}{\|\text{target\_yaw}\|}$$
   $$\text{Yaw\_Factor} = \text{clamp}\left(\frac{1 + \mathbf{u}_{\text{pred}} \cdot \mathbf{u}_{\text{tgt}}}{2}, 0.0, 1.0\right)$$
3. Modulate:
   $$y_{\text{iou}} = \text{IoU}_{\text{footprint}} \times \text{Yaw\_Factor} \in [0, 1]$$

---

## 4. Loss Function & Multi-Task Uncertainty Integration

### 4.1. Loss Formulation
The loss for the IoU branch is Binary Cross-Entropy with Logits, supervised exclusively on positive anchors:
$$\mathcal{L}_{\text{iou}} = \frac{1}{\sum \text{reg\_mask}} \sum_{i, j \in \mathcal{P}_{\text{pos}}} \text{BCEWithLogits}(\text{pred\_logits}_{\text{iou}}(i, j), y_{\text{iou}}(i, j))$$
If $\sum \text{reg\_mask} == 0$ (empty positive mask), return graph-connected zero:
$$\mathcal{L}_{\text{iou}} = 0.0 \times \text{pred\_logits}_{\text{iou}}.\text{sum}()$$

### 4.2. Strategy Integration in `OgaLossStrategy`
1. Read configuration flags:
   - `use_iou = bool(config.get("use_iou", False))`
   - `iou_target_type = str(config.get("iou_target_type", "mgiou"))`
   - `iou_loss_weight = float(config.get("iou_loss_weight", 1.0))`
2. Dynamic Task Allocation for `TemperatureSoftmaxUncertainty`:
   - If `use_iou=True`: `tasks = ("cls", "offset", "size", "yaw", "geo", "iou")`
   - If `use_iou=False`: `tasks = ("cls", "offset", "size", "yaw", "geo")`
3. Loss Output Dictionary:
   Includes `"iou": iou_loss.detach()`, `"weight_iou": w_iou`, and `"mean_iou_target": y_iou.mean().detach()`.

---

## 5. Post-Processing & Joint Quality NMS

### 5.1. Modification in `detector/postprocess.py` (`filter_pred`)
1. Check if `"iou"` exists in `pred`:
   - If absent or `nms_alpha == 0.0`:
     $$S_{\text{final}} = P_{\text{cls}}$$
     (Identical to current baseline behavior).
   - If present and `nms_alpha > 0.0`:
     $$S_{\text{iou}} = \sigma(\text{pred["iou"].squeeze(0)})$$
     $$S_{\text{final}} = (P_{\text{cls}})^{1 - \alpha} \cdot (S_{\text{iou}})^{\alpha}$$
2. Candidate selection and Rotated NMS:
   - Max-pooling and score thresholding use $S_{\text{final}}$:
     $$\text{candidate\_mask} = (S_{\text{final}} == \text{pooled}) \land (S_{\text{final}} > \text{threshold})$$
   - Candidate ranking in NMS sorts by `candidate_scores = S_final[candidate_mask]`.

---

## 6. Configuration Management

### 6.1. Full Pillar 4 Configuration
`configs/kitti/mobilebev/kitti_mobilepixornext_litemla_oga_reparam_iqa.json`:
```json
{
  "header_use_iou": true,
  "use_iou": true,
  "iou_target_type": "mgiou",
  "iou_loss_weight": 1.0,
  "nms_alpha": 0.5
}
```

### 6.2. Baseline Fallback / Standard IoU Behavior
If a user wants standard behavior (no IoU head, standard NMS):
```json
{
  "header_use_iou": false,
  "use_iou": false,
  "nms_alpha": 0.0
}
```
Setting `"header_use_iou": false` (or omitting it) disables the 5th head completely. Setting `"nms_alpha": 0.0` uses pure $P_{\text{cls}}$ for NMS even if an IoU head exists.

---

## 7. Testing & Verification Protocol

### Test File: `tests/test_iqa_header.py`
1. **Head Instantiation & Shape Test**:
   - `use_iou=False` yields 4 head keys.
   - `use_iou=True` yields 5 keys with `pred["iou"]` shape $[B, 1, H, W]$.
2. **Target Generator Fidelity Tests (`mgiou` & `yaw_footprint`)**:
   - Perfect match box ($\text{pred} == \text{target}$) yields $y_{\text{iou}} \approx 1.0$.
   - Orthogonal box ($90^\circ$ rotation) yields $y_{\text{iou}} \approx 0.0$.
   - Distant box ($>50\text{m}$ offset) yields $y_{\text{iou}} = 0.0$.
   - Invariance to batch size and device.
3. **Gradient Isolation Test**:
   - Backward on $\mathcal{L}_{\text{iou}}$ produces valid non-zero gradients on `header.iou.parameters()`.
   - Produces **None** gradients on `header.offset`, `header.size`, and `header.yaw`.
4. **NMS Score Reordering Test**:
   - Case where candidate A has $P_{\text{cls}}=0.95, S_{\text{iou}}=0.30 \implies S_{\text{final}}=0.534$.
   - Case where candidate B has $P_{\text{cls}}=0.80, S_{\text{iou}}=0.90 \implies S_{\text{final}}=0.848$.
   - Verifies NMS selects candidate B and suppresses candidate A.
5. **Full Regression Test**:
   - Run complete suite (`pytest`) to verify all 87 existing tests pass cleanly without regression.

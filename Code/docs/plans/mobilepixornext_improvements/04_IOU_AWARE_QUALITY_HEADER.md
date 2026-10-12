# Pillar 4: IoU-Aware Quality Detection Header (IQA-Header)

## 1. Motivation & Problem Statement

In conventional anchor-free 3D LiDAR detectors (such as PIXOR, PointPillars, and baseline **MobilePixorNeXt**), the classification branch and bounding box regression branches are trained independently:
- **Classification Branch**: Predicts class probability $P_{\text{cls}} \in [0, 1]$.
- **Regression Branches**: Predicts spatial offsets $(dx, dy)$, physical dimensions $(\log l, \log w)$, and heading angle $(\cos \theta, \sin \theta)$.

During post-processing, Non-Maximum Suppression (NMS) sorts candidate proposals **purely based on classification confidence $P_{\text{cls}}$**.

### The "Score-IoU Misalignment" Phenomenon

This decoupling leads to severe performance degradation at high IoU thresholds (such as KITTI $AP_{3D} \ge 0.70$):
1. **False High-Confidence Outliers**: An anchor located near the physical edge of a vehicle may have dense point reflections, yielding high classification confidence ($P_{\text{cls}} = 0.94$). However, its partial view causes its predicted 3D bounding box to be misaligned (actual 3D $\text{IoU} = 0.42$).
2. **Accurate Low-Confidence Suppressions**: Another anchor closer to the geometric center may produce an exceptionally precise 3D bounding box ($\text{IoU} = 0.88$), but have slightly lower classification confidence ($P_{\text{cls}} = 0.86$).
3. **NMS Degradation**: The NMS algorithm picks the $0.94$ score proposal and suppresses the $0.88$ proposal. This results in poor high-IoU true positives and lower mAP scores.

```
       [Standard NMS: Driven strictly by P_cls]
       Candidate A: P_cls = 0.94, IoU = 0.42  ──> Selected (Sloppy Box!)
       Candidate B: P_cls = 0.86, IoU = 0.88  ──> SUPPRESSED (High quality lost)

       [IoU-Aware NMS: Calibrated Joint Score S_final = P_cls^0.5 * S_iou^0.5]
       Candidate A: S_final = sqrt(0.94 * 0.42) = 0.628
       Candidate B: S_final = sqrt(0.86 * 0.88) = 0.870  ──> Selected (Accurate Box!)
```

**Solution**: **IoU-Aware Quality Detection Header (IQA-Header)**:
Add a 5th specialized prediction branch that estimates the expected 3D IoU ($S_{\text{iou}} \in [0, 1]$) of the predicted bounding box, and fuse it with classification probability during NMS ranking.

---

## 2. Mathematical Formulation & Architecture

### 2.1. Decoupled 5-Task Header Architecture

The neck outputs a shared 16-channel feature map at stride 4 ($B \times 16 \times 200 \times 176$). The detection header consists of 5 parallel task heads:

```
                                  Shared Neck Features (16 x 200 x 176)
                                                   │
    ┌──────────────┬──────────────┬────────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼                ▼              ▼              ▼
1. Class Head  2. Offset Head 3. Size Head    4. Yaw Head   5. IQA Head    (Optional Centerness)
 (3 channels)   (2 channels)   (2 channels)   (2 channels)   (1 channel)
  Sigmoid / CE     Linear         Linear         Linear        Sigmoid
```

Each head follows the lightweight modern standard:
$$\text{Head}(X) = \text{Conv}_{1\times 1}(\text{SiLU}(\text{BN}(\text{Conv}_{3\times 3}(X))))$$

### 2.2. Training Supervision for IoU Branch

Let $(i, j)$ denote a grid cell on the BEV feature map. The IoU branch outputs:
$$S_{\text{iou}}(i, j) = \sigma(\text{Head}_{\text{iou}}(X)_{i, j}) \in [0, 1]$$

#### Ground Truth IoU Target:
For each positive anchor cell $(i, j) \in \mathcal{P}_{\text{pos}}$:
1. Decode the predicted 3D bounding box $\widehat{B}_{i, j}$ using current regression outputs:
   $$\widehat{B}_{i, j} = \text{DecodeBox}(\text{Offset}_{i, j}, \text{Size}_{i, j}, \text{Yaw}_{i, j})$$
2. Compute the exact rotated 3D IoU against the assigned ground-truth bounding box $B^*_{i, j}$:
   $$y_{\text{iou}}(i, j) = \text{Rotated-3D-IoU}(\widehat{B}_{i, j}.\text{detach}(), B^*_{i, j})$$
   *(Crucial: $\widehat{B}_{i, j}$ is detached to avoid circular regression gradient instability).*

#### Loss Formulation:
The IoU branch is supervised exclusively on positive anchors using Binary Cross-Entropy (or Smooth $L_1$):
$$\mathcal{L}_{\text{iou}} = \frac{1}{|\mathcal{P}_{\text{pos}}|} \sum_{(i, j) \in \mathcal{P}_{\text{pos}}} \text{BCE}(S_{\text{iou}}(i, j), y_{\text{iou}}(i, j))$$

Where:
$$\text{BCE}(S, y) = - [y \log(S) + (1 - y) \log(1 - S)]$$

### 2.3. Uncertainty-Weighted Loss Integration

The new $\mathcal{L}_{\text{iou}}$ loss is integrated into the homoscedastic uncertainty weighting framework (`TemperatureSoftmaxUncertainty`):
$$\mathcal{L}_{\text{total}} = \sum_{k \in \{\text{cls}, \text{reg}, \text{heading}, \text{corner}, \text{iou}\}} \frac{1}{2 \sigma_k^2} \mathcal{L}_k + \log \sigma_k$$

---

## 3. Joint Quality-Aware Post-Processing & NMS

### 3.1. Calibrated Ranking Score

During inference, candidate bounding boxes are ranked using the composite score $S_{\text{final}}$:
$$S_{\text{final}} = P_{\text{cls}}^{1 - \alpha} \cdot S_{\text{iou}}^{\alpha}$$

Where $\alpha \in [0, 1]$ is a tunable hyperparameter:
- $\alpha = 0.0$: Standard classification-only ranking (Baseline).
- $\alpha = 0.5$: Balanced geometric mean of class confidence and spatial accuracy (Recommended).
- $\alpha = 1.0$: Pure IoU-driven ranking.

### 3.2. Fast Vectorized NMS Implementation

```python
def quality_aware_nms(
    boxes: torch.Tensor,       # (N, 7) [x, y, z, dx, dy, dz, heading]
    scores_cls: torch.Tensor,  # (N,) classification confidence
    scores_iou: torch.Tensor,  # (N,) predicted IoU quality
    alpha: float = 0.5,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.1
) -> torch.Tensor:
    # 1. Compute joint score
    joint_scores = (scores_cls ** (1.0 - alpha)) * (scores_iou ** alpha)
    
    # 2. Threshold filtering
    mask = joint_scores > score_threshold
    boxes, joint_scores = boxes[mask], joint_scores[mask]
    
    # 3. Standard Rotated NMS on joint_scores
    keep_indices = rotated_nms(boxes, joint_scores, iou_threshold)
    return keep_indices
```

---

## 4. Implementation Blueprint

### 4.1. File Locations
- Modify: `detector/core/models/heads/header.py` (Add `iou_head` and `has_iou_branch` flag).
- Modify: `detector/core/losses/oriented_geometric_loss.py` (Compute $\mathcal{L}_{\text{iou}}$ and add to `loss_dict`).
- Modify: `detector/core/postprocessing/eval_nms.py` (Incorporate joint score calibration).

### 4.2. Header Configuration Interface

```json
{
  "header_tasks": ["cls", "offset", "size", "heading", "iou"],
  "iou_loss_weight": 1.0,
  "nms_alpha": 0.5
}
```

---

## 5. Verification & Testing Protocol

### Test File: `tests/test_iqa_header.py`

1. **Forward Output Shape**:
   Verify `Header` with `has_iou_branch=True` outputs `preds["iou"]` with shape $(B, 1, 200, 176)$.
2. **IoU Loss Gradient Verification**:
   Verify that $\mathcal{L}_{\text{iou}}$ computes valid gradients to `iou_head` parameters without propagating gradients to the regression heads (due to `.detach()`).
3. **NMS Score Reordering Test**:
   Construct artificial cases where candidate $A$ has high $P_{\text{cls}}$ but low $S_{\text{iou}}$, and candidate $B$ has moderate $P_{\text{cls}}$ but high $S_{\text{iou}}$. Verify that `joint_scores` correctly inverts ranking in favor of candidate $B$.
4. **Latency Verification**:
   Added parameter cost: $\approx 2.5\text{k}$ parameters. Latency overhead: $\le 0.08\text{ ms}$.

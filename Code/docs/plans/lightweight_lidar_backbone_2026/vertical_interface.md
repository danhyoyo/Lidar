# Vertical prediction and target interface

Implemented in batches18–20, tasks53–58. Heads, owner-consistent targets, weighted loss, calibrated decoding and reference AP3D/APBEV are implemented. Dedicated 3D CUDA/AMP/worker smoke and main/ECA runtime presets are verified in task58; no trained KITTI AP is reported.

## Configuration and construction

The authoritative field is `data.box_mode`, accepting `bev` (default) or `3d`. Do not duplicate it in `model`. The full pipeline passes the resolved value into `CustomModel(..., box_mode=...)`. Direct `Header` and `GroupedHeader` construction also accept `box_mode`, defaulting to BEV.

```python
candidate["data"]["box_mode"] = "3d"
candidate["loss"]["name"] = "oga"  # or baseline
candidate["loss"]["vertical_loss_weight"] = 1.0  # required finite positive coefficient
model = build_model(candidate)
```

Gaussian and binary prediction/target construction support both grouped and single-head topology. Full 3D config validation currently permits only baseline/OGA; UWAG, GW-QAL and Q-OGA extensions are rejected. The full pipeline requires an explicit finite positive vertical weight. BEV IQA remains a one-channel BEV quality branch; it is not 3D IoU.

For the grouped candidate, predictions include `groups.car.vertical` and `groups.ped_cyc.vertical`, each shaped `[B,2,H/4,W/4]`. Channel0 is **LiDAR bottom-center z in meters**; channel1 is **log(height in meters)**. These are separate task branches using the existing two-convolution `Head`; offset/size/yaw/classification/IQA branches and widths are unchanged. Single-head 3D uses the same `vertical` key in its flat prediction mapping.

BEV mode adds no vertical parameters or output key, preserves existing state keys and seeded numerical behavior, and retains historical default configuration behavior. Strict state loading rejects BEV/3D head mismatches. The existing checkpoint identity and candidate run-name digest distinguish `box_mode`; no new checkpoint format is introduced.

## Targets and ownership

Processed labels remain `[class,height,width,length,x,y,z_bottom,yaw]`. `Dataset` reads `box_mode` from its data config and adds `[2,H/4,W/4]` float32 `vertical` targets in 3D mode; DataLoader collation produces `[B,2,H/4,W/4]`. Targets share the existing `reg_mask`.

The explicit low-level API is:

```python
offset, size, yaw, mask, vertical = fill_regression_targets_3d(
    boxes, radii, output_shape, geometry, out_size_factor,
    backend="python",  # or numba
    assignment="nearest_center",  # or legacy
)
```

Internal maps use `[X,Y,C]` and are transposed to `[C,Y,X]` by Dataset, matching BEV. The legacy `fill_regression_targets_python/numba` APIs still return exactly **four arrays**. The 3D API returns those same four maps plus a two-channel vertical array.

Both backends execute the same ownership kernel. It writes vertical values inside the existing accepted BEV assignment, after canonical ordering, reserved quantized peaks, overlap comparisons and tie-breaking. There is no second owner calculation. In nearest-center mode, exact ties retain the first canonical box; legacy mode preserves last-box-wins behavior. Global class IDs remain available to canonical ordering even when local classification channels are reordered. Separate groups can retain different boxes at one cell; within a group, one regression branch retains one owner per cell.

Unsupervised vertical cells and empty groups are zero. The new 3D API rejects non-finite boxes, nonpositive dimensions, mismatched/non-finite/negative radii and unsupported backends. An explicit Numba request fails if unavailable instead of falling back. Existing BEV validation/formulas remain unchanged.

## Weighted objective and consumers

Inside each baseline/OGA criterion, the vertical term is FP32 Smooth L1 with beta1, averaged across both channels of all owned cells. Owned values are selected before arithmetic, so non-finite unsupervised cells cannot contaminate the reduction. A finite binary mask, aligned shapes/devices and finite supervised values are required. An empty mask returns a connected zero. The fixed weighted term is added after the BEV strategy objective, before normalized group aggregation; OGA gains no additional uncertainty task or state parameter. Grouped inputs are validated before any adaptive state update. The coefficient participates in existing strict checkpoint identity and resume.

`filter_pred_3d` returns complete float32 Nx9 rows and preserves vertical/quality association through classwise BEV-footprint NMS and one global cap. The [reference guide](reference_evaluation.md) documents calibrated conversion, genuine projected image boxes and AP3D/APBEV R11/R40. Direct BEV loss/decoder, the local BEV evaluator and ONNX/common deployment guards remain explicit. Select the complete 3D path rather than strip vertical tensors. BEV Nx7 rows cannot be reinterpreted as complete 3D boxes.

## Counts and evidence

For the default width32, BN/SiLU head, each vertical branch adds **18,626 parameters**. Two independent branches add **37,252**. Main focal grouped IQA heads now have **223,413** parameters, detector **883,941**, backbone including neck **660,528**. Counts exclude the train-only criterion; grouped 3D OGA+IQA still has 12 parameters, with no extra vertical adaptive parameter.

Tests: [vertical head](../../../tests/test_vertical_head.py), [vertical targets](../../../tests/test_vertical_targets.py), [vertical loss](../../../tests/test_vertical_loss.py), [3D decode](../../../tests/test_kitti_3d_decode.py) and [reference evaluator](../../../tests/test_kitti_3d_evaluation.py). Gates cover construction/config immutability, widths, group/task gradient isolation, seeded BEV parity, strict state roundtrip and semantic identities; targets cover Python/Numba, unequal XY resolutions, every supervised cell's BEV/vertical owner, exact same-cell ties, small-radius reserved peaks, local class reordering, empty groups, invalid inputs and actual Dataset/DataLoader collation. Runtime BEV recipes and notebook defaults stay unchanged.

See [execution log](execution_progress.md) for actual results and [Milestone A verification](release_a_verification.md) for the verified BEV scope. CPU optimizer/strict training resume and real reference CUDA parity are verified in batch19. Dedicated 3D CUDA/AMP/worker training-resume-decode smoke is verified in task58; see [3D verification](release_b_verification.md). No measured trained KITTI AP or SOTA claim is made.

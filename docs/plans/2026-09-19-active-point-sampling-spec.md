# Active Point Sampling for MobileBEV A4 — Specification

**Status:** Closed — rejected after controller-dev and final validation
**Date:** 2026-09-19; closed 2026-09-20
**Baseline:** MobileBEV-Lite A4 (`RichBEV-8 + SG-FPN + Center3D`), checkpoint reported by the owner as epoch 50

## Final Outcome

Static sampling failed the registered S0 gate, and every label-informed oracle
point failed the safety guardrails. No contextual policy or RL implementation is
authorized.

Controller-dev results:

| Rate | 3D mAP Moderate | mAP drop | Preprocess gain | E2E gain | S0 |
|---:|---:|---:|---:|---:|---|
| 1.00 | 78.0439 | 0.0000 | 0.00% | 0.00% | Dense control |
| 0.75 | 76.8960 | 1.1479 | 12.80% | 9.63% | Fail |
| 0.50 | 69.8734 | 8.1704 | 23.27% | 15.67% | Fail |
| 0.25 | 44.6093 | 33.4345 | 37.93% | 24.58% | Fail |

The best oracle mAP was 73.8533 at mean rate 0.5196, about 4.19 points
below the dense controller-dev result. All 88 unique oracle action vectors
failed the accuracy safety guardrails.

After freezing rate 0.75 as the only plausible deployment candidate, the
sealed validation split was consumed once with three timing repeats per rate:

| Variant | 3D mAP Moderate | E2E mean | FPS | Decision |
|---|---:|---:|---:|---|
| Dense A4 | 69.9429 | 52.0123 ms | 19.2262 | Retain |
| Static 0.75 | 66.9301 | 46.8510 ms | 21.3443 | Reject as default |

Rate 0.75 improved E2E latency by 9.92% but lost 3.0129 mAP, including
2.9360 Car AP and 4.4312 Cyclist AP. It failed the pre-registered deployment
gate. Dense A4 remains the default; rate 0.75 may only be described as an
explicit degraded/fast mode, not as a successful research result.

The dense smoke runs produced identical accuracy. Their one-pass latency
comparison was biased by run order/cache state and exceeded the 5% timing
equivalence bound, so it is not used as evidence for the final latency claim.
The final dense/rate-0.75 comparison instead uses the same evaluator, rotated
run order, and median-of-three timing.

Committed summary artifacts:

- `results/research/active_sampling/gate_report.json`
- `results/research/active_sampling/final_comparison.json`
- `results/research/active_sampling/baseline_a4_epoch50.json`

## 1. Decision

Investigate adaptive point sampling before any spatial model routing. The measured A4 pipeline spends more time in preprocessing than in the model, so sampling raw points before `RichBEV-8` encoding has the clearest end-to-end opportunity.

Implementation is gated:

1. Measure deterministic static rates.
2. Measure a label-informed dynamic oracle.
3. Build a deployable contextual policy only if the oracle materially beats the best static rate.

PPO, backbone changes, adaptive voxel sizes, spatial tiling, and A4 retraining are out of scope for the first implementation. With four actions and offline evaluation of every action, PPO adds complexity without adding information.

## 2. Locked Baseline

Source artifact: `/home/duyennh/Downloads/Eval_pro1/evaluation_a4.json`.

| Item | Locked value |
|---|---:|
| Evaluation frames | 1,497 |
| Source commit | `f881567ca859574792446642dc80a2cab5fef710` (clean) |
| Config SHA-256 | `d4876f09a41e84f0353d058ba79c5f0fbf3210ed18bda47962492abd2df99630` |
| Validation split SHA-256 | `cae7f7b5a72f28af771ebe878faa97b81bdafbcecd2be51205fd8cb5ecd2733a` |
| Checkpoint SHA-256 | `e3ff54229ea2feaa30f8f5a88da6555835fbcfa099d02ac49d9400ea510d222d` |
| Runtime | PyTorch 2.11, FP32, NVIDIA L4 |
| Parameters | 593,075 |
| 3D mAP Moderate | 69.9508% |
| Preprocess mean | 25.2491 ms |
| Model mean | 10.2856 ms |
| End-to-end mean | 47.6528 ms |

The JSON does not encode the checkpoint epoch. Before a final research claim, the experiment manifest must record `checkpoint_epoch: 50` or link to the checkpoint-selection artifact that proves it.

Baseline 3D AP R40 Moderate:

| Class | AP | Max recall |
|---|---:|---:|
| Car | 82.0814% | 0.9062 |
| Pedestrian | 56.8348% | 0.7326 |
| Cyclist | 70.9363% | 0.8287 |

Pedestrian is the primary safety guardrail. Its Moderate recall is 0.4845 at 30–50 m and 0.2000 at 50–70.4 m in the baseline, so an aggregate mAP-only decision is not acceptable.

## 3. Research Claims

The first implementation may support only these claims:

- Reducing raw points can reduce `np.fromfile -> sampling -> RichBEV encoding` preprocessing and total input-to-detections latency while keeping the frozen A4 model and fixed BEV tensor shape.
- A scene-conditioned policy is useful only if it improves the measured accuracy/latency Pareto frontier over every fixed sampling rate.

It must not claim that H2D or model latency scales with point count. Their tensor shape remains `(1, 8, 800, 704)`.

## 4. Scope

### In scope

- Sampling rates `{1.00, 0.75, 0.50, 0.25}` before `encode_bev`.
- Frozen A4 weights and existing KITTI AP R40 evaluator.
- 3D and BEV AP, per-class/distance recall, and stage/E2E latency.
- An offline full-information contextual bandit only after the static and oracle gates pass.

### Out of scope

- PPO or sequential MDP training.
- Using a previous-frame detection count: KITTI samples are not a guaranteed temporal stream in this pipeline.
- Ego velocity: it is unavailable in the current dataset interface.
- Changing `RichBEV-8`, correcting its density channel, retraining A4, sparse CUDA kernels, or spatial feature-map routing.

Density correction may be a separately registered ablation only if the static experiment shows that channel 7 distribution shift, rather than geometric information loss, is the dominant failure. It must not be silently combined with the initial sweep.

## 5. Deterministic Sampling Contract

For frame identifier `frame_id`, global seed `42`, point count `N`, and rate `r`:

1. At `r = 1.00`, return the original point array unchanged.
2. Derive a stable 64-bit seed from the first eight bytes of `SHA-256("42:" + frame_id)`; Python's randomized `hash()` is forbidden.
3. Generate one permutation of `[0, N)` per frame. Each rate uses the first `floor(N * r)` indices, sorted before indexing so source order is retained.
4. Use at least one point for any non-empty input. Preserve an empty input as empty.
5. Apply sampling to the raw `(N, 4)` array before ROI filtering and `encode_bev`.

This makes lower-rate samples nested inside higher-rate samples, repeatable across processes, and free of LiDAR file-order aliasing. Strided slicing is excluded.

The dense control must produce byte-identical BEV input and identical predictions to the existing evaluator.

## 6. Data and Leakage Contract

The 1,497-frame validation split is sealed until the experiment and policy are frozen.

Create policy-train and controller-dev manifests from the 5,984 current training IDs:

- Sort IDs by `SHA-256("42:" + frame_id)`.
- First 1,200 IDs: controller-dev.
- Remaining 4,784 IDs: policy-train.

Use controller-dev for the static sweep, oracle analysis, feature choice, reward weights, and go/no-go decisions. If a policy is authorized, train it only on policy-train and select it only on controller-dev. Run the sealed validation split once for the final comparison; further tuning after seeing it requires a new declared iteration and cannot reuse it as an untouched final result.

## 7. Measurement Contract

Keep the current primary latency boundary:

> Sequential offline binary load + preprocessing + H2D + model + decode/NMS; excludes ROS, tracking, and HMI.

Report these stages separately:

- point-file load;
- sampling;
- `RichBEV-8` encoding;
- total preprocess, compatible with the current baseline;
- H2D;
- model;
- decode/NMS;
- input-to-detections.

Accuracy is deterministic and needs one prediction set per rate. For latency, use 10 warmup frames, three complete repeats per rate on the same NVIDIA L4 software environment, and report the median of the three run means plus p50/p95/p99. Rotate rate order between repeats. Context extraction and policy inference time count inside preprocessing.

Before using sampled results, run the unmodified and modified dense evaluators on
the same first 200 frames of `splits/kitti/train.txt`. The modified dense control
must satisfy:

- byte-identical predictions and identical AP values to the unmodified run;
- preprocess and end-to-end mean within 5% of the unmodified run under the same environment.

If the control fails, no sampled result is valid until the discrepancy is explained.
The locked 1,497-frame baseline is compared again only in the final sealed-validation run.

## 8. Phase S0 — Static Feasibility Gate

Evaluate all four fixed rates on controller-dev. Report:

- 3D/BEV AP R40 for Easy, Moderate, and Hard;
- per-class Moderate AP and max recall;
- Pedestrian and Cyclist Moderate AP/recall for 0–30 m, 30–50 m, and 50–70.4 m;
- point counts, selected rate, all latency stages, and predictions per frame;
- accuracy deltas relative to the dense controller-dev control.

Static feasibility passes if at least one non-dense rate meets all applicable conditions:

- 3D mAP Moderate drop `<= 0.8` percentage points;
- each class's 3D AP Moderate drop `<= 1.5` points;
- each class's Moderate max-recall drop `<= 0.03` absolute;
- for a distance band with at least 30 ground-truth objects, Pedestrian/Cyclist max-recall drop `<= 0.05` absolute;
- mean preprocessing improves by at least 35%;
- mean input-to-detections improves by at least 15%.

Bands with fewer than 30 ground-truth objects are reported with TP/FN counts and are diagnostic, not hard pass/fail metrics.

If no rate passes, stop. Report the negative result; do not implement a policy.

## 9. Dynamic Oracle Gate

For every controller-dev frame, cache predictions for all four rates. Compute a label-informed per-frame quality score from class-aware matching at the evaluator's IoU thresholds:

`quality = 2 * TP / (2 * TP + FP + FN)`

Macro-average over Car, Pedestrian, and Cyclist; a class with no GT and no prediction scores 1, while a class with predictions but no GT scores 0. Sweep a sampling-cost coefficient and select the rate maximizing:

`utility(frame, rate) = quality(frame, rate) - beta * rate`

This oracle uses ground truth and is not deployable. Its selected actions are evaluated with the normal aggregate AP evaluator.

Policy work is authorized only if an oracle point satisfying all S0 safety guardrails does either of the following relative to the best static point:

- improves 3D mAP Moderate by at least `0.3` points at a mean sampling rate within `2.5` percentage points; or
- lowers mean sampling rate by at least `10` percentage points while keeping 3D mAP Moderate within `0.2` points.

If neither condition holds, the best static rate is the final implementation recommendation.

## 10. Phase S1 — Conditional Contextual Policy

If the oracle gate passes, use only current-frame, label-free context:

1. `log1p(N)`;
2. mean `z`;
3. standard deviation of `z`;
4. fraction of points within 15 m;
5. fraction of points beyond 40 m;
6. fraction of points inside the configured XYZ ROI.

The extractor must be one vectorized pass and its latency is included in preprocessing. The first policy is the smallest model that works: predict the four cached counterfactual utilities and choose `argmax`. Start with a linear PyTorch layer; add one hidden layer only if controller-dev results show the linear model fails the oracle-headroom gate.

This is an offline, full-information contextual bandit. No online exploration is required because every action's outcome is available during training.

The policy passes only if it:

- satisfies all S0 safety guardrails on controller-dev;
- improves over every static rate by one of the oracle-gate margins;
- keeps context extraction + policy inference below `0.3 ms` mean;
- is deterministic for a fixed frame and checkpoint.

## 11. Final Acceptance

After freezing sampling, features, reward coefficient, and policy checkpoint, run dense, best-static, and policy variants on the sealed 1,497-frame validation split.

The research direction is successful only if the policy:

- keeps 3D mAP Moderate within `0.8` points of dense A4;
- satisfies the class and recall guardrails from S0;
- reduces mean input-to-detections latency by at least 20%;
- materially beats the best static rate by an oracle-gate margin.

If the policy fails but a static rate passes S0, ship/report the static result. If both fail, retain dense A4 and close this direction without adding RL code.

## 12. Required Artifacts

- immutable baseline manifest including epoch provenance;
- generated policy-train and controller-dev split files plus SHA-256 hashes;
- one JSON result per rate and repeat;
- per-frame action/prediction cache for oracle analysis;
- oracle Pareto summary;
- conditional policy checkpoint and training manifest;
- final comparison JSON containing dense, best-static, and policy results.

Every result must include source commit/dirty state, config/split/checkpoint hashes, runtime versions, device, seed, sampling contract version, and exact command.

# Robust Multi-Scale LiteMLA for MobilePixorNeXt (M2 / Pillar 2) Implementation Plan

**Goal:** Implement the isolated **Pillar 2 / M2 LiteMLA delta** on top of `feature/mobilepixornext-improvements`, then verify numerical stability, runtime cost, and detection impact with controlled ablations. The Pillar 1 branch is reference material only and must not be merged into this feature branch.

**Source specification:** `docs/plans/mobilepixornext_improvements/02_ROBUST_MULTISCALE_LITEMLA.md`

## Execution Snapshot

- Implemented on `feature/m2-robust-multiscale-litemla` from base commit `7d394d7`; Pillar 1 was not merged.
- Added configurable attention scales and QK normalization modes (`none`, `rmsnorm`, `layernorm`).
- Added M2 `(3, 5) + RMSNorm` and `(3, 5) + no norm` ablation configs.
- Full test suite: `89 passed`, `2 subtests passed` on PyTorch `2.12.0.dev20260408+cu128`.
- CUDA BF16 forward/backward and one real KITTI optimizer step completed with finite outputs, loss, and gradients.
- Full model parameters: single-scale reference `693,097`; M2 RMSNorm `709,545` (`+16,448`).
- Preliminary RTX 5060 Ti BF16 model latency: reference mean `8.2324 ms`; M2 mean `8.4647 ms`. This is not the roadmap's RTX 3060 reference platform.
- The normal trainer is currently blocked before model forward by a Windows native exception in the existing augmentation path at `transform.py:110` (`numpy.matmul`). A one-step training verification succeeded when augmentation was disabled in memory; committed experiment configs were not changed.
- Full 100-epoch ablation training and AP evaluation remain pending after the augmentation runtime issue is resolved.

**Roadmap relationship:**

```text
M1 = MobilePixorNeXt + OGA + structural reparameterization
M2 = M1 + multi-scale LiteMLA (3, 5) + QK-RMSNorm
```

This branch delivers only the Pillar 2 code so independent feature branches can be reviewed and merged later. Its isolated ablation must compare the current single-scale LiteMLA baseline with the new multi-scale/QK-normalized variant under the same data split, optimizer, schedule, seed, precision, augmentation, loss, and post-processing.

## 1. Repository Analysis

### Current implementation

- `LiteMLARefinement` already exists in `detector/core/models/backbones/mobilepixornext_blocks.py`.
- It already accepts a `scales` tuple and constructs one regional QKV aggregation branch per scale.
- It already disables autocast and performs the linear-attention reduction in FP32 before converting the result back to the input dtype.
- `MobilePixorNeXtBackbone` currently hardcodes `scales=(5,)`.
- Query and key currently use `ReLU` as the positive linear-attention feature map; QK normalization is not implemented.
- The registry and JSON configs do not expose attention scales or QK normalization.
- Existing tests cover shape, gradients, FP32 numerical sanity, parameter budget, and TorchScript, but do not cover dual-scale attention, QK normalization, or CUDA BF16 stress cases.

### Branch base and Pillar 1 reference

The working branch `feature/m2-robust-multiscale-litemla` was created from `feature/mobilepixornext-improvements` at commit `7d394d7`.

The Pillar 1 implementation exists on the following branch and may be inspected for repository conventions:

```text
origin/feature/pillar1-structural-reparameterization
```

Do not merge or copy Pillar 1 code into this branch. The later integration branch is responsible for composing Pillar 1 and Pillar 2 into the cumulative roadmap M2 model.

### Important evaluation limitation

The current repository evaluator reports local KITTI-style rotated **BEV AP R40**. It does not currently provide official KITTI `AP3D`. Therefore:

- Report BEV AP R40 and validation loss using the current pipeline.
- Do not label BEV AP as AP3D.
- If AP3D is a required deliverable, add or adopt a separate 3D evaluator as a follow-up task before claiming the roadmap's AP3D targets.

## 2. Design Decisions

### Reuse the existing attention class

Extend `LiteMLARefinement` rather than creating a parallel `ms_litemla.py` implementation. The existing class already contains the multi-scale branch machinery and FP32 accumulation. Reusing it avoids duplicate attention implementations and preserves existing state-dict keys when the new options are disabled.

### Preserve backward compatibility

Defaults must reproduce M0/M1 behavior:

```text
scales = (5,)
qk_norm = "none"
FP32 accumulation = enabled
```

Old configs that omit the new keys must continue to construct the existing single-scale LiteMLA.

### Normalize over the head dimension

Inside the attention core, query and key have logical shape:

```text
(batch, heads, head_dim, tokens)
```

QK normalization must operate over `head_dim`, not over the token dimension. Transpose to `(batch, heads, tokens, head_dim)` for `LayerNorm`/`RMSNorm`, then transpose back.

For the linear-attention denominator to remain non-negative, apply the positive feature map after normalization:

```text
Q = ReLU(Norm(Q))
K = ReLU(Norm(K))
```

Run the split, normalization, matrix products, and denominator computation inside the existing FP32 autocast-disabled section.

### Make ablation settings configurable

Expose these model config keys:

```json
{
  "c4_attention": "litemla",
  "c4_attention_scales": [3, 5],
  "c4_attention_qk_norm": "rmsnorm"
}
```

Supported normalization values:

- `"none"`: M1 baseline and raw-attention ablation.
- `"rmsnorm"`: target M2 configuration.
- `"layernorm"`: optional sensitivity experiment defined by the ablation protocol.

Reject unsupported values early with a clear `ValueError`.

## 3. Files Expected to Change

| File | Planned change |
|---|---|
| `detector/core/models/backbones/mobilepixornext_blocks.py` | Extend `LiteMLARefinement` with QK normalization while preserving FP32 accumulation and existing scale branches. |
| `detector/core/models/backbones/mobilepixornext.py` | Accept and forward attention scales and normalization mode. |
| `detector/core/models/backbones/registry.py` | Read the new model config keys. |
| `configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json` | Preserve the current single-scale reference configuration. |
| `configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_ms_litemla_oga.json` | Add the Pillar 2 configuration with `(3, 5)` and RMSNorm. |
| `tests/test_ms_litemla.py` | Add focused unit, stability, compatibility, and scripting tests. |
| `tests/test_mobilepixornext_backbone.py` | Add configuration wiring and full-backbone integration assertions where appropriate. |
| `docs/plans/mobilepixornext_improvements/00_ABLATION_STUDY_PROTOCOL.md` | Fill results only after experiments; do not place predicted gains in measured-result cells. |

Avoid changing the detection head, OGA loss, neck, dataset augmentation, or NMS as part of M2.

## 4. Task Breakdown

### Task 0 — Freeze the isolated Pillar 2 baseline

- [ ] Confirm the branch starts from `feature/mobilepixornext-improvements` commit `7d394d7`.
- [ ] Record the existing single-scale `(5,)` LiteMLA behavior and parameter count before changes.
- [ ] Inspect Pillar 1 only for naming, config, test, and documentation conventions.
- [ ] Do not merge, cherry-pick, or copy Pillar 1 implementation changes into this branch.
- [ ] Record the exact Pillar 2 base commit SHA used by all experiments.

Exit criterion: the current improvements baseline tests pass and its single-scale LiteMLA behavior is recorded before any Pillar 2 implementation commit.

### Task 1 — Add failing focused tests

Create `tests/test_ms_litemla.py` with tests for:

- [ ] Output shape preservation for `(B, 96, 100, 88)` using `scales=(3, 5)`.
- [ ] Forward and backward produce finite values and gradients.
- [ ] Exactly two aggregation branches are created for `(3, 5)`.
- [ ] `qk_norm="rmsnorm"`, `"layernorm"`, and `"none"` construct the expected mode.
- [ ] Invalid scale lists, duplicate scales, even kernels, and unsupported norm names raise `ValueError`.
- [ ] QK normalization operates over `head_dim`; test with token count different from `head_dim` to catch axis mistakes.
- [ ] The default constructor retains single-scale `(5,)`, no-norm behavior.
- [ ] TorchScript eager/scripted outputs agree in evaluation mode.

Add a CUDA-only test guarded by `torch.cuda.is_available()` and BF16 support:

- [ ] Run forward and backward under BF16 autocast with inputs containing values around `1e3` and `1e-4`.
- [ ] Assert all outputs and parameter/input gradients are finite.
- [ ] Confirm the attention reduction uses FP32 internally and the residual output follows PyTorch's input/residual dtype promotion rules.

Run the new tests first and verify the M2-specific assertions fail for the expected missing functionality.

### Task 2 — Implement robust QK normalization in `LiteMLARefinement`

- [ ] Add a validated `qk_norm` argument with default `"none"`.
- [ ] Add shared-per-head-dimension RMSNorm parameters for query and key when `qk_norm="rmsnorm"`.
- [ ] Add LayerNorm support only for the protocol's sensitivity comparison; keep RMSNorm as the M2 default.
- [ ] Reshape/transpose query and key so normalization is applied over `head_dim`.
- [ ] Apply `ReLU` after normalization to retain a positive linear-attention feature map and a non-negative denominator.
- [ ] Keep the entire attention core in FP32 when the input is FP16/BF16.
- [ ] Cast only the final attended tensor back to the original input dtype.
- [ ] Retain the residual connection and LayerScale initialization at `0.01`.
- [ ] Preserve the existing native QKV branch plus one aggregation branch per configured kernel scale.

Do not rewrite the attention implementation into quadratic `N x N` attention. Complexity must remain linear in the number of BEV tokens.

### Task 3 — Wire M2 options through backbone and registry

- [ ] Extend `MobilePixorNeXtBackbone.__init__` with `c4_attention_scales` and `c4_attention_qk_norm`.
- [ ] Convert JSON lists to an immutable tuple before constructing `LiteMLARefinement`.
- [ ] Pass the new keys from `registry.py`, using `(5,)` and `"none"` as compatibility defaults.
- [ ] Keep `c4_attention="none"` behavior unchanged and avoid constructing unused norm parameters.
- [ ] Ensure the M1 reparameterization flags remain independent of attention settings.

Add integration assertions proving that:

- [ ] An old config produces one `5x5` aggregation branch with no QK norm.
- [ ] The M2 config produces `3x3` and `5x5` branches with QK-RMSNorm.
- [ ] Both configurations return `(B, 16, 200, 176)` from the backbone.

### Task 4 — Create controlled configurations

Keep the existing reference config unchanged. Create a separate Pillar 2 config copied from it and change only:

```json
{
  "model": {
    "c4_attention_scales": [3, 5],
    "c4_attention_qk_norm": "rmsnorm"
  },
    "note": "mobilepixornext_m2_ms_litemla_3_5_rmsnorm_oga"
}
```

- [ ] Confirm no Pillar 1 `use_reparam` or deploy-only options are introduced.
- [ ] Keep the KITTI split, Rich8 encoding, augmentation, OGA loss, optimizer, scheduler, seed, batch size, and precision identical.
- [ ] Store the resolved config with each training run.
- [ ] Do not modify the shared baseline config in a way that silently turns M1 runs into M2 runs.

### Task 5 — Run regression and smoke verification

Run the focused tests, backbone tests, model registry tests, OGA training-pipeline test, and then the full suite.

Minimum verification groups:

```text
tests/test_ms_litemla.py
tests/test_mobilepixornext_backbone.py
tests/test_model_registry.py
tests/test_training_pipeline_oga.py
tests/
```

Then run a small training smoke test using the M2 config:

```text
epochs=1
max_train_batches=8
max_val_batches=4
precision=bf16
seed=42
```

Acceptance checks:

- [ ] Loss remains finite.
- [ ] At least one optimizer update occurs.
- [ ] Checkpoint save and reload succeed.
- [ ] No NaN/Inf appears in model or criterion gradients.
- [ ] Existing backbone construction and checkpoint loading remain functional when new options are omitted.

### Task 6 — Measure parameters and latency

Measure the current single-scale reference and Pillar 2 variant on the same machine and software environment.

Protocol:

```text
input: (1, 8, 800, 704)
mode: eval + inference/no_grad
warmup: 100 iterations
measurement: 500 iterations
CUDA synchronization: before and after every timed region
precision: same FP16/BF16 mode for both models
```

Record:

- [ ] Training parameter count.
- [ ] Inference parameter count for each isolated attention variant.
- [ ] Mean, p50, and p95 model latency.
- [ ] Peak CUDA memory.
- [ ] Hardware, PyTorch, CUDA, and precision details.

The planned budget is:

- Total model remains below `2.0M` parameters.
- Pillar 2 latency increase over the single-scale reference is targeted at `<= 0.08 ms` on the reference RTX 3060 environment.

Treat the latency value as a measured benchmark target, not a unit-test assertion, because timing thresholds are hardware-dependent and flaky in CI.

### Task 7 — Train the controlled ablation matrix

Run these minimum variants:

| Run | Scales | QK norm | Purpose |
|---|---|---|---|
| `P2-reference` | `(5,)` | none | Required isolated baseline from the improvements branch. |
| `M2-scale-only` | `(3, 5)` | none | Isolate the benefit/cost of multi-scale aggregation. |
| `M2-final` | `(3, 5)` | RMSNorm | Measure the complete Pillar 2 contribution. |
| `M2-layernorm` | `(3, 5)` | LayerNorm | Optional normalization sensitivity experiment. |

For the main comparison, train every required variant from scratch using the same seed and schedule. Do not resume the Pillar 2 variant from another architecture's checkpoint and present it as a controlled comparison.

Recommended experiment levels:

1. **Smoke:** one epoch, limited batches.
2. **Primary:** full 100 epochs with seed 42.
3. **Confidence:** repeat the winning and reference variants with seeds 43 and 44 if compute allows.

For every run, evaluate the same validation split and report:

- [ ] Car/Pedestrian/Cyclist BEV AP R40 for Easy/Moderate/Hard.
- [ ] Moderate mean BEV AP and overall mean BEV AP.
- [ ] Validation loss and selected checkpoint epoch.
- [ ] Parameter count, latency, FPS, and peak memory.
- [ ] Any non-finite training event or skipped optimizer update.

M2 should not be merged based only on training loss. The decision must use detection AP together with runtime cost.

### Task 8 — Document results and prepare merge

- [ ] Record the isolated Pillar 2 comparison separately; update cumulative M1/M2 rows only after the integration branch combines both pillars.
- [ ] Include exact commit SHAs, config hashes, split hash, seed, hardware, and checkpoint paths.
- [ ] State clearly whether metrics are BEV AP or AP3D.
- [ ] Summarize which component provided the gain: multi-scale branches, normalization, or both.
- [ ] Record deviations from the original spec and why they were necessary.
- [ ] Review the diff to ensure no Pillar 3, Pillar 4, or augmentation work leaked into the branch.
- [ ] Merge only after tests pass and the M2 accuracy/runtime trade-off is accepted by the team.

## 5. Acceptance Criteria

Implementation is complete when all of the following are true:

1. Pillar 2 is implemented independently on the exact `feature/mobilepixornext-improvements` base without merging Pillar 1.
2. Old configs/checkpoints remain valid when the new options are omitted.
3. `(3, 5)` multi-scale LiteMLA preserves input/output shape and has finite gradients.
4. QK-RMSNorm operates over `head_dim` and passes FP32/BF16 stability tests.
5. Linear attention remains `O(N)` and never materializes an `N x N` matrix.
6. The full model remains below the `2.0M` parameter budget.
7. The single-scale reference and Pillar 2 variant are trained and evaluated under an identical controlled protocol.
8. Latency and memory are measured, not inferred from the design document.
9. Results are labeled truthfully as BEV AP unless a genuine 3D evaluator is used.
10. The final report contains enough run metadata for another developer to reproduce the comparison.

## 6. Suggested Commit Sequence

```text
test(attention): define robust multi-scale LiteMLA behavior
feat(attention): add QK normalization to LiteMLA refinement
feat(config): expose M2 LiteMLA scales and normalization
test(backbone): verify M1/M2 compatibility and BF16 stability
docs(ablation): record M2 experiment protocol and measured results
```

Keep implementation commits separate from experiment-result commits so code review does not mix architectural changes with generated reports.

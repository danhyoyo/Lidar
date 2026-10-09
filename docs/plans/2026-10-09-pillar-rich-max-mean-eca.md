# Pillar rich max–mean ECA implementation

### Goal

Add mean-conditioned ECA mixing of per-pillar learned max/mean features, preserving rich8 + learned24 =32 channels. Reduce avoidable scatter/copy and validation overhead, and provide a reproducible rich8 versus pillar timing benchmark without claiming unmeasured CUDA speed or AP.

### Assumptions

- `pillar_rich` defaults remain max-only and retain historical checkpoint semantics/state keys.
- New recipe uses `pooling="max_mean_eca"`, `eca_kernel_size=3`, the existing one-layer point MLP, and no backbone attention.
- Gate is per occupied pillar and channel, generated from learned mean, not global dense-BEV attention.
- Python is available via `/home/duyennh/miniconda3/envs/AI_env/bin/python`. The sandbox hides CUDA, but unrestricted verification can use the local RTX 4050 Laptop GPU.

### Plan

1. Add behavioral regression tests before implementation.
   - Files: `tests/test_pillar_max_mean_eca.py`.
   - Change: schema/checkpoint distinction, local gate/reference pooling, permutation/batch isolation, empty/singleton, FP32/BF16 gradients, model/config/preset smoke.
   - Verify: run new suite and confirm failures are caused by missing pooling support.
2. Add strict pooling semantics while preserving default identity.
   - Files: `detector/core/bev_encoding.py`, `detector/core/models/encoders/pillar.py`.
   - Change: validate options, describe gate/mean precision in semantic identity, FP32 segmented mean and small channel convolution on occupied pillars.
   - Verify: new reference checks plus existing pillar/rich encoding tests; old default/explicit max identity and old checkpoints remain compatible.
3. Remove avoidable forward overhead.
   - Files: `detector/core/models/encoders/pillar.py`.
   - Change: combine bounds checks into one device predicate; scatter directly into contiguous NCHW instead of allocating NHWC and copying the whole BEV.
   - Verify: reference output/gradient equality, malformed-input rejection and contiguous batch output tests.
4. Integrate training configuration and performance measurement.
   - Files: `tools/kitti_training_pipeline/notebook_config.py`, `tools/kitti_training_pipeline/common.py`, `configs/experiments/encoders/pillar_rich_max_mean_eca.json`, `tools/benchmarks/benchmark_pillar_encoders.py`.
   - Change: new preset, unique variant run naming even with default backbone, matched random-weight benchmark with separate CPU preparation/H2D/model boundaries and explicit precision/hardware.
   - Verify: config equals preset; model count=655516 with kernel3; old selection rejected; CLI train/resume/evaluation smoke and small CPU benchmark.
5. Review, document and verify.
   - Files: `docs/learned_pillar_encoder.md`, `tools/kitti_training_pipeline/README.md`, this plan.
   - Change: activation and Colab custom overrides, fresh-run requirement, GPU benchmark instructions and honest speed limitations.
   - Verify: relevant suite then full suite, `git diff --check`, inspect diff for unrelated changes; record hardware-dependent skips/pre-existing failures.

### Risks & mitigations

- BF16 summation over many points: accumulate mean and gate in FP32; only cast fused features to model dtype.
- Sparse input and empty frames: existing packed contract retained; connect ECA weights to zero output for empty-frame backward.
- Checkpoint confusion: include pooling/kernel in effective encoding identity; retain historical max-only identity.
- Added mean reduction can dominate ECA parameter cost: benchmark actual operations; preserve all points and avoid point cap or wider BEV.
- Faster scatter can change tensor layout/gradient behavior: use explicit reference and batch isolation tests.
- No 50-epoch run performed here: verify CUDA FP32/BF16 and record local GPU timing; AP and L4 speed remain measurements for a new run.

### Rollback plan

Select existing `ENCODER_PILLAR_RICH` / `pooling="max"` with its original checkpoint. The new preset and run name are separate; do not overwrite source archives or unrelated working-tree changes.

### Verification and review

- New tests failed for missing pooling support before implementation. Final focused suite outside the sandbox: **127 passed**, including CUDA FP32/BF16, DataLoader spawn workers, BF16 summation/gradient checks, and CLI train/resume/evaluate. No CUDA skips in the unrestricted run.
- Manual channel convolution matches independent `F.conv1d` output and parameter gradients for kernels1/3/5. Direct NCHW scatter passes existing pillar tests and permutation/batch/empty-frame checks.
- Schema review: default max-only retains historical metadata/state keys; gate variants have distinct encoding/checkpoint/run identities. The new preset matches its saved JSON and changes only encoder options relative to the pillar-rich comparison recipe.
- GPU benchmark: RTX4050, batch1, FP32, warmup10/50 iterations, real KITTI frames000000/000001/000010, matched random-weight detector. New forward7.998/8.046/7.962 ms versus rich8 6.783/6.801/6.730 ms; preparation+transfer+forward is5–10% higher than rich8. Mean/gate adds0.17–0.19 ms versus max-only. Detailed reports are ignored artifacts; scopes/limitations documented in `docs/learned_pillar_encoder.md`.
- The initial full-suite attempts encountered sandbox socket restrictions and exhausted temporary-file quota. Their failure totals are not a reliable regression result; fixtures from those attempts were removed. Eleven recipe/config consistency failures were reproduced on an isolated archive of unmodified HEAD, confirming they predate this change. A final unrestricted full-suite run uses an artifacts directory for fixtures.
- Final full suite: **1714 passed,19 skipped,19 failed**, with17 subtests passed. All19 failures reproduced on unmodified HEAD:11 recipe/config mismatches;7 failures involving reference CUDA/Numba dependencies, existing smoke-selection rules and notebook contents;1 GUI failure reproduced after creating the same IPython singleton as earlier notebook tests. The GUI test passes in isolation. No failure is in the new encoder suite. Logs: `artifacts/pillar_max_mean_eca/full_suite_final.log`, `baseline_other_failures.log`, `baseline_gui_failure.log`.
- Review found no remaining blocker in the encoder change. GPU speed figures are preliminary timing evidence, not AP evidence or a promise of L4 throughput.

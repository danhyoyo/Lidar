# Separate pillar-rich encoder names

### Goal

Expose max-only `pillar_rich` and max–mean ECA `pillar_rich_eca` as two distinct encoder selections, preserving the numerical implementation and max-only checkpoint identity.

### Assumptions

- Selecting `pillar_rich_eca` alone enables the existing ECA pooling with kernel3.
- `pillar_rich` accepts max only; configurations requesting ECA under that name receive a migration error.
- Canonical preset/config are `ENCODER_PILLAR_RICH_ECA` and `pillar_rich_eca.json`. The previous long preset remains an alias selecting the new encoder name.
- Naming is part of checkpoint identity: use a new run for the renamed ECA encoder. No raw checkpoints or archived experiment outputs are rewritten.
- Existing unrelated local notebook/database changes are excluded from the commit. The earlier request to push this feature remains authorized.

### Plan

1. Extend regression tests before implementation.
   - Files: `tests/test_pillar_max_mean_eca.py`, `tests/test_pillar_encoder.py`, `tests/test_under1m_notebook_config.py`.
   - Change: distinct names, name-only activation, conflicting-option rejection, historical max identity, canonical/legacy presets, notebook widths, pipeline/AP/resume/spawn paths.
   - Verify: focused tests fail for unsupported `pillar_rich_eca` before changes.
2. Wire the name through encoding and the complete pipeline.
   - Files: `detector/core/{bev_encoding.py,models/model.py,models/encoders/pillar.py,datasets/dataset.py,datasets/utils_1/pillar_backend.py}`, `tools/kitti_training_pipeline/{common.py,train.py,notebook_config.py}`.
   - Change: rich8 fusion for both rich variants; separate pooling defaults and semantic identities; correct packed model/dataset/collate/warmup behavior and run names.
   - Verify: CPU/CUDA output/gradient, old encoder, dummy-input and train/resume/evaluation tests.
3. Rename experiment configuration and update entry points.
   - Files: `configs/experiments/encoders/pillar_rich_eca.json`, `tools/benchmarks/benchmark_pillar_encoders.py`, `3D_Lidar_Object_Detection_Notebook_standard.ipynb`, `docs/learned_pillar_encoder.md`, `tools/kitti_training_pipeline/README.md`.
   - Change: canonical encoder/preset/config names; notebook selection hints; consistent benchmark labels. Preserve hyperparameters and historical performance evidence.
   - Verify: canonical JSON equals resolved preset; benchmark smoke uses the new name; notebook remains valid JSON.
4. Review, verify and publish.
   - Files: all above and this plan.
   - Change: document migration and final checks; commit only the feature files and push the current branch.
   - Verify: focused suite, required full suite, `git diff --check`, remote commit equals local HEAD.

### Risks & mitigations

- Missing a raw name check could send packed data to a dense path: cover all model/dataset/train/warmup adapters and integration tests.
- Accidentally altering max-only identity breaks historical archives: compare metadata/hash with the previous committed implementation.
- Older preset strings: preserve a preset alias but always resolve it to the new encoder name.
- Full-suite baseline failures: compare with the recorded19 failures from the prior verified commit rather than modify unrelated tests.

### Rollback plan

Use `pillar_rich` / `ENCODER_PILLAR_RICH` with its existing max-only checkpoint. Before publishing, review the staged paths; after publishing, use a revert commit if required, without rewriting shared history.

### Verification and review

- Red phase:4 focused cases failed for the unsupported new name, missing preset, accidental baseline gate activation and run-name fallback to legacy35.
- Focused suite: **157 passed**, including CUDA FP32/BF16, name-only activation, strict conflicting options, matched reference output/gradient, default max-only identity, named/legacy presets, notebook compatibility, dummy compile inputs, AP selection, resume/evaluate and spawn workers.
- Max-only semantic hash for the regression geometry remains `e05ae9866d3e51a4ead83821ea5a1144e6f240af860071e08e50655cc7814d60`.
- Canonical experiment JSON equals the resolved preset and differs from `pillar_rich.json` only in encoding options. Standard notebook JSON is valid and lists both encoder choices.
- Real KITTI frame000000 benchmark on RTX4050 BF16, batch1, warmup10/50 iterations completed with distinct labels `rich8`, `pillar_rich`, `pillar_rich_eca`, finite detector outputs and correct semantic metadata. JSON: `artifacts/pillar_rich_eca_verification/rtx4050_kitti_000000_bf16.json`. This is random-weight timing evidence, not AP evidence.
- Review checked every raw name adapter and the shared rich8 preparation/fusion path. The pooling arithmetic and parameter state keys are unchanged. No encoder blocker found; ECA checkpoint identity changes intentionally with its new name.
- Python compilation and `git diff --check` passed.
- Full suite: **1731 passed,19 skipped,19 failed**, plus17 passed subtests. Automated comparison confirms the19 failing cases exactly match those recorded before this change; they were already reproduced on unmodified HEAD in the previous feature verification. No new full-suite failure. Logs and comparison summary: `artifacts/pillar_rich_eca_verification/`.
- Publication scope: the encoder/config/notebook selection changes and their tests/docs on `research/mobilepixornext-under1m`; exclude the existing local database-builder edit, robust notebook/generator, datasets and generated artifacts.

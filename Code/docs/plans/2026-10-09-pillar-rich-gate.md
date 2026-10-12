### Goal
Replace the retired pillar_rich_eca encoder with pillar_rich_gate, leaving the max-only pillar_rich baseline unchanged.

### Assumptions
- One fixed gate design: concatenate learned max24 and existing rich8, Linear(32,4) + ReLU + Linear(4,24), then multiply max24 by 2*sigmoid(logits).
- Both Linear layers have bias (252 extra parameters). Zero only the final Linear weight/bias for identity initialization.
- Rich8 passes through unchanged. No learned mean pooling, point attention, or dense BEV attention.
- Gate construction preserves CPU RNG state; all shared baseline parameters keep identical initialization.
- Old ECA encoder/presets are removed, not silently aliased to a numerically different encoder. Historical experiment records remain.

### Plan
1. Add regression coverage for the replacement.
   - Files: tests/test_pillar_rich_gate.py, tests/test_pillar_encoder.py, tests/test_under1m_notebook_config.py, tests/test_run_name_and_logging.py.
   - Change: Replace ECA tests with gate tests covering identity, rich-conditioned behavior, gradients, AMP, batch independence, config and checkpoint identity.
   - Verify: AI_env Python -m pytest tests/test_pillar_rich_gate.py -q (fails before implementation).
2. Replace the encoder and configuration contract.
   - Files: detector/core/bev_encoding.py, detector/core/models/encoders/pillar.py, detector/core/models/model.py, detector/core/datasets/dataset.py, tools/kitti_training_pipeline/{common,notebook_config,train}.py.
   - Change: Register pillar_rich_gate; keep pooling=max; record fixed gate semantics and remove old options/presets.
   - Verify: Focused pillar, notebook configuration, checkpoint and run-name tests.
3. Update user-facing recipes and benchmark.
   - Files: configs/experiments/encoders/pillar_rich_gate.json, 3D_Lidar_Object_Detection_Notebook_standard.ipynb, tools/benchmarks/benchmark_pillar_encoders.py, docs/learned_pillar_encoder.md, tools/kitti_training_pipeline/README.md.
   - Change: Replace active ECA selections and documentation; do not label old ECA timing results as gate results.
   - Verify: Preset/config equality, CPU benchmark smoke, notebook controls and full pytest suite; inspect diff.

### Risks & mitigations
- Checkpoint incompatibility: distinct semantic identity rejects ECA/baseline resume into gate.
- Kernel/launch overhead: no speed claim without a matched GPU benchmark.
- Existing unrelated edits: leave GT-database changes, robust notebook/tools and data untouched.

### Rollback plan
Restore only files touched by this change from the pre-change revision; preserve user files and experiment artifacts. Do not relabel old checkpoints.

### Verification results
- Regression red check before implementation: 2 failures (new encoder unsupported and retired encoder still accepted), 12 passes.
- Focused pillar/config/run-name/notebook controls: 151 passed on CPU and CUDA/BF16, including spawned DataLoader, train/resume/evaluate smoke and exact shared initialization parity.
- Additional actual notebook controls/schema/model registry/gate suite: 74 passed, including the two added real-cell tests.
- Full suite: 1734 passed, 19 skipped, 19 failed, 17 passed subtests. Failure identifiers exactly match the 19 already recorded in the preceding ECA verification; no new failure. Reports: artifacts/pillar_rich_gate_verification/full_suite.{log,xml}, failure_comparison.json.
- Plain pytest initially encountered an unrelated ROS auto-loaded plugin import failure. Verification used PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. Sandboxed DataLoader sockets were denied; final focused/full verification ran outside the sandbox.
- Self-review: no blocker or major issue introduced. Baseline config/semantic hash/state keys are preserved; active encoder/config/tool/notebook paths contain no retired ECA encoder references. Historical plans and archived results remain.
- Real frame000000 benchmark on RTX4050 BF16, batch1, warmup10/50 iterations: rich8 model5.743ms / input-to-output14.885ms; pillar_rich6.577ms /16.377ms; pillar_rich_gate6.667ms /16.464ms. Random weights, one frame, excludes I/O/decode/NMS/AP; no dataset-wide speed claim. JSON: artifacts/pillar_rich_gate_verification/rtx4050_kitti_000000_bf16.json.

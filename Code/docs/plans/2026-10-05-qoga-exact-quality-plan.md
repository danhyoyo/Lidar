# Q-OGA: exact rotated quality targets

### Goal

Build a testable Q-OGA candidate with detached, exact rotated BEV IoU at classification peaks. Improve alignment between score supervision and evaluation without assuming AP gains before training.

### Assumptions

- Optimize validation Moderate BEV mAP, with Pedestrian and 30–50 m AP monitored.
- One seed (42), one candidate per training round. Existing Q-OGA, IQA and GW-QAL artifacts remain references.
- Keep legacy `q_oga` behavior and checkpoint keys when new options are absent.
- Preserve regression, Gaussian negative masking, RDA and EMA/softmax weighting in the first candidate.
- CUDA and processed KITTI are unavailable here. CPU PyTorch/Shapely checks are available through `/home/duyennh/miniconda3/envs/AI_env/bin/python`.

### Plan

1. Specify exact IoU behavior with failing tests.
   - Files: `tests/test_rotated_iou_targets.py`.
   - Change: independent polygon reference, identity/disjoint/touching/containment/crossing edges, angle and dimension equivalence, scale and translation, empty masks, detach and BF16.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_rotated_iou_targets.py -q`.
2. Implement batched aligned rectangle intersections in PyTorch.
   - Files: `detector/core/losses/rotated_iou.py`, `detector/core/losses/iou_targets.py`.
   - Change: contained corners plus edge intersections; order intersection vertices and calculate area. Decode width/length and doubled yaw with the existing convention. Disable autocast and detach geometry.
   - Verify: tests from step 1, including randomized comparison to Shapely in float64.
3. Add opt-in exact quality and optional curriculum.
   - Files: `detector/core/losses/strategies/q_oga.py`, `tools/kitti_training_pipeline/train.py`, `tests/test_q_oga_exact_quality.py`.
   - Change: `quality_target=rotated_iou`, default legacy `mgiou`; optional `quality_warmup_epochs`, zero-based epoch hook invoked for train and validation; peak-only quality telemetry.
   - Verify: target-only component equality against legacy, finite backward FP32/BF16, classification gradient isolation, empty objects and resume/epoch behavior.
4. Prepare reproducible single-seed experiments and usage.
   - Files: `configs/experiments/qoga_quality/*.json`, `docs/plans/2026-10-05-qoga-exact-quality.md`, configuration tests.
   - Change: matched legacy, target-only and conditional curriculum configs copied from the archive recipe with actual batch 16, 100 epochs and seed 42. Distinct experiment names; full commands with numba/compile flags.
   - Verify: JSON loading, identical recipe except declared quality options, pipeline model forward/backward and CLI help.
5. Review and record verification.
   - Files: touched implementation, tests and documentation.
   - Change: inspect diff for unintended legacy/model changes; record test outcomes and limits.
   - Verify: focused loss and training tests, `git diff --check`, independent geometry error metrics.

### Risks & mitigations

- Exact IoU may be zero at early positive peaks: monitor raw zero fraction; curriculum is a separate optional experiment and ends at exact IoU.
- Polygon arithmetic near tangency: FP32 geometry, pair-local coordinates, dimension normalization, independent reference tests.
- GPU runtime unmeasured: batched device-native operations with no Shapely/CPU transfer in training; benchmark on the target GPU before asserting overhead.
- Existing archive differs from current source: matched legacy config is provided when a clean attribution is required. Do not call old/new AP differences causal without matching recipe/source/split.
- Warmup changes validation objective over time: judge final AP, not minimum validation loss across the curriculum.

### Rollback plan

Use `quality_target=mgiou` and `quality_warmup_epochs=0`, or the legacy experiment config. No model or inference head changes are needed.

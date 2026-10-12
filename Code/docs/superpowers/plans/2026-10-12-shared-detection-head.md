### Goal

Implement the proposed per-group shared Conv3×3, with a selectable architecture
and unchanged historical config/checkpoint behavior by default.

### Assumptions

- `model.header_architecture` selects `independent` (default) or `shared`.
- Shared mode replaces the first private Conv3×3 in every branch with one
  Conv3×3 inside its Header; grouped Headers never share parameters.
- Both modes retain output keys/widths, optional IQA/vertical branches, and
  configured BN/activation. A single Header may also select shared mode.

### Plan

1. Capture compatibility evidence and add regression tests.
   - Files: `tests/test_shared_header.py`.
   - Change: cover output contracts, isolated gradients, parameter budgets,
     construction/validation, checkpoints, training, and notebook controls.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_shared_header.py -q`
     must fail for missing shared architecture; preserve pre-change state and
     predictions in `/tmp/shared-head-baseline`.
2. Implement selectable Header construction and pure config validation.
   - Files: `detector/core/models/heads/cnn.py`, `heads/grouped.py`,
     `detector/core/models/model.py`, `detector/core/detection_config.py`.
   - Change: create one shared block before the branch fork in shared mode;
     keep legacy module registration/initialization order unchanged.
   - Verify: focused regression tests and exact pre-change checkpoint/prediction comparison.
3. Integrate checkpoint semantics, run naming and notebook selection.
   - Files: `tools/kitti_training_pipeline/common.py`, `notebook_config.py`,
     both root notebooks.
   - Change: canonicalize the default away for old identities; record shared
     architecture and distinct run names; expose `HEADER_ARCHITECTURE`.
   - Verify: checkpoint resume/evaluation mismatches reject before mutation,
     matching shared checkpoints resume, notebook controls reach saved configs.
4. Document and verify the production implementation.
   - Files: `tools/kitti_training_pipeline/README.md`, comparison generator/docs,
     a full shared SimAM experiment config.
   - Change: document selection and warm-start behavior; compare the actual
     production implementations instead of the former prototype.
   - Verify: run head/config/checkpoint/training/evaluation/notebook regressions,
     full-size forward, and regenerate comparison figures from production code.

### Risks & mitigations

- Changing default registration would break checkpoints: compare keys, seeded
  tensors and predictions against the captured pre-change models.
- Shared and independent weights are different topologies: semantic identities
  must differ; switching architecture uses a new run or backbone warm start.
- Shared task features may change AP: no accuracy improvement is claimed without training.

### Rollback plan

Select `independent` or omit the key. Existing configs/presets remain unchanged.
Remove only these feature additions if rollback of code is needed.

### Verification results

- Initial feature regressions: 43 failures for missing architecture APIs,
  ignored config keys, unchanged shared budgets/identity, and missing controls.
- Final focused suite: 364 passed (49 shared-head cases plus existing head,
  detection config, checkpoint, training, evaluation, export guard and notebook
  tests). Environment warnings: unavailable NVML and temporary IPython directory.
- Pre-change captured grouped/single models match exactly: state keys, seeded
  tensors, strict loading, predictions, checkpoint identity and run name, for
  omitted and explicit `independent`, using the same CPU thread count.
- Serialized shared resume reproduces the next loss, model/criterion state,
  optimizer state and scheduler after a completed training step. Cross-topology
  resume/evaluation rejects before restoring state. Backbone warm start preserves
  the newly initialized shared heads.
- Full CPU model forward at 1×14×800×704: all ten outputs finite at 200×176;
  shared head 111,921 parameters, Focal/SimAM/fusion32 detector 778,690.
- Standard notebook custom/preset controls tested; actual robust configuration
  cell executed with shared and checked against saved config/model.
- Comparison generator now audits both production GroupedHeaders. Regenerated
  PNG/SVG/vector PDF/TikZ artifacts checked for labels, code hashes and budgets.
- Local review: no blockers or majors. AP/latency comparison remains a future
  trained experiment; this change implements architecture selection.

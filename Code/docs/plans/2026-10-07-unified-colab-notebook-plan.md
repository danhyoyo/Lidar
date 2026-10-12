# Unified Colab notebook

### Goal

Deliver one Colab workflow whose only editable source JSON is `configs/config.json`. Notebook controls resolve model, augmentation, training and metric options; generated run snapshots preserve reproducibility. Keep BEV development as the default and make final reference BEV/3D explicit.

### Assumptions

- The user authorizes notebook and supporting workflow/configuration changes. Existing model, loss, geometry, runtime recipe files and unrelated dirty/deleted files remain intact.
- Main default controls select hist14/local3/focal, grouped Car/Pedestrian-Cyclist heads, 32 head input channels, OGA/IQA and hybrid GT.
- Built-in optional presets and augmentation recipes need no auxiliary JSON when resolving notebook configs. Saved baseline/checkpoint configs are provenance, not user-maintained experiment files.
- Development runs may train with unresolved comparator gates. Benchmark runs require a frozen matched protocol. No real-data training, deployment, remote publishing or SOTA claim is executed locally.

### Plan

1. Establish configuration and workflow regressions.
   - Files: `tests/test_unified_notebook_workflow.py`, `tests/test_under1m_notebook_controls.py`.
   - Change: verify one-master-file resolution, actual default/control propagation, metric/mode guards, own-config selected checkpoints, truthful metrics, command routing and failed/incomplete benchmark rejection.
   - Verify: `python -m pytest tests/test_unified_notebook_workflow.py -q` must fail for missing behavior before implementation.
2. Resolve all notebook options from the master and Python recipes.
   - Files: `tools/kitti_training_pipeline/notebook_config.py`.
   - Change: built-in augmentation/under1m recipe definitions reproduce existing materialized JSON; no auxiliary JSON required for notebook resolution. Preserve legacy APIs and generated recipes.
   - Verify: `python -m pytest tests/test_under1m_notebook_config.py tests/test_grouped_3d_pipeline.py tests/test_unified_notebook_workflow.py -q`.
3. Add small checked workflow helpers.
   - Files: `tools/kitti_training_pipeline/notebook_workflow.py`, `common.py`, `evaluate_kitti_3d.py`, `tools/benchmarks/benchmark_evidence.py`, `tools/benchmarks/benchmark_protocol.py`.
   - Change: validate metric modes, select each run's actual resolved config/best checkpoint, construct safe train/evaluation commands, render real result rows, verify pinned reference CUDA parity and enforce benchmark readiness.
   - Verify: small actual training/evaluation fixtures, negative mode/selection cases, actual compatible GPU reference parity and existing evaluator regressions.
   - Integration finding: actual trainer archives have destination-dependent names and different SHA despite identical payloads. Verify full selected/retained state when file bytes differ; keep each file hash for freshness. Preserve trainer/model formulas. Normalize local raw roots and add an optional reference CLI run label for unambiguous comparison rows.
4. Integrate complete notebook cells.
   - Files: `3D_Lidar_Object_Detection_Notebook_standard.ipynb`.
   - Change: correct branch; explicit source availability preflight; one control cell with main defaults; mode-aware extraction/GT audit; current proposal tests/budget/GPU smoke; comparator evidence/freeze; safe train/resume; own-config BEV/reference evaluation and complete comparison records. Separate optional external-run evaluation and GT preview.
   - Verify: execute configuration/result cells with temporary paths, compile every cell via IPython transform, validate nbformat, inspect all generated commands, run focused notebook tests.
5. Document and verify delivery.
   - Files: `docs/plans/lightweight_lidar_backbone_2026/notebook_runbook.md`, index/execution/runbook links.
   - Change: English guide for development, controlled no-context baseline, benchmark gate and explicit final 3D results; record local verification and external Colab limitations.
   - Verify: focused tests followed by full GPU/reference suite; static/document checks, master/materialized config and model/source preservation audit.

### Risks & mitigations

- Never choose a different checkpoint/config silently. Missing selected checkpoints, incomplete training or mismatched own-config identity fail explicitly.
- BF16 availability varies: development smoke uses the selected precision; benchmark precision policy retains the existing required matrix.
- Reference CUDA compilation is checked on the actual runtime before reference training/evaluation; dependency failure cannot substitute local BEV.
- Notebook remote checkout cannot fetch uncommitted local files. Preflight checks required tools and explains how to use an already uploaded checkout; no push is performed.
- Audit/protocol errors are visible; development status is labeled pending and is never advertised as benchmark-ready.

### Rollback plan

Restore only notebook/config/workflow changes from the batch22 snapshot. Retain existing model/recipes, all datasets/checkpoints and prior work. Generated configs/results remain under ignored artifacts or the configured Colab Drive run directory.

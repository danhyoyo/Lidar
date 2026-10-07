# Candidate benchmark protocol

Updated 2026-10-07, tasks51/58. Status: **task51 audit/evidence implementation available; final Colab benchmark freeze pending actual evidence**. The complete vertical pipeline and pinned reference evaluator are implemented. This record does not establish trained accuracy or matched comparator results.

The current default is **BEV for development, ablations and intermediate comparisons**, as confirmed by the user after task58. The existing BEV notebook/runtime remains the active workflow. The two JSON protocol records below describe the explicitly selected **final 3D comparison stage**, not an automatic switch of the current training config. Record current BEV experiments with their actual BEV config and evaluator; the historical local ROI APBEV R40 must be named separately from reference APBEV.

At the final-results stage, the primary benchmark is **KITTI AP3D, reporting R11 and R40 separately**. The same complete 3D checkpoint supports reference **APBEV**, also reporting R11 and R40. Use `evaluate_kitti_3d.py --metric-mode 3d|bev`; the recorder's `--metric-mode` records the requested protocol but does not run evaluation. The historical local, loader-aligned ROI APBEV R40 evaluator remains a separate diagnostic and rejects 3D configs. A BEV checkpoint must not be reference-evaluated as if it predicted complete 3D boxes: train or fine-tune the explicit 3D model first and match the 3D head/objective across baseline and candidate.

## Candidate settings

| Field | Recorded policy |
| --- | --- |
| Main model | hist14, depths3/4/2, focal, local attention none, SG-FPN output32, Car + Pedestrian/Cyclist heads |
| Parameters | Backbone including neck660,528; 3D heads223,413; detector883,941; train-only criterion12 |
| Objective | Grouped OGA + BEV IQA; owned bottom-z/log-height with fixed vertical weight1 |
| Seed | 42 |
| Split | 3,712 training / 3,769 validation IDs; unique and disjoint |
| Training | Existing hybrid GT recipe; AdamW, lr0.0007, weight decay0.001, cosine, 4 warmup epochs, 50 epochs |
| Precision/batch/workers | BF16 / physical batch16 / accumulation1 / workers6 / compile disabled |
| Checkpoint selection | Minimum validation loss, matching historical trainer behavior |
| Decode | Score0.05, rotated NMS0.10, IQA alpha0.5, per-class peaks, one global cap500 after classwise NMS |
| KITTI metrics | Car IoU0.7; Pedestrian/Cyclist IoU0.5; Easy/Moderate/Hard, both R11 and R40 |
| Evaluation domain | Full benchmark GT with pinned reference ignored/neighbor/DontCare rules; ROI diagnostics reported separately |
| Aggregation | Moderate macro mean across all three classes; AP9 across 3 classes × 3 difficulties; separate value for each metric/sampling |
| Orientation | Pi-symmetric box yaw; no AOS claim |

Split and complete 3D config hashes are in [3D policy](benchmark_protocol_3d.json) and [BEV policy](benchmark_protocol_bev.json). Both now use the materialized main 3D config and verified pinned evaluator source hashes. Actual main CUDA smoke reports cover FP32/FP16/BF16 with workers0/6; the reference verification covers both metric modes and both samplings. Optional ECA has its own complete 3D recipe and corresponding GPU evidence in [Milestone B](release_b_verification.md). SimAM remains a separate BEV comparison, without an advertised 3D runtime preset. Accuracy is unknown for all candidates.

## Reproduce the records

Run from the repository root with the project's Python environment. Existing local evidence paths below reproduce the checked-in records; use new matching evidence/config paths for Colab:

```bash
python tools/benchmarks/benchmark_protocol.py --config configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa_3d.json --metric-mode 3d --smoke-report artifacts/kitti/backbone_audit/batch20/none_workers0.json --smoke-report artifacts/kitti/backbone_audit/batch20/none_workers6.json --reference-verification artifacts/kitti/backbone_audit/batch19/verification.json --output /tmp/protocol_3d.json
python tools/benchmarks/benchmark_protocol.py --config configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa_3d.json --metric-mode bev --smoke-report artifacts/kitti/backbone_audit/batch20/none_workers0.json --smoke-report artifacts/kitti/backbone_audit/batch20/none_workers6.json --reference-verification artifacts/kitti/backbone_audit/batch19/verification.json --output /tmp/protocol_bev.json
```

Omitting evidence leaves the corresponding reference/GPU gates pending. The recorder validates report status, actual CUDA identification, canonical source config hash, all precision/vertical/ownership/restore/full-resolution fields, workers0/configured workers, and reference revision/source hashes with successful non-skipped reference verification. Supplied JSON is trusted verification evidence; these checks do not rerun kernels or authenticate arbitrary reports. Smoke/reference records alone never clear the real-data audit or reproduced-baseline gates. The new `--asset-audit` and `--comparator-evidence` options consume and re-verify those separate records.

Use `--comparator-protocol path.json` for each declared comparator record. The tool reports missing or different metric, recall sampling, class/IoU, GT domain, difficulty/orientation, validation split, checkpoint selection, score/NMS/cap and aggregation fields. Matching declarations alone do not establish implementation parity or reproduced accuracy. Architecture and training settings may differ and must be disclosed for algorithm comparisons.

## Remaining gates

Tasks53–58 now provide the vertical pipeline, immutable reference evaluator revision, actual synthetic reference parity for both metric modes/samplings and main/ECA GPU recipes. **Task51 now implements executable real-data/train-only GT database auditing and reproduced comparator evidence checks; its final freeze still requires actual Colab records.** The user's data and checkpoints are on Colab; local missing paths are not evidence about their existence there. Follow the [Colab runbook](colab_3d_runbook.md) to record actual assets, resolved runtime conditions, checkpoint identity and evaluation inputs.

Both policy records retain `long_training_allowed=false` while those gates are unresolved. No long comparison training was launched. AP improvement and SOTA claims require trained results under matched benchmark conditions.

## Current BEV task51 workflow

The recorder CLI now defaults to `--metric-mode local_bev`, matching the active notebook BEV workflow. This records only actual local ROI R40, with configured geometry; it does not label it reference R11/R40. The Python API retains its historical default for compatibility; new callers should select the mode explicitly. The checked-in primary/reference JSON records above remain final-stage records and are unchanged.

Use [task51 Colab runbook](task51_colab_runbook.md) for concrete own-config baseline evaluation and candidate smoke/audit/freeze commands. The initial internal comparator is the existing no-context hist14/local3/grouped OGA-IQA reference. Backbone/encoding/heads/objective/training conditions must be disclosed; an internal focal ablation is not an external SOTA comparison.

`audit_kitti_assets.py` checks actual configured files, finite points/crops and train-only v1 database/source-label provenance. Freeze re-runs it to detect stale assets. `benchmark_evidence.py` checks full-split evaluator input/config/checkpoint hashes, all class/difficulty AP values and recomputed macros, and minimum-loss selection against complete configured training history and retained checkpoint bytes. Local/reference evaluator reports now include actual input asset hashes; re-evaluate older reports before using them as task51 evidence. Synthetic evidence is rejected for real benchmark freeze. Current initial evidence support covers this repository's PyTorch/selection formats; external methods require an explicit verified adapter.

Only when every applicable GPU, data, reference (if selected), comparator and reproduced-baseline gate passes does the recorder emit `status="protocol frozen for supplied verified evidence"` and `long_training_allowed=true`. A declared comparator protocol alone stays pending. All supplied JSON remains trusted execution evidence, not authenticated remote execution or proof of genuine KITTI origin. Candidate accuracy remains unmeasured until its own actual evaluation. Neither freezing nor fixture AP certifies SOTA.

# Task51: executable asset audit and benchmark evidence

### Goal

Finish the missing implementation for task51: inspect configured KITTI/GT-database assets, record a reproduced baseline with matched evaluation/selection conditions, and consume that evidence before freezing a comparison protocol. BEV remains the current development default; final reference AP3D/APBEV remains explicit. Actual Colab assets and trained checkpoints must supply final evidence.

### Assumptions

- Existing nine BEV/two 3D configs, notebook defaults, model/encoding/loss code and dirty-tree changes are preserved.
- Internal baseline initially uses hist14/local3/grouped OGA-IQA without focal context. It tests focal's contribution and does not establish SOTA against external algorithms.
- Existing local BEV reports contain R40 only and a configured XY ROI; reference BEV/3D reports contain R11/R40 and full-domain GT. Never substitute one protocol for another.
- Supplied evaluator/training JSON is trusted execution evidence, with freshness and consistency checks; it is not authenticated remote execution or proof that arbitrary data is genuine KITTI.

### Plan

1. Implement a read-only KITTI audit.
   - Files: `tools/benchmarks/audit_kitti_assets.py`, `tests/test_kitti_asset_audit.py` (new).
   - Change: validate unique/disjoint splits, finite processed points/labels, original validation labels/calibration and reference image inputs when needed. Verify v1 train-only GT database manifests, entry geometry/source labels and finite crop counts; record config/split/file hashes. An explicit alternate source manifest must have the same train IDs and match database metadata.
   - Verify: failing tests before implementation; run `python -m pytest tests/test_kitti_asset_audit.py -q`, including stale assets and validation leakage.
2. Record reproduced evaluation evidence.
   - Files: `tools/benchmarks/benchmark_evidence.py`, `tests/test_benchmark_evidence.py` (new).
   - Change: validate full-split evaluator output, actual config/checkpoint hashes, mode/domain/classes/difficulty/decode conditions, all class AP values and aggregation. Check minimum-loss selection against complete training JSONL and actual selected checkpoint. Preserve synthetic flags and reject them as real benchmark readiness evidence.
   - Verify: failing tests, then `python -m pytest tests/test_benchmark_evidence.py -q`; use actual small synthetic checkpoint/evaluation fixtures, not invented AP results.
3. Connect evidence and local BEV protocol.
   - Files: `tools/benchmarks/benchmark_protocol.py`, `tests/test_benchmark_protocol.py`, `tests/test_benchmark_freeze.py` (new).
   - Change: add explicit `local_bev` with R40/ROI metadata and make the recorder CLI default to it; preserve existing explicit reference modes and API compatibility. Validate current GPU reports, audited assets and matched baseline/comparator evidence. Permit final freeze only when all relevant gates pass; missing evidence remains pending.
   - Verify: focused tests and actual CLI integration. Partial reports, stale data/checkpoints, missing classes, wrong quality/decode/selection or synthetic evidence must not clear gates.
4. Deliver Colab workflow and final verification.
   - Files: `task51_colab_runbook.md` (new), existing protocol/spec/plan/index/execution docs and recipes README.
   - Change: provide exact BEV-first audit/baseline/candidate/evidence/freeze commands and explicit final reference-mode instructions. Track implementation readiness separately from external Colab benchmark freeze.
   - Verify: relevant tests, full suite in the existing compatible GPU/reference environment, CLI help/reproduction, document links, syntax/whitespace and pre-batch source preservation.

### Risks & mitigations

- Audits read all configured point/crop files; report this cost and avoid full-dataset memory retention. A hash mismatch fails rather than rewriting database metadata.
- Local APBEV can omit missing classes from its historical macro: evidence validation requires all class/difficulty values and recomputes the advertised macro. Preserve historical evaluator formulas.
- A matching declared protocol is insufficient: require actual full evaluation, checkpoint/config provenance and selection history. Keep external algorithm training conditions disclosed.
- No dataset downloads, remote Colab access, long training, AP improvement/SOTA claim, merge or push is part of this implementation.

### Rollback plan

Revert only these audit/evidence/protocol additions if necessary. Leave existing configs, notebook defaults, model/loss code and prior verification artifacts intact. Explicit existing reference-mode protocol commands remain available.

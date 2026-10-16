# CANUPO feasibility on available KITTI data

## Goal
Assess whether multiscale point geometry adds useful information for the existing lightweight KITTI detector, using real local scans and measured CPU costs.

## Constraints
Keep detector training and existing user changes untouched. Use the current 3712/3769 split, never fit on validation frames, and label box-derived supervision as weak labels rather than semantic ground truth. Keep generated data and third-party programs in ignored artifacts or /tmp.

## Known context
Local KITTI scans, calibration and 7481 object-label files are available under `Code/Kitti_Raw`. The existing BEV supports Rich8/Hist14. The AI_env Python environment contains NumPy, SciPy, scikit-learn and plotting packages. CloudCompare/CANUPO is absent from PATH; the author's standalone Linux package is being obtained.

## Risks
Box-contained points include some background; outside annotated boxes is not complete semantic background. Uniform samples underrepresent small objects. Camera annotation coverage and ignored classes must be handled. Sampled core-point timings are not full-cloud deployment latency. A geometry probe cannot establish detector AP improvement.

## Options
1. Run the original CANUPO command-line tools: direct algorithm evidence and reproducibility, but legacy binaries may need compatibility work.
2. Implement a descriptor-only CANUPO-inspired probe: easier feature ablations, but must be identified separately from the original classifier.
3. Retrain the detector immediately: gives AP evidence but premature before establishing geometric utility and cost.

## Recommendation
Attempt the original tools, and combine their descriptors with controlled lightweight feature probes. Start with a bounded, seeded sample across train/validation and report class/distance coverage and limitations explicitly.

## Acceptance criteria
- Exact selected frames, source hashes, scales, sampling, supervision and software provenance recorded.
- Geometry checked against known synthetic shapes and local box conversion.
- Train/tuning/validation sets disjoint; all fitting and threshold selection use train only.
- Compare basic point/pillar cues, single-scale geometry, multiscale geometry and their combination.
- Report held-out point metrics and object-point retention, including distance bins.
- Measure descriptor computation on real scans and distinguish sampled from full ROI timings.
- Produce a Vietnamese report and standalone plots, with no claim of new detector AP.

## Assumptions
The user authorizes a local feasibility experiment and necessary supporting scripts/downloads. Existing KITTI bounding boxes provide weak foreground supervision; no pointwise vegetation/ground labels are available.

## Plan
1. Inventory and provenance
   - Files: `artifacts/canupo_feasibility_20261016/manifest.json`.
   - Change: identify aligned training scans, labels, calibration, active split and available CANUPO executable.
   - Verify: file counts, split disjointness, binary execution and package hash.
2. Implement the bounded analysis
   - Files: `Code/tools/analysis/analyze_canupo_kitti.py`, focused geometry tests if needed.
   - Change: select seeded frames, establish weak labels, compute original descriptors or explicitly named fallback, compare feature probes.
   - Verify: synthetic geometry invariants, label geometry, no validation fitting, real-frame smoke run.
3. Run and inspect results
   - Files: ignored experiment arrays, metrics and runtime JSON/CSV.
   - Change: execute real-data experiment with train-only tuning and save reproducible outputs.
   - Verify: finite outputs, nonempty class/distance strata, recorded commands and timings.
4. Report and review
   - Files: `Code/docs/canupo_feasibility_vi.md`, plots under experiment artifacts.
   - Change: explain measured utility, risks, limitations and next-step recommendation.
   - Verify: cross-check reported numbers against JSON, inspect plots and run focused checks plus `git diff --check`.

## Risks & mitigations
Use FOV/ignore masks and document weak labels. Evaluate uniform samples separately from object-enriched diagnostic samples. Benchmark without claiming legacy CPU results represent optimized GPU latency. If original software fails, document the exact limitation and name any substitute accurately.

## Rollback plan
Only new analysis files and ignored artifacts are created; the detector and user edits are unaffected.

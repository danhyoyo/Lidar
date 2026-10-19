# OpenPCDet validation notebook

## Brainstorm
### Goal
Deliver one standalone Vietnamese notebook for the three audited KITTI checkpoints:
PointPillars, SECOND, PV-RCNN. Evaluate the full 3,769-frame validation split on
the user's GPU and save reference R11/R40 APBEV/AP3D and explicit latency reports.
### Constraints
No training, changes to existing detectors, or replacement of upstream evaluation
code. Preserve checkpoint-specific geometry and postprocessing. No local CUDA or
Torch is available for an actual model run in this editing environment.
### Known context
The split and checkpoint audits are in `paper/research/openpcdet_*_audit.json`.
The current BEV timing uses batch 1 FP32 and excludes ten warmup frames; Final3D
has no stage latency. OpenPCDet's standard timing includes final NMS and GT recall.
### Risks
Changing Colab CUDA/Torch versions can break compiled extensions. Wrong or partial
splits, partially loaded weights, GT work inside timing, and stale cached reports
could invalidate comparisons. Partial runs must be visibly identified.
### Options
1. Call upstream test.py: simple but insufficient timing and warmup control.
2. Standalone notebook embedding a reusable runner: explicit boundaries, tests,
   no dependency on a separately uploaded helper; modest additional maintenance.
### Recommendation
Option 2. Use an isolated dependency overlay retaining the user's installed CUDA
PyTorch. Build the pinned OpenPCDet extensions, audit the environment, then run
smoke and full evaluation in separate processes. Export KITTI predictions and
score the exported predictions with the unchanged pinned evaluator.
### Acceptance criteria
- A single downloaded notebook runs without the user's project checkout.
- Official checkpoint links and canonical split hashes are pinned.
- Original annotations retain all ignored classes and full-frame GT.
- GPU inference excludes GT recall; stage boundaries and caveats are explicit.
- Reports include mean/p50/p95/p99/FPS, per-frame times, AP per class/difficulty,
  config/checkpoint/split/source identities, GPU and dependency versions.
- Smoke and failures cannot appear as complete benchmark results.
- CPU tests cover split/label handling, output aggregation and cache identity;
  notebook validation and executable-cell syntax pass.

## Implementation plan
### Goal
Implement the notebook and its tested runner without changing model training.
### Assumptions
KITTI is already extracted, including original image_2, calib and label_2. The
target is a Linux CUDA GPU session with PyTorch, torchvision and nvcc installed;
Colab is the default path layout. Runtime installation is visible and logged.
### Plan
1. Add focused CPU contract tests.
   - Files: `Code/tests/test_openpcdet_benchmark.py`.
   - Verify: `python3 -m pytest Code/tests/test_openpcdet_benchmark.py -q` fails
     for the missing runner.
2. Implement runner, setup, data audit, profiling, AP and exports.
   - Files: `Code/tools/openpcdet_benchmark.py`.
   - Verify: focused tests, CLI --help, Python compilation.
3. Generate a standalone notebook and short guide.
   - Files: `Code/tools/generate_openpcdet_benchmark_notebook.py`,
     `Code/OpenPCDet_KITTI_Validation_Benchmark.ipynb`,
     `Code/docs/openpcdet_benchmark.md`, `Code/README.md`.
   - Verify: nbformat, compile/execute safe cells, embedded-runner equality,
     deterministic generation and focused tests.
4. Review requirements and document limits.
   - Verify: `git diff --check`, inspect changed files, run all new CPU checks.
### Risks & mitigations
Check CUDA and dependency compatibility before long inference. Keep evaluation
source hashes unchanged; record any mechanical extension compatibility edits.
Check every parameter loaded and require exact canonical split hashes. Run one
model per process and write explicit failure reports.
### Rollback plan
Remove only the new notebook, runner, generator, tests, guide and README link.
Datasets, checkpoints, existing notebooks and paper evidence remain untouched.

## Verification and review

- 15 focused CPU tests pass, covering canonical split rejection, original KITTI
  annotation semantics, PNG dimensions, staging with spaces in paths, GT-free
  inference metadata, percentiles/FPS, error/partial summary behavior, config
  validation, extension API edits and standalone notebook controls.
- nbformat validation, every code cell compilation, CLI help, Python compilation,
  deterministic generation/embedded-source equality and `git diff --check` pass.
- Runner split hashes were checked against the actual fetched 2020 upstream
  manifests: 3712/3769 IDs. Evaluator hashes match the pinned vendored sources.
- Review: no remaining Blocker/Major issue found in the CPU paths inspected.
  CUDA compilation, dependency resolution on the user's runtime, checkpoint
  loading and full inference remain unverified here: this editing interpreter
  has no Torch/CUDA. The notebook executes CUDA overlap and model smoke checks
  before the full benchmark and writes failure reports without fabricated AP.

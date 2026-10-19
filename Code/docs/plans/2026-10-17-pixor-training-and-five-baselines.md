# Five pretrained baselines and a standalone PIXOR training notebook

## Scope and decisions

- Extend the existing OpenPCDet notebook to PointPillars, SECOND, PointRCNN,
  Part-A2-Anchor and PV-RCNN. Preserve native configs, metrics and timing.
- Audit the two additional checkpoint headers before pinning metadata; never
  infer epoch/iteration values from the recipe alone.
- Create a standalone PIXOR reproduction notebook, training from scratch for
  50 epochs on the existing canonical 3712/3769 KITTI partition. No random split.
- PIXOR predicts Car BEV boxes only. Explicitly identify this as our reproduction
  and training recipe, not released original weights or a three-class 3D detector.
- Embed code, manifests and the existing pinned reference evaluator. Train-only
  label statistics, lazy point loading, optional local staging, resumable optimizer
  checkpoints, validation-based selection, R40/R11 BEV and explicit FP32 profiling.

## Implementation

1. Update `tools/openpcdet_benchmark.py`, its notebook generator, documentation
   and CPU tests for five models and the verified additional checkpoints.
2. Add `tools/pixor_training.py`: data audit/geometry/targets, full PIXOR backbone
   and head, loss, smoke, deterministic epoch seeds, train/resume and evaluation.
3. Add `tools/generate_pixor_training_notebook.py`, the generated standalone
   `PIXOR_KITTI_Train_50Epochs.ipynb`, documentation and focused regression tests.
4. Review split leakage, calibration/yaw/dimension transforms, target/decode
   inversion, empty scenes, resume identity and partial-run reporting.
5. Run relevant pytest suites, notebook schema/compilation/embedded-source checks,
   deterministic regeneration and `git diff --check`. GPU training and actual
   AP/FPS require the user's dataset and CUDA runtime; report this limit honestly.

## Verification completed

- Added headers read from actual official download byte ranges: PointRCNN
  epoch 77 / iteration 17864; Part-A2-Anchor epoch 80 / iteration 12400. Recorded
  prefix SHA-256 and the limits of split provenance in the existing audit JSON.
- 28 focused tests passed using PyTorch 2.10.0+cpu installed only under `/tmp`.
  Includes full-network forward/backward on small tensors, actual 800x700 layout
  on Torch meta tensors, calibration and dimension round trips, rotated NMS,
  train-only target/prior data staging and real optimizer/checkpoint resume on a
  small synthetic CPU network after interruption during validation.
- Both notebooks validate as nbformat 4.5; all eight code cells in each compile;
  deterministic generator output and verbatim embedded helper/source checks pass.
- GPU smoke, full checkpoint loading, real KITTI AP/FPS and 50-epoch PIXOR training
  were not run in this workspace. The delivered notebooks perform those actions
  on the user's CUDA runtime and dataset, with no fabricated benchmark values.

## Follow-up: readable notebook installation cells

Replace whole-file `repr()` payloads with readable raw multiline literals. Keep
embedded source in a separate, initially collapsed cell; make setup a short,
visible cell. Apply to both notebook generators, execute preparation cells on
CPU to verify every written byte, and validate schema/generator parity. Keep
the standalone workflow, model settings, training recipe and manifests intact.

Completed: setup cells are now 10 lines (OpenPCDet) and 6 lines (PIXOR), with a
separate collapsed, readable source cell. Five focused notebook checks pass,
including executing preparation cells without GPU/network and verifying every
written source/manifest byte. Both nine-code-cell notebooks pass schema,
compilation, deterministic generation and clean-output checks.

## Follow-up: accept the Robust notebook's archived KITTI layout

Root cause: the new notebooks validate raw training directories immediately after
Drive mount, while the existing Robust notebook extracts separate RAR/ZIP/TAR
components from `KITTI_DATASET_ZIP` first and can use the existing per-frame
image-size metadata instead of PNGs.

1. Add a shared standalone data resolver/extractor for raw or archived inputs,
   nested/flat component layouts, exact selected IDs, local disk preflight,
   safe atomic extraction, reusable files and actionable missing-component errors.
2. Embed that helper, canonical splits and the same image metadata used by Robust.
   Mount Drive, materialize helpers, resolve/extract data, update settings, then
   install the GPU environment. Avoid a second local copy of extracted data.
3. Make both runners accept validated per-frame image metadata as well as PNGs;
   preserve shape order, full GT and native model preprocessing.
4. Reproduce the missing-folder error with small real archives; test ZIP/TAR/RAR,
   test-split exclusion, selected IDs, nested paths, reruns, missing data, metadata
   coverage and notebook configuration-to-resolver execution without CUDA.

Completed verification: 43 focused tests pass with CPU Torch. Real RAR 5 solid
and multi-volume fixtures were created and extracted with official RAR/UnRAR
7.23 tools under `/tmp`; ZIP/TAR, ID filtering, cache reuse, empty label files,
missing components, traversal rejection and disk preflight also pass. Both actual
notebook config/source/data cells execute in isolated Python processes with no
project checkout or CUDA, correctly rewriting settings and disabling duplicate
local copying. Both ten-code-cell notebooks pass nbformat schema, deterministic
generation, compilation and whitespace checks. User Drive archives and GPU model
inference/training remain external runtime verification, not claimed as executed.

## Follow-up: KITTI imports fail on unrelated Argoverse dependencies

The user's captured child traceback identifies `ModuleNotFoundError: av2` in
`pcdet.datasets.argo2.argo2_dataset`, imported eagerly by the upstream dataset
registry while loading KITTI's evaluator. The notebook's inherited subprocess
streams also hide that traceback in Colab, leaving only CalledProcessError.

1. Reproduce the missing optional dependency in a small standalone package.
   Restrict this KITTI-only checkout's registry to DatasetTemplate/KittiDataset,
   keeping the native KITTI class, dataloader and evaluator intact. Record the
   registry's original/modified hashes and retain the manifest on reruns.
2. Apply the import compatibility step during setup and runtime activation so
   an existing built environment can recover without rebuilding extensions.
3. Stream combined child output through notebook print, retain command logs and
   error tails, including setup and gpu-check. Keep pip-freeze output clean.
4. Regenerate embedded runners, test actual subprocess failures/import recovery,
   validate both notebooks, and report separately what still needs a CUDA run.

Verification: 45 focused CPU tests pass, including reproducing an actual child
process import failure without av2, successful KITTI import after the patch,
preserved class/evaluator bytes, idempotent hash manifest and notebook-visible
stderr/log tails. Also fetched the pinned upstream registry and reproduced its
av2 failure in an isolated package with stub dataset classes and real CPU Torch;
after patching, its native DataLoader iterates successfully. Sampler/dataloader
ASTs and pinned evaluator hashes are unchanged. Real CUDA kernel execution still
requires the user's Colab runtime; no GPU success is claimed from CPU tests.

### Goal
Build an opt-in hybrid augmentation on `feature/hybrid-gt-augmentation`, combining the existing ordered pipeline and train provenance checks, batched/JIT geometry inspired by MMDetection3D, and configurable PCU placement/physics/visibility checks.

### Assumptions
- Keep existing `standard`, `openpcdet_global` and `openpcdet_gt` behavior unchanged.
- Reuse the existing train-only JSON + object-bin database. Boxes are `[class,h,w,l,x,y,z_bottom,yaw]`; database points are centered on geometric center without removing yaw.
- Default physics is conservative: source-relative placement, ground support and density adjustment; shadow/radiometric calibration are opt-in heuristics.
- No full KITTI training data or GPU benchmark is available locally. Report synthetic CPU timings and smoke validation, without claiming AP improvements.

### Plan
1. Geometry and cache foundation.
   - Files: `detector/core/datasets/augmentor/hybrid_geometry.py`, `hybrid_sampler.py`, `tests/test_hybrid_augmentation.py`.
   - Change: Implement NumPy/Numba SAT collision with broad-phase rejection, union point membership, ray-box kernels, and a bounded per-worker LRU object cache.
   - Verify: randomized parity against Shapely and existing membership geometry, boundary cases, cache limits/invalidation and pickle behavior.
2. Transactional sampling and optional physics.
   - Files: `hybrid_sampler.py`, `hybrid_physics.py`, augmentation tests.
   - Change: Support source/source-relative placement with bounded retries, ground/static/LOS checks, range density adjustment, optional intensity/shadow, visibility protection and diagnostic provenance.
   - Change: Stage removal masks and accepted object chunks; concatenate scene points once after all accepted insertions; rejected candidates never mutate the scene.
   - Verify: round-trip coordinates, empty scenes, source-mode geometry parity, rejection reasons, original/inserted visibility, deterministic RNG, malformed configuration, worker isolation.
3. Pipeline, notebook and visualization integration.
   - Files: `data_augmentor.py`, `configs/augmentation/hybrid_gt.json`, `common.py`, `notebook_config.py`, standard notebook, `tools/visualization/visualize_gt_sampling.py`, `README.md`.
   - Change: Add `hybrid_gt` selection, GT database preparation and a preview cell with accurate inserted-point coloring even when shadows remove points.
   - Verify: actual configuration cell, real model forward/backward, one-epoch trainer smoke, preview rendering and existing regression suites.
4. Benchmark and finish.
   - Files: `tools/benchmarks/benchmark_gt_samplers.py`, benchmark report.
   - Change: Compare the current sampler, PCU sampler and hybrid sampler on matched synthetic scenes; distinguish common-feature throughput from extra physics work. MMDetection3D supplies the batched geometry design reference, but its full runtime is not benchmarked.
   - Verify: CPU benchmark with JIT warmup, accepted-object counts, allocation-aware results; review diff, commit and push only the isolated branch changes.

### Risks & mitigations
- Physics rules may reject valid objects or alter visibility: independent flags, bounded attempts, rejection diagnostics, lightweight defaults and explicit experimental status.
- Numba may be unavailable: NumPy fallback with parity tests; explicit backend reports/errors.
- Worker RAM growth: bounded object cache, clear warmed caches and sampling cursor state on serialization.
- Box origins differ across frameworks: retain the repo's box convention and test yaw/Z round trips.
- Sampling behavior and collision order can affect acceptance: preserve the old path; benchmark exact source mode and report accepted counts.

### Rollback plan
Choose `AUGMENTATION="openpcdet_gt"` on the new branch, or return to `feature/mobilepixornext-improvements`. No model/loss changes or database migration are required.

### Completion evidence
- Implemented geometry, bounded worker cache, transactional sampler, configurable physics, profile/resolver, notebook preparation/controls/preview, CLI preview, benchmark and usage documentation.
- Final verification: 267 tests passed across hybrid/legacy augmentation, small-object assignment, notebook configuration, run naming, KITTI compatibility, backbone, BEV encodings and model/config registries. Includes a real model backward and a one-epoch CPU trainer checkpoint, plus a spawned two-worker loader after parent cache warmup.
- Multi-worker tests require Unix sockets; the sandbox rejected PyTorch tensor-sharing sockets. The identical checks passed outside the sandbox. The only final warning is an existing TorchScript/Python 3.14 compatibility warning.
- `compileall` and `git diff --check` passed. Synthetic benchmark raw results and interpretation are stored alongside this plan.
- Review: resolved process ownership/cache isolation and shared-boundary visibility protection. No unresolved blocker or major implementation issue identified. Physical heuristics and real KITTI accuracy remain experimental, as documented.
- Work was isolated in `/tmp/lidar-hybrid-augmentation`; unrelated loss/trainer changes in the user's original workspace were excluded.

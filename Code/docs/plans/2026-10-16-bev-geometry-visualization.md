# BEV neighborhood and geometry visualization

## Goal
Generate real KITTI BEV tensors and inspect what Hist14 gains from neighborhood support and multiscale 3D covariance. Deliver full channel maps, object zooms, a local viewer, and quantitative comparisons.

## Assumptions
- `Rich14` means the historical label for Hist14; no independent rich14 runtime encoder exists. Include the actual Rich8 as an additional baseline.
- Build an experimental analysis encoder, without changing detector configuration/training behavior.
- Compare Hist14 (14ch), Hist14+support (18ch), Hist14+support+geometry (22ch), using two XY windows near 1m and 2m.
- Compute 3D covariance from pooled raw-point moments. XY column neighborhoods differ from CANUPO's point-centered 3D spheres and must be identified as such.

## Constraints
Use the existing frame split, samples, and weak labels from the completed CANUPO study. Use no bounding-box labels to construct encoder features. Keep Hist14 channels bitwise identical and empty cells zero. Select display normalization from training frames only.

## Risks & mitigations
RGB composites cannot show 22 channels at once: supply full individual maps with physical legends. Ground-truth boxes are overlays only. A visually salient channel is not evidence of detection AP. Moment cancellation is handled with float64 and centered coordinates. Sparse/degenerate neighborhoods have explicit validity masks; no fabricated geometry.

## Plan
1. Define and test experimental encoding.
   - Files: `Code/tools/analysis/bev_geometry_maps.py`, `Code/tests/test_bev_geometry_maps.py`.
   - Change: pooled count/first/second moments, 3x3 covariance eigenspectra, 18/22-channel tensors.
   - Verify: brute-force neighborhood equivalence, line/plane/volume behavior, sparse masks, empty input, translation and Hist14 preservation.
2. Run paired feature analysis.
   - Files: `Code/tools/visualization/compare_bev_geometry_maps.py`; ignored artifacts.
   - Change: same 144 frames and sampled cores; compare Rich8/Hist14/18ch/22ch with a common probe; record timing, coverage, channel distributions and correlations.
   - Verify: all fitting uses fit frames; tune thresholds on train only; primary validation remains random frames and uniform cores.
3. Render maps and viewer.
   - Files: ignored PNG/NPZ/HTML assets under `artifacts/bev_geometry_maps_20261016/`.
   - Change: overview, all channels, and near/far object crops on four selected validation frames. Use fixed training-derived display scales and label GT overlays explicitly.
   - Verify: inspect rendered maps, shared coordinate alignment and asset links.
4. Report and verify.
   - Files: `Code/docs/bev_geometry_visualization_vi.md`.
   - Change: explain measured information gain, approximation limits, costs and integration recommendation.
   - Verify: focused tests, raw-metric recomputation, report numbers, tensor invariants, and `git diff --check`.

## Rollback plan
Only add analysis/visualization code, tests, documentation and ignored artifacts. Preserve existing user changes and detector behavior.

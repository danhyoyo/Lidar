### Goal
Speed up pillar_rich_gate CPU preparation, including rich8, without changing feature semantics or checkpoint identity. Measure matched latency on real KITTI frames.

### Assumptions
- Fuse finite/ROI filtering for float32/float64 inputs with ordinary JSON bounds; retain NumPy filtering for other dtypes/scalar types. Keep NumPy float32 XY floor-division, intensity and normalized-height arithmetic as the compatibility boundary.
- Replace sorting on normal-size grids with a bounded dense lookup. Fuse feature/statistic accumulation in a serial Numba kernel without fastmath, retaining float64 sums and float32 outputs.
- Auto-select Numba when installed; retain an explicit NumPy reference and fallback when Numba is unavailable. No model architecture changes.

### Plan
1. Add numerical and fallback regression tests.
   - Files: tests/test_pillar_preparation.py.
   - Change: Compare all arrays exactly over boundaries, dtypes, nonfinite/empty/singleton/dense inputs and different encoders/options; exercise sparse-grid lookup fallback.
   - Verify: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_pillar_preparation.py -q (red before implementation).
2. Implement the CPU preparation backend.
   - Files: detector/core/datasets/utils_1/pillar_backend.py, detector/core/datasets/utils_1/pillar_numba.py.
   - Change: Preserve the original NumPy implementation and introduce optional fused preparation with bounded grouping workspace.
   - Verify: Run preparation and all existing pillar regression tests, including CUDA/BF16, DataLoader and CLI integration.
3. Benchmark and document measured behavior.
   - Files: tools/benchmarks/benchmark_pillar_preparation.py, docs/learned_pillar_encoder.md.
   - Change: Benchmark NumPy and Numba preparation on multiple real frames and matched GPU input-to-output latency; record exact feature/output parity, JIT warmup and scope.
   - Verify: Benchmark CPU and CUDA/BF16, inspect reports; full pytest suite and compare known failures.

### Risks & mitigations
- Float32 boundaries/reduction changes: preserve vectorized boundary arithmetic and strict accumulator order; require exact array parity before adopting.
- JIT startup: cache kernels and report first-call separately from steady-state.
- Large sparse grids: guard dense lookup memory and fall back to unique grouping.
- Preserve unrelated GT database, robust notebook/scripts, raw data and checkpoints.

### Rollback plan
Select cpu_backend="numpy" to use the unchanged reference implementation, or revert only these optimization files. No checkpoint migration or retraining is required if parity holds.

### Completed verification
- 32 preparation cases and 132 focused tests passed.
- Bitwise input parity on 100 real frames (11,861,298 points); exact BF16 output parity on nine timed frames.
- CPU prep 8.685 to 3.857 ms (2.252x); prep + H2D + forward 16.365 to 11.245 ms (31.29% lower latency).
- Full suite 1761 passed, 26 failed, 19 skipped; 19 known failures and seven quota failures, all seven passed on rerun on /home. No new regression observed.
- Reports: artifacts/pillar_preparation_speed/report.md and verification.json. No AP experiment.

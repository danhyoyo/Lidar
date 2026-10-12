# Notebook subprocess output

### Goal

Show command startup, child stdout/stderr and failures directly in notebook output while the command runs.

### Assumptions

The current runner inherits file descriptors, which may bypass the notebook's Python output capture. The trainer already flushes epoch summaries. Numerical training behavior and run configuration must remain unchanged.

### Plan

1. Reproduce missing Python-level capture and test live delivery.
   - Files: `tests/test_notebook_command_output.py`.
   - Change: Execute the actual notebook function with real subprocesses. Cover stdout/stderr, an output acknowledgement before child exit, cwd/arguments, nonzero exits, SIGKILL and interruption cleanup.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_notebook_command_output.py -q --tb=short` fails before the fix.
2. Stream subprocess output through the kernel.
   - Files: `3D_Lidar_Object_Detection_Notebook_standard.ipynb`.
   - Change: Use an argv-based Popen, merged output pipe, unbuffered Python child environment, flushed notebook prints and bounded failure-tail retention. Terminate the direct child when the caller interrupts.
   - Verify: The reproduction tests pass, including the acknowledgement handshake before subprocess exit.
3. Document recovery and verify compatibility.
   - Files: `docs/plans/lightweight_lidar_backbone_2026/notebook_runbook.md`.
   - Change: Explain replacing the runner in an existing kernel; preserve epoch-level trainer logging and existing file logs.
   - Verify: Run the command-output, detector-import, notebook-control, backbone-config and BEV-schema tests; validate notebook JSON/schema, code syntax, clean outputs and `git diff --check`.

### Risks & mitigations

Child output without newline remains pending until newline or EOF. Python children use PYTHONUNBUFFERED=1. Retain at most 100 lines for exception diagnostics rather than buffering complete training output. Streaming does not create new trainer messages or cure SIGKILL. Local verification uses real CPU subprocesses; Colab GPU execution is separate.

### Rollback plan

Revert only this runner change and its documentation/tests. Preserve existing configs, datasets, checkpoints and unrelated working-tree files.

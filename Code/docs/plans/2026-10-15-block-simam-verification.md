# Block SimAM: implementation and verification

Implemented selectable parameter-free SimAM inside MobilePixorNeXt blocks.
The insertion is after pointwise projection BN, before existing LayerScale and
shortcut addition. Stage-level attention and Focal remain independent controls.

## Usage

In the configuration cell of either Standard or Robust:

```python
PRESET = "custom"
BLOCK_SIMAM = True
BLOCK_SIMAM_STAGES = ["c4", "c5"]
BLOCK_SIMAM_LAMBDA = 0.0001
LOCAL_ATTENTION = "none"  # block-only SimAM; Focal is controlled independently
```

The stage list defaults to C4+C5; the enable flag is now True at the user's
subsequent request. After-stage local attention remains off by default. Select
`["c5"]`, `["c3"]`, any pair or all three stages.
Set the flag False or use an empty list to disable the feature. With depths
3/4/2, C4+C5 enables six cores, C5 enables two and all stages enable nine.

JSON model settings use `block_simam_stages` and optional positive
`block_simam_lambda` (default 0.0001). Omit lambda when disabled. Stage names are
case-insensitive, canonicalized into depth order, and duplicates are rejected.

Enabled block attention uses feature semantic metadata version 3; disabled
settings preserve existing version 1/2 metadata. Different stages or lambda
produce distinct run/checkpoint identities despite identical parameter keys.

## Verification

Interpreter: `/home/duyennh/miniconda3/envs/AI_env/bin/python` (PyTorch 2.11.0).
Commands run from `Code/`, with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid an
unrelated installed ROS pytest plugin missing `osrf_pycommon`, and with
`PYTHONPATH=detector:tools/kitti_training_pipeline:.`, `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`.

Main focused suite: **429 passed, 10 skipped**, 20.19 seconds:

```sh
/home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -q \
  tests/test_block_simam_config.py tests/test_block_simam.py \
  tests/test_block_simam_notebook.py tests/test_attention_stage_routing.py \
  tests/test_attention_stage_training.py tests/test_attention_stage_checkpoint.py \
  tests/test_attention_stage_compatibility.py tests/test_attention_stage_notebook.py \
  tests/test_backbone_feature_config.py tests/test_backbone_attention_integration.py \
  tests/test_light_attention.py tests/test_lightweight_detector_budget.py \
  tests/test_mobilepixornext_backbone.py tests/test_under1m_notebook_controls.py \
  tests/test_robust_notebook_setup.py tests/test_grouped_checkpoint.py \
  tests/test_notebook_detector_imports.py --maxfail=5
```

Consumer/workflow regression suite: **200 passed, 3 skipped**, 20.61 seconds:

```sh
/home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -q \
  tests/test_attention_placement_visualization.py tests/test_attention_presets.py \
  tests/test_backbone_diagram.py tests/test_under1m_notebook_config.py \
  tests/test_unified_notebook_workflow.py tests/test_simple_notebook_workflow.py \
  tests/test_model_registry.py --maxfail=3
```

Total across these two suites: **629 passed, 13 skipped**. CUDA is unavailable to
this process; FP16/BF16 CUDA checks are skipped. TorchScript numerical checks
pass, with PyTorch's existing deprecation warning for Python 3.14.

After the user's follow-up to enable C4+C5 by default, reran
`test_block_simam_notebook.py`, `test_under1m_notebook_controls.py`,
`test_attention_stage_notebook.py` and `test_robust_notebook_setup.py`:
**91 passed** in 2.66 seconds. The actual notebook defaults now build six block
cores and an Identity after-stage local adapter. Focal C4 remains independently
enabled as before; `git diff --check` passes.

New tests establish:

- All eight stage subsets route exactly once through every selected block.
- Numerical block output matches SimAM after BN and before scaling/addition;
  zero LayerScale preserves the shortcut exactly.
- Disabled features preserve seeded state/outputs; enabling block SimAM adds no
  learned parameters and retains state keys.
- A real grouped OGA/IQA loss backward/update and full checkpoint roundtrip
  succeed; changed stages/lambda reject resume and evaluation before mutation.
- C4+C5/all-stage full KITTI meta shapes and TorchScript pass.
- Focal, after-stage SimAM and RepConv deployment coexist correctly.
- Both actual notebook cells save and summarize the selected stages; empty
  stages or a disabled flag turn it off; trained snapshots are preserved.
- Robust setup detects stale block source in addition to stale resolver,
  backbone and notebook helpers.

An independent meta-device profile using the user's existing Hist14/shared-head
recipe reports **747,329 detector parameters**, unchanged from its no-attention
baseline, and output shape `[1,32,200,176]`. The C4+C5 summary lists four blocks
at C4 and two at C5. Raw profile:
`artifacts/block_simam_20261015/c4c5_model_audit.json` at repository root.

`git diff --check` passes. Comparing notebook JSON against the initial index
confirms only setup/config cells changed; evaluation and dataset cells were
preserved. No existing training experiment JSON or user-staged implementation
was replaced. No dataset training, new AP evaluation or GPU latency benchmark
was run for block SimAM.

## Review

- Blockers: none found in the reviewed change.
- Majors: none found in the reviewed change.
- Minors: GPU execution/performance remains unverified in this environment.
- Nits: none requiring follow-up.
- Overall: the requested runtime/configuration controls and their compatibility
  checks are implemented. Train a new run for mAP and benchmark on the target GPU.

## Rollback

Set `BLOCK_SIMAM=False` / `block_simam_stages=[]` to recover the original topology.
When changing enabled stages or lambda, use a new run rather than resuming an
existing trained snapshot.

# Block SimAM implementation

### Goal
Add selectable SimAM inside MobilePixorNeXt blocks, ready for a C4+C5 ablation.

### Assumptions
- Use the existing SimAM core after projection BN; do not add a second residual beta.
- Existing stage attention retains its behavior. The user subsequently requested enabling block SimAM at C4+C5 by default in both notebooks, with after-stage SimAM off.
- Verification interpreter: `/home/duyennh/miniconda3/envs/AI_env/bin/python` (PyTorch 2.11.0).

### Plan
1. Add focused failing behavior tests.
   - Files: `Code/tests/test_block_simam_config.py`, `Code/tests/test_block_simam.py`, `Code/tests/test_block_simam_notebook.py`.
   - Change: cover validation/canonicalization, residual placement, stage selection, gradients, semantic identity and real notebook controls.
   - Verify: run the new tests and confirm failures arise from missing functionality.
2. Implement resolver and runtime wiring.
   - Files: `Code/detector/core/backbone_config.py`, `Code/detector/core/models/backbones/mobilepixornext_blocks.py`, `Code/detector/core/models/backbones/mobilepixornext.py`.
   - Change: validate stages/lambda, preserve disabled legacy metadata, create a core inside each selected block.
   - Verify: new config/runtime tests plus existing feature, attention-routing and compatibility tests.
3. Integrate run naming and notebook configuration.
   - Files: `Code/tools/kitti_training_pipeline/common.py`, `Code/tools/kitti_training_pipeline/notebook_config.py`, `Code/tools/generate_robust_notebook.py`, both Standard and Robust notebooks.
   - Change: readable distinct names; toggle/stage controls; summaries; inherited-setting cleanup; source compatibility checks.
   - Verify: new notebook/checkpoint tests and existing notebook, checkpoint and generated-notebook checks.
4. Document use, review and finish.
   - Files: `Code/tools/kitti_training_pipeline/README.md`, verification notes alongside this plan.
   - Change: document C4+C5, C5-only, all stages, disabling and fresh-run requirements.
   - Verify: focused integration suite, TorchScript checks, `git diff --check`, and check scope against initial user edits.

### Risks & mitigations
- Preserve old hashes by omitting disabled block settings from canonical serialization and metadata.
- Protect parameter-free topology changes with explicit semantic identity and mismatch tests.
- Use synthetic tensors and full-resolution meta checks; do not claim mAP or GPU speed improvements without training/benchmarking.

### Rollback plan
- Set `BLOCK_SIMAM=False` / `block_simam_stages=[]` to recover the existing architecture.
- Keep new experiments under distinct run names; do not overwrite or resume incompatible existing runs.

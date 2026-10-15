# Selectable SimAM inside MobilePixorNeXt blocks

### Goal
- Allow SimAM inside every block of selected C3/C4/C5 stages, with C4+C5 as the suggested first experiment.

### Constraints
- Preserve existing stage attention, checkpoint identities, state keys and initialization when block SimAM is disabled.
- Keep Focal, heads, losses and training schedules independently configurable.
- Do not change existing user edits in training, postprocessing or evaluation.

### Known context
- Stage depths are configurable; the current experiment uses 3/4/2 blocks at C3/C4/C5.
- Blocks already contain residual addition and per-channel LayerScale.
- Stage SimAM uses a separate learned residual adapter; block SimAM can use the parameter-free core directly.

### Risks
- Parameter-free attention has identical state keys across placements, so semantic identity must distinguish enabled stages and lambda.
- Notebook custom merges must remove inherited inactive settings without hiding explicitly invalid settings.
- Shapes and gradients require numerical checks; CPU checks cannot establish CUDA latency or AMP performance.

### Options
1. Add a separate `block_simam_stages` list and `block_simam_lambda`, independent of stage attention. Enables simple stage selection and preserves all existing behavior; modest resolver/notebook integration.
2. Replace the existing local attention configuration with a generalized placement schema. More uniform, but changes legacy serialization and adds migration risk.
3. Wrap each whole block externally. Easy wiring, but changes the shortcut output and differs from the agreed insertion inside the residual branch.

### Recommendation
- Use option 1. An empty list disables block SimAM; active stages are canonicalized into C3/C4/C5 order.
- Insert `SimAMCore` after projection BN and before LayerScale and residual addition.
- Notebook controls: `BLOCK_SIMAM=True`, `BLOCK_SIMAM_STAGES=["c4", "c5"]`, `BLOCK_SIMAM_LAMBDA=0.0001`. The user subsequently requested enabling C4+C5 by default; set the flag False to disable it. The after-stage adapter remains off by default.
- Stage and block attention may be combined explicitly. For a block-only ablation set `LOCAL_ATTENTION="none"`; Focal remains a separate control.

### Acceptance criteria
- Select any subset of C3/C4/C5; every block in selected stages executes SimAM exactly once; no block in other stages does.
- No extra learned parameters; shortcut identity holds with zero LayerScale; disabled state/output/identity match existing defaults.
- Forward/backward and a real grouped-loss optimizer step remain finite; full KITTI shapes pass a meta-device audit.
- Different stages/lambda have different run/checkpoint identities and incompatible resumes/evaluations are rejected.
- Both notebooks expose controls, print actual stages/block counts and persist canonical configs; toggling off clears inherited settings.
- Invalid stages, duplicate stages, invalid lambda and unsupported backbones fail explicitly.

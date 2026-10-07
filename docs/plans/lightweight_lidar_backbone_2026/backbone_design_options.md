# Backbone design options beyond the initial candidate

Date: 2026-10-06. Status: decision background. Focal context is now part of the accepted main specification, and ECA/SimAM residual adapters are configurable options. See [specification.md](specification.md), [attention_specification.md](attention_specification.md) and [implementation_plan.md](implementation_plan.md) for the authoritative design. Detail/fusion options remain explicit follow-ups; dense stage redesign remains backlog. None has been production-implemented or accuracy-validated by these documents.

### Goal

Develop an architectural improvement beyond adding depth and widening the output of MobilePixorNeXt. Prioritize GPU KITTI accuracy while keeping backbone including neck below 1M and preserving the grouped-head/loss interface.

### Constraints

- No distillation; Jetson/TensorRT optimization is deferred.
- Preserve output stride 4 and 32 channels, global class mapping and OGA/IQA contracts.
- Limited ablation time: choose one new mechanism for the first enhanced candidate, rather than making every option mandatory.
- Complete the 3D prediction/evaluation milestone before any 3D accuracy/SOTA claim.

### Known context

- The accepted candidate keeps the existing DW7 inverted blocks, legacy stem and SG-FPN hierarchy. It removes LiteMLA, adds one stride-4 block, and raises neck output to 32: projected backbone including neck 635,408 parameters.
- The current SG-FPN projects the stride-4 lateral and top-down path to 24 channels, then applies its final 24→32 convolution. Output width 32 alone does not remove this internal compression.
- The stride-8 feature is 96 channels. It is the natural context insertion point previously used by LiteMLA, and is cheaper to process spatially than stride 4.
- The stride-2 stem feature is currently not connected directly to the neck.

### Risks

- Published sparse/pillar results do not prove that a new dense BEV adaptation improves KITTI AP in this repository.
- Added gates, dilation, feature branches and wider fusion increase compute/memory; parameter savings are not measured GPU latency savings.
- An empty BEV cell can be a valid object center. Point support must never hard-mask detection output.
- More architectural changes with only one long run reduce the ability to attribute improvements or regressions.

### Options

1. **Feature-based multiscale focal context at stride 8**
   - Summary: replace the removed LiteMLA hook with residual focal modulation. Project 96→132 into a 64-channel query, a 64-channel context seed and four gate channels. Generate three sequential depthwise 3×3 context levels with dilation 1/2/3 plus global pooled context; fuse them with normalized gates, apply 64→64 projection and Hadamard modulation, then 64→96 projection/residual.
   - Pros / cons: explicit local/global context with a small parameter budget; repeated dilation and global pooling may blur or bias features and need numerical/accuracy verification. Do not advertise sparse execution savings for this dense implementation.
   - Complexity / risk: moderate. A particular bias-free prototype with pre/output BN and 96-channel LayerScale has a theoretical 25,120 learned parameters; candidate backbone estimate becomes 660,528. Extra geometry-conditioning adapters are not included.
   - Evidence: [SFMNet](https://arxiv.org/html/2503.12093v1) studies focal context in sparse 3D detection; [PillarNeXt](https://arxiv.org/abs/2305.04925) motivates larger receptive fields. The exact dense module above is a proposed adaptation, not either paper's drop-in architecture.

2. **Early detail path from stem to the stride-4 neck**
   - Summary: add a small stride-2→stride-4 branch using depthwise downsampling plus pointwise projection, fused residually into the final neck level. Initialize its residual scale near zero. Keep output stride 4.
   - Pros / cons: provides a shorter path for early point/height features and could help Pedestrian/Cyclist. It does not increase prediction-grid resolution or restore information already lost in rasterization. Small-object gains are a hypothesis.
   - Complexity / risk: low to moderate; recount the exact chosen projection/normalization. Spatial alignment, downsampling and fusion require asymmetric-grid fixtures and a trained comparison.

3. **Widen final fusion inside SG-FPN from 24 to 32**
   - Summary: set the stride-4 lateral, top-down projection and final fusion/gate channels to 32 before the final 32-channel output projection. Make this an explicit neck setting rather than pretending that changing only output width solves the bottleneck.
   - Pros / cons: preserves more fused capacity with a small parameter increase and avoids another data path; more stride-4 computation may dominate its practical cost. There is no evidence yet that 24 channels are the cause of lower AP.
   - Complexity / risk: low; changes checkpoint shapes, so strict resume requires matching configuration. Recount both parameters and full-resolution compute.

4. **Stage-level feature reuse with one-shot aggregation**
   - Summary: retain selected intermediate features within a stage, concatenate once and project back to its stage width, inspired by [Dense Backbone](https://arxiv.org/html/2508.00744v1).
   - Pros / cons: reuses local/intermediate representations rather than only the final residual output. Concatenation adds memory traffic; published lightweight parameter counts do not guarantee faster execution.
   - Complexity / risk: moderate to high. Refactors core stage blocks and requires explicit growth/projection budget and stronger legacy isolation. Lower first-run priority than a context adapter.

### Recommendation

Option 1 is selected for the **main research candidate**, retaining the no-context candidate as a functional/reference configuration. Use feature-only focal gates initially. ECA is the first optional local adapter; SimAM is a separate comparison. Their formulas/wrapper budgets are defined in the attention specification. Support/height-conditioned gates remain subsequent research with separate schema/pooling semantics and parameters outside the 25,120 estimate.

Prioritize options 2 or 3 next if class-specific error analysis indicates small-object/detail loss or insufficient fusion capacity. Do not add both automatically without that evidence. If only one long training run is possible, train the focal-context candidate after smoke checks and the required 3D milestone; treat attribution and SOTA conclusions as preliminary without a matched trained baseline.

The macro hierarchy 32/48/96/128 can remain; the proposed architectural contribution is the context operator and, in a later option, detail/fusion path. Starting from a different backbone family is not required to obtain a meaningful architectural change.

### Acceptance criteria

- A separately named config records context type, bottleneck, dilation levels and initialization; legacy/no-context modes retain current behavior.
- The context input/output is `[B,96,H/8,W/8]`; grouped-head input remains `[B,32,H/4,W/4]`.
- Finite forward/backward on empty/sparse/dense synthetic input, proper residual initialization and dtype/device behavior under supported AMP.
- Actual instantiated backbone including neck <1M, with parameter/MAC/memory reports; the 660,528 figure remains an estimate until implementation.
- All existing grouped OGA/IQA, decode and resume acceptance checks pass with the new context config.
- Trained, matched-protocol class-wise AP evidence is required to call the adaptation more accurate; measured GPU timing is required to call it faster.

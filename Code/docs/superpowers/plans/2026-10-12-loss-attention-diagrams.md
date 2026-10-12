### Goal

Draw source-verified OGA, Q-OGA, Focal Context and SimAM diagrams, with separate
Gaussian focal-loss and adaptive task-weighting detail figures in the existing style.

### Assumptions

- Use the current Hist14/Focal/SimAM/FPN32/shared-head example for block dimensions
  and the configured OGA/IQA example. Q-OGA is a separately labeled alternative.
- Depict Gaussian BEV losses. Q-OGA variants explicitly distinguish the legacy
  MGIoU proxy from exact rotated IoU and optional zero-based epoch curriculum.
- Preserve model, loss and training code. Reuse the existing Scene renderer and
  PlotNeuralNet primitives for editable SVG/TikZ and vector PDF exports.

### Plan

1. Inspect loss/backbone implementations and integration points.
   - Files: `detector/core/losses/strategies/{oga,q_oga}.py`, geometry, quality,
     focal and weighting helpers; `models/backbones/{focal_context,light_attention}.py`.
   - Change: record formulas and sequential/branch structure from source.
   - Verify: compare each label with executed production functions, shapes,
     parameter counts and detached quality targets.
2. Build and export the six diagrams.
   - Files: `tools/visualization/draw_loss_attention_diagrams.py`,
     `diagrams/mobilepixornext/loss_attention/`.
   - Change: draw operations, masks, residual paths and task/group aggregation;
     export PNG/SVG/PDF/TikZ plus an HTML gallery and source audit JSON.
   - Verify: run `python tools/visualization/draw_loss_attention_diagrams.py`;
     renderer rejects overlapping/out-of-canvas text, audit validates numerics.
3. Inspect and deliver the artifacts.
   - Files: the generated figures, PDF atlas and README.
   - Change: document default/optional branches and precision details.
   - Verify: inspect all PNGs; parse SVGs, extract PDF text/check vector content;
     rerun the relevant existing loss/attention tests.

### Risks & mitigations

Focal loss and Focal Context are different operations: name both explicitly.
Sequential dilated convolutions and global pooling of the last level must be
shown accurately. SimAM core has zero parameters but its residual adapter has
one. Geometry is derived from prediction branches; it is not an extra head.

### Rollback plan

Remove only the new generator and figure folder; production behavior is unchanged.

### Verification results

- Generator completed six PNG/SVG/vector PDF/PlotNeuralNet TikZ exports and a
  six-page vector PDF atlas. All text bounds and pairwise overlaps passed.
- Inspected all six PNGs. Confirmed sequential focal context, correct branch
  arrows, residual attention paths, quality detachment and loss aggregation.
- Fresh production audit verified Focal reconstruction (25,120 parameters),
  SimAM core/residual reconstruction (0/1 parameters), finite OGA/Q-OGA forwards
  and backwards, detached quality, curriculum epochs 0/2/4 and CQFL equivalence.
- Existing relevant regressions: 137 passed, 7 subtests passed.
- PNGs are 300 DPI; SVGs retain text; PDFs contain no embedded raster images;
  atlas has six pages. Source/config/generator hashes match the audit.
- TikZ exports escape literal formula powers; compilation remains untested
  because no LaTeX compiler is installed. Delivered PDFs are Matplotlib exports.

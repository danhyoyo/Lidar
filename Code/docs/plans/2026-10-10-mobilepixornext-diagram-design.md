### Goal

Create publication diagrams for MobilePixorNeXt using the pastel colors,
dashed panels and encoder-down / decoder-up routing of the two images in
`diagrams/`, with PlotNeuralNet cuboids and editable vector exports.

### Assumptions

- Draw both the master rich8 configuration and the hist14/focal/ECA grouped preset.
- Use English labels for figures intended for papers.
- Preserve the existing reference images and untracked visualization scripts.
- Report backbone parameters including the neck, excluding detection heads.

### Plan

1. Resolve the actual architecture.
   - Files: `tools/visualization/draw_mobilepixornext_paper.py`.
   - Change: Construct the production backbone, capture real forward shapes and
     record configuration hashes and parameter counts.
   - Verify: Run the generator for both configs; assert feature channels/strides.
2. Draw and export the figures.
   - Files: the generator and `diagrams/mobilepixornext/`.
   - Change: Render a shared scene to PNG/SVG/PDF and PlotNeuralNet/TikZ source;
     include NeXt block and exact scale-gate insets.
   - Verify: Inspect previews, check text bounds and PDF text, and ensure arrows
     connect the correct features, especially post-context C4.
3. Document reproduction.
   - Files: `diagrams/mobilepixornext/README.md`.
   - Change: Record configs, commands, dependency requirements and the fact that
     TikZ source is generated but not compiled on this machine.
   - Verify: Read the generated architecture JSON and confirm export paths.

### Risks & mitigations

Avoid hardcoded preset claims: derive labels from actual modules. Show grouped
heads as a repeated per-group template with explicit local class counts.
Keep generated source portable by copying the PlotNeuralNet layer resources and
their license next to the .tex files. Do not install a TeX distribution merely
to render a figure; use the existing Matplotlib renderer for vector files.

### Rollback plan

All deliverables use new paths. Removing those paths restores the previous
workspace without changing any model, config, reference image or existing script.

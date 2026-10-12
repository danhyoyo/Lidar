### Goal

Add a restrained 2D publication figure and a Paper 2D mode to the local NN-SVG
viewer in the main repository. Preserve the 3D renderer and user-owned artwork.

### Assumptions

- The user selected a horizontal figure for two-column IEEE/CVPR-style papers.
- Use English labels and the existing config-derived, validated graph bundle.
- Provide a composite of (a) backbone/FPN, (b) NeXt block, (c) fusion; separate
  detail views remain available. The master config remains the initial example.
- Reference IEEE typography/vector guidance; do not claim a universal journal
  standard. The main figure is 182 mm wide, with restrained grayscale fills,
  regular Arial-compatible typography, thin wires and operator circles.
- SVG is editable source; PDF preserves vector content and embedded fonts;
  PNG is a high-resolution preview. No raster image generation is used.

### Plan

1. Add Paper.js and meaningful rendering/topology tests.
   - Files: `tools/visualization/nn_svg/Paper.js`, `tests/paper.test.cjs`.
   - Change: dedicated composite layout and compact 2D detail renderer.
   - Verify: Node tests, all config-derived node/edge metadata preserved.
2. Integrate a separate presentation choice and CLI export.
   - Files: `Backbone.html`, `backbone-ui.js`, `export-svg.cjs`,
     `tools/visualization/export_backbone_diagram.py`.
   - Change: Paper 2D selector, full vector download, CLI paper output.
   - Verify: browser navigation, export, style switching, imported experiment.
3. Create publication artifacts from current master and optional focal/ECA config.
   - Files: `tools/visualization/export_paper_figure.py`, `diagrams/paper_2d/`.
   - Change: SVG + PDF + PNG, provenance and LaTeX/caption examples.
   - Verify: parse SVG, render previews, inspect PDF fonts/text/page size.
4. Transfer reviewed files into the main repo without touching unrelated edits.
   - Verify: file hashes, relevant tests from main repo, preserved 3D artifacts.

### Risks & mitigations

Keep shapes and conditional modules sourced from the graph contract. Do not
draw removed attention or use labels that imply incorrect convolution order.
Add publication captions outside the figure rather than prominent banners,
marketing headings, gradient fills, shadows or many semantic colors.
Prepare changes under /tmp, then use authorized escalation to write the main
repository, which is outside this session's sandbox write roots.

### Rollback plan

Remove the new 2D artifacts/renderer/generator and revert the added selector and
CLI option. Keep existing 3D figures, model code, configs and user-owned drawings.

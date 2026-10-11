### Goal

Extend a local copy of NN-SVG with a MobilePixorNeXt view: tensor cuboids,
branched FPN wiring, and separate operation graphs for residual blocks and gates.

### Assumptions

- `configs/config.json` is the initial source; other experiment configs can be exported.
- Backbone scope ends at the feature tensor consumed by the detection header.
- SVG cuboids use the logarithmic dimension scaling of NN-SVG's AlexNet view.
  Labels always show actual BCHW sizes. Shapes are inferred from the repository
  contract; an optional PyTorch meta-device check compares them with the model.
- The editor runs locally and exports vector SVG. No hosting or training changes.

### Plan

1. Preserve upstream runtime sources and license in `tools/visualization/nn_svg/`.
   - Files: upstream HTML/JS/fonts, `UPSTREAM.md`.
   - Change: pin snapshot; add a Backbone navigation item to existing styles.
   - Verify: provenance SHA, unchanged source checks, license included.
2. Export configuration-derived graphs without numeric dependencies.
   - Files: `tools/visualization/export_backbone_diagram.py`,
     `tests/test_backbone_diagram.py`.
   - Change: resolve BEV geometry and backbone options using shared validators;
     describe actual stem, stages, FPN, residual/attention/detail/RepConv operators.
   - Verify: `python3 -m pytest tests/test_backbone_diagram.py -q`.
3. Add the SVG renderer and local editor.
   - Files: `Backbone.html`, `Backbone.js`, `backbone-ui.js`, `backbone.css`.
   - Change: cuboids, branch routing, operation detail views, pan/zoom, selectable
     nodes, configurable colors/spacing, JSON loading, SVG export.
   - Verify: Node tests for rendering/escaping/export; browser checks if available.
4. Generate a default model preset and vector figures, document regeneration.
   - Files: `mobilepixornext.js`, `diagrams/nn_svg/`, READMEs.
   - Verify: regenerate from config, parse exported SVG, inspect rendered preview,
     run relevant existing pure contract tests and model tests where available.

### Risks & mitigations

Architecture docs contain retired LiteMLA options. Use current Python code and
shared resolvers. Explicitly reject unsupported backbone/neck types rather than
silently drawing a different graph. Keep SVG text escaped and loaded JSON validated.
Keep numeric/runtime verification optional because the editor needs no PyTorch.

### Rollback plan

Remove the new viewer/exporter/tests/figures and README links. Training code and
configuration stay untouched.

### Verification and review

- Implemented a pinned upstream snapshot, Backbone tab, config exporter, offline
  editor and 13 default SVG views; the focal/ECA experiment generates 15 views.
- Checked upstream Git blob hashes: only the three navigation HTML files differ.
- Relevant Python tests: 43 passed, 1 skipped (PyTorch unavailable).
- Native SVG renderer tests: 6 passed; parsed 28 default/experiment SVG files.
- Playwright Chromium checks passed for offline loading, detail navigation,
  style controls, pan/zoom, full SVG downloads, malformed JSON, escaped labels,
  all default/experiment views and a 540px viewport.
- Default preset equals a fresh export of `configs/config.json`.
- Full pytest suite attempted: collection stops on 59 modules requiring the
  missing PyTorch dependency. No model-forward shape verification was claimed.
- Review: no blocking issues found. Explicit scope is standard MobilePixorNeXt
  SG-FPN; unsupported RC necks/backbones fail before generating a figure.
- Training code, master config and user work remain untouched.

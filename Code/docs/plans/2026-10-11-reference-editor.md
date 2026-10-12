### Goal

Draw the backbone in the style of diagrams/other_paper.png and
other_paper_backbone.png, and customize arbitrary blocks and connections without
editing renderer code. Deliver in the main repository.

### Assumptions

References guide vertical encoder/decoder layout, pastel colors, circles and
dashed group boundaries. Keep the actual bilinear FPN. Extend the existing graph
schema with validated editable geometry; edits affect drawings independently of
training. Keep offline use, 3D, detail navigation and portable JSON.

### Plan

1. Generic document commands and reference renderer.
   - Files: `nn_svg/Editor.js`, `Reference.js`, `tests/editor.test.cjs`.
   - Change: nodes, edges, groups and layout editing; preset placement is data.
   - Verify: arbitrary IDs, transactional edits, undo, deletion cleanup, validation.
2. Browser editing controls.
   - Files: `Backbone.html`, `backbone-ui.js`, `backbone.css`, browser tests.
   - Change: block library, forms, connect mode, drag/resize, groups, history,
     JSON export, draft persistence and reset.
   - Verify: edits/downloads/import/persistence, all views/styles, narrow layout.
3. Reference artwork and editable source.
   - Files: export tools, `diagrams/reference_2d/`, documentation.
   - Change: SVG/PDF/PNG and portable graph document.
   - Verify: PDF fonts/text/vector content, visual preview, exact regeneration.
4. Transfer scoped changes to main.
   - Change: hash-guarded copies preserving existing user artwork and figures.
   - Verify: focused Python/JS/browser checks in main and hash equality.

### Risks & mitigations

Validate untrusted strings, colors, geometry, ports and references. Commit edits
transactionally and allow undo. Clear shape-verification claims after manual
content edits. Retain saved layout on import; reflow only when requested.
Prepare in /tmp and use authorized escalation to write the main repository.

### Rollback plan

Restore backed-up viewer/export files and remove the new reference editor and
artifacts. Model, configs, user artwork and old figure files remain unchanged.

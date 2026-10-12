# NN-SVG editor UX audit

Scope: existing offline editor, sliders, adding/duplicating blocks, connections,
group membership, pending edits, keyboard use and a touch viewport. Source model,
config, existing paper figures and upstream FCNN/LeNet/AlexNet assets are unchanged.

| Observed issue | Change and verification |
| --- | --- |
| Slider reset zoom and replaced an uncommitted label | Preserve viewBox and mounted forms during presentation input. Browser regression reproduces the old failure and checks numeric/range synchronisation, units and reset. |
| Slider targets only 16 CSS px high; spacing display rounded 0.05 steps away | Slider controls are 40 CSS px high, with exact numeric input, output and accessible value text. |
| New blocks could cover existing nodes | Search nearby free positions for additions and duplicates; test 12 consecutive insertions and a duplicate against existing node bounds. |
| Groups required native multi-select with Ctrl/Cmd | Labelled checkbox rows, IDs/shapes to distinguish repeated names, selected count, select all/clear and an explicit empty-group hint. New groups include the currently selected block. |
| Canvas began at y=2158 px on a 390×844 viewport | Canvas is visible immediately, measured at y=176 px and height=608 px before the final group-only layout adjustment. Controls open in a drawer. Touch checks include adding/editing a block and group, closing/reopening and zoom buttons. |
| Connection mode and property forms gave little feedback | Persistent mode instructions, highlighted source, cancel/Escape, connection form, Apply/Cancel and live status. Valid pending forms apply together; invalid forms retain inputs and prevent partial writes. |
| Style edits resubmitted unchanged shape/group fields | Submit changed fields only. Browser checks preserve shape provenance and membership in multiple groups. Editing a wire label preserves imported multi-waypoint routes. |

Accessibility references used as checks:

- [W3C WCAG 2.2 target size](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html): minimum targets/spacing; primary touch controls use 44 CSS px, sliders use 40 CSS px, checkbox rows are at least 44 CSS px. Drawing targets remain zoom-dependent; equivalent form controls are provided for position, size and connections.
- [W3C focus visibility](https://www.w3.org/WAI/WCAG22/Understanding/focus-visible.html): visible keyboard focus, arrow navigation across tabs, named Undo/Redo, keyboard selection of blocks/wires, Escape and focus management for the mobile drawer.
- [W3C contrast](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html): common help text `#53637b` has contrast 6.11:1 on white and 5.79:1 on `#f7f9fc`; white primary button text on `#365fa8` has contrast 6.25:1. User-chosen diagram colors are outside these interface palette checks.

Run the three browser scripts in `tools/visualization/nn_svg/tests/`, plus the
renderer, paper and editor Node tests. Browser checks run in Chromium and include
320/390/540/768 px widths, desktop layouts and touch events. These checks are not
full WCAG certification, screen-reader coverage or a usability study with users.

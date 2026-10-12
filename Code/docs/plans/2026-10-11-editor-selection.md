### Goal
Add drag-to-select multiple blocks in the editable 2D editor, with predictable modifier keys, moving the selection together, direct group creation and undoable deletion.

### Assumptions
- Keep default left-drag canvas panning; add an explicit “Chọn vùng” mode and Shift/Ctrl/Cmd+drag on the background.
- A marquee selects blocks whose bodies intersect it, in either drag direction. Group frames and wires are not blocks.
- Shift/Ctrl/Cmd+click toggles membership; dragging a selected block moves the whole selection. Selection is transient UI state and is excluded from JSON/SVG exports.
- Existing documents, source diagrams, configs and upstream assets remain compatible.

### Plan
1. Define geometry and command behavior with failing tests.
   - Files: `tools/visualization/nn_svg/tests/editor.test.cjs`, `tools/visualization/nn_svg/tests/selection-browser.cjs`.
   - Change: Cover reverse marquee, translation preserving spacing, bounds, atomic commands and browser behavior.
   - Verify: Run Node/browser tests on existing implementation and confirm new cases fail.
2. Implement selection and batch commands.
   - Files: `tools/visualization/nn_svg/Editor.js`, `tools/visualization/nn_svg/backbone-ui.js`.
   - Change: Add pure hit testing, shared translation and transactional multi-delete. Integrate selection state, drag previews, cancel and one-step undo.
   - Verify: `node tools/visualization/nn_svg/tests/editor.test.cjs`; selection browser tests under pan/zoom and pending validation.
3. Make the workflow visible and document it.
   - Files: `tools/visualization/nn_svg/Backbone.html`, `tools/visualization/nn_svg/backbone.css`, `tools/visualization/nn_svg/README.md`.
   - Change: Add a labelled mode button, selection count, multi-block inspector and direct group action. Document mouse and keyboard use.
   - Verify: Browser checks including touch/narrow layouts, keyboard, grouping, exports and unchanged defaults.
4. Review and update the main folder.
   - Files: Only reviewed changed files from the staged implementation.
   - Change: Transfer with baseline hash guards and backups.
   - Verify: Re-run Node and all four browser workflows from main; confirm protected figure hashes.

### Risks & mitigations
- Pan/zoom and SVG offsets can break hit tests: use SVG coordinates, compensate changed rendering offsets and test real pointer interactions at different zoom levels.
- Selection changes can discard edits: validate/apply pending forms before selecting; retain invalid input and previous selection.
- Multi-move must preserve topology and shape claims: shared delta, validated atomic commands, no separate per-node undo steps.

### Rollback plan
Keep backups in `/tmp/lidar-selection-backup`; restore only touched editor files. No commits, training changes or diagram regeneration are required.

### Implementation and review
- Added transient multi-selection, SVG-coordinate marquee intersection, modifier toggles, shared movement/grid snapping and one-command deletion/group creation.
- Pointer capture tracks one pointer; Escape, pointer cancellation and lost capture restore the selection or move. Undo, select-all and deletion cancel an active preview first.
- Fixed mode priority so a selected source can still start a connection after multi-selection. Existing single-block editing and resize remain available.
- Kept the mobile properties action in the toolbar layout. Selecting blocks no longer shifts the canvas underneath a finger.
- Export uses the renderer's clean SVG, independent of selection decorations. Layout/group changes retain source shapes and provenance; deletion marks the architecture as manually edited.

### Verification evidence
- Before implementation: all four added command tests failed on missing selection APIs; the new browser workflow failed on the missing `Chọn vùng` button.
- Browser regressions reproduced before their fixes: Shift held from pointer-down prevented dragging; Shift-click on a selected block did not toggle it under pointer capture; connection mode was intercepted by group movement; mobile selection shifted canvas position by 49 px.
- Staged verification: 24 Node tests and all four browser workflows passed. The final selection workflow also checks Delete/keyboard Undo and select-all during an active drag.
- Coverage includes both marquee directions, partial overlap, Shift/Ctrl/Cmd, movement and snapping, one-step Undo/Redo, cancellation, exact group members, selection under 125% zoom and pan, invalid pending drafts, clean SVG downloads and actual touch events.
- Review confirmed that UI selection is absent from persistent JSON, errors retain pending forms, deletion cleans incident edges and every group, and existing source/config diagrams are outside the transfer.

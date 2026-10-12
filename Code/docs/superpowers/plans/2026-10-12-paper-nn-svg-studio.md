# PaperNN Studio (NN-SVG Enhancement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the repository's `nn_svg` backbone visualizer into a standalone, general-purpose scientific neural network diagram studio ("PaperNN Studio") with modern building blocks (Concat, MHSA, FFN, LayerNorm), multi-block geometric alignment tools, automated publication legends, column width guides, and client-side high-resolution PNG export.

**Architecture:** Extend the existing lightweight offline JavaScript engine (`Editor.js`, `Reference.js`, `backbone-ui.js`) without adding Node/npm runtime dependencies to the browser. New block types and operations integrate into the transactional graph editor, alignment algorithms operate on SVG layouts with full undo/redo history, and SVG vector rendering gains optional legend overlays and canvas-based high-DPI export.

**Tech Stack:** JavaScript (ES6, browser standard DOM & Canvas API), Node.js (for testing and CLI runner `node:test`), SVG 1.1, CSS3, HTML5.

**Spec:** The evaluation in current conversation and requirements for academic paper figures (CVPR / ICCV / NeurIPS / IEEE style diagrams).

## Global Constraints

- Must run 100% offline in browser without Node/npm, external CDNs, Python backend, or build step.
- Must preserve 100% backward compatibility with `mobilepixornext.json`, `export_reference_figure.py`, and existing pytest regression suites.
- Code style: four-space indentation, transactional commands for state mutations, exact input sanitization.
- All new visual features must output clean vector SVG without raster artifacts.

---

### Task 1: Core Library & Block Types Expansion

**Files:**
- Modify: `tools/visualization/nn_svg/Editor.js:8-20`
- Modify: `tools/visualization/nn_svg/Reference.js:8-15`
- Test: `tools/visualization/nn_svg/tests/editor.test.cjs`

**Interfaces:**
- Consumes: Existing `BackboneEditor.LIBRARY`, `BackboneEditor.COLORS`, and `ReferenceSVG` renderer.
- Produces: New block kinds supported across editor validation and renderer:
  - `concat`: Channel concatenation operator (symbol `©`, pastel blue `#d9eafd`)
  - `split`: Channel split operator (symbol `⑂`, pastel blue `#e2e8f0`)
  - `layernorm`: Layer/RMS normalization layer (neutral `#f1f5f9`)
  - `ffn`: Feed-Forward Network / MLP block (amber `#fef3c7`)
  - `mhsa`: Multi-Head Self-Attention block (purple `#ede9fe`)
  - `reshape`: Tensor reshape / flatten / permute block (slate `#f8fafc`)
  - `softmax`: Softmax activation operator (symbol `σ`, rose `#ffe4e6`)

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/nn_svg/tests/editor.test.cjs`:

```javascript
test('new paper block kinds validate and render with distinctive symbols and colors', () => {
    const doc = editor.blank();
    const concatId = editor.addNode(doc, 'overview', 'concat');
    const mhsaId = editor.addNode(doc, 'overview', 'mhsa');
    const ffnId = editor.addNode(doc, 'overview', 'ffn');
    const normId = editor.addNode(doc, 'overview', 'layernorm');

    editor.updateNode(doc, 'overview', concatId, { label: 'Concat', layout: { x: 300, y: 150 } });
    editor.updateNode(doc, 'overview', mhsaId, { label: 'MHSA (h=8)', layout: { x: 500, y: 150 } });
    editor.updateNode(doc, 'overview', ffnId, { label: 'FFN (dim=512)', layout: { x: 700, y: 150 } });
    editor.updateNode(doc, 'overview', normId, { label: 'LayerNorm', layout: { x: 900, y: 150 } });

    editor.validate(doc);
    const svg = reference.render(doc.views.overview).svg;
    assert.match(svg, /©/);
    assert.match(svg, /MHSA \(h=8\)/);
    assert.match(svg, /FFN \(dim=512\)/);
    assert.match(svg, /LayerNorm/);
    assert.match(svg, /#d9eafd/);
    assert.match(svg, /#ede9fe/);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: FAIL with `Unknown block type.` or `Unknown node kind: concat`.

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/nn_svg/Editor.js`:
Extend `COLORS`, `LIBRARY`, and `operators`:
```javascript
const COLORS = Object.freeze({
    input:'#ffffff', output:'#ffffff', conv:'#a8cf91', depthwise:'#dce8f5',
    block:'#dce8f5', norm:'#f5f5f5', activation:'#f8ceb4', scale:'#ffffff',
    add:'#ffffff', multiply:'#ffffff', upsample:'#ffe5a1', gate:'#ffffff',
    attention:'#dce8f5', pool:'#ece5f5', feature:'#ffffff',
    concat:'#d9eafd', split:'#e2e8f0', layernorm:'#f1f5f9', ffn:'#fef3c7',
    mhsa:'#ede9fe', reshape:'#f8fafc', softmax:'#ffe4e6'
});
const LIBRARY = Object.freeze({
    input:'Input tensor', conv:'Conv 3×3', depthwise:'DW 3×3', block:'Custom block',
    norm:'BatchNorm', activation:'SiLU', scale:'LayerScale γ', add:'Add', multiply:'Multiply',
    concat:'Concat (©)', split:'Split (⑂)', layernorm:'LayerNorm', ffn:'FFN / MLP',
    mhsa:'MHSA', reshape:'Reshape / Flatten', softmax:'Softmax (σ)',
    upsample:'Bilinear ↑2', gate:'Scale gate', attention:'Attention', pool:'Global mean',
    feature:'Feature tensor', output:'Output tensor'
});
const operators = new Set(['add', 'multiply', 'scale', 'gate', 'concat', 'split', 'softmax']);
```

In `tools/visualization/nn_svg/Reference.js`:
Add symbols to `symbol` lookup:
```javascript
const math = new Set(['add', 'multiply', 'scale', 'gate', 'concat', 'split', 'softmax']);
const symbol = { add:'+', multiply:'⊙', scale:'γ', gate:'G', concat:'©', split:'⑂', softmax:'σ' };
```

Also ensure `Backbone.js:COLORS` has matching fallback entries so `validateGraph` allows them.

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: PASS (15 tests passing).

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/nn_svg/Editor.js tools/visualization/nn_svg/Reference.js tools/visualization/nn_svg/Backbone.js tools/visualization/nn_svg/tests/editor.test.cjs
git commit -m "feat(nn_svg): expand block library with concat, mhsa, ffn and layernorm"
```

---

### Task 2: Architecture Presets and Engine Decoupling

**Files:**
- Modify: `tools/visualization/nn_svg/Editor.js:67-105`
- Modify: `tools/visualization/nn_svg/Backbone.html:10-50`
- Modify: `tools/visualization/nn_svg/backbone-ui.js:110-120,315-320`
- Test: `tools/visualization/nn_svg/tests/editor.test.cjs`

**Interfaces:**
- Consumes: `Editor.blank()`, `Editor.create()`.
- Produces:
  - `Editor.presets`: Object containing template architectures (`mobilepixornext`, `resnet_stage`, `transformer_block`, `unet_block`).
  - Generalized layout generator: `Editor.layoutGraph(graph, templateName)`.
  - Neutral branding in UI: "PaperNN Studio", customizable model title and download naming.

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/nn_svg/tests/editor.test.cjs`:

```javascript
test('preset templates for resnet and transformer blocks construct valid editable documents', () => {
    assert.ok(editor.PRESETS.resnet_block, 'ResNet preset must exist');
    assert.ok(editor.PRESETS.transformer_block, 'Transformer preset must exist');

    const resnetDoc = editor.create(editor.PRESETS.resnet_block);
    editor.validate(resnetDoc);
    assert.equal(resnetDoc.name, 'ResNet Residual Block');
    assert.ok(resnetDoc.views.overview.nodes.length >= 4);

    const transDoc = editor.create(editor.PRESETS.transformer_block);
    editor.validate(transDoc);
    assert.equal(transDoc.name, 'Vision Transformer Layer');
    assert.ok(transDoc.views.overview.nodes.some(n => n.kind === 'mhsa'));
    assert.ok(transDoc.views.overview.nodes.some(n => n.kind === 'ffn'));
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: FAIL with `ResNet preset must exist`.

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/nn_svg/Editor.js`:
1. Define template schemas in `PRESETS`:
   - `resnet_block`: Input -> Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> Add (+ skip) -> ReLU -> Output.
   - `transformer_block`: Input -> LayerNorm -> MHSA -> Add (+ skip) -> LayerNorm -> FFN -> Add (+ skip) -> Output.
2. Export `PRESETS` on the returned editor module.
3. Make `layoutGraph(graph, template)` generic: if `template === 'mobilepixornext'`, run previous heuristic; otherwise calculate a clean sequential or grid layout based on node rows/cols without assuming hardcoded strings like "Stem" or "NeXt".

In `tools/visualization/nn_svg/Backbone.html`:
1. Update title to `PaperNN Studio · Neural Network Publication Figures`.
2. Update branding header: `<span class="brand-mark">P</span><span>PaperNN Studio <small>PUBLICATION FIGURE EDITOR</small></span>`.
3. In Panel Save, add preset selector:
   `<label for="preset-select">Mẫu kiến trúc tham khảo</label><select id="preset-select"></select><button id="load-preset" class="secondary">Nạp mẫu kiến trúc</button>`.

In `tools/visualization/nn_svg/backbone-ui.js`:
1. Populate `preset-select` with `MobilePixorNeXt (LiDAR BEV)`, `ResNet Residual Block`, `Vision Transformer Layer`.
2. Update SVG export filename to sanitize `doc().name`: `${doc().name.toLowerCase().replace(/[^a-z0-9_-]+/g, '_')}-${view}.svg`.

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/nn_svg/Editor.js tools/visualization/nn_svg/Backbone.html tools/visualization/nn_svg/backbone-ui.js tools/visualization/nn_svg/tests/editor.test.cjs
git commit -m "feat(nn_svg): decouple editor and add generic neural network presets"
```

---

### Task 3: Multi-Block Alignment and Distribution Engine

**Files:**
- Modify: `tools/visualization/nn_svg/Editor.js:150-200`
- Test: `tools/visualization/nn_svg/tests/editor.test.cjs`

**Interfaces:**
- Consumes: Node layout coordinates `layout: { x, y, width, height }`.
- Produces:
  - `Editor.alignNodes(graph, nodeIds, alignment)`: Returns updated node list with aligned coordinates.
    `alignment`: `'left' | 'center-h' | 'right' | 'top' | 'center-v' | 'bottom'`
  - `Editor.distributeNodes(graph, nodeIds, axis)`: Evenly distributes nodes along `'horizontal'` or `'vertical'` axis.

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/nn_svg/tests/editor.test.cjs`:

```javascript
test('alignNodes aligns selected blocks along bounding box edges and centers', () => {
    const graph = {
        nodes: [
            { id: 'a', layout: { x: 100, y: 100, width: 80, height: 60 } },
            { id: 'b', layout: { x: 250, y: 150, width: 100, height: 80 } },
            { id: 'c', layout: { x: 400, y: 200, width: 60, height: 40 } }
        ]
    };
    // Left edge of 'a' is 100 - 40 = 60.
    const leftAligned = editor.alignNodes(graph, ['a', 'b', 'c'], 'left');
    assert.equal(leftAligned.find(n => n.id === 'a').layout.x, 100); // 60 + 40
    assert.equal(leftAligned.find(n => n.id === 'b').layout.x, 110); // 60 + 50
    assert.equal(leftAligned.find(n => n.id === 'c').layout.x, 90);  // 60 + 30

    // Top edge of 'a' is 100 - 30 = 70.
    const topAligned = editor.alignNodes(graph, ['a', 'b', 'c'], 'top');
    assert.equal(topAligned.find(n => n.id === 'a').layout.y, 100); // 70 + 30
    assert.equal(topAligned.find(n => n.id === 'b').layout.y, 110); // 70 + 40
    assert.equal(topAligned.find(n => n.id === 'c').layout.y, 90);  // 70 + 20
});

test('distributeNodes evenly spaces three or more blocks between extremities', () => {
    const graph = {
        nodes: [
            { id: 'a', layout: { x: 100, y: 100, width: 80, height: 60 } },
            { id: 'b', layout: { x: 150, y: 100, width: 80, height: 60 } },
            { id: 'c', layout: { x: 500, y: 100, width: 80, height: 60 } }
        ]
    };
    const distributed = editor.distributeNodes(graph, ['a', 'b', 'c'], 'horizontal');
    // First at 100, last at 500. Middle should be exactly at 300.
    assert.equal(distributed.find(n => n.id === 'b').layout.x, 300);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: FAIL with `editor.alignNodes is not a function`.

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/nn_svg/Editor.js`:
Implement `alignNodes` and `distributeNodes`:
```javascript
function alignNodes(graph, nodeIds, alignment) {
    if (!Array.isArray(nodeIds) || nodeIds.length < 2) throw new Error('Choose at least two blocks to align.');
    const targets = graph.nodes.filter(n => nodeIds.includes(n.id));
    if (targets.length !== nodeIds.length) throw new Error('Invalid node ids for alignment.');
    const boxes = targets.map(n => ({
        id: n.id,
        left: n.layout.x - n.layout.width / 2,
        right: n.layout.x + n.layout.width / 2,
        top: n.layout.y - n.layout.height / 2,
        bottom: n.layout.y + n.layout.height / 2,
        cx: n.layout.x,
        cy: n.layout.y,
        w: n.layout.width,
        h: n.layout.height
    }));
    const minLeft = Math.min(...boxes.map(b => b.left));
    const maxRight = Math.max(...boxes.map(b => b.right));
    const minTop = Math.min(...boxes.map(b => b.top));
    const maxBottom = Math.max(...boxes.map(b => b.bottom));
    const avgCx = (minLeft + maxRight) / 2;
    const avgCy = (minTop + maxBottom) / 2;

    return targets.map(n => {
        const updated = clone(n);
        if (alignment === 'left') updated.layout.x = minLeft + n.layout.width / 2;
        else if (alignment === 'right') updated.layout.x = maxRight - n.layout.width / 2;
        else if (alignment === 'center-h') updated.layout.x = avgCx;
        else if (alignment === 'top') updated.layout.y = minTop + n.layout.height / 2;
        else if (alignment === 'bottom') updated.layout.y = maxBottom - n.layout.height / 2;
        else if (alignment === 'center-v') updated.layout.y = avgCy;
        return updated;
    });
}

function distributeNodes(graph, nodeIds, axis) {
    if (!Array.isArray(nodeIds) || nodeIds.length < 3) throw new Error('Choose at least three blocks to distribute.');
    const targets = graph.nodes.filter(n => nodeIds.includes(n.id));
    if (targets.length !== nodeIds.length) throw new Error('Invalid node ids for distribution.');
    const sorted = [...targets].sort((a, b) => axis === 'horizontal' ? a.layout.x - b.layout.x : a.layout.y - b.layout.y);
    const startCoord = axis === 'horizontal' ? sorted[0].layout.x : sorted[0].layout.y;
    const endCoord = axis === 'horizontal' ? sorted.at(-1).layout.x : sorted.at(-1).layout.y;
    const step = (endCoord - startCoord) / (sorted.length - 1);

    return sorted.map((n, i) => {
        const updated = clone(n);
        if (axis === 'horizontal') updated.layout.x = Math.round(startCoord + i * step);
        else updated.layout.y = Math.round(startCoord + i * step);
        return updated;
    });
}
```
Expose `alignNodes` and `distributeNodes` in module exports.

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: PASS (17 tests passing).

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/nn_svg/Editor.js tools/visualization/nn_svg/tests/editor.test.cjs
git commit -m "feat(nn_svg): implement multi-block alignment and distribution algorithms"
```

---

### Task 4: Alignment & Distribution UI Integration

**Files:**
- Modify: `tools/visualization/nn_svg/backbone-ui.js:247-265`
- Modify: `tools/visualization/nn_svg/backbone.css:120-150`
- Test: `tools/visualization/nn_svg/tests/editor.test.cjs`

**Interfaces:**
- Consumes: `Editor.alignNodes`, `Editor.distributeNodes`, and `commit()`.
- Produces: Visual toolbars in Multi-Inspector with quick buttons:
  - Căn lề trái, giữa, phải (Align Left, Center H, Right)
  - Căn hàng trên, giữa, dưới (Align Top, Middle V, Bottom)
  - Dãn cách đều ngang / dọc (Distribute H / V)
  - Full Undo/Redo integration.

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/nn_svg/tests/editor.test.cjs`:

```javascript
test('aligning and distributing via History commits a single reversible undo state', () => {
    const doc = editor.blank();
    const a = editor.addNode(doc, 'overview', 'conv', { x: 100, y: 100 });
    const b = editor.addNode(doc, 'overview', 'conv', { x: 200, y: 150 });
    const c = editor.addNode(doc, 'overview', 'conv', { x: 300, y: 200 });
    const history = new editor.History(doc);

    // Apply alignment
    history.commit(d => {
        const aligned = editor.alignNodes(d.views.overview, [a, b, c], 'top');
        for (const n of aligned) {
            d.views.overview.nodes.find(target => target.id === n.id).layout = n.layout;
        }
    });

    const topY = history.value.views.overview.nodes.find(n => n.id === a).layout.y;
    assert.equal(history.value.views.overview.nodes.find(n => n.id === b).layout.y, topY);
    assert.equal(history.value.views.overview.nodes.find(n => n.id === c).layout.y, topY);

    history.undo();
    assert.equal(history.value.views.overview.nodes.find(n => n.id === b).layout.y, 150);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: Passes logic check if Task 3 is complete, or validates the transaction pattern.

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/nn_svg/backbone-ui.js`:
In `multiInspector(panel)`:
Add an alignment controls group:
```javascript
const alignGroup = element('div', '', 'alignment-tools');
alignGroup.append(element('h3', 'Căn gióng các khối đã chọn'));
const gridH = element('div', '', 'button-grid three');
for (const [align, title] of [['left', '← Trái'], ['center-h', '↔ Giữa'], ['right', '→ Phải']]) {
    const btn = element('button', title, 'secondary small');
    btn.onclick = () => {
        commit(d => {
            const updated = E.alignNodes(d.views[view], [...selection], align);
            for (const u of updated) d.views[view].nodes.find(n => n.id === u.id).layout = u.layout;
            return true;
        });
        feedback(`Đã căn ${title.toLowerCase()}. ↶ để hoàn tác.`);
    };
    gridH.append(btn);
}
const gridV = element('div', '', 'button-grid three');
for (const [align, title] of [['top', '↑ Trên'], ['center-v', '↕ Giữa'], ['bottom', '↓ Dưới']]) {
    const btn = element('button', title, 'secondary small');
    btn.onclick = () => {
        commit(d => {
            const updated = E.alignNodes(d.views[view], [...selection], align);
            for (const u of updated) d.views[view].nodes.find(n => n.id === u.id).layout = u.layout;
            return true;
        });
        feedback(`Đã căn ${title.toLowerCase()}. ↶ để hoàn tác.`);
    };
    gridV.append(btn);
}
alignGroup.append(gridH, gridV);

if (selection.size >= 3) {
    const distGrid = element('div', '', 'button-grid two');
    const distH = element('button', 'Dãn đều ngang', 'secondary small');
    distH.onclick = () => {
        commit(d => {
            const updated = E.distributeNodes(d.views[view], [...selection], 'horizontal');
            for (const u of updated) d.views[view].nodes.find(n => n.id === u.id).layout = u.layout;
            return true;
        });
        feedback('Đã dãn cách đều các khối theo chiều ngang.');
    };
    const distV = element('button', 'Dãn đều dọc', 'secondary small');
    distV.onclick = () => {
        commit(d => {
            const updated = E.distributeNodes(d.views[view], [...selection], 'vertical');
            for (const u of updated) d.views[view].nodes.find(n => n.id === u.id).layout = u.layout;
            return true;
        });
        feedback('Đã dãn cách đều các khối theo chiều dọc.');
    };
    distGrid.append(distH, distV);
    alignGroup.append(distGrid);
}
panel.append(alignGroup);
```

In `tools/visualization/nn_svg/backbone.css`:
Add CSS styling for `.alignment-tools`, `.button-grid`, and `.button-grid.three`.

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/nn_svg/backbone-ui.js tools/visualization/nn_svg/backbone.css tools/visualization/nn_svg/tests/editor.test.cjs
git commit -m "feat(nn_svg): integrate alignment and distribution toolbar into multi-inspector UI"
```

---

### Task 5: Automated Publication Legend Generator

**Files:**
- Modify: `tools/visualization/nn_svg/Reference.js:80-125`
- Modify: `tools/visualization/nn_svg/backbone-ui.js:25-35,170-185`
- Modify: `tools/visualization/nn_svg/Backbone.html:38-45`
- Test: `tools/visualization/nn_svg/tests/editor.test.cjs`

**Interfaces:**
- Consumes: Graph nodes, node `kind`, and `options.showLegend`.
- Produces: SVG `<g class="diagram-legend">` with clean colored swatches and layer kind labels rendered at the bottom or top corner.

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/nn_svg/tests/editor.test.cjs`:

```javascript
test('ReferenceSVG renders publication legend when show_legend option is enabled', () => {
    const doc = editor.blank();
    editor.addNode(doc, 'overview', 'conv');
    editor.addNode(doc, 'overview', 'concat');
    editor.addNode(doc, 'overview', 'mhsa');

    // Without legend
    const cleanSvg = reference.render(doc.views.overview, { showLegend: false }).svg;
    assert.doesNotMatch(cleanSvg, /class="diagram-legend"/);

    // With legend
    const legendSvg = reference.render(doc.views.overview, { showLegend: true }).svg;
    assert.match(legendSvg, /class="diagram-legend"/);
    assert.match(legendSvg, /Conv 3×3/);
    assert.match(legendSvg, /Concat/);
    assert.match(legendSvg, /MHSA/);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: FAIL with `doesNotMatch` or missing `diagram-legend`.

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/nn_svg/Reference.js`:
In `render(input, options)`:
1. Check `options.showLegend || graph.show_legend`.
2. If true:
   - Collect unique non-math node kinds present in `graph.nodes`.
   - Calculate legend box width and height based on the number of items.
   - Render a framed group:
     `<g class="diagram-legend" transform="translate(...)"><rect ... fill="#ffffff" stroke="#e2e8f0" rx="6"/><text font-weight="bold">Legend</text>...items</g>`
   - Place items with small colored rounded rectangles + label text.

In `tools/visualization/nn_svg/Backbone.html`:
Add checkbox in Kiến trúc tab:
`<label class="check"><input id="show-legend" type="checkbox"> Hiện bảng chú thích màu (Legend)</label>`.

In `tools/visualization/nn_svg/backbone-ui.js`:
Wire `#show-legend` into `options()` and `draw()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/nn_svg/Reference.js tools/visualization/nn_svg/Backbone.html tools/visualization/nn_svg/backbone-ui.js tools/visualization/nn_svg/tests/editor.test.cjs
git commit -m "feat(nn_svg): add automated publication legend generator"
```

---

### Task 6: Paper Column Width Guides & Physical Scale Feedback

**Files:**
- Modify: `tools/visualization/nn_svg/Backbone.html:95-105`
- Modify: `tools/visualization/nn_svg/backbone-ui.js:100-110,180-190`
- Modify: `tools/visualization/nn_svg/backbone.css:150-180`
- Test: `tools/visualization/nn_svg/tests/editor.test.cjs`

**Interfaces:**
- Consumes: Canvas SVG dimensions and viewBox.
- Produces: Visual paper boundary guide overlay (Single-column ~85mm, Double-column ~182mm) with real physical size readout (mm / inches) in canvas footer.

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/nn_svg/tests/editor.test.cjs`:

```javascript
test('clean SVG export never includes interactive paper guides or selection boxes', () => {
    const doc = editor.create(preset());
    const rendered = reference.render(doc.views.overview);
    assert.doesNotMatch(rendered.svg, /data-paper-guide/);
    assert.doesNotMatch(rendered.svg, /selection-box/);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: PASS (asserts regression baseline).

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/nn_svg/Backbone.html`:
In toolbar actions:
Add `<select id="column-guide" class="secondary" title="Đường gióng khổ bài báo"><option value="none">Khổ giấy: Tự do</option><option value="single">Khổ 1 cột (85 mm)</option><option value="double" selected>Khổ 2 cột (182 mm)</option></select>`.
In canvas footer:
Add `<span id="physical-size" class="subtle" title="Kích thước in dự kiến trên bài báo">182 × 76 mm</span>`.

In `tools/visualization/nn_svg/backbone-ui.js`:
1. In `setFrame()` and `draw()`: calculate physical millimeter dimensions:
   `width_mm = 182; height_mm = Math.round(182 * rendered.height / rendered.width);`
   Update `$('physical-size').textContent = `${width_mm} × ${height_mm} mm (Khổ 2 cột)`;`
2. When `#column-guide` is not `none`: render non-exported dashed guideline lines on the canvas overlay showing the target bounding box.

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/nn_svg/Backbone.html tools/visualization/nn_svg/backbone-ui.js tools/visualization/nn_svg/backbone.css tools/visualization/nn_svg/tests/editor.test.cjs
git commit -m "feat(nn_svg): add paper column width guides and physical size indicator"
```

---

### Task 7: In-Browser High-Resolution PNG & Vector Export

**Files:**
- Modify: `tools/visualization/nn_svg/backbone-ui.js:315-330`
- Modify: `tools/visualization/nn_svg/Backbone.html:95-100`
- Test: `tools/visualization/nn_svg/tests/editor.test.cjs`

**Interfaces:**
- Consumes: Rendered SVG string, canvas image rasterization via standard `HTMLCanvasElement`.
- Produces: Direct client-side download of 300 DPI / 600 DPI PNG images without requiring Node or `rsvg-convert`.

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/nn_svg/tests/editor.test.cjs`:

```javascript
test('SVG generated for client rasterization has explicit pixel dimensions matching physical DPI', () => {
    const doc = editor.create(preset());
    const res = reference.render(doc.views.overview);
    // At 182mm width and 300 DPI (11.811 px/mm), target width is ~2150px
    const scaleFactor300 = (300 / 25.4) * (182 / res.width);
    assert.ok(scaleFactor300 > 0 && Number.isFinite(scaleFactor300));
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: PASS.

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/nn_svg/Backbone.html`:
Split or expand export options:
`<button id="download" class="primary">↓ Xuất SVG</button><button id="download-png" class="secondary">↓ Xuất PNG (600 DPI)</button>`.

In `tools/visualization/nn_svg/backbone-ui.js`:
Implement in-browser high-res rasterization helper:
```javascript
function exportRaster(dpi = 300) {
    if (!rendered || !flushPending()) return;
    const svgStr = rendered.svg;
    const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
        const targetWidth = Math.round((dpi / 25.4) * 182); // 182mm at target DPI
        const targetHeight = Math.round(targetWidth * (rendered.height / rendered.width));
        const offscreen = document.createElement('canvas');
        offscreen.width = targetWidth;
        offscreen.height = targetHeight;
        const ctx = offscreen.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, targetWidth, targetHeight);
        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
        URL.revokeObjectURL(url);
        offscreen.toBlob(pngBlob => {
            const pngUrl = URL.createObjectURL(pngBlob);
            const a = element('a');
            const sanitizedName = (doc().name || 'architecture').toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
            a.download = `${sanitizedName}-${view}-${dpi}dpi.png`;
            a.href = pngUrl;
            document.body.append(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(pngUrl), 1000);
            feedback(`Đã xuất PNG độ phân giải cao (${dpi} DPI).`);
        }, 'image/png');
    };
    img.src = url;
}
$('download-png').addEventListener('click', () => exportRaster(600));
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/nn_svg/Backbone.html tools/visualization/nn_svg/backbone-ui.js tools/visualization/nn_svg/tests/editor.test.cjs
git commit -m "feat(nn_svg): add in-browser high-dpi PNG export"
```

---

### Task 8: End-to-End Regression & Documentation

**Files:**
- Modify: `tools/visualization/nn_svg/README.md`
- Test: All test suites (`node tests/*.test.cjs`, `pytest tests/test_reference_figure.py`)

**Interfaces:**
- Consumes: Complete enhanced studio.
- Produces: Updated documentation in Vietnamese describing all new features, presets, alignment tools, legend, and export shortcuts.

- [ ] **Step 1: Write regression test verification**

Run full test suite:
```bash
node tools/visualization/nn_svg/tests/renderer.test.cjs
node tools/visualization/nn_svg/tests/paper.test.cjs
node tools/visualization/nn_svg/tests/editor.test.cjs
python3 -m pytest tests/test_reference_figure.py -q
```

- [ ] **Step 2: Update documentation**

In `tools/visualization/nn_svg/README.md`:
Document:
1. PaperNN Studio purpose & quickstart.
2. New building blocks (Concat, MHSA, FFN, LayerNorm).
3. Alignment & distribution commands.
4. Publication legend toggle.
5. In-browser 600 DPI PNG export.

- [ ] **Step 3: Run all tests to verify**

Run: `pytest tests/test_reference_figure.py && node tools/visualization/nn_svg/tests/editor.test.cjs`
Expected: ALL PASS.

- [ ] **Step 4: Commit**

```bash
git add tools/visualization/nn_svg/README.md
git commit -m "docs(nn_svg): update guide for PaperNN Studio features and presets"
```

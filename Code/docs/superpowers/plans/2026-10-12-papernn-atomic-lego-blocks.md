# PaperNN Atomic Lego Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform PaperNN Studio into a universal, atomic Deep Learning component studio where any neural network architecture can be freely assembled from fundamental primitives (Tensors, Linear, Convolutions, Norms, Activations, Pooling, Math Ops, Attention), each rendered with distinctive academic visual geometry (narrow bars, trapezoids, circles, stacked cards) and selected via an intuitive categorized component palette.

**Architecture:** Extend `Editor.js` with a comprehensive taxonomy of deep learning component primitives and categories (`CATEGORIES`, `SHAPES`). Extend `Reference.js` to render specific SVG geometries (`<polygon>` for trapezoids, slim vertical bars for normalizations and activations, offset shadows for repeated stacked layers, and circles for math operators) while calculating accurate boundary ports for wire routing. Upgrade `Backbone.html` and `backbone-ui.js` with an accordion-based categorized component drawer.

**Tech Stack:** JavaScript (ES6, browser standard DOM & Canvas API), Node.js (for testing via `node:test` and `node:assert/strict`), SVG 1.1, CSS3, HTML5.

**Spec:** User requirement for a versatile neural network drawing web application with all modern atomic components (Linear, Conv2d, Residuals, Norms, Attention) featuring distinct visual semantics rather than uniform rectangular boxes.

## Global Constraints

- Must be implemented inside `tools/visualization/papernn_studio/` without modifying the original `tools/visualization/nn_svg/` directory.
- Must run 100% offline in browser without Node/npm, external CDNs, Python backend, or build step.
- Must preserve 100% backward compatibility with existing tests and presets (`mobilepixornext`, `resnet_block`, `transformer_block`).
- Code style: four-space indentation, transactional commands for state mutations, exact input sanitization.
- All new visual features must output clean vector SVG without raster artifacts.

---

### Task 1: Complete Deep Learning Atomic Taxonomy & Categories

**Files:**
- Modify: `tools/visualization/papernn_studio/Editor.js:8-30`
- Modify: `tools/visualization/papernn_studio/Backbone.js:12-25`
- Test: `tools/visualization/papernn_studio/tests/editor.test.cjs`

**Interfaces:**
- Consumes: Existing `BackboneEditor` and `BackboneSVG`.
- Produces:
  - `Editor.CATEGORIES`: 7 categorized component groups (`tensors`, `layers`, `norms`, `activations`, `pooling`, `operations`, `attention`).
  - `Editor.SHAPE_TYPES`: Map of kind to visual shape (`standard`, `slim`, `trapezoid_down`, `trapezoid_up`, `circle`).
  - Expanded `COLORS` and `LIBRARY` with:
    - Linear, Conv1D, Pointwise (1x1), Dilated Conv, Deconv
    - RMSNorm, GroupNorm, InstanceNorm
    - ReLU, GELU, SiLU, Sigmoid, Tanh
    - MaxPool, AvgPool, PixelShuffle
    - Matmul (⊗), Scale (γ), RoPE, QKV Projection, MoE Router

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/papernn_studio/tests/editor.test.cjs`:

```javascript
test('Editor supports comprehensive atomic deep learning taxonomy and category grouping', () => {
    assert.ok(editor.CATEGORIES, 'CATEGORIES dictionary must exist');
    assert.ok(editor.CATEGORIES.tensors && editor.CATEGORIES.layers && editor.CATEGORIES.norms, 'Standard categories must be defined');
    assert.ok(editor.CATEGORIES.activations && editor.CATEGORIES.pooling && editor.CATEGORIES.operations && editor.CATEGORIES.attention);

    const doc = editor.blank();
    const linearId = editor.addNode(doc, 'overview', 'linear');
    const rmsId = editor.addNode(doc, 'overview', 'rmsnorm');
    const geluId = editor.addNode(doc, 'overview', 'gelu');
    const poolId = editor.addNode(doc, 'overview', 'maxpool');
    const matmulId = editor.addNode(doc, 'overview', 'matmul');

    editor.validate(doc);
    const g = doc.views.overview;
    assert.equal(g.nodes.find(n => n.id === linearId).kind, 'linear');
    assert.equal(g.nodes.find(n => n.id === rmsId).kind, 'rmsnorm');
    assert.equal(g.nodes.find(n => n.id === geluId).kind, 'gelu');
    assert.equal(g.nodes.find(n => n.id === poolId).kind, 'maxpool');
    assert.equal(g.nodes.find(n => n.id === matmulId).kind, 'matmul');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/papernn_studio/tests/editor.test.cjs`
Expected: FAIL with `CATEGORIES dictionary must exist` or `Unknown block type.`

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/papernn_studio/Backbone.js`:
Add all new atomic kinds to `COLORS` lookup to ensure `validateGraph` accepts them:
```javascript
const COLORS = Object.freeze({
    input: '#64748b', output: '#2563eb', feature: '#64748b', embedding: '#818cf8',
    conv: '#5b7fd6', conv1d: '#5b7fd6', depthwise: '#cf8635', pointwise: '#60a5fa', deconv: '#0284c7', linear: '#3b82f6',
    norm: '#67a09b', layernorm: '#67a09b', rmsnorm: '#0d9488', groupnorm: '#14b8a6',
    activation: '#c97f91', relu: '#f43f5e', gelu: '#fb7185', silu: '#e11d48', sigmoid: '#fda4af', tanh: '#f43f5e',
    pool: '#7d95a5', maxpool: '#64748b', avgpool: '#94a3b8', upsample: '#4a96b8', pixelshuffle: '#38bdf8',
    add: '#399d89', multiply: '#399d89', concat: '#2563eb', split: '#64748b', matmul: '#8b5cf6', scale: '#a88435', gate: '#399d89', softmax: '#e11d48',
    block: '#8b6bc1', attention: '#b369ad', mhsa: '#7c3aed', qkv: '#a855f7', ffn: '#d97706', rope: '#c084fc', moe: '#ea580c', reshape: '#475569'
});
```

In `tools/visualization/papernn_studio/Editor.js`:
Define `CATEGORIES`, `SHAPE_TYPES`, and extend `COLORS` / `LIBRARY` / `operators`:
```javascript
const CATEGORIES = Object.freeze({
    tensors: { label: 'Tensors & Data', kinds: ['input', 'feature', 'embedding', 'output'] },
    layers: { label: 'Convolutions & Linear', kinds: ['conv', 'pointwise', 'depthwise', 'conv1d', 'linear', 'deconv'] },
    norms: { label: 'Normalization (Thanh mảnh)', kinds: ['norm', 'layernorm', 'rmsnorm', 'groupnorm'] },
    activations: { label: 'Hàm kích hoạt (Pill / Curve)', kinds: ['relu', 'gelu', 'silu', 'sigmoid', 'tanh'] },
    pooling: { label: 'Pooling & Co dãn (Hình thang)', kinds: ['maxpool', 'avgpool', 'upsample', 'pixelshuffle'] },
    operations: { label: 'Phép toán (Vòng tròn)', kinds: ['add', 'multiply', 'concat', 'split', 'matmul', 'scale', 'softmax', 'gate'] },
    attention: { label: 'Attention & Transformers', kinds: ['mhsa', 'qkv', 'ffn', 'rope', 'moe'] }
});

const SHAPE_TYPES = Object.freeze({
    input: 'standard', output: 'standard', feature: 'standard', embedding: 'standard',
    conv: 'standard', conv1d: 'standard', depthwise: 'standard', pointwise: 'standard', deconv: 'standard', linear: 'standard',
    norm: 'slim', layernorm: 'slim', rmsnorm: 'slim', groupnorm: 'slim',
    activation: 'slim', relu: 'slim', gelu: 'slim', silu: 'slim', sigmoid: 'slim', tanh: 'slim',
    pool: 'trapezoid_down', maxpool: 'trapezoid_down', avgpool: 'trapezoid_down',
    upsample: 'trapezoid_up', pixelshuffle: 'trapezoid_up',
    add: 'circle', multiply: 'circle', concat: 'circle', split: 'circle', matmul: 'circle', scale: 'circle', gate: 'circle', softmax: 'circle',
    block: 'standard', attention: 'standard', mhsa: 'standard', qkv: 'standard', ffn: 'standard', rope: 'standard', moe: 'standard', reshape: 'standard'
});
```
Extend `LIBRARY` with descriptive titles and export `CATEGORIES` and `SHAPE_TYPES`.
In `addNode`: set initial default sizes matching `SHAPE_TYPES[kind]`:
- `circle`: 44x44
- `slim`: 44x76
- `trapezoid_down` / `trapezoid_up`: 96x64
- `standard`: 160x72

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/papernn_studio/tests/editor.test.cjs`
Expected: PASS (23 tests passing).

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/papernn_studio/Editor.js tools/visualization/papernn_studio/Backbone.js tools/visualization/papernn_studio/tests/editor.test.cjs
git commit -m "feat(papernn): define full atomic deep learning taxonomy and shape mappings"
```

---

### Task 2: Geometric Visual Renderer for Specialized Shapes

**Files:**
- Modify: `tools/visualization/papernn_studio/Reference.js:8-40,90-125`
- Test: `tools/visualization/papernn_studio/tests/editor.test.cjs`

**Interfaces:**
- Consumes: Node kinds, `Editor.SHAPE_TYPES`, layout coordinates `(x, y, width, height)`.
- Produces:
  - SVG `<polygon>` for `trapezoid_down` (wider left, narrower right) and `trapezoid_up` (narrower left, wider right).
  - SVG `<rect rx="12">` for `slim` (narrow rounded vertical bar with centered vertical label or activation curve).
  - SVG `<ellipse>` for `circle` math operators with expanded symbols ($\otimes, \copyright, \rightleftharpoons, \oplus, \odot, \sigma$).
  - Offset stacked shadow borders for blocks with `repeat > 1`.
  - Precise wire port docking for non-rectangular nodes.

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/papernn_studio/tests/editor.test.cjs`:

```javascript
test('ReferenceSVG renders distinctive visual geometries for trapezoid, slim bar, math circle, and stacked layers', () => {
    const doc = editor.blank();
    const poolId = editor.addNode(doc, 'overview', 'maxpool');
    const normId = editor.addNode(doc, 'overview', 'rmsnorm');
    const matmulId = editor.addNode(doc, 'overview', 'matmul');
    const stackedId = editor.addNode(doc, 'overview', 'conv');

    editor.updateNode(doc, 'overview', stackedId, { label: 'Conv Block', repeat: 3, layout: { x: 200, y: 150 } });
    editor.updateNode(doc, 'overview', poolId, { label: 'MaxPool 2×2', layout: { x: 400, y: 150 } });
    editor.updateNode(doc, 'overview', normId, { label: 'RMSNorm', layout: { x: 600, y: 150 } });
    editor.updateNode(doc, 'overview', matmulId, { label: 'Matmul', layout: { x: 800, y: 150 } });

    const svg = reference.render(doc.views.overview).svg;
    // Trapezoid must use SVG polygon
    assert.match(svg, /<polygon class="node-trapezoid"/);
    // Slim bar must have node-slim class
    assert.match(svg, /class="node-slim"/);
    // Matmul circle must render tensor multiplication symbol
    assert.match(svg, /⊗/);
    // Stacked layer must have stacked shadow cards
    assert.match(svg, /class="stacked-shadow"/);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/papernn_studio/tests/editor.test.cjs`
Expected: FAIL with `AssertionError: The input did not match the regular expression /<polygon class="node-trapezoid"/`.

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/papernn_studio/Reference.js`:
1. Expand symbol lookup:
```javascript
const symbol = {
    add: '+', multiply: '⊙', scale: 'γ', gate: 'G', concat: '©',
    split: '⑂', softmax: 'σ', matmul: '⊗'
};
```
2. In `render()`, look up `shapeType = editor.SHAPE_TYPES[n.kind] || 'standard'`:
   - If `shapeType === 'circle'`:
     Render circle with `symbol[n.kind]`.
   - If `shapeType === 'trapezoid_down'`:
     Calculate trapezoid points where left height is `h` and right height is `h * 0.55`:
     ```javascript
     const p1 = `${b.left},${b.top}`;
     const p2 = `${b.right},${l.y - l.height * 0.275}`;
     const p3 = `${b.right},${l.y + l.height * 0.275}`;
     const p4 = `${b.left},${b.bottom}`;
     content += `<polygon class="node-trapezoid" points="${p1} ${p2} ${p3} ${p4}" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.3"/>`;
     ```
   - If `shapeType === 'trapezoid_up'`:
     Calculate points where left height is `h * 0.55` and right height is `h`:
     ```javascript
     const p1 = `${b.left},${l.y - l.height * 0.275}`;
     const p2 = `${b.right},${b.top}`;
     const p3 = `${b.right},${b.bottom}`;
     const p4 = `${b.left},${l.y + l.height * 0.275}`;
     content += `<polygon class="node-trapezoid" points="${p1} ${p2} ${p3} ${p4}" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.3"/>`;
     ```
   - If `shapeType === 'slim'`:
     Render slim vertical bar with rounded corners:
     ```javascript
     content += `<rect class="node-slim" x="${b.left}" y="${b.top}" width="${l.width}" height="${l.height}" rx="10" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.3"/>`;
     ```
   - If `n.repeat && n.repeat > 1`:
     Render offset shadow cards behind the main shape:
     ```javascript
     content += `<rect class="stacked-shadow" x="${b.left + 5}" y="${b.top - 5}" width="${l.width}" height="${l.height}" rx="6" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.2" opacity="0.6"/>`;
     content += `<rect class="stacked-shadow" x="${b.left + 2.5}" y="${b.top - 2.5}" width="${l.width}" height="${l.height}" rx="6" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.2" opacity="0.8"/>`;
     ```
3. Update `bounds(n)` and `port(n, side)`:
   - For `trapezoid_down`: right port is at `(b.right, l.y)`.
   - For `trapezoid_up`: left port is at `(b.left, l.y)`.
   Ports remain centered on edges, matching orthogonal connection lines.

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/papernn_studio/tests/editor.test.cjs`
Expected: PASS (24 tests passing).

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/papernn_studio/Reference.js tools/visualization/papernn_studio/tests/editor.test.cjs
git commit -m "feat(papernn): render geometric visual shapes for trapezoids, slim bars, circles and stacked layers"
```

---

### Task 3: Categorized Component Palette Drawer UI

**Files:**
- Modify: `tools/visualization/papernn_studio/Backbone.html:50-65`
- Modify: `tools/visualization/papernn_studio/backbone-ui.js:295-305`
- Test: `tools/visualization/papernn_studio/tests/editor.test.cjs`

**Interfaces:**
- Consumes: `Editor.CATEGORIES`, `Editor.LIBRARY`, `Editor.SHAPE_TYPES`, `commit()`.
- Produces:
  - Interactive accordion drawer in tab **Block** (`#panel-blocks`).
  - Fast 1-click addition button for every component in its category.
  - Automatic optimal dimensions on creation based on shape semantics.

- [ ] **Step 1: Write the failing test**

Add to `tools/visualization/papernn_studio/tests/editor.test.cjs`:

```javascript
test('addNode automatically assigns distinct initial width and height matching shape semantics', () => {
    const doc = editor.blank();
    const normId = editor.addNode(doc, 'overview', 'layernorm');
    const poolId = editor.addNode(doc, 'overview', 'maxpool');
    const convId = editor.addNode(doc, 'overview', 'conv');
    const mathId = editor.addNode(doc, 'overview', 'concat');

    const g = doc.views.overview;
    const norm = g.nodes.find(n => n.id === normId).layout;
    const pool = g.nodes.find(n => n.id === poolId).layout;
    const conv = g.nodes.find(n => n.id === convId).layout;
    const math = g.nodes.find(n => n.id === mathId).layout;

    // Slim bar: width is narrow (~44)
    assert.ok(norm.width <= 50 && norm.height >= 70, 'Norm must have slim vertical dimensions');
    // Trapezoid pooling: width is moderate (~96)
    assert.ok(pool.width >= 80 && pool.width <= 120, 'Pooling must have trapezoid dimensions');
    // Standard conv card: width is wide (~160)
    assert.ok(conv.width >= 140, 'Conv must have card dimensions');
    // Circle math: width and height are equal (44x44)
    assert.equal(math.width, math.height, 'Math operator must be square/circular');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tools/visualization/papernn_studio/tests/editor.test.cjs`
Expected: FAIL if default sizes were not yet differentiated in Task 1/2.

- [ ] **Step 3: Write minimal implementation**

In `tools/visualization/papernn_studio/Backbone.html`:
Replace the single dropdown `<select id="block-library">` with a categorized component palette:
```html
<section id="panel-blocks" role="tabpanel" aria-labelledby="tab-blocks" hidden>
    <h2>Thư viện khối nguyên tử</h2>
    <p class="help">Chọn khối để thêm vào sơ đồ. Mỗi khối có hình dạng đặc thù chuẩn bài báo.</p>
    <div id="palette-categories" class="palette-categories"></div>
    ...
```

In `tools/visualization/papernn_studio/backbone-ui.js`:
Build the categorized component palette dynamically from `E.CATEGORIES`:
```javascript
function buildComponentPalette() {
    const container = $('palette-categories');
    if (!container) return;
    container.replaceChildren();
    for (const [catKey, cat] of Object.entries(E.CATEGORIES)) {
        const details = element('details', '', 'palette-group');
        details.open = ['tensors', 'layers', 'norms', 'operations'].includes(catKey);
        const summary = element('summary', cat.label);
        details.append(summary);
        const grid = element('div', '', 'palette-button-grid');
        for (const kind of cat.kinds) {
            const btn = element('button', E.LIBRARY[kind] || kind, 'palette-btn');
            btn.type = 'button';
            btn.dataset.kind = kind;
            btn.style.borderLeft = `4px solid ${E.COLORS[kind] || '#cbd5e1'}`;
            btn.onclick = () => {
                if (!editable()) return;
                const box = canvas.getBoundingClientRect();
                const point = svgPoint(box.x + box.width / 2, box.y + box.height / 2);
                const offset = rendered.offset || { x: 0, y: 0 };
                const id = commit(d => E.addNode(d, view, kind, { x: point.x - offset.x, y: point.y - offset.y }));
                if (id) {
                    inspect(id);
                    revealNode(id);
                    setMode('edit');
                    feedback(`Đã thêm ${E.LIBRARY[kind]}. Kéo hoặc sửa thuộc tính bên dưới.`);
                }
            };
            grid.append(btn);
        }
        details.append(grid);
        container.append(details);
    }
}
```
Call `buildComponentPalette()` during initial load.

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/papernn_studio/tests/editor.test.cjs`
Expected: PASS (25 tests passing).

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/papernn_studio/Backbone.html tools/visualization/papernn_studio/backbone-ui.js tools/visualization/papernn_studio/tests/editor.test.cjs
git commit -m "feat(papernn): add categorized component palette drawer for atomic primitives"
```

---

### Task 4: Visual Palette & Geometric SVG Styling

**Files:**
- Modify: `tools/visualization/papernn_studio/backbone.css:140-190`
- Test: `tools/visualization/papernn_studio/tests/editor.test.cjs`

**Interfaces:**
- Consumes: Category drawer and geometric SVG classes (`.palette-group`, `.palette-btn`, `.node-trapezoid`, `.node-slim`, `.stacked-shadow`).
- Produces: Clean, modern responsive stylesheet matching academic aesthetics.

- [ ] **Step 1: Write styling test verification**

Verify classes exist in rendered output:
Add to `tools/visualization/papernn_studio/tests/editor.test.cjs`:

```javascript
test('SVG rendered nodes output correct CSS class hooks for styling', () => {
    const doc = editor.blank();
    editor.addNode(doc, 'overview', 'maxpool');
    editor.addNode(doc, 'overview', 'rmsnorm');
    const svg = reference.render(doc.views.overview).svg;
    assert.match(svg, /class="node-trapezoid"/);
    assert.match(svg, /class="node-slim"/);
});
```

- [ ] **Step 2: Run test to verify it passes**

Run: `node tools/visualization/papernn_studio/tests/editor.test.cjs`
Expected: PASS.

- [ ] **Step 3: Write CSS implementation**

In `tools/visualization/papernn_studio/backbone.css`:
Add styles for:
```css
/* Palette Accordion */
.palette-categories { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }
.palette-group { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden; }
.palette-group summary { padding: 8px 12px; font-size: 12px; font-weight: 600; color: #334155; cursor: pointer; user-select: none; }
.palette-group summary:hover { background: #f1f5f9; }
.palette-button-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; padding: 8px 10px; background: #ffffff; }
.palette-btn { text-align: left; padding: 6px 8px; font-size: 11px; font-weight: 500; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer; transition: background .15s, border-color .15s; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.palette-btn:hover { background: #e2e8f0; border-color: #cbd5e1; }

/* SVG Node Visuals */
.node-trapezoid { vector-effect: non-scaling-stroke; }
.node-slim { vector-effect: non-scaling-stroke; }
.stacked-shadow { pointer-events: none; }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node tools/visualization/papernn_studio/tests/editor.test.cjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/papernn_studio/backbone.css tools/visualization/papernn_studio/tests/editor.test.cjs
git commit -m "feat(papernn): add styles for categorized component palette and geometric SVG shapes"
```

---

### Task 5: End-to-End Regression & Comprehensive Documentation

**Files:**
- Modify: `tools/visualization/papernn_studio/README.md`
- Test: All test suites (`node tests/*.test.cjs`, `pytest tests/test_reference_figure.py`)

**Interfaces:**
- Consumes: Complete atomic PaperNN Studio.
- Produces: Updated Vietnamese user documentation explaining how to construct custom neural networks from atomic components without pre-packaged templates.

- [ ] **Step 1: Run full regression testing**

Run:
```bash
node tools/visualization/papernn_studio/tests/renderer.test.cjs
node tools/visualization/papernn_studio/tests/paper.test.cjs
node tools/visualization/papernn_studio/tests/editor.test.cjs
python3 -m pytest tests/test_reference_figure.py -q
```
Expected: ALL PASS.

- [ ] **Step 2: Update documentation**

In `tools/visualization/papernn_studio/README.md`:
Document:
1. Philosophy of Atomic Primitives (Hộp đồ chơi LEGO Deep Learning).
2. Table of 8 visual shape categories and how they render.
3. How to use the component drawer to construct any custom network (CNN, Transformer, U-Net, Diffusion, MoE).
4. Exporting clean vector SVG and publication-ready 600 DPI PNGs.

- [ ] **Step 3: Commit**

```bash
git add tools/visualization/papernn_studio/README.md
git commit -m "docs(papernn): document atomic primitives library and custom architecture design workflow"
```

'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const paper = require('../Paper.js');
const bundle = () => JSON.parse(fs.readFileSync(path.join(__dirname, '../mobilepixornext.json'), 'utf8'));

test('publication figure uses three panels, plain vector shapes, and physical width', () => {
    const result = paper.renderFigure(bundle());
    assert.match(result.svg, /width="182mm"/);
    assert.match(result.svg, /\(a\)/);
    assert.match(result.svg, /\(b\)/);
    assert.match(result.svg, /\(c\)/);
    assert.match(result.svg, /48 × 200 × 176/);
    assert.match(result.svg, /16 × 200 × 176/);
    assert.doesNotMatch(result.svg, /polygon|linearGradient|filter=|<image|NeXt block ×|logarithmic/);
    assert.match(result.svg, /font-family="Arial, Helvetica, sans-serif"/);
});

test('every encoder and lateral/top-down connection remains represented', () => {
    const data = bundle();
    const result = paper.renderFigure(data);
    for (const n of data.views.overview.nodes) assert.ok(result.svg.includes(`data-node="${n.id}"`), n.id);
    for (const edge of data.views.overview.edges) {
        assert.ok(result.svg.includes(`data-source="${edge.source}" data-target="${edge.target}"`), JSON.stringify(edge));
    }
    assert.match(result.svg, /48 → 120/);
    assert.match(result.svg, /120 → 48/);
    assert.match(result.svg, /DW 7×7/);
    assert.match(result.svg, /γ/);
});

test('all detail views are 2D and render without changing the bundle', () => {
    const data = bundle(), before = JSON.stringify(data);
    for (const graph of Object.values(data.views)) {
        const result = paper.render(graph);
        assert.ok(result.width > 100 && result.height > 50);
        assert.doesNotMatch(result.svg, /polygon|linearGradient|filter=|<image/);
        for (const e of graph.edges) assert.ok(result.svg.includes(`data-source="${e.source}" data-target="${e.target}"`));
    }
    assert.equal(JSON.stringify(data), before);
});

test('untrusted labels are escaped and unknown schemas are rejected', () => {
    const data = bundle();
    data.views.overview.nodes[0].label = '<script>alert(1)</script>';
    assert.doesNotMatch(paper.renderFigure(data).svg, /<script>/);
    assert.match(paper.renderFigure(data).svg, /&lt;script&gt;/);
    data.schema_version = 2;
    assert.throws(() => paper.renderFigure(data), /schema_version/);
});

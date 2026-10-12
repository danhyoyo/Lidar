const test = require('node:test');
const assert = require('node:assert/strict');
const renderer = require('../Backbone.js');

function graph() {
    return {title: 'Residual <block>', description: 'C × H × W', nodes: [
        {id: 'x', label: 'X <script>', shape: [48, 200, 176], col: 0, row: 1, kind: 'input'},
        {id: 'dw', label: 'DW 7×7', shape: [48, 200, 176], col: 1, row: 1, kind: 'depthwise'},
        {id: 'add', label: 'Residual add', shape: [48, 200, 176], col: 2, row: 1, kind: 'add'}
    ], edges: [{source: 'x', target: 'dw'}, {source: 'dw', target: 'add'},
              {source: 'x', target: 'add', kind: 'skip', label: 'identity'}]};
}

test('export contains vector faces, tensor labels, and residual wiring', () => {
    const result = renderer.render(graph());
    assert.match(result.svg, /xmlns="http:\/\/www.w3.org\/2000\/svg"/);
    assert.match(result.svg, /48 × 200 × 176/);
    assert.match(result.svg, /data-node="dw"/);
    assert.match(result.svg, /data-source="x" data-target="add"/);
    assert.match(result.svg, /polygon/);
    assert.ok(result.width > 400 && result.height > 300);
});

test('all text and attributes are escaped when importing external graphs', () => {
    const result = renderer.render(graph());
    assert.match(result.svg, /Residual &lt;block&gt;/);
    assert.match(result.svg, /X &lt;script&gt;/);
    assert.doesNotMatch(result.svg, /<script>/);
    assert.doesNotMatch(result.svg, /https?:\/\/[^w]/);
});

test('unknown edge endpoints, duplicate nodes and invalid geometry fail clearly', () => {
    const g = graph();
    g.edges.push({source: 'missing', target: 'add'});
    assert.throws(() => renderer.render(g), /endpoint/);
    const duplicate = graph();
    duplicate.nodes.push({...duplicate.nodes[0]});
    assert.throws(() => renderer.render(duplicate), /duplicate/);
    const invalid = graph();
    invalid.nodes[0].shape = [0, 200, 176];
    assert.throws(() => renderer.render(invalid), /shape/);
    invalid.nodes[0].shape = [48, 200, 176];
    invalid.nodes[0].col = Infinity;
    assert.throws(() => renderer.render(invalid), /position/);
});

test('style controls alter geometry and remain embedded in standalone export', () => {
    const normal = renderer.render(graph());
    const spaced = renderer.render(graph(), {spacing: 1.6, fontSize: 18, theme: 'dark'});
    assert.ok(spaced.width > normal.width);
    assert.match(spaced.svg, /#111827/);
    assert.match(spaced.svg, /font-size="18"/);
    assert.throws(() => renderer.render(graph(), {spacing: NaN}), /spacing/);
});

test('bundle validates version and links between detail views', () => {
    const bundle = {schema_version: 1, views: {overview: graph()}};
    renderer.validateBundle(bundle);
    bundle.views.overview.nodes[0].detail_view = 'unknown';
    assert.throws(() => renderer.validateBundle(bundle), /detail_view/);
    bundle.schema_version = 2;
    assert.throws(() => renderer.validateBundle(bundle), /schema_version/);
});

test('explicit branch routing separates paths and preserves chosen operator symbols', () => {
    const g = graph();
    g.edges[0].route_row = 0;
    g.nodes[1].symbol = 'β';
    const result = renderer.render(g);
    assert.match(result.svg, />β<\/text>/);
    assert.ok(result.height > 500);
    g.edges[0].route_row = NaN;
    assert.throws(() => renderer.render(g), /route_row/);
});

/* NN-SVG backbone renderer. MIT; see LICENSE and UPSTREAM.md.
 * Dimension scaling adapted from AlexNet.js (Alexander Lenail, 2018).
 * Native SVG adds branched graphs, operation nodes and portable text labels.
 */
(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    else root.BackboneSVG = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    const COLORS = Object.freeze({
        input: '#64748b', output: '#2563eb', feature: '#64748b', embedding: '#818cf8',
        conv: '#5b7fd6', conv1d: '#5b7fd6', depthwise: '#cf8635', pointwise: '#60a5fa', deconv: '#0284c7', linear: '#3b82f6',
        norm: '#67a09b', layernorm: '#67a09b', rmsnorm: '#0d9488', groupnorm: '#14b8a6',
        activation: '#c97f91', relu: '#f43f5e', gelu: '#fb7185', silu: '#e11d48', sigmoid: '#fda4af', tanh: '#f43f5e',
        pool: '#7d95a5', maxpool: '#64748b', avgpool: '#94a3b8', upsample: '#4a96b8', pixelshuffle: '#38bdf8',
        add: '#399d89', multiply: '#399d89', concat: '#2563eb', split: '#64748b', matmul: '#8b5cf6', scale: '#a88435', gate: '#399d89', softmax: '#e11d48',
        block: '#8b6bc1', attention: '#b369ad', mhsa: '#7c3aed', qkv: '#a855f7', ffn: '#d97706', rope: '#c084fc', moe: '#ea580c', reshape: '#475569'
    });
    const escape = value => String(value).replace(/[&<>"']/g, ch =>
        ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;'}[ch]));

    function validateGraph(graph) {
        if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges) ||
            !graph.nodes.length || graph.nodes.length > 500 || graph.edges.length > 4000) {
            throw new Error('Graph must contain 1–500 nodes and at most 4000 edges.');
        }
        const ids = new Set();
        for (const n of graph.nodes) {
            if (typeof n.id !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(n.id)) throw new Error('Invalid node id.');
            if (ids.has(n.id)) throw new Error(`duplicate node id: ${n.id}`);
            ids.add(n.id);
            if (!Array.isArray(n.shape) || n.shape.length !== 3 ||
                n.shape.some(d => !Number.isSafeInteger(d) || d < 1 || d > 1000000)) {
                throw new Error(`Invalid C,H,W shape for ${n.id}.`);
            }
            if (![n.col, n.row].every(v => Number.isFinite(v) && v >= 0 && v <= 1000)) {
                throw new Error(`Invalid position for ${n.id}.`);
            }
            if (typeof n.label !== 'string' || n.label.length > 500) throw new Error('Invalid node label.');
            if (n.kind && !Object.hasOwn(COLORS, n.kind)) throw new Error(`Unknown node kind: ${n.kind}`);
            if (n.symbol !== undefined && (typeof n.symbol !== 'string' || n.symbol.length > 8)) throw new Error('Invalid symbol.');
        }
        for (const e of graph.edges) {
            if (!ids.has(e.source) || !ids.has(e.target)) throw new Error(`Unknown edge endpoint: ${e.source} → ${e.target}`);
            if (e.route_row !== undefined && (!Number.isFinite(e.route_row) || e.route_row < 0 || e.route_row > 1000)) {
                throw new Error('Invalid edge route_row.');
            }
        }
    }

    function validateBundle(bundle) {
        if (!bundle || bundle.schema_version !== 1) throw new Error('Unsupported schema_version; expected 1.');
        if (!bundle.views || !bundle.views.overview || Object.keys(bundle.views).length > 100) {
            throw new Error('Bundle must include views.overview (at most 100 views).');
        }
        for (const [id, graph] of Object.entries(bundle.views)) {
            if (!/^[a-zA-Z0-9_-]+$/.test(id)) throw new Error('Invalid view id.');
            validateGraph(graph);
            for (const n of graph.nodes) {
                if (n.detail_view && !Object.hasOwn(bundle.views, n.detail_view)) {
                    throw new Error(`Unknown detail_view: ${n.detail_view}`);
                }
            }
        }
        return bundle;
    }

    function wrap(value, limit) {
        const words = String(value || '').split(/\s+/);
        const lines = [''];
        for (const word of words) {
            const i = lines.length - 1;
            if (lines[i] && lines[i].length + word.length + 1 > limit) lines.push(word);
            else lines[i] += (lines[i] ? ' ' : '') + word;
        }
        return lines;
    }

    function mix(color, target, fraction) {
        const a = color.slice(1).match(/../g).map(x => parseInt(x, 16));
        const b = target.slice(1).match(/../g).map(x => parseInt(x, 16));
        return '#' + a.map((x, i) => Math.round(x * (1-fraction) + b[i] * fraction)
            .toString(16).padStart(2, '0')).join('');
    }

    function render(graph, supplied = {}) {
        validateGraph(graph);
        const options = {spacing: 1, fontSize: 15, depth: 1, theme: 'light', shapes: true, ...supplied};
        for (const [key, min, max] of [['spacing', 0.6, 2], ['fontSize', 11, 24], ['depth', 0, 2]]) {
            if (!Number.isFinite(options[key]) || options[key] < min || options[key] > max) {
                throw new Error(`Invalid ${key}; expected ${min}–${max}.`);
            }
        }
        if (!['light', 'dark'].includes(options.theme)) throw new Error('Invalid theme.');
        const palette = {...COLORS};
        for (const [key, value] of Object.entries(options.colors || {})) {
            if (!Object.hasOwn(palette, key) || !/^#[0-9a-fA-F]{6}$/.test(value)) throw new Error('Invalid color.');
            palette[key] = value;
        }
        const dark = options.theme === 'dark';
        const background = dark ? '#111827' : '#ffffff';
        const ink = dark ? '#edf2f7' : '#243247';
        const muted = dark ? '#aab7ca' : '#6b7a90';
        const wire = dark ? '#aab7ca' : '#7f8da2';
        const cellW = 260 * options.spacing;
        const cellH = 275;
        const rows = [...graph.nodes.map(n => n.row), ...graph.edges.filter(e => e.route_row !== undefined).map(e => e.route_row)];
        const minRow = Math.min(...rows);
        const maxRow = Math.max(...rows);
        const maxCol = Math.max(...graph.nodes.map(n => n.col));
        const width = Math.max(860, 340 + maxCol * cellW);
        const height = 460 + (maxRow - minRow) * cellH;
        // Adapted from NN-SVG AlexNet: depthFn = log(C)*depthScale,
        // widthFn = log(W)*widthScale. These are visual scales, not metric axes.
        const depthFn = c => Math.max(8, Math.log(c) * 10) * options.depth;
        const widthFn = v => Math.max(26, Math.log(v) * 17);
        const positions = new Map(graph.nodes.map(n => [n.id, {
            x: 160 + n.col * cellW, y: 205 + (n.row-minRow) * cellH,
            w: widthFn(n.shape[2]), h: widthFn(n.shape[1]), d: depthFn(n.shape[0])
        }]));
        const text = (x, y, value, size = options.fontSize, attrs = '') =>
            `<text x="${x}" y="${y}" font-size="${size}" fill="${ink}" ${attrs}>${escape(value)}</text>`;
        const arrow = `<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9" fill="${wire}"/></marker></defs>`;
        let body = `<rect width="${width}" height="${height}" fill="${background}"/>`;
        body += text(38, 38, graph.title || 'Backbone architecture', 23, 'font-weight="650"');
        const description = wrap(graph.description || '', Math.floor((width-76)/7.3));
        description.slice(0, 2).forEach((line, i) => {
            body += `<text x="38" y="${63+i*19}" font-size="13" fill="${muted}">${escape(line)}</text>`;
        });
        if (graph.nodes.some(n => n.id === 'stage4') && graph.nodes.some(n => n.id === 'p4')) {
            for (const [y, bandH] of [[93, 250], [640, 250]]) {
                body += `<rect x="25" y="${y}" width="${width-50}" height="${bandH}" rx="16" fill="${dark ? '#172133' : '#f7f9fc'}"/>`;
            }
        }
        let skipIndex = 0;
        for (const e of graph.edges) {
            const a = positions.get(e.source), b = positions.get(e.target);
            let points, labelX, labelY;
            const sameRow = a.y === b.y;
            const sameCol = a.x === b.x;
            const dir = Math.sign(b.x-a.x) || 1;
            const ax = a.x + dir * (a.w/2 + a.d*0.55 + 10);
            const bx = b.x - dir * (b.w/2 + b.d*0.55 + 12);
            if (e.route_row !== undefined) {
                const lane = 205 + (e.route_row-minRow)*cellH;
                points = [[ax, a.y], [ax+16*dir, a.y], [ax+16*dir, lane],
                    [bx-16*dir, lane], [bx-16*dir, b.y], [bx, b.y]];
                labelX = (ax+bx)/2; labelY = lane-8;
            } else if (sameRow && e.kind !== 'skip') {
                points = [[ax, a.y], [bx, b.y]];
                labelX = (ax+bx)/2; labelY = a.y-11;
            } else if (sameCol && e.kind !== 'skip') {
                // Route vertical branches beside tensor labels and titles.
                const x1 = a.x+a.w/2+a.d/2+10;
                const x2 = b.x+b.w/2+b.d/2+12;
                const lane = Math.max(x1, x2)+26;
                points = [[x1, a.y], [lane, a.y], [lane, b.y], [x2, b.y]];
                labelX = lane+4; labelY = (a.y+b.y)/2;
            } else if (e.kind === 'skip') {
                const lane = Math.max(a.y, b.y)+140+(skipIndex++ % 3)*18;
                const sx = a.x-a.w/2-a.d/2-10;
                const tx = b.x+b.w/2+b.d/2+12;
                points = [[sx, a.y], [sx-16, a.y], [sx-16, lane],
                    [tx+16, lane], [tx+16, b.y], [tx, b.y]];
                labelX = (sx+tx)/2; labelY = lane-7;
            } else {
                const midX = (ax+bx)/2;
                points = [[ax, a.y], [midX, a.y], [midX, b.y], [bx, b.y]];
                labelX = midX+8; labelY = (a.y+b.y)/2;
            }
            const d = points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
            const stroke = e.kind === 'lateral' ? palette.gate : wire;
            body += `<g data-source="${escape(e.source)}" data-target="${escape(e.target)}"><path d="${d}" fill="none" stroke="${stroke}" stroke-width="1.6" ${e.kind === 'skip' ? 'stroke-dasharray="6 4"' : ''} marker-end="url(#arrow)"/>`;
            if (e.label) body += `<text x="${labelX}" y="${labelY}" text-anchor="middle" font-size="11" fill="${muted}" stroke="${background}" stroke-width="3" paint-order="stroke">${escape(e.label)}</text>`;
            body += '</g>';
        }
        for (const n of graph.nodes) {
            const p = positions.get(n.id);
            const color = palette[n.kind || 'conv'];
            const x = -p.w/2-p.d/2, y = -p.h/2+p.d*0.28;
            const dx = p.d, dy = -p.d*0.56;
            const right = x+p.w, bottom = y+p.h;
            const pts = arr => arr.map(([a,b]) => `${a.toFixed(2)},${b.toFixed(2)}`).join(' ');
            const title = [n.label, `B × ${n.shape.join(' × ')}`, n.source, n.formula,
                ...(n.operations || [])].filter(Boolean).join('\n');
            body += `<g class="tensor-node" data-node="${escape(n.id)}" transform="translate(${p.x} ${p.y})" tabindex="0" role="button" aria-label="${escape(n.label)}"><title>${escape(title)}</title>`;
            body += `<rect class="selection-outline" x="-114" y="-91" width="228" height="184" rx="12" fill="none" stroke="${color}" stroke-width="2.5" opacity="0"/>`;
            body += `<polygon points="${pts([[x,y],[x+dx,y+dy],[right+dx,y+dy],[right,y]])}" fill="${mix(color, background, 0.57)}" stroke="${color}"/>`;
            body += `<polygon points="${pts([[right,y],[right+dx,y+dy],[right+dx,bottom+dy],[right,bottom]])}" fill="${mix(color, background, 0.22)}" stroke="${color}"/>`;
            body += `<rect x="${x}" y="${y}" width="${p.w}" height="${p.h}" fill="${mix(color, background, 0.78)}" stroke="${color}" stroke-width="1.4"/>`;
            const symbol = n.symbol || {add: '+', multiply: '×', gate: 'G', scale: 'γ', upsample: '↑2', pool: 'μ'}[n.kind];
            if (symbol) body += text(x+p.w/2, y+p.h/2+9, symbol, 28, `text-anchor="middle" font-weight="500"`);
            const labelLines = wrap(n.label, Math.floor(cellW / (options.fontSize * 0.56)));
            labelLines.forEach((line, i) => {
                body += text(0, -82-(labelLines.length-1-i)*(options.fontSize+3), line,
                    options.fontSize, 'text-anchor="middle" font-weight="600"');
            });
            if (options.shapes) body += text(0, 103, n.shape.join(' × '), options.fontSize-1, 'text-anchor="middle"');
            if (n.repeat) body += `<text x="0" y="124" text-anchor="middle" font-size="12" fill="${muted}">NeXt block × ${n.repeat}</text>`;
            else if (n.detail_view) body += `<text x="0" y="124" text-anchor="middle" font-size="11" fill="${muted}">↗ ${escape(n.detail_view)}</text>`;
            body += '</g>';
        }
        const kinds = [...new Set(graph.nodes.map(n => n.kind || 'conv'))];
        let legendX = 38;
        for (const kind of kinds) {
            if (legendX+140 > width-38) break;
            body += `<rect x="${legendX}" y="${height-47}" width="10" height="10" rx="2" fill="${palette[kind]}"/>`;
            body += `<text x="${legendX+17}" y="${height-38}" font-size="11" fill="${muted}">${escape(kind)}</text>`;
            legendX += 35 + kind.length*6.5;
        }
        body += `<text x="38" y="${height-16}" font-size="11" fill="${muted}">B omitted · C × H × W · logarithmic visual scaling · dashed line = skip / residual</text>`;
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="Inter, DejaVu Sans, Arial, sans-serif"><title>${escape(graph.title)}</title>${arrow}${body}</svg>`;
        return {svg, width, height};
    }

    return {render, validateGraph, validateBundle, colors: COLORS};
}));

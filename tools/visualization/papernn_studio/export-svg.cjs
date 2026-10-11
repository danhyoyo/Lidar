#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {render, validateBundle} = require('./Backbone.js');
const paper = require('./Paper.js');
const editor = require('./Editor.js');
const reference = require('./Reference.js');
const args = process.argv.slice(2);
const style = args.length === 4 && args[2] === '--style' ? args[3] : '3d';

if (![2,4].includes(args.length) || (args.length === 4 && (args[2] !== '--style' || !['3d','paper','reference','mono'].includes(style)))) {
    console.error('Usage: node export-svg.cjs bundle.json output-directory [--style 3d|paper|reference|mono]');
    process.exit(2);
}
try {
    let bundle = validateBundle(JSON.parse(fs.readFileSync(process.argv[2], 'utf8')));
    if (['reference', 'mono'].includes(style)) bundle = editor.create(bundle);
    fs.mkdirSync(process.argv[3], {recursive: true});
    for (const [id, graph] of Object.entries(bundle.views)) {
        fs.writeFileSync(path.join(process.argv[3], `${id}.svg`), (['reference','mono'].includes(style) ? reference.render(graph, {monochrome: style === 'mono'}) : style === 'paper' ? (id === 'overview' ? paper.renderFigure(bundle) : paper.render(graph)) : render(graph)).svg + '\n');
    }
    console.log(`Rendered ${Object.keys(bundle.views).length} ${style} SVG figures.`);
} catch (error) {
    console.error(`Cannot render backbone: ${error.message}`);
    process.exit(2);
}

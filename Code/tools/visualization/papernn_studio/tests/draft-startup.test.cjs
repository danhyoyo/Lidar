/* Exercise the actual UI startup against saved drawings, without a browser. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const editor = require('../Editor.js');
const studio = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(studio, 'backbone-ui.js'), 'utf8');
const preset = JSON.parse(fs.readFileSync(path.join(studio, 'mobilepixornext.json'), 'utf8'));
const legacy = JSON.parse(fs.readFileSync(path.join(studio, 'presets/legacy_rich8.json'), 'utf8'));
const keyExpression = source.match(/const DRAFT_KEY=(.*), mobile=matchMedia/)[1];
const bootstrap = source.slice(source.lastIndexOf('    try{\n        let saved;'), source.lastIndexOf('\n}());'));
const oldKey = 'nn-svg-backbone-editor-v1';

function start(entries) {
    const storage = new Map(entries);
    let loaded;
    const context = {
        window: {BACKBONE_DIAGRAM: preset},
        E: editor,
        localStorage: {getItem: key => storage.get(key) ?? null},
        load: (document, presentation) => { loaded = {document, presentation}; },
        value: () => loaded.presentation,
        doc: () => loaded.document,
        changeView: () => {},
        error: message => { throw Error(message); },
        lastPresentation: '',
    };
    vm.createContext(context);
    vm.runInContext('const DRAFT_KEY=' + keyExpression + ';\n' + bootstrap, context);
    return {loaded, storage};
}

const oldDraft = JSON.stringify({document: editor.create(legacy), view: 'overview', presentation: 'reference'});
const restored = start([[oldKey, oldDraft]]);
assert.equal(restored.loaded.document.options.backbone_out_dim, 32, 'Old output16 draft must not replace P16 on startup');
assert.equal(restored.loaded.document.options.neck_fusion_channels, 32);
assert.equal(restored.loaded.document.options.local_attention, 'simam');
assert.equal(restored.storage.get(oldKey), oldDraft, 'Keep the old drawing available for recovery');

const keyContext = {window: {BACKBONE_DIAGRAM: preset}};
const currentKey = vm.runInNewContext(keyExpression, keyContext);
const edited = editor.create(preset);
edited.views.overview.nodes.find(n => n.id === 'stage2').caption = 'My saved stage';
const saved = JSON.stringify({document: edited, view: 'overview', presentation: 'paper'});
const current = start([[oldKey, oldDraft], [currentKey, saved]]);
assert.equal(current.loaded.document.views.overview.nodes.find(n => n.id === 'stage2').caption, 'My saved stage');
assert.equal(current.loaded.presentation, 'paper');
assert.equal(start([[currentKey, '{invalid json']]).loaded.document.options.backbone_out_dim, 32);
console.log('Draft startup: old 24/16 drawing preserved; P16 32/32 loads; current edits resume; corrupt drafts fall back to P16.');

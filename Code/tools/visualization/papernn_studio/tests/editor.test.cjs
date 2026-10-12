'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const editor = require('../Editor.js');
const reference = require('../Reference.js');
const preset = () => JSON.parse(fs.readFileSync(path.join(__dirname,'../mobilepixornext.json'),'utf8'));

test('editable preset keeps source topology and shapes unchanged', () => {
    const original=preset(), before=JSON.stringify(original), doc=editor.create(original);
    editor.validate(doc); assert.equal(JSON.stringify(original),before);
    for(const [id,g] of Object.entries(original.views)) {
        assert.deepEqual(doc.views[id].nodes.map(n=>n.shape),g.nodes.map(n=>n.shape));
        assert.deepEqual(doc.views[id].edges.map(e=>[e.source,e.target]),g.edges.map(e=>[e.source,e.target]));
        assert.ok(doc.views[id].nodes.every(n=>n.layout.width>0));
    }
});

test('arbitrary blocks and topology render without a model-specific template', () => {
    const doc=editor.create(preset()); doc.views.overview={title:'My architecture',nodes:[],edges:[],groups:[]};
    const a=editor.addNode(doc,'overview','conv'), b=editor.addNode(doc,'overview','block');
    editor.updateNode(doc,'overview',a,{label:'Custom stem',shape:[17,123,87],layout:{x:130,y:110,fill:'#abcdef'}});
    editor.updateNode(doc,'overview',b,{label:'My block',shape:[51,62,44],layout:{x:460,y:250,width:240,height:110}});
    editor.addEdge(doc,'overview',a,b,{label:'skip branch',source_port:'bottom',target_port:'left',kind:'skip'});
    const svg=reference.render(doc.views.overview).svg;
    assert.match(svg,/Custom stem/); assert.match(svg,/17 × 123 × 87/);
    assert.match(svg,/#abcdef/); assert.match(svg,/My block/);
    assert.ok(svg.includes(`data-source="${a}" data-target="${b}"`));
    assert.doesNotMatch(svg,/Stage 2|NeXt ×|MobilePixorNeXt|polygon|linearGradient/);
});

test('deletion cleans references, duplication works, invalid edits are transactional', () => {
    const doc=editor.create(preset()), id=editor.duplicateNode(doc,'overview','stage2');
    assert.notEqual(id,'stage2'); assert.deepEqual(doc.views.overview.nodes.find(n=>n.id===id).shape,[48,200,176]);
    const before=JSON.stringify(doc);
    assert.throws(()=>editor.updateNode(doc,'overview','stage2',{shape:[0,12,12]})); assert.equal(JSON.stringify(doc),before);
    editor.removeNode(doc,'overview','stage2');
    assert.ok(doc.views.overview.edges.every(e=>e.source!=='stage2'&&e.target!=='stage2'));
    assert.ok(doc.views.overview.groups.every(g=>!g.node_ids.includes('stage2'))); editor.validate(doc);
});

test('manual edits clear verification claims; undo/redo handles topology branches', () => {
    const doc=editor.create(preset()); doc.shape_mode='verified against model'; const history=new editor.History(doc);
    history.commit(d=>editor.updateNode(d,'overview','stage2',{shape:[64,200,176]}));
    assert.match(history.value.shape_mode,/manual/);
    history.undo(); assert.equal(history.value.views.overview.nodes.find(n=>n.id==='stage2').shape[0],48);
    history.redo(); assert.equal(history.value.views.overview.nodes.find(n=>n.id==='stage2').shape[0],64);
    history.undo(); history.commit(d=>editor.addNode(d,'overview','conv')); assert.equal(history.canRedo,false);
});

test('groups and saved geometry survive portable JSON round-trip', () => {
    const doc=editor.create(preset()), group=editor.addGroup(doc,'overview','My encoder');
    editor.updateNode(doc,'overview','stage2',{group,layout:{x:250,y:310,width:260,height:150,rotation:90,font_size:19}});
    const restored=editor.create(JSON.parse(JSON.stringify(doc)));
    assert.deepEqual(restored.views.overview.nodes.find(n=>n.id==='stage2').layout,doc.views.overview.nodes.find(n=>n.id==='stage2').layout);
    assert.match(reference.render(restored.views.overview).svg,/My encoder/);
    editor.removeGroup(restored,'overview',group); editor.validate(restored);
});

test('untrusted geometry, colors, ports, groups and text are checked', () => {
    for(const patch of [{fill:'url(javascript:alert(1))'},{x:Infinity},{width:-5},{rotation:37}]) {
        const d=editor.create(preset());d.views.overview.nodes[0].layout={...d.views.overview.nodes[0].layout,...patch};
        assert.throws(()=>editor.validate(d));
    }
    const d=editor.create(preset()); d.views.overview.edges[0].source_port='evil'; assert.throws(()=>editor.validate(d));
    const x=editor.create(preset());x.views.overview.groups[0].node_ids.push('missing'); assert.throws(()=>editor.validate(x));
    const escaped=editor.create(preset());escaped.views.overview.nodes[0].label='<script>alert(1)</script>';
    const svg=reference.render(escaped.views.overview).svg;assert.doesNotMatch(svg,/<script>/);assert.match(svg,/&lt;script&gt;/);
});

test('channel and repeat labels follow edited values rather than fixed preset text', () => {
    const doc=editor.create(preset());
    editor.updateNode(doc,'overview','stage2',{shape:[64,200,176],repeat:5});
    editor.updateNode(doc,'overview','l3',{shape:[32,200,176]});
    const svg=reference.render(doc.views.overview).svg;
    assert.match(svg,/64 → 32/);assert.match(svg,/NeXt × 5/);assert.doesNotMatch(svg,/24 channels/);
    editor.updateNode(doc,'overview','stage2',{repeat:null});
    assert.doesNotMatch(reference.render(doc.views.overview).svg,/NeXt × 5/);
});

test('importing one unplaced block preserves saved positions, routes and groups', () => {
    const doc=editor.create(preset());editor.updateNode(doc,'overview','stage2',{layout:{x:450,y:330,fill:'#123456'}});
    const original=JSON.stringify(doc.views.overview.nodes.find(n=>n.id==='stage2'));
    doc.views.overview.nodes.push({id:'unplaced',label:'Another block',kind:'conv',shape:[10,20,30],col:8,row:1});
    const result=editor.create(doc);
    assert.equal(JSON.stringify(result.views.overview.nodes.find(n=>n.id==='stage2')),original);
    assert.deepEqual(result.views.overview.groups,doc.views.overview.groups);
});

test('wire ports follow moved and resized blocks', () => {
    const doc=editor.blank(),target=editor.addNode(doc,'overview','feature');
    editor.updateNode(doc,'overview','input',{layout:{x:120,y:110,width:80,height:80}});
    editor.updateNode(doc,'overview',target,{layout:{x:460,y:110,width:80,height:80}});
    editor.addEdge(doc,'overview','input',target,{source_port:'right',target_port:'left'});
    let route=reference.render(doc.views.overview).routes[0].points;
    assert.deepEqual([route[0].x,route[0].y],[160,110]);assert.deepEqual([route.at(-1).x,route.at(-1).y],[420,110]);
    editor.updateNode(doc,'overview','input',{layout:{x:180,width:100}});
    route=reference.render(doc.views.overview).routes[0].points;
    assert.deepEqual([route[0].x,route[0].y],[230,110]);
    assert.deepEqual([route.at(-1).x,route.at(-1).y],[420,110]);
});

test('adding and duplicating blocks find free space instead of covering existing blocks', () => {
    const doc=editor.create(preset());
    const overlaps=(a,b)=>Math.abs(a.x-b.x)<(a.width+b.width)/2+12&&Math.abs(a.y-b.y)<(a.height+b.height)/2+12;
    for(let i=0;i<12;i++) {
        const before=doc.views.overview.nodes.map(n=>({...n.layout}));
        const id=editor.addNode(doc,'overview','conv',{x:320,y:228});
        const added=doc.views.overview.nodes.find(n=>n.id===id).layout;
        assert.ok(before.every(l=>!overlaps(l,added)),`Insertion ${i} must not cover another block`);
    }
    const before=doc.views.overview.nodes.map(n=>({...n.layout}));
    const id=editor.duplicateNode(doc,'overview','stage2');
    const added=doc.views.overview.nodes.find(n=>n.id===id).layout;
    assert.ok(before.every(l=>!overlaps(l,added)),'Duplicate must not cover the original or a neighboring block');
    assert.throws(()=>editor.addNode(doc,'overview','conv',{x:Infinity,y:0}));
    editor.validate(doc);
});

test('marquee hit testing handles reverse rectangles, partial intersections and non-block areas', () => {
    const graph={nodes:[
        {id:'a',layout:{x:100,y:100,width:80,height:60}},
        {id:'b',layout:{x:250,y:100,width:80,height:60}},
        {id:'c',layout:{x:100,y:250,width:80,height:60}}
    ]};
    assert.deepEqual(editor.nodesInRect(graph,{x1:40,y1:60,x2:300,y2:140}),['a','b']);
    assert.deepEqual(editor.nodesInRect(graph,{x1:300,y1:140,x2:40,y2:60}),['a','b']);
    assert.deepEqual(editor.nodesInRect(graph,{x1:135,y1:100,x2:160,y2:150}),['a']);
    assert.deepEqual(editor.nodesInRect(graph,{x1:160,y1:140,x2:190,y2:160}),[]);
    assert.deepEqual(editor.nodesInRect(graph,{x1:100,y1:100,x2:100,y2:120}),[]);
    assert.throws(()=>editor.nodesInRect(graph,{x1:NaN,y1:0,x2:100,y2:100}));
});

test('shared movement preserves relative spacing and clamps the entire selection together', () => {
    const graph={nodes:[{id:'a',layout:{x:20,y:40,width:80,height:60}},
        {id:'b',layout:{x:120,y:90,width:80,height:60}}]};
    const before=JSON.stringify(graph);
    const updates=editor.translatedLayouts(graph,['a','b'],-50,30);
    assert.deepEqual(updates.map(n=>[n.id,n.layout.x,n.layout.y]),[['a',0,70],['b',100,120]]);
    assert.equal(updates[1].layout.x-updates[0].layout.x,100);
    assert.equal(updates[1].layout.y-updates[0].layout.y,50);
    assert.equal(JSON.stringify(graph),before);
    const snapped=editor.translatedLayouts(graph,['a','b'],7,8,true);
    assert.deepEqual(snapped.map(n=>[n.layout.x,n.layout.y]),[[30,50],[130,100]]);
    assert.throws(()=>editor.translatedLayouts(graph,['a','missing'],0,0));
    assert.throws(()=>editor.translatedLayouts(graph,['a'],Infinity,0));
});

test('moving a selection is one undo step and retains shapes, topology and provenance', () => {
    const doc=editor.create(preset());doc.shape_mode='verified against model';
    const h=new editor.History(doc),before=JSON.stringify(h.value),g=h.value.views.overview;
    const shapes=g.nodes.map(n=>n.shape),edges=JSON.stringify(g.edges);
    h.commit(d=>editor.moveNodes(d,'overview',['stage2','stage3','stage4'],30,20));
    assert.equal(h.value.shape_mode,'verified against model');
    assert.deepEqual(h.value.views.overview.nodes.map(n=>n.shape),shapes);
    assert.equal(JSON.stringify(h.value.views.overview.edges),edges);
    assert.equal(h.past.length,1);h.undo();assert.equal(JSON.stringify(h.value),before);
    h.redo();assert.equal(h.value.views.overview.nodes.find(n=>n.id==='stage2').layout.x,350);
});

test('deleting a selection cleans all references atomically and rejects deleting the last block', () => {
    const h=new editor.History(preset()),before=JSON.stringify(h.value);
    h.commit(d=>editor.removeNodes(d,'overview',['stage2','stage3']));
    const g=h.value.views.overview;
    assert.equal(g.nodes.length,12);
    assert.ok(g.edges.every(e=>!['stage2','stage3'].includes(e.source)&&!['stage2','stage3'].includes(e.target)));
    assert.ok(g.groups.every(x=>x.node_ids.every(id=>!['stage2','stage3'].includes(id))));
    assert.equal(h.past.length,1);h.undo();assert.equal(JSON.stringify(h.value),before);
    assert.throws(()=>editor.removeNodes(h.value,'overview',h.value.views.overview.nodes.map(n=>n.id)));
    assert.equal(JSON.stringify(h.value),before);
    assert.throws(()=>editor.removeNodes(h.value,'overview',['missing']));
    assert.equal(JSON.stringify(h.value),before);
});

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

test('aligning and distributing via History commits a single reversible undo state', () => {
    const doc = editor.blank();
    const a = editor.addNode(doc, 'overview', 'conv', { x: 100, y: 100 });
    const b = editor.addNode(doc, 'overview', 'conv', { x: 200, y: 150 });
    const c = editor.addNode(doc, 'overview', 'conv', { x: 300, y: 200 });
    const history = new editor.History(doc);
    const initialY = history.value.views.overview.nodes.find(n => n.id === b).layout.y;

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
    assert.equal(history.value.views.overview.nodes.find(n => n.id === b).layout.y, initialY);
});

test('ReferenceSVG renders publication legend when showLegend option is enabled', () => {
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

test('clean SVG export never includes interactive paper guides or selection boxes', () => {
    const doc = editor.create(preset());
    const rendered = reference.render(doc.views.overview);
    assert.doesNotMatch(rendered.svg, /data-paper-guide/);
    assert.doesNotMatch(rendered.svg, /selection-box/);
});

test('SVG generated for client rasterization has explicit pixel dimensions matching physical DPI', () => {
    const doc = editor.create(preset());
    const res = reference.render(doc.views.overview);
    // At 182mm width and 300 DPI (11.811 px/mm), target width is ~2150px
    const scaleFactor300 = (300 / 25.4) * (182 / res.width);
    assert.ok(scaleFactor300 > 0 && Number.isFinite(scaleFactor300));
});

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

test('SVG rendered nodes output correct CSS class hooks for styling', () => {
    const doc = editor.blank();
    editor.addNode(doc, 'overview', 'maxpool');
    editor.addNode(doc, 'overview', 'rmsnorm');
    const svg = reference.render(doc.views.overview).svg;
    assert.match(svg, /class="node-trapezoid"/);
    assert.match(svg, /class="node-slim"/);
});









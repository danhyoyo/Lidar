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

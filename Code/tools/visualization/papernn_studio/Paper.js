/* Publication-oriented 2D SVG. MIT; shares the validated NN-SVG graph bundle. */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory(require('./Backbone.js'));
    else root.PaperSVG = factory(root.BackboneSVG);
}(typeof globalThis !== 'undefined' ? globalThis : this, function (core) {
    'use strict';
    const escape = value => String(value).replace(/[&<>"']/g, ch =>
        ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;'}[ch]));
    const shape = n => n.shape.join(' × ');

    function drawing(width, height, title, description) {
        const wires = [], nodes = [], annotations = [];
        const text = (x, y, value, anchor = 'middle', size = 13, weight = 'normal') =>
            `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}">${escape(value)}</text>`;
        function path(points, edge, dashed = false, arrow = true) {
            const attrs = edge ? ` data-source="${escape(edge.source)}" data-target="${escape(edge.target)}"` : '';
            wires.push(`<path${attrs} d="${points.map((p, i) => `${i ? 'L' : 'M'} ${p[0]} ${p[1]}`).join(' ')}" fill="none" stroke="#333333" stroke-width="0.9"${dashed ? ' stroke-dasharray="4 3"' : ''}${arrow ? ' marker-end="url(#paper-arrow)"' : ''}/>`);
        }
        function box(n, x, y, w, h, lines, fill = '#ffffff') {
            const attrs = n ? ` class="tensor-node" data-node="${n.id}" tabindex="0" role="button" aria-label="${escape(n.label)}"` : '';
            nodes.push(`<g${attrs}>${n ? `<title>${escape(n.label)}; ${shape(n)}</title>` : ''}<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="#333333" stroke-width="0.8"/>${lines.map(line => text(x+w/2, y+line[1], line[0], 'middle', line[2] || 13, line[3] || 'normal')).join('')}${n ? `<rect class="selection-outline" x="${x-2}" y="${y-2}" width="${w+4}" height="${h+4}" fill="none" stroke="#46657b" stroke-width="2" opacity="0"/>` : ''}</g>`);
        }
        function circle(n, x, y, symbol, radius = 11) {
            nodes.push(`<g${n ? ` class="tensor-node" data-node="${n.id}" tabindex="0" role="button" aria-label="${escape(n.label)}"` : ''}><circle cx="${x}" cy="${y}" r="${radius}" fill="#ffffff" stroke="#333333" stroke-width="0.9"/>${text(x, y+4.5, symbol, 'middle', 15)}${n ? `<circle class="selection-outline" cx="${x}" cy="${y}" r="${radius+3}" fill="none" stroke="#46657b" stroke-width="2" opacity="0"/>` : ''}</g>`);
        }
        const note = (...args) => annotations.push(text(...args));
        const dot = (x, y) => nodes.push(`<circle cx="${x}" cy="${y}" r="1.7" fill="#333333"/>`);
        function finish() {
            const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="182mm" height="${(182*height/width).toFixed(3)}mm" viewBox="0 0 ${width} ${height}" font-family="Arial, Helvetica, sans-serif" fill="#202020" role="img" aria-labelledby="paper-title paper-description"><title id="paper-title">${escape(title)}</title><desc id="paper-description">${escape(description || '')}</desc><defs><marker id="paper-arrow" viewBox="0 0 6 6" refX="5.6" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 6 3 L 0 6" fill="none" stroke="#333333" stroke-width="0.8"/></marker></defs><rect width="${width}" height="${height}" fill="#ffffff"/>${wires.join('')}${nodes.join('')}${annotations.join('')}</svg>`;
            return {svg, width, height};
        }
        return {text, path, box, circle, note, dot, finish, annotations};
    }

    // Detail views preserve every primitive operation and connection in the bundle.
    // Their physical height follows their topology; wide views may need separate plates.
    function render(graph) {
        core.validateGraph(graph);
        const rows = [...graph.nodes.map(n => n.row), ...graph.edges.filter(e => e.route_row !== undefined).map(e => e.route_row)];
        const minRow = Math.min(...rows), maxRow = Math.max(...rows);
        const stepX = 190, stepY = 136, cardW = 150, cardH = 78;
        const width = 60 + Math.max(...graph.nodes.map(n => n.col))*stepX + cardW;
        const height = 142 + (maxRow-minRow)*stepY + 70;
        const d = drawing(width, height, graph.title, graph.description);
        const positions = new Map(graph.nodes.map(n => [n.id,
            {x: 30+n.col*stepX, y: 62+(n.row-minRow)*stepY, w: cardW, h: cardH}]));
        for (const e of graph.edges) {
            const a = positions.get(e.source), b = positions.get(e.target);
            const ax = a.x+a.w, ay = a.y+a.h/2, bx = b.x, by = b.y+b.h/2;
            let points;
            if (e.kind === 'skip' || e.route_row !== undefined) {
                const lane = e.route_row !== undefined ? 62+(e.route_row-minRow)*stepY+cardH/2 :
                    Math.max(a.y+a.h, b.y+b.h)+30;
                points = [[a.x+a.w/2, a.y+a.h], [a.x+a.w/2, lane], [b.x+b.w/2, lane], [b.x+b.w/2, b.y+b.h]];
            } else if (a.x === b.x) {
                const down = b.y > a.y;
                points = [[a.x+a.w/2, down ? a.y+a.h : a.y], [b.x+b.w/2, down ? b.y : b.y+b.h]];
            } else {
                const right = b.x > a.x;
                const sx = right ? ax : a.x, tx = right ? bx : b.x+b.w;
                const mid = (sx+tx)/2;
                points = [[sx, ay], [mid, ay], [mid, by], [tx, by]];
            }
            d.path(points, e, e.kind === 'skip');
        }
        d.note(30, 28, graph.title, 'start', 15);
        for (const n of graph.nodes) {
            const p = positions.get(n.id);
            const words = n.label.split(/\s+/), lines = [''];
            for (const word of words) {
                const i = lines.length-1;
                if (lines[i] && lines[i].length+word.length > 20) lines.push(word);
                else lines[i] += (lines[i] ? ' ' : '')+word;
            }
            // Titles remain selectable and escaped even for long imported labels.
            d.box(n, p.x, p.y, p.w, p.h, [
                ...lines.slice(0, 3).map((line, i) => [line, 18+i*16]),
                [shape(n), 68, 12]
            ], ['block', 'gate', 'attention'].includes(n.kind) ? '#f1f3f4' : '#ffffff');
        }
        return d.finish();
    }

    function renderFigure(bundle) {
        core.validateBundle(bundle);
        const graph = bundle.views.overview, byId = new Map(graph.nodes.map(n => [n.id, n]));
        const required = ['input','stem','stage2','stage3','stage4','l5','l4','l3','u4','p4','u3','p3','out_conv','output'];
        const supported = new Set([...required, 'local', 'context', 'detail', 'detail_add']);
        if (required.some(id => !byId.has(id)) || graph.nodes.some(n => !supported.has(n.id)) || !bundle.views.block2) {
            return render(graph);
        }
        const blockNodes = new Set(bundle.views.block2.nodes.map(node => node.id));
        if (['input', 'expand', 'project', 'scale', 'add'].some(id => !blockNodes.has(id))) return render(graph);
        const n = id => byId.get(id), opts = bundle.options || {};
        const optionalHeight = n('local') || n('context') ? 48 : 0;
        const detailHeight = n('detail') ? 60 : 0;
        const gateY = 284+optionalHeight, lateralY = 204+optionalHeight;
        const panelY = 375+optionalHeight+detailHeight;
        const d = drawing(720, panelY+180, 'MobilePixorNeXt backbone', graph.description);
        const pos = new Map();
        const addBox = (id, x, y, w, h, lines, fill) => {
            pos.set(id, {x,y,w,h}); d.box(n(id), x,y,w,h,lines,fill);
        };
        const addCircle = (id, x, y, symbol) => {
            pos.set(id, {x:x-11,y:y-11,w:22,h:22}); d.circle(n(id),x,y,symbol);
        };
        d.note(16, 23, '(a) Backbone and feature pyramid', 'start', 14, 'bold');
        addBox('input', 16,54,95,124,[[n('input').label,24],[shape(n('input')),86],[`C × H × W`,111]], '#ffffff');
        addBox('stem',134,54,112,124,[['Stem (/2)',24],['Conv 3×3 × 2',46],['s=2, then s=1',66],['each BN · SiLU',86],[shape(n('stem')),111]]);
        for (const [id,x,stride] of [['stage2',269,4],['stage3',417,8],['stage4',565,16]]) {
            addBox(id,x,54,135,124,[[`Stage ${id.slice(-1)} (/${stride})`,24],['Conv 3×3, s=2',46],
                ['BN · SiLU',66],[`NeXt × ${n(id).repeat}`,86],[shape(n(id)),111]], '#f1f3f4');
        }
        if (n('local')) {
            const label = /ECA/i.test(n('local').label) ? 'Residual ECA' : n('local').label;
            addBox('local',269,188,135,38,[[label,24]], '#f1f3f4');
        }
        if (n('context')) addBox('context',417,188,135,38,[[n('context').label.split(' · ')[0],24]], '#f1f3f4');
        // Laterals are bare 1×1 projections; BN and SiLU occur only in refinements.
        for (const [id,x,source] of [['l3',286,'stage2'],['l4',434,'stage3'],['l5',582,'stage4']]) {
            addBox(id,x,lateralY,101,36,[[`1×1: ${n(source).shape[0]} → ${n(id).shape[0]}`,23]]);
        }
        addBox('u4',512,gateY-28,101,56,[['↑2 · DW 3×3',21],['BN · SiLU',43]]);
        addBox('u3',363,gateY-28,101,56,[['↑2 · Conv 3×3',21],['BN · SiLU',43]]);
        addCircle('p4',484.5,gateY,opts.scale_gated_fpn === false ? '+' : 'G');
        addCircle('p3',336.5,gateY,opts.scale_gated_fpn === false ? '+' : 'G');
        const outY = gateY+detailHeight;
        addBox('out_conv',145,outY-28,112,56,[['Conv 3×3',21],['BN · SiLU',43]]);
        addBox('output',16,outY-28,109,56,[['Output (/4)',21],[shape(n('output')),43]]);
        if (n('detail')) {
            addBox('detail',144,lateralY,114,36,[['Detail projection',23]], '#f1f3f4');
            addCircle('detail_add',336.5,outY,'+');
        }
        // Explicit port routing follows the actual resolved graph, including options.
        for (const e of graph.edges) {
            const a = pos.get(e.source), b = pos.get(e.target);
            const acx=a.x+a.w/2, acy=a.y+a.h/2, bcx=b.x+b.w/2, bcy=b.y+b.h/2;
            let points;
            if (e.target === 'detail_add' && e.source === 'detail') {
                points=[[acx,a.y+a.h],[acx,gateY+34],[bcx+39,gateY+34],[bcx+39,bcy],[b.x+b.w,bcy]];
            } else if (e.target === 'u4' && e.source === 'l5') {
                points=[[acx,a.y+a.h],[acx,bcy],[b.x+b.w,bcy]];
            } else if (Math.abs(acx-bcx)<2) {
                points=[[acx,a.y+a.h],[bcx,b.y]];
            } else if (e.kind === 'lateral' || e.target === 'detail') {
                points=[[acx,a.y+a.h],[acx,(a.y+a.h+b.y)/2],[bcx,(a.y+a.h+b.y)/2],[bcx,b.y]];
            } else {
                const right=bcx>acx, sx=right?a.x+a.w:a.x, tx=right?b.x:b.x+b.w;
                const mid=(sx+tx)/2;
                points=[[sx,acy],[mid,acy],[mid,bcy],[tx,bcy]];
            }
            d.path(points,e);
        }
        if (!n('detail')) d.note(336.5,gateY+48,`P3: ${shape(n('p3'))}`);
        else d.circle(null,270,gateY+34,'γ');
        d.note(484.5,gateY+48,`P4: ${shape(n('p4'))}`);
        d.note(16,panelY-17,`↑2: bilinear resize; align_corners=False.  ${opts.scale_gated_fpn===false?'+':'G'}: fusion in (c).`, 'start');
        if (n('detail')) d.note(16,panelY-35,'Detail path: DW 3×3, s=2 → PW 1×1; both BN + SiLU; learned channel scale.', 'start');

        // One representative stage-2 block; the same topology uses C=96/128 later.
        d.note(16,panelY,`(b) NeXt block (C = ${n('stage2').shape[0]})`, 'start', 14, 'bold');
        const by=panelY+44, bcy=by+27;
        const block=bundle.views.block2, bn=id=>block.nodes.find(x=>x.id===id);
        const hidden=bn('expand').shape[0], channels=bn('input').shape[0];
        const reparam=opts.use_reparam && !opts.deploy;
        d.box(null,53,by,80,54,[[reparam?'RepDW 7×7':'DW 7×7',20],
            [reparam?'3 branches':opts.deploy&&opts.use_reparam?'fused':'BN',40]]);
        d.box(null,149,by,81,54,[['PW 1×1',17],['SiLU',33],[`${channels} → ${hidden}`,49]]);
        d.box(null,246,by,81,54,[['PW 1×1',17],['BN',33],[`${hidden} → ${channels}`,49]]);
        d.circle(null,353,bcy,'γ'); d.circle(null,389,bcy,'+');
        d.note(25,bcy+4,'X'); d.note(417,bcy+4,'Y');
        for(const [a,b] of [[31,53],[133,149],[230,246],[327,342],[364,378],[400,411]]) d.path([[a,bcy],[b,bcy]]);
        d.path([[39,bcy],[39,by+78],[389,by+78],[389,bcy+11]],null,true);
        d.dot(39,bcy);
        d.note(16,by+119,reparam?'RepDW: DW7 + BN, DW3 + BN, identity + BN.':'LayerScale γ: learned per channel, initialized to 10⁻⁵.', 'start', 13);
        // Allow opening the fully expanded primitive view from either panel.
        d.annotations.push(`<rect x="16" y="${panelY+8}" width="414" height="117" fill="transparent" data-view="block2"/>`);

        d.note(452,panelY,opts.scale_gated_fpn===false?'(c) Additive fusion':'(c) Scale-gated fusion', 'start',14,'bold');
        const gy=panelY+66, ly=gy-30, uy=gy+42;
        d.note(454,ly+4,'L'); d.note(454,uy+4,'U');
        if (opts.scale_gated_fpn===false) {
            d.circle(null,620,gy,'+');
            d.path([[466,ly],[620,ly],[620,gy-11]]);
            d.path([[466,uy],[620,uy],[620,gy+11]]);
            d.path([[631,gy],[690,gy]]); d.note(702,gy+4,'P');
            d.note(452,by+119,'P = U + L', 'start');
        } else {
            d.circle(null,482,gy,'+');
            d.box(null,509,gy-17,62,34,[['DW 3×3',22]]);
            d.box(null,587,gy-17,39,34,[['2σ',22]]);
            d.circle(null,649,gy,'⊙'); d.circle(null,649,uy,'+');
            d.path([[466,ly],[482,ly],[482,gy-11]]);
            d.path([[482,ly],[649,ly],[649,gy-11]]);
            d.path([[466,uy],[482,uy],[482,gy+11]]);
            d.path([[482,uy],[638,uy]]);
            for(const [a,b] of [[493,509],[571,587],[626,638]]) d.path([[a,gy],[b,gy]]);
            d.path([[649,gy+11],[649,uy-11]]);
            d.path([[660,uy],[690,uy]]); d.note(702,uy+4,'P');
            d.dot(482,ly); d.dot(482,uy);
            d.note(452,by+119,'P = U + 2σ(DW(L + U)) ⊙ L', 'start');
        }
        d.annotations.push(`<rect x="446" y="${panelY+8}" width="262" height="117" fill="transparent" data-view="gate4"/>`);
        return d.finish();
    }
    return {render, renderFigure};
}));

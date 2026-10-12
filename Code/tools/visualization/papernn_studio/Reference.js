/* Data-driven paper diagrams: editable blocks, orthogonal wires and group frames. */
(function(root,factory){
    if(typeof module==='object'&&module.exports)module.exports=factory(require('./Editor.js'));
    else root.ReferenceSVG=factory(root.BackboneEditor);
}(typeof globalThis!=='undefined'?globalThis:this,function(editor){
    'use strict';
    const escape=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
    const math=new Set(['add','multiply','scale','gate','concat','split','softmax','matmul']);
    const symbol={add:'+',multiply:'⊙',scale:'γ',gate:'G',concat:'©',split:'⑂',softmax:'σ',matmul:'⊗'};
    function wrap(value,limit){
        const out=[];
        for(const paragraph of String(value || '').split('\n')){
            let line='';for(const word of paragraph.split(/\s+/)){
                if(line&&line.length+word.length+1>limit){out.push(line);line=word;}else line+=(line?' ':'')+word;
            }out.push(line);
        }return out;
    }
    function bounds(n){const l=n.layout;return {left:l.x-l.width/2,right:l.x+l.width/2,top:l.y-l.height/2,bottom:l.y+l.height/2};}
    function port(n,side){const l=n.layout,b=bounds(n);return {x:side==='left'?b.left:side==='right'?b.right:l.x,y:side==='top'?b.top:side==='bottom'?b.bottom:l.y,side};}
    const step=(p,d)=>({x:p.x+(p.side==='right'?d:p.side==='left'?-d:0),y:p.y+(p.side==='bottom'?d:p.side==='top'?-d:0)});
    function hit(a,b,n){const box=bounds(n),pad=6;
        if(a.x===b.x)return a.x>box.left-pad&&a.x<box.right+pad&&Math.max(a.y,b.y)>box.top-pad&&Math.min(a.y,b.y)<box.bottom+pad;
        if(a.y===b.y)return a.y>box.top-pad&&a.y<box.bottom+pad&&Math.max(a.x,b.x)>box.left-pad&&Math.min(a.x,b.x)<box.right+pad;
        return false;
    }
    function route(edge,nodes){
        const source=nodes.get(edge.source),target=nodes.get(edge.target),a=source.layout,b=target.layout;
        const horizontal=Math.abs(b.x-a.x)>Math.abs(b.y-a.y);
        const sp=edge.source_port&&edge.source_port!=='auto'?edge.source_port:horizontal?(b.x>a.x?'right':'left'):(b.y>a.y?'bottom':'top');
        const tp=edge.target_port&&edge.target_port!=='auto'?edge.target_port:horizontal?(b.x>a.x?'left':'right'):(b.y>a.y?'top':'bottom');
        const start=port(source,sp),end=port(target,tp);
        const facing=({left:'right',right:'left',top:'bottom',bottom:'top'})[sp]===tp;
        const gap=sp==='right'?end.x-start.x:sp==='left'?start.x-end.x:sp==='bottom'?end.y-start.y:start.y-end.y;
        const lead=facing&&gap>0?Math.min(16,gap/3):16,sa=step(start,lead),ta=step(end,lead);
        let candidates;
        if(edge.waypoints&&edge.waypoints.length){
            const result=[start,sa];
            for(const p of [...edge.waypoints,ta]){const last=result[result.length-1];result.push({x:p.x,y:last.y},p);}
            return [...result,end];
        }
        const midX=(sa.x+ta.x)/2,midY=(sa.y+ta.y)/2;
        candidates=[
            [start,sa,{x:midX,y:sa.y},{x:midX,y:ta.y},ta,end],
            [start,sa,{x:sa.x,y:midY},{x:ta.x,y:midY},ta,end],
            [start,sa,{x:ta.x,y:sa.y},ta,end],
            [start,sa,{x:sa.x,y:ta.y},ta,end]
        ];
        // Route residuals around the operation chain, rather than through blocks.
        if(edge.kind==='skip'){
            const bottom=Math.max(...[...nodes.values()].filter(n=>n.id!==edge.source&&n.id!==edge.target&&
                n.layout.x>=Math.min(a.x,b.x)&&n.layout.x<=Math.max(a.x,b.x)).map(n=>bounds(n).bottom),bounds(source).bottom,bounds(target).bottom)+38;
            const left=Math.min(...[...nodes.values()].map(n=>bounds(n).left))-32;
            const ps=port(source,horizontal?'bottom':'left'),pt=port(target,horizontal?'bottom':'left');
            candidates.unshift(horizontal?[ps,{x:ps.x,y:bottom},{x:pt.x,y:bottom},pt]:[ps,{x:left,y:ps.y},{x:left,y:pt.y},pt]);
        }
        const obstacles=[...nodes.values()].filter(n=>n.id!==edge.source&&n.id!==edge.target);
        function cost(points){let length=0,crossings=0;
            for(let i=1;i<points.length;i++){length+=Math.abs(points[i].x-points[i-1].x)+Math.abs(points[i].y-points[i-1].y);
                crossings+=obstacles.filter(n=>hit(points[i-1],points[i],n)).length;}
            return length+crossings*10000;
        }
        candidates.sort((p,q)=>cost(p)-cost(q));return candidates[0];
    }
    function render(input,options={}){
        editor.validateGraph(input);
        const graph=input.nodes.every(n=>n.layout)?input:editor.layoutGraph(input);
        const nodes=new Map(graph.nodes.map(n=>[n.id,n]));
        const routes=graph.edges.map(e=>({edge:e,points:route(e,nodes)}));
        const groupBounds=(graph.groups || []).filter(g=>g.node_ids.length).map(g=>{
            const boxes=g.node_ids.map(id=>bounds(nodes.get(id))),p=g.padding || 26;
            return {...g,x:Math.min(...boxes.map(b=>b.left))-p,y:Math.min(...boxes.map(b=>b.top))-p-16,
                right:Math.max(...boxes.map(b=>b.right))+p,bottom:Math.max(...boxes.map(b=>b.bottom))+p};
        });
        const minX=Math.min(0,...graph.nodes.map(n=>{
            const font=n.layout.font_size || 17;
            const extra=(math.has(n.kind)||(editor.SHAPE_TYPES&&editor.SHAPE_TYPES[n.kind]==='circle'))?Math.max(String(n.caption??n.label).length*font*.58,n.layout.show_shape?n.shape.join(' × ').length*(font-1)*.58:0)+12:0;
            return bounds(n).left-36-extra;
        }),...groupBounds.map(g=>g.x),...routes.flatMap(r=>r.points.map(p=>p.x-10)));
        const minY=Math.min(0,...graph.nodes.map(n=>bounds(n).top-28),...groupBounds.map(g=>g.y),...routes.flatMap(r=>r.points.map(p=>p.y-10)));
        const ox=-minX,oy=-minY;
        const width=Math.max(480,Math.max(...graph.nodes.map(n=>bounds(n).right),...groupBounds.map(g=>g.right),...routes.flatMap(r=>r.points.map(p=>p.x)))+36+ox);
        const height=Math.max(220,Math.max(...graph.nodes.map(n=>bounds(n).bottom),...groupBounds.map(g=>g.bottom),...routes.flatMap(r=>r.points.map(p=>p.y)))+40+oy);
        const mono=!!options.monochrome,frames=[],wires=[],blocks=[];
        const text=(x,y,value,anchor='middle',size=17,weight='normal')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}">${escape(value)}</text>`;
        for(const g of groupBounds){frames.push(`<g data-group="${g.id}"><rect x="${g.x}" y="${g.y}" width="${g.right-g.x}" height="${g.bottom-g.y}" rx="12" fill="${mono?'#fafafa':g.fill || '#fafafa'}" stroke="${g.stroke || '#303030'}" stroke-width="1.4" stroke-dasharray="7 5"/>${text(g.label_position==='center'?(g.x+g.right)/2:g.x+16,g.y+24,g.label,g.label_position==='center'?'middle':'start',18,'bold')}</g>`);}
        for(const {edge:e,points} of routes){
            const color=e.color || '#303030',data=`data-source="${e.source}" data-target="${e.target}" data-edge="${e.id || ''}"`;
            const path=points.map((p,i)=>`${i?'L':'M'} ${p.x} ${p.y}`).join(' ');
            wires.push(`<g class="edge-node" ${data}><path class="edge-hit" d="${path}" stroke="transparent" stroke-width="14" fill="none"/><path class="edge-line" d="${path}" stroke="${color}" stroke-width="1.3" fill="none"${e.kind==='skip'?' stroke-dasharray="5 4"':''} marker-end="url(#reference-arrow)"/>`);
            if(e.label&&!['lateral','identity'].includes(e.label)){
                const mid=points[Math.floor(points.length/2)];wires.push(text(mid.x+8,mid.y-7,e.label,'start',14));
            }wires.push('</g>');
        }
        for(const n of graph.nodes){
            const l=n.layout,b=bounds(n),shapeType=(editor.SHAPE_TYPES&&editor.SHAPE_TYPES[n.kind])||(math.has(n.kind)?'circle':'standard'),isMath=shapeType==='circle'||math.has(n.kind),size=l.font_size || 17;
            const fill=mono?(n.kind==='block'?'#eeeeee':'#ffffff'):(l.fill || editor.COLORS[n.kind] || '#ffffff');
            const label=n.caption!==undefined?n.caption:n.label;
            let content=`<title>${escape(n.label)}; ${n.shape.join(' × ')}</title>`;

            // Render offset stacked shadows for repeated blocks
            if(n.repeat && n.repeat > 1){
                content+=`<rect class="stacked-shadow" x="${b.left+5}" y="${b.top-5}" width="${l.width}" height="${l.height}" rx="6" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.2" opacity="0.6"/>`;
                content+=`<rect class="stacked-shadow" x="${b.left+2.5}" y="${b.top-2.5}" width="${l.width}" height="${l.height}" rx="6" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.2" opacity="0.8"/>`;
            }

            if(isMath){
                content+=`<ellipse cx="${l.x}" cy="${l.y}" rx="${l.width/2}" ry="${l.height/2}" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.4"/>${text(l.x,l.y+7,n.symbol || symbol[n.kind] || '+','middle',size+5)}`;
                if(label&&!['Add','Multiply','LayerScale γ','Residual add','Matmul','Concat','Split'].includes(label))content+=text(b.left-10,l.y-25,label,'end',size);
                if(l.show_shape&&options.shapes!==false)content+=text(b.left-10,l.y-6,n.shape.join(' × '), 'end',size-1);
            }else if(shapeType==='trapezoid_down'){
                const p1=`${b.left},${b.top}`, p2=`${b.right},${l.y-l.height*0.275}`, p3=`${b.right},${l.y+l.height*0.275}`, p4=`${b.left},${b.bottom}`;
                content+=`<polygon class="node-trapezoid" points="${p1} ${p2} ${p3} ${p4}" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.3"/>`;
            }else if(shapeType==='trapezoid_up'){
                const p1=`${b.left},${l.y-l.height*0.275}`, p2=`${b.right},${b.top}`, p3=`${b.right},${b.bottom}`, p4=`${b.left},${l.y+l.height*0.275}`;
                content+=`<polygon class="node-trapezoid" points="${p1} ${p2} ${p3} ${p4}" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.3"/>`;
            }else if(shapeType==='slim'){
                content+=`<rect class="node-slim" x="${b.left}" y="${b.top}" width="${l.width}" height="${l.height}" rx="12" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.3"/>`;
            }else{
                content+=`<rect x="${b.left}" y="${b.top}" width="${l.width}" height="${l.height}" rx="6" fill="${fill}" stroke="${l.stroke || '#303030'}" stroke-width="1.3"/>`;
            }

            if(!isMath){
                const textSize = shapeType==='slim' && l.width <= 50 ? Math.min(size, 11) : size;
                const lineWidth=l.rotation?l.height:l.width,limit=Math.max(5,Math.floor((lineWidth-16)/(textSize*0.52)));
                const labels=wrap(label,limit),subs=n.subtitle?wrap(n.subtitle,limit):[];
                const upstream=nodes.get(graph.edges.find(e=>e.target===n.id)?.source);
                const channels=n.channel_projection?[upstream?`${upstream.shape[0]} → ${n.shape[0]}`:`${n.shape[0]} channels`]:[];
                const repeat=n.repeat?[`${n.repeat_label || 'Repeat'} × ${n.repeat}`]:[];
                const lines=[...labels,...subs,...channels,...repeat],showShape=l.show_shape&&options.shapes!==false;
                const usable=l.rotation?l.width:l.height,step=textSize+3,maxLines=Math.max(1,Math.floor((usable-14)/step));
                const displayed=lines.slice(0,Math.max(1,maxLines-(showShape?1:0)));
                if(lines.length>displayed.length)displayed[displayed.length-1]+='…';
                const all=[...displayed,...(showShape?[n.shape.join(' × ')]:[])];
                content+=`<g${l.rotation?` transform="rotate(${l.rotation} ${l.x} ${l.y})"`:''}>${all.map((line,i)=>text(l.x,l.y-(all.length-1)*step/2+i*step+textSize*0.34,line,'middle',textSize)).join('')}</g>`;
            }
            content+=`<rect class="selection-outline" x="${b.left-3}" y="${b.top-3}" width="${l.width+6}" height="${l.height+6}" rx="6" fill="none" stroke="#2473aa" stroke-width="2" opacity="0"/><rect class="resize-handle" data-resize="${n.id}" x="${b.right-5}" y="${b.bottom-5}" width="10" height="10" fill="#2473aa" stroke="#ffffff"/>`;
            blocks.push(`<g class="tensor-node editable-node" data-node="${n.id}" tabindex="0" role="button" aria-label="${escape(label)}">${content}</g>`);
        }
        const showLegend = !!(options.showLegend || graph.show_legend);
        const kinds = [...new Set(graph.nodes.map(n=>n.kind).filter(Boolean))];
        const legendItems = showLegend ? kinds.map(k => ({
            kind: k,
            label: editor.LIBRARY[k] || k,
            fill: mono ? (k==='block'?'#eeeeee':'#ffffff') : (editor.COLORS[k] || '#ffffff'),
            shapeType: (editor.SHAPE_TYPES && editor.SHAPE_TYPES[k]) || 'standard',
            isMath: math.has(k) || ((editor.SHAPE_TYPES && editor.SHAPE_TYPES[k]) === 'circle'),
            symbol: symbol[k] || ''
        })) : [];
        const legW = 170, legH = legendItems.length ? 28 + legendItems.length * 20 : 0;
        let legendSvg = '';
        if (showLegend && legendItems.length) {
            const lx = width - legW - 16, ly = 20;
            const itemsSvg = legendItems.map((item, i) => {
                const iy = ly + 34 + i * 20;
                const swatch = item.isMath
                    ? `<circle cx="${lx + 20}" cy="${iy - 4}" r="7" fill="${item.fill}" stroke="#303030" stroke-width="1"/><text x="${lx + 20}" y="${iy - 1}" text-anchor="middle" font-size="10">${escape(item.symbol)}</text>`
                    : item.shapeType === 'slim'
                    ? `<rect x="${lx + 15}" y="${iy - 10}" width="10" height="13" rx="4" fill="${item.fill}" stroke="#303030" stroke-width="1"/>`
                    : item.shapeType === 'trapezoid_down'
                    ? `<polygon points="${lx + 11},${iy - 10} ${lx + 27},${iy - 7} ${lx + 27},${iy - 1} ${lx + 11},${iy + 2}" fill="${item.fill}" stroke="#303030" stroke-width="1"/>`
                    : item.shapeType === 'trapezoid_up'
                    ? `<polygon points="${lx + 11},${iy - 7} ${lx + 27},${iy - 10} ${lx + 27},${iy + 2} ${lx + 11},${iy - 1}" fill="${item.fill}" stroke="#303030" stroke-width="1"/>`
                    : `<rect x="${lx + 12}" y="${iy - 10}" width="16" height="12" rx="3" fill="${item.fill}" stroke="#303030" stroke-width="1"/>`;
                return `${swatch}<text x="${lx + 36}" y="${iy}" font-size="12" fill="#334155">${escape(item.label)}</text>`;
            }).join('');
            legendSvg = `<g class="diagram-legend"><rect x="${lx}" y="${ly}" width="${legW}" height="${legH}" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/><text x="${lx + 12}" y="${ly + 18}" font-size="12" font-weight="bold" fill="#1e293b">Legend</text>${itemsSvg}</g>`;
        }
        const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="182mm" height="${(182*height/width).toFixed(3)}mm" viewBox="0 0 ${width} ${height}" font-family="Arial, Helvetica, sans-serif" fill="#202020" role="img" aria-labelledby="reference-title"><title id="reference-title">${escape(graph.title)}</title><desc>${escape(graph.description || '')}</desc><defs><marker id="reference-arrow" viewBox="0 0 6 6" refX="5.7" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 6 3 L 0 6" fill="#303030"/></marker></defs><style>.resize-handle{display:none}.edge-hit{pointer-events:stroke}</style><rect width="${width}" height="${height}" fill="#ffffff"/><g transform="translate(${ox} ${oy})">${frames.join('')}${wires.join('')}${blocks.join('')}</g>${legendSvg}</svg>`;
        return {svg,width,height,offset:{x:ox,y:oy},routes};
    }
    return {render,route,bounds};
}));

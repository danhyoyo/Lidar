/* Editable NN-SVG documents and transactional graph commands. MIT. */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory(require('./Backbone.js'));
    else root.BackboneEditor = factory(root.BackboneSVG);
}(typeof globalThis !== 'undefined' ? globalThis : this, function (core) {
    'use strict';
    const clone = value => JSON.parse(JSON.stringify(value));
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

    const COLORS = Object.freeze({
        input: '#ffffff', output: '#ffffff', feature: '#ffffff', embedding: '#ede9fe',
        conv: '#a8cf91', conv1d: '#a8cf91', depthwise: '#dce8f5', pointwise: '#dce8f5', deconv: '#bae6fd', linear: '#dbeafe',
        norm: '#f5f5f5', layernorm: '#f1f5f9', rmsnorm: '#ccfbf1', groupnorm: '#ccfbf1',
        activation: '#f8ceb4', relu: '#ffe4e6', gelu: '#ffe4e6', silu: '#f8ceb4', sigmoid: '#ffe4e6', tanh: '#ffe4e6',
        pool: '#ece5f5', maxpool: '#f1f5f9', avgpool: '#f1f5f9', upsample: '#ffe5a1', pixelshuffle: '#e0f2fe',
        add: '#ffffff', multiply: '#ffffff', concat: '#d9eafd', split: '#e2e8f0', matmul: '#ede9fe', scale: '#ffffff', gate: '#ffffff', softmax: '#ffe4e6',
        block: '#dce8f5', attention: '#dce8f5', mhsa: '#ede9fe', qkv: '#fae8ff', ffn: '#fef3c7', rope: '#f3e8ff', moe: '#ffedd5', reshape: '#f8fafc'
    });

    const LIBRARY = Object.freeze({
        input: 'Input tensor', output: 'Output tensor', feature: 'Feature tensor', embedding: 'Embedding',
        conv: 'Conv 3×3', conv1d: 'Conv 1D', depthwise: 'DW 3×3', pointwise: 'Pointwise (1×1)', deconv: 'Transposed Conv', linear: 'Linear / Dense',
        norm: 'BatchNorm', layernorm: 'LayerNorm', rmsnorm: 'RMSNorm', groupnorm: 'GroupNorm',
        activation: 'Activation', relu: 'ReLU', gelu: 'GELU', silu: 'SiLU / Swish', sigmoid: 'Sigmoid', tanh: 'Tanh',
        pool: 'Pooling', maxpool: 'MaxPool 2×2', avgpool: 'AvgPool 2×2', upsample: 'Upsample ↑2', pixelshuffle: 'PixelShuffle ↑2',
        add: 'Add (+)', multiply: 'Multiply (⊙)', concat: 'Concat (©)', split: 'Split (⑂)', matmul: 'Matmul (⊗)', scale: 'LayerScale (γ)', gate: 'Gate (G)', softmax: 'Softmax (σ)',
        block: 'Custom block', attention: 'Attention', mhsa: 'MHSA', qkv: 'QKV Proj', ffn: 'FFN / MLP', rope: 'RoPE', moe: 'MoE Router', reshape: 'Reshape / Flatten'
    });

    const PORTS = ['auto', 'top', 'right', 'bottom', 'left'];
    const operators = new Set(['add', 'multiply', 'scale', 'gate', 'concat', 'split', 'softmax', 'matmul']);
    const idOK = value => typeof value === 'string' && /^[a-zA-Z0-9_-]+$/.test(value);
    const colorOK = value => typeof value === 'string' && /^#[0-9a-fA-F]{6}$/.test(value);
    const bounded = (value,min,max) => Number.isFinite(value) && value>=min && value<=max;
    const textOK = value => typeof value==='string' && value.length<=1000;

    function validateGraph(graph) {
        core.validateGraph(graph);
        const ids=new Set(graph.nodes.map(n=>n.id)), groups=new Set(), edgeIds=new Set();
        if(graph.groups!==undefined&&(!Array.isArray(graph.groups)||graph.groups.length>200))throw new Error('Invalid groups array.');
        for(const g of graph.groups || []) {
            if(!idOK(g.id)||groups.has(g.id)||!textOK(g.label)||!Array.isArray(g.node_ids)||
                g.node_ids.some(id=>!ids.has(id))||(g.fill!==undefined&&!colorOK(g.fill))||
                (g.stroke!==undefined&&!colorOK(g.stroke))||
                (g.padding!==undefined&&!bounded(g.padding,8,100))) throw new Error('Invalid group.');
            groups.add(g.id);
        }
        for(const n of graph.nodes) {
            if(n.kind!==undefined&&!Object.hasOwn(COLORS,n.kind))throw new Error('Unknown block type.');
            if(n.caption!==undefined&&!textOK(n.caption)) throw new Error('Invalid block caption.');
            if(n.subtitle!==undefined&&!textOK(n.subtitle)) throw new Error('Invalid block subtitle.');
            for(const key of ['kernel','stride','repeat','groups'])if(n[key]!==undefined&&(!Number.isSafeInteger(n[key])||n[key]<1||n[key]>1000000))throw new Error(`Invalid ${key}.`);
            if(n.repeat_label!==undefined&&!textOK(n.repeat_label))throw new Error('Invalid repetition label.');
            if(n.channel_projection!==undefined&&typeof n.channel_projection!=='boolean')throw new Error('Invalid channel label option.');
            if(!n.layout) continue;
            const l=n.layout;
            if(!bounded(l.x,0,20000)||!bounded(l.y,0,20000)||!bounded(l.width,24,1200)||!bounded(l.height,24,1200)||
                (l.fill!==undefined&&!colorOK(l.fill))||(l.stroke!==undefined&&!colorOK(l.stroke))||
                (l.font_size!==undefined&&!bounded(l.font_size,8,40))||
                (l.rotation!==undefined&&![0,90,-90].includes(l.rotation))||
                (l.show_shape!==undefined&&typeof l.show_shape!=='boolean')) throw new Error(`Invalid layout for ${n.id}.`);
        }
        for(const e of graph.edges) {
            if(e.id!==undefined&&(!idOK(e.id)||edgeIds.has(e.id))) throw new Error('Invalid edge id.');
            if(e.id) edgeIds.add(e.id);
            if(!PORTS.includes(e.source_port || 'auto')||!PORTS.includes(e.target_port || 'auto')||
                (e.label!==undefined&&!textOK(e.label))||(e.color!==undefined&&!colorOK(e.color))||
                (e.waypoints!==undefined&&(!Array.isArray(e.waypoints)||e.waypoints.length>16||
                    e.waypoints.some(p=>!bounded(p.x,0,20000)||!bounded(p.y,0,20000))))) throw new Error('Invalid edge style or route.');
        }
        return graph;
    }
    function validate(doc) {
        core.validateBundle(doc);
        for(const graph of Object.values(doc.views)) validateGraph(graph);
        return doc;
    }
    function nextId(items,prefix) {
        const used=new Set(items.map(x=>x.id));let index=1;
        while(used.has(`${prefix}${index}`)) index++;
        return `${prefix}${index}`;
    }
    function layoutGraph(graph,overview=false) {
        const result=clone(graph), rowMin=Math.min(...result.nodes.map(n=>n.row));
        const detail=result.nodes.some(n=>n.row>3), outX=detail?1070:853;
        for(const n of result.nodes) {
            const math=operators.has(n.kind);
            let x=120+n.col*204, y=104+(n.row-rowMin)*144;
            let width=math?44:176,height=math?44:88;
            let caption=n.label, subtitle='';
            if(overview) {
                if(n.row===1) {
                    x=n.col===0?83:320;y=n.col===0||n.col===1?64:228+(n.col-2)*93;
                    width=n.col===0?146:n.col===1?214:212;height=n.col===1?110:n.col===0?94:112;
                    if(n.col===1) {caption='Stem';subtitle='Conv 3×3, s=2 → s=1\neach BN + SiLU';}
                    else if(n.repeat) {caption=n.label.replace(' · stride ',' /');subtitle='DS 3×3 + BN + SiLU';n.repeat_label='NeXt';}
                    else if(n.col>1) height=58;
                } else if(n.kernel===1) {
                    x=588;y=228+(n.col-2)*93;width=148;height=86;
                    caption='Conv 1×1';n.channel_projection=true;
                } else {
                    x=853;y=228+(n.col-2)*93;
                    if(n.col<=1&&n.row>=3) {x=outX;y=n.col===0?16:86;width=n.col===0?210:178;height=n.col===0?44:80;
                        caption=n.col===0?'Output /4':'Conv 3×3';subtitle=n.col===0?'':'BN + SiLU';}
                    else if(n.row>3) {x=outX;y=228;}
                    else if(n.kind==='upsample') {width=194;height=98;caption='Bilinear ↑2';
                        subtitle=(n.operations || []).some(op=>op.startsWith('DW'))?'DW 3×3\nBN + SiLU':'Conv 3×3\nBN + SiLU';}
                    else if(n.row===2&&n.col===1) {x=83;y=414;width=150;height=90;caption='Detail branch';subtitle='DW s2 → PW';}
                }
            }
            n.layout={x,y,width,height,fill:overview&&n.row===1&&n.col===1?'#f8ceb4':COLORS[n.kind || 'conv'],stroke:'#303030',font_size:18,rotation:0,show_shape:overview||!math};
            if(overview) {n.caption=caption;n.subtitle=subtitle;}
        }
        if(overview) {
            result.groups=[{id:'backbone',label:'Backbone + SG-FPN',fill:'#fafafa',stroke:'#303030',padding:26,label_position:'center',
                node_ids:result.nodes.filter(n=>!(n.row===1&&n.col<2)&&!(n.row>=3&&n.col<2)).map(n=>n.id)}];
        } else result.groups=[];
        result.edges.forEach(e=>{e.id=e.id || nextId(result.edges,'edge');e.source_port=e.source_port || 'auto';e.target_port=e.target_port || 'auto';});
        return result;
    }
    function create(bundle) {
        // Validate saved geometry before cloning: JSON cloning must not hide NaN/Infinity.
        validate(bundle);
        const doc=clone(bundle);
        for(const [id,g] of Object.entries(doc.views)) {
            if(!g.nodes.every(n=>n.layout)) {
                const initial=layoutGraph(g,id==='overview'&&doc.name==='MobilePixorNeXt');
                g.nodes=g.nodes.map((node,index)=>node.layout?node:{...initial.nodes[index],...node,layout:initial.nodes[index].layout});
                g.groups=g.groups || initial.groups;
            }
            g.groups=g.groups || [];g.edges.forEach(e=>{e.id=e.id || nextId(g.edges,'edge');});
        }
        doc.editor={version:1,modified:false,...doc.editor};
        return validate(doc);
    }
    function graphAt(doc,view) {
        if(!doc.views[view]) throw new Error('Unknown diagram view.');
        return doc.views[view];
    }
    function transaction(doc,mutate,semantic=true) {
        const draft=clone(doc), result=mutate(draft);
        draft.editor={...draft.editor,version:1,modified:true};
        if(semantic) draft.shape_mode='manual drawing; tensor shapes entered by user';
        validate(draft);
        Object.assign(doc,draft);
        return result;
    }
    function updateNode(doc,view,id,patch) {
        return transaction(doc,d=>{
            const g=graphAt(d,view), node=g.nodes.find(n=>n.id===id);
            if(!node) throw new Error('Unknown block.');
            const {group,...fields}=patch;
            Object.assign(node,{...fields,layout:{...node.layout,...fields.layout}});
            for(const key of ['kernel','stride','repeat','groups','symbol'])if(fields[key]===null)delete node[key];
            if(fields.label!==undefined&&fields.caption===undefined) node.caption=fields.label;
            if(group!==undefined) {
                if(group&&!g.groups.some(x=>x.id===group)) throw new Error('Unknown group.');
                g.groups.forEach(x=>{x.node_ids=x.node_ids.filter(v=>v!==id);if(x.id===group)x.node_ids.push(id);});
            }
        },Object.keys(patch).some(key=>!['layout','group'].includes(key)));
    }
    function freePosition(nodes,size,preferred={x:180,y:130}) {
        if(!Number.isFinite(preferred.x)||!Number.isFinite(preferred.y)) throw new Error('Invalid insertion position.');
        const fits=p=>nodes.every(n=>Math.abs(n.layout.x-p.x)>=(n.layout.width+size.width)/2+16 || Math.abs(n.layout.y-p.y)>=(n.layout.height+size.height)/2+16);
        const clamp=p=>({x:Math.max(size.width/2+8,Math.min(19000,p.x)),y:Math.max(size.height/2+8,Math.min(19000,p.y))});
        // Search nearby free cells, then expand outwards. The source graph is untouched.
        for(let ring=0;ring<80;ring++) {
            for(let y=-ring;y<=ring;y++) for(let x=-ring;x<=ring;x++) {
                if(Math.max(Math.abs(x),Math.abs(y))!==ring)continue;
                const point=clamp({x:preferred.x+x*(size.width+32),y:preferred.y+y*(size.height+32)});
                if(fits(point))return point;
            }
        }
        throw new Error('No free space near this position. Move some blocks before adding another.');
    }
    function defaultSize(kind) {
        const shape = SHAPE_TYPES[kind] || 'standard';
        if (shape === 'circle') return { width: 44, height: 44 };
        if (shape === 'slim') return { width: 44, height: 76 };
        if (shape === 'trapezoid_down' || shape === 'trapezoid_up') return { width: 96, height: 64 };
        return { width: 160, height: 72 };
    }
    function addNode(doc,view,kind='conv',preferred) {
        if(!Object.hasOwn(LIBRARY,kind)) throw new Error('Unknown block type.');
        return transaction(doc,d=>{
            const g=graphAt(d,view),id=nextId(g.nodes,'block'),index=g.nodes.length;
            const size=defaultSize(kind);
            const point=freePosition(g.nodes,size,preferred);
            g.nodes.push({id,label:LIBRARY[kind],kind,shape:[48,200,176],col:index,row:1,
                layout:{...point,...size,fill:COLORS[kind],stroke:'#303030',font_size:17,rotation:0,show_shape:true}});
            return id;
        });
    }
    function duplicateNode(doc,view,id) {
        return transaction(doc,d=>{
            const g=graphAt(d,view),source=g.nodes.find(n=>n.id===id);
            if(!source) throw new Error('Unknown block.');
            const copy=clone(source);copy.id=nextId(g.nodes,'block');
            Object.assign(copy.layout,freePosition(g.nodes,copy.layout,{x:source.layout.x+source.layout.width+32,y:source.layout.y}));
            g.nodes.push(copy);for(const group of g.groups)if(group.node_ids.includes(id))group.node_ids.push(copy.id);
            return copy.id;
        });
    }
    function removeNode(doc,view,id) {
        return transaction(doc,d=>{const g=graphAt(d,view);g.nodes=g.nodes.filter(n=>n.id!==id);
            g.edges=g.edges.filter(e=>e.source!==id&&e.target!==id);g.groups.forEach(x=>{x.node_ids=x.node_ids.filter(v=>v!==id);});});
    }
    function nodesInRect(graph,rect) {
        if(!rect||!['x1','y1','x2','y2'].every(k=>Number.isFinite(rect[k])))throw new Error('Invalid selection rectangle.');
        const left=Math.min(rect.x1,rect.x2),right=Math.max(rect.x1,rect.x2);
        const top=Math.min(rect.y1,rect.y2),bottom=Math.max(rect.y1,rect.y2);
        if(left===right||top===bottom)return [];
        return graph.nodes.filter(n=>{
            const l=n.layout;
            return l&&l.x+l.width/2>left&&l.x-l.width/2<right&&l.y+l.height/2>top&&l.y-l.height/2<bottom;
        }).map(n=>n.id);
    }
    function selectionNodes(graph,ids) {
        if(!Array.isArray(ids)||!ids.length)throw new Error('Choose at least one block.');
        const byId=new Map(graph.nodes.map(n=>[n.id,n]));
        return [...new Set(ids)].map(id=>{const n=byId.get(id);if(!n)throw new Error('Unknown block in selection.');return n;});
    }
    function translatedLayouts(graph,ids,dx,dy,snap=false) {
        if(!Number.isFinite(dx)||!Number.isFinite(dy))throw new Error('Invalid movement.');
        const nodes=selectionNodes(graph,ids);
        if(nodes.some(n=>!n.layout))throw new Error('Position blocks before moving them.');
        if(snap){const anchor=nodes[0].layout;dx=Math.round((anchor.x+dx)/10)*10-anchor.x;dy=Math.round((anchor.y+dy)/10)*10-anchor.y;}
        // Clamp one shared delta so spacing survives at the document boundary.
        dx=Math.max(Math.max(...nodes.map(n=>-n.layout.x)),Math.min(Math.min(...nodes.map(n=>20000-n.layout.x)),dx));
        dy=Math.max(Math.max(...nodes.map(n=>-n.layout.y)),Math.min(Math.min(...nodes.map(n=>20000-n.layout.y)),dy));
        return nodes.map(n=>({id:n.id,layout:{...n.layout,x:n.layout.x+dx,y:n.layout.y+dy}}));
    }
    function moveNodes(doc,view,ids,dx,dy,snap=false) {
        return transaction(doc,d=>{
            const graph=graphAt(d,view),updates=new Map(translatedLayouts(graph,ids,dx,dy,snap).map(n=>[n.id,n.layout]));
            graph.nodes.forEach(n=>{if(updates.has(n.id))n.layout=updates.get(n.id);});
        },false);
    }
    function alignNodes(graph, nodeIds, alignment) {
        if (!Array.isArray(nodeIds) || nodeIds.length < 2) throw new Error('Choose at least two blocks to align.');
        const targets = selectionNodes(graph, nodeIds);
        if (targets.some(n => !n.layout)) throw new Error('Position blocks before aligning them.');
        const boxes = targets.map(n => ({
            id: n.id,
            left: n.layout.x - n.layout.width / 2,
            right: n.layout.x + n.layout.width / 2,
            top: n.layout.y - n.layout.height / 2,
            bottom: n.layout.y + n.layout.height / 2,
            cx: n.layout.x,
            cy: n.layout.y,
            w: n.layout.width,
            h: n.layout.height
        }));
        const minLeft = Math.min(...boxes.map(b => b.left));
        const maxRight = Math.max(...boxes.map(b => b.right));
        const minTop = Math.min(...boxes.map(b => b.top));
        const maxBottom = Math.max(...boxes.map(b => b.bottom));
        const avgCx = (minLeft + maxRight) / 2;
        const avgCy = (minTop + maxBottom) / 2;

        return targets.map(n => {
            const updated = clone(n);
            if (alignment === 'left') updated.layout.x = minLeft + n.layout.width / 2;
            else if (alignment === 'right') updated.layout.x = maxRight - n.layout.width / 2;
            else if (alignment === 'center-h') updated.layout.x = avgCx;
            else if (alignment === 'top') updated.layout.y = minTop + n.layout.height / 2;
            else if (alignment === 'bottom') updated.layout.y = maxBottom - n.layout.height / 2;
            else if (alignment === 'center-v') updated.layout.y = avgCy;
            else throw new Error(`Unknown alignment: ${alignment}`);
            return updated;
        });
    }
    function distributeNodes(graph, nodeIds, axis) {
        if (!Array.isArray(nodeIds) || nodeIds.length < 3) throw new Error('Choose at least three blocks to distribute.');
        const targets = selectionNodes(graph, nodeIds);
        if (targets.some(n => !n.layout)) throw new Error('Position blocks before distributing them.');
        const sorted = [...targets].sort((a, b) => axis === 'horizontal' ? a.layout.x - b.layout.x : a.layout.y - b.layout.y);
        const startCoord = axis === 'horizontal' ? sorted[0].layout.x : sorted[0].layout.y;
        const endCoord = axis === 'horizontal' ? sorted.at(-1).layout.x : sorted.at(-1).layout.y;
        const step = (endCoord - startCoord) / (sorted.length - 1);

        return sorted.map((n, i) => {
            const updated = clone(n);
            if (axis === 'horizontal') updated.layout.x = Math.round(startCoord + i * step);
            else if (axis === 'vertical') updated.layout.y = Math.round(startCoord + i * step);
            else throw new Error(`Unknown axis: ${axis}`);
            return updated;
        });
    }
    function removeNodes(doc,view,ids) {
        return transaction(doc,d=>{
            const g=graphAt(d,view),removed=new Set(selectionNodes(g,ids).map(n=>n.id));
            g.nodes=g.nodes.filter(n=>!removed.has(n.id));
            g.edges=g.edges.filter(e=>!removed.has(e.source)&&!removed.has(e.target));
            for(const group of g.groups || [])group.node_ids=group.node_ids.filter(id=>!removed.has(id));
        });
    }
    function addEdge(doc,view,source,target,patch={}) {
        return transaction(doc,d=>{const g=graphAt(d,view),id=nextId(g.edges,'edge');
            if(source===target) throw new Error('Choose two different blocks.');
            if(g.edges.some(e=>e.source===source&&e.target===target)) throw new Error('These blocks are already connected.');
            g.edges.push({id,source,target,label:'',kind:'flow',source_port:'auto',target_port:'auto',...patch});return id;});
    }
    function updateEdge(doc,view,id,patch) {
        return transaction(doc,d=>{const e=graphAt(d,view).edges.find(x=>x.id===id);if(!e)throw new Error('Unknown connection.');Object.assign(e,patch);if(e.source===e.target)throw new Error('Choose two different blocks.');});
    }
    function removeEdge(doc,view,id) {
        return transaction(doc,d=>{const g=graphAt(d,view);g.edges=g.edges.filter(e=>e.id!==id);});
    }
    function addGroup(doc,view,label) {
        return transaction(doc,d=>{const g=graphAt(d,view),id=nextId(g.groups,'group');
            g.groups.push({id,label,node_ids:[],fill:'#fafafa',stroke:'#303030',padding:26});return id;},false);
    }
    function updateGroup(doc,view,id,patch) {
        return transaction(doc,d=>{const g=graphAt(d,view).groups.find(x=>x.id===id);if(!g)throw new Error('Unknown group.');Object.assign(g,patch);},false);
    }
    function removeGroup(doc,view,id) {
        return transaction(doc,d=>{const g=graphAt(d,view);g.groups=g.groups.filter(x=>x.id!==id);},false);
    }
    class History {
        constructor(value) {this.value=create(value);this.past=[];this.future=[];}
        get canUndo() {return this.past.length>0;}
        get canRedo() {return this.future.length>0;}
        commit(command) {
            const draft=clone(this.value),result=command(draft);validate(draft);
            if(JSON.stringify(draft)!==JSON.stringify(this.value)) {this.past.push(this.value);if(this.past.length>80)this.past.shift();this.value=draft;this.future=[];}
            return result;
        }
        undo() {if(this.canUndo){this.future.push(this.value);this.value=this.past.pop();}}
        redo() {if(this.canRedo){this.past.push(this.value);this.value=this.future.pop();}}
    }
    function blank() {
        const doc={schema_version:1,name:'Custom architecture',source:'Custom drawing',shape_mode:'manual drawing',
            views:{overview:{title:'My architecture',nodes:[{id:'input',label:'Input tensor',kind:'input',shape:[8,800,704],col:0,row:1}],edges:[]}}};
        return create(doc);
    }
    const PRESETS = Object.freeze({
        resnet_block: {
            schema_version: 1,
            name: 'ResNet Residual Block',
            source: 'He et al., CVPR 2016',
            shape_mode: 'manual drawing',
            views: {
                overview: {
                    title: 'ResNet Residual Block',
                    nodes: [
                        {id:'in',label:'Input x',kind:'input',shape:[64,56,56],col:0,row:1,layout:{x:100,y:120,width:130,height:65,fill:'#ffffff'}},
                        {id:'c1',label:'Conv 3×3',subtitle:'stride 1',kind:'conv',shape:[64,56,56],col:1,row:1,layout:{x:280,y:120,width:130,height:65,fill:'#a8cf91'}},
                        {id:'bn1',label:'BatchNorm',kind:'norm',shape:[64,56,56],col:2,row:1,layout:{x:440,y:120,width:120,height:60,fill:'#f5f5f5'}},
                        {id:'r1',label:'ReLU',kind:'activation',shape:[64,56,56],col:3,row:1,layout:{x:580,y:120,width:90,height:55,fill:'#f8ceb4'}},
                        {id:'c2',label:'Conv 3×3',subtitle:'stride 1',kind:'conv',shape:[64,56,56],col:4,row:1,layout:{x:710,y:120,width:130,height:65,fill:'#a8cf91'}},
                        {id:'bn2',label:'BatchNorm',kind:'norm',shape:[64,56,56],col:5,row:1,layout:{x:870,y:120,width:120,height:60,fill:'#f5f5f5'}},
                        {id:'add',label:'Add',symbol:'+',kind:'add',shape:[64,56,56],col:6,row:1,layout:{x:1020,y:120,width:44,height:44,fill:'#ffffff'}},
                        {id:'r2',label:'ReLU',kind:'activation',shape:[64,56,56],col:7,row:1,layout:{x:1110,y:120,width:90,height:55,fill:'#f8ceb4'}},
                        {id:'out',label:'Output',kind:'output',shape:[64,56,56],col:8,row:1,layout:{x:1250,y:120,width:120,height:60,fill:'#ffffff'}}
                    ],
                    edges: [
                        {id:'e1',source:'in',target:'c1'},{id:'e2',source:'c1',target:'bn1'},
                        {id:'e3',source:'bn1',target:'r1'},{id:'e4',source:'r1',target:'c2'},
                        {id:'e5',source:'c2',target:'bn2'},{id:'e6',source:'bn2',target:'add'},
                        {id:'e7',source:'add',target:'r2'},{id:'e8',source:'r2',target:'out'},
                        {id:'eskip',source:'in',target:'add',kind:'skip',label:'Identity shortcut',source_port:'bottom',target_port:'bottom'}
                    ],
                    groups: [
                        {id:'g_res',label:'Residual Block',node_ids:['c1','bn1','r1','c2','bn2','add','r2'],fill:'#fafafa',stroke:'#303030',padding:24}
                    ]
                }
            }
        },
        transformer_block: {
            schema_version: 1,
            name: 'Vision Transformer Layer',
            source: 'Dosovitskiy et al., ICLR 2021',
            shape_mode: 'manual drawing',
            views: {
                overview: {
                    title: 'Vision Transformer Layer',
                    nodes: [
                        {id:'in',label:'Input x',kind:'input',shape:[768,196,1],col:0,row:1,layout:{x:100,y:120,width:130,height:65,fill:'#ffffff'}},
                        {id:'ln1',label:'LayerNorm',kind:'layernorm',shape:[768,196,1],col:1,row:1,layout:{x:270,y:120,width:120,height:60,fill:'#f1f5f9'}},
                        {id:'mhsa',label:'MHSA',subtitle:'heads = 12',kind:'mhsa',shape:[768,196,1],col:2,row:1,layout:{x:440,y:120,width:150,height:70,fill:'#ede9fe'}},
                        {id:'add1',label:'Add',symbol:'+',kind:'add',shape:[768,196,1],col:3,row:1,layout:{x:590,y:120,width:44,height:44,fill:'#ffffff'}},
                        {id:'ln2',label:'LayerNorm',kind:'layernorm',shape:[768,196,1],col:4,row:1,layout:{x:680,y:120,width:120,height:60,fill:'#f1f5f9'}},
                        {id:'ffn',label:'FFN',subtitle:'dim = 3072',kind:'ffn',shape:[768,196,1],col:5,row:1,layout:{x:850,y:120,width:140,height:70,fill:'#fef3c7'}},
                        {id:'add2',label:'Add',symbol:'+',kind:'add',shape:[768,196,1],col:6,row:1,layout:{x:1000,y:120,width:44,height:44,fill:'#ffffff'}},
                        {id:'out',label:'Output',kind:'output',shape:[768,196,1],col:7,row:1,layout:{x:1130,y:120,width:120,height:60,fill:'#ffffff'}}
                    ],
                    edges: [
                        {id:'e1',source:'in',target:'ln1'},{id:'e2',source:'ln1',target:'mhsa'},
                        {id:'e3',source:'mhsa',target:'add1'},{id:'e4',source:'add1',target:'ln2'},
                        {id:'e5',source:'ln2',target:'ffn'},{id:'e6',source:'ffn',target:'add2'},
                        {id:'e7',source:'add2',target:'out'},
                        {id:'eskip1',source:'in',target:'add1',kind:'skip',label:'Residual',source_port:'bottom',target_port:'bottom'},
                        {id:'eskip2',source:'add1',target:'add2',kind:'skip',label:'Residual',source_port:'bottom',target_port:'bottom'}
                    ],
                    groups: [
                        {id:'g_vit',label:'Transformer Encoder Layer',node_ids:['ln1','mhsa','add1','ln2','ffn','add2'],fill:'#fafafa',stroke:'#303030',padding:24}
                    ]
                }
            }
        }
    });
    return {create,validate,validateGraph,layoutGraph,clone,History,blank,PRESETS,COLORS,LIBRARY,PORTS,CATEGORIES,SHAPE_TYPES,
        nodesInRect,translatedLayouts,moveNodes,removeNodes,alignNodes,distributeNodes,updateNode,addNode,duplicateNode,removeNode,addEdge,updateEdge,removeEdge,addGroup,updateGroup,removeGroup};
}));


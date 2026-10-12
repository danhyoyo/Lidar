/* Editable NN-SVG documents and transactional graph commands. MIT. */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory(require('./Backbone.js'));
    else root.BackboneEditor = factory(root.BackboneSVG);
}(typeof globalThis !== 'undefined' ? globalThis : this, function (core) {
    'use strict';
    const clone = value => JSON.parse(JSON.stringify(value));
    const COLORS = Object.freeze({input:'#ffffff',output:'#ffffff',conv:'#a8cf91',depthwise:'#dce8f5',
        block:'#dce8f5',norm:'#f5f5f5',activation:'#f8ceb4',scale:'#ffffff',add:'#ffffff',
        multiply:'#ffffff',upsample:'#ffe5a1',gate:'#ffffff',attention:'#dce8f5',pool:'#ece5f5',feature:'#ffffff'});
    const LIBRARY = Object.freeze({input:'Input tensor',conv:'Conv 3×3',depthwise:'DW 3×3',block:'Custom block',
        norm:'BatchNorm',activation:'SiLU',scale:'LayerScale γ',add:'Add',multiply:'Multiply',
        upsample:'Bilinear ↑2',gate:'Scale gate',attention:'Attention',pool:'Global mean',feature:'Feature tensor',output:'Output tensor'});
    const PORTS = ['auto','top','right','bottom','left'];
    const operators = new Set(['add','multiply','scale','gate']);
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
    function addNode(doc,view,kind='conv',preferred) {
        if(!Object.hasOwn(LIBRARY,kind)) throw new Error('Unknown block type.');
        return transaction(doc,d=>{
            const g=graphAt(d,view),id=nextId(g.nodes,'block'),index=g.nodes.length;
            const size={width:operators.has(kind)?44:176,height:operators.has(kind)?44:88};
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
    return {create,validate,validateGraph,layoutGraph,clone,History,blank,COLORS,LIBRARY,PORTS,
        nodesInRect,translatedLayouts,moveNodes,removeNodes,updateNode,addNode,duplicateNode,removeNode,addEdge,updateEdge,removeEdge,addGroup,updateGroup,removeGroup};
}));

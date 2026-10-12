/* Offline NN-SVG editor. Documents remain ordinary JSON; no server is required. */
(function () {
    'use strict';
    const $=id=>document.getElementById(id), E=BackboneEditor, canvas=$('canvas');
    const DRAFT_KEY=`papernn-studio-backbone-editor-v1-${window.BACKBONE_DIAGRAM.provenance?.config_sha256 || 'default'}-original-layout-v1`, mobile=matchMedia('(max-width:720px)');
    const tabs=['setup','blocks','groups','save'], pending=new Map(),selection=new Set();
    let history,view='overview',selected=null,selectedEdge=null,rendered,frame,drag=null;
    let mode='view',linkStart=null,groupSelection=null,suppressClick=false,feedbackTimer,flushing=false,flushingDraft;
    const doc=()=>history.value,graph=()=>doc().views[view];
    const editable=()=>['reference','paper'].includes($('presentation').value);
    const error=message=>{message={'Choose two different blocks.':'Chọn hai block khác nhau để nối.','These blocks are already connected.':'Hai block này đã có đường nối. Chọn đường đó để sửa.'}[message] || message;$('error').textContent=message;$('error').hidden=!message;};
    function feedback(message){clearTimeout(feedbackTimer);$('feedback').textContent=message;$('feedback').hidden=false;feedbackTimer=setTimeout(()=>$('feedback').hidden=true,5000);}
    function element(tag,text='',className){const n=document.createElement(tag);n.textContent=text;if(className)n.className=className;return n;}
    function option(value,label){const o=element('option',label);o.value=value;return o;}
    function input(id,value,type='text'){
        const n=element('input');n.id=id;n.name=id;n.type=type;n.value=value??'';
        if(type==='number')n.step='any';else if(type==='text')n.maxLength=1000;
        return n;
    }
    function field(parent,label,id,value,type='text',limits={}){
        const wrap=element('label',label);wrap.htmlFor=id;const n=input(id,value,type);
        Object.assign(n,limits);wrap.append(n);parent.append(wrap);return n;
    }
    function select(parent,label,id,values,value){
        const wrap=element('label',label),s=element('select');s.id=id;s.name=id;wrap.htmlFor=id;
        for(const [key,name] of values)s.append(option(key,name));s.value=value;wrap.append(s);parent.append(wrap);return s;
    }
    const value=id=>$(id).value,number=id=>Number(value(id));
    const options=()=>({spacing:number('spacing'),depth:number('depth'),fontSize:number('font-size'),
        shapes:$('shapes').checked,theme:$('dark').checked?'dark':'light',colors:{block:value('color-block'),depthwise:value('color-dw'),gate:value('color-gate'),add:value('color-gate'),multiply:value('color-gate')}});
    function activateTab(name,focus=false){
        for(const tab of tabs){const active=tab===name;$(`tab-${tab}`).setAttribute('aria-selected',String(active));$(`tab-${tab}`).tabIndex=active?0:-1;$(`panel-${tab}`).hidden=!active;}
        $('controls').scrollTop=0;if(focus)$(`tab-${name}`).focus();
    }
    function drawer(open){
        const show=mobile.matches&&open,aside=$('controls');aside.classList.toggle('is-open',show);
        aside.inert=mobile.matches&&!show;document.querySelector('main').inert=show;
        $('controls-backdrop').hidden=!show;$('toggle-controls').setAttribute('aria-expanded',String(show));
        if(show){aside.setAttribute('role','dialog');aside.setAttribute('aria-modal','true');$('close-controls').focus();}
        else{aside.removeAttribute('role');aside.removeAttribute('aria-modal');}
    }
    function showPanel(name){activateTab(name);if(mobile.matches)drawer(true);}
    for(const name of tabs)$(`tab-${name}`).addEventListener('click',()=>activateTab(name));
    document.querySelector('.panel-tabs').addEventListener('keydown',event=>{
        const index=tabs.findIndex(name=>`tab-${name}`===event.target.id);if(index<0)return;
        let next;if(event.key==='ArrowRight')next=(index+1)%tabs.length;else if(event.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;
        else if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else return;
        event.preventDefault();activateTab(tabs[next],true);
    });
    $('toggle-controls').addEventListener('click',()=>drawer(true));
    for(const id of ['close-controls','controls-backdrop'])$(id).addEventListener('click',()=>{drawer(false);$('toggle-controls').focus();});
    mobile.addEventListener('change',()=>drawer(false));drawer(false);
    function snapshot(form){return JSON.stringify([...form.querySelectorAll('input,select,textarea')].map(n=>[n.id,n.type==='checkbox'?n.checked:n.value]));}
    function registerForm(form,save,applyId,cancelId,cancel){
        const initial=snapshot(form),state=form.querySelector('.form-state');
        const item={form,dirty:()=>snapshot(form)!==initial,save:()=>{
            if(!form.checkValidity()){reportValidity(form);return false;}
            return save();
        }};
        pending.set(form.id,item);
        const update=()=>{const dirty=item.dirty();state.textContent=dirty?'Chưa áp dụng':'Đã áp dụng';state.classList.toggle('dirty',dirty);$(applyId).disabled=!dirty;$(cancelId).disabled=!dirty;};
        form.oninput=update;form.onchange=update;
        form.onsubmit=event=>{event.preventDefault();if(flushPending())feedback('Đã áp dụng thay đổi. Có thể hoàn tác bằng ↶.');};
        $(cancelId).onclick=()=>{pending.delete(form.id);cancel();error('');feedback('Đã hủy phần sửa chưa áp dụng.');};update();
    }
    function reportValidity(form){
        const panel=form.closest('[role=tabpanel]');if(panel)showPanel(panel.id.replace('panel-',''));
        const invalid=form.querySelector(':invalid');
        for(let n=invalid?.parentElement;n&&n!==form;n=n.parentElement)if(n.tagName==='DETAILS')n.open=true;
        form.reportValidity();error('Giá trị chưa hợp lệ. Kiểm tra trường được đánh dấu rồi áp dụng lại.');
    }
    function flushPending(){
        if(flushing)return true;
        const dirty=[...pending.values()].filter(item=>item.form.isConnected&&item.dirty());
        if(!dirty.length)return true;
        flushing=true;
        try{
            // Keep all forms mounted until every pending form has been processed.
            // A node and its group can both be edited before the next canvas action.
            for(const item of dirty)if(!item.form.checkValidity()){
                reportValidity(item.form);return false;
            }
            flushingDraft=E.clone(doc());
            for(const item of dirty)if(!item.save()){
                const panel=item.form.closest('[role=tabpanel]');if(panel)showPanel(panel.id.replace('panel-',''));return false;
            }
            history.commit(d=>Object.assign(d,flushingDraft));viewOptions();draw(false);persist();return true;
        }finally{flushing=false;flushingDraft=null;}
    }
    function formActions(form,label,applyId,cancelId){
        const actions=element('div','','form-actions'),state=element('span','','form-state');state.setAttribute('aria-live','polite');
        const apply=element('button',label,'primary');apply.type='submit';apply.id=applyId;
        const cancel=element('button','Hủy sửa','secondary');cancel.type='button';cancel.id=cancelId;actions.append(state,apply,cancel);form.append(actions);
    }
    function sizeHandles(){
        const svg=canvas.querySelector('svg'),matrix=svg?.getScreenCTM();if(!matrix)return;
        const scale=Math.hypot(matrix.a,matrix.b);if(!Number.isFinite(scale)||scale<=0)return;const size=24/scale;
        canvas.querySelectorAll('[data-resize]').forEach(handle=>{
            if(!handle.dataset.centerX){handle.dataset.centerX=Number(handle.getAttribute('x'))+Number(handle.getAttribute('width'))/2;handle.dataset.centerY=Number(handle.getAttribute('y'))+Number(handle.getAttribute('height'))/2;}
            handle.setAttribute('x',Number(handle.dataset.centerX)-size/2);handle.setAttribute('y',Number(handle.dataset.centerY)-size/2);handle.setAttribute('width',size);handle.setAttribute('height',size);
        });
    }
    function updatePhysicalSize(){
        if(!rendered) return;
        const guide = $('column-guide')?.value || 'double';
        const targetWidthMm = guide === 'single' ? 85 : 182;
        const targetHeightMm = Math.round(targetWidthMm * (rendered.height / rendered.width));
        const colText = guide === 'single' ? '1 cột (85mm)' : guide === 'double' ? '2 cột (182mm)' : 'Tự do';
        if ($('physical-size')) $('physical-size').textContent = `In: ${targetWidthMm} × ${targetHeightMm} mm [${colText}]`;
        let guideEl = canvas.querySelector('[data-paper-guide]');
        if (guide === 'none') { guideEl?.remove(); return; }
        const svg = canvas.querySelector('svg');
        if (!svg) return;
        if (!guideEl) {
            guideEl = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            guideEl.setAttribute('data-paper-guide', '');
            guideEl.setAttribute('class', 'paper-guide-box');
            guideEl.setAttribute('pointer-events', 'none');
            svg.prepend(guideEl);
        }
        guideEl.setAttribute('x', '2');
        guideEl.setAttribute('y', '2');
        guideEl.setAttribute('width', String(Math.max(10, rendered.width - 4)));
        guideEl.setAttribute('height', String(Math.max(10, rendered.height - 4)));
    }
    function setFrame(){const svg=canvas.querySelector('svg');if(svg&&frame){svg.setAttribute('viewBox',`${frame.x} ${frame.y} ${frame.w} ${frame.h}`);$('zoom-percent').value=`${Math.round(rendered.width/frame.w*100)}%`;sizeHandles();updatePhysicalSize();}}
    function fit(){if(rendered){frame={x:0,y:0,w:rendered.width,h:rendered.height};setFrame();}}
    new ResizeObserver(sizeHandles).observe(canvas);
    function persist(){
        try{localStorage.setItem(DRAFT_KEY,JSON.stringify({document:doc(),view,presentation:value('presentation')}));$('draft-status').textContent='Bản nháp đã lưu trên trình duyệt này. Lưu JSON để chuyển sang máy/trình duyệt khác.';}
        catch{$('draft-status').textContent='Trình duyệt không lưu được bản nháp. Dùng “Lưu JSON chỉnh sửa” để giữ bản vẽ.';}
    }
    function metadata(){
        $('model-name').textContent=doc().name || 'Custom architecture';
        const opts=doc().options || {},depths=Array.isArray(opts.stage_depths)?opts.stage_depths:[];
        $('model-summary').textContent=depths.length?`${doc().input_encoding || 'BEV'} · stages ${depths.join(' / ')} · output ${opts.backbone_out_dim || '?'} ch · fusion ${opts.neck_fusion_channels || '?'} ch`:'Thêm block và kết nối để tạo kiến trúc của bạn.';
        $('source').textContent=doc().editor?.modified?'Bản vẽ tùy chỉnh':doc().source || 'JSON import';
        $('shape-mode').textContent=String(doc().shape_mode || '').includes('manual')?'Shapes: nhập thủ công':String(doc().shape_mode || '').startsWith('verified')?'Shapes: PyTorch verified':'Shapes: config + code';
        $('undo').disabled=!history.canUndo;$('redo').disabled=!history.canRedo;
    }
    function viewOptions(){const current=view;$('view').replaceChildren();for(const [id,g] of Object.entries(doc().views))$('view').append(option(id,g.title));view=Object.hasOwn(doc().views,current)?current:'overview';$('view').value=view;}
    function groupCount(){const count=$('group-nodes').querySelectorAll('input:checked').length;$('group-count').textContent=`${count} block`;$('group-empty').hidden=count>0;}
    function groupOptions(){
        pending.delete('group-form');const groups=graph().groups || [];$('group-select').replaceChildren(option('','Chọn nhóm…'));
        for(const g of groups)$('group-select').append(option(g.id,g.label));
        if(!groups.some(g=>g.id===groupSelection))groupSelection=groups[0]?.id || null;
        $('group-select').value=groupSelection || '';const g=groups.find(g=>g.id===groupSelection);
        $('group-form').hidden=!g;$('remove-group').disabled=!g;$('add-group').disabled=!editable();
        if(!g)return;
        $('group-label').value=g.label;$('group-fill').value=g.fill || '#fafafa';$('group-nodes').replaceChildren();
        for(const n of graph().nodes){const label=element('label'),check=input(`member-${n.id}`,n.id,'checkbox');check.checked=g.node_ids.includes(n.id);const text=element('span',n.caption??n.label);text.append(element('small',`${n.id} · ${n.shape.join(' × ')}`));label.append(check,text);$('group-nodes').append(label);}
        groupCount();registerForm($('group-form'),()=>commit(d=>{E.updateGroup(d,view,g.id,{label:value('group-label'),fill:value('group-fill'),node_ids:[...$('group-nodes').querySelectorAll('input:checked')].map(n=>n.value)});return true;},false,true),'group-apply','group-cancel',groupOptions);
    }
    function edgeOptions(){
        for(const id of ['new-edge-source','new-edge-target']){
            const current=value(id);$(id).replaceChildren();for(const n of graph().nodes)$(id).append(option(n.id,`${n.caption??n.label} (${n.id})`));
            if(graph().nodes.some(n=>n.id===current))$(id).value=current;
        }
        if(value('new-edge-source')===value('new-edge-target')&&graph().nodes.length>1)$('new-edge-target').selectedIndex=1;
        $('add-edge').disabled=!editable()||graph().nodes.length<2;
    }
    function selectedClasses(){
        const valid=new Set(graph().nodes.map(n=>n.id));for(const id of selection)if(!valid.has(id))selection.delete(id);
        selected=selection.size===1?[...selection][0]:null;
        canvas.querySelectorAll('.tensor-node').forEach(n=>{
            const chosen=selection.has(n.dataset.node);n.classList.toggle('selected',chosen);n.classList.toggle('link-source',n.dataset.node===linkStart);n.setAttribute('aria-pressed',String(chosen));
        });
        canvas.querySelectorAll('.edge-node').forEach(n=>n.classList.toggle('selected',n.dataset.edge===selectedEdge));
        canvas.classList.toggle('multi-selection',selection.size>1);document.querySelector('main').classList.toggle('multi-selection',selection.size>1);
        $('selection-properties').disabled=!selection.size&&!selectedEdge;
        $('selection-count').hidden=!selection.size;$('selection-count').textContent=`${selection.size} đang chọn`;
    }
    function modeText(){
        const text=mode==='connect'?(linkStart?`Nguồn: ${graph().nodes.find(n=>n.id===linkStart)?.caption || linkStart}. Chọn block đích.`:'Chọn block nguồn, sau đó chọn block đích.'):
            mode==='select'?'Kéo nền để chọn block; Shift/Ctrl/Cmd để thêm vùng. Kéo block đã chọn để di chuyển cả cụm.':
            mode==='edit'?'Kéo thân block để di chuyển; kéo góc để đổi kích thước. Giữ Shift để căn lưới.':'Kéo nền: di chuyển · Shift + kéo nền: chọn vùng · Cuộn: zoom';
        $('interaction-state').textContent=text;$('mode-instruction').textContent=text;$('mode-banner').hidden=mode==='view';selectedClasses();
    }
    function setMode(next){
        mode=editable()?next:'view';linkStart=null;canvas.classList.toggle('editing',mode==='edit');canvas.classList.toggle('selecting',mode==='select');
        for(const [id,name] of [['select-mode','select'],['edit-mode','edit'],['connect-mode','connect']])$(id).setAttribute('aria-pressed',String(mode===name));modeText();
    }
    function syncSliders(){
        for(const [id,output,unit] of [['spacing','spacing-value','lần'],['depth','depth-value','lần'],['font-size','font-value','px']]){
            const current=Number(value(id));$(id+'-number').value=current;$(output).value=`${current}${unit==='px'?' px':'×'}`;$(id).setAttribute('aria-valuetext',`${current} ${unit}`);
        }
    }
    function draw(reset=true,preview,preservePanels=false){
        if(!history)return;
        try{
            const current=preview?preview.views[view]:graph(),style=value('presentation');
            $('3d-controls').hidden=style!=='3d';$('paper-help').hidden=style==='3d';
            $('legend-label').hidden=style==='3d';
            $('paper-help').textContent=editable()?'Chọn block để sửa; dùng tab Nhóm để tạo khung. Lưu JSON để mở lại bản vẽ.':'Bản này hiển thị từ config. Chọn kiểu 2D màu/trắng đen để chỉnh bố cục.';
            for(const id of ['select-mode','edit-mode','connect-mode','add-node','add-block'])$(id).disabled=!editable();
            $('editing-help').textContent=editable()?'Áp dụng để lưu thay đổi. Khi chuyển block, phần sửa hợp lệ cũng được áp dụng.':'Chọn kiểu 2D màu/trắng đen trong tab Kiến trúc để chỉnh bản vẽ.';
            if(style==='reference'||style==='paper')rendered=ReferenceSVG.render(current,{monochrome:style==='paper',showLegend:$('show-legend').checked});
            else if(style==='paper-composite')rendered=view==='overview'?PaperSVG.renderFigure(preview || doc()):PaperSVG.render(current);
            else rendered=BackboneSVG.render(current,options());
            const previousOffset=canvas.dataset.offset?JSON.parse(canvas.dataset.offset):{x:0,y:0};
            const nextOffset=rendered.offset || {x:0,y:0};
            if(!reset&&frame){frame.x+=nextOffset.x-previousOffset.x;frame.y+=nextOffset.y-previousOffset.y;}
            canvas.dataset.offset=JSON.stringify(nextOffset);
            canvas.innerHTML=rendered.svg;const svg=canvas.querySelector('svg');svg.setAttribute('role','group');svg.setAttribute('aria-label',current.title);
            canvas.querySelectorAll('.edge-node').forEach(edge=>{edge.tabIndex=0;edge.setAttribute('role','button');edge.setAttribute('aria-label',`Kết nối ${edge.dataset.source} đến ${edge.dataset.target}`);});
            if(reset||!frame)fit();else setFrame();canvas.classList.toggle('editing',mode==='edit');canvas.classList.toggle('selecting',mode==='select');selectedClasses();
            $('view-title').textContent=current.title;$('graph-count').textContent=`${current.nodes.length} khối · ${current.edges.length} kết nối`;syncSliders();
            if(!preview&&!preservePanels){const scroll=$('controls').scrollTop;metadata();inspect();groupOptions();edgeOptions();$('controls').scrollTop=scroll;error('');}
        }catch(err){error(err.message);}
    }
    function commit(command,reset=false,fromForm=false){
        if(!fromForm&&!flushPending())return null;
        try{if(fromForm&&flushing){const result=command(flushingDraft);E.validate(flushingDraft);return result;}
            const result=history.commit(command);viewOptions();draw(reset);persist();return result;}
        catch(err){error(err.message);return null;}
    }
    function changeView(id){if(!Object.hasOwn(doc().views,id)||!flushPending())return;view=id;selection.clear();selected=selectedEdge=null;groupSelection=null;linkStart=null;$('view').value=id;draw();setMode('view');persist();activateTab('setup');}
    function inspect(id,edgeId){
        if(arguments.length){selection.clear();if(id)selection.add(id);selectedEdge=edgeId || null;}
        selectedClasses();const panel=$('inspector');panel.replaceChildren();pending.delete('node-form');pending.delete('edge-form');
        const n=graph().nodes.find(n=>n.id===selected),edge=graph().edges.find(e=>e.id===selectedEdge);
        if(selection.size>1){multiInspector(panel);return;}
        if(!n&&!edge){selection.clear();selected=selectedEdge=null;selectedClasses();panel.append(element('p','Chọn block hoặc đường nối trên sơ đồ để sửa thuộc tính.','subtle'));return;}
        if(edge){edgeInspector(panel,edge);return;}
        panel.append(element('h2',n.caption??n.label),element('p',`B × ${n.shape.join(' × ')}`,'shape'));
        const form=element('form');form.id='node-form';form.noValidate=true;panel.append(form);
        const label=element('label','Nhãn block');label.htmlFor='node-label';const area=element('textarea');area.id='node-label';area.name=area.id;area.value=n.caption??n.label;area.rows=2;area.maxLength=500;area.required=true;label.append(area);form.append(label);
        select(form,'Loại block','node-kind',Object.entries(E.LIBRARY),n.kind || 'conv');
        const shapeRow=element('div','','field-grid three');form.append(shapeRow);
        for(const [i,name] of ['C','H','W'].entries())field(shapeRow,name,`node-shape-${i}`,n.shape[i],'number',{min:1,step:1,required:true});
        const subtitleLabel=element('label','Phụ đề / phép toán (mỗi dòng một phép)');subtitleLabel.htmlFor='node-subtitle';
        const subtitle=element('textarea');subtitle.id='node-subtitle';subtitle.name=subtitle.id;subtitle.rows=2;subtitle.maxLength=1000;subtitle.value=n.subtitle || '';subtitleLabel.append(subtitle);form.append(subtitleLabel);
        const row=element('div','','field-grid two');form.append(row);field(row,'Màu nền','node-fill',n.layout.fill || E.COLORS[n.kind],'color');
        field(row,'Cỡ chữ 2D (đơn vị SVG)','node-font',n.layout.font_size || 17,'number',{min:8,max:40,required:true});
        const font=field(form,'Kéo để chỉnh cỡ chữ','node-font-slider',n.layout.font_size || 17,'range',{min:8,max:40,step:.5});font.addEventListener('input',()=>{$('node-font').value=font.value;});$('node-font').addEventListener('input',()=>{font.value=value('node-font');});
        const position=element('details');position.open=true;position.append(element('summary','Vị trí và kích thước (đơn vị SVG)'));form.append(position);
        const grid=element('div','','field-grid two');position.append(grid);
        for(const [key,label] of [['x','X'],['y','Y'],['width','Rộng'],['height','Cao']])field(grid,label,`node-${key}`,n.layout[key],'number',{min:['width','height'].includes(key)?24:0,max:['width','height'].includes(key)?1200:20000,required:true});
        select(position,'Xoay chữ','node-rotation',[['0','Ngang'],['-90','Dọc ↑'],['90','Dọc ↓']],String(n.layout.rotation || 0));
        select(form,'Khung nhóm','node-group',[['','Không nhóm'],...graph().groups.map(g=>[g.id,g.label])],graph().groups.find(g=>g.node_ids.includes(n.id))?.id || '');
        const check=element('label','','check'),show=input('node-show-shape',null,'checkbox');show.checked=n.layout.show_shape!==false;check.append(show,document.createTextNode('Hiện C × H × W'));form.append(check);
        const channels=element('label','','check'),channelCheck=input('node-channel-projection',null,'checkbox');channelCheck.checked=!!n.channel_projection;channels.append(channelCheck,document.createTextNode('Hiện kênh đầu vào → đầu ra'));form.append(channels);
        const extras=element('details');extras.append(element('summary','Thông số block'));form.append(extras);
        const opGrid=element('div','','field-grid two');extras.append(opGrid);
        for(const [key,label] of [['kernel','Kernel'],['stride','Stride'],['repeat','Repeat'],['groups','Groups']])field(opGrid,label,`node-${key}`,n[key] ?? '','number',{min:1,max:1000000,step:1});
        field(extras,'Tên block lặp','node-repeat-label',n.repeat_label || 'Repeat');field(extras,'Ký hiệu vòng tròn (tùy chọn)','node-symbol',n.symbol || '');
        if(n.source)extras.append(element('code',n.source));for(const op of n.operations || [])extras.append(element('p',op));if(n.formula)extras.append(element('p',n.formula));
        formActions(form,'Áp dụng block','node-apply','node-cancel');
        const readPatch=()=>{
            const patch={label:value('node-label'),subtitle:value('node-subtitle'),kind:value('node-kind'),shape:[0,1,2].map(i=>number(`node-shape-${i}`)),group:value('node-group'),channel_projection:$('node-channel-projection').checked,repeat_label:value('node-repeat-label'),symbol:value('node-symbol') || null,
                layout:{x:number('node-x'),y:number('node-y'),width:number('node-width'),height:number('node-height'),fill:value('node-fill'),font_size:number('node-font'),rotation:number('node-rotation'),show_shape:$('node-show-shape').checked}};
            for(const key of ['kernel','stride','repeat','groups'])patch[key]=value(`node-${key}`)===''?null:number(`node-${key}`);
            return patch;
        };
        const baseline=readPatch();
        registerForm(form,()=>{
            const patch={},current=readPatch();
            for(const [key,v] of Object.entries(current)){
                if(key==='layout'){
                    const layout=Object.fromEntries(Object.entries(v).filter(([name,val])=>JSON.stringify(val)!==JSON.stringify(baseline.layout[name])));
                    if(Object.keys(layout).length)patch.layout=layout;
                }else if(JSON.stringify(v)!==JSON.stringify(baseline[key]))patch[key]=v;
            }
            return commit(d=>{E.updateNode(d,view,n.id,patch);return true;},false,true);
        },'node-apply','node-cancel',()=>inspect());
        const actions=element('div','','inline-controls');panel.append(actions);
        const duplicate=element('button','Nhân bản','secondary');duplicate.id='duplicate-node';duplicate.disabled=!editable();duplicate.addEventListener('click',()=>{const id=commit(d=>E.duplicateNode(d,view,n.id));if(id){inspect(id);revealNode(id);feedback('Đã nhân bản block ở vị trí trống.');}});
        const remove=element('button','Xóa block','secondary danger');remove.id='remove-node';remove.disabled=!editable()||graph().nodes.length<2;remove.addEventListener('click',()=>{if(commit(d=>{E.removeNode(d,view,n.id);return true;})){inspect(null);feedback('Đã xóa block và các đường nối liên quan. ↶ để hoàn tác.');}});actions.append(duplicate,remove);
        if(n.detail_view){const detail=element('button','Mở sơ đồ chi tiết ↗','secondary');detail.id='open-detail';detail.addEventListener('click',()=>changeView(n.detail_view));panel.append(detail);}
    }
    function multiInspector(panel){
        panel.append(element('h2',`${selection.size} block đang chọn`),element('p','Kéo một block đã chọn để di chuyển cả cụm. Shift/Ctrl/Cmd + nhấp để thêm hoặc bỏ block.','help'));
        const list=element('ul','','selected-block-list');for(const n of graph().nodes)if(selection.has(n.id))list.append(element('li',`${n.caption??n.label} (${n.id})`));panel.append(list);

        const alignTools = element('div', '', 'alignment-tools');
        alignTools.append(element('h3', 'Căn gióng các khối đã chọn'));
        const alignRowH = element('div', '', 'inline-controls');
        for (const [align, label] of [['left', '← Trái'], ['center-h', '↔ Giữa'], ['right', '→ Phải']]) {
            const btn = element('button', label, 'secondary small');
            btn.onclick = () => {
                commit(d => {
                    const updated = E.alignNodes(d.views[view], [...selection], align);
                    for (const u of updated) d.views[view].nodes.find(n => n.id === u.id).layout = u.layout;
                    return true;
                });
                feedback(`Đã căn ${label.toLowerCase()}. ↶ để hoàn tác.`);
            };
            alignRowH.append(btn);
        }
        const alignRowV = element('div', '', 'inline-controls');
        for (const [align, label] of [['top', '↑ Trên'], ['center-v', '↕ Giữa'], ['bottom', '↓ Dưới']]) {
            const btn = element('button', label, 'secondary small');
            btn.onclick = () => {
                commit(d => {
                    const updated = E.alignNodes(d.views[view], [...selection], align);
                    for (const u of updated) d.views[view].nodes.find(n => n.id === u.id).layout = u.layout;
                    return true;
                });
                feedback(`Đã căn ${label.toLowerCase()}. ↶ để hoàn tác.`);
            };
            alignRowV.append(btn);
        }
        alignTools.append(alignRowH, alignRowV);

        if (selection.size >= 3) {
            const distRow = element('div', '', 'inline-controls');
            const distH = element('button', 'Dãn đều ngang', 'secondary small');
            distH.onclick = () => {
                commit(d => {
                    const updated = E.distributeNodes(d.views[view], [...selection], 'horizontal');
                    for (const u of updated) d.views[view].nodes.find(n => n.id === u.id).layout = u.layout;
                    return true;
                });
                feedback('Đã dãn đều ngang các khối.');
            };
            const distV = element('button', 'Dãn đều dọc', 'secondary small');
            distV.onclick = () => {
                commit(d => {
                    const updated = E.distributeNodes(d.views[view], [...selection], 'vertical');
                    for (const u of updated) d.views[view].nodes.find(n => n.id === u.id).layout = u.layout;
                    return true;
                });
                feedback('Đã dãn đều dọc các khối.');
            };
            distRow.append(distH, distV);
            alignTools.append(distRow);
        }
        panel.append(alignTools);

        const group=element('button','Tạo nhóm từ vùng chọn','primary');group.id='group-selection';group.disabled=!editable();group.onclick=createSelectionGroup;panel.append(group);
        const remove=element('button','Xóa các block đã chọn','secondary danger');remove.id='remove-selection';remove.disabled=!editable()||selection.size>=graph().nodes.length;
        if(selection.size>=graph().nodes.length)remove.title='Cần giữ ít nhất một block trong bản vẽ.';remove.onclick=deleteSelection;panel.append(remove);
        const clear=element('button','Bỏ chọn tất cả','secondary');clear.id='clear-selection';clear.onclick=()=>{if(flushPending())inspect(null);};panel.append(clear);
    }
    function deleteSelection(){
        if(drag)endDrag(true);
        if(!editable()||!selection.size)return;
        if(selection.size>=graph().nodes.length){error('Cần giữ ít nhất một block trong bản vẽ.');return;}
        const ids=[...selection];if(commit(d=>{E.removeNodes(d,view,ids);return true;})){inspect(null);feedback(`Đã xóa ${ids.length} block và các đường nối liên quan. ↶ để hoàn tác.`);}
    }
    function createSelectionGroup(){
        if(!editable())return;const members=[...selection];
        const id=commit(d=>{const id=E.addGroup(d,view,`Group ${(graph().groups || []).length+1}`);if(members.length)E.updateGroup(d,view,id,{node_ids:members});return id;});
        if(id){groupSelection=id;groupOptions();showPanel('groups');$('group-label').focus();$('group-label').select();feedback(members.length?`Đã tạo nhóm từ ${members.length} block đang chọn.`:'Nhóm mới: tích block để tạo khung, rồi áp dụng.');}
    }
    function edgeInspector(panel,e){
        panel.append(element('h2','Kết nối block'));const form=element('form');form.id='edge-form';form.noValidate=true;panel.append(form);
        const values=graph().nodes.map(n=>[n.id,`${n.caption??n.label} (${n.id})`]);select(form,'Nguồn','edge-source',values,e.source);select(form,'Đích','edge-target',values,e.target);
        field(form,'Nhãn đường nối','edge-label',e.label || '');select(form,'Kiểu nét','edge-kind',[['flow','Liền'],['lateral','Lateral'],['skip','Đứt / residual']],e.kind || 'flow');
        const ports=E.PORTS.map(p=>[p,{auto:'Tự động',top:'Trên',right:'Phải',bottom:'Dưới',left:'Trái'}[p]]);
        select(form,'Cổng ra','edge-source-port',ports,e.source_port || 'auto');select(form,'Cổng vào','edge-target-port',ports,e.target_port || 'auto');
        const row=element('div','','field-grid two');form.append(row);field(row,'Điểm rẽ đầu tiên X (tùy chọn)','edge-bend-x',e.waypoints?.[0]?.x ?? '','number',{min:0,max:20000});field(row,'Điểm rẽ Y','edge-bend-y',e.waypoints?.[0]?.y ?? '','number',{min:0,max:20000});
        const edgeValues=()=>({source:value('edge-source'),target:value('edge-target'),label:value('edge-label'),kind:value('edge-kind'),source_port:value('edge-source-port'),target_port:value('edge-target-port')});
        const baseline=edgeValues(),bend=[value('edge-bend-x'),value('edge-bend-y')];
        formActions(form,'Áp dụng kết nối','edge-apply','edge-cancel');
        registerForm(form,()=>{
            const hasX=value('edge-bend-x')!=='',hasY=value('edge-bend-y')!=='';if(hasX!==hasY){error('Nhập cả X và Y của điểm rẽ, hoặc bỏ trống cả hai.');return false;}
            const patch=Object.fromEntries(Object.entries(edgeValues()).filter(([key,val])=>val!==baseline[key]));
            if(bend[0]!==value('edge-bend-x')||bend[1]!==value('edge-bend-y'))patch.waypoints=hasX?[{x:number('edge-bend-x'),y:number('edge-bend-y')},...(e.waypoints || []).slice(1)]:[];
            return commit(d=>{E.updateEdge(d,view,e.id,patch);return true;},false,true);
        },'edge-apply','edge-cancel',()=>inspect());
        const remove=element('button','Xóa kết nối','secondary danger');remove.id='remove-edge';remove.disabled=!editable();remove.addEventListener('click',()=>{if(commit(d=>{E.removeEdge(d,view,e.id);return true;})){inspect(null);feedback('Đã xóa kết nối. ↶ để hoàn tác.');}});panel.append(remove);
    }
    function revealNode(id){const node=canvas.querySelector(`[data-node="${id}"]`),a=node?.getBoundingClientRect(),b=canvas.getBoundingClientRect();if(a&&(a.left<b.left||a.right>b.right||a.top<b.top||a.bottom>b.bottom))fit();}
    function load(bundle,presentation){
        const next=new E.History(bundle);pending.clear();history=next;view='overview';selection.clear();selected=selectedEdge=null;groupSelection=null;
        if(presentation&&[...$('presentation').options].some(o=>o.value===presentation))$('presentation').value=presentation;
        viewOptions();setMode('view');draw();persist();
    }
    function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type})),a=element('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
    $('view').addEventListener('change',e=>{const next=e.target.value;changeView(next);$('view').value=view;});$('overview').addEventListener('click',()=>changeView('overview'));
    let lastPresentation='reference';
    $('presentation').addEventListener('change',()=>{if(!flushPending()){$('presentation').value=lastPresentation;return;}lastPresentation=value('presentation');setMode('view');draw();persist();});
    $('fit').addEventListener('click',fit);$('cancel-mode').addEventListener('click',()=>setMode('view'));
    $('select-mode').addEventListener('click',()=>{if(flushPending())setMode(mode==='select'?'view':'select');});
    $('edit-mode').addEventListener('click',()=>{if(flushPending())setMode(mode==='edit'?'view':'edit');});$('connect-mode').addEventListener('click',()=>{if(flushPending())setMode(mode==='connect'?'view':'connect');});
    function buildComponentPalette() {
        const container = $('palette-categories');
        if (!container || !E.CATEGORIES) return;
        container.replaceChildren();
        for (const [catKey, cat] of Object.entries(E.CATEGORIES)) {
            const details = element('details', '', 'palette-group');
            details.open = ['layers', 'norms', 'operations'].includes(catKey);
            const summary = element('summary', cat.label);
            details.append(summary);
            const grid = element('div', '', 'palette-button-grid');
            for (const kind of cat.kinds) {
                const btn = element('button', E.LIBRARY[kind] || kind, 'palette-btn');
                btn.type = 'button';
                btn.dataset.kind = kind;
                btn.title = `Thêm ${E.LIBRARY[kind] || kind}`;
                btn.style.borderLeft = `4px solid ${E.COLORS[kind] || '#cbd5e1'}`;
                btn.onclick = () => {
                    if (!editable()) return;
                    const box = canvas.getBoundingClientRect();
                    const point = svgPoint(box.x + box.width / 2, box.y + box.height / 2);
                    const offset = rendered.offset || { x: 0, y: 0 };
                    const id = commit(d => E.addNode(d, view, kind, { x: point.x - offset.x, y: point.y - offset.y }));
                    if (id) {
                        inspect(id);
                        revealNode(id);
                        setMode('edit');
                        feedback(`Đã thêm ${E.LIBRARY[kind] || kind}. Kéo hoặc sửa thuộc tính bên dưới.`);
                    }
                };
                grid.append(btn);
            }
            details.append(grid);
            container.append(details);
        }
    }

    $('add-block').addEventListener('click',()=>{showPanel('blocks');$('palette-categories')?.focus();});$('selection-properties').addEventListener('click',()=>showPanel('blocks'));
    for(const [kind,label] of Object.entries(E.LIBRARY))$('block-library').append(option(kind,label));$('block-library').value='conv';
    buildComponentPalette();
    $('add-node').addEventListener('click',()=>{
        if(!editable())return;const box=canvas.getBoundingClientRect(),point=svgPoint(box.x+box.width/2,box.y+box.height/2),offset=rendered.offset || {x:0,y:0};
        const id=commit(d=>E.addNode(d,view,value('block-library'),{x:point.x-offset.x,y:point.y-offset.y}));
        if(id){inspect(id);revealNode(id);setMode('edit');feedback('Đã thêm block ở vị trí trống. Sửa thuộc tính bên dưới.');}
    });
    function moveHistory(redo){if(drag)endDrag(true);if(!flushPending())return;redo?history.redo():history.undo();selection.clear();selected=selectedEdge=null;viewOptions();draw();setMode('view');persist();feedback(redo?'Đã làm lại.':'Đã hoàn tác.');}
    $('undo').addEventListener('click',()=>moveHistory(false));$('redo').addEventListener('click',()=>moveHistory(true));
    for(const id of ['spacing','depth','font-size']){
        $(id).addEventListener('input',()=>draw(false,null,true));
        $(id+'-number').addEventListener('change',()=>{const n=$(id+'-number');if(!n.checkValidity()||n.value===''){n.reportValidity();return;}$(id).value=n.value;draw(false,null,true);});
        $(id+'-number').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();event.target.dispatchEvent(new Event('change'));}});
    }
    for(const id of ['shapes','dark','color-block','color-dw','color-gate','show-legend'])$(id).addEventListener('input',()=>draw(false,null,true));
    $('column-guide')?.addEventListener('change', updatePhysicalSize);
    $('reset-3d').addEventListener('click',()=>{for(const id of ['spacing','depth','font-size','color-block','color-dw','color-gate'])$(id).value=$(id).defaultValue;$('shapes').checked=true;$('dark').checked=false;draw(false,null,true);feedback('Đã đặt lại trình bày 3D.');});
    $('import-button').addEventListener('click',()=>{if(flushPending())$('import-file').click();});
    $('import-file').addEventListener('change',async event=>{try{const file=event.target.files[0];if(!file||!flushPending())return;if(file.size>8*1024*1024)throw new Error('JSON vượt giới hạn 8 MB.');const bundle=JSON.parse(await file.text());load(bundle,bundle.editor?.presentation || 'reference');lastPresentation=value('presentation');feedback('Đã nạp bản vẽ JSON.');}catch(err){error(`Không nạp được JSON: ${err.message}`);}finally{event.target.value='';}});
    $('download').addEventListener('click',()=>{
        if(!rendered||!flushPending())return;
        const suffix={reference:'reference2d',paper:'mono2d','paper-composite':'paper2d','3d':''}[value('presentation')];
        const docName=(doc().name || 'architecture').toLowerCase().replace(/[^a-z0-9_-]+/g,'_');
        download(`${docName}-${view}${suffix?'-'+suffix:''}.svg`,rendered.svg,'image/svg+xml;charset=utf-8');
        feedback('Đã xuất SVG toàn bộ hình.');
    });
    function exportRaster(dpi = 600) {
        if (!rendered || !flushPending()) return;
        const svgStr = rendered.svg;
        const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
            const guide = $('column-guide')?.value || 'double';
            const widthMm = guide === 'single' ? 85 : 182;
            const targetWidth = Math.round((dpi / 25.4) * widthMm);
            const targetHeight = Math.round(targetWidth * (rendered.height / rendered.width));
            const offscreen = document.createElement('canvas');
            offscreen.width = targetWidth;
            offscreen.height = targetHeight;
            const ctx = offscreen.getContext('2d');
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, targetWidth, targetHeight);
            ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
            URL.revokeObjectURL(url);
            offscreen.toBlob(pngBlob => {
                if (!pngBlob) return;
                const pngUrl = URL.createObjectURL(pngBlob);
                const a = element('a');
                const docName = (doc().name || 'architecture').toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
                a.download = `${docName}-${view}-${dpi}dpi.png`;
                a.href = pngUrl;
                document.body.append(a);
                a.click();
                a.remove();
                setTimeout(() => URL.revokeObjectURL(pngUrl), 1000);
                feedback(`Đã xuất PNG chất lượng cao (${dpi} DPI).`);
            }, 'image/png');
        };
        img.src = url;
    }
    $('download-png')?.addEventListener('click', () => exportRaster(600));
    $('download-json').addEventListener('click',()=>{
        if(!flushPending())return;
        const saved=E.clone(doc());
        saved.editor.presentation=value('presentation');
        const docName=(doc().name || 'architecture').toLowerCase().replace(/[^a-z0-9_-]+/g,'_');
        download(`${docName}-custom.json`,JSON.stringify(saved,null,2)+'\n','application/json;charset=utf-8');
        feedback('Đã lưu bản vẽ JSON.');
    });
    function replaceDrawing(value){if(!flushPending())return false;selection.clear();selected=selectedEdge=null;pending.clear();commit(d=>{for(const key of Object.keys(d))delete d[key];Object.assign(d,E.create(value));},true);setMode('view');return true;}
    $('new-drawing').addEventListener('click',()=>{if(replaceDrawing(E.blank()))feedback('Bản vẽ mới. Thêm block trong tab Block; ↶ để quay lại.');});
    $('reset-preset').addEventListener('click',()=>{if(replaceDrawing(window.BACKBONE_DIAGRAM))feedback(`Đã khôi phục ${window.BACKBONE_DIAGRAM.name}. ↶ để quay lại.`);});
    try{$('download-legacy-draft').hidden=!localStorage.getItem('nn-svg-backbone-editor-v1');}catch{}
    $('download-legacy-draft').addEventListener('click',()=>{
        try{
            const saved=JSON.parse(localStorage.getItem('nn-svg-backbone-editor-v1') || 'null');
            if(!saved?.document)return;
            E.validate(saved.document);
            download('backbone-before-p16-custom.json',JSON.stringify(saved.document,null,2)+'\n','application/json;charset=utf-8');
            feedback('Đã tải bản nháp trước cập nhật. Dùng Nạp kiến trúc JSON để mở lại.');
        }catch(err){error(`Không đọc được bản nháp trước cập nhật: ${err.message}`);}
    });
    const presetOptions = [
        ['mobilepixornext', window.BACKBONE_DIAGRAM.name || 'MobilePixorNeXt (LiDAR BEV)'],
        ['resnet_block', 'ResNet Residual Block (CVPR 2016)'],
        ['transformer_block', 'Vision Transformer Layer (ICLR 2021)']
    ];
    for (const [key, label] of presetOptions) $('preset-select').append(option(key, label));
    $('load-preset').addEventListener('click', () => {
        const key = value('preset-select');
        const target = key === 'mobilepixornext' ? window.BACKBONE_DIAGRAM : E.PRESETS[key];
        if (target && replaceDrawing(target)) feedback(`Đã nạp mẫu: ${$('preset-select').selectedOptions[0]?.text || key}. ↶ để quay lại.`);
    });
    $('group-select').addEventListener('change',()=>{const next=value('group-select');if(flushPending()){groupSelection=next;groupOptions();}else $('group-select').value=groupSelection || '';});
    $('add-group').addEventListener('click',createSelectionGroup);
    $('remove-group').addEventListener('click',()=>{if(groupSelection&&commit(d=>{E.removeGroup(d,view,groupSelection);return true;}))feedback('Đã xóa nhóm; các block được giữ lại.');});
    $('group-nodes').addEventListener('change',groupCount);
    for(const [id,checked] of [['group-all',true],['group-none',false]])$(id).addEventListener('click',()=>{$('group-nodes').querySelectorAll('input').forEach(n=>n.checked=checked);$('group-form').dispatchEvent(new Event('input'));groupCount();});
    $('new-edge-form').addEventListener('submit',event=>{event.preventDefault();const source=value('new-edge-source'),target=value('new-edge-target');const id=commit(d=>E.addEdge(d,view,source,target));if(id){inspect(null,id);feedback('Đã tạo kết nối. Có thể sửa cổng và đường rẽ bên dưới.');}});
    function chooseNode(id,event={}){
        if(!flushPending())return;
        if(mode==='connect'){
            if(!linkStart){linkStart=id;inspect(id);modeText();}
            else if(id!==linkStart){const source=linkStart;if(commit(d=>E.addEdge(d,view,source,id))){setMode('edit');inspect(id);feedback('Đã nối hai block.');}}
            else{linkStart=null;inspect(null);modeText();feedback('Đã bỏ chọn nguồn. Chọn block nguồn mới.');}
        }else if(editable()&&(event.shiftKey||event.ctrlKey||event.metaKey)){
            selection.has(id)?selection.delete(id):selection.add(id);selectedEdge=null;inspect();showPanel('blocks');
        }else{inspect(id);showPanel('blocks');}
    }
    canvas.addEventListener('click',event=>{
        if(suppressClick){suppressClick=false;return;}const node=event.target.closest('.tensor-node'),edge=event.target.closest('.edge-node');
        if(node)chooseNode(node.dataset.node,event);else if(edge){if(flushPending()){inspect(null,edge.dataset.edge);showPanel('blocks');}}
        else if(mode==='view'&&flushPending())inspect(null);
    });
    canvas.addEventListener('dblclick',event=>{if(mode==='connect'||mode==='select'||event.shiftKey||event.ctrlKey||event.metaKey)return;const panel=event.target.closest('[data-view]');if(panel){changeView(panel.dataset.view);return;}const node=event.target.closest('.tensor-node'),n=node&&graph().nodes.find(n=>n.id===node.dataset.node);if(n?.detail_view)changeView(n.detail_view);});
    canvas.addEventListener('keydown',event=>{
        if(!['Enter',' '].includes(event.key))return;const node=event.target.closest('.tensor-node'),edge=event.target.closest('.edge-node');
        if(node){event.preventDefault();chooseNode(node.dataset.node,event);}else if(edge&&flushPending()){event.preventDefault();inspect(null,edge.dataset.edge);showPanel('blocks');}
    });
    function svgPoint(x,y,inverse){return new DOMPoint(x,y).matrixTransform(inverse || canvas.querySelector('svg').getScreenCTM().inverse());}
    function zoom(factor,anchor){
        if(!frame)return;anchor=anchor || {x:frame.x+frame.w/2,y:frame.y+frame.h/2};
        if(frame.w*factor<rendered.width*.12||frame.w*factor>rendered.width*4)return;
        frame.x=anchor.x-(anchor.x-frame.x)*factor;frame.y=anchor.y-(anchor.y-frame.y)*factor;frame.w*=factor;frame.h*=factor;setFrame();
    }
    $('zoom-in').addEventListener('click',()=>zoom(1/1.25));$('zoom-out').addEventListener('click',()=>zoom(1.25));
    canvas.addEventListener('wheel',event=>{if(!frame)return;event.preventDefault();if(drag)return;zoom(event.deltaY>0?1.12:1/1.12,svgPoint(event.clientX,event.clientY));},{passive:false});
    function paintSelectionBox(point){
        let box=canvas.querySelector('[data-selection-box]');
        if(!box){box=document.createElementNS('http://www.w3.org/2000/svg','rect');box.setAttribute('data-selection-box','');box.setAttribute('class','selection-box');box.setAttribute('aria-hidden','true');canvas.querySelector('svg').append(box);}
        box.setAttribute('x',Math.min(drag.anchor.x,point.x));box.setAttribute('y',Math.min(drag.anchor.y,point.y));
        box.setAttribute('width',Math.abs(point.x-drag.anchor.x));box.setAttribute('height',Math.abs(point.y-drag.anchor.y));
    }
    canvas.addEventListener('pointerdown',event=>{
        if(![0,1].includes(event.button)||!frame||drag)return;suppressClick=false;
        const node=event.target.closest('.tensor-node'),resize=event.target.closest('[data-resize]'),modified=event.shiftKey||event.ctrlKey||event.metaKey;
        const pan=event.button===1;
        const canMove=node&&editable()&&(mode==='edit'||mode==='select'||mode==='view'&&selection.size>1&&selection.has(node.dataset.node));
        const snapDrag=canMove&&selection.has(node.dataset.node)&&event.shiftKey&&!event.ctrlKey&&!event.metaKey;
        if(node&&modified&&!pan&&mode!=='connect'&&!snapDrag)return;
        if(!pan&&!node&&editable()&&(mode==='select'||modified)){
            if(!flushPending())return;
            const inverse=canvas.querySelector('svg').getScreenCTM().inverse();
            drag={type:'selection',anchor:svgPoint(event.clientX,event.clientY,inverse),inverse,offset:rendered.offset || {x:0,y:0},base:[...selection],edge:selectedEdge,additive:modified,moved:false};
        }else if(!pan&&canMove){
            if(!flushPending())return;const id=node.dataset.node,n=graph().nodes.find(n=>n.id===id);if(!n)return;
            if(!selection.has(id))inspect(id);else inspect();
            const inverse=canvas.querySelector('svg').getScreenCTM().inverse();
            drag={type:resize&&selection.size===1?'resize':'node',id,ids:[...selection],anchor:svgPoint(event.clientX,event.clientY,inverse),inverse,layout:{...n.layout},base:graph(),draft:E.clone(doc()),moved:false};
        }else{
            if(!pan&&(node||event.target.closest('.edge-node,[data-view]')))return;
            const inverse=canvas.querySelector('svg').getScreenCTM().inverse();drag={type:'pan',anchor:svgPoint(event.clientX,event.clientY,inverse),inverse,frame:{...frame},moved:false};
        }
        drag.modifiers={shiftKey:event.shiftKey,ctrlKey:event.ctrlKey,metaKey:event.metaKey};
        drag.toggleClick=modified;drag.pointerId=event.pointerId;drag.client={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);event.preventDefault();
    });
    canvas.addEventListener('pointermove',event=>{
        if(!drag||event.pointerId!==drag.pointerId)return;
        if(Math.hypot(event.clientX-drag.client.x,event.clientY-drag.client.y)<3&&!drag.moved)return;drag.moved=true;
        const point=svgPoint(event.clientX,event.clientY,drag.inverse),dx=point.x-drag.anchor.x,dy=point.y-drag.anchor.y;
        if(drag.type==='pan'){frame.x=drag.frame.x-dx;frame.y=drag.frame.y-dy;setFrame();return;}
        if(drag.type==='selection'){
            paintSelectionBox(point);const o=drag.offset,ids=E.nodesInRect(graph(),{x1:drag.anchor.x-o.x,y1:drag.anchor.y-o.y,x2:point.x-o.x,y2:point.y-o.y});
            selection.clear();for(const id of [...(drag.additive?drag.base:[]),...ids])selection.add(id);selectedEdge=null;selectedClasses();return;
        }
        if(drag.type==='node'){
            drag.translation={dx,dy,snap:event.shiftKey};
            const updates=new Map(E.translatedLayouts(drag.base,drag.ids,dx,dy,event.shiftKey).map(n=>[n.id,n.layout]));
            drag.draft.views[view].nodes.forEach(n=>{if(updates.has(n.id))n.layout=updates.get(n.id);});
        }else{
            const l=drag.layout,updated={...l,width:Math.max(24,Math.min(1200,l.width+dx)),height:Math.max(24,Math.min(1200,l.height+dy))};
            updated.x=l.x+(updated.width-l.width)/2;updated.y=l.y+(updated.height-l.height)/2;
            drag.draft.views[view].nodes.find(n=>n.id===drag.id).layout=updated;
        }
        draw(false,drag.draft);
    });
    function endDrag(cancel){
        const last=drag;drag=null;if(!last)return;
        if(last.type==='selection'){
            canvas.querySelector('[data-selection-box]')?.remove();
            if(cancel||!last.moved){selection.clear();for(const id of cancel||last.additive?last.base:[])selection.add(id);selectedEdge=cancel?last.edge:null;}
            inspect();if(!cancel&&selection.size&&!mobile.matches)activateTab('blocks');suppressClick=true;
        }else if(last.type==='pan'){
            if(cancel){frame=last.frame;setFrame();}if(last.moved)suppressClick=true;
        }else if(!last.moved){
            if(!cancel){if(last.toggleClick)chooseNode(last.id,last.modifiers);else showPanel('blocks');}
            suppressClick=true;
        }
        else{
            suppressClick=true;
            if(cancel)draw(false);
            else commit(d=>last.type==='node'?
                E.moveNodes(d,view,last.ids,last.translation.dx,last.translation.dy,last.translation.snap):
                E.updateNode(d,view,last.id,{layout:{...last.draft.views[view].nodes.find(n=>n.id===last.id).layout}}));
        }
        if(canvas.hasPointerCapture(last.pointerId))canvas.releasePointerCapture(last.pointerId);
    }
    canvas.addEventListener('pointerup',event=>{if(drag?.pointerId===event.pointerId)endDrag(false);});
    canvas.addEventListener('pointercancel',event=>{if(drag?.pointerId===event.pointerId)endDrag(true);});
    canvas.addEventListener('lostpointercapture',()=>{if(drag)endDrag(true);});
    document.addEventListener('keydown',event=>{
        if(event.key==='Escape'){if(drag)endDrag(true);else if(!event.target.closest('input,textarea,select')&&flushPending())inspect(null);setMode('view');if(mobile.matches&&$('controls').classList.contains('is-open')){drawer(false);$('toggle-controls').focus();}return;}
        if(mobile.matches&&$('controls').classList.contains('is-open')&&event.key==='Tab'){
            const focusable=[...$('controls').querySelectorAll('a,button,input,select,textarea,summary,[tabindex]')].filter(n=>n.tabIndex>=0&&!n.disabled&&n.getClientRects().length);
            const first=focusable[0],last=focusable.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}return;
        }
        if(event.target.closest('input,textarea,select')||!history)return;
        if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();moveHistory(event.shiftKey);}
        else if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='a'&&editable()){event.preventDefault();if(drag)endDrag(true);if(flushPending()){selection.clear();graph().nodes.forEach(n=>selection.add(n.id));selectedEdge=null;inspect();showPanel('blocks');}}
        else if(event.key==='Delete'&&editable()&&selection.size){event.preventDefault();deleteSelection();}
    });
    try{
        let saved;try{saved=JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null');if(saved)E.validate(saved.document);}catch{saved=null;}
        load(saved?.document || window.BACKBONE_DIAGRAM,saved?.presentation || 'reference');lastPresentation=value('presentation');
        if(saved?.view&&Object.hasOwn(doc().views,saved.view))changeView(saved.view);
    }catch(err){error(`Không nạp được bản vẽ: ${err.message}`);}
}());

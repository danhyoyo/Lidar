'use strict';
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require('playwright-core');
const viewer=path.resolve(__dirname,'..');
const drawing=page=>page.evaluate(()=>JSON.parse(localStorage.getItem('nn-svg-backbone-editor-v1')).document);
const selected=page=>page.locator('.tensor-node.selected').evaluateAll(ns=>ns.map(n=>n.dataset.node).sort());
async function body(page,id){return page.locator(`[data-node="${id}"] > rect`).first().boundingBox();}
async function dragBox(page,ids,reverse=false,modifiers=[]){
    const boxes=await Promise.all(ids.map(id=>body(page,id)));
    const from={x:Math.min(...boxes.map(b=>b.x))-10,y:Math.min(...boxes.map(b=>b.y))-10};
    const to={x:Math.max(...boxes.map(b=>b.x+b.width))+10,y:Math.max(...boxes.map(b=>b.y+b.height))+10};
    for(const key of modifiers)await page.keyboard.down(key);
    const start=reverse?to:from,end=reverse?from:to;
    await page.mouse.move(start.x,start.y);await page.mouse.down();
    await page.mouse.move(end.x,end.y,{steps:8});
    assert.equal(await page.locator('[data-selection-box]').count(),1);
    await page.mouse.up();for(const key of modifiers)await page.keyboard.up(key);
    assert.equal(await page.locator('[data-selection-box]').count(),0);
}
(async()=>{
    const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
    try{
        const page=await browser.newPage({viewport:{width:1700,height:1100},acceptDownloads:true});page.setDefaultTimeout(10000);
        const errors=[];page.on('pageerror',e=>errors.push(e.message));
        await page.goto(pathToFileURL(path.join(viewer,'Backbone.html')).href);
        assert.equal(await page.locator('#select-mode').count(),1,'The editor must offer drag-to-select mode');
        const original=await drawing(page);
        await page.locator('#select-mode').click();
        await dragBox(page,['stage2','stage3','stage4']);
        assert.deepEqual(await selected(page),['stage2','stage3','stage4']);
        assert.match(await page.locator('#selection-count').textContent(),/3/);
        assert.deepEqual(await drawing(page),original,'Selection must not mutate the drawing');
        await dragBox(page,['stage2','stage3','stage4'],true);
        assert.deepEqual(await selected(page),['stage2','stage3','stage4']);
        await page.locator('[data-node="stage4"]').click({modifiers:['Shift']});assert.deepEqual(await selected(page),['stage2','stage3']);
        await page.locator('[data-node="stage4"]').click({modifiers:['Shift']});assert.deepEqual(await selected(page),['stage2','stage3','stage4']);
        await page.locator('[data-node="l3"]').click({modifiers:['Shift']});
        assert.deepEqual(await selected(page),['l3','stage2','stage3','stage4']);
        await page.locator('[data-node="l3"]').click({modifiers:['Control']});
        assert.deepEqual(await selected(page),['stage2','stage3','stage4']);
        const before=await drawing(page),b=await body(page,'stage3');
        await page.mouse.move(b.x+b.width/2,b.y+b.height/2);await page.mouse.down();await page.mouse.move(b.x+b.width/2+35,b.y+b.height/2+18,{steps:6});await page.mouse.up();
        const moved=await drawing(page),delta=moved.views.overview.nodes.find(n=>n.id==='stage2').layout.x-before.views.overview.nodes.find(n=>n.id==='stage2').layout.x;
        assert.ok(delta>10);
        for(const id of ['stage2','stage3','stage4']){
            const a=before.views.overview.nodes.find(n=>n.id===id),z=moved.views.overview.nodes.find(n=>n.id===id);
            assert.ok(Math.abs((z.layout.x-a.layout.x)-delta)<1e-6);assert.deepEqual(z.shape,a.shape);
        }
        assert.deepEqual(moved.views.overview.edges,before.views.overview.edges);
        assert.equal(moved.shape_mode,before.shape_mode);
        await page.locator('#undo').click();assert.deepEqual(await drawing(page),before);
        await page.locator('#redo').click();assert.deepEqual(await drawing(page),moved);
        // Shift held from pointer-down snaps the whole selection without toggling it.
        await page.locator('#select-mode').click();await dragBox(page,['stage2','stage3','stage4']);
        const snapBefore=await drawing(page),snapBody=await body(page,'stage3');
        await page.keyboard.down('Shift');await page.mouse.move(snapBody.x+snapBody.width/2,snapBody.y+snapBody.height/2);
        await page.mouse.down();await page.mouse.move(snapBody.x+snapBody.width/2+26,snapBody.y+snapBody.height/2+11,{steps:5});await page.mouse.up();await page.keyboard.up('Shift');
        const snapped=await drawing(page),snapAnchor=snapped.views.overview.nodes.find(n=>n.id==='stage2');
        assert.notDeepEqual(snapped,snapBefore,'Shift+drag must move the selection');
        assert.equal(snapAnchor.layout.x%10,0);assert.equal(snapAnchor.layout.y%10,0);
        assert.deepEqual(await selected(page),['stage2','stage3','stage4']);
        await page.locator('#undo').click();assert.deepEqual(await drawing(page),snapBefore);
        // A cancelled move restores positions without adding an undo entry.
        await page.locator('#select-mode').click();await dragBox(page,['stage2','stage3','stage4']);
        const cancelBefore=await drawing(page),cancelBody=await body(page,'stage3');
        await page.mouse.move(cancelBody.x+cancelBody.width/2,cancelBody.y+cancelBody.height/2);await page.mouse.down();
        await page.mouse.move(cancelBody.x+cancelBody.width/2+44,cancelBody.y+cancelBody.height/2+20,{steps:5});
        await page.keyboard.press('Escape');await page.mouse.up();assert.deepEqual(await drawing(page),cancelBefore);
        assert.deepEqual(await selected(page),['stage2','stage3','stage4']);
        // Escape cancels an in-progress marquee and restores its original selection.
        await page.locator('#select-mode').click();await dragBox(page,['stage2','stage3','stage4']);
        const c=await page.locator('#canvas').boundingBox();
        await page.mouse.move(c.x+5,c.y+5);await page.mouse.down();await page.mouse.move(c.x+30,c.y+30,{steps:3});
        await page.keyboard.press('Escape');await page.mouse.up();
        assert.deepEqual(await selected(page),['stage2','stage3','stage4']);
        assert.equal(await page.locator('[data-selection-box]').count(),0);
        await page.locator('#select-mode').click();
        await page.mouse.move(c.x+5,c.y+5);await page.mouse.down();await page.mouse.move(c.x+30,c.y+30,{steps:3});
        await page.locator('#canvas').dispatchEvent('pointercancel',{pointerId:1});await page.mouse.up();
        assert.deepEqual(await selected(page),['stage2','stage3','stage4']);assert.equal(await page.locator('[data-selection-box]').count(),0);
        // Group creation uses exactly the selected block IDs.
        await page.locator('#group-selection').click();
        const grouped=await drawing(page),newGroup=grouped.views.overview.groups.at(-1);
        assert.deepEqual(newGroup.node_ids.sort(),['stage2','stage3','stage4']);
        assert.equal(await page.locator(`[data-group="${newGroup.id}"]`).count(),1);
        // Selection stays accurate after pan/zoom and never appears in exported SVG.
        await page.locator('#zoom-in').click();assert.equal(await page.locator('#zoom-percent').textContent(),'125%');
        const panStart=await page.locator('#canvas').boundingBox(),frameBeforePan=await page.locator('#canvas svg').getAttribute('viewBox');
        await page.mouse.move(panStart.x+5,panStart.y+5);await page.mouse.down({button:'middle'});
        await page.mouse.move(panStart.x+26,panStart.y+19,{steps:3});await page.mouse.up({button:'middle'});
        assert.notEqual(await page.locator('#canvas svg').getAttribute('viewBox'),frameBeforePan);
        await dragBox(page,['stage2','stage3']);
        assert.deepEqual(await selected(page),['stage2','stage3']);
        await page.locator('[data-node="l3"]').click({modifiers:['Meta']});assert.deepEqual(await selected(page),['l3','stage2','stage3']);
        await page.locator('[data-node="l3"]').click({modifiers:['Meta']});
        const downloadPromise=page.waitForEvent('download');await page.locator('#download').click();
        const download=await downloadPromise,stream=await download.createReadStream(),chunks=[];for await(const chunk of stream)chunks.push(chunk);
        const svg=Buffer.concat(chunks).toString();assert.doesNotMatch(svg,/data-selection-box|multi-selection|class="tensor-node[^\"]*selected/);
        await page.locator('#remove-selection').click();assert.equal(await page.locator('[data-node="stage2"]').count(),0);assert.equal(await page.locator('[data-node="stage3"]').count(),0);
        await page.locator('#undo').click();assert.equal(await page.locator('[data-node="stage2"]').count(),1);assert.equal(await page.locator('[data-node="stage3"]').count(),1);
        await page.locator('#canvas').focus();await page.keyboard.press('Control+a');assert.equal((await selected(page)).length,14);
        const allBefore=await drawing(page),allBody=await body(page,'stage3');
        await page.mouse.move(allBody.x+allBody.width/2,allBody.y+allBody.height/2);await page.mouse.down();
        await page.mouse.move(allBody.x+allBody.width/2+15,allBody.y+allBody.height/2+8,{steps:3});
        await page.keyboard.press('Control+a');await page.mouse.up();assert.deepEqual(await drawing(page),allBefore);assert.equal((await selected(page)).length,14);
        await page.keyboard.press('Escape');assert.deepEqual(await selected(page),[]);
        // Default mode pans; Shift+background drag temporarily selects.
        const viewBox=await page.locator('#canvas svg').getAttribute('viewBox'),r=await page.locator('#canvas').boundingBox();
        await page.mouse.move(r.x+5,r.y+5);await page.mouse.down();await page.mouse.move(r.x+25,r.y+15,{steps:3});await page.mouse.up();
        assert.notEqual(await page.locator('#canvas svg').getAttribute('viewBox'),viewBox);
        await page.locator('#fit').click();await dragBox(page,['stage2','stage3'],false,['Shift']);assert.deepEqual(await selected(page),['stage2','stage3']);
        // Add a second marquee without clearing the first one.
        await dragBox(page,['stage4'],true,['Control']);assert.deepEqual(await selected(page),['stage2','stage3','stage4']);
        // Switching to connection mode must work even when its source is in a multi-selection.
        const beforeLink=await drawing(page);await page.locator('#connect-mode').click();await page.locator('[data-node="stage2"]').click();
        assert.equal(await page.locator('[data-node="stage2"]').evaluate(n=>n.classList.contains('link-source')),true);
        await page.locator('[data-node="output"]').click();assert.equal(await page.locator('.edge-node[data-source="stage2"][data-target="output"]').count(),1);
        await page.locator('#undo').click();assert.deepEqual(await drawing(page),beforeLink);
        // Invalid pending fields block a new selection and retain both the draft and selection.
        await page.locator('[data-node="l3"]').click();await page.locator('#node-shape-0').fill('0');
        const rejectedBefore=await drawing(page),rejectedBody=await body(page,'stage2');
        await page.keyboard.down('Shift');await page.mouse.move(rejectedBody.x-10,rejectedBody.y-10);await page.mouse.down();
        await page.mouse.move(rejectedBody.x+rejectedBody.width+10,rejectedBody.y+rejectedBody.height+10,{steps:4});await page.mouse.up();await page.keyboard.up('Shift');
        assert.equal(await page.locator('#node-shape-0').inputValue(),'0');assert.deepEqual(await selected(page),['l3']);assert.deepEqual(await drawing(page),rejectedBefore);
        await page.locator('#node-cancel').click();await page.locator('#select-mode').click();await dragBox(page,['stage2','stage3']);
        const beforeKeyboardDelete=await drawing(page);await page.locator('#canvas').focus();await page.keyboard.press('Delete');
        assert.equal(await page.locator('[data-node="stage2"]').count(),0);assert.equal(await page.locator('[data-node="stage3"]').count(),0);
        await page.keyboard.press('Control+z');assert.deepEqual(await drawing(page),beforeKeyboardDelete);
        await page.locator('#select-mode').click();await dragBox(page,['stage2','stage3']);
        if(process.argv[2])await page.screenshot({path:process.argv[2]});
        // Real touch pointer events use the same selection and group-move workflow.
        const phone=await browser.newPage({viewport:{width:390,height:844},hasTouch:true,isMobile:true});
        phone.on('pageerror',e=>errors.push(e.message));await phone.goto(pathToFileURL(path.join(viewer,'Backbone.html')).href);
        await phone.locator('#select-mode').tap();
        const session=await phone.context().newCDPSession(phone);
        const touch=(type,p)=>session.send('Input.dispatchTouchEvent',{type,touchPoints:p?[{x:p.x,y:p.y,id:1}]:[]});
        async function swipe(from,to){
            await touch('touchStart',from);
            for(let i=1;i<=8;i++)await touch('touchMove',{x:from.x+(to.x-from.x)*i/8,y:from.y+(to.y-from.y)*i/8});
            await touch('touchEnd');
        }
        const phoneBodies=await Promise.all(['stage2','stage3'].map(id=>body(phone,id)));
        const phoneCanvas=await phone.locator('#canvas').boundingBox(),phoneSelection=['input','stage2','stage3','stem'];
        // Start on clear background, outside Chromium's expanded touch targets.
        await swipe({x:phoneCanvas.x+5,y:phoneCanvas.y+20},
            {x:Math.max(...phoneBodies.map(b=>b.x+b.width))+3,y:Math.max(...phoneBodies.map(b=>b.y+b.height))+3});
        assert.deepEqual(await selected(phone),phoneSelection);assert.equal(await phone.locator('main').evaluate(n=>n.inert),false,'Touch marquee must leave the canvas available');
        assert.deepEqual(await phone.locator('#canvas').boundingBox(),phoneCanvas,'Selection must not reflow the toolbar and move the canvas under a finger');
        const phoneBefore=await drawing(phone),phoneBody=await body(phone,'stage3');
        await swipe({x:phoneBody.x+phoneBody.width/2,y:phoneBody.y+phoneBody.height/2},{x:phoneBody.x+phoneBody.width/2+18,y:phoneBody.y+phoneBody.height/2+5});
        const phoneMoved=await drawing(phone),phoneDx=id=>phoneMoved.views.overview.nodes.find(n=>n.id===id).layout.x-phoneBefore.views.overview.nodes.find(n=>n.id===id).layout.x;
        assert.ok(phoneDx('stage2')>0);assert.ok(Math.abs(phoneDx('stage2')-phoneDx('stage3'))<1e-6);
        await phone.locator('#selection-properties').tap();await phone.locator('#group-selection').tap();
        assert.deepEqual((await drawing(phone)).views.overview.groups.at(-1).node_ids.sort(),phoneSelection);
        await phone.locator('#close-controls').tap();assert.equal(await phone.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
        if(process.argv[2])await phone.screenshot({path:process.argv[2].replace(/\.png$/,'-mobile.png')});
        assert.deepEqual(errors,[]);
        console.log('Selection browser checks passed: forward/reverse marquee, modifier toggles, shared move/undo and grid snap, cancel, exact group membership, zoom/pan, delete/undo, pending validation, select all, clean SVG export and touch selection/move/group.');
    }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

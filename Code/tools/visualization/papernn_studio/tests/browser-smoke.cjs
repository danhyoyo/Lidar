/* Optional browser checks: requires Playwright and its Chromium headless shell. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright-core');
const viewer = path.resolve(__dirname, '..');

(async () => {
    const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
    try {
        const page = await browser.newPage({viewport: {width: 1500, height: 960}, acceptDownloads: true});
        const errors = [], external = [];
        page.on('pageerror', e => errors.push(e.message));
        page.on('request', request => { if (/^https?:/.test(request.url())) external.push(request.url()); });
        await page.goto(pathToFileURL(path.join(viewer, 'Backbone.html')).href);
        await page.waitForSelector('.tensor-node');
        assert.equal(await page.locator('#error').isVisible(), false);
        assert.equal(await page.locator('#view option').count(), 13);
        assert.equal(await page.locator('.tensor-node').count(), 14);
        assert.equal(await page.locator('#presentation').inputValue(), 'reference');
        await page.locator('#presentation').selectOption('paper-composite');
        assert.equal(await page.locator('#canvas polygon').count(), 0);
        assert.match(await page.locator('#canvas').innerText(), /\(b\) NeXt block/);
        // Verify text bounds at the publication layout before testing custom labels.
        const outside = await page.locator('#canvas svg').evaluate(svg => {
            const frame = svg.viewBox.baseVal;
            return [...svg.querySelectorAll('text')].filter(t => {
                const b = t.getBBox();
                return b.x < -0.1 || b.y < -0.1 || b.x+b.width > frame.width+0.1 || b.y+b.height > frame.height+0.1;
            }).map(t => t.textContent);
        });
        assert.deepEqual(outside, []);
        await page.locator('[data-view="block2"]').dblclick();
        assert.equal(await page.locator('#view').inputValue(), 'block2');
        await page.locator('#overview').click();
        const paperDownloading = page.waitForEvent('download');
        await page.locator('#download').click();
        assert.equal((await paperDownloading).suggestedFilename(), 'mobilepixornext-overview-paper2d.svg');
        const preset = JSON.parse(fs.readFileSync(path.join(viewer, 'mobilepixornext.json'), 'utf8'));
        for (const id of Object.keys(preset.views)) {
            await page.locator('#view').selectOption(id);
            assert.equal(await page.locator('#error').isVisible(), false, id);
            assert.equal(await page.locator('#canvas polygon').count(), 0);
        }
        await page.locator('#overview').click();
        await page.locator('#presentation').selectOption('3d');
        assert.ok(await page.locator('#canvas polygon').count());

        await page.locator('[data-node="stage2"]').click();
        assert.match(await page.locator('#inspector').innerText(), /48 × 200 × 176/);
        await page.locator('#open-detail').click();
        assert.equal(await page.locator('#view').inputValue(), 'stage2');
        await page.locator('[data-node="block0"]').dblclick();
        assert.equal(await page.locator('#view').inputValue(), 'block2');
        assert.match(await page.locator('#canvas').innerText(), /120 × 200 × 176/);
        await page.locator('#tab-setup').click();await page.locator('#view').selectOption('gate4');
        assert.equal(await page.locator('[data-node="multiply"]').count(), 1);

        await page.locator('#overview').click();
        await page.locator('#spacing').fill('1.3');
        await page.locator('#dark').check();
        assert.equal(await page.locator('#canvas svg > rect').first().getAttribute('fill'), '#111827');
        const original = await page.locator('#canvas svg').getAttribute('viewBox');
        const rect = await page.locator('#canvas').boundingBox();
        await page.mouse.move(rect.x+rect.width/2, rect.y+rect.height/2);
        await page.mouse.wheel(0, -120);
        await page.waitForFunction(old => document.querySelector('#canvas svg').getAttribute('viewBox') !== old, original);
        const zoomed = await page.locator('#canvas svg').getAttribute('viewBox');
        await page.mouse.move(rect.x+20, rect.y+20);
        await page.mouse.down();
        await page.mouse.move(rect.x+70, rect.y+40, {steps: 5});
        await page.mouse.up();
        assert.notEqual(await page.locator('#canvas svg').getAttribute('viewBox'), zoomed);

        const downloading = page.waitForEvent('download');
        await page.locator('#download').click();
        const download = await downloading;
        assert.equal(download.suggestedFilename(), 'mobilepixornext-overview.svg');
        const stream = await download.createReadStream();
        const chunks = [];
        for await (const chunk of stream) chunks.push(chunk);
        const svg = Buffer.concat(chunks).toString('utf8');
        assert.match(svg, /viewBox="0 0 /);
        assert.match(svg, /data-node="stage4"/);
        assert.doesNotMatch(svg, /<script/);

        const invalid = Buffer.from(JSON.stringify({schema_version: 7}));
        await page.locator('#import-file').setInputFiles({name: 'invalid.json', mimeType: 'application/json', buffer: invalid});
        await page.waitForFunction(() => !document.getElementById('error').hidden);
        assert.match(await page.locator('#error').innerText(), /schema_version/);
        assert.equal(await page.locator('.tensor-node').count(), 14);

        const valid = JSON.parse(fs.readFileSync(path.join(viewer, 'mobilepixornext.json'), 'utf8'));
        valid.views.overview.nodes[0].label = '<script>alert(1)</script>';
        await page.locator('#import-file').setInputFiles({name: 'valid.json', mimeType: 'application/json',
            buffer: Buffer.from(JSON.stringify(valid))});
        await page.waitForFunction(() => document.getElementById('error').hidden);
        assert.equal(await page.locator('#canvas script').count(), 0);
        assert.match(await page.locator('#canvas').innerText(), /<script>alert\(1\)<\/script>/);
        for (const id of Object.keys(valid.views)) {
            await page.locator('#view').selectOption(id);
            assert.equal(await page.locator('#error').isVisible(), false, id);
            assert.ok(await page.locator('.tensor-node').count(), id);
        }

        if (process.argv[3]) {
            const optional = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
            await page.locator('#import-file').setInputFiles({name: 'experiment.json', mimeType: 'application/json',
                buffer: Buffer.from(JSON.stringify(optional))});
            await page.waitForFunction(() => document.getElementById('view').value === 'overview');
            assert.match(await page.locator('#model-summary').innerText(), /hist14/);
            for (const style of ['3d', 'paper-composite', 'paper', 'reference']) {
                await page.locator('#presentation').selectOption(style);
                for (const id of Object.keys(optional.views)) {
                    await page.locator('#view').selectOption(id);
                    assert.equal(await page.locator('#error').isVisible(), false, `${style}: ${id}`);
                    assert.ok(await page.locator('.tensor-node').count(), id);
                }
            }
        }

        await page.goto(pathToFileURL(path.join(viewer, 'Backbone.html')).href);
        if (process.argv[2]) await page.screenshot({path: process.argv[2], fullPage: true});
        await page.setViewportSize({width: 540, height: 900});
        assert.ok(await page.locator('.tensor-node').count());
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        assert.deepEqual(errors, []);
        assert.deepEqual(external, []);
        console.log('Browser checks passed: offline load, detail navigation, styles, pan/zoom, full SVG download, JSON validation/escaping, all views, narrow layout.');
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });

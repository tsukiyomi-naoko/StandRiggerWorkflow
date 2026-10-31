// Load UI-format workflows into the real ComfyUI frontend (headless Chromium) and report problems.
// Usage: node check_ui_load.js <comfy_url> <shots_dir> wf1.json [wf2.json ...]
const { chromium } = require('playwright');
const fs = require('fs'); const path = require('path');
(async () => {
  const [url, shots, ...files] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1800, height: 1100 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  let failed = 0;
  for (const f of files) {
    errors.length = 0;
    const wf = JSON.parse(fs.readFileSync(f, 'utf8'));
    const r = await page.evaluate(async (wf) => {
      await window.app.loadGraphData(wf);
      await new Promise(r => setTimeout(r, 1500));
      const g = window.app.graph;
      const nodes = g._nodes || g.nodes;
      const missing = nodes.filter(n => n.has_errors || n.type === undefined || (n.constructor && n.constructor.name === 'LGraphNode' && !n.comfyClass && n.type !== 'MarkdownNote' && n.type !== 'Note')).map(n => n.type);
      const links = g.links instanceof Map ? g.links.size : Object.keys(g.links || {}).length;
      // widget check: every StandRig node's widgets must hold the serialized values
      const widgetMismatch = [];
      for (const n of nodes) {
        const src = wf.nodes.find(x => x.id === n.id);
        if (!src || !n.widgets) continue;
        const vals = n.widgets.map(w => w.value);
        const sv = src.widgets_values || [];
        if (JSON.stringify(vals.slice(0, sv.length)) !== JSON.stringify(sv)) widgetMismatch.push([n.type, sv, vals]);
      }
      return { nodes: nodes.length, links, groups: (g._groups || g.groups || []).length, missing, widgetMismatch: widgetMismatch.slice(0, 5) };
    }, wf);
    const bad = r.nodes !== wf.nodes.length || r.links !== wf.links.length || r.missing.length || r.widgetMismatch.length;
    if (bad) failed++;
    console.log(path.basename(f), JSON.stringify(r), errors.length ? 'console errors: ' + errors.slice(0, 5).join(' | ') : '');
    await page.evaluate(() => { window.app.canvas.ds.scale = 0.22; window.app.canvas.ds.offset = [40, 40]; window.app.canvas.setDirty(true, true); });
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(shots, path.basename(f, '.json') + '.png') });
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();

// Records B-roll from the project's own visualisations, frame-exact via Playwright's fake clock.
//   node capture.mjs lab   -> build/clip_lab/0000.jpg ...   (Live Lab, seizure onset -> peak)
//   node capture.mjs dev   -> build/clip_dev/0000.jpg ...   (3D ear-hook vs ear+back hologram)
import { createRequire } from "module"; import { execSync } from "child_process";
import fs from "fs"; import path from "path";
const require = createRequire(import.meta.url);
const { chromium } = require(execSync("npm root -g").toString().trim() + "/playwright");
const HERE = path.dirname(new URL(import.meta.url).pathname);
const VIZ = path.resolve(HERE, "../../epilepsy_sim/viz");
const which = process.argv[2], FPS = 30;
const cfg = {
  lab: { url: `file://${VIZ}/epilepsy_live.html?t=8.6`, frames: 150, skip: false },
  dev: { url: `file://${VIZ}/device_compare.html`, frames: 150, skip: true },
}[which];
const out = path.join(HERE, "build", "clip_" + which); fs.mkdirSync(out, { recursive: true });
const b = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
p.on("pageerror", e => console.log("ERR", e.message));
if (cfg.skip) {
  // three.js page: hand-drive requestAnimationFrame with exact timestamps (fake clocks stall WebGL screenshots)
  await p.addInitScript(() => {
    window.__rafs = []; window.requestAnimationFrame = cb => { window.__rafs.push(cb); return 1; };
    window.__step = now => { const q = window.__rafs; window.__rafs = []; q.forEach(cb => cb(now)); };
  });
  await p.goto(cfg.url, { waitUntil: "domcontentloaded" });
  await p.waitForTimeout(1500); await p.mouse.click(960, 540); await p.waitForTimeout(2500);
} else {
  await p.clock.install();
  await p.goto(cfg.url, { waitUntil: "domcontentloaded" });
  await p.clock.runFor(200);
}
const base = cfg.skip ? await p.evaluate(() => performance.now()) : 0;
const t0 = Date.now();
const done = fs.readdirSync(out).length;   // resumable: fast-forward past frames already on disk
for (let f = 0; f < cfg.frames; f++) {
  if (f < done) { if (!cfg.skip) await p.clock.runFor(1000 / FPS); continue; }
  if (cfg.skip) { await p.evaluate(n => { window.__step(n); window.__step(n + 0.01); }, base + 4000 + f * 1000 / FPS); }
  else await p.clock.runFor(1000 / FPS);
  await p.screenshot({ path: path.join(out, String(f).padStart(4, "0") + ".jpg"), type: "jpeg", quality: 92, timeout: 180000 });
  if (f % 30 === 0) console.log(which, f, ((Date.now() - t0) / 1000).toFixed(0) + "s");
}
await b.close();

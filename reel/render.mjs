// Step 2: render reel.html frame by frame with headless Chromium and pipe to ffmpeg.
//   node render.mjs                 -> build/frames.mp4 (silent video)
//   node render.mjs --stills 1,5,9  -> build/still_<t>.png for quick checks
import { createRequire } from "module";
import { execSync, spawn } from "child_process";
import fs from "fs";
import path from "path";
import http from "http";

const require = createRequire(import.meta.url);
const { chromium } = require(path.join(execSync("npm root -g").toString().trim(), "playwright"));
const HERE = path.dirname(new URL(import.meta.url).pathname);
const BUILD = path.join(HERE, "build");
const timeline = JSON.parse(fs.readFileSync(path.join(BUILD, "timeline.json")));
const FPS = timeline.fps;

// tiny static server so fonts load from http (file:// font loading is flaky)
const server = http.createServer((req, res) => {
  const p = path.join(HERE, decodeURIComponent(req.url.split("?")[0]));
  if (!p.startsWith(HERE) || !fs.existsSync(p)) { res.writeHead(404); return res.end(); }
  const type = { ".html": "text/html", ".ttf": "font/ttf", ".json": "application/json" }[path.extname(p)] || "application/octet-stream";
  res.writeHead(200, { "Content-Type": type }); fs.createReadStream(p).pipe(res);
}).listen(0);
const port = server.address().port;

const browser = await chromium.launch({ args: ["--disable-gpu-vsync"] });
const page = await browser.newPage({ viewport: { width: 400, height: 700 } });
page.on("console", m => console.log("[page]", m.text()));
page.on("pageerror", e => { console.error("[page error]", e); process.exit(1); });
await page.goto(`http://127.0.0.1:${port}/reel.html`);
const duration = await page.evaluate(tl => window.init(tl), timeline);

const grab = t => page.evaluate(t => { window.draw(t); return document.getElementById("c").toDataURL("image/jpeg", 0.93); }, t);

const stillsArg = process.argv.indexOf("--stills");
if (stillsArg > 0) {
  for (const t of process.argv[stillsArg + 1].split(",").map(Number)) {
    const png = await page.evaluate(t => { window.draw(t); return document.getElementById("c").toDataURL("image/png"); }, t);
    fs.writeFileSync(path.join(BUILD, `still_${t}.png`), Buffer.from(png.split(",")[1], "base64"));
  }
} else {
  const n = Math.ceil((duration + 0.6) * FPS);  // +0.6 s hold, matches mix.py
  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", String(FPS), "-i", "-",
    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", path.join(BUILD, "frames.mp4")], { stdio: ["pipe", "inherit", "inherit"] });
  const t0 = Date.now();
  for (let f = 0; f < n; f++) {
    const url = await grab(f / FPS);
    const ok = ff.stdin.write(Buffer.from(url.split(",")[1], "base64"));
    if (!ok) await new Promise(r => ff.stdin.once("drain", r));
    if (f % 150 === 0) console.log(`frame ${f}/${n}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on("close", r));
  console.log(`done: ${n} frames in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
await browser.close();
server.close();

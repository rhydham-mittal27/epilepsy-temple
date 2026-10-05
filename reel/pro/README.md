# Epilepsy Temple: Build Log reel (professional cut)

`epilepsy_temple_reel_pro.mp4`: 9:16, 1080×1920, 30 fps, -14 LUFS, about 62 s.

A founder-style mini-documentary: it opens on the result, walks through the problem, the spark, the twist, the lab, the data, the device and the verdict, is honest about false alarms, and ends on what's next. It uses real footage from the project's own Live Lab and 3D device comparison, editorial typography (Inter + JetBrains Mono), chapter markers, a progress bar, sentence-case karaoke captions, and a minimal cinematic score.

## Rebuild

```bash
python voice_pro.py          # Kokoro narration -> build/voice.wav + build/timeline.json
node capture.mjs lab         # B-roll: epilepsy_sim/viz/epilepsy_live.html (seizure onset -> peak)
node capture.mjs dev         # B-roll: epilepsy_sim/viz/device_compare.html (rotating hologram)
node render_pro.mjs          # reel_pro.html frame by frame -> build/frames.mp4
python mix_pro.py            # score + SFX + ducking + loudnorm -> epilepsy_temple_reel_pro.mp4
```

The Kokoro model files go in `../models/` (see `../README.md`). `node render_pro.mjs --stills 5,20,40` writes check frames to `build/`.

## Design choices (from short-form research)

| Practice | Applied |
|---|---|
| Lead with the result ("show the after first") and land the hook in the first 3 s | Opens on 96% vs 91% |
| Founder videos lead with a problem and show the real work | Problem, then the build, using real project footage instead of stock |
| Restrained typography, negative space, smooth motion (Apple-style) | Near-black canvas, one accent colour, masked word reveals, slow push-ins |
| Clean sentence-case captions read as more credible than meme captions | Karaoke fill, with the active word in the accent colour |
| A visual change every ~2 s | Every narration line adds a new element, footage crop or data beat |
| Safe zones | Content between y≈290 and 1290; captions at y≈1395 |
| Credibility | On-screen footnotes give dataset, cohort and method for every number |

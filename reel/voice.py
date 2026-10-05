"""Step 1: narrate the reel script with Kokoro TTS and write the timeline.

Outputs (in build/):
  voice.wav      the full narration
  timeline.json  scenes, lines, per-word timings, mouth envelope, SFX cues

Word timings are estimated: each line is synthesised on its own, leading and
trailing silence is trimmed, and the spoken span is split across words by
length (lines are short, so the error is a few frames).
"""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
MODEL_DIR = os.environ.get("KOKORO_DIR", os.path.join(HERE, "models"))
VOICE = os.environ.get("REEL_VOICE", "am_michael")
SPEED = float(os.environ.get("REEL_SPEED", "1.12"))
FPS = 30
SR = 24000

# (scene, spoken text, caption text or None for same, pause after in s, sfx at line start)
# Caption words wrapped in *stars* are highlighted.
SCRIPT = [
    ("hook", "Two tiny sensors behind your ear...", "*Two* tiny sensors behind your *ear*...", 0.15, "whoosh"),
    ("hook", "versus a hospital cap with eighteen.", "versus a hospital cap with *18*", 0.15, "pop"),
    ("hook", "Which one catches more seizures?", "Which one catches *more* *seizures?*", 0.35, "riser"),
    ("gadget", "This case started with a gadget.", "This *case* started with a *gadget*", 0.1, "whoosh"),
    ("gadget", "Zomato's founder wears this on his temple.", "Zomato's founder wears this on his *temple*", 0.1, "pop"),
    ("gadget", "Fifty million people live with epilepsy.", "*50* *million* people live with *epilepsy*", 0.1, "boom"),
    ("gadget", "Could something like this warn them?", "Could something like this *warn* them?", 0.3, "ding"),
    ("twist", "So I investigated.", "So I *investigated*", 0.1, "whoosh"),
    ("twist", "Plot twist! It measures blood flow. Not brain signals.", "*Plot* *twist!* It measures *blood* *flow*. Not brain signals", 0.1, "scratch"),
    ("twist", "Dead end.", "*Dead* *end.*", 0.35, "fail"),
    ("sim", "So I built a seizure. Inside a computer.", "So I *built* a *seizure*. Inside a computer", 0.1, "whoosh"),
    ("sim", "Potassium floods out.", "*Potassium* floods out", 0.05, "bubbles"),
    ("sim", "Neurons start firing together...", "Neurons start *firing* *together*...", 0.05, "zap"),
    ("sim", "then the brain goes silent.", "then the brain goes *silent*", 0.4, "down"),
    ("evidence", "Next: real evidence.", "Next: *real* *evidence*", 0.05, "whoosh"),
    ("evidence", "Hospital brainwaves from over a hundred and sixty patients.", "Hospital brainwaves from *160+* *patients*", 0.05, "tick"),
    ("evidence", "Over eleven hundred seizures.", "Over *1,100* *seizures*", 0.3, "tick"),
    ("suspect", "My suspect? A question-mark hook.", "My *suspect?* A *question-mark* *hook*", 0.05, "whoosh"),
    ("suspect", "Just two sensors, near the ears.", "Just *two* *sensors*, near the ears", 0.3, "pop"),
    ("verdict", "The verdict?", "The *verdict?*", 0.15, "drumroll"),
    ("verdict", "The hook caught ninety-six percent of seizures.", "The hook caught *96%* of seizures", 0.1, "tada"),
    ("verdict", "The full hospital cap? Ninety-one.", "The full hospital cap? *91%*", 0.35, "pop"),
    ("catch", "But every detective story has a catch.", "But every detective story has a *catch*", 0.05, "whoosh"),
    ("catch", "False alarms. Lots of them.", "*False* *alarms.* Lots of them", 0.1, "alarm"),
    ("catch", "The fix? Listen to the heart.", "The fix? Listen to the *heart*", 0.05, "heartbeat"),
    ("catch", "It cut false alarms by nearly half.", "It cut false alarms by *nearly* *half*", 0.3, "ding"),
    ("cta", "Case not closed.", "Case *not* *closed*", 0.1, "stamp"),
    ("cta", "Next, I build prototype one.", "Next, I build *Prototype* *1*", 0.05, "whoosh"),
    ("cta", "Follow, to watch it come alive.", "*Follow* to watch it come *alive*", 0.9, "pop"),
]

LEAD_IN = 0.25


def trim(samples, thresh=0.012):
    """Return (start, end) sample indices of the voiced span."""
    win = int(0.01 * SR)
    n = len(samples) // win
    rms = np.sqrt(np.mean(samples[: n * win].reshape(n, win) ** 2, axis=1))
    voiced = np.where(rms > thresh)[0]
    if len(voiced) == 0:
        return 0, len(samples)
    return max(0, voiced[0] * win - win), min(len(samples), (voiced[-1] + 2) * win)


def word_weight(w):
    core = re.sub(r"[^A-Za-z0-9%+,]", "", w)
    # numbers are spoken long ("96%" -> "ninety-six percent")
    if re.search(r"\d", core):
        return 4 + 2 * len(re.sub(r"\D", "", core)) + (3 if "%" in core else 0)
    return 1.5 + len(core)


def main():
    os.makedirs(BUILD, exist_ok=True)
    kokoro = Kokoro(os.path.join(MODEL_DIR, "kokoro-v1.0.onnx"), os.path.join(MODEL_DIR, "voices-v1.0.bin"))
    lang = "en-us" if VOICE.startswith("a") else "en-gb"

    audio = [np.zeros(int(LEAD_IN * SR), dtype=np.float32)]
    t = LEAD_IN
    lines, scenes = [], []
    for i, (scene, spoken, caption, pause, sfx) in enumerate(SCRIPT):
        samples, sr = kokoro.create(spoken, voice=VOICE, speed=SPEED, lang=lang)
        assert sr == SR
        a, b = trim(samples)
        clip = samples[a:b].astype(np.float32)
        dur = len(clip) / SR

        words = (caption or spoken).split()
        weights = np.array([word_weight(w) for w in words], dtype=float)
        edges = np.concatenate([[0], np.cumsum(weights) / weights.sum()]) * dur
        wlist = []
        for j, w in enumerate(words):
            hi = w.startswith("*")
            wlist.append({"w": w.replace("*", ""), "hi": hi, "t0": round(t + edges[j], 3), "t1": round(t + edges[j + 1], 3)})

        if not scenes or scenes[-1]["name"] != scene:
            scenes.append({"name": scene, "t0": round(t, 3)})
        lines.append({"i": i, "scene": scene, "t0": round(t, 3), "t1": round(t + dur, 3), "words": wlist, "sfx": sfx})
        audio.append(clip)
        audio.append(np.zeros(int(pause * SR), dtype=np.float32))
        t += dur + pause
        print(f"{t:6.2f}s  {scene:9s} {spoken}", file=sys.stderr)

    voice = np.concatenate(audio)
    total = len(voice) / SR
    for k, s in enumerate(scenes):
        s["t1"] = scenes[k + 1]["t0"] if k + 1 < len(scenes) else round(total, 3)

    # mouth envelope, one value per video frame, 0..1
    hop = SR // FPS
    nf = int(np.ceil(len(voice) / hop))
    padded = np.pad(voice, (0, nf * hop - len(voice)))
    env = np.sqrt(np.mean(padded.reshape(nf, hop) ** 2, axis=1))
    env = np.clip(env / (np.percentile(env[env > 0.01], 90) + 1e-9), 0, 1)

    sf.write(os.path.join(BUILD, "voice.wav"), voice, SR)
    with open(os.path.join(BUILD, "timeline.json"), "w") as f:
        json.dump({"fps": FPS, "duration": round(total, 3), "scenes": scenes, "lines": lines,
                   "mouth": [round(float(x), 2) for x in env]}, f)
    print(f"total {total:.2f}s, voice={VOICE}", file=sys.stderr)


if __name__ == "__main__":
    main()

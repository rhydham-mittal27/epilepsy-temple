"""Pro cut sound: cinematic minimal score + restrained SFX, ducked under the voice, muxed with the video."""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mix import SR, bandpass, norm, sweep, tvec  # noqa: E402

BUILD = os.path.join(HERE, "build")
rng = np.random.default_rng(11)


def delay(x, d_s, fb=.35, taps=4):
    d = int(d_s * SR); y = x.copy()
    for k in range(1, taps + 1):
        if d * k < len(x):
            y[d * k:] += x[:-d * k] * fb ** k
    return y


def reverb(x, t60=1.6):
    ir = rng.standard_normal(int(t60 * SR)) * np.exp(-tvec(t60) * 6.9 / t60)
    ir = bandpass(ir, 200, 6000)
    return np.fft.irfft(np.fft.rfft(x, len(x) + len(ir)) * np.fft.rfft(ir, len(x) + len(ir)))[:len(x)] * .02


# ---------------- SFX ----------------
def impact():
    d = 2.2; t = tvec(d)
    sub = np.sin(2 * np.pi * (38 + 50 * np.exp(-t * 18)) * t) * np.exp(-t * 2.2)
    hit = bandpass(rng.standard_normal(len(t)), 60, 2500) * np.exp(-t * 14)
    return norm(sub + .35 * hit + reverb(hit, 1.8) * 4, .9)
def whoosh():
    d = .6; t = tvec(d); n = rng.standard_normal(len(t)); out = np.zeros_like(n)
    for k in range(6):  # moving band
        m = (t >= k * d / 6) & (t < (k + 1) * d / 6); out[m] = bandpass(n, 300 + 500 * k, 1200 + 900 * k)[m]
    return norm(out * np.sin(np.pi * t / d) ** 2, .28)
def pulse():
    out = np.zeros(int(1.4 * SR))
    for at, g in ((0, 1), (.22, .65)):
        t = tvec(.3); b = np.sin(2 * np.pi * (48 + 25 * np.exp(-t * 30)) * t) * np.exp(-t * 14) * g
        s = int(at * SR); out[s:s + len(b)] += b
    return norm(out, .8)
def tick():
    t = tvec(.04); return norm(np.sin(2 * np.pi * 3200 * t) * np.exp(-t * 160), .22)
def riser():
    d = 1.6; t = tvec(d)
    x = bandpass(rng.standard_normal(len(t)), 500, 9000) * (t / d) ** 3 + .4 * sweep(110, 880, d) * (t / d) ** 2
    return norm(x, .3)
def glitch():
    d = .45; t = tvec(d); out = np.zeros_like(t)
    for k in range(9):
        s = int(rng.uniform(0, d - .05) * SR); L = int(rng.uniform(.015, .05) * SR)
        f = rng.choice([180, 440, 1200, 2600]); seg = np.sign(np.sin(2 * np.pi * f * np.arange(L) / SR))
        out[s:s + L] += seg[: len(out[s:s + L])]
    crushed = np.round(bandpass(rng.standard_normal(len(t)), 1000, 8000) * 4) / 4 * np.exp(-t * 6)
    return norm(out * .6 + crushed * .5, .38)
def count():
    out = np.zeros(int(1.4 * SR))
    for k in range(14):
        b = tick() * (.6 + .4 * k / 14); s = int(k * .085 * SR); out[s:s + len(b)] += b
    return norm(out, .3)

SFX = dict(impact=impact, whoosh=whoosh, pulse=pulse, tick=tick, riser=riser, glitch=glitch, count=count)


# ---------------- score: 100 bpm, A minor ----------------
CHORDS = [[220, 261.6, 329.6, 493.9], [174.6, 220, 261.6, 329.6], [130.8, 196, 261.6, 329.6], [196, 246.9, 293.7, 329.6]]
LEVELS = {  # pad, pulse, arp, hats, kick
    "hook": (1, 1, .6, 0, 0), "problem": (1, 1, 0, 0, 0), "spark": (1, .8, .7, 0, 0), "twist": (.7, 0, 0, 0, 0),
    "build": (1, 1, 1, .6, 0), "data": (1, 1, 1, .8, .8), "design": (1, 1, 1, .8, .8), "result": (1, 1, 1, 1, 1),
    "honest": (.9, 1, .6, .4, 0), "next": (1, 1, 1, 1, 1)}


def score(duration, scenes):
    beat = 60 / 100; bar = 4 * beat
    out = np.zeros(int((duration + 3) * SR)); arp_bus = np.zeros_like(out)
    def add(buf, x, at, g):
        s = int(at * SR); e = min(len(buf), s + len(x))
        if 0 <= s < len(buf) and g > 0: buf[s:e] += g * x[: e - s]
    def scene_at(t):
        cur = scenes[0]["name"]
        for s in scenes:
            if t >= s["t0"]: cur = s["name"]
        return cur
    kick = np.sin(2 * np.pi * (45 + 70 * np.exp(-tvec(.35) * 30)) * tvec(.35)) * np.exp(-tvec(.35) * 9)
    hat = bandpass(rng.standard_normal(int(.06 * SR)), 7000, 14000) * np.exp(-tvec(.06) * 70)
    for b in range(int(duration / bar) + 2):
        t0 = b * bar
        ch = CHORDS[(b // 2) % 4]
        for q in range(8):  # per eighth, pick levels at that moment (lets drops land mid-bar)
            at = t0 + q * beat / 2
            pad_l, pul, arp, hats, kk = LEVELS[scene_at(at)]
            if q % 2 == 0:
                tt = tvec(.4); add(out, np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 9), at, .35 * pul)
                if q in (0, 4): add(out, kick, at, .5 * kk)
            add(out, hat, at + (beat / 4 if q % 2 else 0), .05 * hats)
            for s16 in range(2):
                f = ch[(q * 2 + s16) % 4] * 2; dd = .35; tt = tvec(dd)
                note = (np.sin(2 * np.pi * f * tt) + .25 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 11)
                add(arp_bus, note, at + s16 * beat / 4, .055 * arp)
        pad_l = LEVELS[scene_at(t0)][0]
        tt = tvec(bar)
        pad = sum(np.sin(2 * np.pi * f / 2 * tt + np.sin(2 * np.pi * .2 * tt)) + .6 * np.sin(2 * np.pi * f / 2 * 1.004 * tt) for f in ch)
        add(out, bandpass(pad, 60, 1400) * np.minimum(1, np.minimum(tt / .5, (bar - tt) / .5)), t0, .03 * pad_l)
    out += delay(arp_bus, beat * .75, .4)
    return out[: int(duration * SR)]


def main():
    tl = json.load(open(os.path.join(BUILD, "timeline.json")))
    dur = tl["duration"] + 0.6
    n = int(dur * SR)
    voice, vsr = sf.read(os.path.join(BUILD, "voice.wav"))
    voice = np.interp(np.arange(int(len(voice) * SR / vsr)) * vsr / SR, np.arange(len(voice)), voice)
    voice = np.pad(voice, (0, max(0, n - len(voice))))[:n]

    mus = score(dur, tl["scenes"])
    talk = np.zeros(n)
    for l in tl["lines"]:
        talk[int(l["t0"] * SR):int(l["t1"] * SR)] = 1
    k = int(.2 * SR); talk = np.convolve(talk, np.ones(k) / k, mode="same")
    mus *= 1 - .45 * talk
    fl = int(1.2 * SR); mus[-fl:] *= np.linspace(1, 0, fl)

    fx = np.zeros(n)
    def put(name, at, g=1.0):
        x = SFX[name](); s = max(0, int(at * SR)); e = min(n, s + len(x)); fx[s:e] += g * x[: e - s]
    for l in tl["lines"]:
        cue = l["sfx"]
        if cue == "riser":   # build into the line's first highlighted word
            hi = next((w for w in l["words"] if w["hi"]), l["words"][-1])
            put("riser", hi["t0"] - 1.6, .9)
        elif cue == "whoosh": put("whoosh", l["t0"] - .3)
        else: put(cue, l["t0"])
    w96 = tl["lines"][15]["words"][8]["t0"]
    put("impact", w96 - .05, .9)                       # the 96% reveal
    put("count", tl["lines"][12]["t0"], .8)
    put("pulse", tl["lines"][18]["t0"] + .8, .7)

    mix = voice * 1.0 + mus * .5 + fx * .55
    mix = np.tanh(mix * 1.05) / np.tanh(1.05)
    mix = mix / np.max(np.abs(mix)) * .95
    sf.write(os.path.join(BUILD, "mix.wav"), mix.astype(np.float32), SR)
    out = os.path.join(HERE, "epilepsy_temple_reel_pro.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(BUILD, "frames.mp4"), "-i", os.path.join(BUILD, "mix.wav"),
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()

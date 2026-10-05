"""Step 3: synthesise music + cartoon SFX, mix under the voice, mux with the video.

All audio is generated here with numpy (no samples, no licences to worry about).
Outputs build/mix.wav and the final reel: epilepsy_temple_reel.mp4
"""
import json
import os
import subprocess

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
SR = 44100
rng = np.random.default_rng(4)


def tvec(d):
    return np.arange(int(d * SR)) / SR


def env(d, a=0.005, r=None):
    t = tvec(d)
    e = np.minimum(1, t / a)
    return e * (np.exp(-t / r) if r else np.clip((d - t) / max(d * 0.3, 1e-3), 0, 1))


def bandpass(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def sweep(f0, f1, d, wave=np.sin):
    t = tvec(d)
    f = np.geomspace(f0, f1, len(t))
    return wave(2 * np.pi * np.cumsum(f) / SR)


def saw(ph):
    return 2 * ((ph / (2 * np.pi)) % 1) - 1


def square(ph):
    return np.sign(np.sin(ph))


def norm(x, g=1.0):
    return g * x / (np.max(np.abs(x)) + 1e-9)


# ---------------- SFX ----------------
def sfx_whoosh():
    d = .45; n = rng.standard_normal(int(d * SR))
    t = tvec(d); e = np.sin(np.pi * t / d) ** 2
    return norm(bandpass(n, 400, 4000) * e, .5)
def sfx_pop():
    d = .12; return norm(sweep(900, 260, d) * env(d, .002, .04), .7)
def sfx_riser():
    d = .9; t = tvec(d)
    x = sweep(200, 1400, d, lambda p: saw(p)) * .4 + bandpass(rng.standard_normal(len(t)), 800, 6000) * .6
    return norm(x * (t / d) ** 2, .45)
def sfx_boom():
    d = .9; t = tvec(d)
    x = np.sin(2 * np.pi * (55 + 40 * np.exp(-t * 12)) * t) * np.exp(-t * 4) + .3 * bandpass(rng.standard_normal(len(t)), 40, 300) * np.exp(-t * 8)
    return norm(x, .9)
def sfx_ding():
    d = 1.0; t = tvec(d)
    return norm((np.sin(2 * np.pi * 1318 * t) + .5 * np.sin(2 * np.pi * 2637 * t) + .2 * np.sin(2 * np.pi * 3950 * t)) * np.exp(-t * 4), .45)
def sfx_scratch():
    d = .45; t = tvec(d)
    n = rng.standard_normal(len(t))
    out = np.zeros_like(n)
    for k, (a, b) in enumerate([(0, .15), (.15, .3), (.3, .45)]):
        m = (t >= a) & (t < b)
        out[m] = bandpass(n, 300 + 900 * (k % 2), 2500 + 2500 * (k % 2))[m]
    return norm(out * np.exp(-t * 2), .7)
def sfx_fail():  # sad trombone
    notes = [392, 370, 349, 311]; parts = []
    for i, f in enumerate(notes):
        d = .32 if i < 3 else .8; t = tvec(d)
        vib = 1 + (.02 * np.sin(2 * np.pi * 6 * t) if i == 3 else 0)
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        parts.append((saw(ph) * .6 + np.sin(ph) * .4) * env(d, .02, None))
    return norm(bandpass(np.concatenate(parts), 80, 2500), .55)
def sfx_bubbles():
    d = 1.0; out = np.zeros(int(d * SR))
    for k in range(9):
        s = int(rng.uniform(0, .8) * SR); b = sweep(rng.uniform(400, 700), rng.uniform(900, 1600), .07) * env(.07, .002, .02)
        out[s:s + len(b)] += b
    return norm(out, .45)
def sfx_zap():
    d = .7; t = tvec(d)
    x = square(2 * np.pi * (120 + 60 * np.sin(2 * np.pi * 30 * t)) * t) * .5 + bandpass(rng.standard_normal(len(t)), 2000, 8000) * .5
    return norm(bandpass(x, 80, 9000) * np.exp(-t * 3), .35)
def sfx_down():
    d = .9; return norm(sweep(700, 90, d) * env(d, .01, .5), .45)
def sfx_tick():
    out = np.zeros(int(.8 * SR))
    for k in range(4):
        b = sweep(2500, 1800, .03) * env(.03, .001, .008); s = int(k * .18 * SR); out[s:s + len(b)] += b
    return norm(out, .4)
def sfx_drumroll():
    d = 1.3; t = tvec(d); n = bandpass(rng.standard_normal(len(t)), 150, 6000)
    am = .55 + .45 * np.sin(2 * np.pi * 22 * t)
    return norm(n * am * (.2 + .8 * t / d), .5)
def sfx_tada():
    out = np.zeros(int(1.6 * SR))
    for k, f in enumerate([523, 659, 784, 1046]):
        d = 1.4 - k * .08; t = tvec(d); ph = 2 * np.pi * f * t
        b = (saw(ph) * .5 + np.sin(ph)) * env(d, .01, .45); s = int(k * .07 * SR); out[s:s + len(b)] += b
    return norm(bandpass(out, 100, 5000), .6)
def sfx_alarm():
    d = 1.3; t = tvec(d); f = np.where((t * 6).astype(int) % 2, 660, 880)
    return norm(square(2 * np.pi * np.cumsum(f) / SR) * env(d, .01, None), .22)
def sfx_heartbeat():
    out = np.zeros(int(1.8 * SR))
    for beat in (0, .8):
        for off, g in ((0, 1), (.2, .7)):
            d = .25; t = tvec(d); b = np.sin(2 * np.pi * (60 + 30 * np.exp(-t * 30)) * t) * np.exp(-t * 18) * g
            s = int((beat + off) * SR); out[s:s + len(b)] += b
    return norm(out, .95)
def sfx_stamp():
    d = .5; t = tvec(d)
    return norm(np.sin(2 * np.pi * 70 * t) * np.exp(-t * 14) + .5 * bandpass(rng.standard_normal(len(t)), 200, 3000) * np.exp(-t * 30), .9)

SFX = {k[4:]: v for k, v in globals().items() if k.startswith("sfx_")}


# ---------------- music: 100 bpm detective groove in A minor ----------------
def music(duration, scenes):
    bpm = 100; beat = 60 / bpm; bar = 4 * beat
    out = np.zeros(int((duration + 2) * SR))
    def add(x, at, g=1.0):
        s = int(at * SR); e = min(len(out), s + len(x))
        if s < len(out): out[s:e] += g * x[: e - s]
    kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-tvec(.3) * 35)) * tvec(.3)) * np.exp(-tvec(.3) * 10)
    hat = bandpass(rng.standard_normal(int(.05 * SR)), 6000, 12000) * np.exp(-tvec(.05) * 80)
    snap = bandpass(rng.standard_normal(int(.12 * SR)), 1200, 5000) * np.exp(-tvec(.12) * 40)
    bassline = [110, 110, 130.8, 146.8, 164.8, 146.8, 130.8, 98]          # A A C D E D C G
    chords = [[220, 261.6, 329.6], [196, 246.9, 293.7], [174.6, 220, 261.6], [164.8, 207.7, 246.9]]  # Am G F E
    energy = {"hook": 1, "gadget": .8, "twist": .7, "sim": 1, "evidence": .9, "suspect": .9, "verdict": 1, "catch": .9, "cta": 1}
    def scene_at(t):
        cur = scenes[0]
        for s in scenes:
            if t >= s["t0"]: cur = s
        return cur["name"]
    nbars = int(duration / bar) + 2
    for b in range(nbars):
        t0 = b * bar
        e = energy[scene_at(t0)]
        # stop-time drop for the twist's "dead end" and the verdict drumroll
        for q in range(4):
            add(kick, t0 + q * beat, .55 * e if q in (0, 2) else 0)
            add(snap, t0 + q * beat, .25 * e if q in (1, 3) else 0)
            for h in range(2):
                add(hat, t0 + q * beat + h * beat / 2, .08 * e)
        for i in range(8):  # pizzicato bass in eighths
            f = bassline[i] * (1 if b % 2 == 0 else (.89 if i > 3 else 1))
            d = beat / 2; ph = 2 * np.pi * f * tvec(d)
            add((np.sin(ph) + .3 * saw(ph)) * np.exp(-tvec(d) * 9), t0 + i * d, .32 * e)
        ch = chords[b % 4]; d = bar; t = tvec(d)
        pad = sum(np.sin(2 * np.pi * f * t) + .5 * np.sin(2 * np.pi * f * 1.003 * t) for f in ch)
        add(pad * np.sin(np.pi * t / d) ** .5, t0, .045 * e)
        # music-box sparkle on the off-beats
        for i in (1, 3):
            f = ch[i % 3] * 4; dd = .4
            add(np.sin(2 * np.pi * f * tvec(dd)) * np.exp(-tvec(dd) * 9), t0 + i * beat + beat / 2, .05 * e)
    return out[: int(duration * SR)]


def main():
    tl = json.load(open(os.path.join(BUILD, "timeline.json")))
    dur = tl["duration"] + 0.6
    voice, vsr = sf.read(os.path.join(BUILD, "voice.wav"))
    idx = np.arange(int(len(voice) * SR / vsr))
    voice = np.interp(idx * vsr / SR, np.arange(len(voice)), voice)
    n = int(dur * SR)
    voice = np.pad(voice, (0, max(0, n - len(voice))))[:n]

    mus = music(dur, tl["scenes"])
    # sidechain-style ducking: music dips while the narrator talks
    talk = np.zeros(n)
    for l in tl["lines"]:
        talk[int(l["t0"] * SR):int(l["t1"] * SR)] = 1
    k = int(.15 * SR)
    talk = np.convolve(talk, np.ones(k) / k, mode="same")
    duck = 1 - .55 * talk
    # silence the beat briefly for the two big stingers (dead end, verdict drumroll)
    for li in (9, 19):
        a, b = int((tl["lines"][li]["t0"] - .05) * SR), int((tl["lines"][li]["t1"] + .3) * SR)
        duck[a:b] *= .15
    fade = np.ones(n); fl = int(.8 * SR); fade[-fl:] = np.linspace(1, 0, fl)
    mus = mus * duck * fade

    fx = np.zeros(n)
    for l in tl["lines"]:
        x = SFX[l["sfx"]]()
        at = l["t0"] - (.18 if l["sfx"] in ("whoosh", "riser") else 0)
        s = max(0, int(at * SR)); e = min(n, s + len(x)); fx[s:e] += x[: e - s]
    # extra beats synced to the visuals
    extra = [(tl["lines"][20]["t0"] + 1.35, "tada"), (tl["lines"][17]["t0"] + .33, "stamp"),
             (tl["lines"][26]["t0"] + .4, "stamp"), (tl["lines"][1]["t0"] + .5, "boom")]
    for at, name in extra:
        x = SFX[name](); s = int(at * SR); e = min(n, s + len(x)); fx[s:e] += .8 * x[: e - s]
    for k2, l in enumerate(tl["lines"][25:26]):   # poofs as false alarms vanish
        for j in range(6):
            x = sfx_pop(); s = int((l["t0"] + .2 + j * .14) * SR); fx[s:s + len(x)] += .6 * x

    mix = 1.0 * voice + .55 * mus + .5 * fx
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)          # soft limiter
    mix = mix / np.max(np.abs(mix)) * .95
    sf.write(os.path.join(BUILD, "mix.wav"), mix.astype(np.float32), SR)

    out = os.path.join(HERE, "epilepsy_temple_reel.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(BUILD, "frames.mp4"), "-i", os.path.join(BUILD, "mix.wav"),
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()

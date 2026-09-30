"""Fourier analysis and noise reduction on single SeizeIT2 patients: can cleaning the ear EEG cut false alarms?

For each chosen patient, every run's behind-the-ear EEG (2 channels, 256 Hz) is streamed from OpenNeuro into
RAM (nothing written to disk), then:

1. Fourier analysis. Welch power spectra (0.5-100 Hz) of seizure windows, background windows, and the windows
   where a patient-specific detector raised FALSE alarms; plus spectrograms around a seizure and a false alarm.
   Shows which frequencies separate seizures from false alarms, and whether mains (50 Hz) or muscle (> 30 Hz)
   noise is present.
2. Noise reduction. The same 14 EEG features per channel are recomputed after each cleaning step:
     baseline      band-pass 0.5-45 Hz (the pipeline as used everywhere else)
     notch         + 50 Hz and 100 Hz notch (mains)
     lowpass30     band-pass 0.5-30 Hz (drops most muscle / high-frequency artefact)
     clip          robust artefact clipping: samples beyond 6 x the running robust SD (median absolute deviation
                   over 10 s) are clipped, so electrode pops and big movement spikes cannot dominate a window
     lowpass30+clip
   and a patient-specific detector (HistGradientBoosting, EEG only, runs split into 5 chronological folds,
   causal 10-min baseline normalisation) is scored with the usual event rules.
   Same patient, same folds, same model; only the cleaning differs.

Usage (from epilepsy_sim/):  uv run detect/fourier_denoise.py sub-103 sub-047   -> results_seizeit2/fourier_denoise.json + PNGs
"""
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, welch, spectrogram
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402
import detect as D  # noqa: E402  (fast weighted binning)

VARIANTS = ["baseline", "notch", "lowpass30", "clip", "lowpass30+clip", "baseline+spectral"]
if "--variants" in sys.argv:
    VARIANTS = sys.argv[sys.argv.index("--variants") + 1].split(",")
    del sys.argv[sys.argv.index("--variants"): sys.argv.index("--variants") + 2]


def spectral(x, fs):
    """Artefact-aware spectral features per channel and 4-s window, on the UNFILTERED signal (keeps the
    45-100 Hz band the pipeline normally cuts): muscle index (45-100 Hz share of 0.5-100 Hz power),
    spectral flatness (2-45 Hz, geometric / arithmetic mean: 1 = white/broadband burst, 0 = peaky rhythm),
    spectral entropy (2-45 Hz), 1/f slope (log-log fit 2-45 Hz) and peak frequency (1-30 Hz)."""
    w, s, nwin = S.windows(x.shape[1], fs)
    f = np.fft.rfftfreq(w, 1 / fs)
    hann = np.hanning(w)
    m_all, m_mus, m_fit = (f >= .5) & (f <= 100), (f >= 45) & (f <= 100) & ~((f > 48) & (f < 52)), (f >= 2) & (f <= 45)
    m_pk = (f >= 1) & (f <= 30)
    lf = np.log10(f[m_fit])
    out = np.empty((nwin, 5 * x.shape[0]), np.float32)
    for c in range(x.shape[0]):
        xc = x[c] - np.median(x[c])
        for a, b in S.chunked(nwin):
            seg = xc[np.arange(w)[None] + (np.arange(a, b) * s)[:, None]]
            P = np.abs(np.fft.rfft((seg - seg.mean(1, keepdims=True)) * hann, axis=1)) ** 2 + 1e-9
            pf = P[:, m_fit]
            pn = pf / pf.sum(1, keepdims=True)
            lp = np.log10(pf)
            slope = ((lf - lf.mean()) * (lp - lp.mean(1, keepdims=True))).sum(1) / ((lf - lf.mean()) ** 2).sum()
            out[a:b, 5 * c + 0] = P[:, m_mus].sum(1) / P[:, m_all].sum(1)
            out[a:b, 5 * c + 1] = np.exp(np.log(pf).mean(1)) / pf.mean(1)
            out[a:b, 5 * c + 2] = -(pn * np.log(pn)).sum(1) / np.log(pn.shape[1])
            out[a:b, 5 * c + 3] = slope
            out[a:b, 5 * c + 4] = f[m_pk][np.argmax(P[:, m_pk], 1)]
    return out


def stream_patient(sub):
    runs = sorted([r for r in json.loads((S.FEAT / "index.json").read_text()) if r["sub"] == sub and "eeg" in r["files"]], key=S.run_no)

    def one(r):
        sig = S.read_edf(S.get(f"{S.S3}/{r['files']['eeg']}"))
        ch = list(sig.values())[:2]
        n = min(len(c[0]) for c in ch)
        return r, np.stack([c[0][:n].astype(np.float64) for c in ch]), ch[0][1]

    with cf.ThreadPoolExecutor(4) as ex:
        out = list(ex.map(one, runs))
    print(f"{sub}: {len(out)} runs, {sum(x[1].shape[1] / x[2] for x in out) / 3600:.1f} h, "
          f"{sum(len(r['seizures']) for r in runs)} seizures, fs={out[0][2]:.0f} Hz", flush=True)
    return out


def clean(x, fs, how):
    x = x - np.median(x, axis=1, keepdims=True)
    if "notch" in how:
        for f0 in (50, 100):
            if f0 < fs / 2:
                b, a = iirnotch(f0, 30, fs)
                x = filtfilt(b, a, x, axis=1)
    if "lowpass30" in how:
        x = sosfiltfilt(butter(4, [0.5, 30], btype="band", fs=fs, output="sos"), x, axis=1)
    if "clip" in how:
        y = sosfiltfilt(butter(4, [0.5, 45], btype="band", fs=fs, output="sos"), x, axis=1)
        w = int(10 * fs) | 1
        for c in range(y.shape[0]):
            # running robust SD from the median absolute deviation, on a decimated copy for speed
            dec = max(1, int(fs // 8))
            mad = median_filter(np.abs(y[c, ::dec]), size=max(3, w // dec), mode="nearest") * 1.4826 + 1e-6
            lim = 6 * np.repeat(mad, dec)[: y.shape[1]]
            y[c] = np.clip(y[c], -lim, lim)
        x = y
    return x


def features(x, fs):
    n = x.shape[1]
    _, _, nwin = S.windows(n, fs)
    return np.hstack([S.eeg_features(x[c], fs, nwin) for c in range(2)])


def run_patient(sub):
    data = stream_patient(sub)
    res = {"sub": sub, "variants": {}}
    feats = {v: [] for v in VARIANTS}
    meta = []
    for r, x, fs in data:
        t = (np.arange(S.windows(x.shape[1], fs)[2]) * S.STEP + S.WIN).astype(float)
        y = np.zeros(len(t), np.int8)
        for a, b in r["seizures"]:
            y[(t - S.WIN / 2 >= a) & (t - S.WIN / 2 <= b)] = 1
        meta.append((r, t, y))
        for v in VARIANTS:
            if v.endswith("+spectral"):  # cleaned EEG features + artefact-aware spectral features of the raw signal
                F = np.hstack([features(clean(x, fs, v[: -len("+spectral")]), fs), spectral(x, fs)])
            else:
                F = features(clean(x, fs, v), fs)
            feats[v].append(S.normalise(F))
    k = 5
    fold = np.array_split(np.arange(len(meta)), k)  # chronological run blocks
    rng = np.random.default_rng(0)
    keep = [(y == 1) | (rng.random(len(y)) < 0.10) for _, _, y in meta]
    outs = {}
    for v in VARIANTS:
        recs = []
        for fi in range(k):
            te = set(fold[fi].tolist())
            tr = [i for i in range(len(meta)) if i not in te]
            Xtr = np.vstack([feats[v][i][keep[i]] for i in tr])
            ytr = np.concatenate([meta[i][2][keep[i]] for i in tr])
            if ytr.sum() == 0:
                continue
            m = D.clf().fit(Xtr, ytr)
            for i in sorted(te):
                r, t, _ = meta[i]
                recs.append((m.predict_proba(feats[v][i])[:, 1].astype(np.float32), t, [tuple(s) for s in r["seizures"]],
                             r["duration"] / 3600, i))
        outs[v] = recs
        curve = S.score_fast([q[:4] for q in recs], S.PT_GRID)
        at = next(iter(S.score_fast([q[:4] for q in recs], [0.5]).values()))
        s = {"fa_at_sens": {str(p): S.fa_at_sensitivity(curve, p / 100) for p in (50, 70, 90)},
             "sens_at_budget": {str(b): S.sens_at_fa(curve, b) for b in (1, 5, 20, 50)},
             "at_0.5": {"sensitivity": at["sensitivity"], "fa_per_24h": at["fa_per_24h"]}}
        res["variants"][v] = s
        fa = " / ".join("--" if q is None else f"{q:.1f}" for q in s["fa_at_sens"].values())
        se = " / ".join("--" if q is None else f"{q:.0%}" for q in s["sens_at_budget"].values())
        print(f"  {v:15s} FA/day @50/70/90% caught: {fa:>20s} | caught @<=1/5/20/50 FA/day: {se} | "
              f"@0.5: {at['sensitivity']:.0%}, {at['fa_per_24h']:.0f} FA/day", flush=True)
    if "baseline" in outs:
        fourier(sub, data, meta, outs["baseline"], res)
    return res


def fourier(sub, data, meta, recs, res):
    """Welch spectra of seizure / background / false-alarm windows + two spectrograms."""
    fs = data[0][2]
    seg = int(4 * fs)
    P = {"seizure": [], "background": [], "false alarm": []}
    fa_examples, sz_example = [], None
    rng = np.random.default_rng(1)
    by_i = {q[4]: q for q in recs}
    for i, (r, x, fsr) in enumerate(data):
        _, t, y = meta[i]
        xr = x - np.median(x, axis=1, keepdims=True)
        p = by_i[i][0] if i in by_i else np.zeros(len(t))
        ps = np.convolve(p, np.ones(3) / 3)[: len(p)]
        near = np.zeros(len(t), bool)
        for a, b in r["seizures"]:
            near |= (t >= a - 60) & (t <= b + 120)
        fa_w = np.flatnonzero((ps > 0.5) & ~near)
        sz_w = np.flatnonzero(y == 1)
        bg_w = np.flatnonzero(~near & (ps < 0.2))
        bg_w = rng.choice(bg_w, min(300, len(bg_w)), replace=False) if len(bg_w) else bg_w
        for name, ws in (("seizure", sz_w), ("false alarm", fa_w[:: max(1, len(fa_w) // 200)]), ("background", bg_w)):
            for w in ws:
                a0 = int((t[w] - S.WIN) * fsr)
                if a0 < 0 or a0 + seg > x.shape[1]:
                    continue
                f, pxx = welch(xr[:, a0:a0 + seg], fs=fsr, nperseg=int(2 * fsr), axis=1)
                P[name].append(pxx.mean(0))
        if sz_example is None and len(r["seizures"]):
            sz_example = (i, r["seizures"][0][0])
        if len(fa_w):
            fa_examples.append((i, t[fa_w[len(fa_w) // 2]]))
    spec = {k: np.median(np.array(v), 0) for k, v in P.items() if len(v)}
    f = np.fft.rfftfreq(int(2 * fs), 1 / fs)
    bands = {"delta 0.5-4": (0.5, 4), "theta 4-8": (4, 8), "alpha 8-13": (8, 13), "beta 13-30": (13, 30),
             "gamma 30-45": (30, 45), "muscle 45-100": (45, 100), "mains 49-51": (49, 51)}
    table = {}
    for k, s in spec.items():
        tot = s[(f >= .5) & (f <= 100)].sum()
        table[k] = {b: float(s[(f >= lo) & (f < hi)].sum() / tot) for b, (lo, hi) in bands.items()}
        table[k]["total power (uV^2)"] = float(tot)
    res["fourier"] = {"n_windows": {k: len(v) for k, v in P.items()}, "relative_band_power": table}
    print("  relative band power (median spectra):")
    for k, v in table.items():
        print("    %-12s " % k + "  ".join(f"{b.split()[0]} {v[b]:.1%}" for b in bands) + f"  total {v['total power (uV^2)']:.0f} uV^2")
    # figure
    fig = plt.figure(figsize=(13, 8.2), facecolor="white")
    ax = fig.add_subplot(2, 2, (1, 2))
    col = {"seizure": "#d62728", "false alarm": "#ff7f0e", "background": "#1f77b4"}
    for k, s in spec.items():
        ax.semilogy(f, s, label=f"{k} (n={len(P[k])} windows)", color=col[k], lw=1.8)
    for x0, lab in ((4, "δ|θ"), (8, "θ|α"), (13, "α|β"), (30, "β|γ"), (45, "pipeline cut 45 Hz"), (50, "mains 50 Hz")):
        ax.axvline(x0, color="#999", lw=.8, ls="--")
        ax.text(x0, ax.get_ylim()[1] if False else 1, lab, fontsize=7, rotation=90, va="bottom", ha="right", color="#666", transform=ax.get_xaxis_transform())
    ax.set_xlim(0.5, 100)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (uV²/Hz), median over windows")
    ax.set_title(f"{sub}: Fourier spectrum of seizure vs false-alarm vs background windows (behind-the-ear EEG)")
    ax.legend()
    for j, (lab, ex) in enumerate((("seizure onset", sz_example), ("a false alarm", fa_examples[len(fa_examples) // 2] if fa_examples else None))):
        a2 = fig.add_subplot(2, 2, 3 + j)
        if ex is None:
            a2.axis("off")
            continue
        i, tc = ex
        x = data[i][1] - np.median(data[i][1], axis=1, keepdims=True)
        fsr = data[i][2]
        a0, a1 = int(max(0, tc - 60) * fsr), int(min(x.shape[1] / fsr, tc + 90) * fsr)
        ff, tt, Sxx = spectrogram(x[0, a0:a1], fs=fsr, nperseg=int(2 * fsr), noverlap=int(1.5 * fsr))
        a2.pcolormesh(tt + a0 / fsr - tc, ff, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap="magma")
        a2.set_ylim(0, 60)
        a2.axvline(0, color="cyan", lw=1)
        a2.set_title(f"spectrogram around {lab} (t=0), channel 1")
        a2.set_xlabel("seconds")
        a2.set_ylabel("Hz")
    fig.tight_layout()
    out = S.RES / f"fourier_{sub}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  figure: {out}")


if __name__ == "__main__":
    S.keep_awake()
    subs = sys.argv[1:] or ["sub-103"]
    allres = {}
    path = S.RES / "fourier_denoise.json"
    if path.exists():
        allres = json.loads(path.read_text())
    for s in subs:
        print(f"\n=== {s}", flush=True)
        r = run_patient(s)
        if s in allres:  # merge variant results into an earlier run
            allres[s]["variants"].update(r["variants"])
            allres[s].setdefault("fourier", r.get("fourier"))
        else:
            allres[s] = r
        path.write_text(json.dumps(allres, indent=1, default=float))
    print(f"\nsaved {path}")

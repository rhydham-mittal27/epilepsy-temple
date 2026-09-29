"""Seizure detection performance on real scalp EEG (CHB-MIT): full cap vs temple-only electrodes.

Pipeline (all causal, i.e. usable in real time on a device):
  1. band-pass 0.5-45 Hz, 2 s windows every 1 s (a window is scored at its END)
  2. 14 features per channel: line length, variance, Hjorth mobility & complexity,
     log band power (delta/theta/alpha/beta/gamma), relative band power (x5)
  3. gradient-boosted trees (HistGradientBoosting, class-balanced)
  4. causal 5 s moving average of the probability -> alarm when it crosses a threshold,
     then 60 s refractory period (one alarm per event)

Evaluation:
  * patient-specific, leave-one-recording-out (each hour is tested by a model that never saw it)
  * cross-patient: train on the other patients, test on the held-out patient
  * event metrics: sensitivity (seizures detected), false alarms / 24 h, detection latency
    (alarm time - clinician-marked onset). An alarm up to 15 s before onset counts as a
    detection; alarms during the 60 s after a seizure are not counted as false alarms.

Usage:
    uv run detect/fetch_chbmit.py        # download data first
    uv run detect/detect.py              # features are cached in data/chbmit/features/
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier


def _weighted_edges(x, w, pct):
    """Vectorised twin of sklearn.utils.stats._weighted_percentile(x, w, pct, average=True) for a 1D
    float64 array with no zero weights (what the binner passes): same sort call, same cumulative
    weights, same "averaged_inverted_cdf" rule, but one searchsorted for all percentiles instead of
    a Python loop over them."""
    idx = np.argsort(x.reshape(-1, 1), axis=0)[:, 0]  # same (unstable) sort sklearn uses internally
    xs, cdf = x[idx], np.cumsum(w[idx])
    adj = pct / 100 * cdf[-1]
    zero = adj == 0
    adj[zero] = np.nextafter(adj[zero], adj[zero] + 1)
    last = len(x) - 1
    pi = np.clip(np.searchsorted(cdf, adj), 0, last)
    above = (cdf[pi] - adj) > np.finfo(np.float64).eps
    return np.where(above, xs[pi], (xs[pi] + xs[np.clip(pi + 1, 0, last)]) / 2)


def _speed_up_weighted_binning():
    """scikit-learn 1.9 bins each feature with sample weights (class_weight="balanced") by calling
    _weighted_percentile once PER BIN EDGE (254 calls x n_features, each re-sorting the column,
    through array-API wrappers). With fewer than 200k training rows (no subsampling, so the weighted
    path is taken) this dominated fit time: 4,500 rows x 252 features took ~55 s. Replace it with a
    vectorised equivalent (_weighted_edges); thresholds and model outputs are identical (checked
    bit-for-bit on random, tied, discrete and missing-value features)."""
    import sklearn.ensemble._hist_gradient_boosting.binning as B

    orig = B._find_binning_thresholds
    if getattr(orig, "_vectorised", False):
        return

    def fast(col_data, max_bins, sample_weight=None):
        if sample_weight is None:
            return orig(col_data, max_bins, None)
        missing = np.isnan(col_data)
        if missing.any():
            col_data, sample_weight = col_data[~missing], sample_weight[~missing]
        nz = sample_weight != 0
        col_data, sample_weight = col_data[nz], sample_weight[nz]
        order = np.argsort(col_data)
        col_data, sample_weight = col_data[order], sample_weight[order]
        distinct = np.unique(col_data).astype(B.X_DTYPE)
        if len(distinct) <= max_bins:  # few distinct values: the original (unweighted) midpoint rule applies
            return orig(col_data, max_bins, sample_weight)
        pct = np.linspace(0, 100, num=max_bins + 1)[1:-1]
        if col_data.dtype != np.float64 or sample_weight.dtype != np.float64:
            return orig(col_data, max_bins, sample_weight)  # only the float64 case is mirrored exactly
        thr = _weighted_edges(col_data, sample_weight, pct)
        thr = np.unique(thr) if np.unique(thr).shape[0] != thr.shape[0] else thr
        np.clip(thr, a_min=None, a_max=B.ALMOST_INF, out=thr)
        return thr

    fast._vectorised = True
    B._find_binning_thresholds = fast


_speed_up_weighted_binning()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "chbmit"
FEAT = DATA / "features"
RES = Path(__file__).resolve().parent / "results"
FS = 256
WIN, STEP = 2, 1
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

FULL18 = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4",
          "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]
EXTRA = ["T7-FT9", "FT10-T8", "FT9-FT10"]
ALL_CH = FULL18 + EXTRA
CONFIGS = {  # name: channels (FT9/FT10 sit just in front of the ears, i.e. the temple)
    "Full cap (18 ch)": FULL18,
    "Both temples (4 ch)": ["T7-FT9", "F7-T7", "FT10-T8", "F8-T8"],
    "Right temple (2 ch)": ["FT10-T8", "F8-T8"],
    "Left temple (2 ch)": ["T7-FT9", "F7-T7"],
    "Right temple (1 ch)": ["FT10-T8"],
}
N_F = 14
PRE, POST, REFRACT, SMOOTH = 15, 60, 60, 5


# ------------------------------------------------------------------ features
def resolve(names, wanted):
    out = []
    for w in wanted:
        hit = [i for i, n in enumerate(names) if n == w or n.startswith(w + "-")]
        out.append(hit[0])
    return out


def features(x):
    """x: (C, N) float -> (n_windows, C, 14) float32."""
    w, s = WIN * FS, STEP * FS
    nwin = (x.shape[1] - w) // s + 1
    idx = np.arange(w)[None, :] + (np.arange(nwin) * s)[:, None]
    freqs = np.fft.rfftfreq(w, 1 / FS)
    hann = np.hanning(w).astype(np.float32)
    out = np.empty((nwin, x.shape[0], N_F), np.float32)
    for c in range(x.shape[0]):
        seg = x[c][idx]
        d1 = np.diff(seg, axis=1)
        d2 = np.diff(d1, axis=1)
        v0, v1, v2 = seg.var(1) + 1e-12, d1.var(1) + 1e-12, d2.var(1) + 1e-12
        mob = np.sqrt(v1 / v0)
        comp = np.sqrt(v2 / v1) / mob
        P = np.abs(np.fft.rfft((seg - seg.mean(1, keepdims=True)) * hann, axis=1)) ** 2
        bp = np.stack([P[:, (freqs >= a) & (freqs < b)].sum(1) for a, b in BANDS], 1) + 1e-12
        rel = bp / bp.sum(1, keepdims=True)
        out[:, c, 0] = np.log(np.abs(d1).sum(1) / WIN + 1e-12)
        out[:, c, 1] = np.log(v0)
        out[:, c, 2] = mob
        out[:, c, 3] = comp
        out[:, c, 4:9] = np.log(bp)
        out[:, c, 9:14] = rel
    return out


def load_file(patient, fname, seizures):
    cache = FEAT / f"{patient}_{fname}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"], z["t"]
    import mne

    raw = mne.io.read_raw_edf(DATA / patient / fname, preload=True, verbose="error")
    idx = resolve(raw.ch_names, ALL_CH)
    x = raw.get_data(picks=idx) * 1e6  # uV
    sos = butter(4, [0.5, 45], btype="band", fs=FS, output="sos")
    x = sosfiltfilt(sos, x, axis=1).astype(np.float32)
    X = features(x)
    t_end = np.arange(X.shape[0]) * STEP + WIN  # window END time = when it can be scored
    centre = t_end - WIN / 2
    y = np.zeros(len(t_end), np.int8)
    for a, b in seizures:
        y[(centre >= a) & (centre <= b)] = 1
    FEAT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, t=t_end)
    return X, y, t_end


# ------------------------------------------------------------------ alarms & scoring
def alarms_from_prob(p, t, thr):
    ps = np.convolve(p, np.ones(SMOOTH) / SMOOTH)[: len(p)]  # causal moving average
    out, last = [], -1e9
    for k in np.flatnonzero(ps > thr):
        if t[k] - last >= REFRACT:
            out.append(t[k])
            last = t[k]
    return out


def score(records, thr):
    """records: list of (prob, t, seizures, hours). Returns sensitivity, FA/24h, latencies."""
    n_sz = det = fa = 0
    hours = 0.0
    lat = []
    for p, t, sz, h in records:
        al = alarms_from_prob(p, t, thr)
        hours += h
        used = set()
        for a, b in sz:
            n_sz += 1
            hits = [x for x in al if a - PRE <= x <= b]
            if hits:
                det += 1
                lat.append(hits[0] - a)
                used.update(hits)
        for x in al:
            if x in used or any(a - PRE <= x <= b + POST for a, b in sz):
                continue
            fa += 1
    return {"sensitivity": det / max(n_sz, 1), "detected": det, "seizures": n_sz,
            "fa_per_24h": fa / hours * 24, "false_alarms": fa, "hours": hours,
            "latency_median_s": float(np.median(lat)) if lat else None,
            "latency_iqr_s": [float(np.percentile(lat, 25)), float(np.percentile(lat, 75))] if lat else None,
            "latencies": [float(v) for v in lat]}


def clf():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                          class_weight="balanced", random_state=0)


# ------------------------------------------------------------------ experiments
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset of config indices to run (0-4); default all. Finished configs are skipped.")
    args = ap.parse_args()
    manifest = json.loads((DATA / "manifest.json").read_text())
    data = {}
    t0 = time.time()
    for pat, files in manifest.items():
        data[pat] = []
        for f, sz in files.items():
            X, y, t = load_file(pat, f, sz)
            data[pat].append((f, X, y, t, sz))
        print(f"[features] {pat}: {len(files)} files, {sum(len(s) for s in files.values())} seizures "
              f"({time.time() - t0:.0f} s)", flush=True)

    thresholds = np.round(np.arange(0.1, 0.96, 0.05), 2)
    results = {"patient_specific": {}, "cross_patient": {}}
    probs = {}
    RES.mkdir(exist_ok=True)
    names = list(CONFIGS)
    todo = names if args.configs is None else [names[int(i)] for i in args.configs]
    for cname in names:
        ck = RES / f"partial_{names.index(cname)}.npz"
        if cname not in todo and not ck.exists():
            continue
        if ck.exists():  # resume: results for this config were already computed
            z = np.load(ck, allow_pickle=True)
            results["patient_specific"][cname] = z["ps"].item()
            results["cross_patient"][cname] = z["cp"].item()
            probs.update(z["probs"].item())
            print(f"[{cname}] loaded checkpoint", flush=True)
            continue
        chans = CONFIGS[cname]
        cidx = [ALL_CH.index(c) for c in chans]

        def mat(X):
            return np.ascontiguousarray(X[:, cidx, :].reshape(len(X), -1), dtype=np.float32)

        # --- patient-specific, leave-one-recording-out
        recs = []
        for pat, items in data.items():
            for k, (f, X, y, t, sz) in enumerate(items):
                tr = [j for j in range(len(items)) if j != k]
                Xtr = np.vstack([mat(items[j][1]) for j in tr])
                ytr = np.concatenate([items[j][2] for j in tr])
                m = clf().fit(Xtr, ytr)
                del Xtr
                p = m.predict_proba(mat(X))[:, 1].astype(np.float32)
                recs.append((p, t, sz, len(t) / 3600))
                probs[(cname, pat, f)] = p
        results["patient_specific"][cname] = {f"{th:.2f}": score(recs, th) for th in thresholds}

        # --- cross-patient: train on other patients
        crecs = []
        for pat in data:
            Xtr = np.vstack([mat(it[1]) for q in data if q != pat for it in data[q]])
            ytr = np.concatenate([it[2] for q in data if q != pat for it in data[q]])
            m = clf().fit(Xtr, ytr)
            del Xtr
            for f, X, y, t, sz in data[pat]:
                crecs.append((m.predict_proba(mat(X))[:, 1], t, sz, len(t) / 3600))
        results["cross_patient"][cname] = {f"{th:.2f}": score(crecs, th) for th in thresholds}
        np.savez(ck, ps=np.array(results["patient_specific"][cname], dtype=object),
                 cp=np.array(results["cross_patient"][cname], dtype=object),
                 probs=np.array({k: v for k, v in probs.items() if k[0] == cname}, dtype=object))
        ps = results["patient_specific"][cname]["0.50"]
        cp = results["cross_patient"][cname]["0.50"]
        print(f"[{cname}] patient-specific @0.5: sens {ps['sensitivity']:.0%} "
              f"({ps['detected']}/{ps['seizures']}), FA/24h {ps['fa_per_24h']:.1f}, "
              f"latency {ps['latency_median_s']} s | cross-patient: sens {cp['sensitivity']:.0%}, "
              f"FA/24h {cp['fa_per_24h']:.1f}  ({time.time() - t0:.0f} s)", flush=True)

    if len(results["patient_specific"]) < len(CONFIGS):
        done = list(results["patient_specific"])
        print(f"partial run: {len(done)}/{len(CONFIGS)} configs done {done}; re-run to finish")
        return
    (RES / "metrics.json").write_text(json.dumps(results, indent=1))
    figure(results, data, probs)
    print(f"saved {RES / 'metrics.json'} and {RES / 'detection.png'}")


def figure(results, data, probs):
    names = list(CONFIGS)
    ps = [results["patient_specific"][n]["0.50"] for n in names]
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3)
    colors = ["k", "tab:blue", "tab:red", "tab:green", "tab:orange"]
    for i, (key, lab) in enumerate([("sensitivity", "Seizures detected (%)"),
                                    ("fa_per_24h", "False alarms per 24 h"),
                                    ("latency_median_s", "Median alert latency (s)")]):
        a = fig.add_subplot(gs[0, i])
        vals = [(r[key] or 0) * (100 if key == "sensitivity" else 1) for r in ps]
        a.bar(range(len(names)), vals, color=colors)
        a.set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=7)
        a.set_title(lab + "  (patient-specific, thr 0.5)")
    a = fig.add_subplot(gs[1, :2])
    for n, c in zip(names, colors):
        for mode, ls in (("patient_specific", "-"), ("cross_patient", ":")):
            r = results[mode][n]
            fa = [v["fa_per_24h"] for v in r.values()]
            se = [100 * v["sensitivity"] for v in r.values()]
            a.plot(fa, se, ls, color=c, marker="o", ms=3, label=f"{n} ({'patient' if ls == '-' else 'cross'})")
    a.set_xscale("symlog", linthresh=1)
    a.set(xlabel="false alarms per 24 h", ylabel="seizures detected (%)",
          title="Sensitivity vs false-alarm trade-off (threshold sweep; solid = patient-specific, dotted = cross-patient)")
    a.legend(fontsize=6, ncol=2)
    a = fig.add_subplot(gs[1, 2])
    for n, c in zip(names, colors):
        lat = results["patient_specific"][n]["0.50"]["latencies"]
        if lat:
            a.hist(lat, bins=np.arange(-15, 61, 5), histtype="step", color=c, lw=1.5, label=n)
    a.set(xlabel="alarm time - clinical onset (s)", ylabel="seizures", title="Latency distribution")
    a.legend(fontsize=6)
    # example seizure
    pat = next(iter(data))
    f, X, y, t, sz = next(it for it in data[pat] if it[4])
    a = fig.add_subplot(gs[2, :])
    a0, b0 = sz[0]
    m = (t > a0 - 300) & (t < b0 + 300)
    for n, c in zip(names, colors):
        p = probs[(n, pat, f)]
        ps_ = np.convolve(p, np.ones(SMOOTH) / SMOOTH)[: len(p)]
        a.plot(t[m], ps_[m], color=c, lw=1, label=n)
    a.axvspan(a0, b0, color="tab:red", alpha=0.15, label="clinician-marked seizure")
    a.axhline(0.5, color="grey", ls="--", lw=0.8)
    a.set(xlabel=f"time in {pat}/{f} (s)", ylabel="smoothed seizure probability",
          title="Example: detector output around a held-out seizure")
    a.legend(fontsize=7, ncol=3)
    fig.suptitle("Seizure detection on CHB-MIT scalp EEG: full cap vs temple-only electrodes", fontsize=14)
    fig.tight_layout()
    fig.savefig(RES / "detection.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()

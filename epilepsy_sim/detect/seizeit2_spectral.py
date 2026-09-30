"""The proper test of EEG cleaning + artefact-aware spectral features on ALL SeizeIT2 patients.

fetch  Stream every run's behind-the-ear EEG EDF from OpenNeuro into RAM (nothing written to disk) and store,
       per run, on the SAME 4-s / 2-s window grid as the main features (data/seizeit2/features_spec/<run>.npz):
         eeg_notch  28 EEG features after a 50 + 100 Hz notch (then the usual 0.5-45 Hz band-pass)
         eeg_lp30   28 EEG features after a 0.5-30 Hz band-pass
         spec       10 artefact-aware spectral features of the unfiltered signal (per channel: 45-100 Hz muscle
                    share, spectral flatness, spectral entropy, 1/f slope, peak frequency) - see fourier_denoise.py
run    Patient-wise 5-fold CV (same folds, same training windows per twin, same model and scoring as seizeit2.py run)
       for the ear-EEG-only and the best general setup, each with the original EEG features and with cleaned EEG
       features + spectral features. Results -> results_seizeit2/spectral_general.json.

Usage (from epilepsy_sim/):  uv run detect/seizeit2_spectral.py fetch --workers 3 ;  uv run detect/seizeit2_spectral.py run
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402
import detect as D  # noqa: E402,F401  (fast weighted binning)
from fourier_denoise import clean, spectral  # noqa: E402

OUT = S.FEAT.parent / "features_spec"


def process(r):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{r['run']}.npz"
    if out.exists():
        return r["run"], "cached"
    sig = S.read_edf(S.get(f"{S.S3}/{r['files']['eeg']}"))
    ch = list(sig.values())[:2]
    fs = ch[0][1]
    n_sec = min(len(c[0]) / c[1] for c in ch)
    nwin = S.windows(int(n_sec), 1)[2]  # the same window count as the main feature files
    n = min(len(c[0]) for c in ch)
    x = np.stack([c[0][:n].astype(np.float64) for c in ch])
    del sig, ch
    feats = lambda xx: np.hstack([S.eeg_features(xx[c], fs, nwin) for c in range(2)])
    res = {"eeg_notch": feats(clean(x, fs, "notch")).astype(np.float16),
           "eeg_lp30": feats(clean(x, fs, "lowpass30")).astype(np.float16),
           "spec": spectral(x, fs)[:nwin].astype(np.float32)}
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    tmp.replace(out)
    return r["run"], f"{nwin * S.STEP / 3600:.1f} h"


def fetch(workers=3):
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [r for r in json.loads((S.FEAT / "index.json").read_text()) if "eeg" in r["files"] and (S.FEAT / f"{r['run']}.npz").exists()]
    todo = [r for r in runs if not (OUT / f"{r['run']}.npz").exists()]
    print(f"{len(runs)} runs, {len(todo)} to stream", flush=True)
    failed = []
    with cf.ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(process, r): r for r in todo}
        for f in tqdm(cf.as_completed(futs), total=len(futs), desc="EEG spectral features", unit="run"):
            try:
                f.result()
            except Exception as e:
                failed.append(futs[f]["run"])
                tqdm.write(f"{futs[f]['run']}: FAILED ({type(e).__name__}: {e})")
    done = sum((OUT / f"{r['run']}.npz").exists() for r in runs)
    print(f"{done}/{len(runs)} runs" + (f"; {len(failed)} failed - re-run to retry" if failed else ""))
    S.notify("SeizeIT2 spectral fetch DONE", f"{done}/{len(runs)} runs")
    if failed:
        sys.exit(1)


SETUPS = {  # name: (groups, EEG variant or None = original, add spectral features, seed twin)
    "EEG only (ear)": (S.CONFIGS["EEG only (ear)"], None, False, "EEG only (ear)"),
    "EEG only, notch": (S.CONFIGS["EEG only (ear)"], "eeg_notch", False, "EEG only (ear)"),
    "EEG only, notch + spectral": (S.CONFIGS["EEG only (ear)"], "eeg_notch", True, "EEG only (ear)"),
    "EEG only, lowpass30 + spectral": (S.CONFIGS["EEG only (ear)"], "eeg_lp30", True, "EEG only (ear)"),
    "Best (EEG+ECG+ctx+HRx)": (S.CONFIGS["EEG + ECG + context + HR-excess"], None, False, "EEG + ECG + context + HR-excess"),
    "Best, notch + spectral": (S.CONFIGS["EEG + ECG + context + HR-excess"], "eeg_notch", True, "EEG + ECG + context + HR-excess"),
    "Best, lowpass30 + spectral": (S.CONFIGS["EEG + ECG + context + HR-excess"], "eeg_lp30", True, "EEG + ECG + context + HR-excess"),
}


def design(run, groups, variant, spec):
    X, y, t = S.load(run)
    if variant or spec:
        z = np.load(OUT / f"{run}.npz")
        m = min(len(X), len(z["spec"]))
        X, y, t = X[:m], y[:m], t[:m]
        if variant:
            X = X.copy()
            X[:, :len(S.EEG_NAMES)] = z[variant][:m].astype(np.float32)
        Z = S.design(X, groups)
        if spec:
            Z = np.hstack([Z, S.normalise(z["spec"][:m])])
        return Z, y, t
    return S.design(X, groups), y, t


def run(only=None, folds=5):
    runs = [r for r in json.loads((S.FEAT / "index.json").read_text())
            if (S.FEAT / f"{r['run']}.npz").exists() and (OUT / f"{r['run']}.npz").exists()]
    subs, fold = S.subject_folds(runs, folds)
    print(f"{len(runs)} runs, {len(subs)} patients, {sum(len(r['seizures']) for r in runs)} seizures", flush=True)
    path = S.RES / "spectral_general.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.time()
    for name, (groups, variant, spec, twin) in SETUPS.items():
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        rng = S.pool_rng(twin)  # twins train on exactly the same windows
        Xs, ys, fs_ = [], [], []
        for r in tqdm(runs, desc=f"{name}: loading", leave=False):
            Z, y, _ = design(r["run"], groups, variant, spec)
            keep = (y == 1) | (rng.random(len(y)) < S.NEG_FRAC)
            Xs.append(Z[keep])
            ys.append(y[keep])
            fs_.append(np.full(keep.sum(), fold[r["sub"]], np.int8))
        Xp, yp, fp = np.vstack(Xs), np.concatenate(ys), np.concatenate(fs_)
        del Xs
        recs = []
        for k in tqdm(range(folds), desc=f"{name}: folds", leave=False):
            m = D.clf().fit(Xp[fp != k], yp[fp != k])
            for r in runs:
                if fold[r["sub"]] == k:
                    Z, _, t = design(r["run"], groups, variant, spec)
                    recs.append((m.predict_proba(Z)[:, 1].astype(np.float32), t, [tuple(v) for v in r["seizures"]], r["duration"] / 3600))
        del Xp
        curve = S.score_fast(recs, S.PT_GRID)
        at = next(iter(S.score_fast(recs, [0.5]).values()))
        s = {"fa_at_sens": {str(p): S.fa_at_sensitivity(curve, p / 100) for p in (50, 60, 70, 80)},
             "sens_at_budget": {str(b): S.sens_at_fa(curve, b) for b in (1, 5, 20, 50, 200)},
             "at_0.5": {"sensitivity": at["sensitivity"], "fa_per_24h": at["fa_per_24h"]}}
        out[name] = s
        path.write_text(json.dumps(out, indent=1, default=float))
        fa = " / ".join("--" if q is None else f"{q:.1f}" for q in s["fa_at_sens"].values())
        se = " / ".join("--" if q is None else f"{q:.0%}" for q in s["sens_at_budget"].values())
        print(f"{name:34s} FA/day @50/60/70/80%: {fa:>28s} | caught @<=1/5/20/50/200: {se} | @0.5 {at['sensitivity']:.0%}, "
              f"{at['fa_per_24h']:.0f} FA/day  ({time.time() - t0:.0f} s)", flush=True)
    S.notify("SeizeIT2 spectral run DONE", "see spectral_general.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "run"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--configs", nargs="*")
    a = ap.parse_args()
    S.keep_awake()
    fetch(a.workers) if a.cmd == "fetch" else run(a.configs)

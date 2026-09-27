"""General (cross-patient) seizure detector on all CHB-MIT subjects, plus per-person fine-tuning.

Steps
  fetch  Stream every seizure recording + 3 seizure-free hours per subject from the PhysioNet
         AWS mirror: download one EDF, extract features, delete it (disk use stays ~1 GB).
  run    1. General model, leave-one-subject-out (LOSO): trained on N-1 subjects, tested on a
            subject it has never seen ("new user, day one").
         2. Ablation: raw features vs per-person causal baseline normalisation (each feature
            z-scored against an exponential moving average of that recording's past ~10 min).
         3. Fine-tuning: for subjects with >= 3 seizure recordings, add the user's first k
            seizure recordings (k = 1, 2) + 1 seizure-free hour, test on everything later.
            Compared with a patient-only model trained on the same k recordings.
         Channel sets: temple (F7-T7, T7-P7, F8-T8, T8-P8; present for every subject) and the
         full 18-channel cap.

chb01 and chb21 are the same child (recorded 1.5 years apart) and are kept in one group.

Usage:
    uv run detect/general.py fetch     # ~30-45 min, resumable
    uv run detect/general.py run       # ~20-30 min
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import (FULL18, FS, N_F, WIN, STEP, clf, features, resolve, score)  # noqa: E402
from fetch_chbmit import BASE, parse_summary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "chbmit"
FEAT = DATA / "features_general"
RES = Path(__file__).resolve().parent / "results_general"
PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]
SUBJECT = {p: ("chb01" if p == "chb21" else p) for p in PATIENTS}
CONFIGS = {
    "Temple (4 ch)": ["F7-T7", "T7-P7", "F8-T8", "T8-P8"],
    # "?"-shaped ear-hook proxy: T7/T8 sit just above the ear, P7/P8 above-behind it, i.e. where
    # the curve of an ear-hook rests. Closest standard 10-20 channels available in CHB-MIT.
    "Ear-hook both ears (2 ch)": ["T7-P7", "T8-P8"],
    "Ear-hook one ear (1 ch)": ["T8-P8"],
    "Full cap (18 ch)": FULL18,
}
NORMS = {"Temple (4 ch)": (False, True)}  # raw-vs-normalised ablation on one setup; others normalised
NEG_FRAC = 0.05  # fraction of non-seizure windows kept for training (all seizure windows kept)
EMA_N = 600  # baseline memory ~10 min of 1 s windows


# ------------------------------------------------------------------ fetch + features (streaming)
def fetch(n_free=3):
    FEAT.mkdir(parents=True, exist_ok=True)
    tmp = DATA / "tmp.edf"
    man_path = FEAT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    sos = butter(4, [0.5, 45], btype="band", fs=FS, output="sos")
    import mne

    for p in PATIENTS:
        summ = urllib.request.urlopen(f"{BASE}/{p}/{p}-summary.txt", timeout=120).read().decode()
        ann = parse_summary(summ)
        sz = [f for f, s in ann.items() if s]
        free = [f for f, s in ann.items() if not s]
        step = max(len(free) // n_free, 1)
        files = sz + free[::step][:n_free]
        manifest.setdefault(p, {})
        for f in files:
            cache = FEAT / f"{p}_{f}.npz"
            if cache.exists() or manifest[p].get(f) == "skipped":
                continue
            t0 = time.time()
            try:
                url = f"{BASE}/{p}/{urllib.parse.quote(f)}"  # some names contain "+"
                with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                raw = mne.io.read_raw_edf(tmp, preload=True, verbose="error")
                idx = resolve(raw.ch_names, FULL18)
                x = raw.get_data(picks=idx) * 1e6
                del raw
                x = sosfiltfilt(sos, x, axis=1).astype(np.float32)
                X = features(x)
                del x
                t_end = np.arange(X.shape[0]) * STEP + WIN
                centre = t_end - WIN / 2
                y = np.zeros(len(t_end), np.int8)
                for a, b in ann[f]:
                    y[(centre >= a) & (centre <= b)] = 1
                np.savez_compressed(cache, X=X.astype(np.float16), y=y, t=t_end.astype(np.int32))
                manifest[p][f] = ann[f]
                print(f"{p}/{f}: {len(t_end) / 3600:.1f} h, {len(ann[f])} seizures ({time.time() - t0:.0f} s)",
                      flush=True)
            except (IndexError, ValueError) as e:  # montage without the 18 standard channels
                manifest[p][f] = "skipped"
                print(f"{p}/{f}: skipped ({type(e).__name__}: channels missing)", flush=True)
            except (urllib.error.URLError, TimeoutError, OSError) as e:  # network: retry on next run
                print(f"{p}/{f}: download failed ({e}); will retry on re-run", flush=True)
            finally:
                tmp.unlink(missing_ok=True)
            man_path.write_text(json.dumps(manifest, indent=1))
    ok = {p: {f: s for f, s in m.items() if s != "skipped"} for p, m in manifest.items()}
    print(f"done: {sum(len(m) for m in ok.values())} recordings, "
          f"{sum(len(s) for m in ok.values() for s in m.values())} seizures")


# ------------------------------------------------------------------ helpers
def normalise(X):
    """Causal per-recording baseline: z-score vs EMA mean/var of PREVIOUS windows (~10 min)."""
    W = X.shape[0]
    x = X.reshape(W, -1).astype(np.float32)
    a = 1.0 / EMA_N
    n0 = min(60, W)  # initialise from the first minute (only look-ahead: 60 s at file start)
    m0, v0 = x[:n0].mean(0), x[:n0].var(0) + 1e-6
    m = lfilter([a], [1, a - 1], x, axis=0, zi=((1 - a) * m0)[None, :])[0]
    s2 = lfilter([a], [1, a - 1], x ** 2, axis=0, zi=((1 - a) * (v0 + m0 ** 2))[None, :])[0]
    m_prev = np.vstack([m0[None], m[:-1]])
    v_prev = np.maximum(np.vstack([(v0 + m0 ** 2)[None], s2[:-1]]) - m_prev ** 2, 1e-6)
    return ((x - m_prev) / np.sqrt(v_prev)).reshape(X.shape)


def load(p, f, cidx, norm):
    z = np.load(FEAT / f"{p}_{f}.npz")
    X = z["X"][:, cidx, :].astype(np.float32)
    if norm:
        X = normalise(X)
    return X.reshape(len(X), -1), z["y"], z["t"].astype(float)


def recordings():
    man = json.loads((FEAT / "manifest.json").read_text())
    recs = []
    for p, m in man.items():
        for f, s in sorted(m.items()):
            if s != "skipped":
                recs.append((SUBJECT[p], p, f, [tuple(x) for x in s]))
    return recs


# ------------------------------------------------------------------ experiments
def run():
    RES.mkdir(exist_ok=True)
    recs = recordings()
    subjects = sorted({r[0] for r in recs})
    print(f"{len(recs)} recordings, {len(subjects)} subjects, "
          f"{sum(len(r[3]) for r in recs)} seizures", flush=True)
    rng = np.random.default_rng(0)
    thresholds = np.round(np.arange(0.1, 0.96, 0.05), 2)
    out = {"general": {}, "finetune": {}, "per_subject": {}}
    t0 = time.time()

    for cname, chans in CONFIGS.items():
        cidx = [FULL18.index(c) for c in chans]
        for norm in NORMS.get(cname, (True,)):
            key = f"{cname} | {'normalised' if norm else 'raw'}"
            # training pool: all seizure windows + NEG_FRAC of the rest, tagged by subject
            Xs, ys, gs = [], [], []
            for subj, p, f, sz in recs:
                X, y, _ = load(p, f, cidx, norm)
                keep = (y == 1) | (rng.random(len(y)) < NEG_FRAC)
                Xs.append(X[keep])
                ys.append(y[keep])
                gs.append(np.full(keep.sum(), subjects.index(subj)))
            Xp, yp, gp = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
            del Xs, ys
            # leave-one-subject-out
            scored, per_subj = [], {}
            for si, subj in enumerate(subjects):
                m = clf().fit(Xp[gp != si], yp[gp != si])
                srec = []
                for s_, p, f, sz in recs:
                    if s_ != subj:
                        continue
                    X, y, t = load(p, f, cidx, norm)
                    srec.append((m.predict_proba(X)[:, 1], t, sz, len(t) / 3600))
                scored += srec
                per_subj[subj] = score(srec, 0.5)
            out["general"][key] = {f"{th:.2f}": score(scored, th) for th in thresholds}
            out["per_subject"][key] = {s: {k: v for k, v in d.items() if k != "latencies"}
                                       for s, d in per_subj.items()}
            r = out["general"][key]["0.50"]
            print(f"[general] {key}: sens {r['sensitivity']:.0%} ({r['detected']}/{r['seizures']}), "
                  f"FA/24h {r['fa_per_24h']:.1f}, latency {r['latency_median_s']} s  ({time.time() - t0:.0f} s)",
                  flush=True)
            if not norm:
                continue

            # ---------------- fine-tuning (normalised features only)
            ft = {k: {"general": [], "patient_only": [], "finetuned": []} for k in (1, 2)}
            for si, subj in enumerate(subjects):
                srecs = [r for r in recs if r[0] == subj]
                szr = [r for r in srecs if r[3]]
                fre = [r for r in srecs if not r[3]]
                if len(szr) < 3 or not fre:
                    continue
                general = clf().fit(Xp[gp != si], yp[gp != si])
                for k in (1, 2):
                    tune = szr[:k] + fre[:1]
                    test = szr[k:] + fre[1:]
                    Xt, yt = [], []
                    for _, p, f, _ in tune:
                        X, y, _ = load(p, f, cidx, norm)
                        Xt.append(X)
                        yt.append(y)
                    Xt, yt = np.vstack(Xt), np.concatenate(yt)
                    pat = clf().fit(Xt, yt)
                    w = np.r_[np.ones((gp != si).sum()), np.full(len(yt), 10.0)]  # upweight the user
                    fin = clf().fit(np.vstack([Xp[gp != si], Xt]), np.r_[yp[gp != si], yt], sample_weight=w)
                    for _, p, f, sz in test:
                        X, y, t = load(p, f, cidx, norm)
                        h = len(t) / 3600
                        for name, mdl in (("general", general), ("patient_only", pat), ("finetuned", fin)):
                            ft[k][name].append((mdl.predict_proba(X)[:, 1], t, sz, h))
            out["finetune"][key] = {
                str(k): {name: {f"{th:.2f}": score(v, th) for th in thresholds} for name, v in d.items()}
                for k, d in ft.items()}
            for k in (1, 2):
                line = f"[finetune k={k}] {key}:"
                for name in ("general", "patient_only", "finetuned"):
                    r = out["finetune"][key][str(k)][name]["0.50"]
                    line += f" {name} sens {r['sensitivity']:.0%} FA/24h {r['fa_per_24h']:.1f} |"
                print(line, f"({time.time() - t0:.0f} s)", flush=True)
            del Xp, yp, gp
        (RES / "metrics.json").write_text(json.dumps(out, indent=1))
    figure(out)
    print(f"saved {RES / 'metrics.json'} and {RES / 'general.png'}")


def figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    a = ax[0, 0]
    for (key, d), c in zip(out["general"].items(), ["tab:cyan", "tab:blue", "tab:purple", "tab:pink", "k"]):
        fa = [v["fa_per_24h"] for v in d.values()]
        se = [100 * v["sensitivity"] for v in d.values()]
        a.plot(fa, se, "o-", ms=3, color=c, label=key)
    a.set_xscale("symlog", linthresh=1)
    a.set(xlabel="false alarms per 24 h", ylabel="seizures detected (%)",
          title="General model, never-seen subject (leave-one-subject-out)")
    a.legend(fontsize=8)

    a = ax[0, 1]
    key = next(k for k in out["per_subject"] if k.startswith("Temple") and "normalised" in k)
    ps = out["per_subject"][key]
    subs = list(ps)
    a.bar(range(len(subs)), [100 * ps[s]["sensitivity"] for s in subs], color="tab:blue")
    a2 = a.twinx()
    a2.plot(range(len(subs)), [ps[s]["fa_per_24h"] for s in subs], "r.", ms=8)
    a2.set_ylabel("false alarms / 24 h", color="tab:red")
    a.set_xticks(range(len(subs)), subs, rotation=90, fontsize=7)
    a.set(ylabel="seizures detected (%)", title=f"Per subject at threshold 0.5 ({key})")

    ax[0, 2].set_axis_off()
    for i, (key, d) in enumerate(list(out["finetune"].items())[:3]):
        a = ax[1, i]
        for name, c in (("general", "grey"), ("patient_only", "tab:orange"), ("finetuned", "tab:green")):
            for k, ls in (("1", ":"), ("2", "-")):
                v = d[k][name]
                a.plot([x["fa_per_24h"] for x in v.values()], [100 * x["sensitivity"] for x in v.values()],
                       ls, color=c, marker="o", ms=3, label=f"{name}, {k} seizure rec.")
        a.set_xscale("symlog", linthresh=1)
        a.set(xlabel="false alarms per 24 h", ylabel="seizures detected (%)",
              title=f"Fine-tuning with the user's first k seizure recordings\n{key}")
        a.legend(fontsize=7)
    fig.suptitle("General seizure detector on all CHB-MIT subjects + per-person fine-tuning", fontsize=14)
    fig.tight_layout()
    fig.savefig(RES / "general.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "run"])
    a = ap.parse_args()
    fetch() if a.cmd == "fetch" else run()

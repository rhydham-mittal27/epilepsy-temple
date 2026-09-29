"""Two-stage detection: a second model VERIFIES each candidate alarm of the first-stage detector.

Stage 1 = the cached general model (EEG + ECG + context + HR-excess; outputs are out-of-fold: the
model never saw the patient). It raises CANDIDATES at a loose threshold. Stage 2 looks at each
candidate with event-level evidence, available D seconds after the trigger (causal):
  * probability shape: peak, mean, time above thresholds over [trigger - 20 s, trigger + D]
  * the patient's recent "noisiness": candidates in the previous hour, mean probability over 5 min
  * body/EEG change vs the 5 min before: heart rate, HR not explained by movement, activity,
    EEG power / line length / gamma and delta share, shoulder EMG, beat quality
and outputs P(real seizure). Accepted candidates become alarms (60 s refractory among ACCEPTED
alarms only, so a rejected false candidate never blocks a later true one).

Protocol: 71 patients of the k=2 split (cache from `seizeit2.py personal`). Stage 2 is trained with
patient-wise 5-fold CV on candidates from other patients (their tuning + later data) and scored on
each held-out patient's LATER data only, i.e. comparable with the one-stage tables.

Usage (from epilepsy_sim/):  uv run detect/stage2.py
"""
import json
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402
from detect import POST, PRE, REFRACT  # noqa: E402

CFG = "EEG + ECG + context + HR-excess"
THETA0 = (0.5, 0.7)  # stage-1 candidate thresholds (loose)
PERSONAL_W = 20.0  # weight of the user's own tuning-period candidates in the personal verifier
QUALITY = False  # + ear-EEG signal-quality features (set by --quality)
DELAYS = (0, 10)  # s after the trigger at which stage 2 decides
CAND_REFRACT = 10  # s between candidates
BUDGETS = (0.5, 1, 2, 5, 10, 20, 50)
RAW = ["hr_10s", "hr_slope_30s", "rmssd_30s", "beat_quality", "ch1_logvar", "ch2_logvar", "ch1_ll", "ch2_ll",
       "ch1_rel_g", "ch2_rel_g", "ch1_rel_d", "ch2_rel_d", "emg_logrms", "back_acc_logstd", "chest_acc_logstd"]


def candidates(p, t, thr):
    ps = np.convolve(p, np.ones(3) / 3)[: len(p)]
    cand, out, i = t[ps > thr].astype(float), [], 0
    while i < len(cand):
        out.append(cand[i])
        i = int(np.searchsorted(cand, cand[i] + CAND_REFRACT, side="left"))
    return np.array(out), ps


def event_features(ps, t, X, H, a, delay, hist):
    """Features of the candidate triggered at time a, using data up to a + delay only."""
    i = int(np.searchsorted(t, a))
    j = min(int(np.searchsorted(t, a + delay, side="right")) - 1, len(t) - 1)
    w = ps[max(0, i - 10): j + 1]
    pre = slice(max(0, i - 150), max(1, i - 5))  # 5 min before (excluding the last 10 s)
    f = [w.max(), w.mean(), (w > 0.8).mean(), (w > 0.95).mean(), ps[pre].mean(), ps[j],
         np.sum((hist > a - 3600) & (hist < a)), np.sum((hist > a - 600) & (hist < a))]
    for k in RAW:
        c = X[:, S.NAMES.index(k)]
        now = np.nanmean(c[max(0, j - 2): j + 1]) if np.isfinite(c[max(0, j - 2): j + 1]).any() else np.nan
        base = np.nanmedian(c[pre]) if np.isfinite(c[pre]).any() else np.nan
        f += [now, now - base]
    for q in range(H.shape[1]):
        c = H[:, q]
        seg = c[max(0, j - 2): j + 1]
        f.append(np.nanmean(seg) if np.isfinite(seg).any() else np.nan)
    if QUALITY:  # ear-EEG signal quality (unstable amplitude = the one trait linked to false alarms, §6.7)
        lv = X[:, [S.NAMES.index("ch1_logvar"), S.NAMES.index("ch2_logvar")]]
        long = slice(max(0, i - 900), max(1, i - 5))  # previous 30 min
        for ch in range(2):
            v5, v30 = lv[pre, ch], lv[long, ch]
            ok5, ok30 = np.isfinite(v5), np.isfinite(v30)
            f.append(np.std(v5[ok5]) if ok5.sum() > 5 else np.nan)  # amplitude instability, 5 min
            f.append(np.mean(v5[ok5] > np.median(v30[ok30]) + 3) if ok5.sum() > 5 and ok30.sum() > 5 else np.nan)  # artefact bursts
        imb_now = abs(np.nanmean(lv[max(0, j - 2): j + 1, 0]) - np.nanmean(lv[max(0, j - 2): j + 1, 1]))
        imb_base = np.nanmedian(np.abs(lv[long, 0] - lv[long, 1])) if np.isfinite(lv[long]).any() else np.nan
        f.append(imb_now - imb_base)  # channel balance shift (one electrode losing contact)
    return f


def build(data, by, model="general"):
    """All candidates (tune + test data) of every patient, with features and labels."""
    rows = []
    for s in tqdm(sorted(data), desc="candidates", unit="patient"):
        tune, test = S.split_patient(by[s], 2)
        for part, lst in (("tune", tune), ("test", test)):
            recs = data[s][part][model]
            k = 0
            for r, cut in lst:
                z = np.load(S.FEAT / f"{r['run']}.npz")
                tt = z["t"]
                if cut is None:
                    sel = np.ones(len(tt), bool)
                elif part == "tune":
                    sel = tt <= cut
                else:
                    sel = tt - S.WIN > cut
                if not sel.any():
                    continue
                p, t, sz, h = recs[k]
                k += 1
                X = z["X"].astype(np.float32)
                H = S.hr_vs_motion(X)[sel]
                X = X[sel]
                for th0 in THETA0:
                    cand, ps = candidates(p, t, th0)
                    for a in cand:
                        if any(x - PRE <= a <= y for x, y in sz):
                            lab = 1
                        elif any(x - PRE <= a <= y + POST for x, y in sz):
                            lab = -1  # just after a seizure: neither (ignored, as in the scoring)
                        else:
                            lab = 0
                        for d in DELAYS:
                            rows.append((f"{s}|{r['run']}", part, th0, d, a, lab, event_features(ps, t, X, H, a, d, cand)))
                if part == "test":  # remember the record for scoring
                    rows.append((f"{s}|{r['run']}", part, None, None, None, None, (t, sz, h)))
    return rows


def score(accepted, test_recs):
    """accepted: {recording id: accepted trigger times}; test_recs: {recording id: (t, seizures, hours)}.
    Scored per RECORDING (each recording's clock starts at 0), 60 s refractory among accepted alarms."""
    det = nsz = fa = 0
    hours = 0.0
    for rid, (t, sz, h) in test_recs.items():
        al, last = [], -1e9
        for a in sorted(accepted.get(rid, [])):
            if a - last >= REFRACT:
                al.append(a)
                last = a
        hours += h
        for x, y in sz:
            nsz += 1
            det += any(x - PRE <= a <= y for a in al)
        fa += sum(1 for a in al if not any(x - PRE <= a <= y + POST for x, y in sz))
    return det / max(nsz, 1), fa * 24 / max(hours, 1e-9), nsz


def main():
    idx = json.loads((S.FEAT / "index.json").read_text())
    by = {}
    for r in idx:
        by.setdefault(r["sub"], []).append(r)
    data = pickle.loads((S.PROBS / f"{S.slug(CFG)}_k2.pkl").read_bytes())
    resp = json.loads((S.RES / "responders.json").read_text())[CFG]["rise_bpm"]
    cache = S.PROBS / ("stage2_rows_quality.pkl" if QUALITY else "stage2_rows.pkl")
    if cache.exists():
        rows = pickle.loads(cache.read_bytes())
    else:
        rows = build(data, by)
        cache.write_bytes(pickle.dumps(rows))
    test_recs = {rid: f for rid, part, th0, d, a, lab, f in rows if th0 is None}
    pat = lambda rid: rid.split("|")[0]
    subs = sorted({pat(rid) for rid in test_recs})
    fold = {s: zlib.crc32(s.encode()) % 5 for s in subs}
    groups = {"all": subs, "responders": [s for s in subs if (resp.get(s) or 0) > 50],
              "non-responders": [s for s in subs if (resp.get(s) or 0) <= 50]}
    out = {}
    thr_grid = np.r_[np.linspace(0.01, 0.9, 60), 1 - np.logspace(-1, -3.5, 40)]
    for th0 in THETA0:
        for d in DELAYS:
            C = [(s, part, a, lab, f) for s, part, t0, dd, a, lab, f in rows if t0 == th0 and dd == d]
            Xc = np.array([f for *_, f in C], np.float32)
            y = np.array([lab for _, _, _, lab, _ in C])
            rids = np.array([s for s, *_ in C])
            ss = np.array([pat(r) for r in rids])
            parts = np.array([part for _, part, *_ in C])
            aa = np.array([a for _, _, a, _, _ in C])
            p2 = np.full(len(C), np.nan)
            mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                        class_weight="balanced", random_state=0)
            if not PERSONAL:  # general verifier: patient-wise 5-fold
                for k in range(5):
                    tr = np.array([fold[s] != k for s in ss]) & (y >= 0)
                    te = np.array([fold[s] == k for s in ss]) & (parts == "test")
                    p2[te] = mk().fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1]
            else:  # personal verifier: other patients' candidates + this patient's TUNING candidates (x PERSONAL_W)
                for s in tqdm(subs, desc=f"personal verifier {th0}/{d}s", unit="patient", leave=False):
                    own = ss == s
                    tr = ((~own) | (parts == "tune")) & (y >= 0)
                    w = np.where(own[tr], PERSONAL_W, 1.0)
                    te = own & (parts == "test")
                    if te.any():
                        p2[te] = mk().fit(Xc[tr], y[tr], sample_weight=w).predict_proba(Xc[te])[:, 1]
            key = f"theta0={th0} | delay={d}s" + (f" | personal w={PERSONAL_W:g}" if PERSONAL else "") + (" | +quality" if QUALITY else "")
            out[key] = {}
            base_n = int(((parts == "test") & (y == 0)).sum())
            print(f"\n=== stage 1 at {th0} (+{d} s to decide): {int((y == 1).sum())} true / {int((y == 0).sum())} false candidates "
                  f"({base_n} false on later data)")
            for g, gs in groups.items():
                gset = set(gs)
                curve = []
                for tau in thr_grid:
                    acc = {}
                    for rid, s, a, pv, part in zip(rids, ss, aa, p2, parts):
                        if part == "test" and s in gset and pv > tau:
                            acc.setdefault(rid, []).append(a)
                    sens, far, nsz = score(acc, {rid: v for rid, v in test_recs.items() if pat(rid) in gset})
                    curve.append((float(tau), sens, far))
                best = {}
                for B in BUDGETS:
                    ok = [c[1] for c in curve if c[2] <= B]
                    best[str(B)] = max(ok) if ok else None
                out[key][g] = {"curve": curve, "sens_at_budget": best, "patients": len(gs)}
                print(f"  {g:15s} ({len(gs):2d} pts): " + "  ".join(
                    f"<={B}/d: {best[str(B)]:.0%}" if best[str(B)] is not None else f"<={B}/d:  - " for B in BUDGETS))
    path = S.RES / "stage2.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old, indent=1))
    print(f"\nsaved {path}")


PERSONAL = False

if __name__ == "__main__":
    S.keep_awake()
    PERSONAL = "--personal" in sys.argv
    QUALITY = "--quality" in sys.argv
    if "--w" in sys.argv:
        PERSONAL_W = float(sys.argv[sys.argv.index("--w") + 1])
    if PERSONAL or QUALITY:
        THETA0, DELAYS = (0.5,), (10,)  # the best general variant
    main()

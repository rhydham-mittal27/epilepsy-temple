"""What KINDS of false alarms are there? (SeizeIT2)

The dataset does not say why a false alarm happened, so the false alarms are grouped by their own
signature: every candidate alarm of the best general model (stage 1 at 0.5, evidence up to 10 s
after the trigger; cached by stage2.py) carries ~40 measurements: detector-output shape, recent alarm
history, and for heart rate, HR slope, RMSSD, beat quality, ear-EEG power / line length / gamma and
delta share, shoulder EMG, back/chest movement the CURRENT value and the change vs the 5 min
before, plus heart rate not explained by movement.

Steps
  1. Standardise the change features robustly (median / IQR) and cluster the FALSE alarms
     (Gaussian mixture, k chosen by BIC among 3..8).
  2. Describe each cluster by its median profile next to the true seizures' profile, name it
     from the profile, and count how many patients it comes from.
  3. Separability: for each cluster, a patient-wise cross-validated classifier "this cluster vs
     true seizure alarms" -> AUC. High AUC = a targeted fix can remove this kind; AUC ~0.5 =
     physiologically indistinguishable from a seizure at this sensor set.

Usage (from epilepsy_sim/):  uv run detect/analysis_fa_types.py   -> results_seizeit2/fa_types.json
"""
import json
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402
import detect  # noqa: E402,F401  (fast weighted binning)
import stage2 as ST  # noqa: E402

SHAPE = ["peak", "mean", "frac>0.8", "frac>0.95", "pre5min_mean", "current", "cands_last_hour", "cands_last_10min"]
NAMES = SHAPE + [f"{k}{suf}" for k in ST.RAW for suf in ("", "_change")] + S.HRX_NAMES
# signature used for clustering: what CHANGED at the moment of the alarm (+ absolute heart rate)
SIG = ["hr_10s", "hr_10s_change", "hr_slope_30s", "rmssd_30s_change", "beat_quality_change",
       "ch1_logvar_change", "ch2_logvar_change", "ch1_ll_change", "ch2_ll_change",
       "ch1_rel_g_change", "ch2_rel_g_change", "ch1_rel_d_change", "ch2_rel_d_change",
       "emg_logrms_change", "back_acc_logstd_change", "chest_acc_logstd_change", "hr_excess", "cands_last_hour"]


def name_cluster(med):
    """A plain-language label from the median signature (z-scores relative to all false alarms)."""
    g = lambda k: med.get(k, 0.0)
    eeg_pow = (g("ch1_logvar_change") + g("ch2_logvar_change")) / 2
    gamma = (g("ch1_rel_g_change") + g("ch2_rel_g_change")) / 2
    delta = (g("ch1_rel_d_change") + g("ch2_rel_d_change")) / 2
    move = max(g("back_acc_logstd_change"), g("chest_acc_logstd_change"))
    hr = g("hr_10s_change")
    parts = []
    if move > 0.5:
        parts.append("movement")
    if hr > 0.5:
        parts.append("heart-rate rise" + (" beyond movement" if g("hr_excess") > 0.5 else ""))
    if gamma > 0.5 or g("emg_logrms_change") > 0.5:
        parts.append("muscle / high-frequency EEG")
    if eeg_pow > 0.5 and move <= 0.5 and gamma <= 0.5:
        parts.append("EEG power jump without body change (electrode / contact)")
    if delta > 0.5:
        parts.append("slow-wave (delta) increase")
    if g("cands_last_hour") > 0.5:
        parts.append("in a burst of repeated alarms")
    return ", ".join(parts) if parts else "small changes everywhere (subtle, seizure-like)"


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.mixture import GaussianMixture

    rows = pickle.loads((S.PROBS / "stage2_rows.pkl").read_bytes())
    C = [(rid, part, a, lab, f) for rid, part, th0, d, a, lab, f in rows if th0 == 0.5 and d == 10]
    F = np.array([f for *_, f in C], np.float64)
    y = np.array([c[3] for c in C])
    pat = np.array([c[0].split("|")[0] for c in C])
    assert F.shape[1] == len(NAMES), (F.shape, len(NAMES))
    fa, sz = y == 0, y == 1
    print(f"{fa.sum()} false and {sz.sum()} true candidate alarms (stage 1 at 0.5), {len(set(pat))} patients")

    cols = [NAMES.index(k) for k in SIG]
    Z = F[:, cols]
    med = np.nanmedian(Z[fa], 0)
    iqr = np.nanpercentile(Z[fa], 75, 0) - np.nanpercentile(Z[fa], 25, 0)
    Z = (Z - med) / np.where(iqr > 0, iqr, 1.0)
    Z = np.clip(np.where(np.isfinite(Z), Z, 0.0), -6, 6)  # missing sensor -> "no change"

    rng = np.random.default_rng(0)
    fa_idx = np.flatnonzero(fa)
    sub = rng.choice(fa_idx, min(20000, len(fa_idx)), replace=False)
    bics = {}
    for k in range(3, 9):
        gm = GaussianMixture(k, covariance_type="diag", random_state=0, n_init=3).fit(Z[sub])
        bics[k] = gm.bic(Z[sub])
    k = min(bics, key=bics.get)
    gm = GaussianMixture(k, covariance_type="diag", random_state=0, n_init=3).fit(Z[sub])
    lab = gm.predict(Z)
    print(f"clusters: k = {k} (BIC: " + ", ".join(f"{kk}:{v:.0f}" for kk, v in bics.items()) + ")\n")

    sz_prof = {s: float(np.median(Z[sz][:, i])) for i, s in enumerate(SIG)}
    out = {"n_false": int(fa.sum()), "n_true": int(sz.sum()), "k": k, "bic": bics, "true_seizure_profile": sz_prof,
           "clusters": []}
    fold = {p: zlib.crc32(p.encode()) % 5 for p in set(pat)}
    order = sorted(range(k), key=lambda c: -np.sum(lab[fa] == c))
    for c in order:
        m = fa & (lab == c)
        prof = {s: float(np.median(Z[m][:, i])) for i, s in enumerate(SIG)}
        pts = pat[m]
        counts = np.array(sorted([np.sum(pts == p) for p in set(pts)], reverse=True))
        top10 = counts[: max(1, len(set(pat)) // 10)].sum() / max(counts.sum(), 1)
        # separability from true seizure alarms (patient-wise 5-fold; all ~40 features)
        sel = m | sz
        Xs, ys, ps = F[sel], (y[sel] == 1).astype(int), pat[sel]
        pr = np.full(len(ys), np.nan)
        for f_ in range(5):
            te = np.array([fold[p] == f_ for p in ps])
            if ys[~te].min() == ys[~te].max() or not te.any():
                continue
            mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                 class_weight="balanced", random_state=0)
            pr[te] = mdl.fit(Xs[~te], ys[~te]).predict_proba(Xs[te])[:, 1]
        ok = np.isfinite(pr)
        auc = float(roc_auc_score(ys[ok], pr[ok])) if len(set(ys[ok])) == 2 else float("nan")
        name = name_cluster(prof)
        raw_hr = float(np.nanmedian(F[m][:, NAMES.index("hr_10s")]))
        info = {"cluster": int(c), "share_of_false_alarms": float(m.sum() / fa.sum()), "n": int(m.sum()),
                "name": name, "patients": int(len(set(pts))), "top10pct_patients_share": float(top10),
                "median_hr_bpm": raw_hr, "auc_vs_true_seizures": auc, "profile_z": prof}
        out["clusters"].append(info)
        print(f"[{info['share_of_false_alarms']:5.1%}] {name}")
        print(f"        n={m.sum()}, from {len(set(pts))} patients (top 10% of patients: {top10:.0%}); "
              f"median HR {raw_hr:.0f} bpm; separable from real seizures: AUC {auc:.2f}")
        key = sorted(prof.items(), key=lambda kv: -abs(kv[1]))[:5]
        print("        strongest signals (z vs typical false alarm): " + ", ".join(f"{k_} {v:+.1f}" for k_, v in key))
    key = sorted(sz_prof.items(), key=lambda kv: -abs(kv[1]))[:6]
    print("\nTRUE seizure alarms, strongest signals: " + ", ".join(f"{k_} {v:+.1f}" for k_, v in key))
    (S.RES / "fa_types.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {S.RES / 'fa_types.json'}")


def verify():
    """Type-aware alarm verifier: a multi-class model (real seizure + each false-alarm type) decides each
    candidate; an alarm is accepted when P(seizure) is high enough. Compared with the plain yes/no verifier
    on the same candidates, folds and scoring (general verifier, later data of held-out patients)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.mixture import GaussianMixture

    rows = pickle.loads((S.PROBS / "stage2_rows.pkl").read_bytes())
    test_recs = {rid: f for rid, part, th0, d, a, lab, f in rows if th0 is None}
    C = [(rid, part, a, lab, f) for rid, part, th0, d, a, lab, f in rows if th0 == 0.5 and d == 10]
    F = np.array([f for *_, f in C], np.float64)
    y = np.array([c[3] for c in C])
    rid = np.array([c[0] for c in C])
    pat = np.array([r.split("|")[0] for r in rid])
    parts = np.array([c[1] for c in C])
    aa = np.array([c[2] for c in C])
    fa = y == 0
    cols = [NAMES.index(k) for k in SIG]
    Z = F[:, cols]
    med = np.nanmedian(Z[fa], 0)
    iqr = np.nanpercentile(Z[fa], 75, 0) - np.nanpercentile(Z[fa], 25, 0)
    Z = np.clip(np.where(np.isfinite(Z := (Z - med) / np.where(iqr > 0, iqr, 1.0)), Z, 0.0), -6, 6)
    rng = np.random.default_rng(0)
    sub = rng.choice(np.flatnonzero(fa), min(20000, fa.sum()), replace=False)
    # false-alarm types are defined on the TRAINING folds only (no peeking at held-out patients' types)
    fold = {p: zlib.crc32(p.encode()) % 5 for p in set(pat)}
    fk = np.array([fold[p] for p in pat])
    mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                class_weight="balanced", random_state=0)
    p_bin, p_multi = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in range(5):
        tr, te = (fk != k) & (y >= 0), (fk == k) & (parts == "test")
        p_bin[te] = mk().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        trs = np.intersect1d(sub, np.flatnonzero(tr & fa))
        gm = GaussianMixture(8, covariance_type="diag", random_state=0, n_init=3).fit(Z[trs])
        cls = np.where(y == 1, 0, gm.predict(Z) + 1)  # 0 = seizure, 1..8 = false-alarm types
        m = mk().fit(F[tr], cls[tr])
        p_multi[te] = m.predict_proba(F[te])[:, list(m.classes_).index(0)]
    grid = np.r_[np.linspace(0.01, 0.9, 60), 1 - np.logspace(-1, -3.5, 40)]
    out = {}
    for name, pv in (("yes/no verifier", p_bin), ("type-aware verifier (seizure vs 8 false-alarm types)", p_multi)):
        curve = []
        for tau in grid:
            acc = {}
            for r_, a, v, part in zip(rid, aa, pv, parts):
                if part == "test" and v > tau:
                    acc.setdefault(r_, []).append(a)
            sens, far, _ = ST.score(acc, test_recs)
            curve.append((float(tau), sens, far))
        best = {str(B): max([c[1] for c in curve if c[2] <= B], default=None) for B in ST.BUDGETS}
        out[name] = best
        print(f"  {name:55s} " + "  ".join(f"<={B}/d: {v:.0%}" if v is not None else f"<={B}/d: -" for B, v in best.items()))
    old = json.loads((S.RES / "fa_types.json").read_text())
    old["type_aware_verifier"] = out
    (S.RES / "fa_types.json").write_text(json.dumps(old, indent=1, default=float))


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else main()

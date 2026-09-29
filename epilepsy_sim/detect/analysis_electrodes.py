"""Is electrode quality what limits detection? (SeizeIT2)

Hypothesis: the ear-EEG electrodes are poor, and that is why seizures are missed and false alarms
are high. Test: measure each patient's ear-EEG contact stability, and compare detection between
stable- and unstable-contact patients on their LATER data.

Contact stability = IQR over time of the ear-EEG log-variance (mean of both channels), over
non-seizure windows of the TUNING period only (a device knows this after the first days; no
peek at test data). Higher = amplitude jumps around = unstable contact.

For each group: seizures caught within false-alarm budgets, false alarms at matched sensitivity,
and the ceiling (share of later seizures whose smoothed output ever exceeds 0.5 / 0.95), using the
cached outputs of `seizeit2.py personal` (k = 2).

Usage (from epilepsy_sim/):  uv run detect/analysis_electrodes.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402

BUDGETS = (1, 5, 20, 50, 200)
SENS = (50, 60, 70)
MODELS = ["EEG only (ear)", "EEG + ECG + context + HR-excess"]
LV = [S.NAMES.index("ch1_logvar"), S.NAMES.index("ch2_logvar")]


def instability(s, by):
    """IQR of ear-EEG log-variance over the patient's non-seizure tuning windows."""
    tune, _ = S.split_patient(by[s], 2)
    vals = []
    for r, cut in tune:
        X, y, t = S.load(r["run"])
        sel = (y == 0) if cut is None else (y == 0) & (t <= cut)
        vals.append(X[sel][:, LV])
    v = np.vstack(vals)
    iq = [np.nanpercentile(v[:, c], 75) - np.nanpercentile(v[:, c], 25) for c in range(2)]
    return float(np.nanmean(iq))


def peaks(recs, smooth=3):
    """Peak causal 3-window-mean output inside each seizure (the detector's best look at it)."""
    out = []
    for p, t, szs, _ in recs:
        ps = np.convolve(p, np.ones(smooth) / smooth)[:len(p)]
        for a, b in szs:
            m = (t >= a) & (t <= b)
            if m.any():
                out.append(float(ps[m].max()))
    return np.array(out)


def main():
    idx = json.loads((S.FEAT / "index.json").read_text())
    by = {}
    for r in idx:
        by.setdefault(r["sub"], []).append(r)
    first = pickle.loads((S.PROBS / f"{S.slug(MODELS[0])}_k2.pkl").read_bytes())
    inst = {s: instability(s, by) for s in first}
    med = float(np.median(list(inst.values())))
    groups = {"stable contact": [s for s in inst if inst[s] <= med],
              "unstable contact": [s for s in inst if inst[s] > med],
              "all": list(inst)}
    print(f"{len(inst)} patients; contact instability (IQR of log-variance) median {med:.2f}, "
          f"range {min(inst.values()):.2f}-{max(inst.values()):.2f}")
    out = {"instability": inst, "median": med, "results": {}}
    for cname in MODELS:
        data = pickle.loads((S.PROBS / f"{S.slug(cname)}_k2.pkl").read_bytes())
        for model in ("general", "finetuned"):
            print(f"\n=== {cname} | {model}")
            for g, subs in groups.items():
                recs = [r for s in subs for r in data[s]["test"][model]]
                curve = S.score_fast(recs, S.PT_GRID)
                pk = peaks(recs)
                row = {"patients": len(subs), "seizures": int(len(pk)),
                       "sens_at_budget": {str(b): S.sens_at_fa(curve, b) for b in BUDGETS},
                       "fa_at_sens": {str(x): S.fa_at_sensitivity(curve, x / 100) for x in SENS},
                       "ceiling_gt_0.5": float((pk > 0.5).mean()), "ceiling_gt_0.95": float((pk > 0.95).mean())}
                out["results"].setdefault(cname, {}).setdefault(model, {})[g] = row
                se = "  ".join(f"<={b}/d:{'-' if v is None else f'{v:4.0%}'}" for b, v in row["sens_at_budget"].items())
                fa = "  ".join(f"{x}%:{'-' if v is None else f'{v:5.0f}'}" for x, v in row["fa_at_sens"].items())
                print(f"  {g:17s} ({row['patients']:2d} pts, {row['seizures']:3d} sz) caught {se} | FA/d at {fa} | "
                      f"seizures ever >0.5: {row['ceiling_gt_0.5']:.0%}, >0.95: {row['ceiling_gt_0.95']:.0%}")
            # per patient: does instability track how visible that patient's seizures are, and their false alarms?
            vis, fa5 = [], []
            for s in inst:
                pk = peaks(data[s]["test"][model])
                vis.append(np.median(pk) if len(pk) else np.nan)
                c = S.score_fast(data[s]["test"][model], [0.9])
                fa5.append(next(iter(c.values()))["fa_per_24h"])
            x = np.array([inst[s] for s in inst])
            ok = ~np.isnan(vis)
            r1 = spearmanr(x[ok], np.array(vis)[ok])
            r2 = spearmanr(x, fa5)
            out["results"][cname][model]["spearman"] = {"instability_vs_seizure_visibility": [float(r1.statistic), float(r1.pvalue)],
                                                        "instability_vs_fa_at_0.9": [float(r2.statistic), float(r2.pvalue)]}
            print(f"  per patient: instability vs median seizure peak rho={r1.statistic:+.2f} (p={r1.pvalue:.3f}); "
                  f"vs FA/day at threshold 0.9 rho={r2.statistic:+.2f} (p={r2.pvalue:.3g})")
    (S.RES / "electrodes.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {S.RES / 'electrodes.json'}")


if __name__ == "__main__":
    main()

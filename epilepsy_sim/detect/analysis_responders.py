"""Autonomic responders vs non-responders (SeizeIT2).

A patient is a "responder" if their heart rate rose by more than 50 bpm in at least one of their
FIRST TWO seizures (the tuning seizures; a device could know this). Rise = max 10-s heart rate from
onset to 60 s after the seizure end, minus the median over the 5 min before onset.
Performance is then scored separately for both groups on each patient's later data, using the
cached model outputs of `seizeit2.py personal` (k = 2).

Usage (from epilepsy_sim/):  uv run detect/analysis_responders.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402

RISE_BPM = 50
BUDGETS = (1, 5, 20, 50, 100, 200)


def hr_rise(run, a, b):
    z = np.load(S.FEAT / f"{run}.npz")
    X, t = z["X"], z["t"].astype(float)
    hr = X[:, S.NAMES.index("hr_10s")].astype(float)
    pre = hr[(t >= a - 300) & (t < a)]
    dur = hr[(t >= a) & (t <= b + 60)]
    if np.isnan(pre).all() or np.isnan(dur).all():
        return np.nan
    return float(np.nanmax(dur) - np.nanmedian(pre))


def main():
    idx = json.loads((S.FEAT / "index.json").read_text())
    by = {}
    for r in idx:
        by.setdefault(r["sub"], []).append(r)
    grid = S.PT_GRID
    out = {}
    for cname in [c for c in S.CONFIGS if (S.PROBS / f"{S.slug(c)}_k2.pkl").exists()]:
        data = pickle.loads((S.PROBS / f"{S.slug(cname)}_k2.pkl").read_bytes())
        resp = {}
        for s in data:
            rs = sorted(by[s], key=S.run_no)
            first = [(r["run"], a, b) for r in rs for a, b in r["seizures"]][:2]
            rises = [hr_rise(*f) for f in first]
            resp[s] = np.nanmax(rises) if not np.isnan(rises).all() else np.nan
        groups = {"responders": [s for s in data if resp[s] > RISE_BPM],
                  "non-responders": [s for s in data if resp[s] <= RISE_BPM],
                  "all": list(data)}
        out[cname] = {"rise_bpm": resp}
        print(f"\n=== {cname}: {len(groups['responders'])} responders, {len(groups['non-responders'])} non-responders, "
              f"{sum(np.isnan(v) for v in resp.values())} unknown (no ECG)")
        for model in ("general", "finetuned"):
            for g, subs in groups.items():
                recs = [r for s in subs for r in data[s]["test"][model]]
                curve = S.score_fast(recs, grid)
                cells = []
                for B in BUDGETS:
                    v = S.sens_at_fa(curve, B)
                    cells.append(f"{v:5.0%}" if v is not None else "    -")
                n_sz = next(iter(curve.values()))["seizures"]
                print(f"  {model:9s} {g:15s} ({len(subs):2d} pts, {n_sz:3d} sz) caught at <= " +
                      "  ".join(f"{B}/d:{c}" for B, c in zip(BUDGETS, cells)))
                out[cname].setdefault(model, {})[g] = {"patients": len(subs), "seizures": n_sz,
                                                      "sens_at_budget": {str(B): S.sens_at_fa(curve, B) for B in BUDGETS}}
    (S.RES / "responders.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {S.RES / 'responders.json'}")


if __name__ == "__main__":
    main()

"""Which seizures does the detector catch? Split by seizure type, vigilance, electrode side and lobe.

Uses the cached model outputs of `seizeit2.py personal` (k = 2 split, 71 patients, each patient's
LATER data) and the dataset's own annotations (eventType, vigilance, lateralization, localization
from every events.tsv; EEG montage side from each patient's eeg.json). Both metadata files are
downloaded once from OpenNeuro and cached in data/seizeit2/.

For every test seizure: its peak smoothed detector probability, and whether it is caught at the
SHARED thresholds that were calibrated in advance on tuning data for 5 and 20 false alarms/day
(honest operating points, see Multimodal_Detection_SeizeIT2.md §5.1).

Usage (from epilepsy_sim/):  uv run detect/analysis_seizure_types.py
"""
import concurrent.futures as cf
import json
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402
from detect import PRE, REFRACT  # noqa: E402

META = S.FEAT.parent


def fetch_json(url):
    return urllib.request.urlopen(url, timeout=60).read()


def events():
    path = META / "events_all.json"
    if path.exists():
        return json.loads(path.read_text())
    idx = json.loads((S.FEAT / "index.json").read_text())
    keys = [f"{S.DS}/{r['sub']}/ses-01/eeg/{r['run']}_events.tsv" for r in idx]
    with cf.ThreadPoolExecutor(24) as ex:
        out = dict(zip(keys, ex.map(lambda k: fetch_json(f"{S.S3}/{k}").decode(), keys)))
    path.write_text(json.dumps(out))
    return out


def montages():
    path = META / "subject_meta.json"
    if path.exists():
        return json.loads(path.read_text())
    idx = json.loads((S.FEAT / "index.json").read_text())
    first = {}
    for r in sorted(idx, key=lambda r: r["run"]):
        first.setdefault(r["sub"], r["run"])
    def one(s):
        j = json.loads(fetch_json(f"{S.S3}/{S.DS}/{s}/ses-01/eeg/{first[s]}_eeg.json"))
        return s, {"montage": j.get("EEGMontage")}
    with cf.ThreadPoolExecutor(24) as ex:
        out = dict(ex.map(one, first))
    path.write_text(json.dumps(out))
    return out


def seizure_type(ev):
    t = ev["eventType"]
    if "f2b" in t:
        return "focal to bilateral tonic-clonic"
    if "_m" in t:
        return "focal motor (automatisms, hyperkinetic, tonic, clonic)"
    if t.startswith("sz_foc_a_nm"):
        return "focal aware, non-motor"
    if t.startswith("sz_foc_ia_nm"):
        return "focal impaired awareness, non-motor"
    return "other / unclassified"


def main():
    ev = events()
    meta = montages()
    info = {}  # (run, onset) -> event dict
    for k, txt in ev.items():
        run = k.split("/")[-1][: -len("_events.tsv")]
        rows = [r.rstrip("\r").split("\t") for r in txt.strip().split("\n")]
        for r in rows[1:]:
            e = dict(zip(rows[0], r))
            if e["eventType"].startswith("sz"):
                info[(run, round(float(e["onset"]), 1))] = e
    idx = json.loads((S.FEAT / "index.json").read_text())
    by = {}
    for r in idx:
        by.setdefault(r["sub"], []).append(r)
    thr = json.loads((S.RES / "personal_threshold.json").read_text())
    out = {}
    for cname in [c for c in S.CONFIGS if (S.PROBS / f"{S.slug(c)}_k2.pkl").exists()]:
        key = f"{cname} | k=2"
        if key not in thr:
            continue
        th5 = thr[key]["general"]["shared"]["5"]["threshold_median"]
        th20 = thr[key]["general"]["shared"]["20"]["threshold_median"]
        data = pickle.loads((S.PROBS / f"{S.slug(cname)}_k2.pkl").read_bytes())
        rows = []
        for s, d in data.items():
            _, test = S.split_patient(by[s], 2)
            runs = []
            for r, cut in test:
                tt = np.load(S.FEAT / f"{r['run']}.npz")["t"]
                sel = np.ones(len(tt), bool) if cut is None else (tt - S.WIN > cut)
                if sel.any():
                    runs.append(r["run"])
            for (p, t, sz, h), run in zip(d["test"]["general"], runs):
                ps = np.convolve(p, np.ones(3) / 3)[: len(p)]
                al = {}
                for name, thv in (("5", th5), ("20", th20)):
                    cand, a_, i = t[ps > thv].astype(float), [], 0
                    while i < len(cand):
                        a_.append(cand[i])
                        i = int(np.searchsorted(cand, cand[i] + REFRACT, side="left"))
                    al[name] = np.array(a_)
                for a, b in sz:
                    e = info.get((run, round(a, 1)), {})
                    m = (t >= a - PRE) & (t <= b)
                    mont = (meta.get(s) or {}).get("montage")
                    lat = e.get("lateralization", "n/a")
                    side = ("electrodes on BOTH sides (generalized montage)" if mont == "generalized"
                            else "same side as onset" if lat in ("left", "right") and lat == mont
                            else "opposite side to onset" if lat in ("left", "right") and mont in ("left", "right")
                            else "onset side unknown / bilateral")
                    loc = e.get("localization", "n/a")
                    rows.append({
                        "type": seizure_type(e) if e else "other / unclassified",
                        "vigilance": {"awake": "awake", "asleep": "asleep"}.get(e.get("vigilance"), "unknown"),
                        "side": side,
                        "lobe": "temporal" if loc == "temp" else ("unknown" if loc in ("n/a", "", None) else "extratemporal"),
                        "duration": "< 30 s" if b - a < 30 else ("30-90 s" if b - a <= 90 else "> 90 s"),
                        "peak": float(ps[m].max()) if m.any() else 0.0,
                        "c5": bool(any(a - PRE <= x <= b for x in al["5"])),
                        "c20": bool(any(a - PRE <= x <= b for x in al["20"])),
                    })
        out[cname] = rows
        print(f"\n################ {cname} | general model | {len(rows)} later seizures | "
              f"shared thresholds: {th5:.3f} (5/day), {th20:.3f} (20/day)")
        for dim in ("type", "vigilance", "side", "lobe", "duration"):
            print(f"  by {dim}:")
            for g in sorted({r[dim] for r in rows}):
                v = [r for r in rows if r[dim] == g]
                print(f"    {g:52s} n={len(v):3d} | caught @5/day {np.mean([r['c5'] for r in v]):4.0%} "
                      f"@20/day {np.mean([r['c20'] for r in v]):4.0%} | never above 0.5: {np.mean([r['peak'] <= 0.5 for r in v]):4.0%}")
    (S.RES / "seizure_types.json").write_text(json.dumps(out, indent=0))
    print(f"\nsaved {S.RES / 'seizure_types.json'}")


if __name__ == "__main__":
    main()

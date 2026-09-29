"""Siena Scalp EEG (PhysioNet): does the ear-level vs full-cap result hold in ADULTS with focal
epilepsy, and does ECG help there too?

14 adults (20-71 y), 47 focal seizures (mostly temporal), 128 h of video-EEG at 512 Hz with a
full 10-20 cap (29 EEG channels) and EKG. The data are STREAMED from PhysioNet's AWS open-data
mirror into RAM (no EDF is written to disk); only features are stored (data/siena/features/).

Montages (bipolar, same as the CHB-MIT study):
  ear-hook, both ears  T3-T5, T4-T6   (= T7-P7, T8-P8: just above / behind each ear)
  ear-hook, one ear    T4-T6
  full cap             the 18-channel double banana
Heart: the EKG channel -> the same 8 ECG features as SeizeIT2.
Features: the SeizeIT2 pipeline (4 s windows every 2 s, 14 EEG features per channel, causal
10-min per-recording baseline normalisation). Model: HistGradientBoosting, leave-one-patient-out.

Usage (from epilepsy_sim/):
    uv run detect/siena.py index      # parse the seizure lists (checks all 47 seizures)
    uv run detect/siena.py fetch      # stream all 41 EDFs (~21 GB through RAM), features only
    uv run detect/siena.py run        # leave-one-patient-out, all montages; -> results_siena/
"""
import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402

BASE = "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0"
ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "siena" / "features"
RES = Path(__file__).resolve().parent / "results_siena"

# 18-channel double banana (modern names; Siena uses T3/T4/T5/T6 for T7/T8/P7/P8)
CAP = [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"),
       ("P3", "O1"), ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"), ("Fp2", "F8"), ("F8", "T4"),
       ("T4", "T6"), ("T6", "O2"), ("Fz", "Cz"), ("Cz", "Pz")]
NF = 14  # EEG features per channel
CONFIGS = {
    "Ear-hook, both ears (2 ch)": {"eeg": [("T3", "T5"), ("T4", "T6")]},
    "Ear-hook, both ears + ECG": {"eeg": [("T3", "T5"), ("T4", "T6")], "ecg": True},
    "Ear-hook, one ear (1 ch)": {"eeg": [("T4", "T6")]},
    "Full cap (18 ch)": {"eeg": CAP},
    "Full cap + ECG": {"eeg": CAP, "ecg": True},
}
NEG_FRAC = 0.2  # small dataset: keep 20% of non-seizure windows for training


def hms(s):
    h, m, x = (int(v) for v in re.findall(r"\d+", s)[:3])
    return h * 3600 + m * 60 + x


def keys():
    xml = S.get(f"https://physionet-open.s3.amazonaws.com/?list-type=2&prefix=siena-scalp-eeg/1.0.0/&max-keys=1000").decode()
    return {k.split("1.0.0/")[1]: int(s) for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", xml)}


def edf_header(path):
    """Start time and duration from the EDF header (HTTP range request, 256 bytes)."""
    import urllib.request
    req = urllib.request.Request(f"{BASE}/{path}", headers={"Range": "bytes=0-255"})
    h = urllib.request.urlopen(req, timeout=60).read()
    return hms(h[176:184].decode()), int(h[236:244]) * float(h[244:252])


def build_index():
    FEAT.mkdir(parents=True, exist_ok=True)
    path = FEAT / "index.json"
    if path.exists():
        return json.loads(path.read_text())
    k = keys()
    edfs = sorted(p for p in k if p.endswith(".edf"))
    lists = sorted(p for p in k if "Seizures-list" in p)
    heads = dict(zip(edfs, cf.ThreadPoolExecutor(16).map(edf_header, edfs)))
    runs = {p: {"sub": p.split("/")[0], "run": p.split("/")[1][:-4], "path": p, "bytes": k[p],
                "start": heads[p][0], "duration": heads[p][1], "seizures": []} for p in edfs}
    for lp in lists:
        txt = S.get(f"{BASE}/{lp}").decode("latin-1")
        sub = lp.split("/")[0]
        mine = [p for p in edfs if p.startswith(sub + "/")]
        fn, a = None, None
        # line by line: the file name may come once before several seizures (PN01), and a named
        # file may not exist as such (PN01 lists "PN01.edf"; the file is PN01-1.edf)
        for line in txt.splitlines():
            m = re.search(r"File name:\s*(\S+)", line)
            if m:
                fn = m.group(1).replace("PNO", "PN0")  # PN06's list has a letter O typo ("PNO6-1.edf")
                continue
            if re.search(r"registration", line, re.I):
                continue
            m = re.search(r"start time:\s*([\d.:]+)", line, re.I)
            if m:
                a = hms(m.group(1))
                continue
            m = re.search(r"end time:\s*([\d.:]+)", line, re.I)
            if m and a is not None:
                b = hms(m.group(1))
                key = f"{sub}/{fn}"
                if key not in runs:
                    assert len(mine) == 1, f"{lp}: unknown file {fn}"
                    key = mine[0]
                r = runs[key]
                on, off = (a - r["start"]) % 86400, (b - r["start"]) % 86400
                if off - on > 3600 - 1200:  # PN00-3 lists 18.28.29-19.29.29: an hour typo for a 1-min seizure
                    print(f"  note: {key} seizure {on:.0f}-{off:.0f} s looks like an hour typo; end -> {off - 3600:.0f} s")
                    off -= 3600
                assert on < r["duration"], f"{lp}: seizure at {on} s outside {key}"
                r["seizures"].append([float(on), float(min(off, r["duration"]))])
                a = None
    out = list(runs.values())
    path.write_text(json.dumps(out, indent=0))
    return out


def process(r):
    """Stream one EDF into RAM, decode only the needed channels, save features."""
    out = FEAT / f"{r['run']}.npz"
    if out.exists():
        return r["run"], "cached"
    buf = S.get(f"{BASE}/{r['path']}")
    h = buf[:256]
    nhdr, nrec, dur, ns = int(h[184:192]), int(h[236:244]), float(h[244:252]), int(h[252:256])

    def field(off, w):
        return [buf[256 + off * ns + i * w: 256 + off * ns + (i + 1) * w].decode("latin-1").strip() for i in range(ns)]

    lab = [lb.replace("EEG ", "").lower() for lb in field(0, 16)]
    pmin, pmax = np.array(field(104, 8), float), np.array(field(112, 8), float)
    dmin, dmax = np.array(field(120, 8), float), np.array(field(128, 8), float)
    nsamp = np.array(field(216, 8), int)
    rec = nsamp.sum()
    nrec = min(nrec, (len(buf) - nhdr) // (2 * rec))
    data = np.frombuffer(buf, "<i2", count=nrec * rec, offset=nhdr).reshape(nrec, rec)
    offs = np.r_[0, np.cumsum(nsamp)[:-1]]
    fs = float(nsamp[0] / dur)

    def sig(name):
        i = lab.index(name.lower())
        g = (pmax[i] - pmin[i]) / max(dmax[i] - dmin[i], 1)
        x = data[:, offs[i]:offs[i] + nsamp[i]].reshape(-1).astype(np.float32)
        return (x - dmin[i]) * g + pmin[i]

    n = nrec * nsamp[0]
    _, _, nwin = S.windows(n, fs)
    # decode channel pairs on demand (not all at once: a 13-h file x 20 channels would be ~2 GB of float32)
    feats = [S.eeg_features(sig(a) - sig(b), fs, nwin) for a, b in CAP]
    ecg = (S.ecg_features(sig("ekg ekg").astype(np.float64), fs, nwin) if "ekg ekg" in lab
           else np.full((nwin, len(S.ECG_NAMES)), np.nan, np.float32))
    del buf, data
    t = (np.arange(nwin) * S.STEP + S.WIN).astype(np.int32)  # window END time (s)
    ctr = t - S.WIN / 2
    y = np.zeros(nwin, np.int8)
    for on, off in r["seizures"]:
        y[(ctr >= on) & (ctr <= off)] = 1
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, eeg=np.hstack(feats), ecg=ecg, y=y, t=t)
    tmp.replace(out)
    return r["run"], f"{n / fs / 3600:.1f} h"


def fetch(workers=2):
    runs = build_index()
    todo = [r for r in runs if not (FEAT / f"{r['run']}.npz").exists()]
    print(f"{len(runs)} EDFs ({sum(r['bytes'] for r in runs) / 1e9:.1f} GB), {len(todo)} to stream", flush=True)
    with cf.ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(process, r): r for r in todo}
        for f in tqdm(cf.as_completed(futs), total=len(futs), desc="Siena EDFs", unit="file"):
            try:
                tqdm.write(f"{futs[f]['run']}: {f.result()[1]}")
            except Exception as e:
                tqdm.write(f"{futs[f]['run']}: FAILED ({type(e).__name__}: {e})")
    done = sum((FEAT / f"{r['run']}.npz").exists() for r in runs)
    S.notify("Siena streaming DONE", f"{done}/{len(runs)} EDFs")


def design(z, cfg):
    cols = [i * NF + j for i, pair in enumerate(CAP) if pair in cfg["eeg"] for j in range(NF)]
    Z = S.normalise(z["eeg"][:, cols])
    if cfg.get("ecg"):
        e = z["ecg"]
        Z = np.hstack([Z, S.normalise(e), e[:, [S.ECG_NAMES.index("hr_10s"), S.ECG_NAMES.index("hr_30s"),
                                               S.ECG_NAMES.index("beat_quality")]]])
    return Z


def run():
    runs = [r for r in build_index() if (FEAT / f"{r['run']}.npz").exists()]
    subs = sorted({r["sub"] for r in runs})
    nsz = sum(len(r["seizures"]) for r in runs)
    hours = sum(r["duration"] for r in runs) / 3600
    print(f"{len(runs)} files, {len(subs)} patients, {nsz} seizures, {hours:.0f} h; leave-one-patient-out", flush=True)
    RES.mkdir(exist_ok=True)
    data = {r["run"]: dict(np.load(FEAT / f"{r['run']}.npz")) for r in runs}
    out = {"dataset": {"files": len(runs), "patients": len(subs), "seizures": nsz, "hours": hours}, "results": {}}
    for cname, cfg in CONFIGS.items():
        rng = np.random.default_rng(0)
        D = {r["run"]: design(data[r["run"]], cfg) for r in runs}
        keep = {k: (data[k]["y"] == 1) | (rng.random(len(data[k]["y"])) < NEG_FRAC) for k in D}
        recs = []
        for s in tqdm(subs, desc=cname, unit="patient", leave=False):
            tr = [r["run"] for r in runs if r["sub"] != s]
            m = S.clf().fit(np.vstack([D[k][keep[k]] for k in tr]), np.concatenate([data[k]["y"][keep[k]] for k in tr]))
            for r in runs:
                if r["sub"] == s:
                    recs.append((m.predict_proba(D[r["run"]])[:, 1].astype(np.float32), data[r["run"]]["t"].astype(float),
                                 [tuple(v) for v in r["seizures"]], r["duration"] / 3600))
        curve = S.score_fast(recs, S.PT_GRID)
        at05 = S.score_fast(recs, [0.5], keep_lat=True)
        row = {"at_0.5": next(iter(at05.values())),
               "fa_at_sens": {str(x): S.fa_at_sensitivity(curve, x / 100) for x in (50, 60, 70, 80, 90)},
               "sens_at_budget": {str(b): S.sens_at_fa(curve, b) for b in (1, 5, 20, 50, 100, 200)}}
        out["results"][cname] = row
        a = row["at_0.5"]
        fa = " / ".join("--" if v is None else f"{v:.0f}" for v in row["fa_at_sens"].values())
        se = " / ".join("--" if v is None else f"{v:.0%}" for v in row["sens_at_budget"].values())
        print(f"{cname:28s} @0.5: {a['sensitivity']:.0%} caught, {a['fa_per_24h']:.0f} FA/day | "
              f"FA/day @50/60/70/80/90%: {fa} | caught @<=1/5/20/50/100/200: {se}", flush=True)
        (RES / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    S.notify("Siena results DONE", "see results_siena/metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["index", "fetch", "run"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    S.keep_awake()
    if a.cmd == "index":
        idx = build_index()
        for r in idx:
            print(f"{r['run']:14s} {r['duration'] / 3600:5.2f} h  seizures (s): {[[round(x) for x in s] for s in r['seizures']]}")
        print(f"{len(idx)} files, {sum(len(r['seizures']) for r in idx)} seizures, "
              f"{sum(r['duration'] for r in idx) / 3600:.1f} h, {sum(r['bytes'] for r in idx) / 1e9:.1f} GB")
    else:
        {"fetch": lambda: fetch(a.workers), "run": run}[a.cmd]()

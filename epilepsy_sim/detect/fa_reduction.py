"""Five ways to cut false alarms, tested on data (SeizeIT2 + CHB-MIT).

  chb      (2) Ear + back of the head on CHB-MIT: 23 children, 185 seizures, leave-one-subject-out,
               the stored 18-channel features re-combined into ear / ear+back / ear+back+parietal / cap.
  clock    Recording start times of every SeizeIT2 run from the EDF headers (HTTP range request,
           256 bytes each; cached). Needed for (4) and (5).
  target   (3) Alarm only for the seizures that matter. (a) Best general model, alarms scored only
               against target seizures (tonic-clonic; > 90 s; >= 30 s): other seizures are logged, not
               counted as misses, and alarms during them are not false. (b) A dedicated convulsive
               model trained with focal-to-bilateral tonic-clonic seizures as the only positives
               (EEG + ECG + EMG + motion + context + HR-excess), all 125 patients, patient-wise CV.
  night    (4) Night-alarm / day-diary mode: alarms only 22:00-07:00 (clock time), scored against
               night seizures; plus a device-side "lying still" mode from the accelerometer.
  prior    (5) Personal risk priors: seizure clusters (the hours after a confirmed seizure) and
               time of day (population onset-hour histogram from OTHER patients), added to the
               detector's log-odds.
  stack    (1) Combine the winners: fine-tuned detector (k = 1, so the stretch between the 1st and
               2nd seizure is honest held-out data) + personal alarm verifier (+ averaging with the
               general model), scored on exactly the same later data as the one-stage and
               two-stage tables.

Usage (from epilepsy_sim/):  uv run detect/fa_reduction.py <cmd>      -> detect/results_seizeit2/fa_reduction.json
"""
import argparse
import json
import pickle
import sys
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402
import detect as D  # noqa: E402  (also installs the fast weighted binning)

BEST = "EEG + ECG + context + HR-excess"
BUDGETS = (0.5, 1, 2, 5, 10, 20, 50)
SENS = (50, 70, 90, 100)
OUT = S.RES / "fa_reduction.json"


def save(key, val):
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old[key] = val
    OUT.write_text(json.dumps(old, indent=1, default=float))


def summary(curve):
    return {"fa_at_sens": {str(x): S.fa_at_sensitivity(curve, x / 100) for x in SENS},
            "sens_at_budget": {str(b): S.sens_at_fa(curve, b) for b in BUDGETS},
            "seizures": next(iter(curve.values()))["seizures"]}


def show(name, s):
    fa = " / ".join("--" if v is None else f"{v:.1f}" for v in s["fa_at_sens"].values())
    se = " / ".join("--" if v is None else f"{v:.0%}" for v in s["sens_at_budget"].values())
    print(f"  {name:58s} ({s['seizures']:3d} sz) FA/day @{'/'.join(map(str, SENS))}%: {fa} | "
          f"caught @<={'/'.join(map(str, BUDGETS))}/d: {se}", flush=True)


def by_patient():
    by = {}
    for r in json.loads((S.FEAT / "index.json").read_text()):
        by.setdefault(r["sub"], []).append(r)
    return by


def test_runs(rs, k=2):
    """(run dict, cut) of a patient's later data, in the same order as the cached output lists."""
    _, test = S.split_patient(rs, k)
    out = []
    for r, cut in test:
        tt = np.load(S.FEAT / f"{r['run']}.npz")["t"]
        sel = np.ones(len(tt), bool) if cut is None else (tt - S.WIN > cut)
        if sel.any():
            out.append((r, cut))
    return out


def sz_types(r):
    """(onset, end, eventType) of a run's seizures."""
    return [(a, b, t) for (a, b), t in zip(r["seizures"], r.get("types", ["sz"] * len(r["seizures"])))]


# ------------------------------------------------------------------ (2) CHB-MIT: ear + back of head
def chb():
    import general as G
    montages = {
        "Ear-hook, both ears (2 ch)": ["T7-P7", "T8-P8"],
        "Back of head only (2 ch)": ["P7-O1", "P8-O2"],
        "Ear + back of head (4 ch)": ["T7-P7", "T8-P8", "P7-O1", "P8-O2"],
        "Ear + back + parietal (6 ch)": ["T7-P7", "T8-P8", "P7-O1", "P8-O2", "P3-O1", "P4-O2"],
        "Full cap (18 ch)": D.FULL18,
    }
    recs = G.recordings()
    subjects = sorted({r[0] for r in recs})
    print(f"CHB-MIT: {len(recs)} recordings, {len(subjects)} subjects, {sum(len(r[3]) for r in recs)} seizures; "
          f"leave-one-subject-out", flush=True)
    res = {}
    for name, chans in montages.items():
        cidx = [D.FULL18.index(c) for c in chans]
        rng = np.random.default_rng(zlib.crc32(b"ear+back"))  # same training windows for every montage
        cache = {(p, f): G.load(p, f, cidx, True) for _, p, f, _ in tqdm(recs, desc=f"{name}: loading", leave=False)}
        keep = {k: (v[1] == 1) | (rng.random(len(v[1])) < G.NEG_FRAC) for k, v in cache.items()}
        scored = []
        for si, subj in enumerate(tqdm(subjects, desc=name, unit="subject", leave=False)):
            tr = [(p, f) for s_, p, f, _ in recs if s_ != subj]
            m = D.clf().fit(np.vstack([cache[k][0][keep[k]] for k in tr]),
                            np.concatenate([cache[k][1][keep[k]] for k in tr]))
            for s_, p, f, sz in recs:
                if s_ == subj:
                    X, _, t = cache[(p, f)]
                    scored.append((m.predict_proba(X)[:, 1].astype(np.float32), t, sz, len(t) / 3600))
        curve = S.score_fast(scored, S.PT_GRID, smooth=D.SMOOTH)  # 1-s windows: 5-s smoothing as in the CHB-MIT study
        at5 = next(iter(S.score_fast(scored, [0.5], smooth=D.SMOOTH).values()))
        res[name] = {**summary(curve), "at_0.5": {k: at5[k] for k in ("sensitivity", "fa_per_24h", "latency_median_s")},
                     "szcore_best_f1": best_f1(curve)}
        show(name, res[name])
        del cache
    save("chb_ear_back", res)


def best_f1(curve):
    best = None
    for t, v in curve.items():
        tp, fp, n = v["detected"], v["false_alarms"], v["seizures"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        sens = tp / n if n else 0.0
        f1 = 2 * sens * prec / (sens + prec) if sens + prec else 0.0
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(t), "sensitivity": sens, "precision": prec, "f1": f1, "fp_per_24h": v["fa_per_24h"]}
    return best


# ------------------------------------------------------------------ clock times
def clock():
    path = S.FEAT.parent / "starttimes.json"
    if path.exists():
        return json.loads(path.read_text())
    idx = json.loads((S.FEAT / "index.json").read_text())
    part = path.with_suffix(".partial.json")
    got = json.loads(part.read_text()) if part.exists() else {}

    def one(r):
        import time
        for i in range(5):  # the server occasionally resets a connection (WinError 10054): retry with back-off
            try:
                req = urllib.request.Request(f"{S.S3}/{r['files']['eeg']}", headers={"Range": "bytes=0-255"})
                h = urllib.request.urlopen(req, timeout=60).read()
                return r["run"], {"date": h[168:176].decode().strip(), "time": h[176:184].decode().strip()}
            except OSError:
                if i == 4:
                    raise
                time.sleep(2 * (i + 1))

    import concurrent.futures as cf
    todo = [r for r in idx if "eeg" in r["files"] and r["run"] not in got]
    with cf.ThreadPoolExecutor(8) as ex:
        for n, (run, v) in enumerate(tqdm(ex.map(one, todo), total=len(todo), desc="EDF headers")):
            got[run] = v
            if n % 200 == 0:
                part.write_text(json.dumps(got))
    out = got
    path.write_text(json.dumps(out))
    part.unlink(missing_ok=True)
    times = [v["time"] for v in out.values()]
    print(f"{len(out)} runs; distinct start times: {len(set(times))}; examples: {times[:5]}")
    return out


def hours_of_day(run, t, st):
    """Clock hour (0-24, float) of each window end time t (s from the recording start)."""
    h, m, s = (int(x) for x in st[run]["time"].split("."))
    return ((h * 3600 + m * 60 + s + t) % 86400) / 3600


# ------------------------------------------------------------------ (3) target seizures
def target_scores(recs_typed, is_target):
    """recs_typed: [(p, t, [(a, b, type)], h)]. False alarms counted against ALL seizures; sensitivity
    against target seizures only (the alarm sequence does not depend on the seizure list)."""
    allr = [(p, t, [(a, b) for a, b, _ in sz], h) for p, t, sz, h in recs_typed]
    tgt = [(p, t, [(a, b) for a, b, ty in sz if is_target(a, b, ty)], h) for p, t, sz, h in recs_typed]
    ca, ct = S.score_fast(allr, S.PT_GRID), S.score_fast(tgt, S.PT_GRID)
    return {th: {**ct[th], "fa_per_24h": ca[th]["fa_per_24h"], "false_alarms": ca[th]["false_alarms"]} for th in ca}


TARGETS = {
    "all seizures": lambda a, b, ty: True,
    "focal-to-bilateral tonic-clonic": lambda a, b, ty: "f2b" in ty,
    "long seizures (> 90 s)": lambda a, b, ty: b - a > 90,
    "tonic-clonic or > 90 s": lambda a, b, ty: "f2b" in ty or b - a > 90,
    "seizures >= 30 s": lambda a, b, ty: b - a >= 30,
}


def cached_typed(model="general", cname=BEST):
    """Cached k=2 outputs on each patient's later data, with seizure types attached."""
    data = pickle.loads((S.PROBS / f"{S.slug(cname)}_k2.pkl").read_bytes())
    by = by_patient()
    out = {}
    for s, d in data.items():
        out[s] = []
        for (p, t, sz, h), (r, _) in zip(d["test"][model], test_runs(by[s])):
            ty = {round(a, 1): tt for a, _, tt in sz_types(r)}
            out[s].append((p, t, [(a, b, ty.get(round(a, 1), "sz")) for a, b in sz], h, r["run"]))
    return out


def target():
    res = {}
    for model in ("general", "finetuned"):
        data = cached_typed(model)
        recs = [x[:4] for v in data.values() for x in v]
        print(f"\n=== best model ({model}), 71 patients, later data: alarm only for target seizures")
        for name, f in TARGETS.items():
            s = summary(target_scores(recs, f))
            res[f"best {model} | {name}"] = s
            show(name, s)
    # (b) a dedicated convulsive model on all 125 patients
    idx = json.loads((S.FEAT / "index.json").read_text())
    runs = [r for r in idx if (S.FEAT / f"{r['run']}.npz").exists()]
    subs, fold = S.subject_folds(runs)
    n_tc = sum("f2b" in ty for r in runs for _, _, ty in sz_types(r))
    print(f"\n=== all 125 patients ({n_tc} tonic-clonic seizures), patient-wise 5-fold")
    groups = {"best general model (all seizures as positives)": S.CONFIGS[BEST],
              "dedicated convulsive model (tonic-clonic positives, + EMG + motion)": ["eeg", "ecg", "emg", "mov", "ctx", "hrx"]}
    for gname, groups_ in groups.items():
        conv = "dedicated" in gname
        rng = np.random.default_rng(1)
        Xs, ys, fs_ = [], [], []
        for r in tqdm(runs, desc=f"{gname[:30]}: loading", leave=False):
            X, y, t = S.load(r["run"])
            if conv:  # positives: tonic-clonic windows only; other seizures left out of training
                tc = np.zeros(len(y), bool)
                oth = np.zeros(len(y), bool)
                c = t - S.WIN / 2
                for a, b, ty in sz_types(r):
                    (tc if "f2b" in ty else oth)[(c >= a) & (c <= b)] = True
                keep = tc | ((~oth) & (y == 0) & (rng.random(len(y)) < S.NEG_FRAC))
                yy = tc.astype(np.int8)
            else:
                keep = (y == 1) | (rng.random(len(y)) < S.NEG_FRAC)
                yy = y
            Xs.append(S.design(X, groups_)[keep])
            ys.append(yy[keep])
            fs_.append(np.full(keep.sum(), fold[r["sub"]], np.int8))
        Xp, yp, fp = np.vstack(Xs), np.concatenate(ys), np.concatenate(fs_)
        del Xs
        recs = []
        for k in tqdm(range(5), desc=f"{gname[:30]}: folds", leave=False):
            m = D.clf().fit(Xp[fp != k], yp[fp != k])
            for r in runs:
                if fold[r["sub"]] == k:
                    X, _, t = S.load(r["run"])
                    recs.append((m.predict_proba(S.design(X, groups_))[:, 1].astype(np.float32), t, sz_types(r),
                                 r["duration"] / 3600))
        del Xp
        for name in ("focal-to-bilateral tonic-clonic", "all seizures"):
            s = summary(target_scores(recs, TARGETS[name]))
            res[f"125 pts | {gname} | target: {name}"] = s
            show(f"{gname[:40]} -> {name}", s)
    save("target_seizures", res)


# ------------------------------------------------------------------ (4) night mode
def night():
    st = clock()
    res = {}
    for model in ("general", "finetuned"):
        data = cached_typed(model)
        print(f"\n=== best model ({model}): alarms only at night (22:00-07:00), day = diary")
        for mode, lo, hi in (("24 h alarms", 0, 24), ("night only 22-07", 22, 7), ("night only 23-06", 23, 6)):
            recs, n_hours = [], 0.0
            for v in data.values():
                for p, t, sz, h, run in v:
                    hr = hours_of_day(run, t, st)
                    on = np.ones(len(t), bool) if hi == 24 else ((hr >= lo) | (hr < hi))
                    pp = np.where(on, p, 0.0).astype(np.float32)  # outside the alarm window: log only
                    # every seizure stays in the list (an alarm near a day seizure is not false); only
                    # seizures starting inside the alarm window are targets
                    tsz = [(a, b, ("N|" if on[min(np.searchsorted(t, a), len(t) - 1)] else "D|") + ty) for a, b, ty in sz]
                    recs.append((pp, t, tsz, on.sum() * S.STEP / 3600))  # FA per 24 h of ACTIVE alarm time
                    n_hours += on.sum() * S.STEP / 3600
            s = summary(target_scores(recs, lambda a, b, ty: ty.startswith("N|")))
            s["active_hours"] = n_hours
            res[f"best {model} | {mode}"] = s
            show(f"{mode} (FA per 24 h of alarm time)", s)
        # which fraction of seizures happen at night, and how asleep/awake labels line up
        nsz = [(ty, hours_of_day(run, np.array([a]), st)[0]) for v in data.values() for _, _, sz, _, run in v for a, _, ty in sz]
        night_frac = np.mean([(h >= 22) | (h < 7) for _, h in nsz])
        res[f"best {model} | share of later seizures at night"] = float(night_frac)
        print(f"  share of later seizures between 22:00 and 07:00: {night_frac:.0%}")
    save("night_mode", res)


# ------------------------------------------------------------------ (4b) resting mode (no clock needed)
def activity(X):
    """Movement per window: the larger of the two accelerometer log-SDs (NaN if motion is missing)."""
    mv = X[:, [S.NAMES.index("back_acc_logstd"), S.NAMES.index("chest_acc_logstd")]].astype(np.float64)
    return np.where(np.isnan(mv).all(1), np.nan, np.nanmax(np.where(np.isnan(mv), -np.inf, mv), 1))


def trailing_median(x, n, block=30):
    """Causal ~10-min movement level: mean movement per completed 1-min block, then the median of the
    last n/block completed blocks, mapped back to every window (NaN until enough history)."""
    nb = len(x) // block
    if nb == 0:
        return np.full(len(x), np.nan)
    with np.errstate(all="ignore"):
        bm = np.nanmean(x[: nb * block].reshape(nb, block), 1)
    k = n // block
    pad = np.r_[np.full(k - 1, np.nan), bm]
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.lib.stride_tricks.sliding_window_view(pad, k), 1)  # median of blocks j-k+1..j
    ok = np.lib.stride_tricks.sliding_window_view(np.isfinite(pad), k).sum(1) >= k // 3
    med = np.where(ok, med, np.nan)
    out = np.full(len(x), np.nan)
    j = np.arange(len(x)) // block - 1  # last COMPLETED block before window i
    has = (j >= 0) & (j < nb)
    out[has] = med[j[has]]
    return out


def vigilance_map():
    """(run, onset rounded to 0.1 s) -> 'awake' / 'asleep' / other, from the cached events.tsv files."""
    ev = json.loads((S.FEAT.parent / "events_all.json").read_text())
    vm = {}
    for k, txt in ev.items():
        run = k.split("/")[-1][: -len("_events.tsv")]
        rows = [r.rstrip("\r").split("\t") for r in txt.strip().split("\n")]
        for r in rows[1:]:
            e = dict(zip(rows[0], r))
            if e["eventType"].startswith("sz"):
                vm[(run, round(float(e["onset"]), 1))] = e.get("vigilance", "n/a")
    return vm


def rest():
    """Alarms only while the wearer has been lying still (trailing 10-min median movement at or below the
    user's own q-th percentile, learned on the tuning period); otherwise log only. The clock is not needed:
    the device's accelerometer defines the state. Checked against the dataset's asleep/awake seizure labels."""
    by = by_patient()
    vm = vigilance_map()
    res = {}
    step_n = int(600 / S.STEP)  # 10 min
    for model in ("general", "finetuned"):
        data = cached_typed(model)
        # per-patient still threshold from the tuning period
        thr_q = {}
        for q in (20, 35, 50):
            thr_q[q] = {}
            for s in data:
                tune, _ = S.split_patient(by[s], 2)
                vals = []
                for r, cut in tune:
                    X, _, t = S.load(r["run"])
                    a = activity(X)
                    if cut is not None:
                        a = a[t <= cut]
                    vals.append(a[np.isfinite(a)])
                v = np.concatenate(vals) if vals else np.array([])
                thr_q[q][s] = float(np.percentile(v, q)) if len(v) > 100 else np.nan
        # state per test window (computed once)
        state_cache = {}
        for s, v in data.items():
            for p, t, sz, h, run in v:
                X, _, tt = S.load(run)
                am = trailing_median(activity(X), step_n)
                idx = np.clip(np.round((t - S.WIN) / S.STEP).astype(int), 0, len(am) - 1)
                state_cache[(s, run, float(t[0]))] = am[idx]
        print(f"\n=== best model ({model}): alarms only while lying still for 10+ min (day activity = diary)")
        base = summary(target_scores([x[:4] for v in data.values() for x in v], TARGETS["all seizures"]))
        res[f"best {model} | 24 h alarms"] = base
        show("24 h alarms (reference)", base)
        for q in (20, 35, 50):
            recs, act_h, tot_h, cnt = [], 0.0, 0.0, {"asleep": [0, 0], "awake": [0, 0]}
            for s, v in data.items():
                th = thr_q[q][s]
                for p, t, sz, h, run in v:
                    am = state_cache[(s, run, float(t[0]))]
                    on = np.isfinite(am) & (am <= th) if np.isfinite(th) else np.zeros(len(t), bool)
                    pp = np.where(on, p, 0.0).astype(np.float32)
                    tsz = []
                    for a, b, ty in sz:
                        st_on = bool(on[min(np.searchsorted(t, a), len(t) - 1)])
                        tsz.append((a, b, ("N|" if st_on else "D|") + ty))
                        vg = vm.get((run, round(a, 1)), "n/a")
                        if vg in cnt:
                            cnt[vg][0] += st_on
                            cnt[vg][1] += 1
                    recs.append((pp, t, tsz, on.sum() * S.STEP / 3600))
                    act_h += on.sum() * S.STEP / 3600
                    tot_h += h
            s_ = summary(target_scores(recs, lambda a, b, ty: ty.startswith("N|")))
            s_["share_of_time_in_alarm_mode"] = act_h / max(tot_h, 1e-9)
            s_["asleep_seizures_in_rest_state"] = cnt["asleep"][0] / max(cnt["asleep"][1], 1)
            s_["awake_seizures_in_rest_state"] = cnt["awake"][0] / max(cnt["awake"][1], 1)
            res[f"best {model} | rest mode q{q}"] = s_
            show(f"rest mode (still <= own {q}th pct; {act_h / max(tot_h, 1e-9):.0%} of time; FA per 24 h of it)", s_)
            print(f"      seizures in rest state: asleep-labelled {s_['asleep_seizures_in_rest_state']:.0%} "
                  f"({cnt['asleep'][1]}), awake-labelled {s_['awake_seizures_in_rest_state']:.0%} ({cnt['awake'][1]})")
    save("rest_mode", res)


# ------------------------------------------------------------------ (5) risk priors
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def prior():
    st = {r: {"time": "00.00.00"} for r in json.loads((S.FEAT.parent / "starttimes.json").read_text())}  # unused
    res = {}
    data = cached_typed("general")
    by = by_patient()
    # onset-hour histogram per patient (all their seizures), used leave-one-patient-out
    onset_h = {s: [hours_of_day(r["run"], np.array([a]), st)[0] for r in by[s] for a, _ in r["seizures"]]
               for s in by}
    print("\n=== best general model + risk priors (71 patients, later data)")
    variants = {"no prior": (0, 0, 0)}
    for w in (1.0, 2.0):
        variants[f"cluster: +{w:g} log-odds for 12 h after a confirmed seizure"] = (w, 12, 0)
        variants[f"cluster: +{w:g} log-odds for 24 h after a confirmed seizure"] = (w, 24, 0)
    # time-of-day priors dropped: SeizeIT2 anonymises every recording start time to 00.00.00
    for vname, (wc, hc, wt) in variants.items():
        recs = []
        for s, v in data.items():
            others = [h for s2, hs in onset_h.items() if s2 != s for h in hs]
            hist = np.histogram(others, bins=24, range=(0, 24))[0] + 1.0
            rel = np.log(hist / hist.mean())  # > 0 at hours when seizures are common
            rs = sorted(by[s], key=S.run_no)
            # timeline of the patient's seizures across runs (run order; gaps between runs unknown: 60 s assumed).
            # A seizure counts as "confirmed" only after it has ENDED (+60 s), as a caregiver or diary would
            # confirm it; the boost therefore never uses the seizure currently being detected.
            t_off, abs_start = 0.0, {}
            for r in rs:
                abs_start[r["run"]] = t_off
                t_off += r["duration"] + 60
            ends = np.array(sorted(abs_start[r["run"]] + b for r in rs for _, b in r["seizures"]))
            for p, t, sz, h, run in v:
                z = logit(np.convolve(p, np.ones(3) / 3)[:len(p)])
                ta = abs_start[run] + t
                if wc:
                    j = np.searchsorted(ends, ta - 60, side="right") - 1  # last seizure that ended >= 60 s ago
                    since = np.where(j >= 0, ta - ends[np.maximum(j, 0)], np.inf)
                    z = z + wc * (since < hc * 3600)
                if wt:
                    z = z + wt * rel[np.floor(hours_of_day(run, t, st)).astype(int) % 24]
                pp = (1 / (1 + np.exp(-z))).astype(np.float32)
                recs.append((pp, t, [(a, b) for a, b, _ in sz], h))
        s = summary(S.score_fast(recs, S.PT_GRID, smooth=1))  # already smoothed above
        res[vname] = s
        show(vname, s)
    save("risk_priors", res)


# ------------------------------------------------------------------ (1) stack the winners
def stack():
    import stage2 as ST
    by = by_patient()
    k2 = pickle.loads((S.PROBS / f"{S.slug(BEST)}_k2.pkl").read_bytes())
    k1 = S.collect_probs(BEST, 1)  # fine-tuned after the 1st seizure; cached after the first call
    subs = sorted(set(k2) & set(k1))
    print(f"{len(subs)} patients with both splits", flush=True)

    def segments(s, which):
        """Records (run, sel, p, t, sz, h) for stage-1 = `which` on: 'own' = the patient's personal-training
        stretch, 'test' = the SAME later data as every k=2 table."""
        _, test2 = S.split_patient(by[s], 2)
        tune2, _ = S.split_patient(by[s], 2)
        cut2_run, cut2 = tune2[-1][0]["run"], tune2[-1][1]
        order = {r["run"]: i for i, r in enumerate(sorted(by[s], key=S.run_no))}
        out = {"own": [], "test": []}
        if which == "general":  # baseline two-stage: own = tuning data up to cut2 (general model is out-of-fold)
            for part, src in (("own", "tune"), ("test", "test")):
                lst = S.split_patient(by[s], 2)[0 if part == "own" else 1]
                recs = k2[s][src]["general"]
                k = 0
                for r, cut in lst:
                    tt = np.load(S.FEAT / f"{r['run']}.npz")["t"]
                    sel = (np.ones(len(tt), bool) if cut is None else (tt <= cut) if part == "own" else (tt - S.WIN > cut))
                    if sel.any():
                        out[part].append((r["run"], sel) + tuple(recs[k]))
                        k += 1
            return out
        # fine-tuned (k=1) outputs cover everything after cut1: split them at cut2
        _, test1 = S.split_patient(by[s], 1)
        recs_ft, recs_gen = k1[s]["test"]["finetuned"], k1[s]["test"]["general"]
        k = 0
        for r, cut in test1:
            tt = np.load(S.FEAT / f"{r['run']}.npz")["t"]
            sel = np.ones(len(tt), bool) if cut is None else (tt - S.WIN > cut)
            if not sel.any():
                continue
            p_ft, t, sz, h = recs_ft[k]
            p_g = recs_gen[k][0]
            k += 1
            p = p_ft if which == "finetuned" else (p_ft + p_g) / 2
            tsel = tt[sel]
            before = order[r["run"]] < order[cut2_run] or (r["run"] == cut2_run)
            after = order[r["run"]] > order[cut2_run] or (r["run"] == cut2_run)
            for part, m in (("own", (tsel <= cut2) if r["run"] == cut2_run else np.full(len(tsel), before)),
                            ("test", (tsel - S.WIN > cut2) if r["run"] == cut2_run else np.full(len(tsel), after))):
                if m.any():
                    full = np.zeros(len(tt), bool)
                    full[np.flatnonzero(sel)[m]] = True
                    ts = t[m]
                    szs = [(a, b) for a, b in sz if (a <= cut2 if part == "own" else a > cut2) or r["run"] != cut2_run]
                    out[part].append((r["run"], full, p[m], ts, szs, m.sum() * S.STEP / 3600))
        return out

    res = {}
    for which in ("general", "finetuned", "finetuned+general mean"):
        rows = []  # (patient, part, anchor, label, features) + test records
        test_recs = {}
        for s in tqdm(subs, desc=f"stage-1 = {which}: candidates", unit="patient"):
            seg = segments(s, which)
            for part, lst in seg.items():
                for run, sel, p, t, sz, h in lst:
                    z = np.load(S.FEAT / f"{run}.npz")
                    X = z["X"].astype(np.float32)
                    H = S.hr_vs_motion(X)[sel]
                    X = X[sel]
                    cand, ps = ST.candidates(p, t, 0.5)
                    for a in cand:
                        lab = 1 if any(x - D.PRE <= a <= y for x, y in sz) else (
                            -1 if any(x - D.PRE <= a <= y + D.POST for x, y in sz) else 0)
                        rows.append((s, part, f"{s}|{run}|{part}", a, lab, ST.event_features(ps, t, X, H, a, 10, cand)))
                    if part == "test":
                        test_recs[f"{s}|{run}|test"] = (t, sz, h)
        Xc = np.array([f for *_, f in rows], np.float32)
        y = np.array([r[4] for r in rows])
        ss = np.array([r[0] for r in rows])
        parts = np.array([r[1] for r in rows])
        rid = np.array([r[2] for r in rows])
        aa = np.array([r[3] for r in rows])
        p2 = np.full(len(rows), np.nan)
        from sklearn.ensemble import HistGradientBoostingClassifier as H_
        for s in tqdm(subs, desc=f"stage-1 = {which}: personal verifier", unit="patient", leave=False):
            own = ss == s
            tr = ((~own) | (parts == "own")) & (y >= 0)
            te = own & (parts == "test")
            if te.any() and (y[tr] == 1).any():
                w = np.where(own[tr], ST.PERSONAL_W, 1.0)
                m = H_(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced", random_state=0)
                p2[te] = m.fit(Xc[tr], y[tr], sample_weight=w).predict_proba(Xc[te])[:, 1]
        curve = []
        for tau in np.r_[np.linspace(0.01, 0.9, 60), 1 - np.logspace(-1, -3.5, 40)]:
            acc = {}
            for r_, a, pv, part in zip(rid, aa, p2, parts):
                if part == "test" and pv > tau:
                    acc.setdefault(r_, []).append(a)
            sens, far, nsz = ST.score(acc, test_recs)
            curve.append((float(tau), sens, far))
        best = {str(B): max([c[1] for c in curve if c[2] <= B], default=None) for B in BUDGETS}
        res[f"personal verifier on stage-1 = {which}"] = {"sens_at_budget": best, "patients": len(subs), "curve": curve}
        print(f"  stage-1 = {which:28s}: " + "  ".join(
            f"<={B}/d: {v:.0%}" if v is not None else f"<={B}/d: -" for B, v in best.items()), flush=True)
    save("stack", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["chb", "clock", "target", "night", "rest", "prior", "stack"])
    a = ap.parse_args()
    S.keep_awake()
    {"chb": chb, "clock": clock, "target": target, "night": night, "rest": rest, "prior": prior, "stack": stack}[a.cmd]()

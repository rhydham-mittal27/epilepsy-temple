"""Multimodal seizure detector on SeizeIT2 (OpenNeuro ds005873): behind-the-ear EEG + ECG + EMG + motion.

Question: how many false alarms do the body signals remove, compared with ear-EEG alone?

SeizeIT2 = 125 patients with focal epilepsy, 883 seizures, ~11,600 h, recorded in 5 European EMUs
with the Byteflies Sensor Dot: 2 behind-the-ear EEG channels (256 Hz), chest ECG (256 Hz), left
deltoid EMG (256 Hz) and accelerometer/gyroscope on two devices (upper back + chest, 25 Hz).

Steps
  fetch  STREAM every run straight from OpenNeuro's S3 bucket into memory (no EDF is ever written to
         disk), decode the EDF in RAM, compute per-window features from the full-resolution signal,
         keep only the features (~few GB). Parallel + resumable: re-run to continue.
  run    Subject-wise 5-fold cross-validation (every patient is tested by a model that never saw them):
           EEG only | ECG only | body only (ECG+EMG+motion) | EEG+ECG | EEG+ECG+EMG+motion
         Metrics: seizures detected, false alarms / 24 h, latency; and false alarms at MATCHED
         sensitivity (the fair comparison: same seizures caught, how many fewer false alarms).

Usage (run in your own terminal, from epilepsy_sim/):
    uv run detect/seizeit2.py fetch     # ~1.5-2.5 h at ~20 MB/s (126 GB streamed), resumable
    uv run detect/seizeit2.py run       # ~30-60 min
    uv run detect/seizeit2.py finetune  # personal fine-tuning + sustained-alarm rules (~2-3 h)
    uv run detect/seizeit2.py personal  # shared vs personal alarm threshold (~35 min; outputs cached)

Data: SeizeIT2 (CC0), Bhagubai et al. 2025, doi:10.18112/openneuro.ds005873.v1.1.0.
"""

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, lfilter, sosfilt, sosfiltfilt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect  # noqa: E402
from detect import clf, score  # noqa: E402

S3 = "https://s3.amazonaws.com/openneuro.org"
DS = "ds005873"
ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "seizeit2" / "features"
RES = Path(__file__).resolve().parent / "results_seizeit2"
WIN, STEP = 4, 2  # s; 4 s windows hold enough heartbeats for HR, 2 s step keeps 11,600 h manageable
detect.SMOOTH = 3  # causal smoothing over 3 windows = 6 s (detect.alarms_from_prob reads this global)
EMA_N = 300  # per-recording baseline memory: 300 windows x 2 s = 10 min
NEG_FRAC = 0.03  # fraction of non-seizure windows used for training (all seizure windows kept)
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

EEG_NAMES = [f"{c}_{n}" for c in ("ch1", "ch2") for n in
             ("ll", "logvar", "mob", "comp", "bp_d", "bp_t", "bp_a", "bp_b", "bp_g",
              "rel_d", "rel_t", "rel_a", "rel_b", "rel_g")]
ECG_NAMES = ["hr_4s", "hr_10s", "hr_30s", "hr_slope_30s", "rmssd_30s", "sdnn_30s", "beat_quality", "ecg_ll"]
EMG_NAMES = ["emg_logrms", "emg_ll", "emg_zcr", "emg_hf_frac", "emg_lf_logpow"]
MOV_NAMES = [f"{d}_{n}" for d in ("back", "chest") for n in ("acc_logstd", "acc_ll", "acc_2_8hz_frac", "gyr_logmean")]
NAMES = EEG_NAMES + ECG_NAMES + EMG_NAMES + MOV_NAMES
GROUPS = {"eeg": EEG_NAMES, "ecg": ECG_NAMES, "emg": EMG_NAMES, "mov": MOV_NAMES}
CONFIGS = {
    "EEG only (ear)": ["eeg"],
    "ECG only": ["ecg"],
    "Body only (ECG+EMG+motion)": ["ecg", "emg", "mov"],
    "EEG + ECG": ["eeg", "ecg"],
    "EEG + ECG + EMG + motion": ["eeg", "ecg", "emg", "mov"],
}


def notify(title, msg, tags="white_check_mark"):
    """Ring the phone via ntfy (topic in ntfy_topic.txt, not committed). Silent if not set up."""
    f = Path(__file__).resolve().parent / "ntfy_topic.txt"
    if f.exists():
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{f.read_text().strip()}", data=msg.encode(),
                                         headers={"Title": title, "Priority": "urgent", "Tags": tags})
            urllib.request.urlopen(req, timeout=30)
        except OSError:
            pass


# ------------------------------------------------------------------ S3 index
def s3_keys():
    keys, tok = {}, None
    while True:
        q = f"list-type=2&prefix={DS}/&max-keys=1000" + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else "")
        xml = urllib.request.urlopen(f"{S3}?{q}", timeout=120).read().decode()
        for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", xml):
            keys[k] = int(s)
        m = re.search(r"<NextContinuationToken>([^<]*)", xml)
        if not m:
            return keys
        tok = m.group(1)


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError):
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def build_index():
    """List all runs, their modalities and seizures. Cached in index.json."""
    FEAT.mkdir(parents=True, exist_ok=True)
    path = FEAT / "index.json"
    if path.exists():
        return json.loads(path.read_text())
    print("listing the dataset on S3 ...", flush=True)
    keys = s3_keys()
    ev = sorted(k for k in keys if k.endswith("_events.tsv"))

    def one(k):
        txt = get(f"{S3}/{k}").decode()
        rows = [r.rstrip("\r").split("\t") for r in txt.strip().split("\n")]
        h = rows[0]
        evs = [dict(zip(h, r)) for r in rows[1:]]
        stem = k[: -len("_events.tsv")]  # ds/sub/ses/eeg/sub_ses_task_run
        sub, ses = stem.split("/")[1:3]
        base = stem.split("/")[-1]
        mods = {m: f"{DS}/{sub}/{ses}/{m}/{base}_{m}.edf" for m in ("eeg", "ecg", "emg", "mov")}
        mods = {m: p for m, p in mods.items() if p in keys}
        sz = [(float(e["onset"]), float(e["onset"]) + float(e["duration"]))
              for e in evs if e["eventType"].startswith("sz")]
        types = [e["eventType"] for e in evs if e["eventType"].startswith("sz")]
        return {"sub": sub, "run": base, "files": mods, "seizures": sz, "types": types,
                "duration": float(evs[0]["recordingDuration"]),
                "bytes": sum(keys[p] for p in mods.values())}

    with cf.ThreadPoolExecutor(32) as ex:
        runs = list(tqdm(ex.map(one, ev), total=len(ev), desc="events", unit="run"))
    path.write_text(json.dumps(runs, indent=0))
    return runs


# ------------------------------------------------------------------ in-memory EDF reader
def read_edf(buf):
    """Decode an EDF/EDF+ byte string. Returns {label: (float32 signal in physical units, fs)}."""
    h = buf[:256]
    nhdr, nrec, dur, ns = int(h[184:192]), int(h[236:244]), float(h[244:252]), int(h[252:256])

    def field(off, w):
        return [buf[256 + off * ns + i * w: 256 + off * ns + (i + 1) * w].decode("latin-1").strip() for i in range(ns)]

    lab = field(0, 16)
    pmin, pmax = np.array(field(104, 8), float), np.array(field(112, 8), float)
    dmin, dmax = np.array(field(120, 8), float), np.array(field(128, 8), float)
    nsamp = np.array(field(216, 8), int)
    rec = nsamp.sum()
    nrec = min(nrec, (len(buf) - nhdr) // (2 * rec))
    data = np.frombuffer(buf, "<i2", count=nrec * rec, offset=nhdr).reshape(nrec, rec)
    out, o = {}, 0
    for i in range(ns):
        if lab[i] != "EDF Annotations":
            g = (pmax[i] - pmin[i]) / max(dmax[i] - dmin[i], 1)
            x = data[:, o:o + nsamp[i]].reshape(-1).astype(np.float32)
            out[lab[i]] = ((x - dmin[i]) * g + pmin[i]).astype(np.float32), nsamp[i] / dur
        o += nsamp[i]
    return out


# ------------------------------------------------------------------ features
def windows(n, fs):
    w, s = int(WIN * fs), int(STEP * fs)
    return w, s, max((n - w) // s + 1, 0)


def chunked(nwin, size=4000):
    for a in range(0, nwin, size):
        yield a, min(a + size, nwin)


def eeg_features(x, fs, nwin):
    """Same 14 features per channel as the CHB-MIT detector, on a 4 s window."""
    x = sosfiltfilt(butter(4, [0.5, 45], btype="band", fs=fs, output="sos"), x).astype(np.float32)
    w, s, _ = windows(len(x), fs)
    freqs = np.fft.rfftfreq(w, 1 / fs)
    hann = np.hanning(w).astype(np.float32)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    out = np.empty((nwin, 14), np.float32)
    for a, b in chunked(nwin):
        seg = x[np.arange(w)[None] + (np.arange(a, b) * s)[:, None]]
        d1 = np.diff(seg, axis=1)
        d2 = np.diff(d1, axis=1)
        v0, v1, v2 = seg.var(1) + 1e-6, d1.var(1) + 1e-6, d2.var(1) + 1e-6
        mob = np.sqrt(v1 / v0)
        P = np.abs(np.fft.rfft((seg - seg.mean(1, keepdims=True)) * hann, axis=1)) ** 2
        bp = np.stack([P[:, m].sum(1) for m in masks], 1) + 1e-6
        out[a:b, 0] = np.log(np.abs(d1).sum(1) / WIN + 1e-6)
        out[a:b, 1] = np.log(v0)
        out[a:b, 2] = mob
        out[a:b, 3] = np.sqrt(v2 / v1) / mob
        out[a:b, 4:9] = np.log(bp)
        out[a:b, 9:14] = bp / bp.sum(1, keepdims=True)
    return out


def rpeaks(x, fs):
    """Pan-Tompkins-style R-peak detector with an adaptive (10 s block) threshold."""
    y = sosfiltfilt(butter(3, [5, 20], btype="band", fs=fs, output="sos"), x)
    e = np.diff(y, prepend=y[0]) ** 2
    e = np.convolve(e, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same")
    blk = int(10 * fs)
    nb = max(len(e) // blk, 1)
    lvl = np.percentile(e[: nb * blk].reshape(nb, -1), 98, axis=1)
    thr = np.interp(np.arange(len(e)), (np.arange(nb) + 0.5) * blk, lvl) * 0.3
    pk, _ = find_peaks(e, height=thr, distance=int(0.27 * fs))  # refractory 270 ms (max ~220 bpm)
    return pk / fs


def ecg_features(x, fs, nwin):
    T = np.arange(nwin) * STEP + WIN  # window end times
    bt = rpeaks(x, fs)
    rr = np.diff(bt)
    ok = (rr > 0.27) & (rr < 2.0)
    if len(rr) > 2:  # reject beats deviating >30 % from the local median RR (missed / extra beats)
        med = np.convolve(np.clip(rr, 0.27, 2.0), np.ones(9) / 9, mode="same")
        ok &= np.abs(rr - med) < 0.3 * med
    tb, hr = bt[1:][ok], 60 / rr[ok]
    dd = np.r_[np.nan, np.diff(rr[ok])]  # successive RR differences, only between adjacent clean beats
    if len(dd):
        dd[~np.r_[False, ok[1:] & ok[:-1]][ok]] = np.nan

    def wmean(v, L):
        c = np.r_[0, np.cumsum(np.nan_to_num(v))]
        n = np.r_[0, np.cumsum(~np.isnan(v))]
        lo, hi = np.searchsorted(tb, T - L), np.searchsorted(tb, T)
        cnt = n[hi] - n[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt >= 2, (c[hi] - c[lo]) / cnt, np.nan), cnt

    h4, _ = wmean(hr, 4)
    h10, _ = wmean(hr, 10)
    h30, n30 = wmean(hr, 30)
    m2, _ = wmean(dd ** 2 * 1e6, 30)  # ms^2
    rr_ms = 60000 / hr if len(hr) else hr
    mr, _ = wmean(rr_ms, 30)
    mr2, _ = wmean(rr_ms ** 2, 30)
    lag = int(30 / STEP)
    slope = h10 - np.r_[np.full(lag, np.nan), h10[:-lag]] if nwin > lag else np.full(nwin, np.nan)
    expect = np.clip(h30, 30, 220) / 2 if np.isfinite(h30).any() else np.full(nwin, np.nan)
    ll = np.abs(np.diff(x[: (nwin - 1) * int(STEP * fs) + int(WIN * fs)]))
    w, s, _ = windows(len(x), fs)
    c = np.r_[0, np.cumsum(ll)]
    ends = np.minimum(np.arange(nwin) * s + w - 1, len(ll))
    ecg_ll = np.log((c[ends] - c[np.arange(nwin) * s]) / WIN + 1e-3)
    with np.errstate(invalid="ignore"):
        return np.stack([h4, h10, h30, slope, np.sqrt(m2), np.sqrt(np.maximum(mr2 - mr ** 2, 0)),
                         n30 / expect, ecg_ll], 1).astype(np.float32)


def emg_features(x, fs, nwin):
    sos = butter(4, [20, min(120, fs / 2 - 1)], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(butter(2, [48, 52], btype="bandstop", fs=fs, output="sos"), sosfiltfilt(sos, x))
    lo = sosfiltfilt(butter(4, [0.5, 20], btype="band", fs=fs, output="sos"), x)
    w, s, _ = windows(len(x), fs)
    freqs = np.fft.rfftfreq(w, 1 / fs)
    out = np.empty((nwin, 5), np.float32)
    for a, b in chunked(nwin):
        ix = np.arange(w)[None] + (np.arange(a, b) * s)[:, None]
        seg = y[ix]
        P = np.abs(np.fft.rfft(seg, axis=1)) ** 2 + 1e-9
        out[a:b, 0] = 0.5 * np.log(np.mean(seg ** 2, 1) + 1e-6)
        out[a:b, 1] = np.log(np.abs(np.diff(seg, axis=1)).sum(1) / WIN + 1e-6)
        out[a:b, 2] = (np.diff(np.signbit(seg), axis=1)).sum(1) / WIN
        out[a:b, 3] = P[:, freqs >= 60].sum(1) / P.sum(1)
        out[a:b, 4] = np.log(np.var(lo[ix], 1) + 1e-6)
    return out


def mov_features(sig, nwin):
    """Accelerometer/gyroscope of the back (EEG) and chest (ECG/EMG) devices; NaN where absent."""
    out = np.full((nwin, 8), np.nan, np.float32)
    for d, dev in enumerate(("EEG SD", "ECGEMG SD")):
        acc = [sig.get(f"{dev} ACC {a}") for a in "XYZ"]
        if any(v is None for v in acc):
            continue
        fs = acc[0][1]
        n = min(len(v[0]) for v in acc)
        mag = np.sqrt(sum(v[0][:n].astype(np.float64) ** 2 for v in acc))
        gyr = [sig.get(f"{dev} GYR {a}") for a in "ABC"]
        gmag = np.sqrt(sum(v[0][:n].astype(np.float64) ** 2 for v in gyr)) if all(v is not None for v in gyr) else None
        w, s, nw = windows(n, fs)
        nw = min(nw, nwin)
        ix = np.arange(w)[None] + (np.arange(nw) * s)[:, None]
        seg = mag[ix]
        freqs = np.fft.rfftfreq(w, 1 / fs)
        P = np.abs(np.fft.rfft(seg - seg.mean(1, keepdims=True), axis=1)) ** 2 + 1e-12
        out[:nw, 4 * d + 0] = np.log(seg.std(1) + 1e-5)
        out[:nw, 4 * d + 1] = np.log(np.abs(np.diff(seg, axis=1)).sum(1) / WIN + 1e-5)
        out[:nw, 4 * d + 2] = P[:, (freqs >= 2) & (freqs <= 8)].sum(1) / P[:, freqs >= 0.3].sum(1)
        if gmag is not None:
            out[:nw, 4 * d + 3] = np.log(gmag[ix].mean(1) + 1e-5)
    return out


def process_run(r):
    """Stream the run's EDFs into memory, compute features, save only the features."""
    cache = FEAT / f"{r['run']}.npz"
    if cache.exists():
        return r["run"], "cached", 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:  # the 4 modalities stream in parallel
        bufs = dict(zip(r["files"], ex.map(lambda p: get(f"{S3}/{p}"), r["files"].values())))
    nbytes = sum(len(b) for b in bufs.values())
    sig = {m: read_edf(b) for m, b in bufs.items()}
    del bufs
    eeg = sig.get("eeg", {})
    eeg = [v for k, v in eeg.items()][:2]
    n_sec = min(len(v[0]) / v[1] for v in eeg)
    nwin = windows(int(n_sec), 1)[2]
    X = np.full((nwin, len(NAMES)), np.nan, np.float32)
    for c, (x, fs) in enumerate(eeg):
        X[:, 14 * c:14 * c + 14] = eeg_features(x, fs, nwin)
    o = len(EEG_NAMES)

    def fill(feats, width):
        m = min(len(feats), nwin)
        X[:m, o:o + width] = feats[:m]

    if "ecg" in sig and sig["ecg"]:
        x, fs = next(iter(sig["ecg"].values()))
        if len(x) / fs >= WIN * 10:
            fill(ecg_features(x.astype(np.float64), fs, min(nwin, windows(len(x), fs)[2])), 8)
    o += len(ECG_NAMES)
    if "emg" in sig and sig["emg"]:
        x, fs = next(iter(sig["emg"].values()))
        fill(emg_features(x.astype(np.float64), fs, min(nwin, windows(len(x), fs)[2])), 5)
    o += len(EMG_NAMES)
    if "mov" in sig:
        fill(mov_features(sig["mov"], nwin), 8)
    del sig
    t_end = np.arange(nwin) * STEP + WIN
    centre = t_end - WIN / 2
    y = np.zeros(nwin, np.int8)
    for a, b in r["seizures"]:
        y[(centre >= a) & (centre <= b)] = 1
    tmp = cache.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, X=X.astype(np.float16), y=y, t=t_end.astype(np.int32))
    tmp.replace(cache)  # atomic: a half-written file never looks finished
    return r["run"], f"{nwin * STEP / 3600:.1f} h, {len(r['seizures'])} sz, {time.time() - t0:.0f} s", nbytes


def fetch(workers=3):
    runs = build_index()
    todo = [r for r in runs if not (FEAT / f"{r['run']}.npz").exists() and "eeg" in r["files"]]
    total = sum(r["bytes"] for r in todo)
    print(f"{len(runs)} runs, {len({r['sub'] for r in runs})} patients, "
          f"{sum(len(r['seizures']) for r in runs)} seizures; to stream: {len(todo)} runs, {total / 1e9:.1f} GB",
          flush=True)
    bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="streaming", smoothing=0.05)
    failed = []
    with cf.ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(process_run, r): r for r in todo}
        for f in cf.as_completed(futs):
            r = futs[f]
            try:
                name, msg, _ = f.result()
                tqdm.write(f"{name}: {msg}")
            except Exception as e:  # network or corrupt file: skip now, retried on the next run
                failed.append(r["run"])
                tqdm.write(f"{r['run']}: FAILED ({type(e).__name__}: {e}); re-run fetch to retry")
            bar.update(r["bytes"])
    bar.close()
    done = sum((FEAT / f"{r['run']}.npz").exists() for r in runs)
    msg = f"{done}/{len(runs)} runs have features" + (f"; {len(failed)} failed, re-run fetch to retry" if failed else "")
    print("done: " + msg)
    notify("SeizeIT2 streaming DONE", msg + ". Next: uv run detect/seizeit2.py run")


# ------------------------------------------------------------------ evaluation
def normalise(X):
    """Causal per-recording baseline: z-score against an EMA of the PREVIOUS ~10 min (NaN-safe)."""
    x = np.nan_to_num(X.astype(np.float32), nan=0.0)
    a = 1.0 / EMA_N
    n0 = min(30, len(x))
    m0, v0 = x[:n0].mean(0), x[:n0].var(0) + 1e-6
    m = lfilter([a], [1, a - 1], x, axis=0, zi=((1 - a) * m0)[None, :])[0]
    s2 = lfilter([a], [1, a - 1], x ** 2, axis=0, zi=((1 - a) * (v0 + m0 ** 2))[None, :])[0]
    mp = np.vstack([m0[None], m[:-1]])
    vp = np.maximum(np.vstack([(v0 + m0 ** 2)[None], s2[:-1]]) - mp ** 2, 1e-6)
    z = (x - mp) / np.sqrt(vp)
    z[np.isnan(X)] = np.nan  # a missing sensor stays missing (the trees route NaN separately)
    return z


def design(X, groups):
    """Model input: baseline-normalised features of the chosen modalities, plus absolute heart rate
    / beat quality (absolute tachycardia matters, not only the change)."""
    cols = [i for g in groups for i in (NAMES.index(n) for n in GROUPS[g])]
    Z = normalise(X[:, cols])
    if "ecg" in groups:
        Z = np.hstack([Z, X[:, [NAMES.index("hr_10s"), NAMES.index("hr_30s"), NAMES.index("beat_quality")]]])
    return Z


def load(run):
    z = np.load(FEAT / f"{run}.npz")
    return z["X"].astype(np.float32), z["y"], z["t"].astype(float)


def fa_at_sensitivity(curve, target):
    """Fewest false alarms/24 h among thresholds that catch at least `target` of seizures.
    (Sensitivity is not monotonic in the threshold: at low thresholds a false alarm just before a
    seizure starts the 60 s refractory period and blocks the true alarm, so no interpolation.)"""
    ok = [v["fa_per_24h"] for v in curve.values() if v["sensitivity"] >= target]
    return float(min(ok)) if ok else None


def sens_at_fa(curve, budget):
    """Best sensitivity among thresholds with at most `budget` false alarms/24 h."""
    ok = [v["sensitivity"] for v in curve.values() if v["fa_per_24h"] <= budget]
    return float(max(ok)) if ok else None


def run(only=None, folds=5, snapshot=False):
    global RES
    idx = json.loads((FEAT / "index.json").read_text())
    runs = [r for r in idx if (FEAT / f"{r['run']}.npz").exists()]
    if snapshot:  # interim result while fetch is still streaming: complete patients only, separate folder
        RES = RES.parent / "results_seizeit2_partial"
        incomplete = {r["sub"] for r in idx if not (FEAT / f"{r['run']}.npz").exists()}
        runs = [r for r in runs if r["sub"] not in incomplete]
        print(f"SNAPSHOT: {len({r['sub'] for r in runs})} complete patients (fetch still running); results -> {RES.name}/")
    RES.mkdir(exist_ok=True)
    subs = sorted({r["sub"] for r in runs})
    print(f"{len(runs)} runs, {len(subs)} patients, {sum(len(r['seizures']) for r in runs)} seizures, "
          f"{sum(r['duration'] for r in runs) / 3600:.0f} h", flush=True)
    # subject-wise folds, balanced by seizure count (round-robin over subjects sorted by #seizures)
    nsz = {s: sum(len(r["seizures"]) for r in runs if r["sub"] == s) for s in subs}
    fold = {s: i % folds for i, s in enumerate(sorted(subs, key=lambda s: (-nsz[s], zlib.crc32(s.encode()))))}
    thresholds = np.round(np.r_[np.arange(0.05, 0.95, 0.05), 0.95, 0.97, 0.98, 0.99], 3)
    out = json.loads((RES / "metrics.json").read_text()) if (RES / "metrics.json").exists() else {}
    out.setdefault("general", {})
    out.setdefault("per_subject", {})
    out.setdefault("latency", {})
    out["dataset"] = {"runs": len(runs), "patients": len(subs), "seizures": sum(nsz.values()),
                      "hours": sum(r["duration"] for r in runs) / 3600}
    t0 = time.time()
    for cname, groups in CONFIGS.items():
        if only and not any(o.lower() in cname.lower() for o in only):
            continue
        if cname in out["general"] and not only:
            tqdm.write(f"{cname}: already in metrics.json, skipping")
            continue
        rng = np.random.default_rng(zlib.crc32(cname.encode()))
        Xs, ys, fs_ = [], [], []
        for r in tqdm(runs, desc=f"{cname}: loading", unit="run", leave=False):
            X, y, _ = load(r["run"])
            keep = (y == 1) | (rng.random(len(y)) < NEG_FRAC)
            Xs.append(design(X, groups)[keep])
            ys.append(y[keep])
            fs_.append(np.full(keep.sum(), fold[r["sub"]], np.int8))
        Xp, yp, fp = np.vstack(Xs), np.concatenate(ys), np.concatenate(fs_)
        del Xs, ys, fs_
        scored, per = [], {}
        for k in tqdm(range(folds), desc=f"{cname}: {folds}-fold (unseen patients)", unit="fold"):
            m = clf().fit(Xp[fp != k], yp[fp != k])
            for s in subs:
                if fold[s] != k:
                    continue
                srec = []
                for r in runs:
                    if r["sub"] == s:
                        X, _, t = load(r["run"])
                        srec.append((m.predict_proba(design(X, groups))[:, 1].astype(np.float32), t,
                                     [tuple(v) for v in r["seizures"]], r["duration"] / 3600))
                scored += srec
                per[s] = srec
        del Xp, yp, fp
        out["general"][cname] = {f"{th:.2f}": {k: v for k, v in score(scored, th).items() if k != "latencies"}
                                 for th in thresholds}
        out["latency"][cname] = score(scored, 0.5)["latencies"]
        out["per_subject"][cname] = {s: {k: v for k, v in score(v, 0.5).items() if k != "latencies"}
                                     for s, v in per.items()}
        (RES / "metrics.json").write_text(json.dumps(out, indent=1))  # saved per config: survives a stop
        r = out["general"][cname]["0.50"]
        tqdm.write(f"[{cname}] sens {r['sensitivity']:.0%} ({r['detected']}/{r['seizures']}), "
                   f"FA/24h {r['fa_per_24h']:.1f}, latency {r['latency_median_s']} s  ({time.time() - t0:.0f} s)")
        del scored, per
    summary(out)
    figure(out)
    if "matched" in out:
        notify(f"SeizeIT2 {'SNAPSHOT ' if snapshot else ''}results DONE", "FA/24h at 70% caught: " + ", ".join(
            f"{c.split(' (')[0]}: {v['0.7']:.0f}" for c, v in out["matched"].items() if v["0.7"] is not None), "tada")
    print(f"saved {RES / 'metrics.json'} and {RES / 'seizeit2.png'}")


def summary(out):
    g = out["general"]
    if "EEG only (ear)" not in g:
        return
    print("\nFalse alarms / 24 h at MATCHED sensitivity (same seizures caught):")
    print(f"{'setup':32s}" + "".join(f"{f'@{int(t * 100)}%':>10s}" for t in (0.5, 0.6, 0.7, 0.8)))
    out["matched"] = {}
    for c, curve in g.items():
        vals = [fa_at_sensitivity(curve, t) for t in (0.5, 0.6, 0.7, 0.8)]
        out["matched"][c] = dict(zip(("0.5", "0.6", "0.7", "0.8"), vals))
        print(f"{c:32s}" + "".join(f"{v:10.1f}" if v is not None else f"{'-':>10s}" for v in vals))
    print("\nSeizures caught at a FIXED false-alarm budget (per 24 h):")
    print(f"{'setup':32s}" + "".join(f"{f'<={b}':>10s}" for b in (5, 20, 50, 100, 200)))
    out["budget"] = {}
    for c, curve in g.items():
        vals = [sens_at_fa(curve, b) for b in (5, 20, 50, 100, 200)]
        out["budget"][c] = dict(zip(("5", "20", "50", "100", "200"), vals))
        print(f"{c:32s}" + "".join(f"{v:10.0%}" if v is not None else f"{'-':>10s}" for v in vals))
    (RES / "metrics.json").write_text(json.dumps(out, indent=1))


def figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(20, 6.5))
    cols = dict(zip(CONFIGS, ["tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:green"]))
    for c, d in out["general"].items():
        ax[0].plot([v["fa_per_24h"] for v in d.values()], [100 * v["sensitivity"] for v in d.values()],
                   "o-", ms=3, color=cols.get(c), label=c)
    ax[0].set_xscale("symlog", linthresh=1)
    ax[0].set(xlabel="false alarms per 24 h", ylabel="seizures detected (%)",
              title="Unseen patients (subject-wise 5-fold CV)")
    ax[0].legend(fontsize=8)
    if "matched" in out:
        ts = ["0.5", "0.6", "0.7", "0.8"]
        w = 0.8 / len(out["matched"])
        for i, (c, d) in enumerate(out["matched"].items()):
            ax[1].bar(np.arange(4) + i * w, [d[t] or 0 for t in ts], w, color=cols.get(c), label=c)
        ax[1].set_xticks(np.arange(4) + 0.4 - w / 2, [f"{int(float(t) * 100)}% caught" for t in ts])
        ax[1].set(ylabel="false alarms per 24 h", title="False alarms at matched sensitivity (lower = better)")
        ax[1].legend(fontsize=8)
    for c, lat in out["latency"].items():
        if lat:
            ax[2].hist(np.clip(lat, -15, 120), bins=np.arange(-15, 121, 5), histtype="step", lw=2,
                       color=cols.get(c), label=f"{c} (median {np.median(lat):.0f} s)")
    ax[2].set(xlabel="alarm time after EEG onset (s)", ylabel="seizures", title="Detection latency (threshold 0.5)")
    ax[2].legend(fontsize=8)
    fig.suptitle("SeizeIT2: behind-the-ear EEG vs + ECG / EMG / motion", fontsize=14)
    fig.tight_layout()
    fig.savefig(RES / "seizeit2.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ personal fine-tuning
FT_CONFIGS = ["EEG only (ear)", "EEG + ECG"]
PATIENT_W = 10.0  # weight of the user's own windows when fine-tuning (as in the CHB-MIT study)
TUNE_MARGIN = 600  # s after the k-th seizure that still belongs to the tuning period
POST = {  # alarm rule: (smoothing windows, consecutive smoothed windows above threshold); 1 window = 2 s
    "6 s smoothing": (3, 1),
    "sustained 10 s": (3, 5),
    "sustained 20 s": (5, 10),
}
FT_THR = np.round(np.r_[np.arange(0.1, 0.95, 0.05), 0.95, 0.97, 0.98, 0.99, 0.995], 3)


def run_no(r):
    return int(re.search(r"run-(\d+)", r["run"]).group(1))


def subject_folds(runs, folds=5):
    """Same patient-wise folds as run(): round-robin over patients sorted by seizure count."""
    subs = sorted({r["sub"] for r in runs})
    nsz = {s: sum(len(r["seizures"]) for r in runs if r["sub"] == s) for s in subs}
    return subs, {s: i % folds for i, s in enumerate(sorted(subs, key=lambda s: (-nsz[s], zlib.crc32(s.encode()))))}


def split_patient(rs, k):
    """Chronological split of one patient's runs. Tune = everything up to TUNE_MARGIN after the k-th
    seizure; test = everything after (windows that straddle the cut are dropped). None if no test seizure."""
    rs = sorted(rs, key=run_no)
    sz = [(i, a, b) for i, r in enumerate(rs) for a, b in r["seizures"]]
    if len(sz) < k + 1:
        return None
    ri, _, e = sz[k - 1]
    cut = e + TUNE_MARGIN
    if not any(i > ri or (i == ri and a > cut) for i, a, _ in sz):
        return None
    tune = [(r, None) for r in rs[:ri]] + [(rs[ri], cut)]
    test = [(rs[ri], cut)] + [(r, None) for r in rs[ri + 1:]]
    return tune, test


def score_fast(recs, thresholds, smooth=3, persist=1, as_list=False, keep_lat=False):
    """Event-level scoring for many thresholds at once, with the same rules as detect.score()
    (caught = alarm from PRE s before onset to seizure end; alarms up to POST s after a seizure are
    not false; REFRACT s between alarms) plus a persistence rule: the smoothed probability must stay
    above threshold for `persist` consecutive windows before an alarm fires.
    Returns {"0.500": row, ...}, or a list of rows in threshold order if as_list."""
    from detect import POST as POST_S, PRE, REFRACT

    P, T, S, hours, off = [], [], [], 0.0, 0.0
    pad = np.full(persist + smooth, -1.0)
    for p, t, sz, h in recs:
        if not len(t):
            continue
        P += [np.convolve(p, np.ones(smooth) / smooth)[: len(p)], pad]  # causal smoothing, per record
        T += [t.astype(np.float64) + off, np.full(len(pad), np.nan)]
        S += [(a + off, b + off) for a, b in sz]
        hours += h
        off += float(t[-1]) + 10_000  # records never interact
    if not P:  # nothing to score
        P, T = np.zeros(0), np.zeros(0)
    else:
        P, T = np.concatenate(P), np.concatenate(T)
    S = np.array(sorted(S), float).reshape(-1, 2)
    lo, hi = S[:, 0] - PRE, S[:, 1] + POST_S  # merged "not a false alarm" intervals
    mlo, mhi = [], []
    for a, b in zip(lo, hi):
        if mlo and a <= mhi[-1]:
            mhi[-1] = max(mhi[-1], b)
        else:
            mlo.append(a)
            mhi.append(b)
    mlo, mhi = np.array(mlo), np.array(mhi)
    rows = []
    for thr in thresholds:
        above = P > thr
        if persist > 1:
            above = np.convolve(above.astype(np.int32), np.ones(persist, np.int32))[: len(above)] >= persist
        cand = T[above]
        al, i = [], 0
        while i < len(cand):  # refractory: next alarm = first candidate >= last + REFRACT
            al.append(cand[i])
            i = int(np.searchsorted(cand, cand[i] + REFRACT, side="left"))
        al = np.array(al)
        j = np.searchsorted(al, S[:, 0] - PRE, side="left")
        hit = (j < len(al)) & (al[np.minimum(j, max(len(al) - 1, 0))] <= S[:, 1]) if len(al) else np.zeros(len(S), bool)
        lat = (al[j[hit]] - S[hit, 0]) if len(al) else np.array([])
        m = np.searchsorted(mlo, al, side="right") - 1
        covered = (m >= 0) & (al <= mhi[np.maximum(m, 0)]) if len(al) and len(mhi) else np.zeros(len(al), bool)
        fa = int((~covered).sum())
        row = {"sensitivity": float(hit.mean()) if len(S) else 0.0, "detected": int(hit.sum()),
               "seizures": int(len(S)), "fa_per_24h": fa / max(hours, 1e-9) * 24, "false_alarms": fa,
               "hours": hours, "latency_median_s": float(np.median(lat)) if len(lat) else None}
        if keep_lat:
            row["latencies"] = [float(v) for v in lat]
        rows.append(row)
    return rows if as_list else {f"{thr:.3f}": r for thr, r in zip(thresholds, rows)}


def finetune(configs=None, ks=(1, 2), folds=5):
    """General model vs general + the user's own first k seizures, on the same later data."""
    RES.mkdir(exist_ok=True)
    path = RES / "finetune.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    runs = [r for r in json.loads((FEAT / "index.json").read_text()) if (FEAT / f"{r['run']}.npz").exists()]
    subs, fold = subject_folds(runs, folds)
    by = {s: [r for r in runs if r["sub"] == s] for s in subs}
    t0 = time.time()
    names = FT_CONFIGS if not configs else [c for c in CONFIGS if any(o.lower() in c.lower() for o in configs)]
    for cname in names:
        groups = CONFIGS[cname]
        todo = [k for k in ks if f"{cname} | k={k}" not in out]
        if not todo:
            tqdm.write(f"{cname}: fine-tuning already in finetune.json, skipping")
            continue
        rng = np.random.default_rng(zlib.crc32(cname.encode()))  # same training pool as run()
        Xs, ys, fs_ = [], [], []
        for r in tqdm(runs, desc=f"{cname}: loading pool", unit="run", leave=False):
            X, y, _ = load(r["run"])
            keep = (y == 1) | (rng.random(len(y)) < NEG_FRAC)
            Xs.append(design(X, groups)[keep])
            ys.append(y[keep])
            fs_.append(np.full(keep.sum(), fold[r["sub"]], np.int8))
        Xp, yp, fp = np.vstack(Xs), np.concatenate(ys), np.concatenate(fs_)
        del Xs, ys, fs_
        gen = {f: clf().fit(Xp[fp != f], yp[fp != f]) for f in tqdm(range(folds), desc=f"{cname}: general models", unit="fold")}
        for k in todo:
            recs = {"general": [], "finetuned": []}
            elig = [s for s in subs if split_patient(by[s], k)]
            for s in tqdm(elig, desc=f"{cname} | k={k}: fine-tuning", unit="patient"):
                tune, test = split_patient(by[s], k)
                Xt, yt = [], []
                for r, cut in tune:
                    X, y, t = load(r["run"])
                    sel = np.ones(len(t), bool) if cut is None else (t <= cut)
                    Xt.append(design(X, groups)[sel])
                    yt.append(y[sel])
                Xt, yt = np.vstack(Xt), np.concatenate(yt)
                tr = fp != fold[s]  # exactly the general model's training data ...
                w = np.r_[np.ones(tr.sum()), np.full(len(yt), PATIENT_W)]  # ... plus the user's own, upweighted
                fin = clf().fit(np.vstack([Xp[tr], Xt]), np.r_[yp[tr], yt], sample_weight=w)
                for r, cut in test:
                    X, _, t = load(r["run"])
                    sel = np.ones(len(t), bool) if cut is None else (t - WIN > cut)
                    if not sel.any():
                        continue
                    Z = design(X, groups)[sel]
                    szs = [tuple(v) for v in r["seizures"] if cut is None or v[0] > cut]
                    h = sel.sum() * STEP / 3600
                    for name, mdl in (("general", gen[fold[s]]), ("finetuned", fin)):
                        recs[name].append((mdl.predict_proba(Z)[:, 1].astype(np.float32), t[sel].astype(np.int32), szs, h))
            key = f"{cname} | k={k}"
            out[key] = {"patients": len(elig),
                        **{name: {pp: score_fast(v, FT_THR, *rule) for pp, rule in POST.items()} for name, v in recs.items()}}
            path.write_text(json.dumps(out, indent=1))  # saved per block: a stop loses at most one block
            ft_summary({key: out[key]})
            tqdm.write(f"({time.time() - t0:.0f} s)")
            del recs
        del Xp, yp, fp, gen
    ft_summary(out)
    ft_figure(out)
    best = []
    for key, d in out.items():
        v = fa_at_sensitivity(d["finetuned"]["sustained 10 s"], 0.6)
        best.append(f"{key.split(' (')[0]}: {v:.0f}" if v is not None else f"{key}: -")
    notify("SeizeIT2 fine-tuning DONE", "FA/24h at 60% caught (fine-tuned, 10 s rule): " + "; ".join(best), "tada")
    print(f"saved {path} and {RES / 'finetune.png'}")


def ft_summary(out):
    for key, d in out.items():
        print(f"\n=== {key}  ({d['patients']} patients; tested only on data AFTER their first seizures)")
        print(f"{'model | alarm rule':42s}" + "".join(f"{f'@{t}%':>8s}" for t in (40, 50, 60, 70))
              + "".join(f"{f'<={b}/d':>8s}" for b in (1, 5, 20)) + f"{'delay':>8s}")
        for name in ("general", "finetuned"):
            for pp, curve in d[name].items():
                fa = [fa_at_sensitivity(curve, t / 100) for t in (40, 50, 60, 70)]
                se = [sens_at_fa(curve, b) for b in (1, 5, 20)]
                ok = [v for v in curve.values() if v["sensitivity"] >= 0.6 and v["latency_median_s"] is not None]
                lat = min(ok, key=lambda v: v["fa_per_24h"])["latency_median_s"] if ok else None
                print(f"{name + ' | ' + pp:42s}" + "".join(f"{v:8.1f}" if v is not None else f"{'-':>8s}" for v in fa)
                      + "".join(f"{v:8.0%}" if v is not None else f"{'-':>8s}" for v in se)
                      + (f"{lat:7.0f}s" if lat is not None else f"{'-':>8s}"))


def ft_figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(out)
    fig, ax = plt.subplots(1, len(keys), figsize=(6.5 * len(keys), 5.5), squeeze=False)
    for a, key in zip(ax[0], keys):
        for name, ls in (("general", ":"), ("finetuned", "-")):
            for (pp, curve), c in zip(out[key][name].items(), ["tab:blue", "tab:orange", "tab:green"]):
                a.plot([v["fa_per_24h"] for v in curve.values()], [100 * v["sensitivity"] for v in curve.values()],
                       ls, color=c, marker="o", ms=2.5, label=f"{name}, {pp}")
        a.set_xscale("symlog", linthresh=1)
        a.axvline(1, color="grey", lw=0.8, ls="--")
        a.set(xlabel="false alarms per 24 h", ylabel="seizures detected (%)", title=f"{key}\n({out[key]['patients']} patients)")
        a.legend(fontsize=7)
    fig.suptitle("SeizeIT2: personal fine-tuning + sustained-alarm rules (tested on each patient's later data)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RES / "finetune.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ personal alarm thresholds
PROBS = FEAT.parent / "probs"  # cached model outputs (gitignored with the rest of data/)
PT_BUDGETS = [0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200]  # target false alarms/day on the TUNING data
PT_GRID = np.unique(np.round(np.r_[np.linspace(0.02, 0.9, 45), 1 - np.logspace(-1, -3.3, 40)], 5))
MIN_TUNE_H = 6.0  # less tuning data than this -> fall back to the shared threshold


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def collect_probs(cname, k, folds=5):
    """General and fine-tuned model outputs on every eligible patient's tuning AND test data.
    Same models and split as finetune(); cached so threshold experiments need no retraining."""
    import pickle

    PROBS.mkdir(parents=True, exist_ok=True)
    cache = PROBS / f"{slug(cname)}_k{k}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    groups = CONFIGS[cname]
    runs = [r for r in json.loads((FEAT / "index.json").read_text()) if (FEAT / f"{r['run']}.npz").exists()]
    subs, fold = subject_folds(runs, folds)
    by = {s: [r for r in runs if r["sub"] == s] for s in subs}
    rng = np.random.default_rng(zlib.crc32(cname.encode()))
    Xs, ys, fs_ = [], [], []
    for r in tqdm(runs, desc=f"{cname}: loading pool", unit="run", leave=False):
        X, y, _ = load(r["run"])
        keep = (y == 1) | (rng.random(len(y)) < NEG_FRAC)
        Xs.append(design(X, groups)[keep])
        ys.append(y[keep])
        fs_.append(np.full(keep.sum(), fold[r["sub"]], np.int8))
    Xp, yp, fp = np.vstack(Xs), np.concatenate(ys), np.concatenate(fs_)
    del Xs, ys, fs_
    gen = {f: clf().fit(Xp[fp != f], yp[fp != f]) for f in tqdm(range(folds), desc=f"{cname}: general models", unit="fold")}
    data = {}
    for s in tqdm([s for s in subs if split_patient(by[s], k)], desc=f"{cname} | k={k}: models + outputs", unit="patient"):
        tune, test = split_patient(by[s], k)
        parts = {"tune": [], "test": []}
        for which, lst in (("tune", tune), ("test", test)):
            for r, cut in lst:
                X, y, t = load(r["run"])
                if cut is None:
                    sel, szs = np.ones(len(t), bool), [tuple(v) for v in r["seizures"]]
                elif which == "tune":
                    sel, szs = t <= cut, [tuple(v) for v in r["seizures"] if v[0] <= cut]
                else:
                    sel, szs = t - WIN > cut, [tuple(v) for v in r["seizures"] if v[0] > cut]
                if sel.any():
                    parts[which].append((design(X, groups)[sel], y[sel], t[sel].astype(np.int32), szs, sel.sum() * STEP / 3600))
        Xt = np.vstack([z for z, *_ in parts["tune"]])
        yt = np.concatenate([y for _, y, *_ in parts["tune"]])
        tr = fp != fold[s]
        w = np.r_[np.ones(tr.sum()), np.full(len(yt), PATIENT_W)]
        fin = clf().fit(np.vstack([Xp[tr], Xt]), np.r_[yp[tr], yt], sample_weight=w)
        data[s] = {which: {name: [(m.predict_proba(z)[:, 1].astype(np.float32), t, szs, h) for z, _, t, szs, h in parts[which]]
                           for name, m in (("general", gen[fold[s]]), ("finetuned", fin))}
                   for which in ("tune", "test")}
    cache.write_bytes(pickle.dumps(data))
    return data


def personal(configs=None, k=2):
    """Shared vs personal alarm threshold, both calibrated ONLY on tuning data, scored on later data.
    Personal: per patient, the lowest threshold whose false alarms on that patient's own tuning data
    stay within the budget. Shared: the same rule on everyone's tuning data pooled."""
    path = RES / "personal_threshold.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    names = FT_CONFIGS if not configs else [c for c in CONFIGS if any(o.lower() in c.lower() for o in configs)]
    for cname in names:
        key = f"{cname} | k={k}"
        data = collect_probs(cname, k)
        res = {"patients": len(data)}
        for model in ("general", "finetuned"):
            tune = {}
            for s, d in tqdm(data.items(), desc=f"{key} | {model}: calibrating", unit="patient", leave=False):
                rows = score_fast(d["tune"][model], PT_GRID, as_list=True)
                tune[s] = (sum(r[3] for r in d["tune"][model]), np.array([r["false_alarms"] for r in rows]))
            H = sum(h for h, _ in tune.values())
            pooled = sum(fa for _, fa in tune.values()) * 24 / max(H, 1e-9)

            def pick(fa_day, budget):  # lowest threshold within budget (most sensitive), else the highest
                ok = np.flatnonzero(fa_day <= budget)
                return float(PT_GRID[ok[0]]) if len(ok) else float(PT_GRID[-1])

            res[model] = {}
            for mode in ("shared", "personal"):
                res[model][mode] = {}
                for B in PT_BUDGETS:
                    shared_thr = pick(pooled, B)
                    det = nsz = fa = 0
                    hours, lats, thrs, fallback = 0.0, [], [], 0
                    for s, d in data.items():
                        h_t, fa_t = tune[s]
                        if mode == "shared":
                            thr = shared_thr
                        elif h_t < MIN_TUNE_H:
                            thr, fallback = shared_thr, fallback + 1
                        else:
                            thr = pick(fa_t * 24 / h_t, B)
                        r = score_fast(d["test"][model], [thr], as_list=True, keep_lat=True)[0]
                        det, nsz, fa, hours = det + r["detected"], nsz + r["seizures"], fa + r["false_alarms"], hours + r["hours"]
                        lats += r["latencies"]
                        thrs.append(thr)
                    res[model][mode][str(B)] = {
                        "sensitivity": det / max(nsz, 1), "detected": det, "seizures": nsz, "fa_per_24h": fa * 24 / max(hours, 1e-9),
                        "latency_median_s": float(np.median(lats)) if lats else None,
                        "threshold_median": float(np.median(thrs)), "threshold_iqr": [float(np.percentile(thrs, 25)), float(np.percentile(thrs, 75))],
                        "fallback_patients": fallback}
        out[key] = res
        path.write_text(json.dumps(out, indent=1))
        pt_summary({key: res})
    pt_figure(out)
    lines = []
    for key, res in out.items():
        for model in ("general", "finetuned"):
            sh = pt_frontier(res[model]["shared"], 5)
            pe = pt_frontier(res[model]["personal"], 5)
            lines.append(f"{key.split(' (')[0].split(' |')[0]} {model}: {sh:.0%}->{pe:.0%}" if sh is not None and pe is not None else key)
    notify("SeizeIT2 personal thresholds DONE", "Caught at <=5 FA/day, shared->personal: " + "; ".join(lines), "tada")
    print(f"saved {path} and {RES / 'personal_threshold.png'}")


def pt_frontier(curve, fa_max):
    """Best test sensitivity among budget settings whose TEST false alarms stay <= fa_max/day."""
    ok = [v["sensitivity"] for v in curve.values() if v["fa_per_24h"] <= fa_max]
    return max(ok) if ok else None


def pt_summary(out):
    for key, res in out.items():
        print(f"\n=== {key}  ({res['patients']} patients; thresholds set on tuning data, scored on later data)")
        print(f"{'model':10s}{'budget/day':>11s} |{'SHARED: caught':>16s}{'FA/day':>8s}{'thr':>7s} |{'PERSONAL: caught':>18s}{'FA/day':>8s}{'thr (median)':>14s}{'delay':>7s}")
        for model in ("general", "finetuned"):
            for B in PT_BUDGETS:
                a, b = res[model]["shared"][str(B)], res[model]["personal"][str(B)]
                print(f"{model:10s}{B:>11g} |{a['sensitivity']:>16.0%}{a['fa_per_24h']:>8.1f}{a['threshold_median']:>7.3f} |"
                      f"{b['sensitivity']:>18.0%}{b['fa_per_24h']:>8.1f}{b['threshold_median']:>14.3f}"
                      + (f"{b['latency_median_s']:>6.0f}s" if b["latency_median_s"] is not None else f"{'-':>7s}"))
        print("Seizures caught with at most X false alarms/day on the LATER data (best budget setting):")
        for model in ("general", "finetuned"):
            cells = []
            for X in (1, 5, 20):
                sh, pe = pt_frontier(res[model]["shared"], X), pt_frontier(res[model]["personal"], X)
                f = lambda v: f"{v:.0%}" if v is not None else "-"
                cells.append(f"<={X}/day: shared {f(sh)} vs personal {f(pe)}")
            print(f"  {model:10s} " + " | ".join(cells))


def pt_figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(out)
    fig, ax = plt.subplots(1, len(keys), figsize=(7 * len(keys), 5.5), squeeze=False)
    for a, key in zip(ax[0], keys):
        for model, c in (("general", "tab:grey"), ("finetuned", "tab:green")):
            for mode, ls in (("shared", ":"), ("personal", "-")):
                cur = out[key][model][mode]
                a.plot([v["fa_per_24h"] for v in cur.values()], [100 * v["sensitivity"] for v in cur.values()],
                       ls, color=c, marker="o", ms=3, label=f"{model}, {mode} threshold")
        a.set_xscale("symlog", linthresh=1)
        a.axvline(1, color="grey", lw=0.8, ls="--")
        a.set(xlabel="false alarms per 24 h (later data)", ylabel="seizures detected (%)",
              title=f"{key}\n({out[key]['patients']} patients)")
        a.legend(fontsize=8)
    fig.suptitle("SeizeIT2: shared vs personal alarm threshold (set on tuning data only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RES / "personal_threshold.png", dpi=110)
    plt.close(fig)


def main(a):
    if a.cmd == "fetch":
        fetch(a.workers)
    elif a.cmd == "test":  # smoke test on the smallest runs
        idx = build_index()
        for r in sorted([r for r in idx if r["seizures"] and len(r["files"]) == 4], key=lambda r: r["bytes"])[:2]:
            print(process_run(r))
            X, y, t = load(r["run"])
            print(X.shape, "seizure windows", y.sum())
            for i, n in enumerate(NAMES):
                v = X[:, i]
                print(f"  {n:22s} nan {np.isnan(v).mean():5.0%}  median {np.nanmedian(v):9.3f}  "
                      f"p5 {np.nanpercentile(v, 5):9.3f}  p95 {np.nanpercentile(v, 95):9.3f}")
    elif a.cmd == "report":  # recompute tables + figure from saved metrics (no retraining)
        global RES
        if a.snapshot:
            RES = RES.parent / "results_seizeit2_partial"
        out = json.loads((RES / "metrics.json").read_text())
        summary(out)
        figure(out)
    elif a.cmd == "finetune":
        finetune(a.configs, tuple(a.k))
    elif a.cmd == "personal":
        for k in (a.k if a.k != [1, 2] else [2]):  # default: tune on 2 own seizures
            personal(a.configs, k)
    else:
        run(a.configs, snapshot=a.snapshot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "run", "report", "finetune", "personal", "test"])
    ap.add_argument("--workers", type=int, default=3, help="runs streamed in parallel (each uses 4 connections)")
    ap.add_argument("--configs", nargs="*", default=None, help="re-run only configs containing these words")
    ap.add_argument("--snapshot", action="store_true",
                    help="run: interim results on the patients streamed so far (separate folder; fetch can keep running)")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2],
                    help="finetune: number of the user's own seizures used for tuning (default: 1 2)")
    a = ap.parse_args()
    try:
        main(a)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        notify(f"SeizeIT2 {a.cmd} CRASHED", f"{type(e).__name__}: {e}"[:200], "rotating_light")
        raise

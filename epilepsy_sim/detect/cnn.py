"""Deep learning on the RAW behind-the-ear EEG (SeizeIT2): does a CNN find information the hand-made
features miss?

Steps
  fetch  Stream every run's EEG EDF from OpenNeuro into RAM (no EDF on disk), band-pass 0.5-14 Hz,
         downsample 256 -> 32 Hz, store int16 (0.1 uV steps) in data/seizeit2/raw32/ (~5.4 GB).
  train  1D CNN on 16 s of 2-channel EEG ending at each 2-s decision point (same window grid as the
         features). Patient-wise 5-fold CV with the SAME folds as seizeit2.py; every run's CNN output
         comes from a network that never saw that patient -> data/seizeit2/cnn/<run>.npy (float16).
  eval   Score the CNN alone like the other detectors (matched sensitivity, false-alarm budgets).
The CNN output is then used as an extra input to the feature model: `seizeit2.py run --configs CNN`.

Usage (from epilepsy_sim/; needs the optional extra: uv sync --extra dl):
    uv run --extra dl detect/cnn.py fetch     # ~40-60 min (47.6 GB streamed)
    uv run --extra dl detect/cnn.py train     # ~1-2 h on CPU
    uv run --extra dl detect/cnn.py eval
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seizeit2 as S  # noqa: E402

RAW = S.FEAT.parent / "raw32"
OUT = S.FEAT.parent / "cnn"
FS = 32  # Hz after downsampling
CTX_S = 16  # s of EEG the CNN sees for each decision
SCALE = 10.0  # int16 units per uV (0.1 uV resolution, +-3276 uV range)
NEG_FRAC = 0.015
MIN_FREE_GB = 1.5  # fetch refuses to write when free disk space drops below this
CLIP = 50.0  # normalised EEG is clipped to +-50 x the recording's median |amplitude|
EPOCHS = 6
BATCH = 256


# ------------------------------------------------------------------ fetch raw EEG
def fetch_one(r):
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{r['run']}.npy"
    if out.exists():
        return r["run"], "cached"
    import shutil
    if shutil.disk_usage(RAW).free < MIN_FREE_GB * 1e9:  # never fill the drive
        raise RuntimeError(f"disk nearly full (<{MIN_FREE_GB} GB free); stopped")
    buf = S.get(f"{S.S3}/{r['files']['eeg']}")
    sig = S.read_edf(buf)
    del buf
    chans = list(sig.values())[:2]
    fs = chans[0][1]
    n = min(len(c[0]) for c in chans)
    sos = butter(4, [0.5, 14], btype="band", fs=fs, output="sos")
    step = int(round(fs / FS))
    x = np.stack([sosfiltfilt(sos, c[0][:n].astype(np.float64))[::step] for c in chans])
    x = np.clip(np.round(x * SCALE), -32767, 32767).astype(np.int16)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, x)
    tmp.replace(out)
    return r["run"], f"{x.shape[1] / FS / 3600:.1f} h"


def fetch(workers=2):
    RAW.mkdir(parents=True, exist_ok=True)
    runs = [r for r in json.loads((S.FEAT / "index.json").read_text()) if "eeg" in r["files"]]
    todo = [r for r in runs if not (RAW / f"{r['run']}.npy").exists()]
    print(f"{len(runs)} runs, {len(todo)} to stream", flush=True)
    failed = []
    with cf.ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_one, r): r for r in todo}
        for f in tqdm(cf.as_completed(futs), total=len(futs), desc="raw EEG", unit="run"):
            try:
                f.result()
            except Exception as e:  # network / corrupt: retried on the next run
                failed.append(futs[f]["run"])
                tqdm.write(f"{futs[f]['run']}: FAILED ({type(e).__name__}: {e})")
    done = sum((RAW / f"{r['run']}.npy").exists() for r in runs)
    msg = f"{done}/{len(runs)} runs of raw EEG" + (f"; {len(failed)} failed, re-run to retry" if failed else "")
    print(msg)
    S.notify("SeizeIT2 raw EEG streaming DONE", msg)


# ------------------------------------------------------------------ data
def windows_of(run):
    """Raw EEG (float32, uV / per-recording robust scale) and the feature-window grid of this run."""
    x = np.load(RAW / f"{run}.npy").astype(np.float32) / SCALE
    z = np.load(S.FEAT / f"{run}.npz")
    t, y = z["t"].astype(np.int64), z["y"]
    # per-recording, per-channel robust scale. Floor at 1 uV: some runs have flat (all-zero) channels, and
    # dividing by ~0 gave values ~3e6 that overflowed float16 -> inf -> NaN loss. Clip huge artefacts too.
    scale = np.maximum(np.median(np.abs(x), axis=1, keepdims=True), 1.0)
    return np.clip(x / scale, -CLIP, CLIP), t, y


def cut(x, t_end):
    """(n, 2, CTX_S*FS) windows ending at each t_end (zero-padded at the start of the recording)."""
    L = CTX_S * FS
    pad = np.concatenate([np.zeros((2, L), np.float32), x], axis=1)
    ends = np.minimum(t_end * FS, x.shape[1]) + L
    idx = ends[:, None] - L + np.arange(L)[None, :]
    return np.stack([pad[0][idx], pad[1][idx]], 1)


def model():
    import torch
    import torch.nn as nn

    def block(i, o, k, s):
        return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.ReLU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(block(2, 16, 7, 2), block(16, 32, 7, 2), block(32, 64, 7, 2), block(64, 64, 5, 2))
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

        def forward(self, x):
            h = self.f(x)
            return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1)).squeeze(1)

    return Net()


def train(folds=5):
    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [r for r in json.loads((S.FEAT / "index.json").read_text()) if (RAW / f"{r['run']}.npy").exists()]
    subs, fold = S.subject_folds(runs, folds)
    rng = np.random.default_rng(0)
    # training pool: every seizure window + NEG_FRAC of the rest, with their fold
    Xs, ys, fs_ = [], [], []
    for r in tqdm(runs, desc="building training windows", unit="run"):
        x, t, y = windows_of(r["run"])
        keep = (y == 1) | (rng.random(len(y)) < NEG_FRAC)
        if keep.any():
            Xs.append(cut(x, t[keep]).astype(np.float16))
            ys.append(y[keep])
            fs_.append(np.full(keep.sum(), fold[r["sub"]], np.int8))
    Xp, yp, fp = np.concatenate(Xs), np.concatenate(ys).astype(np.float32), np.concatenate(fs_)
    del Xs
    assert np.isfinite(Xp).all(), "non-finite values in the CNN training pool"
    print(f"pool: {len(yp)} windows ({int(yp.sum())} seizure), {Xp.nbytes / 1e9:.2f} GB", flush=True)
    log = {}
    for k in range(folds):
        done_runs = [r for r in runs if fold[r["sub"]] == k]
        if all((OUT / f"{r['run']}.npy").exists() for r in done_runs):
            print(f"fold {k}: outputs exist, skipping", flush=True)
            continue
        tr = np.flatnonzero(fp != k)
        pos_w = torch.tensor((len(tr) - yp[tr].sum()) / max(yp[tr].sum(), 1))
        torch.manual_seed(k)
        net = model()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * (len(tr) // BATCH + 1))
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        t0 = time.time()
        for ep in range(EPOCHS):
            net.train()
            perm = rng.permutation(tr)
            tot = 0.0
            for b in tqdm(range(0, len(perm), BATCH), desc=f"fold {k} epoch {ep + 1}/{EPOCHS}", unit="batch", leave=False):
                ii = np.sort(perm[b:b + BATCH])
                xb = torch.from_numpy(Xp[ii].astype(np.float32))
                xb = xb * torch.empty(len(ii), 1, 1).uniform_(0.8, 1.25)  # amplitude augmentation
                loss = lossf(net(xb), torch.from_numpy(yp[ii]))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss in fold {k} epoch {ep + 1} batch {b // BATCH}")
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss) * len(ii)
            tqdm.write(f"fold {k} epoch {ep + 1}: loss {tot / len(tr):.4f} ({time.time() - t0:.0f} s)")
        log[k] = {"train_windows": int(len(tr)), "seconds": time.time() - t0}
        # out-of-fold outputs for every window of the held-out patients
        net.eval()
        with torch.no_grad():
            for r in tqdm(done_runs, desc=f"fold {k}: scoring held-out runs", unit="run"):
                x, t, _ = windows_of(r["run"])
                p = np.empty(len(t), np.float32)
                for a in range(0, len(t), 4096):
                    xb = torch.from_numpy(cut(x, t[a:a + 4096]))
                    p[a:a + 4096] = torch.sigmoid(net(xb)).numpy()
                tmp = OUT / f"{r['run']}.tmp.npy"
                np.save(tmp, p.astype(np.float16))
                tmp.replace(OUT / f"{r['run']}.npy")
        (S.RES / "cnn_train_log.json").write_text(json.dumps(log, indent=1))
    S.notify("SeizeIT2 CNN training DONE", f"{len(runs)} runs scored out-of-fold")


def evaluate():
    runs = [r for r in json.loads((S.FEAT / "index.json").read_text()) if (OUT / f"{r['run']}.npy").exists()]
    recs = []
    for r in runs:
        t = np.load(S.FEAT / f"{r['run']}.npz")["t"]
        recs.append((np.load(OUT / f"{r['run']}.npy").astype(np.float32), t, [tuple(v) for v in r["seizures"]],
                     r["duration"] / 3600))
    curve = S.score_fast(recs, S.PT_GRID)
    fa = {str(t): S.fa_at_sensitivity(curve, t / 100) for t in (50, 60, 70, 80)}
    se = {str(b): S.sens_at_fa(curve, b) for b in (1, 5, 20, 50, 100, 200)}
    print(f"CNN alone (raw ear EEG, {len(runs)} runs): FA/24h at 50/60/70/80% caught: {fa}")
    print(f"caught at <=1/5/20/50/100/200 FA/day: {se}")
    (S.RES / "cnn_eval.json").write_text(json.dumps({"fa_at_sens": fa, "sens_at_budget": se}, indent=1))


CMP_CONFIGS = ["EEG only (ear)", "EEG + ECG + context + HR-excess"]


def feature_oof(runs, fold, cname, folds=5):
    """Out-of-fold outputs of a feature model trained on exactly these runs and folds (as run())."""
    groups = S.CONFIGS[cname]
    rng = S.pool_rng(cname)
    Xs, ys, fs_ = [], [], []
    for r in tqdm(runs, desc=f"{cname}: loading", unit="run", leave=False):
        X, y, _ = S.load(r["run"])
        keep = (y == 1) | (rng.random(len(y)) < S.neg_frac(cname))
        Xs.append(S.design(X, groups)[keep])
        ys.append(y[keep])
        fs_.append(np.full(keep.sum(), fold[r["sub"]], np.int8))
    Xp, yp, fp = np.vstack(Xs), np.concatenate(ys), np.concatenate(fs_)
    out = {}
    for k in tqdm(range(folds), desc=f"{cname}: folds", unit="fold"):
        m = S.model_for(cname).fit(Xp[fp != k], yp[fp != k])
        for r in runs:
            if fold[r["sub"]] == k:
                X, _, _ = S.load(r["run"])
                out[r["run"]] = m.predict_proba(S.design(X, groups))[:, 1].astype(np.float32)
    return out


def compare(folds=5):
    """CNN vs hand-made features on the SAME runs, patients and folds (the CNN's subset), plus a
    late fusion (mean of the two probabilities: no learned weights, so no label leakage)."""
    from sklearn.metrics import roc_auc_score

    runs = [r for r in json.loads((S.FEAT / "index.json").read_text()) if (OUT / f"{r['run']}.npy").exists()]
    subs, fold = S.subject_folds(runs, folds)
    probs = {"CNN (raw ear EEG)": {r["run"]: np.load(OUT / f"{r['run']}.npy").astype(np.float32) for r in runs}}
    for c in CMP_CONFIGS:
        probs[c] = feature_oof(runs, fold, c, folds)
        probs[f"{c} + CNN (mean)"] = {k: (v + probs["CNN (raw ear EEG)"][k]) / 2 for k, v in probs[c].items()}
    meta = {"runs": len(runs), "patients": len(subs), "seizures": sum(len(r["seizures"]) for r in runs),
            "hours": round(sum(r["duration"] for r in runs) / 3600, 1)}
    res = {"subset": meta, "results": {}}
    print(f"subset: {meta}")
    for name, p in probs.items():
        recs, yy, pp = [], [], []
        for r in runs:
            z = np.load(S.FEAT / f"{r['run']}.npz")
            recs.append((p[r["run"]], z["t"], [tuple(v) for v in r["seizures"]], r["duration"] / 3600))
            yy.append(z["y"])
            pp.append(p[r["run"]])
        curve = S.score_fast(recs, S.PT_GRID)
        yy, pp = np.concatenate(yy), np.concatenate(pp)
        row = {"window_auroc": float(roc_auc_score(yy, pp)),
               "fa_at_sens": {str(t): S.fa_at_sensitivity(curve, t / 100) for t in (50, 60, 70, 80)},
               "sens_at_budget": {str(b): S.sens_at_fa(curve, b) for b in (1, 5, 20, 50, 100, 200)}}
        res["results"][name] = row
        fa = " / ".join("--" if v is None else f"{v:.0f}" for v in row["fa_at_sens"].values())
        se = " / ".join("--" if v is None else f"{v:.0%}" for v in row["sens_at_budget"].values())
        print(f"{name:45s} AUROC {row['window_auroc']:.3f} | FA/day @50/60/70/80%: {fa} | caught @<=1/5/20/50/100/200: {se}")
    (S.RES / "cnn_subset.json").write_text(json.dumps(res, indent=1))
    S.notify("SeizeIT2 CNN comparison DONE", f"{meta['patients']} patients; see cnn_subset.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "train", "eval", "compare"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    S.keep_awake()
    try:
        {"fetch": lambda: fetch(a.workers), "train": train, "eval": evaluate, "compare": compare}[a.cmd]()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        S.notify(f"SeizeIT2 CNN {a.cmd} CRASHED", f"{type(e).__name__}: {e}"[:200], "rotating_light")
        raise

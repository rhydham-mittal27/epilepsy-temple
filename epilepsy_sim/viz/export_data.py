"""Pack real project data into viz/data.js for the live HTML visualisation.

Sources (nothing synthetic is invented here):
  * output/ionic_{epileptic,healthy}.npz   ionic.py simulation (run with --save)
  * data/chbmit/chb01/chb01_03.edf           real scalp EEG, seizure at 2996-3036 s (CHB-MIT)
  * data/chbmit/features/...                 detector features for that recording
  * detect/results/partial_1.npz             detector probabilities, "Both temples (4 ch)" model
                                             (patient-specific, leave-one-recording-out)
  * infer.py                                 Kalman-filter K+ estimate, chloride inference

Usage:
    uv run viz/export_data.py
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detect"))
import ionic  # noqa: E402
import infer  # noqa: E402

FPS = 25  # model frames per second in the export
OUT = Path(__file__).resolve().parent / "data.js"


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, np.uint8).tobytes()).decode()


def rnd(x, n=3):
    return [round(float(v), n) for v in x]


def block(x, k):
    n = len(x) // k
    return x[: n * k].reshape(n, k, *x.shape[1:]).mean(axis=1)


def model(name):
    z = np.load(ROOT / "output" / f"ionic_{name}.npz")
    rec = z["rec"]
    t = rec[:, 0] / 1000
    k = 1000 // FPS  # rec is 1 ms
    stim_on, stim_off = float(z["stim_on"]), float(z["stim_off"])
    rate_s = np.convolve(rec[:, 8], np.ones(1000) / 1000, mode="same")
    base = max(rate_s[t < stim_on].mean(), 0.1)
    cbf, hbo, hbr = ionic.balloon(rate_s, 0.001, base, float(z["nv_gain"]))
    series = {
        "lfp": rec[:, 1], "ko": rec[:, 2], "nai": rec[:, 3], "cli": rec[:, 4], "egaba": rec[:, 5],
        "ek": rec[:, 6], "pump": rec[:, 7], "rate": np.convolve(rec[:, 8], np.ones(200) / 200, mode="same"),
        "ado": rec[:, 9], "cbf": cbf, "hbo": hbo, "hbr": hbr,
    }
    out = {k_: rnd(block(v, k), 3) for k_, v in series.items()}
    # LFP at 200 Hz for a crisp trace
    out["lfp200"] = rnd(block(rec[:, 1], 5), 2)
    n = len(out["ko"])
    # spatial K+ map (recorded every 10 ms) -> FPS, quantised 3..13 mM -> 0..255
    km = z["ko_map"]
    km = block(km, 100 // FPS)[:n]
    out["komap"] = b64(np.clip((km - 3.0) / 10.0 * 255, 0, 255))
    # per-cell spike counts per frame (0..15)
    sp_t, sp_i = z["sp_t"], z["sp_i"]
    ncell = km.shape[1]
    cnt = np.zeros((n, ncell), np.int32)
    fi = (sp_t / 1000 * FPS).astype(int)
    ok = fi < n
    np.add.at(cnt, (fi[ok], sp_i[ok]), 1)
    out["fire"] = b64(np.clip(cnt, 0, 15))
    out.update(n=n, ncell=int(ncell), inh=[int(v) for v in z["is_inh"]], stim_on=stim_on, stim_off=stim_off,
               kcc2=float(z["kcc2"]), glia=float(z["glia"]), clear=float(z["clear"]))
    return out


def inference(he_name="healthy", ep_name="epileptic"):
    he, ep = infer.load(he_name), infer.load(ep_name)
    res, (t, ko, naive, ol, ekf, dc) = infer.e2_dc_ekf(he, ep)  # 10 ms grid
    k = 100 // FPS
    r3, (x, cl_true, inv) = infer.e3_chloride(ep)  # 1 Hz
    return {"ko_ekf": rnd(block(ekf, k), 3), "ko_naive": rnd(block(naive, k), 3), "dc_uV": rnd(block(dc, k), 1),
            "cl_inferred_1hz": rnd(inv["chord (matches model)"], 2),
            "ekf_rmse": res["ekf_rmse_mM"], "naive_rmse": res["naive_rmse_mM"]}


def eeg():
    import mne
    from scipy.signal import butter, sosfiltfilt, resample_poly
    from detect import resolve, alarms_from_prob, ALL_CH

    fname = "chb01_03.edf"
    onset, end = 2996, 3036
    t0, t1 = onset - 70, onset + 110
    raw = mne.io.read_raw_edf(ROOT / "data" / "chbmit" / "chb01" / fname, preload=True, verbose="error")
    chans = ["F7-T7", "T7-P7", "F8-T8", "T8-P8", "FT10-T8"]
    x = raw.get_data(picks=resolve(raw.ch_names, chans)) * 1e6
    sos = butter(4, [0.5, 45], btype="band", fs=256, output="sos")
    x = sosfiltfilt(sos, x, axis=1)[:, t0 * 256: t1 * 256]
    x = resample_poly(x, 1, 2, axis=1)  # 128 Hz
    # detector features & probability (per second, window scored at its end)
    fz = np.load(ROOT / "data" / "chbmit" / "features" / f"chb01_{fname}.npz")
    X, tf = fz["X"], fz["t"]
    ci = ALL_CH.index("T8-P8")
    sel = (tf >= t0) & (tf < t1)
    pz = np.load(ROOT / "detect" / "results" / "partial_1.npz", allow_pickle=True)["probs"].item()
    p = pz[("Both temples (4 ch)", "chb01", fname)]
    ps = np.convolve(p, np.ones(5) / 5)[: len(p)]
    alarms = [float(a) for a in alarms_from_prob(p, tf, 0.5) if t0 <= a < t1]
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    return {
        "fs": 128, "t0": t0, "onset": onset, "end": end, "chans": chans,
        "data": [rnd(c, 1) for c in x],
        "feat_t": rnd(tf[sel], 0), "prob": rnd(ps[sel], 3), "alarms": alarms, "threshold": 0.5,
        "linelen": rnd(X[sel, ci, 0], 3),
        "bands": {b: rnd(X[sel, ci, 9 + i], 3) for i, b in enumerate(bands)},  # relative band power
        "model": "Both temples (4 ch), patient-specific, leave-one-recording-out",
    }


def pink_noise(n, rng):
    """1/f ('pink') noise with unit standard deviation."""
    f = np.fft.rfftfreq(n)
    spec = rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def virtual_eeg(name, gain=None, fs=256):
    """Forward model: what scalp-like electrodes around the modelled patch would record.

    channel_e(t) = gain * sum_c w_ec * s_c * (spikes_c * synaptic kernel)(t)   # synaptic currents
                 - alpha * 51 mV * log10(Ko_local / Ko_rest)                    # glial K+ DC shift
                 + pink background noise
    w_ec: Gaussian weight of cell c by ring distance to electrode e (sigma = 20 cells);
    s_c = +1 excitatory, -1.5 inhibitory. Returns (dict, gain) so healthy and epileptic share
    one gain (calibrated so the healthy baseline has ~10 uV of neural signal).
    """
    from scipy.signal import butter, sosfiltfilt, resample_poly
    from detect import features

    z = np.load(ROOT / "output" / f"ionic_{name}.npz")
    rec, ko_map, sp_t, sp_i, inh = z["rec"], z["ko_map"], z["sp_t"], z["sp_i"], z["is_inh"]
    T = rec.shape[0] / 1000
    N = int(T * fs)
    ncell = ko_map.shape[1]
    stim_on, stim_off = float(z["stim_on"]), float(z["stim_off"])
    rng = np.random.default_rng(11 if name == "healthy" else 12)

    # spikes -> per-cell counts at fs, then electrode-weighted sums
    idx = np.clip((sp_t / 1000 * fs).astype(int), 0, N - 1)
    sign = np.where(inh, -1.5, 1.0)
    elec_angles = [0, 20, 50, 70, 35]  # cell positions (of 100) under each virtual electrode
    labels = ["V1 R-ear", "V2 R-ear", "V3 L-ear", "V4 L-ear"]  # virtual ear-hook electrodes
    cells = np.arange(ncell)
    t_k = np.arange(int(0.06 * fs)) / fs
    kernel = np.exp(-t_k / 0.008) - np.exp(-t_k / 0.002)  # synaptic current shape
    chans = []
    for pos in elec_angles[:4]:
        d = np.minimum(np.abs(cells - pos), ncell - np.abs(cells - pos))
        w = np.exp(-0.5 * (d / 20.0) ** 2)
        drive = np.zeros(N)
        np.add.at(drive, idx, (w * sign)[sp_i])
        chans.append(np.convolve(drive, kernel)[:N])
    chans = np.array(chans)
    if gain is None:  # healthy baseline -> ~10 uV RMS neural signal
        gain = 10.0 / chans[:, : int(stim_on * fs)].std()
    chans *= gain

    # glial K+ DC shift (DC-coupled channel) from the local K+ under electrode 1
    t_map = (np.arange(ko_map.shape[0]) + 1) * 0.01
    tt = np.arange(N) / fs
    d = np.minimum(np.abs(cells - elec_angles[0]), ncell - np.abs(cells - elec_angles[0]))
    w = np.exp(-0.5 * (d / 20.0) ** 2)
    ko_loc = np.interp(tt, t_map, (ko_map * w).sum(1) / w.sum())
    k0 = np.median(ko_loc[: int(stim_on * fs)])
    dc = -0.01 * 51.0 * np.log10(ko_loc / k0) * 1000  # uV (skull-attenuated field fraction 1%)

    # background brain noise (shared + independent pink noise)
    shared = pink_noise(N, rng)
    for k in range(4):
        chans[k] += 6.0 * (0.5 * shared + 0.87 * pink_noise(N, rng))
    dc = dc + 3.0 * pink_noise(N, rng)

    sos = butter(4, [0.5, 45], btype="band", fs=fs, output="sos")
    ac = sosfiltfilt(sos, chans, axis=1)
    # features exactly as in detect/detect.py (shown in the page)
    X = features(ac.astype(np.float32))
    t_end = np.arange(X.shape[0]) + 2.0
    # Detector: the model's ictal event is tonic firing, which band-passed EEG barely separates
    # from interictal bursts, so we detect the K+ DC shift instead (a known ictal marker).
    # Causal 1 s mean of the DC channel; alarm below -150 uV (~ local K+ > 8 mM, seizure-level).
    # NOTE: threshold chosen after inspecting this model's data (not independently validated).
    dc_1s = np.array([dc[max(0, int((te - 1) * fs)): int(te * fs)].mean() for te in t_end])
    prob = 1 / (1 + np.exp((dc_1s + 150.0) / 15.0))
    alarms, last = [], -1e9  # prob is already a 1 s mean: alarm directly, 60 s refractory
    for te, pv in zip(t_end, prob):
        if pv > 0.5 and te - last >= 60:
            alarms.append(float(te))
            last = te
    ds = np.vstack([resample_poly(ac, 1, 2, axis=1), resample_poly(dc, 1, 2)[None]])  # 128 Hz
    return {
        "fs": fs // 2, "t0": 0.0, "onset": stim_on, "end": stim_off, "chans": labels + ["DC · K⁺"],
        "data": [rnd(c, 1) for c in ds], "feat_t": rnd(t_end, 0), "prob": rnd(prob, 3),
        "alarms": alarms, "threshold": 0.5, "linelen": rnd(X[:, 0, 0], 3),
        "bands": {b: rnd(X[:, 0, 9 + i], 3) for i, b in enumerate(["delta", "theta", "alpha", "beta", "gamma"])},
        "model": "K⁺-DC detector: alarm when the 1 s DC shift < -150 µV (≈ local K⁺ > 8 mM); threshold tuned on this model",
        "source": "model",
    }, gain


def main():
    v_h, g = virtual_eeg("healthy")
    v_e, _ = virtual_eeg("epileptic", gain=g)
    data = {"fps": FPS, "epileptic": model("epileptic"), "healthy": model("healthy"),
            "inference": inference(), "eeg": eeg()}
    data["epileptic"]["veeg"], data["healthy"]["veeg"] = v_e, v_h
    data["eeg"]["source"] = "real"
    js = "window.SIM_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    OUT.write_text(js)
    print(f"wrote {OUT} ({len(js) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

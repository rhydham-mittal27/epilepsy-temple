"""Twin experiment: can the derived inference formulas recover hidden seizure variables?

Ground truth comes from ionic.py (run with --save). From it we synthesise the signals a
sensor would really record, with realistic confounds, then invert them with the formulas
derived in ../Inference_Formulas.md and score the result against the truth.

  E1  K+ from an intracellular glial electrode    Nernst inverse, assumed vs measured slope
  E2  K+ from scalp DC-EEG                          naive inverse vs process model vs EKF
  E3  [Cl]i from an E_GABA estimate                 chord inverse vs GHK inverse (r = 0.2-0.44)
  E4  E/I balance from the EEG spectrum             FOOOF aperiodic exponent (fixed + knee)
  E5  HbO/HbR from fNIRS optical density            MBLL, + short-channel, + partial volume
  E6  adenosine from postictal silence              tests the withdrawn exponential formula
  E7  KCC2 strength (tissue parameter)              Bayesian grid / ABC over ionic.py runs

Usage:
    uv run ionic.py healthy --t 90 --save
    uv run ionic.py epileptic --t 120 --save
    uv run infer.py --abc      # E7 grid: ~8 short ionic.py runs, cached to output/abc_grid.npz
    uv run infer.py            # E1-E7, figure + metrics
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import ionic

OUT = Path(__file__).parent / "output"
RNG = np.random.default_rng(7)
RTF = ionic.RTF
E_HCO3 = ionic.E_HCO3
LN10 = np.log(10)


def load(name):
    d = np.load(OUT / f"ionic_{name}.npz")
    rec = d["rec"]
    return SimpleNamespace(
        t=rec[:, 0] / 1000, lfp=rec[:, 1], ko=rec[:, 2], nai=rec[:, 3], cli=rec[:, 4],
        egaba=rec[:, 5], rate=rec[:, 8], ado=rec[:, 9], stim_on=float(d["stim_on"]),
        stim_off=float(d["stim_off"]), nv_gain=float(d["nv_gain"]), name=name)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def block_mean(x, k):
    n = len(x) // k
    return x[: n * k].reshape(n, k).mean(axis=1)


# ------------------------------------------------------------------ E1
def e1_glial(d):
    """Glial membrane: true slope 51 mV/decade (mammalian glia). Invert assuming 61.5 vs 51."""
    k0 = np.median(d.ko[d.t < d.stim_on])
    s_true = 51.0
    v = s_true * np.log10(d.ko / k0) + RNG.normal(0, 0.3, len(d.ko))  # 0.3 mV electrode noise
    inv = {s: k0 * 10 ** (v / s) for s in (61.54, 51.0)}
    pk = np.argmax(d.ko)
    res = {f"S={s}": {"rmse_mM": rmse(inv[s], d.ko), "peak_true": float(d.ko[pk]),
                      "peak_inferred": float(inv[s][pk])} for s in inv}
    return res, inv


# ------------------------------------------------------------------ E2
def fit_k_process(d, dt):
    """Calibrate dKo/dt = a*rate - (Ko - K0)/tau on data with known Ko (calibration cohort)."""
    ko = block_mean(d.ko, int(dt * 1000))
    r = block_mean(d.rate, int(dt * 1000))
    k0 = np.median(ko[: int(d.stim_on / dt)])
    dk = np.diff(ko) / dt
    A = np.c_[r[:-1], -(ko[:-1] - k0)]
    (a, inv_tau), *_ = np.linalg.lstsq(A, dk, rcond=None)
    return a, 1 / inv_tau, k0


def e2_dc_ekf(cal, d, dt=0.01):
    """Scalp DC = -alpha * glial depolarisation + slow drift + noise. alpha unknown."""
    a, tau, _ = fit_k_process(cal, dt)
    ko = block_mean(d.ko, int(dt * 1000))
    rate_true = block_mean(d.rate, int(dt * 1000))
    t = np.arange(len(ko)) * dt
    k0 = np.median(ko[: int(d.stim_on / dt)])
    S = 51.0
    alpha_true = 0.01  # cortex-to-scalp field fraction (unknown to the observer)
    drift = np.cumsum(RNG.normal(0, 0.4, len(ko)))  # uV random-walk electrode drift
    dc = -alpha_true * S * np.log10(ko / k0) * 1000 + drift + RNG.normal(0, 5, len(ko))  # uV
    rate_obs = np.clip(rate_true * (1 + RNG.normal(0, 0.2, len(ko))), 0, None)  # EEG activity index

    # (i) naive: constant-gain conversion with a textbook guess alpha=0.005, Nernstian slope
    naive = k0 * 10 ** (-dc / (0.005 * 61.54 * 1000))
    # (ii) open-loop process model driven by EEG activity only
    ol = np.empty_like(ko)
    x = k0
    for k in range(len(ko)):
        ol[k] = x
        x += dt * (a * rate_obs[k] - (x - k0) / tau)
    # (iii) EKF: state [Ko, alpha, drift]
    xs = np.array([k0, 0.005, 0.0])
    P = np.diag([0.25, 0.005 ** 2 * 4, 25.0])
    Q = np.diag([(0.05 * np.sqrt(dt)) ** 2 + 1e-4, 1e-10, 0.4 ** 2])
    R = 5.0 ** 2
    ekf = np.empty_like(ko)
    for k in range(len(ko)):
        # predict
        kx, al, dr = xs
        xs = np.array([kx + dt * (a * rate_obs[k] - (kx - k0) / tau), al, dr])
        F = np.array([[1 - dt / tau, 0, 0], [0, 1, 0], [0, 0, 1]])
        P = F @ P @ F.T + Q
        # update
        kx, al, dr = xs
        kx = max(kx, 0.5)
        h = -al * S * np.log10(kx / k0) * 1000 + dr
        H = np.array([[-al * S * 1000 / (kx * LN10), -S * np.log10(kx / k0) * 1000, 1.0]])
        Sy = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T / Sy).ravel()
        xs = xs + K * (dc[k] - h)
        P = (np.eye(3) - np.outer(K, H)) @ P
        ekf[k] = xs[0]
    res = {"process_model": {"a": float(a), "tau_s": float(tau)},
           "naive_rmse_mM": rmse(naive, ko), "openloop_rmse_mM": rmse(ol, ko),
           "ekf_rmse_mM": rmse(ekf, ko), "ekf_final_alpha": float(xs[1]), "alpha_true": alpha_true,
           "peak_true": float(ko.max()), "peak_naive": float(naive.max()),
           "peak_openloop": float(ol.max()), "peak_ekf": float(ekf.max())}
    return res, (t, ko, naive, ol, ekf, dc)


# ------------------------------------------------------------------ E3
def e3_chloride(d):
    sec = 1000  # one E_GABA estimate per second (e.g. repeated TMS-EEG / model fit)
    eg_true = block_mean(d.egaba, sec)
    cl_true = block_mean(d.cli, sec)
    eg_obs = eg_true + RNG.normal(0, 2.0, len(eg_true))  # +/- 2 mV estimation error
    ecl = (eg_obs - 0.2 * E_HCO3) / 0.8
    inv = {"chord (matches model)": ionic.CL_O * np.exp(ecl / RTF)}
    for r in (0.2, 0.25, 0.44):
        inv[f"GHK r={r}"] = (ionic.CL_O + r * ionic.HCO3_O) * np.exp(eg_obs / RTF) - r * ionic.HCO3_I
    res = {k: {"rmse_mM": rmse(v, cl_true), "bias_mM": float(np.mean(v - cl_true))} for k, v in inv.items()}
    return res, (np.arange(len(cl_true)), cl_true, inv)


# ------------------------------------------------------------------ E4
def e4_aperiodic(d):
    from fooof import FOOOF

    mask = (d.t > d.stim_off + 30) if d.name == "epileptic" else (d.t > d.stim_off + 10)
    f, p = welch(d.lfp[mask], fs=1000, nperseg=2048)
    out = {}
    for mode, rng in (("fixed", [30, 50]), ("knee", [2, 200])):
        fm = FOOOF(aperiodic_mode=mode, max_n_peaks=4, verbose=False)
        fm.fit(f, p, rng)
        out[mode] = {"exponent": float(fm.aperiodic_params_[-1]), "r2": float(fm.r_squared_)}
    return out, (f, p)


# ------------------------------------------------------------------ E5
EPS = {760: (1486.5845, 3843.707), 850: (2526.391, 1798.643)}  # Prahl, cm^-1/M (decadic)


def dpf(lam, age=30):
    return 223.3 + 0.05624 * age ** 0.8493 - 5.723e-7 * lam ** 3 + 0.001245 * lam ** 2 - 0.9025 * lam


def e5_fnirs(d, fs=10):
    k = 1000 // fs
    rate_s = np.convolve(d.rate, np.ones(1000) / 1000, mode="same")
    baseline = max(rate_s[d.t < d.stim_on].mean(), 0.1)
    _, hbo, hbr = ionic.balloon(rate_s, 0.001, baseline, d.nv_gain)
    dhbo = block_mean(60e-6 * hbo / 100, k)  # M, baseline HbO 60 uM
    dhbr = block_mean(40e-6 * hbr / 100, k)  # M, baseline HbR 40 uM
    t = np.arange(len(dhbo)) / fs
    sys_o = 1.0e-6 * np.sin(2 * np.pi * 0.1 * t) + 0.5e-6 * np.sin(2 * np.pi * 1.2 * t)  # scalp
    sys_r = -0.3 * sys_o
    f_brain = 0.15  # fraction of the photon path inside cortex (partial volume)
    dpf_true = {lam: dpf(lam) * 1.1 for lam in EPS}  # individual DPF 10% above the formula

    def od(d_cm, brain_frac, lam):
        eo, er = EPS[lam]
        L = d_cm * dpf_true[lam]
        return ((eo * dhbo + er * dhbr) * L * brain_frac
                + (eo * sys_o + er * sys_r) * L * (1 - brain_frac)
                + RNG.normal(0, 1e-4, len(t)))

    def mbll(ods, d_cm):
        E = np.array([EPS[760], EPS[850]])
        rhs = np.vstack([ods[760] / dpf(760), ods[850] / dpf(850)]) / d_cm
        c = np.linalg.solve(E, rhs)
        return c[0], c[1]

    long_ = {lam: od(3.0, f_brain, lam) for lam in EPS}
    short = {lam: od(0.8, 0.0, lam) for lam in EPS}
    o_naive, r_naive = mbll(long_, 3.0)
    o_s, r_s = mbll(short, 0.8)
    beta = np.dot(o_s, o_naive) / np.dot(o_s, o_s)
    o_ssr = o_naive - beta * o_s
    beta_r = np.dot(r_s, r_naive) / np.dot(r_s, r_s)
    r_ssr = r_naive - beta_r * r_s
    o_pv, r_pv = o_ssr / f_brain, r_ssr / f_brain

    def score(est, true):
        return {"rmse_uM": rmse(est, true) * 1e6, "corr": float(np.corrcoef(est, true)[0, 1]),
                "amp_ratio": float(np.ptp(est) / np.ptp(true))}

    res = {"naive MBLL": score(o_naive, dhbo), "+ short-channel": score(o_ssr, dhbo),
           "+ partial-volume": score(o_pv, dhbo), "HbR full pipeline": score(r_pv, dhbr)}
    return res, (t, dhbo * 1e6, o_naive * 1e6, o_ssr * 1e6, o_pv * 1e6)


# ------------------------------------------------------------------ E6
def e6_adenosine(ds):
    out = {}
    for d in ds:
        rate_s = np.convolve(d.rate, np.ones(1000) / 1000, mode="same")
        baseline = rate_s[d.t < d.stim_on].mean()
        rec_idx = np.flatnonzero((d.t >= d.stim_off + 1.0) & (rate_s > 0.5 * baseline))
        t_rec = d.t[rec_idx[0]] if len(rec_idx) else np.nan
        silence = t_rec - d.stim_off
        a_peak = float(d.ado.max())
        a_at_recovery = float(d.ado[rec_idx[0]]) if len(rec_idx) else np.nan
        out[d.name] = {"A_peak": a_peak, "silence_s": float(silence),
                       "A_th_formula": float(a_peak * np.exp(-silence / 20.0)),
                       "A_at_recovery_true": a_at_recovery}
    return out


# ------------------------------------------------------------------ E7
ABC_GRID = [0.15, 0.2, 0.3, 0.45, 0.7, 1.0]


def features(rec, t):
    m = t > 5
    return np.array([rec[m, 5].mean(), rec[m, 8].mean()])  # resting E_GABA, mean E rate


def run_abc():
    rows = []
    for k in ABC_GRID:
        a = SimpleNamespace(
            n=100, radius=10, g_exc=1.3, g_inh=2.6, seed=3, kcc2=k, kcc2_base=0.4, dt=0.02,
            kbath=4.0, glia=0.3, clear=0.35, bg_rate=60.0, bg_w=0.07, stim_rate=0.0,
            stim_on=1e6, stim_off=1e6, d_lat=0.625, g_kna=1.33, pump_i=1.0, std_u=0.0,
            g_ado=0.3, ado_alpha=0.5, ado_tau=20.0, warmup=60.0, t=25.0)
        rec, *_ = ionic.network(a)
        f = features(rec, rec[:, 0] / 1000)
        print(f"[abc] kcc2 x{k:<4}  E_GABA {f[0]:6.1f} mV  E rate {f[1]:5.1f} Hz")
        rows.append(f)
    np.savez(OUT / "abc_grid.npz", grid=np.array(ABC_GRID), feats=np.array(rows))


def e7_posterior(ep):
    path = OUT / "abc_grid.npz"
    if not path.exists():
        return None, None
    g = np.load(path)
    grid, feats = g["grid"], g["feats"]
    m = ep.t > ep.stim_off + 30
    obs = np.array([ep.egaba[m].mean() + RNG.normal(0, 1.0), ep.rate[m].mean()])
    sig = np.array([1.5, 0.25 * obs[1]])  # E_GABA +/-1.5 mV, rate +/-25%
    loglik = -0.5 * (((feats - obs) / sig) ** 2).sum(axis=1)
    prior = np.ones_like(grid) / len(grid)  # flat over the grid
    post = np.exp(loglik - loglik.max()) * prior
    post /= post.sum()
    mean = float((post * grid).sum())
    sd = float(np.sqrt((post * (grid - mean) ** 2).sum()))
    return {"true_kcc2": 0.3, "map": float(grid[np.argmax(post)]), "mean": mean, "sd": sd,
            "posterior": dict(zip(map(float, grid), map(float, post))), "observed": obs.tolist()}, (grid, post)


# ------------------------------------------------------------------ figure
def figure(e1, e2, e3, e4, e5, e7, ep, he):
    fig, ax = plt.subplots(3, 2, figsize=(15, 13))
    a = ax[0, 0]
    a.plot(ep.t, ep.ko, "k", lw=1.2, label="true [K]o")
    a.plot(ep.t, e1[1][61.54], "tab:red", lw=0.8, label="inferred, S = 61.5 (textbook Nernst)")
    a.plot(ep.t, e1[1][51.0], "tab:green", lw=0.8, ls="--", label="inferred, S = 51 (measured glia)")
    a.set(title="E1  K$^+$ from glial electrode (Nernst inverse)", xlabel="s", ylabel="mM", xlim=(5, 40))
    a.legend(fontsize=8)

    t, ko, naive, ol, ekf, _ = e2[1]
    a = ax[0, 1]
    a.plot(t, ko, "k", lw=1.2, label="true [K]o")
    a.plot(t, naive, "tab:red", lw=0.7, label=f"naive DC conversion (RMSE {e2[0]['naive_rmse_mM']:.2f})")
    a.plot(t, ol, "tab:orange", lw=0.7, label=f"EEG-driven model only ({e2[0]['openloop_rmse_mM']:.2f})")
    a.plot(t, ekf, "tab:blue", lw=0.9, label=f"EKF: model + DC ({e2[0]['ekf_rmse_mM']:.2f})")
    a.set(title="E2  K$^+$ from scalp DC-EEG (Kalman filter)", xlabel="s", ylabel="mM", xlim=(5, 60))
    a.legend(fontsize=8)

    x, cl, inv = e3[1]
    a = ax[1, 0]
    a.plot(x, cl, "k", lw=1.5, label="true [Cl]$_i$")
    for (k, v), c in zip(inv.items(), ["tab:green", "tab:purple", "tab:red", "tab:brown"]):
        a.plot(x, v, color=c, lw=0.8, label=f"{k} (RMSE {e3[0][k]['rmse_mM']:.2f})")
    a.set(title="E3  [Cl]$_i$ from E$_{GABA}$ estimate (+/-2 mV)", xlabel="s", ylabel="mM")
    a.legend(fontsize=7)

    a = ax[1, 1]
    for d, c in ((he, "tab:blue"), (ep, "tab:red")):
        f, p = e4[d.name][1]
        m = (f >= 2) & (f <= 200)
        a.loglog(f[m], p[m], color=c,
                 label=f"{d.name}: chi(30-50 Hz) = {e4[d.name][0]['fixed']['exponent']:.2f}, "
                       f"chi(knee) = {e4[d.name][0]['knee']['exponent']:.2f}")
    a.set(title="E4  aperiodic exponent (FOOOF) of LFP proxy", xlabel="Hz", ylabel="power")
    a.legend(fontsize=8)

    t, tru, nv, ssr, pv = e5[1]
    a = ax[2, 0]
    a.plot(t, tru, "k", lw=1.5, label="true cortical $\\Delta$HbO")
    a.plot(t, nv, "tab:red", lw=0.7, label="naive MBLL")
    a.plot(t, ssr, "tab:orange", lw=0.7, label="+ short-channel regression")
    a.plot(t, pv, "tab:blue", lw=0.9, label="+ partial-volume correction")
    a.set(title="E5  $\\Delta$HbO from fNIRS (MBLL)", xlabel="s", ylabel="$\\mu$M", xlim=(0, 60))
    a.legend(fontsize=8)

    a = ax[2, 1]
    if e7[0]:
        grid, post = e7[1]
        a.bar(range(len(grid)), post, color="tab:blue")
        a.set_xticks(range(len(grid)), [f"x{g}" for g in grid])
        a.axvline(list(grid).index(0.3), color="k", ls="--", label="true KCC2 = x0.3")
        a.set(title=f"E7  posterior over KCC2 strength (mean {e7[0]['mean']:.2f} +/- {e7[0]['sd']:.2f})",
              xlabel="KCC2 strength", ylabel="posterior probability")
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, "run  uv run infer.py --abc  first", ha="center", va="center")
        a.set_axis_off()
    fig.suptitle("Twin experiment: derived inference formulas vs ground truth (ionic.py)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "infer_results.png", dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--abc", action="store_true", help="run the E7 simulation grid (~10 min)")
    args = p.parse_args()
    if args.abc:
        run_abc()
    he, ep = load("healthy"), load("epileptic")
    e1 = e1_glial(ep)
    e2 = e2_dc_ekf(he, ep)
    e3 = e3_chloride(ep)
    e4 = {d.name: e4_aperiodic(d) for d in (he, ep)}
    e5 = e5_fnirs(ep)
    e6 = e6_adenosine([he, ep])
    e7 = e7_posterior(ep)
    figure(e1, e2, e3, e4, e5, e7, ep, he)
    metrics = {"E1_glial_K": e1[0], "E2_dc_K": e2[0], "E3_chloride": e3[0],
               "E4_aperiodic": {k: v[0] for k, v in e4.items()}, "E5_fnirs": e5[0],
               "E6_adenosine": e6, "E7_kcc2": e7[0]}
    (OUT / "infer_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"saved {OUT / 'infer_results.png'} and infer_metrics.json")


if __name__ == "__main__":
    main()

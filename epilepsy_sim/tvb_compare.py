"""Compare against an established simulator: The Virtual Brain (TVB).

T1  Cross-validation: TVB's official Epileptor vs our main.py Epileptor (same equations,
    parameters and initial state). Any mismatch would expose a bug in our code.
T2  VEP-style inference: 76-region human connectome, one epileptogenic zone (EZ). The same
    Bayesian framework used in infer.py recovers the hidden excitability x0 of the EZ and
    localises it from seizure onset times alone.

Usage:
    uv run --extra tvb tvb_compare.py
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
from tvb.simulator.lab import connectivity, coupling, integrators, models, monitors, simulator  # noqa: E402

import main as ours  # noqa: E402

OUT = Path(__file__).parent / "output"
IC = np.array([-1.5, -10.25, 4.0, -1.0, 0.0, 0.0])  # same start as main.py: rest, z high


def seizures_from_z(t, z, merge=50.0, min_len=20.0):
    """Seizure = z rising (same definition as main.py). Returns [(start, end), ...]."""
    ictal = np.gradient(z) > 0
    e = np.diff(np.r_[0, ictal.astype(int), 0])
    segs = []
    for s0, s1 in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if segs and t[s0] - segs[-1][1] < merge:
            segs[-1][1] = t[s1 - 1]
        else:
            segs.append([t[s0], t[s1 - 1]])
    return [(a, b) for a, b in segs if b - a >= min_len]


def run_tvb(conn, x0, t_end, dt=0.05, ks=0.0, a_coup=0.0):
    n = conn.number_of_regions
    model = models.Epileptor(x0=np.asarray(x0, float), Ks=np.array([ks]),
                             variables_of_interest=("x1", "z", "x2"))

    def make(ic):
        return simulator.Simulator(
            model=model, connectivity=conn, coupling=coupling.Difference(a=np.array([a_coup])),
            integrator=integrators.HeunDeterministic(dt=dt), monitors=(monitors.Raw(),),
            simulation_length=t_end, initial_conditions=ic)

    probe = make(None)
    probe.configure()
    horizon = probe.connectivity.horizon
    ic = np.tile(IC[None, :, None, None], (horizon, 1, n, 1))
    sim = make(ic)
    sim.configure()
    (t, y), = sim.run()
    return t, y[:, 0, :, 0], y[:, 1, :, 0], y[:, 2, :, 0]  # x1, z, x2 per node


# ------------------------------------------------------------------ T1
def t1_crossval(x0=-1.6, t_end=6000.0):
    conn = connectivity.Connectivity(
        weights=np.zeros((1, 1)), tract_lengths=np.zeros((1, 1)), region_labels=np.array(["n0"]),
        centres=np.zeros((1, 3)), speed=np.array([3.0]))
    conn.configure()
    t0 = time.time()
    tt, x1, z, x2 = run_tvb(conn, [x0], t_end)
    tvb_s = time.time() - t0
    tt, z_tvb, lfp_tvb = tt, z[:, 0], (x2 - x1)[:, 0]

    res = {}
    traces = {"TVB (Heun, dt 0.05)": (tt, z_tvb, lfp_tvb)}
    for dt in (0.05, 0.01):
        t0 = time.time()
        # our integrator records every 20 steps; resample onto TVB's grid
        t_o, out = ours.simulate_epileptor(t_end=t_end, dt=dt, x0=x0)
        run_s = time.time() - t0
        x1o, x2o, zo = out.T
        traces[f"ours (Euler, dt {dt})"] = (t_o, zo, x2o - x1o)
        res[f"ours_dt{dt}_runtime_s"] = run_s
    res["tvb_runtime_s"] = tvb_s

    table = {}
    for k, (t, z_, l_) in traces.items():
        sz = seizures_from_z(t, z_)
        table[k] = {"n_seizures": len(sz), "onsets": [round(a, 1) for a, _ in sz],
                    "durations": [round(b - a, 1) for a, b in sz]}
    ref_t, ref_z, ref_l = traces["TVB (Heun, dt 0.05)"]
    for k, (t, z_, l_) in traces.items():
        if k.startswith("ours"):
            zi = np.interp(ref_t, t, z_)
            li = np.interp(ref_t, t, l_)
            table[k]["z_rmse_vs_tvb"] = float(np.sqrt(np.mean((zi - ref_z) ** 2)))
            table[k]["lfp_corr_vs_tvb"] = float(np.corrcoef(li, ref_l)[0, 1])
    res["table"] = table
    return res, traces


# ------------------------------------------------------------------ T2
def t2_vep(t_end=5000.0, ez_label="rAMYG", grid=(-2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4)):
    conn = connectivity.Connectivity.from_file()
    conn.speed = np.array([3.0])
    conn.configure()
    labels = [str(x) for x in conn.region_labels]
    ez = labels.index(ez_label) if ez_label in labels else int(np.argmax(conn.weights.sum(0)))
    base = np.full(conn.number_of_regions, -2.2)
    # propagation zone (PZ): the 3 regions most strongly connected to the EZ, mildly
    # hyperexcitable (x0 = -2.0) as in VEP studies, so the seizure can spread
    pz = [int(j) for j in np.argsort(conn.weights[ez])[::-1] if j != ez][:3]
    base[pz] = -2.0

    def simulate(x0_ez):
        x0 = base.copy()
        x0[ez] = x0_ez
        t, x1, z, x2 = run_tvb(conn, x0, t_end, ks=-0.2, a_coup=1.0)
        onsets = np.full(x1.shape[1], np.nan)
        for j in range(x1.shape[1]):
            idx = np.flatnonzero(x1[:, j] > 0)
            if len(idx):
                onsets[j] = t[idx[0]]
        return t, x1, z, onsets

    true_x0 = -1.6
    t0 = time.time()
    t, x1, z, onsets = simulate(true_x0)
    sim_s = time.time() - t0
    rng = np.random.default_rng(1)
    obs = onsets + rng.normal(0, 20.0, onsets.shape)  # 20 ms onset-timing noise (SEEG picking)
    seized = ~np.isnan(onsets)
    loc_ez = int(np.nanargmin(obs)) if seized.any() else -1

    # Bayesian grid over the EZ's x0 using the observed EZ onset time (Gaussian, sigma 20 ms;
    # 'no seizure' when the simulation never seizes is scored as a very poor fit)
    ll = []
    sims = {}
    for g in grid:
        _, _, _, on_g = simulate(g)
        sims[g] = on_g
        pred = on_g[ez]
        if np.isnan(pred) and np.isnan(obs[ez]):
            ll.append(0.0)
        elif np.isnan(pred) or np.isnan(obs[ez]):
            ll.append(-50.0)
        else:
            ll.append(-0.5 * ((obs[ez] - pred) / 20.0) ** 2)
    ll = np.array(ll)
    post = np.exp(ll - ll.max())
    post /= post.sum()
    g = np.array(grid)
    mean = float((post * g).sum())
    sd = float(np.sqrt((post * (g - mean) ** 2).sum()))
    res = {"n_regions": conn.number_of_regions, "ez_label": labels[ez], "ez_index": ez,
           "pz_labels": [labels[j] for j in pz],
           "regions_seizing": int(seized.sum()),
           "recruited": [labels[j] for j in np.argsort(np.nan_to_num(onsets, nan=1e9))[: int(seized.sum())]],
           "localised_ez": labels[loc_ez] if loc_ez >= 0 else None,
           "localisation_correct": loc_ez == ez, "x0_true": true_x0, "x0_map": float(g[np.argmax(post)]),
           "x0_mean": mean, "x0_sd": sd, "posterior": dict(zip(map(float, g), map(float, post))),
           "runtime_per_sim_s": sim_s}
    return res, (t, x1, z, onsets, labels, ez, g, post)


def figure(t1, t2):
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2)
    a = fig.add_subplot(gs[0, :])
    for (k, (t, z_, l_)), c, ls in zip(t1[1].items(), ["k", "tab:red", "tab:blue"], ["-", "--", ":"]):
        a.plot(t, l_, color=c, ls=ls, lw=0.6, label=k)
    a.set(title="T1  Epileptor cross-validation: TVB vs our main.py (x2 - x1)", xlabel="time (a.u.)")
    a.legend(fontsize=8, loc="upper right")
    a = fig.add_subplot(gs[1, :])
    for (k, (t, z_, l_)), c, ls in zip(t1[1].items(), ["k", "tab:red", "tab:blue"], ["-", "--", ":"]):
        a.plot(t, z_, color=c, ls=ls, lw=1.0, label=k)
    a.set(title="T1  slow permittivity z", xlabel="time (a.u.)", ylabel="z")
    a.legend(fontsize=8, loc="upper right")

    t, x1, z, onsets, labels, ez, g, post = t2[1]
    a = fig.add_subplot(gs[2, 0])
    order = np.argsort(np.nan_to_num(onsets, nan=1e9))[:8]
    for k, j in enumerate(order):
        a.plot(t, x1[:, j] - 3 * k, lw=0.6, color="tab:red" if j == ez else "k")
        a.text(t[0], -3 * k + 0.5, labels[j], fontsize=7)
    a.set(title=f"T2  TVB 76-region connectome: first recruited regions (red = EZ {labels[ez]})",
          xlabel="time (ms)", yticks=[])
    a = fig.add_subplot(gs[2, 1])
    a.bar([f"{x:.1f}" for x in g], post, color="tab:blue")
    a.axvline(list(np.round(g, 1)).index(-1.6), color="k", ls="--", label="true x0 = -1.6")
    a.set(title=f"T2  posterior over EZ excitability x0 (mean {t2[0]['x0_mean']:.2f} +/- {t2[0]['x0_sd']:.2f})",
          xlabel="x0", ylabel="posterior")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "tvb_compare.png", dpi=110)
    plt.close(fig)


def main():
    t1 = t1_crossval()
    print(json.dumps(t1[0], indent=2))
    t2 = t2_vep()
    print(json.dumps(t2[0], indent=2))
    figure(t1, t2)
    (OUT / "tvb_metrics.json").write_text(json.dumps({"T1": t1[0], "T2": t2[0]}, indent=2))
    print(f"saved {OUT / 'tvb_compare.png'} and tvb_metrics.json")


if __name__ == "__main__":
    main()

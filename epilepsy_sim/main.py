"""Two classic ways to simulate a seizure.

1. network   - 1000 spiking neurons (Izhikevich 2003, 80% excitatory / 20% inhibitory).
               Mid-run we weaken GABA inhibition, mimicking interneuron exhaustion and
               chloride loading. The asynchronous "healthy" network tips into a
               hypersynchronous seizure, then recovers when inhibition returns.
2. epileptor - Jirsa et al. 2014 (Brain) 6-variable model. A slow "permittivity"
               variable z (energy / ion build-up) drives spontaneous seizure onset and
               offset, reproducing fast discharges, spike-wave events and the
               abrupt termination seen in human recordings.

Usage:
    uv run main.py                      # both models
    uv run main.py --model network --inhibition 0.3
    uv run main.py --model epileptor --x0 -2.2   # healthy excitability: no seizures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "output"


# ---------------------------------------------------------------- spiking network
def simulate_network(t_ms=4000, seizure=(1500, 2800), inhibition_during=0.35, seed=1):
    """Izhikevich network with a time window of weakened inhibition."""
    rng = np.random.default_rng(seed)
    ne, ni = 800, 200
    re, ri = rng.random(ne), rng.random(ni)

    # Neuron parameters: regular-spiking excitatory, fast-spiking inhibitory cells
    a = np.r_[0.02 * np.ones(ne), 0.02 + 0.08 * ri]
    b = np.r_[0.2 * np.ones(ne), 0.25 - 0.05 * ri]
    c = np.r_[-65 + 15 * re**2, -65 * np.ones(ni)]
    d = np.r_[8 - 6 * re**2, 2 * np.ones(ni)]

    # Synaptic weights: column j = effect of neuron j on every neuron
    s = np.c_[0.5 * rng.random((ne + ni, ne)), -rng.random((ne + ni, ni))]

    v = -65 * np.ones(ne + ni)
    u = b * v
    spikes_t, spikes_i = [], []
    lfp = np.zeros(t_ms)
    g_inh = np.ones(t_ms)
    g_inh[seizure[0] : seizure[1]] = inhibition_during

    for t in range(t_ms):
        drive = np.r_[5 * rng.standard_normal(ne), 2 * rng.standard_normal(ni)]
        fired = np.flatnonzero(v >= 30)
        if fired.size:
            spikes_t.append(np.full(fired.size, t))
            spikes_i.append(fired)
            v[fired] = c[fired]
            u[fired] += d[fired]
            exc = fired[fired < ne]
            inh = fired[fired >= ne]
            drive += s[:, exc].sum(axis=1) + g_inh[t] * s[:, inh].sum(axis=1)
        # LFP proxy: summed synaptic input onto excitatory (pyramidal) cells
        lfp[t] = -drive[:ne].mean()
        for _ in range(2):  # two 0.5 ms half-steps for numerical stability
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + drive)
        u += a * (b * v - u)

    st = np.concatenate(spikes_t) if spikes_t else np.array([])
    si = np.concatenate(spikes_i) if spikes_i else np.array([])
    return st, si, lfp, g_inh, ne


def synchrony(st, si, ne, t0, t1, bin_ms=5):
    """Population synchrony: variance of pop. rate / mean single-cell variance (0..1)."""
    mask = (st >= t0) & (st < t1) & (si < ne)
    bins = np.arange(t0, t1 + bin_ms, bin_ms)
    counts = np.zeros((ne, len(bins) - 1))
    idx = np.digitize(st[mask], bins) - 1
    np.add.at(counts, (si[mask], idx), 1)
    pop_var = counts.mean(axis=0).var()
    cell_var = counts.var(axis=1).mean()
    return pop_var / cell_var if cell_var else 0.0


def plot_network(args):
    st, si, lfp, g_inh, ne = simulate_network(
        t_ms=args.duration, seizure=(args.onset, args.offset), inhibition_during=args.inhibition
    )
    t = np.arange(args.duration)
    rate = np.bincount(st.astype(int), minlength=args.duration)
    rate_hz = np.convolve(rate, np.ones(10) / 10, mode="same") * 1000 / (ne + 200)

    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [1, 3, 1.3, 1.3]})
    ax[0].plot(t, g_inh, color="tab:blue")
    ax[0].set_ylabel("GABA\nstrength")
    ax[0].set_ylim(0, 1.1)
    ax[1].scatter(st, si, s=0.3, c=np.where(si < ne, "k", "tab:red"), rasterized=True)
    ax[1].set_ylabel("neuron #\n(red = inhibitory)")
    ax[2].plot(t, rate_hz, color="tab:purple", lw=0.8)
    ax[2].set_ylabel("pop. rate\n(Hz/neuron)")
    ax[3].plot(t, lfp, color="tab:green", lw=0.6)
    ax[3].set_ylabel("LFP proxy\n(a.u.)")
    ax[3].set_xlabel("time (ms)")
    for a_ in ax:
        a_.axvspan(args.onset, args.offset, color="tab:red", alpha=0.08)
    fig.suptitle("Spiking network: weakening inhibition -> hypersynchronous seizure")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "network_seizure.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    base = synchrony(st, si, ne, 0, args.onset)
    ictal = synchrony(st, si, ne, args.onset, args.offset)
    post = synchrony(st, si, ne, args.offset, args.duration)
    print(f"[network] synchrony  baseline={base:.3f}  seizure={ictal:.3f}  recovery={post:.3f}")
    print(f"[network] saved {path}")


# ---------------------------------------------------------------- Epileptor
def simulate_epileptor(t_end=6000.0, dt=0.05, x0=-1.6, noise=0.0, seed=0):
    """Jirsa et al. 2014 Epileptor, forward Euler(-Maruyama)."""
    rng = np.random.default_rng(seed)
    y0, tau0, tau2 = 1.0, 2857.0, 10.0
    i1, i2, gamma = 3.1, 0.45, 0.01
    # Start at rest (lower branch of x1) with z high, so any seizure is spontaneous
    x1, y1, z, x2, y2, g = -1.5, -10.25, 4.0, -1.0, 0.0, 0.0

    n = int(t_end / dt)
    keep = 20  # store every 20th step
    out = np.empty((n // keep, 3))
    for k in range(n):
        if x1 < 0:
            f1 = x1**3 - 3 * x1**2
        else:
            f1 = (x2 - 0.6 * (z - 4) ** 2) * x1
        f2 = 0.0 if x2 < -0.25 else 6 * (x2 + 0.25)

        dx1 = y1 - f1 - z + i1
        dy1 = y0 - 5 * x1**2 - y1
        dz = (4 * (x1 - x0) - z) / tau0
        dx2 = -y2 + x2 - x2**3 + i2 + 2 * g - 0.3 * (z - 3.5)
        dy2 = (-y2 + f2) / tau2
        dg = -gamma * (g - 0.1 * x1)

        x1 += dt * dx1
        y1 += dt * dy1
        z += dt * dz
        x2 += dt * dx2 + noise * np.sqrt(dt) * rng.standard_normal()
        y2 += dt * dy2
        g += dt * dg
        if k % keep == 0 and k // keep < len(out):
            out[k // keep] = (x1, x2, z)
    t = np.arange(len(out)) * dt * keep
    return t, out


def plot_epileptor(args):
    t, out = simulate_epileptor(t_end=args.t_end, x0=args.x0, noise=args.noise)
    x1, x2, z = out.T
    lfp = -x1 + x2  # the model's simulated field potential

    # Seizure state: z (slow build-up) rises only while the fast subsystem discharges
    ictal = np.gradient(z) > 0
    edges = np.diff(np.r_[0, ictal.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    # Merge flickers: join segments separated by < 50 time units, drop ones < 20
    step = t[1] - t[0]
    segs = []
    for s0, s1 in zip(starts, ends):
        if segs and (s0 - segs[-1][1]) * step < 50:
            segs[-1][1] = s1
        else:
            segs.append([s0, s1])
    segs = [s for s in segs if (s[1] - s[0]) * step >= 20]
    starts = [s[0] for s in segs]
    ends = [s[1] for s in segs]
    ictal = np.zeros_like(ictal)
    for s0, s1 in segs:
        ictal[s0:s1] = True

    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    for s0, s1 in zip(starts, ends):
        for a_ in ax:
            a_.axvspan(t[s0], t[s1 - 1], color="tab:red", alpha=0.08)
    ax[0].plot(t, lfp, color="k", lw=0.5)
    ax[0].set_ylabel("simulated LFP\n(x2 - x1)")
    ax[1].plot(t, z, color="tab:orange")
    ax[1].set_ylabel("z (slow\npermittivity)")
    ax[1].set_xlabel("time (a.u.)")
    fig.suptitle(f"Epileptor (Jirsa 2014), excitability x0 = {args.x0}")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "epileptor_seizures.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    print(f"[epileptor] x0={args.x0}: {len(starts)} seizure(s), "
          f"time in seizure {100 * ictal.mean():.1f}%")
    print(f"[epileptor] saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["network", "epileptor", "both"], default="both")
    p.add_argument("--duration", type=int, default=4000, help="network: ms to simulate")
    p.add_argument("--onset", type=int, default=1500, help="network: ms inhibition drops")
    p.add_argument("--offset", type=int, default=2800, help="network: ms inhibition returns")
    p.add_argument("--inhibition", type=float, default=0.35,
                   help="network: GABA strength during seizure window (1 = normal)")
    p.add_argument("--x0", type=float, default=-1.6,
                   help="epileptor: excitability; > about -2.05 seizes, lower is healthy")
    p.add_argument("--t-end", type=float, default=6000.0, help="epileptor: time units")
    p.add_argument("--noise", type=float, default=0.0, help="epileptor: noise on x2")
    args = p.parse_args()

    if args.model in ("network", "both"):
        plot_network(args)
    if args.model in ("epileptor", "both"):
        plot_epileptor(args)


if __name__ == "__main__":
    main()

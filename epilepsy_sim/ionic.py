"""Biophysical seizure model with potassium, sodium and chloride dynamics.

Neuron core: Cressman et al. 2009 (J Comput Neurosci 26:159), using the corrected
equations from the ModelDB code (see erratum, J Comput Neurosci 30:781, 2011):
Hodgkin-Huxley Na/K spiking + Ca-activated AHP, with dynamic [K]o and [Na]i,
Na/K-ATPase pump, glial K+ uptake and K+ diffusion to the vasculature.

Added on top (following Krishnan & Bazhenov 2011, J Neurosci 31:8870, and
Doyon et al. 2016):
  * dynamic intracellular chloride [Cl]i per neuron
  * KCC2 cotransporter, driven by ln([K]i[Cl]i / [K]o[Cl]o); it extrudes Cl- AND K+,
    so it loads the extracellular space with K+ exactly when Cl- is high
  * GABA-A synapses carrying Cl- (80%) and HCO3- (20%), so E_GABA follows [Cl]i
    and inhibition can turn depolarising
  * AMPA synapses whose Na+/K+ fluxes feed the concentration equations
  * a ring network (80% excitatory, 20% inhibitory) with lateral K+ diffusion
    through the extracellular space, so seizures spread spatially
  * a Balloon-Windkessel neurovascular readout (Friston et al. 2000):
    cerebral blood flow, oxy- and deoxy-haemoglobin, i.e. what an optical
    (NIRS/PPG) sensor would see

Seizure-termination / postictal mechanisms:
  * electrogenic Na/K pump current (3 Na+ out, 2 K+ in); scale set by --pump-i
  * Na+-activated K+ current (Slack, Compte et al. 2003), EC50 shifted to this
    model's resting [Na]i
  * adenosine from ATP breakdown (proportional to pump ATP use above rest), acting
    on A1 receptors: GIRK K+ current + reduced transmitter release (During &
    Spencer 1992: adenosine surges during human seizures)
  * optional Tsodyks-Markram synaptic depression (--std-u; off by default)

Numerics: exponential-Euler for voltage and gates (stable at seizure-level
conductances), numba-compiled; 60 s warm-up so every tissue starts from its own
resting ion concentrations.

Usage:
    uv run ionic.py validate               # reproduce Cressman 2009 Fig. 1 (single neuron)
    uv run ionic.py healthy                # normal tissue: stimulus -> brief response
    uv run ionic.py epileptic --t 300      # KCC2/glia/clearance loss -> interictal
                                           # discharges + triggered ictal event
    uv run ionic.py epileptic --kcc2 0.2 --glia 0.4 --t 120
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

OUT = Path(__file__).parent / "output"

# Physical constants and baseline concentrations (Cressman 2009, corrected code)
RTF = 26.64  # RT/F in mV (Cressman 2009; = 36 C. At 37 C it is 26.73)
GAMMA = 0.044494542  # current (uA/cm2) -> intracellular conc. rate (mM/s)
TAU = 1000.0  # ms per s: concentration equations are written per second
BETA = 7.0  # intracellular / extracellular volume ratio
CL_O = 130.0  # extracellular chloride (mM), large reservoir
HCO3_I, HCO3_O = 16.0, 26.0  # bicarbonate (mM)
E_HCO3 = RTF * np.log(HCO3_I / HCO3_O)  # ~ -13 mV (anion: ln(in/out))
P_NA_AMPA = 0.64  # Na fraction of AMPA conductance (gives E_AMPA ~ 0 mV at rest)
GABA_CL_FRAC = 0.8  # Cl- vs HCO3- share of GABA-A conductance
KNA_EC50 = 68.6  # mM, Na+ half-activation of I_KNa (see note in _simulate)
PUMP_REST = 0.25  # mM/s, pump flux above which ATP use produces adenosine


@nb.njit(cache=True)
def _simulate(t_ms, dt, n, is_inh, ptr, post, wts, kbath, glia_s, kcc2_u, cl_dynamic,
              rate_bg, w_bg, stim_t0, stim_t1, stim_rate, d_lat, eps, rho,
              tau_e, tau_i, rec_every, ko_every, seed,
              v0, n0, h0, ko0, nai0, cli0, ca0, g_kna, electrogenic, std_u, std_tau,
              g_ado, ado_alpha, ado_tau):
    np.random.seed(seed)
    steps = int(t_ms / dt)
    nrec = steps // rec_every
    nko = steps // ko_every

    v = v0.copy()
    ng = n0.copy()
    hg = h0.copy()
    ko = ko0.copy()
    nai = nai0.copy()
    cli = cli0.copy()
    ca = ca0.copy()
    ge = np.zeros(n)
    gi = np.zeros(n)
    xres = np.ones(n)  # presynaptic vesicle resources (Tsodyks-Markram depression)
    rec_x = np.exp(-dt / std_tau) if std_u > 0 else 0.0
    # extracellular adenosine near each cell (normalised; 1 = A1 half-activation).
    # Produced from ATP breakdown in proportion to Na/K-pump ATP use above rest
    # (PUMP_REST), cleared with time constant ado_tau.
    ado = np.zeros(n)
    dec_ado = np.exp(-dt / ado_tau)

    rec = np.zeros((nrec, 10))  # t, lfp, ko, nai, cli, egaba, ek, pump, rate_e, adenosine
    ko_map = np.zeros((nko, n))
    max_sp = 20_000_000
    sp_t = np.zeros(max_sp, np.float32)
    sp_i = np.zeros(max_sp, np.int32)
    nsp = 0

    g_na, g_nal, g_k, g_kl, g_ahp, g_cll, g_ca = 100.0, 0.0175, 40.0, 0.05, 0.01, 0.05, 0.1
    phi = 3.0
    de = np.exp(-dt / tau_e)
    di = np.exp(-dt / tau_i)
    n_exc = 0
    for i in range(n):
        if not is_inh[i]:
            n_exc += 1

    acc = np.zeros(10)
    acc_n = 0
    spikes_e_window = 0
    new_ge = np.zeros(n)
    new_gi = np.zeros(n)
    ko_old = np.zeros(n)

    for step in range(steps):
        t = step * dt
        in_stim = stim_t0 <= t < stim_t1
        for i in range(n):
            ko_old[i] = ko[i]
        new_ge[:] = 0.0
        new_gi[:] = 0.0

        lfp = 0.0
        for i in range(n):
            vi = v[i]
            # --- gating rates (Cressman 2009) ---
            x = vi + 34.0
            an = 0.01 * x / (1.0 - np.exp(-0.1 * x)) if abs(x) > 1e-6 else 0.1
            bn = 0.125 * np.exp(-(vi + 44.0) / 80.0)
            x = vi + 30.0
            am = 0.1 * x / (1.0 - np.exp(-0.1 * x)) if abs(x) > 1e-6 else 1.0
            bm = 4.0 * np.exp(-(vi + 55.0) / 18.0)
            ah = 0.07 * np.exp(-(vi + 44.0) / 20.0)
            bh = 1.0 / (1.0 + np.exp(-0.1 * (vi + 14.0)))
            minf = am / (am + bm)

            # --- reversal potentials from current concentrations ---
            ki = 158.0 - nai[i]
            nao = 144.0 - BETA * (nai[i] - 18.0)
            ek = RTF * np.log(ko[i] / ki)
            ena = RTF * np.log(nao / nai[i])
            ecl = RTF * np.log(cli[i] / CL_O)

            # --- membrane currents (positive = outward), uA/cm2 ---
            gahp = 0.0 if is_inh[i] else g_ahp  # interneurons: no spike adaptation
            # Na-activated K+ current (Slack), pyramidal cells only. Compte et al. 2003
            # curve, EC50 shifted from 38.7 mM to this model's resting [Na]i (~19.5 vs ~11
            # mM) so it is ~5% of leak at rest and recruits as Na+ accumulates.
            gkna = 0.0 if is_inh[i] else g_kna * 0.37 / (1.0 + (KNA_EC50 / nai[i]) ** 3.5)
            # adenosine A1 receptor -> GIRK K+ channels (postsynaptic hyperpolarisation)
            girk = g_ado * ado[i] / (1.0 + ado[i])
            i_na = g_na * minf ** 3 * hg[i] * (vi - ena) + g_nal * (vi - ena)
            i_k = ((g_k * ng[i] ** 4 + gahp * ca[i] / (1.0 + ca[i]) + gkna + girk) * (vi - ek)
                   + g_kl * (vi - ek))
            i_cl = g_cll * (vi - ecl)
            # synapses
            i_ampa_na = P_NA_AMPA * ge[i] * (vi - ena)
            i_ampa_k = (1.0 - P_NA_AMPA) * ge[i] * (vi - ek)
            i_gaba_cl = GABA_CL_FRAC * gi[i] * (vi - ecl)
            i_gaba_hco3 = (1.0 - GABA_CL_FRAC) * gi[i] * (vi - E_HCO3)
            i_syn = i_ampa_na + i_ampa_k + i_gaba_cl + i_gaba_hco3
            if not is_inh[i]:
                lfp -= i_syn  # LFP ~ summed synaptic current into pyramidal cells

            # --- pumps, glia, KCC2, diffusion (mM/s) ---
            pump = (rho / (1.0 + np.exp((25.0 - nai[i]) / 3.0))) / (1.0 + np.exp(5.5 - ko[i]))
            glia = glia_s * (200.0 / 3.0) / (1.0 + np.exp((18.0 - ko[i]) / 2.5))
            diff = eps * (ko[i] - kbath)
            kcc2 = kcc2_u * np.log((ki * cli[i]) / (ko[i] * CL_O)) if cl_dynamic else 0.0
            if pump > PUMP_REST:
                ado[i] += dt / 1000.0 * ado_alpha * (pump - PUMP_REST)
            lat = 0.0
            if n > 1:
                lat = d_lat * (ko_old[(i - 1) % n] + ko_old[(i + 1) % n] - 2.0 * ko_old[i])

            # --- derivatives ---
            # electrogenic pump: 3 Na+ out / 2 K+ in = 1 net charge out per cycle
            i_pump = electrogenic * pump / GAMMA  # electrogenic = scale factor (0 = off)
            dv = -(i_na + i_k + i_cl + i_syn + i_pump)
            dko = (GAMMA * BETA * (i_k + i_ampa_k) - 2.0 * BETA * pump - glia - diff
                   + BETA * kcc2 + lat) / TAU
            dnai = (-GAMMA * (i_na + i_ampa_na) - 3.0 * pump) / TAU
            dcli = (GAMMA * (i_cl + i_gaba_cl) - kcc2) / TAU if cl_dynamic else 0.0
            dca = -0.002 * g_ca * (vi - 120.0) / (1.0 + np.exp(-(vi + 25.0) / 2.5)) - ca[i] / 80.0

            # Exponential Euler (exact for frozen conductances): stable even when
            # synaptic conductances become large during seizures
            g_tot = (g_na * minf ** 3 * hg[i] + g_nal
                     + g_k * ng[i] ** 4 + gahp * ca[i] / (1.0 + ca[i]) + gkna + girk + g_kl + g_cll
                     + ge[i] + gi[i])
            v_inf = vi + dv / g_tot  # dv = g_tot * (v_inf - v)
            v_new = v_inf + (vi - v_inf) * np.exp(-dt * g_tot)
            tn = 1.0 / (phi * (an + bn))
            ng[i] = an / (an + bn) + (ng[i] - an / (an + bn)) * np.exp(-dt / tn)
            th = 1.0 / (phi * (ah + bh))
            hg[i] = ah / (ah + bh) + (hg[i] - ah / (ah + bh)) * np.exp(-dt / th)
            ko[i] += dt * dko
            nai[i] += dt * dnai
            cli[i] += dt * dcli
            ca[i] += dt * dca

            # spike: upward crossing of 0 mV
            if vi < 0.0 <= v_new:
                if nsp < max_sp:
                    sp_t[nsp] = t
                    sp_i[nsp] = i
                    nsp += 1
                if not is_inh[i]:
                    spikes_e_window += 1
                # short-term depression: each spike releases a fraction U of resources
                eff = 1.0
                if std_u > 0:
                    eff = xres[i]
                    xres[i] -= std_u * xres[i]
                # A1 receptors also cut presynaptic transmitter release
                eff /= 1.0 + ado[i]
                for k in range(ptr[i], ptr[i + 1]):
                    if is_inh[i]:
                        new_gi[post[k]] += wts[k] * eff
                    else:
                        new_ge[post[k]] += wts[k] * eff
            v[i] = v_new

            acc[2] += ko[i]
            acc[3] += nai[i]
            acc[4] += cli[i]
            # E_GABA = reversal of the modelled GABA current, i.e. the chord-conductance
            # average of E_Cl and E_HCO3 (not the GHK form, which the current does not use)
            acc[5] += GABA_CL_FRAC * ecl + (1 - GABA_CL_FRAC) * E_HCO3
            acc[6] += ek
            acc[7] += pump
            acc[9] += ado[i]

        # synaptic conductance update + Poisson background drive
        r = rate_bg + (stim_rate if in_stim else 0.0)
        p_bg = r * dt / 1000.0
        for i in range(n):
            ge[i] = ge[i] * de + new_ge[i]
            gi[i] = gi[i] * di + new_gi[i]
            if std_u > 0:
                xres[i] = 1.0 - (1.0 - xres[i]) * rec_x
            ado[i] *= dec_ado
            if np.random.random() < p_bg:
                ge[i] += w_bg

        acc[1] += lfp / max(n_exc, 1)
        acc_n += 1
        if (step + 1) % rec_every == 0:
            k = (step + 1) // rec_every - 1
            rec[k, 0] = t
            rec[k, 1] = acc[1] / acc_n
            for j in range(2, 8):
                rec[k, j] = acc[j] / (acc_n * n)
            rec[k, 9] = acc[9] / (acc_n * n)
            rec[k, 8] = spikes_e_window / max(n_exc, 1) / (acc_n * dt / 1000.0)  # Hz/neuron
            acc[:] = 0.0
            acc_n = 0
            spikes_e_window = 0
        if (step + 1) % ko_every == 0:
            ko_map[(step + 1) // ko_every - 1] = ko

    state = np.empty((7, n))
    state[0], state[1], state[2], state[3] = v, ng, hg, ko
    state[4], state[5], state[6] = nai, cli, ca
    return rec, ko_map, sp_t[:nsp], sp_i[:nsp], state


def build_ring(n, frac_inh, radius, p_e, p_i, g_e_tot, g_i_tot, seed):
    """Ring network: every 1/frac_inh-th cell is inhibitory; local random wiring."""
    rng = np.random.default_rng(seed)
    is_inh = np.zeros(n, bool)
    is_inh[:: int(round(1 / frac_inh))] = True
    edges = []  # (pre, post, w)
    for post_i in range(n):
        pres = [(post_i + d) % n for d in range(-radius, radius + 1) if d != 0]
        e_pre = [j for j in pres if not is_inh[j] and rng.random() < p_e]
        i_pre = [j for j in pres if is_inh[j] and rng.random() < p_i]
        for j in e_pre:
            edges.append((j, post_i, g_e_tot / max(len(e_pre), 1)))
        for j in i_pre:
            edges.append((j, post_i, g_i_tot / max(len(i_pre), 1)))
    edges.sort()
    ptr = np.zeros(n + 1, np.int64)
    for pre, _, _ in edges:
        ptr[pre + 1] += 1
    ptr = np.cumsum(ptr)
    post = np.array([e[1] for e in edges], np.int64)
    wts = np.array([e[2] for e in edges], np.float64)
    return is_inh, ptr, post, wts


def balloon(rate, dt_s, baseline, gain):
    """Balloon-Windkessel (Friston 2000): neural activity -> CBF, CBV, HbR, HbO.

    Returns % changes. Drive u = gain * ln(rate / baseline), clipped to [0.3x, 30x]
    baseline: neurovascular coupling saturates, so a 10x rise in firing gives a
    physiological ~+50-100% CBF; below-baseline activity (postictal depression)
    constricts vessels (postictal hypoperfusion).
    """
    kappa, gam, tau0, alpha, e0 = 0.65, 0.41, 0.98, 0.32, 0.34
    s, f, vol, q = 0.0, 1.0, 1.0, 1.0
    out = np.zeros((len(rate), 4))
    u = gain * np.log(np.clip(rate / baseline, 0.3, 30.0))
    for k, uk in enumerate(u):
        ds = uk - kappa * s - gam * (f - 1.0)
        df = s
        dvol = (f - vol ** (1 / alpha)) / tau0
        dq = (f * (1 - (1 - e0) ** (1 / f)) / e0 - vol ** (1 / alpha) * q / vol) / tau0
        s += dt_s * ds
        f = max(f + dt_s * df, 0.05)
        vol += dt_s * dvol
        q += dt_s * dq
        out[k] = f, vol, q, 0.0
    hbr0 = 0.4  # baseline deoxy fraction of total haemoglobin (venous-weighted)
    hbt = out[:, 1]
    hbr = out[:, 2]
    hbo = (hbt - hbr0 * hbr) / (1 - hbr0)
    return 100 * (out[:, 0] - 1), 100 * (hbo - 1), 100 * (hbr - 1)


# ------------------------------------------------------------------ scenarios
def validate(args):
    """Single neuron, fixed Cl, kbath = 8 mM: Cressman 2009 Fig. 1."""
    one = lambda x: np.array([x], float)
    # Original Cressman equations: no I_KNa, no electrogenic pump current, fixed Cl-
    rec, _, sp_t, _, _ = _simulate(
        args.t * 1000, 0.01, 1, np.zeros(1, bool), np.zeros(2, np.int64),
        np.zeros(0, np.int64), np.zeros(0), 8.0, 1.0, 0.0, False,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0 / 3.0, 1.25, 5.0, 8.0, 100, 10 ** 9, 0,
        one(-50.0), one(0.08), one(0.96), one(7.8), one(15.5), one(6.0), one(0.0),
        0.0, 0.0, 0.0, 700.0, 0.0, 0.0, 1000.0)
    t = rec[:, 0] / 1000
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].eventplot(sp_t / 1000, lineoffsets=0, colors="k", linewidths=0.3)
    ax[0].set_ylabel("spikes")
    ax[0].set_yticks([])
    ax[1].plot(t, rec[:, 2], "k")
    ax[1].set_ylabel("[K]o (mM)")
    ax[2].plot(t, rec[:, 3], "k--")
    ax[2].set_ylabel("[Na]i (mM)")
    ax[2].set_xlabel("time (s)")
    fig.suptitle("Validation: Cressman et al. 2009 single neuron, bath K+ = 8 mM\n"
                 "(paper Fig. 1: seizure-like events every ~30 s, [K]o 7-10 mM, [Na]i 16-19 mM)")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "ionic_validation.png", dpi=130)
    plt.close(fig)

    bursts = np.flatnonzero(np.diff(sp_t) > 2000) if len(sp_t) else []
    print(f"[validate] spikes={len(sp_t)}  events~{len(bursts) + (len(sp_t) > 0)}  "
          f"[K]o range {rec[:, 2].min():.2f}-{rec[:, 2].max():.2f} mM  "
          f"[Na]i range {rec[:, 3].min():.2f}-{rec[:, 3].max():.2f} mM")


SCENARIOS = {
    # Multipliers of normal strength. The epileptic set mirrors hippocampal sclerosis:
    # KCC2 downregulation, loss of astrocytic Kir4.1 K+ buffering, and gliosis that
    # slows K+ clearance into the vasculature.
    "healthy": dict(kcc2=1.0, glia=1.0, clear=1.0),
    "epileptic": dict(kcc2=0.3, glia=0.3, clear=0.35),
}


def network(args):
    n = args.n
    is_inh, ptr, post, wts = build_ring(n, 0.2, args.radius, 0.6, 0.8,
                                        args.g_exc, args.g_inh, args.seed)
    kcc2_u = args.kcc2_base * args.kcc2
    eps = args.clear * 4.0 / 3.0

    def run(t_ms, stim_on, stim_off, init):
        return _simulate(
            t_ms, args.dt, n, is_inh, ptr, post, wts, args.kbath, args.glia,
            kcc2_u, True, args.bg_rate, args.bg_w, stim_on, stim_off,
            args.stim_rate, args.d_lat, eps, 1.25, 5.0, 8.0,
            int(1.0 / args.dt), int(10.0 / args.dt), args.seed,
            *init, args.g_kna, args.pump_i, args.std_u, 700.0,
            args.g_ado, args.ado_alpha, args.ado_tau * 1000)

    init = (np.full(n, -70.0), np.full(n, 0.07), np.full(n, 0.97), np.full(n, args.kbath),
            np.full(n, 21.0), np.full(n, 4.8), np.zeros(n))
    # Warm-up: let [Na]i, [Cl]i and [K]o settle to this tissue's own resting state
    *_, state = run(args.warmup * 1000, 1e12, 1e12, init)
    rec, ko_map, sp_t, sp_i, _ = run(args.t * 1000, args.stim_on * 1000, args.stim_off * 1000,
                                     tuple(state[k].copy() for k in range(7)))
    return rec, ko_map, sp_t, sp_i, is_inh


def count_ieds(t, rate, baseline, seizures):
    """Interictal epileptiform discharges: brief population bursts outside seizures."""
    fast = np.convolve(rate, np.ones(20) / 20, mode="same")  # 20 ms smoothing
    thr = max(50.0, 8 * baseline)
    peaks = np.flatnonzero((fast[1:-1] > thr) & (fast[1:-1] >= fast[:-2]) & (fast[1:-1] > fast[2:])) + 1
    keep, last = [], -1e9
    for p in peaks:
        if t[p] - last > 0.2 and not any(t[s0] <= t[p] <= t[s1 - 1] for s0, s1 in seizures):
            keep.append(p)
            last = t[p]
    return np.array(keep, int)


def detect_seizures(t, rate_s, baseline, min_dur=5.0, merge_gap=1.0):
    """Seizure = 1 s-smoothed rate above max(20 Hz, 4x baseline) for >= min_dur s.

    Brief interictal bursts do not qualify: they are too short to raise the
    1 s average for several seconds."""
    high = rate_s > max(20.0, 4 * baseline)
    edges = np.diff(np.r_[0, high.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    segs = []
    for s0, s1 in zip(starts, ends):
        if segs and t[s0] - t[segs[-1][1] - 1] < merge_gap:
            segs[-1][1] = s1
        else:
            segs.append([s0, s1])
    return [(s0, s1) for s0, s1 in segs if t[s1 - 1] - t[s0] >= min_dur]


def plot_network(args, rec, ko_map, sp_t, sp_i, is_inh, name):
    t = rec[:, 0] / 1000
    base_mask = t < args.stim_on
    rate = rec[:, 8]
    rate_s = np.convolve(rate, np.ones(1000) / 1000, mode="same")  # 1 s window
    baseline = max(rate_s[base_mask].mean(), 0.1) if base_mask.any() else 1.0
    cbf, hbo, hbr = balloon(rate_s, 0.001, baseline, args.nv_gain)
    seizures = detect_seizures(t, rate_s, baseline)
    ieds = count_ieds(t, rate, baseline, seizures)

    fig, ax = plt.subplots(8, 1, figsize=(13, 17), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1, 1.6, 1, 1, 1.2, 1, 1.3]})
    colors = np.where(is_inh[sp_i], "tab:red", "k")
    ax[0].scatter(sp_t / 1000, sp_i, s=0.15, c=colors, rasterized=True)
    ax[0].set_ylabel("neuron\n(red = inh.)")
    ax[1].plot(t, rec[:, 1], "tab:green", lw=0.4)
    ax[1].set_ylabel("LFP proxy\n(uA/cm2)")
    im = ax[2].imshow(ko_map.T, aspect="auto", origin="lower", cmap="magma",
                      extent=[0, t[-1], 0, args.n], vmin=args.kbath, vmax=max(ko_map.max(), args.kbath + 1))
    ax[2].set_ylabel("[K]o per cell\n(space)")
    fig.colorbar(im, ax=ax[2], pad=0.005, fraction=0.02, label="mM")
    ax[3].plot(t, rec[:, 2], "tab:purple")
    ax[3].axhline(args.kbath, color="grey", ls=":", lw=0.8)
    ax[3].set_ylabel("mean [K]o\n(mM)")
    ax[4].plot(t, rec[:, 3], "tab:orange")
    ax[4].set_ylabel("mean [Na]i\n(mM)")
    ax[5].plot(t, rec[:, 5], "tab:blue", label="E_GABA")
    ax[5].plot(t, rec[:, 6], "tab:red", label="E_K")
    ax[5].axhline(-60, color="grey", ls=":", lw=0.8)
    ax[5].set_ylabel("reversal\n(mV)")
    ax5b = ax[5].twinx()
    ax5b.plot(t, rec[:, 4], "tab:cyan", lw=0.8, label="[Cl]i")
    ax5b.set_ylabel("[Cl]i (mM)", color="tab:cyan")
    ax[5].legend(loc="upper left", fontsize=8, ncol=2)
    ax[6].plot(t, rec[:, 7], "tab:brown")
    ax[6].set_ylabel("Na/K pump\n(mM/s)")
    ax6b = ax[6].twinx()
    ax6b.plot(t, rec[:, 9], "tab:olive", lw=1)
    ax6b.set_ylabel("adenosine\n(x A1 EC50)", color="tab:olive")
    ax[7].plot(t, cbf, "k", label="CBF")
    ax[7].plot(t, hbo, "tab:red", label="HbO")
    ax[7].plot(t, hbr, "tab:blue", label="HbR")
    ax[7].axhline(0, color="grey", lw=0.6)
    ax[7].set_ylabel("haemodynamics\n(% change)")
    ax[7].legend(loc="upper right", fontsize=8, ncol=3)
    ax[7].set_xlabel("time (s)")
    for a in ax:
        a.axvspan(args.stim_on, args.stim_off, color="gold", alpha=0.25, lw=0)
        for s0, s1 in seizures:
            a.axvspan(t[s0], t[s1 - 1], color="tab:red", alpha=0.08, lw=0)
    fig.suptitle(f"Ionic seizure model - {name}  (KCC2 x{args.kcc2}, glia x{args.glia}, "
                 f"K+ clearance x{args.clear}; yellow = trigger, red = detected seizure)")
    fig.tight_layout()

    # --- summary metrics (printed before saving, so a locked PNG cannot lose them) ---
    b = base_mask
    print(f"[{name}] baseline: E rate {baseline:.1f} Hz | [K]o {rec[b, 2].mean():.2f} mM | "
          f"[Na]i {rec[b, 3].mean():.1f} mM | [Cl]i {rec[b, 4].mean():.1f} mM | "
          f"E_GABA {rec[b, 5].mean():.1f} mV")
    minutes = (t[-1] - t[0]) / 60
    print(f"[{name}] interictal discharges: {len(ieds)} ({len(ieds) / minutes:.0f}/min) | "
          f"seizures detected: {len(seizures)}")
    for k, (s0, s1) in enumerate(seizures, 1):
        seg = slice(s0, s1)
        post = slice(s1, min(s1 + 20000, len(t)))  # 20 s after offset
        line = (f"  #{k}: {t[s0]:.1f}-{t[s1 - 1]:.1f} s ({t[s1 - 1] - t[s0]:.1f} s) | "
                f"peak rate {rate_s[seg].max():.0f} Hz | peak [K]o {rec[seg, 2].max():.1f} mM | "
                f"peak [Na]i {rec[seg, 3].max():.1f} | peak [Cl]i {rec[seg, 4].max():.1f} | "
                f"E_GABA {rec[seg, 5].max():.0f} mV | CBF {cbf[seg].max():+.0f}%")
        if s1 < len(t):
            line += (f" | postictal: min rate {rate_s[post].min():.1f} Hz, "
                     f"min [K]o {rec[post, 2].min():.2f} mM, CBF {cbf[post].min():+.0f}%")
        print(line)

    # Response to the trigger (too short to count as a >=5 s seizure, but the key readout)
    trig = (t >= args.stim_on) & (t < args.stim_off + 3)
    after = t >= args.stim_off
    if trig.any() and after.any():
        # start 1 s after the trigger: the 1 s smoothing window still contains it before that
        recovered = np.flatnonzero((t >= args.stim_off + 1.0) & (rate_s > 0.5 * baseline))
        silence = t[recovered[0]] - args.stim_off if len(recovered) else float("nan")
        win = after & (t < args.stim_off + 30)
        print(f"[{name}] trigger response: peak [K]o {rec[trig, 2].max():.1f} mM | "
              f"peak [Na]i {rec[trig, 3].max():.1f} mM | peak [Cl]i {rec[trig, 4].max():.1f} mM | "
              f"peak E_GABA {rec[trig, 5].max():.0f} mV | peak adenosine {rec[:, 9].max():.2f}")
        print(f"[{name}] after trigger: silence {silence:.1f} s | min [K]o {rec[win, 2].min():.2f} mM | "
              f"CBF peak {cbf[trig].max():+.0f}% then min {cbf[win].min():+.0f}%")

    OUT.mkdir(exist_ok=True)
    path = OUT / f"ionic_{name}.png"
    try:
        fig.savefig(path, dpi=110)
    except OSError:  # e.g. the PNG is open (locked) in an image viewer on Windows
        path = OUT / f"ionic_{name}_{time.strftime('%Y%m%d-%H%M%S')}.png"
        fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[{name}] saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", choices=["validate", *SCENARIOS])
    p.add_argument("--t", type=float, default=None, help="seconds to simulate")
    p.add_argument("--dt", type=float, default=0.02, help="ms")
    p.add_argument("--n", type=int, default=100, help="neurons (20%% inhibitory)")
    p.add_argument("--radius", type=int, default=10, help="connection radius (cells)")
    p.add_argument("--kcc2", type=float, default=None, help="KCC2 strength (x normal)")
    p.add_argument("--glia", type=float, default=None, help="glial K+ uptake (x normal)")
    p.add_argument("--clear", type=float, default=None, help="vascular K+ clearance (x normal)")
    p.add_argument("--kcc2-base", type=float, default=0.4, help="normal KCC2 rate U (mM/s)")
    p.add_argument("--kbath", type=float, default=4.0, help="blood/bath K+ (mM)")
    p.add_argument("--g-exc", type=float, default=1.3, help="total recurrent AMPA (mS/cm2)")
    p.add_argument("--g-inh", type=float, default=2.6, help="total recurrent GABA-A (mS/cm2)")
    p.add_argument("--g-kna", type=float, default=1.33, help="max Na-activated K+ (mS/cm2)")
    p.add_argument("--std-u", type=float, default=0.0, help="synaptic depression U (0 = off)")
    p.add_argument("--g-ado", type=float, default=0.3, help="max adenosine-GIRK (mS/cm2)")
    p.add_argument("--ado-alpha", type=float, default=0.5, help="adenosine per excess pump flux")
    p.add_argument("--ado-tau", type=float, default=20.0, help="adenosine clearance (s)")
    p.add_argument("--bg-rate", type=float, default=60.0, help="background Poisson rate (Hz)")
    p.add_argument("--bg-w", type=float, default=0.07, help="background AMPA jump (mS/cm2)")
    p.add_argument("--warmup", type=float, default=60.0, help="settling time before t=0 (s)")
    p.add_argument("--stim-on", type=float, default=10.0, help="trigger start (s)")
    p.add_argument("--stim-off", type=float, default=12.0, help="trigger end (s)")
    p.add_argument("--stim-rate", type=float, default=400.0, help="extra input during trigger (Hz)")
    p.add_argument("--d-lat", type=float, default=0.625, help="lateral K+ diffusion D/dx^2 (1/s)")
    p.add_argument("--nv-gain", type=float, default=0.12, help="neurovascular coupling gain")
    p.add_argument("--pump-i", type=float, default=1.0,
                   help="electrogenic pump current scale (calibrate to 1-10 mV post-train AHP)")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--save", action="store_true", help="also save raw arrays to output/ionic_<scenario>.npz")
    args = p.parse_args()

    t0 = time.time()
    if args.scenario == "validate":
        args.t = args.t or 100.0
        validate(args)
    else:
        for k, v in SCENARIOS[args.scenario].items():
            if getattr(args, k) is None:
                setattr(args, k, v)
        args.t = args.t or 90.0
        res = network(args)
        if args.save:
            rec, ko_map, sp_t, sp_i, is_inh = res
            OUT.mkdir(exist_ok=True)
            np.savez_compressed(OUT / f"ionic_{args.scenario}.npz", rec=rec, ko_map=ko_map, sp_t=sp_t,
                                sp_i=sp_i, is_inh=is_inh, stim_on=args.stim_on, stim_off=args.stim_off,
                                nv_gain=args.nv_gain, kcc2=args.kcc2, glia=args.glia, clear=args.clear)
        plot_network(args, *res, args.scenario)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

# epilepsy-sim

Seizure simulations in Python, managed with uv:

- `main.py`: two classic, fast models (spiking network and Epileptor).
- `ionic.py`: a biophysical model with potassium, sodium, chloride and adenosine dynamics, numba-compiled.

```
uv run main.py                                   # both models -> output/*.png
uv run main.py --model network --inhibition 0.3  # spiking network
uv run main.py --model epileptor --x0 -2.2       # healthy brain: no seizures
```

## Models

**1. Spiking neural network** (Izhikevich 2003): 800 excitatory + 200 inhibitory neurons.
Between `--onset` and `--offset` ms, GABA inhibition is scaled to `--inhibition`
(this mimics interneuron exhaustion and chloride loading). The network switches from
asynchronous firing (synchrony ~0.02) to hypersynchronous population bursts
(synchrony ~0.98), then recovers when inhibition returns.

**2. Epileptor** (Jirsa et al., *Brain* 2014): the standard phenomenological seizure
model, also used in The Virtual Brain. A slow variable `z` (energy / ion build-up)
drives spontaneous seizure onset and offset. `x0` is tissue excitability:

- `x0 = -1.6`: recurrent seizures
- `x0 = -2.2`: healthy, no seizures

## Options

| Flag | Model | Meaning |
|---|---|---|
| `--inhibition` | network | GABA strength during the seizure window (1 = normal) |
| `--onset` / `--offset` | network | when inhibition drops and returns (ms) |
| `--duration` | network | simulation length (ms) |
| `--x0` | epileptor | excitability |
| `--noise` | epileptor | stochastic noise on the fast subsystem |
| `--t-end` | epileptor | simulation length (time units) |

---

## ionic.py: ion-concentration seizure model

**Full documentation (equations, parameters, validation, results, calibration log): [SIMULATION.md](SIMULATION.md)**

**Inference twin experiments (`infer.py`) and The Virtual Brain comparison (`tvb_compare.py`): [COMPARISON.md](COMPARISON.md)**

```
uv run ionic.py validate            # ~7 s: reproduces Cressman et al. 2009 Fig. 1
uv run ionic.py healthy --t 120     # ~3 min
uv run ionic.py epileptic --t 300   # ~6 min
```

Each network run simulates 100 neurons (80 excitatory, 20 inhibitory) on a ring. It does a
60 s warm-up first, then applies a 2 s trigger at t = 10 s.

### What is modelled

| Component | Source |
|---|---|
| Hodgkin-Huxley neuron, dynamic [K]o and [Na]i, Na/K pump, glial K+ uptake, vascular K+ clearance | Cressman et al. 2009, using the corrected ModelDB code (see the 2011 erratum) |
| Dynamic [Cl]i, KCC2 (extrudes Cl- and K+), GABA-A with Cl-/HCO3- (so E_GABA follows [Cl]i) | Krishnan & Bazhenov 2011; Doyon et al. 2016 |
| AMPA Na+/K+ fluxes feeding the concentration equations | - |
| Lateral K+ diffusion through extracellular space (seizure spread) | D = 2.5e-6 cm2/s, 20 um spacing |
| Electrogenic pump current, Na+-activated K+ current (Slack) | Krishnan & Bazhenov 2011; Compte et al. 2003 |
| Adenosine from pump ATP use, acting on A1 receptors (GIRK current + reduced release) | During & Spencer 1992 |
| Blood flow / HbO / HbR (Balloon-Windkessel model), i.e. the optical-sensor view | Friston et al. 2000 |

### Scenarios

The epileptic scenario models hippocampal-sclerosis pathology:

| Scenario | KCC2 | Glial K+ buffering | Vascular K+ clearance |
|---|---|---|---|
| healthy | ×1.0 | ×1.0 | ×1.0 |
| epileptic | ×0.3 | ×0.3 | ×0.35 |

### Validation and results so far (full-strength pump current, `--pump-i 1`)

- **validate:** matches Cressman Fig. 1. Events every ~30 s, [K]o 6.7-10.7 mM, [Na]i 15.5-19.1 mM.
- **healthy:**
    - Resting state: asynchronous firing at ~5 Hz, [Cl]i 5.6 mM, E_GABA -71 mV.
    - The trigger gives a transient response with [K]o up to 6 mM, then about 5 s of suppression. No seizure.
- **epileptic:**
    - Resting state: [Cl]i 11.6 mM and E_GABA -58 mV (depolarised, as measured in human epileptic tissue).
    - Spontaneous interictal discharges at ~1/s, with K+ transients to 7-9 mM.
    - The same trigger gives an ictal event: [K]o 12 mM (the physiological "K+ ceiling"), [Cl]i 15 mM and [Na]i 30 mM.
    - The event is followed by an adenosine surge (0.68), 25 s of postictal silence with the K+ undershooting baseline (3.66 mM), and a +31% blood-flow surge followed by postictal hypoperfusion (-46%).

### Known limitation

With the full pump current, the triggered ictal event ends when the trigger ends. It is not
self-sustaining, whereas human focal seizures run 30-120 s on their own.

- With the pump current at `--pump-i 0.15` (calibrated to the measured 1-10 mV post-train
  hyperpolarisation, Gulledge et al. 2013), the network produces realistic self-sustained
  seizures: 10-20 s long, self-terminating, with 30-40 s of postictal silence, recurring.
- At that setting, however, healthy tissue loses its low-rate asynchronous state. It is either
  silent or seizing.
- An intermediate setting (`--pump-i 0.35`, bg drive ~0.045-0.055, recurrent E/I 0.7/1.4) is the
  next thing to test. That sweep was not completed.

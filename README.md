# Epilepsy Temple

### Toward an ear-worn seizure monitor: from research, to physics-based simulation, to detection on real patient EEG

> **What this is.** A research project exploring whether a small wearable (inspired by Deepinder Goyal's temple-worn *Temple* device, redesigned as a **"?"-shaped ear-hook**) could detect epileptic seizures.
>
> It covers:
>
> - a technical teardown of Temple;
> - epilepsy and seizure physiology;
> - a validated biophysical seizure simulator with potassium, sodium, chloride, adenosine and blood-flow dynamics;
> - mathematics for inferring hidden brain chemistry from non-invasive sensors;
> - a cross-check against **The Virtual Brain**;
> - seizure detectors trained and tested on **real hospital EEG** (CHB-MIT, 23 subjects, 185 seizures);
> - a live, single-screen visual lab of the whole thing.
>
> ⚠️ **Research prototype, not a medical device.** Nothing here is validated for clinical use.

![Epilepsy Live Lab](docs/images/live_lab.png)
<sub>The live lab (`epilepsy_sim/viz/epilepsy_live.html`) during a simulated seizure. The orb field shows neurons, K⁺, Na⁺, Cl⁻, adenosine and blood; nine live charts show the hidden variables; a virtual ear-hook EEG and detector alarm sit below.</sub>

---

## Contents

1. [Key findings](#1-key-findings)
2. [Project journey](#2-project-journey)
3. [Repository map](#3-repository-map)
4. [Quick start](#4-quick-start)
5. [The device concept: a "?" ear-hook](#5-the-device-concept-a--ear-hook)
6. [Results](#6-results)
7. [Handling false alarms](#7-handling-false-alarms)
8. [Honest limitations](#8-honest-limitations)
9. [Roadmap](#9-roadmap)
10. [Documents index](#10-documents-index)
11. [Data, licences and disclaimer](#11-data-licences-and-disclaimer)
12. [Key references](#12-key-references)

---

## 1. Key findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **A temple-worn optical sensor alone cannot detect seizures.** Temple measures scalp blood flow (PPG), not brain electrical activity. | `Temple_Technical_Dossier.md` |
| 2 | **Hidden seizure chemistry can be *inferred*, not measured, from outside the head.** Formulas were derived and audited; 3 errors in early drafts were caught and fixed. | `Inference_Formulas.md` |
| 3 | **A physics-based simulator reproduces real seizure hallmarks.** It shows K⁺ reaching the ~12 mM ceiling, chloride loading, depolarising GABA, an adenosine surge, postictal silence, a K⁺ undershoot and postictal hypoperfusion. It is validated against Cressman et al. 2009. | `epilepsy_sim/SIMULATION.md` |
| 4 | **The derived inference methods recover hidden values in twin experiments.** KCC2 strength was inferred as 0.36 ± 0.09 (true 0.30). Naive formulas fail exactly where theory predicts (e.g. 2.4× K⁺ error from scalp DC). | `epilepsy_sim/COMPARISON.md` |
| 5 | **Our Epileptor matches The Virtual Brain's official implementation.** First seizure onset is identical (886.2), and differences vanish as the time step shrinks. | `epilepsy_sim/COMPARISON.md` |
| 6 | **On real EEG, temple and ear electrodes catch seizures as well as a full cap, but with more false alarms.** Patient-specific (3 patients, 19 seizures): both temples caught 100% with 9.8 false alarms/day; the full cap caught 95% with 5.2/day. | `epilepsy_sim/detect/` |
| 7 | **The ear-hook (2 channels) catches as many seizures as a full 18-channel cap, but with ~2× the false alarms.** On 23 subjects and 185 seizures: new user 88% vs 85%; after fine-tuning on 2 own seizures 86% (162 FA/day) vs 89% (88 FA/day). Both ears beat one ear on every measure. | `epilepsy_sim/detect/general.py` |
| 8 | **False alarms are the real problem.** EEG-only models give 90–550 false alarms/day. Per-person normalisation and fine-tuning cut them by 40–60%, but reaching ≤1/day needs multimodal fusion (heart rate, motion, jaw EMG). | §7 |

---

## 2. Project journey

```mermaid
flowchart LR
    A[1. Research<br/>Temple device,<br/>epilepsy] --> B[2. Physiology<br/>what happens<br/>in a seizure]
    B --> C[3. Simulation<br/>ions, neurons,<br/>blood]
    C --> D[4. Inference<br/>formulas for<br/>hidden variables]
    D --> E[5. Validation<br/>twin experiments +<br/>The Virtual Brain]
    E --> F[6. Real EEG<br/>CHB-MIT detection<br/>temple vs full cap]
    F --> G[7. General model<br/>23 subjects +<br/>fine-tuning]
    G --> H[8. Device concept<br/>'?' ear-hook]
    C --> L[Live Lab<br/>visualisation]
    F --> L
```

| Phase | Question | Output |
|---|---|---|
| 1 | What exactly is Temple, technically? | `Temple_Technical_Dossier.md` |
| 2 | What is epilepsy, and what happens in the brain during a seizure? | `Epilepsy_Overview.md`, `Seizure_Neurophysiology.md` |
| 3 | Can we simulate seizures realistically, including ion concentrations? | `epilepsy_sim/ionic.py`, `SIMULATION.md` |
| 4 | Which hidden variables can a wearable infer, and with what formulas? | `Inference_Formulas.md` |
| 5 | Do the formulas work? Does our code match an established simulator? | `infer.py`, `tvb_compare.py`, `COMPARISON.md` |
| 6 | How well do temple or ear electrodes detect seizures on real EEG? | `detect/detect.py` |
| 7 | Can one model work for new users? How much does personal fine-tuning help? | `detect/general.py` |
| 8 | What should the device look like? | §5 |

---

## 3. Repository map

```
epilepsy_temple/
├── README.md                        ← you are here
├── Temple_Technical_Dossier.md      Temple wearable: hardware, software, validation, gaps
├── Epilepsy_Overview.md             causes, seizure types, diagnosis, treatment, wearables
├── Seizure_Neurophysiology.md       second-by-second: ions, cells, networks, blood
├── Inference_Formulas.md            derivations + verification audit for inferring hidden variables
├── docs/images/                     README figures
└── epilepsy_sim/                    all code (Python 3.13, uv)
    ├── main.py                      Model 1: Izhikevich network · Model 2: Epileptor
    ├── ionic.py                     Model 3: biophysical K⁺/Na⁺/Cl⁻/adenosine network (numba)
    ├── infer.py                     twin experiments for the derived inference formulas (E1–E7)
    ├── tvb_compare.py               cross-validation + VEP-style inference with The Virtual Brain
    ├── SIMULATION.md                full model documentation (equations, parameters, calibration log)
    ├── COMPARISON.md                inference + TVB results
    ├── output/                      figures and metrics from the simulations
    ├── detect/
    │   ├── fetch_chbmit.py          download a CHB-MIT subset (PhysioNet AWS mirror)
    │   ├── detect.py                patient-specific detection: full cap vs temple/ear channels
    │   ├── general.py               general model on all CHB-MIT subjects + per-person fine-tuning
    │   ├── notify_watch.sh          optional: ring your phone (ntfy) when a long run finishes
    │   ├── notify_final.sh          optional: ring your phone when general.py finishes / crashes / stops
    │   ├── results/                 3-patient detection metrics + figure
    │   └── results_general/         23-subject general-model metrics
    └── viz/
        ├── epilepsy_live.html       Epilepsy Live Lab (open in a browser, works offline)
        ├── data.js                  data for the lab (regenerate with export_data.py)
        └── export_data.py           packs simulation + EEG data for the lab
```

---

## 4. Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.13 (uv installs it). All commands run from `epilepsy_sim/`.

```bash
cd epilepsy_sim
uv sync                                    # numpy, scipy, matplotlib, numba, mne, scikit-learn, fooof
```

| What | Command | Time |
|---|---|---|
| **Open the live lab** | double-click `viz/epilepsy_live.html` | instant |
| Classic models (spiking network + Epileptor) | `uv run main.py` | ~10 s |
| Validate the ionic model (Cressman 2009 Fig. 1) | `uv run ionic.py validate` | ~7 s |
| Healthy / epileptic tissue | `uv run ionic.py healthy --t 90 --save` / `uv run ionic.py epileptic --t 120 --save` | ~6 / 7 min |
| Inference twin experiments | `uv run infer.py --abc` | ~12 min |
| The Virtual Brain comparison | `uv run --extra tvb tvb_compare.py` | ~12 min |
| Download CHB-MIT subset (3 patients) | `uv run detect/fetch_chbmit.py` | ~4 min, 1.4 GB |
| Temple vs full-cap detection | `uv run detect/detect.py --configs 4` (then 2, 3, 1, 0) | ~2–40 min each |
| General model, all subjects (streaming, ~1 GB disk) | `uv run detect/general.py fetch` then `run` (add `--configs "Full cap"` to re-run one setup; progress bars show time left) | ~1 h + 2–3 h |
| Rebuild live-lab data | `uv run viz/export_data.py` | ~2 min |

> **Memory:** network simulations use ~1–2 GB. On machines with little free RAM, run one long job at a time.

---

## 5. The device concept: a "?" ear-hook

Temple sits on the temple and measures blood flow optically. Our research points to an **ear-hook, worn on both ears**, carrying several sensors:

```
        ___
      /     \   ← top curve: dry EEG electrodes above/behind the ear (≈ T7/T8, P7/P8)
     |       |
      \      |  ← back stem: reference electrode on the mastoid bone
       ?     |
       |    /   ← ear-canal tip: in-ear EEG + PPG (heart rate, blood flow) + temperature
       •   /    ← optional earlobe PPG
   + accelerometer/gyroscope · jaw-muscle EMG · Bluetooth to phone
```

| Why the ear | Evidence |
|---|---|
| Sits over the temporal lobe, the most common seizure origin in adults | Ear-EEG matched scalp EEG for temporal-lobe seizures ([Zibrandtsen 2017](https://www.sciencedirect.com/science/article/pii/S1388245717310763)) |
| Proven for seizure detection | Behind-the-ear 2-channel wearable caught 98% of absence seizures ([Swinnen 2021](https://onlinelibrary.wiley.com/doi/10.1111/epi.17061)) |
| Best PPG site | Earlobe PPG up to 6.3× better signal than the wrist ([source](https://pmc.ncbi.nlm.nih.gov/articles/PMC13210545/)) |
| Discreet and stable | Looks like a hearing aid or earphone; held in place by the ear |
| **Both ears > one ear** | Our 3-patient test: one-sided setups gave 3–6× more false alarms |

---

## 6. Results

### 6.1 Simulation: healthy vs epileptic tissue (`ionic.py`)

The two tissues share the same neurons, wiring and 2 s trigger. Only the pathology differs: KCC2 ×0.3, glial K⁺ buffering ×0.3 and K⁺ clearance ×0.35, modelled on hippocampal sclerosis.

| Measure | Healthy | Epileptic |
|---|---|---|
| Resting [Cl]i / E_GABA | 5.6 mM / −69.6 mV | **11.6 mM / −54.1 mV** |
| Spontaneous activity | Asynchronous ~6 Hz | Interictal discharges (~35/min) |
| Peak [K]o after trigger | 6.1 mM | **12.2 mM** (physiological ceiling) |
| Peak [Cl]i / E_GABA | 7.4 mM / −63.7 mV | **15.2 mM / −48.3 mV** |
| Adenosine surge | 0.12 | **0.68** |
| Postictal silence | 13.7 s | **25 s**, with a K⁺ undershoot to 3.66 mM |
| Blood flow (peak → dip) | +42% → −47% | +31% → −46% |

Validation: the model reproduces Cressman et al. 2009 Fig. 1 (events every ~30 s, [K]o 6.7–10.7 mM). Full details: [`epilepsy_sim/SIMULATION.md`](epilepsy_sim/SIMULATION.md).

### 6.2 Inference: can hidden variables be recovered? (`infer.py`)

| Test | Naive / wrong approach | Derived method | Truth |
|---|---|---|---|
| K⁺ from glial electrode | Peak 10.3 mM (textbook slope) | **12.3 mM** (measured 51 mV/decade) | 12.2 mM |
| K⁺ from scalp DC-EEG | Peak 29.2 mM (2.4× high) | **0.50 mM RMSE** (Kalman filter) | – |
| Cl⁻ from E_GABA | +2.7 mM bias (wrong formula) | **1.1 mM RMSE** | – |
| HbO from fNIRS | Amplitude 22% of true | **1.4 µM RMSE**, r = 0.996 | – |
| KCC2 strength (Bayesian) | – | **0.36 ± 0.09** | 0.30 |

### 6.3 Cross-check with The Virtual Brain (`tvb_compare.py`)

| | TVB official | Ours, dt 0.01 |
|---|---|---|
| First seizure onset | 886.2 | **886.2** |
| Seizure duration | 968.7 | 976.2 |
| z RMSE vs TVB | – | 0.018 |

On a 76-region connectome, a right-amygdala seizure spread to 36/38 right-hemisphere regions. Onset-time inference located the focus correctly. That result is optimistic, because the same model generated and fitted the data. Details: [`COMPARISON.md`](epilepsy_sim/COMPARISON.md).

### 6.4 Real EEG, patient-specific: 3 patients, 19 seizures, 36.6 h (`detect/detect.py`)

| Electrodes | Seizures caught | False alarms / 24 h | Median delay |
|---|---|---|---|
| Full cap (18 ch) | 95% | **5.2** | 4.5 s |
| **Both temples (4 ch)** | **100%** | 9.8 | 6 s |
| Right temple (2 ch) | 100% | 36.0 | 5 s |
| Left temple (2 ch) | 100% | 32.1 | 10 s |
| Right temple (1 ch) | 100% | 62.9 | 5 s |

![Detection](epilepsy_sim/detect/results/detection.png)

### 6.5 Real EEG, general model: 23 subjects, 185 seizures, 207 recordings (`detect/general.py`)

**New user, day one** (leave-one-subject-out; each subject tested by a model that never saw them):

| Electrodes | Seizures caught | False alarms / 24 h | Median delay |
|---|---|---|---|
| Temple (4 ch), raw features | 62% | 524 | 6.5 s |
| Temple (4 ch), **per-person normalised** | 85% | 342 | 6.0 s |
| **Ear-hook both ears (2 ch)**, normalised | **88%** | 382 | **5.0 s** |
| Ear-hook one ear (1 ch), normalised | 83% | 556 | 6.0 s |
| Full cap (18 ch), normalised | 85% | **228** | 5.0 s |

**After learning the user's own seizures** (fine-tuning on their first 2 seizure recordings, tested on later ones):

| Electrodes | General only | Patient-only model | **General + fine-tuned** |
|---|---|---|---|
| Temple (4 ch) | 84% · 361 FA/day | 68% · 16 FA/day | **86% · 138 FA/day** |
| **Ear-hook both ears (2 ch)** | 87% · 397 FA/day | 65% · 19 FA/day | **86% · 162 FA/day** |
| Ear-hook one ear (1 ch) | 81% · 550 FA/day | 66% · 79 FA/day | 80% · 216 FA/day |
| Full cap (18 ch) | 84% · 247 FA/day | 68% · 6 FA/day | **89% · 88 FA/day** |

At a stricter alarm threshold (chosen after seeing results, so optimistic), fine-tuned detectors reach ~18–23 false alarms/day while catching 64% (ear-hook both ears) to 75% (full cap) of seizures.

![General model](epilepsy_sim/detect/results_general/general.png)

Takeaways:

1. **Per-person normalisation** (comparing each person with their own recent baseline) adds 43 detected seizures to the temple setup.
2. **Fine-tuning** cuts false alarms by 40–65% without losing seizures.
3. **The ear-hook is worth it:** with 2 electrodes it matches the temple setup (4) and catches as many seizures as the full cap (18). What the extra channels mainly buy is **false-alarm rejection**, which the ear-hook can recover with its own heart-rate and motion sensors (§7).
4. **Both ears > one ear:** about 6% more seizures caught and 25–30% fewer false alarms.
5. **Patient-only models** have few false alarms but miss a third of seizures. Two examples are too few to learn from alone.

### 6.6 Epilepsy Live Lab (`viz/epilepsy_live.html`)

A single-screen, offline dashboard:

- a glowing orb field (neurons, K⁺, Na⁺, Cl⁻, adenosine, blood vessels) driven by the simulation;
- 9 live charts of hidden variables, including **true vs device-inferred** K⁺ and Cl⁻;
- a **virtual ear-hook EEG** computed from the simulated neurons by a forward model (synaptic currents, glial K⁺ DC shift, 1/f noise), with a live detector;
- a switch to the **real CHB-MIT patient EEG** and its trained detector.

Deep links: `?scen=healthy&t=9&speed=0.5&paused=1`.

---

## 7. Handling false alarms

EEG alone gives far too many false alarms (§6.5). The planned fixes:

| Layer | How it helps |
|---|---|
| **Multimodal agreement** | Alarm only if EEG **and** a body signal agree (heart-rate surge, which occurs in 82% of patients; motion pattern; blood-flow change) |
| **Artefact sensors** | Jaw-muscle EMG flags chewing and talking; the accelerometer flags walking; impedance checks flag loose electrodes |
| **Tiered alerts** | Log silently → "Are you OK?" vibration → caregiver alert |
| **Personal learning** | Every confirmed false alarm teaches that person's model |
| **Risk forecasting** | Multi-day seizure cycles adjust sensitivity |

A published EEG + ECG detector reached 84.8% sensitivity at **0.25 false alarms/day** ([Jeppesen 2024](https://onlinelibrary.wiley.com/doi/10.1002/epd2.20196)). The next planned step is testing this on **SeizeIT2**, a public dataset recorded with a behind-the-ear EEG + ECG + EMG + motion device ([dataset](https://www.nature.com/articles/s41597-025-05580-x)).

---

## 8. Honest limitations

- **The model's seizures are tonic, not rhythmic, and not self-sustaining at default settings.** The triggered event ends with the trigger. Calibrating the pump current fixes this, but costs the healthy tissue's normal low-level firing (`SIMULATION.md` §11–12).
- **The ear-hook results are approximated with standard scalp electrodes** (T7-P7, T8-P8) from a hospital gel cap, not real dry ear electrodes.
- **CHB-MIT is paediatric, recorded in hospital, mostly at rest.** Real-world false alarms will be higher.
- **The small-sample figures are coarse.** With 36.6 h of data, one false alarm counts as 0.66 per day, and the tuned thresholds in the 3-patient study were picked after seeing results.
- **The inference results are simulation-only.** The "perfect" TVB localisation is an inverse crime.
- **Virtual EEG detector threshold:** tuned on the model itself, not independently validated.

---

## 9. Roadmap

| Priority | Next step |
|---|---|
| ★★★ | Multimodal detection on SeizeIT2 (behind-the-ear EEG + ECG + EMG + motion): measure the false-alarm reduction |
| ✅ | ~~Finish the general-model comparison: one ear vs both ears vs full cap~~ (done, §6.5) |
| ★★ | Recalibrate the model for self-sustained, rhythmic seizures (pump-current scale 0.25–0.5) |
| ★★ | Deep-learning detector and a larger training set (e.g. TUH EEG, requires registration) |
| ★★ | Hardware prototype: dry ear electrodes + in-ear PPG + IMU, recording on a phone |
| ★ | Seizure forecasting from multi-day cycles |

---

## 10. Documents index

| Document | What's inside |
|---|---|
| [`Temple_Technical_Dossier.md`](Temple_Technical_Dossier.md) | Temple wearable: company, hardware (multi-wavelength PPG), software (Brain Flow index, Entropy), 3 validation preprints, engineering inference, epilepsy relevance |
| [`Epilepsy_Overview.md`](Epilepsy_Overview.md) | Definition (ILAE 2014), prevalence, causes (6 aetiologies), 2025 seizure classification, diagnosis, drugs, surgery, SUDEP, wearables |
| [`Seizure_Neurophysiology.md`](Seizure_Neurophysiology.md) | Seizure life cycle; second-by-second ions, synapses, spread, energy, termination; what each sensor can see |
| [`Inference_Formulas.md`](Inference_Formulas.md) | Measured vs inferred; Bayes, least squares, Kalman, ABC; derivations for K⁺, Cl⁻, E/I, fNIRS; verification audit |
| [`epilepsy_sim/SIMULATION.md`](epilepsy_sim/SIMULATION.md) | Full model documentation: equations, parameters with sources, validation, results, 14-step calibration log |
| [`epilepsy_sim/COMPARISON.md`](epilepsy_sim/COMPARISON.md) | Twin experiments E1–E7, TVB cross-validation and whole-brain inference, head-to-head comparison |

---

## 11. Data, licences and disclaimer

- **The full patient dataset is not in this repository.** The scripts download it from PhysioNet: the **CHB-MIT Scalp EEG Database** ([PhysioNet](https://physionet.org/content/chbmit/1.0.0/), Open Data Commons Attribution licence v1.0; cite Shoeb 2009 and Goldberger et al. 2000).
- **The one exception** is `epilepsy_sim/viz/data.js`. It includes a 3-minute, 5-channel excerpt of CHB-MIT recording `chb01_03.edf` (2926–3106 s, band-passed, 128 Hz) so the live lab works out of the box. It is redistributed under that licence, with attribution to the source above.
- **The Virtual Brain** (`tvb-library`, `tvb-data`) is an optional dependency under its own licence.
- **No licence has been chosen for this code yet.** Until one is added, default copyright applies.
- **Medical disclaimer:** this is a research and educational project. It is **not** a medical device, has not been clinically validated, and must not be used to diagnose, monitor or treat epilepsy.
- **Temple** is a product of Temple Wearables / Continue Research. This project is independent and not affiliated with it.

---

## 12. Key references

- Cressman JR et al. (2009). *The influence of sodium and potassium dynamics on excitability, seizures, and the stability of persistent states.* J Comput Neurosci. (+ 2011 erratum)
- Krishnan GP, Bazhenov M (2011). *Ionic dynamics mediate spontaneous termination of seizures and postictal depression state.* J Neurosci 31:8870.
- Jirsa VK et al. (2014). *On the nature of seizure dynamics.* Brain 137:2210.
- Sanz-Leon P et al. (2013). *The Virtual Brain.* Front Neuroinform 7:10.
- Shoeb AH (2009). *Application of machine learning to epileptic seizure onset detection and treatment.* MIT PhD thesis (CHB-MIT).
- Zibrandtsen I et al. (2017). *Ear-EEG detects ictal and interictal abnormalities in focal and generalized epilepsy.* Clin Neurophysiol.
- Swinnen L et al. (2021). *Accurate detection of typical absence seizures using a two-channel EEG wearable behind the ears.* Epilepsia.
- Friston KJ et al. (2000). *Nonlinear responses in fMRI: the Balloon model.* NeuroImage 12:466.
- Donoghue T et al. (2020). *Parameterizing neural power spectra.* Nat Neurosci 23:1655.
- Jeppesen J et al. (2024). *Seizure detection with ictal tachycardia using HRV and patient-adaptive ML.* Epileptic Disord.

Full reference lists are in each document.

# Cutting false alarms: five strategies, a false-alarm taxonomy, and the case for a tonic-clonic Prototype 1

*30 September 2026. Scripts: `epilepsy_sim/detect/fa_reduction.py`, `epilepsy_sim/detect/analysis_fa_types.py`, `epilepsy_sim/detect/siena.py`. Results: `detect/results_seizeit2/fa_reduction.json`, `fa_types.json`, `detect/results_siena/metrics.json`, `szcore.json`.*

> **Short answer.**
> - **Alarm only for tonic-clonic seizures** (the SUDEP-risk type) and the same technology reaches **cleared-device level**: a dedicated convulsive model caught **87% at ≤1 false alarm/day, 90% at 1.1/day, 96% at ≤5/day** (125 patients, 55 tonic-clonic seizures, never-seen patients).
> - **Ear + back of the head** cuts false alarms by about half vs the ear alone, in adults (Siena) and in children (CHB-MIT). 6 channels match or beat a full 18-channel cap in the middle of the curve.
> - A **seizure-cluster prior** (lower the bar for 12 h after a confirmed seizure) cuts false alarms by **24–33%**.
> - A **resting-only mode** lowers the total number of alarms but not their rate: false alarms happen at rest too.
> - **False alarms fall into 8 types.** About a third are artefacts a sensor can flag (muscle/chewing, big movements, electrode contact). A type-aware verifier gives no gain on SeizeIT2's sensors, so the types are a hardware to-do list.
>
> **Decision: Prototype 1 = tonic-clonic alarm + diary for everything else.**

---

## Contents

1. [How to read the numbers](#1-how-to-read-the-numbers)
2. [Tonic-clonic and long-seizure alarm modes](#2-tonic-clonic-and-long-seizure-alarm-modes)
3. [Ear + back of the head (Siena and CHB-MIT)](#3-ear--back-of-the-head-siena-and-chb-mit)
4. [Seizure-cluster prior](#4-seizure-cluster-prior)
5. [Night mode, resting mode, and anonymised clocks](#5-night-mode-resting-mode-and-anonymised-clocks)
6. [Combined pipeline (fine-tuned detector + personal verifier)](#6-combined-pipeline)
7. [What kinds of false alarms are there?](#7-what-kinds-of-false-alarms-are-there)
8. [Precision: how many alarms are real? (SzCORE)](#8-precision-how-many-alarms-are-real-szcore)
9. [Conclusions for Prototype 1](#9-conclusions-for-prototype-1)
10. [Engineering notes and incidents](#10-engineering-notes-and-incidents)
11. [Limitations](#11-limitations)
12. [Reproduce](#12-reproduce)
13. [References](#13-references)

---

## 1. How to read the numbers

- **"Caught at ≤ X / day"**: the share of seizures caught at the best threshold whose false-alarm rate stays within X per 24 h.
- **"FA/day at Y% caught"**: the fewest false alarms per 24 h at any threshold that catches at least Y% of seizures.
- **Held-out data:** all SeizeIT2 results use each patient's **later** data (after the 2nd seizure, k = 2 split, 71 patients) or patient-wise cross-validation (125 patients). CHB-MIT and Siena use leave-one-patient-out. Every number comes from patients the model never saw.
- **Slightly optimistic:** these are best-along-the-curve numbers (see `Multimodal_Detection_SeizeIT2.md` §5.1).
- **Scoring:** event scoring is identical to all earlier tables (alarm from 15 s before onset to seizure end; alarms up to 60 s after a seizure are not false; 60 s refractory).

## 2. Tonic-clonic and long-seizure alarm modes

**Idea.** Cleared wearables (Embrace, NightWatch, EpiCare) keep false alarms low by alarming only for **convulsive** seizures. Here the device alarms only for **target** seizures and logs the rest:
- a non-target seizure is neither a miss nor a false alarm;
- an alarm during it is not counted as false;
- the alarm sequence is exactly the same as before; only what counts as a hit changes.

### 2.1 Existing best model, 71 patients, later data

**General model:**

| Target | Seizures | FA/day at 50 / 70 / 90 / 100% caught | Caught at ≤0.5 / 1 / 2 / 5 / 10 / 20 / 50 FA/day |
|---|---|---|---|
| All seizures | 649 | 58.7 / 170.6 / – / – | 10 / 14 / 17 / 25 / 31 / 36 / 47% |
| **Focal-to-bilateral tonic-clonic** | 22 | **0.1 / 0.4 / 1.0 / 4.3** | **77 / 91 / 95 / 100** / 100 / 100 / 100% |
| Long (> 90 s) | 126 | 2.0 / 10.9 / 73.1 / 408 | 32 / 39 / 48 / 62 / 70 / 75 / 84% |
| Tonic-clonic or > 90 s | 134 | 1.6 / 9.0 / 73.1 / 408 | 34 / 42 / 51 / 64 / 72 / 77 / 85% |
| ≥ 30 s | 300 | 5.6 / 30.9 / 171 / – | 22 / 30 / 36 / 49 / 57 / 65 / 75% |

**Fine-tuned model (2 own seizures):**

| Target | FA/day at 50 / 70 / 90 / 100% caught | Caught at ≤0.5 / 1 / 2 / 5 / 10 / 20 / 50 FA/day |
|---|---|---|
| All seizures | 44.6 / 149.3 / – / – | 12 / 15 / 21 / 27 / 33 / 41 / 52% |
| **Tonic-clonic** | **0.1 / 0.2 / 0.9 / 2.3** | **86 / 91 / 91 / 100** / 100 / 100 / 100% |
| Long (> 90 s) | 1.9 / 10.8 / 117 / 402 | 33 / 42 / 52 / 64 / 69 / 79 / 86% |
| Tonic-clonic or > 90 s | 1.4 / 8.9 / 91 / 402 | 36 / 45 / 54 / 66 / 71 / 80 / 87% |
| ≥ 30 s | 4.6 / 23.3 / 183 / – | 24 / 31 / 42 / 52 / 60 / 69 / 76% |

### 2.2 All 125 patients, and a dedicated convulsive model

With only 22 tonic-clonic seizures in the later data, the estimate is coarse. So both models were run on **all 125 patients**, patient-wise 5-fold, all data: **55 tonic-clonic seizures**. The two models:
- **general model** = the best setup (EEG + ECG + context + HR-excess), trained on all seizures;
- **dedicated convulsive model** = trained with tonic-clonic windows as the **only** positives (other seizures left out of training), with EEG + ECG + **shoulder EMG + motion** + context + HR-excess.

| Model → target | FA/day at 50 / 70 / 90 / 100% caught | Caught at ≤0.5 / 1 / 2 / 5 / 10 / 20 / 50 FA/day |
|---|---|---|
| General → all 883 seizures | 42.4 / 132.0 / – / – | 14 / 17 / 22 / 28 / 35 / 42 / 53% |
| General → 55 tonic-clonic | 0.1 / 0.2 / 3.1 / 22.6 | 78 / 85 / 87 / 95 / 95 / 98 / 100% |
| **Dedicated → 55 tonic-clonic** | **0.0 / 0.1 / 1.1 / 9.0** | **84 / 87 / 93 / 96 / 100 / 100 / 100%** |
| Dedicated → all 883 seizures | – | 12 / 16 / 19 / 25 / 30 / 34 / 47% |

**What it means:**
1. **Tonic-clonic alarm mode works at product level:** 87% caught at ≤1 false alarm/day, 90% at 1.1/day, 96% at ≤5/day, 100% at ≤10/day. For comparison, Embrace reported 92–100% at 0.2–2 FA/day for generalised tonic-clonic seizures in monitoring units.
2. **A specialist helps.** The dedicated model beats the general one at every strict budget (90% caught at 1.1 vs 3.1 FA/day; 100% at 9 vs 23). Muscle and motion carry the convulsive signature.
3. **The specialist is only a specialist.** It is slightly worse on all seizures, so the device should run it **alongside** the general detector: the dedicated model raises alarms, the general model feeds the diary.
4. **Long seizures (> 90 s) are a middle ground:** 39–42% at ≤1/day, 62–64% at ≤5/day. That is useful for a "long-seizure" alert, but not yet alarm-grade.

## 3. Ear + back of the head (Siena and CHB-MIT)

**Idea.** An ear-hook joined by a band around the back of the head could add electrodes at P7/P8 (behind the ear), O1/O2 (back of the head) and optionally P3/P4 (upper back of the head, parietal). Head map (nose at top):

```
              Fp1   Fp2
          F7   F3  Fz  F4   F8
   (ear) T7   C3  Cz  C4   T8 (ear)
          P7   P3  Pz  P4   P8
              O1     O2
```

| Name | Bipolar channels |
|---|---|
| Ear-hook | T7–P7, T8–P8 |
| Back of head | P7–O1, P8–O2 |
| Ear + back (4 ch) | both of the above |
| Ear + back + parietal (6 ch) | + P3–O1, P4–O2 |

### 3.1 Siena (14 adults, 47 seizures, leave-one-patient-out)

| Setup | Caught / FA/day @0.5 | FA/day at 50 / 60 / 70 / 80% caught | Caught at ≤1 / 5 / 20 / 50 FA/day |
|---|---|---|---|
| Ear-hook, both ears (2 ch) | 96% / 310 | 27 / 49 / 60 / 70 | 11 / 32 / 47 / 66% |
| Back of head only (2 ch) | 98% / 276 | 15 / 23 / 39 / 64 | 11 / 19 / 55 / 77% |
| **Ear + back of head (4 ch)** | 94% / 223 | **18 / 32 / 47** / 71 | 13 / 34 / 55 / 70% |
| Ear + back + ECG | 85% / 144 | 14 / 28 / 60 / 69 | 26 / 32 / 55 / 68% |
| **Ear + back + parietal (6 ch)** | 91% / 203 | **10 / 12 / 32 / 45** | 19 / **38 / 66 / 81**% |
| Full cap (18 ch) | 91% / 132 | 10 / 15 / 23 / 44 | 4 / 34 / 68 / 83% |

### 3.2 CHB-MIT (23 children, 185 seizures, leave-one-subject-out; stored 18-channel features re-combined, same training windows for every setup)

| Setup | FA/day at 50 / 70% caught | Caught at ≤0.5 / 1 / 2 / 5 / 10 / 20 / 50 FA/day |
|---|---|---|
| Ear-hook, both ears (2 ch) | 26.1 / 113.1 | 6 / 10 / 15 / 26 / 39 / 50 / 55% |
| Back of head only (2 ch) | 12.6 / 101.2 | 4 / 11 / 14 / 30 / 46 / 58 / 66% |
| **Ear + back of head (4 ch)** | **14.3 / 89.9** | 15 / 15 / 21 / 34 / 45 / 54 / 63% |
| **Ear + back + parietal (6 ch)** | **9.9 / 69.4** | 17 / 18 / 25 / 39 / 51 / 59 / 68% |
| Full cap (18 ch) | 14.1 / 104.4 | 17 / **26 / 32 / 42** / 49 / 54 / 61% |

**What it means:**
1. **Same pattern in adults and children.** Adding the back of the head cuts false alarms by about 45% at 50% caught (CHB-MIT 26 → 14/day; Siena 27 → 18/day).
2. **6 channels ≈ full cap.** At 50–70% caught, 6 channels beat the full cap on CHB-MIT (9.9 vs 14.1; 69 vs 104) and matched it on Siena. The cap stays ahead only at the strictest budgets (CHB-MIT ≤1/day: 26% vs 18%).
3. **P7/P8 (behind the ear) is a strong position.** "Back of head only" beat the ear-hook on both datasets: more evidence that **placement matters more than electrode quality** (`Electrodes_and_Siena_Validation.md`).
4. **Practical cost.** P7/O1/O2 are under hair (dry comb or spring-pin electrodes), and P3/P4 need a strap over the top of the head. Ear + back (4 ch) is the discreet compromise.

## 4. Seizure-cluster prior

**Idea.** Seizures cluster, so after a **confirmed** seizure the device lowers its alarm bar for a few hours. Implementation:
- add w to the detector's log-odds (after 3-window smoothing) for H hours after the end of the previous seizure;
- a seizure counts only once it has **ended** (+60 s), as a diary entry or caregiver would confirm it, so the boost never uses the seizure currently being detected.

The runs of a patient were placed in order with 60 s gaps (true gaps unknown). Best general model, 71 patients, 649 later seizures:

| Variant | FA/day at 50 / 70% caught | Caught at ≤0.5 / 1 / 2 / 5 / 10 / 20 / 50 FA/day |
|---|---|---|
| No prior | 58.7 / 170.6 | 10 / 14 / 17 / 25 / 31 / 36 / 47% |
| +1 for 12 h | 42.0 / 132.3 | 12 / **16** / **21** / **28** / 33 / **41** / 54% |
| **+2 for 12 h** | **39.3 / 129.9** | 11 / 14 / 20 / 26 / 33 / **41 / 55**% |
| +1 for 24 h | 50.0 / 150.8 | 11 / 14 / 20 / 27 / 32 / 39 / 52% |
| +2 for 24 h | 45.3 / 142.6 | 13 / 13 / 16 / 28 / 33 / 39 / 51% |

**What it means:** **−24% to −33% false alarms** at matched sensitivity, and +4–8 points at 20–50 FA/day, at zero hardware cost. 12 h beats 24 h because clusters are short. It needs a seizure diary or caregiver confirmation.

## 5. Night mode, resting mode, and anonymised clocks

### 5.1 The clock problem

A night-alarm / day-diary mode needs clock time. The EDF headers of all 2,850 SeizeIT2 recordings were read (256-byte range requests), and **every start time is `00.00.00`**: the dataset anonymises clock time.
- **Night mode:** the first night-mode run treated "hours since recording start" as clock time. Its numbers are **invalid and discarded** (they remain in `fa_reduction.json` under `night_mode` for the record only).
- **Time-of-day prior:** dropped for the same reason.

### 5.2 Resting mode (no clock needed)

A device knows from its own accelerometer when the wearer has been still. Resting mode works as follows:
- **Movement per window:** the larger of the two accelerometer log-SDs.
- **Movement level:** the median over the last 10 completed minutes (causal).
- **Alarm window:** alarms are allowed only while that level is at or below the user's own q-th percentile, learned on the **tuning period**. Other times are diary-only.
- **Targets:** seizures starting in the alarm window.
- **Scoring:** FA/day per 24 h of alarm-on time.

**Best general model:**

| Mode | Alarm-on share of time | Seizures in window | FA/day at 50 / 70% caught | Caught at ≤1 / 5 / 20 FA/day | Asleep / awake-labelled seizures in window |
|---|---|---|---|---|---|
| 24 h alarms | 100% | 649 | 58.7 / 170.6 | 14 / 25 / 36% | – |
| Still ≤ 20th pct | 16% | 153 | 59.9 / 173.2 | 5 / 18 / 26% | 38% / 13% |
| Still ≤ 35th pct | 27% | 261 | 51.2 / 145.5 | 8 / 20 / 33% | 61% / 25% |
| Still ≤ 50th pct | 40% | 370 | 51.9 / 152.1 | 11 / 23 / 35% | 78% / 44% |

**Fine-tuned model:**

| Mode | FA/day at 50 / 70% caught | Caught at ≤1 / 5 / 20 FA/day |
|---|---|---|
| 24 h alarms | 44.6 / 149.3 | 15 / 27 / 41% |
| Still ≤ 20th pct | 62.2 / 168.7 | 7 / 16 / 28% |
| Still ≤ 35th pct | 53.0 / 137.7 | 13 / 23 / 37% |
| Still ≤ 50th pct | 46.6 / 133.8 | 13 / 27 / 39% |

**What it means:**
1. **The accelerometer's "still" state tracks sleep:** 78% of seizures labelled *asleep* fall in it, vs 44% of *awake* ones.
2. **But false alarms are not rarer at rest.** Per hour of alarm-on time the rate is about the same, and seizures at rest are slightly harder to catch. False alarms are seizure-like body events, not just movement (§7).
3. **What resting mode does give is fewer alarms in total.** At 40% alarm-on time the absolute number of alarms per calendar day falls by about 60%. The cost is that only rest-time seizures raise alarms. It is a product choice (a NightWatch-style sleep-safety mode), not an accuracy gain.

## 6. Combined pipeline

**Result: no gain.** Personalisation saturates, so one personal layer is enough.

| Stage 1 under the personal verifier (71 patients, 649 later seizures) | Caught at ≤0.5 / 1 / 2 / 5 / 10 / 20 / 50 FA/day |
|---|---|
| General model (existing two-stage baseline, reproduced) | 15 / **19** / 22 / **30** / **37** / **45** / **60**% |
| Fine-tuned model (k = 1) | 14 / 17 / **24** / **30** / **37** / **45** / 58% |
| Mean of fine-tuned + general | 14 / 17 / 22 / 29 / 36 / 44 / 59% |

The fine-tuned detector and the personal verifier learn the same patient-specific information. Differences of 1–2 points are within noise (1 point ≈ 6–7 seizures). For Prototype 1 the simpler design wins: **general detector + personal alarm verifier**.

**Design** (`fa_reduction.py stack`):
- **Stage 1:** the best model **fine-tuned after the 1st seizure** (k = 1).
- **Personal verifier training data:** its own candidates from the stretch **between the 1st and the 2nd seizure**, which that model never saw (honest held-out data, weighted ×20), plus other patients' candidates.
- **Scoring:** exactly the **same later data** as the k = 2 tables.
- **Variants:** stage 1 = the general model (the existing two-stage baseline); the fine-tuned model; or their mean.

The CNN is not included: its outputs exist only for a 33-patient subset.

## 7. What kinds of false alarms are there?

**Method** (`analysis_fa_types.py`):
- **Data:** all 124,563 candidate alarms of the best general model (stage 1 at 0.5, evidence up to 10 s after the trigger; 71 patients): 121,564 false, 2,999 true. Each carries about 40 measurements.
- **Signature:** the false alarms were described by what **changed** at the alarm vs the 5 min before (heart rate and its slope, RMSSD, beat quality, ear-EEG power / line length / gamma and delta shares, shoulder EMG, back and chest movement, heart rate not explained by movement, recent alarm count), robust-standardised.
- **Clustering:** Gaussian mixture, k chosen by BIC (k = 8).
- **Naming:** each cluster is named from its median profile.

| Share | Type | Signature (vs a typical false alarm) | Patients (share from the top 10%) | Median HR | Likely cause | Fix |
|---|---|---|---|---|---|---|
| 12.2% | **Muscle / high-frequency EEG + movement** | EEG gamma share +1.5, line length +1.1, shoulder EMG +0.9 | 70 (32%) | 86 | Chewing, talking, jaw clenching | **Jaw (temporalis) EMG** on the ear-hook |
| 12.0% | **Big-movement artefact** | EEG power +1.5, EMG +1.1, delta share +1.0 | 69 (37%) | 90 | Turning, getting up, touching the head | **Head IMU** gate |
| 7.8% | **Electrode / contact** | EEG power +1.1 and line length +0.9 with **no body change**; ECG beat quality −0.8 | 69 (34%) | 99 | Loose or shifting electrode, pressure on the ear | **Contact-impedance sensing** + "reseat" prompt |
| 9.7% | **Heart surge beyond movement** | HR +0.7, HR-excess +0.6 | 67 (30%) | 102 | Arousal, stress, waking up | Hardest: needs EEG agreement / personal learning |
| 16.1% | **Alarm bursts** | Many alarms in the last hour (+1.7) | **13 (99%)** | 77 | A few noisy patients / recordings | Personal verifier; contact check after a burst |
| 42.3% | **Subtle** (3 clusters) | Small changes everywhere | 64–70 | 77–88 | Model background noise near threshold | Verifier, priors |

**True seizure alarms:** HR change +0.9 together with EEG line length +0.8 and power +0.7-0.8 (heart and EEG rise together), without a big muscle or movement signature.

**Separability.** Each false-alarm type separates from true seizure alarms with patient-wise AUC 0.93–0.99. With 40 false candidates per true one, even 97% rejection leaves more false than true alarms, so the types say **which** fix to build rather than solving the problem.

**Type-aware verifier.** The false-alarm types were learned on the training folds only. A multi-class verifier (seizure vs 8 types) accepts an alarm when P(seizure) is high enough; it was compared with the yes/no verifier on the same candidates, folds and scoring:

| Verifier | Caught at ≤0.5 / 1 / 2 / 5 / 10 / 20 / 50 FA/day |
|---|---|
| Yes/no | 13 / **15** / **18** / **25** / **30** / **38** / **54**% |
| Type-aware | 11 / **15** / 16 / 22 / 29 / 37 / 50% |

**No gain:** both see the same measurements, and splitting "not a seizure" into 8 classes adds no information. The types become useful once each has its own sensor (jaw EMG, head IMU, contact impedance), which SeizeIT2 lacks. They are also useful for explaining cancelled alarms to users ("alarm cancelled: chewing detected").

## 8. Precision: how many alarms are real? (SzCORE)

Seizure detection is judged by event-based **sensitivity, precision, F1 and false positives per 24 h** (SzCORE, Dan et al., *Epilepsia* 2024). "Accuracy" is misleading, because a detector that never alarms is > 99% "accurate".

- **Sensitivity** = TP / (TP + FN)
- **Precision** = TP / (TP + FP), the share of alarms that are real seizures
- **F1** = 2·Sens·Prec / (Sens + Prec)

**Siena (14 adults, 47 seizures), at each setup's best-F1 threshold** (chosen on the same data, so optimistic):

| Setup | Sensitivity | Precision | F1 | FP/day |
|---|---|---|---|---|
| Ear-hook, both ears | 45% (21/47) | 29% (21/72) | 0.35 | 8.7 |
| Ear-hook + ECG | 30% | 36% | 0.33 | 4.3 |
| Ear-hook, one ear | 49% | 30% | 0.37 | 9.0 |
| Back of head only | 53% | 23% | 0.32 | 14.6 |
| Ear + back of head (4 ch) | 32% | **48%** | 0.38 | 2.7 |
| **Ear + back + ECG** | 28% | **65%** (13/20) | 0.39 | **1.2** |
| **Ear + back + parietal (6 ch)** | **64%** (30/47) | 29% | **0.40** | 12.3 |
| Full cap | 53% | 31% | 0.39 | 9.5 |
| Full cap + ECG | 38% | 31% | 0.34 | 6.8 |

At threshold 0.5 every setup catches 85–98% of seizures, but precision is only **2–6%**.

**Precision depends on how often seizures happen:**

Precision(real life) = Sens × R / (Sens × R + FP/day), where R = seizures per day.

The hospital patients above had about 8 seizures/day (medication reduced to provoke seizures). For ear + back + ECG:
- **Hospital (R ≈ 8):** 0.28 × 8 = 2.2 true vs 1.2 false alarms/day → **65%** of alarms real.
- **Daily life (1 seizure/week, R ≈ 0.14):** 0.04 true vs 1.2 false → **about 3%** of alarms real.

This is why alarms need well below 1 false alarm/day, and why tonic-clonic mode (≤0.1–1 FA/day) is the realistic first alarm product.

## 9. Conclusions for Prototype 1

| Finding | Design consequence |
|---|---|
| Tonic-clonic mode: 87% at ≤1 FA/day, 96% at ≤5 (dedicated model) | **Prototype 1 alarms only for tonic-clonic seizures**; everything else goes to the diary |
| The dedicated convulsive model uses muscle + motion | **Jaw (temporalis) EMG + head IMU** on the ear-hook (SeizeIT2 had shoulder EMG and chest/back motion, so this must be validated) |
| Ear + back of head halves false alarms (adults and children) | Prototype 1 should be able to record **ear + back of head** (band behind the head) and compare it with ear-only on the same wearer |
| Cluster prior −24% to −33% | App: seizure diary + caregiver confirmation drive a 12-h lowered threshold |
| Resting mode = fewer alarms, same rate | Optional "sleep-safety" mode, not an accuracy fix |
| Fine-tuning + personal verifier = no gain over the verifier alone | One personal layer: general detector + personal alarm verifier |
| Fourier analysis: false alarms are broadband muscle/movement bursts; low-pass 30 Hz + spectral features help ear EEG (up to −48%) but only −6 to −15% on the best model | Add the low-pass + 5 spectral features to the pipeline (free); notch where mains noise is detected; details in `Fourier_and_Denoising.md` |
| False-alarm types | Sensors: jaw EMG (chewing), head IMU (movement), contact impedance (electrodes); label artefacts during self-wear to re-run §7 |
| Precision collapses at real-life seizure rates | Alarm target must be ≪1 FA/day; tonic-clonic mode is the only one there today |
| SUDEP relevance | In-ear SpO₂ for peri-/postictal desaturation |

## 10. Engineering notes and incidents

- **Connection reset.** The first EDF-header pass stopped at the first connection reset (WinError 10054), and the queue then stopped at its first step. Fixed with 5 retries with back-off, 8 parallel requests instead of 24, and resumable partial saves; the queue now continues past a failed step.
- **Anonymised clocks** (§5.1): the night-mode results were discarded, and resting mode was built instead.
- **Slow rolling median.** The first rolling-median implementation looped in Python over about 13 million windows. It was replaced by 1-min block means + a sliding median over the last 10 completed blocks (causal, vectorised).
- **Data leakage prevented by design:**
  - the cluster prior uses a seizure only after it ended;
  - night/rest mode keeps all seizures in the false-alarm scoring and targets only in-window ones;
  - false-alarm types for the type-aware verifier are learned on training folds only;
  - the combined pipeline trains the personal verifier on data the fine-tuned model never saw.
- **Launch.** Long jobs ran as detached queue files (`detect/logs/queue13.cmd`), launched by the user when the assistant's command-safety check was temporarily unavailable.

## 11. Limitations

- **Tonic-clonic:** 55 seizures (22 in the later-data analysis), so one seizure is 2–5 percentage points. Hospital recordings; motion and EMG on the shoulder/chest/back, not the head.
- **Siena / CHB-MIT montages:** hospital gel scalp electrodes stand in for the ear-hook and band; Siena is 14 patients.
- **Cluster prior:** the gaps between runs are unknown (60 s assumed), and it assumes every seizure is confirmed.
- **Resting mode:** the movement threshold is per patient from 2-seizure tuning data; SeizeIT2 motion is on the back/chest, not the head.
- **False-alarm types:** unsupervised, named from profiles, with no ground-truth artefact labels.
- **General:** best-along-the-curve numbers are slightly optimistic; no confidence intervals yet.

## 12. Reproduce

From `epilepsy_sim/`:

```bash
uv run detect/fa_reduction.py clock      # EDF start times (all anonymised to 00.00.00)
uv run detect/fa_reduction.py rest       # resting mode (accelerometer), ~15 min
uv run detect/fa_reduction.py prior      # seizure-cluster prior, ~5 min
uv run detect/fa_reduction.py chb        # CHB-MIT ear / ear+back / 6 ch / cap, ~30 min
uv run detect/fa_reduction.py target     # tonic-clonic / long modes + dedicated convulsive model, ~1-1.5 h
uv run detect/fa_reduction.py stack      # combined pipeline, ~1.5 h (builds the k=1 fine-tuned cache)
uv run detect/analysis_fa_types.py       # false-alarm taxonomy, ~10 min
uv run detect/analysis_fa_types.py --verify   # type-aware vs yes/no verifier, ~20 min
uv run detect/siena.py run               # Siena incl. ear + back of head montages
uv run detect/siena.py szcore            # sensitivity / precision / F1 / FP per day (SzCORE)
```

## 13. References

- Dan J, Pale U, Amirshahi A, et al. SzCORE: Seizure Community Open-Source Research Evaluation framework for the validation of EEG-based automated seizure detection algorithms. *Epilepsia*, 2024. [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/epi.18113) · [arXiv:2402.13005](https://arxiv.org/abs/2402.13005)
- Onorati F et al. Multicenter clinical assessment of improved wearable multimodal convulsive seizure detectors. *Epilepsia*, 2017.
- Arends J et al. Multimodal nocturnal seizure detection in a residential care setting (NightWatch). *Neurology*, 2018.
- Beniczky S et al. Automated seizure detection using wearable devices: ILAE–IFCN clinical practice guideline. *Epilepsia* 62:632–646, 2021.
- Ryvlin P et al. Incidence and mechanisms of cardiorespiratory arrests in epilepsy monitoring units (MORTEMUS). *Lancet Neurol* 12:966–977, 2013.
- Detti P et al. Siena Scalp EEG Database, PhysioNet, 2020 (CC BY 4.0). Shoeb A. CHB-MIT Scalp EEG Database, PhysioNet, 2009.
- Bhagubai M et al. SeizeIT2. *Scientific Data*, 2025.

# Are the electrodes the problem? Electrode-stability test (SeizeIT2) and adult validation (Siena)

### Two tests of the hypothesis "SeizeIT2 and CHB-MIT used bad electrodes, and that is why detection plateaus"

*29 September 2026. Scripts: `epilepsy_sim/detect/analysis_electrodes.py`, `epilepsy_sim/detect/siena.py`. Results: `detect/results_seizeit2/electrodes.json`, `detect/results_siena/metrics.json`.*

> **Short answer.** Partly.
> - **Electrode contact drives false alarms.** Stable-contact patients have up to 2× fewer.
> - **It is not why seizures are missed.** Their seizures are no more visible to the detector.
>
> On an independent adult dataset (Siena, 14 patients, 47 seizures, full 10-20 cap):
> - **Two ear-level channels caught 96% of seizures in never-seen patients**, as many as the full cap.
> - The **cap's advantage is 2–3× fewer false alarms**.
> - **ECG closes most of that gap** for the ear-hook, but adds nothing to the cap.
>
> What limits detection is **electrode position and the number of channels** (and the seizure type), not electrode quality.

---

## Contents

1. [Why this question](#1-why-this-question)
2. [What the literature says](#2-what-the-literature-says)
3. [Test 1: electrode stability within SeizeIT2](#3-test-1-electrode-stability-within-seizeit2)
4. [Test 2: adult validation on Siena](#4-test-2-adult-validation-on-siena)
5. [Test 3: TUSZ (pending access)](#5-test-3-tusz-pending-access)
6. [Engineering notes: streaming Siena, list typos, and a scikit-learn slowdown](#6-engineering-notes)
7. [Conclusions for Prototype 1](#7-conclusions-for-prototype-1)
8. [Limitations](#8-limitations)
9. [Reproduce](#9-reproduce)
10. [References](#10-references)

---

## 1. Why this question

After the raw-EEG CNN failed to break the detection ceiling (`Multimodal_Detection_SeizeIT2.md` §6.19), the obvious suspect was the data itself: *maybe SeizeIT2 and CHB-MIT used poor electrodes, so every model hits the same wall.* If true, the fix would be a better dataset or better electrodes. If false, the fix is electrode **position / number** or more signal types.

What the two datasets actually use:

| Dataset | Electrodes | Contact quality |
|---|---|---|
| **CHB-MIT** | Clinical wet cup electrodes, full 10-20 cap, hospital | Gold standard |
| **SeizeIT2** | Stick-on Ag/AgCl gel electrodes on the mastoid bone (Byteflies Sensor Dot), 5 hospitals | Better than a dry ear-hook will get |

Neither is obviously "bad". So the hypothesis was tested directly.

## 2. What the literature says

**Only one study has recorded the same seizures with full scalp EEG and behind-the-ear EEG at the same time** (Gu et al. 2018; 12 patients with focal epilepsy, 47 seizures, 431 h):

| | Full scalp EEG | Behind-the-ear EEG |
|---|---|---|
| Median sensitivity | 100% | 94.5% |
| False alarms | 1.14 / h | 0.52 / h |
| Seizures detected | 41 / 47 | 38 / 47 |

The difference was **not statistically significant**: on the same seizures, behind-the-ear electrodes were about as good as a cap (patient-specific models). The dataset behind it, **SeizeIT1** (82 patients, behind-the-ear + full scalp + ECG simultaneously), would be the perfect test, but it is **no longer shared** because its ethical approval expired.

No public **ear-EEG** seizure dataset exists; SeizeIT2 is the closest.

## 3. Test 1: electrode stability within SeizeIT2

**Idea.** If electrode quality limits detection, patients with *stable* electrode contact should do clearly better than patients with *unstable* contact, both in false alarms and in how visible their seizures are.

**Measure of contact stability.** For each patient, the interquartile range (IQR) over time of the ear-EEG log-variance, averaged over both channels:
- non-seizure windows only;
- computed **only on the tuning period** (data up to 10 min after the 2nd seizure), i.e. something a device knows after its first days of wear.

A large IQR means the signal amplitude jumps around (loose or shifting contact). Scoring uses each patient's **later** data only, from the cached outputs of `seizeit2.py personal` (k = 2): 71 patients, 649 later seizures.

**Split.** At the median (1.20; range 0.52–2.83): 36 "stable" patients (263 seizures) vs 35 "unstable" (386 seizures).

### 3.1 Results

**Ear-EEG only, general model:**

| | Stable contact | Unstable contact |
|---|---|---|
| FA/day at 50% caught | **95** | 150 |
| FA/day at 60% caught | **136** | 201 |
| FA/day at 70% caught | **183** | 374 |
| Caught at ≤1 / 5 / 20 FA/day | 4% / 14% / 27% | 4% / 11% / 20% |
| Caught at ≤50 / 200 FA/day | 40% / 71% | 31% / 57% |
| Seizures ever > 0.5 | 91% | 84% |
| **Seizures ever > 0.95** | **21%** | **20%** |

**Ear-EEG only, fine-tuned:**

| | Stable | Unstable |
|---|---|---|
| FA/day at 50 / 60 / 70% caught | **56 / 79 / 135** | 104 / 168 / 322 |
| Caught at ≤1 / 5 / 20 FA/day | 12% / 23% / 34% | 5% / 11% / 23% |
| Seizures ever > 0.95 | 20% | 12% |

**Best model (EEG + ECG + context + HR-excess), general:**

| | Stable | Unstable |
|---|---|---|
| FA/day at 50 / 60 / 70% caught | **53 / 75 / 123** | 71 / 132 / 233 |
| Caught at ≤1 / 5 / 20 FA/day | 12% / 23% / 34% | 15% / 27% / 40% |
| Seizures ever > 0.95 | 27% | 33% |

**Best model, fine-tuned:**

| | Stable | Unstable |
|---|---|---|
| FA/day at 50 / 60 / 70% caught | **40 / 59 / 83** | 49 / 110 / 207 |
| Caught at ≤1 / 5 / 20 FA/day | 17% / 26% / 38% | 14% / 28% / 42% |
| Seizures ever > 0.95 | 23% | 28% |

**Per patient (Spearman, 71 patients):**

| Model | Instability vs median seizure peak (visibility) | Instability vs FA/day at threshold 0.9 |
|---|---|---|
| EEG only, general | ρ = −0.10 (p = 0.39) | **ρ = +0.34 (p = 0.004)** |
| EEG only, fine-tuned | ρ = −0.11 (p = 0.34) | ρ = +0.04 (p = 0.74) |
| Best, general | ρ = −0.06 (p = 0.62) | ρ = +0.17 (p = 0.15) |
| Best, fine-tuned | ρ = −0.06 (p = 0.60) | ρ = +0.17 (p = 0.15) |

### 3.2 What it means

1. **False alarms: electrodes matter.** Stable-contact patients have up to **2× fewer false alarms** at mid-to-high sensitivity (e.g. 183 vs 374 at 70% caught, EEG only; 83 vs 207 with the best fine-tuned model). Per patient, instability tracks false alarms for the plain EEG model (ρ = +0.34). Heart features and personalisation weaken the link: they absorb part of the artefact problem.
2. **Seizure visibility: electrodes don't matter.** The share of seizures the detector sees clearly (> 0.95) is **the same** in both groups (21% vs 20%), and instability does not correlate with how high seizures score (ρ ≈ −0.1, n.s.). At the strict end (≤1–5 FA/day) the groups are close, and with heart features the unstable group is even slightly ahead.
3. **So the hypothesis is half right.** Electrode contact is a false-alarm problem. It is **not** why seizures are missed; that ceiling comes from where the electrodes are (behind the ear, 2 channels) and how small many focal seizures are at that distance.

## 4. Test 2: adult validation on Siena

**Why Siena.** CHB-MIT is children; SeizeIT2 has only behind-the-ear EEG (no cap to compare against). The **Siena Scalp EEG Database** (PhysioNet, CC BY 4.0; Detti et al. 2020) has **adults with focal epilepsy**, a **full 10-20 cap** at 512 Hz and **EKG**. So it answers two questions:
- does ear-level vs full cap hold in adults?
- does ECG help there too?

**Data** (all streamed, nothing downloaded):

| | |
|---|---|
| Patients | 14 adults (20–71 y; 9 male) |
| Epilepsy | 13 temporal (IAS / WIAS), 1 frontal with focal-to-bilateral tonic-clonic seizures (PN10) |
| Seizures | **47** |
| Recording | **141 h** in 41 EDF files (21.8 GB) |
| Channels | 29 EEG (10-20 + extras), EKG, SpO₂, HR |

**Montages** (bipolar, identical to the CHB-MIT study; Siena uses the older names T3/T4/T5/T6 = T7/T8/P7/P8):

| Setup | Channels |
|---|---|
| Ear-hook, both ears (2 ch) | T3–T5, T4–T6 (= T7–P7, T8–P8: just above and behind each ear) |
| Ear-hook, one ear | T4–T6 |
| Full cap | 18-channel double banana |
| + ECG | The 8 SeizeIT2 ECG features from the EKG channel |

**Pipeline.** Exactly the SeizeIT2 pipeline:
- 4 s windows every 2 s, 14 EEG features per channel;
- causal 10-min per-recording baseline normalisation;
- HistGradientBoosting (150 trees, class-balanced) on all seizure windows + 20% of the others;
- smoothing over 3 windows, 60 s refractory, same event scoring.

**Evaluation: leave-one-patient-out.** Each patient is scored by a model trained on the other 13, so these are **new-user** numbers.

### 4.1 Results (14 patients, 47 seizures, new users)

| Setup | Caught @0.5 | FA/day @0.5 | Median delay | FA/day at 50 / 60 / 70 / 80 / 90% caught | Caught at ≤1 / 5 / 20 / 50 / 100 / 200 FA/day |
|---|---|---|---|---|---|
| Ear-hook, both ears (2 ch) | **96%** (45/47) | 310 | 9 s | 27 / 49 / 60 / 70 / 167 | 11 / 32 / 47 / 66 / 83 / 91% |
| **Ear-hook, both ears + ECG** | 91% (43/47) | **172** | 15 s | **14** / 32 / 44 / 102 / 172 | **17** / 30 / 51 / 72 / 79 / 91% |
| Ear-hook, one ear (1 ch) | 94% (44/47) | 398 | 9.5 s | 32 / 57 / 82 / 172 / 348 | 9 / 34 / 49 / 57 / 77 / 83% |
| **Full cap (18 ch)** | 91% (43/47) | 132 | 13 s | **10 / 15 / 23 / 44 / 112** | 4 / 34 / **68 / 83 / 89 / 94**% |
| Full cap + ECG | 89% (42/47) | **113** | 14 s | 14 / 23 / 34 / 56 / 121 | 13 / 28 / 57 / 79 / 89 / 91% |

### 4.2 What it means

1. **Two ear-level channels see adult focal seizures.** The ear-hook caught **96%** of seizures in never-seen adults, as many as the 18-channel cap (91%). Same pattern as CHB-MIT in children (88% vs 85%).
2. **The cap's advantage is false alarms, about 2–3× fewer** at matched sensitivity (e.g. 23 vs 60 at 70% caught). CHB-MIT showed about 1.7×.
3. **ECG helps the ear-hook, not the cap.**
   - Ear-hook: false alarms fall **27–48%** at 50–70% caught (27 → 14 at 50%), and 45% at the default threshold (310 → 172). Same direction as SeizeIT2 (−18% to −31%).
   - Full cap: ECG makes things slightly worse (10 → 14 at 50%).
   - With 18 EEG channels there is little left for the heart to add. With 2 channels, the heart substitutes for the missing spatial information.
4. **Two ears beat one**: fewer false alarms at every sensitivity, as in CHB-MIT.
5. **Ear-EEG here is far better than on SeizeIT2** (27 vs 104 FA/day at 50% caught). **This does not show that SeizeIT2's electrodes were bad.** The datasets differ in several ways at once:
   - 13/14 Siena patients have **temporal-lobe** epilepsy (on SeizeIT2, temporal seizures were caught 2× more often: 43% vs 19%);
   - the "ear" channels here are **scalp** electrodes at T7–P7, higher up than the mastoid;
   - there are only 14 patients.

   Together with Test 1, the evidence points to **position and seizure type**, not electrode quality.

### 4.2b Follow-up: ear + back of the head, and precision

Extending the ear-hook towards the back of the head helps a lot. At 50% caught:
- ear + back of head (4 ch): 18 false alarms/day, against 27 for the ear alone;
- ear + back + parietal (6 ch): 10 false alarms/day, matching the full cap.

CHB-MIT (23 children) confirms the same pattern. In SzCORE terms, the ear + back + ECG setup had the most trustworthy alarms: 65% of its alarms were real seizures, at 1.2 false alarms/day.

Full tables are in [`False_Alarm_Reduction.md`](False_Alarm_Reduction.md), §3 and §8.

### 4.3 First look: one patient (PN00)

Before the full run, a patient-specific test on PN00 (55-year-old man, right temporal; 5 seizures, 3.2 h; leave-one-recording-out):

| Setup | @0.5 | Best threshold catching all 5 |
|---|---|---|
| Ear-hook, both ears | 5/5, 44 FA/day | 30 FA/day |
| **Ear-hook + ECG** | 5/5, 37 FA/day | **0 FA/day** |
| Ear-hook, one ear | 5/5, 59 FA/day | 7 FA/day |
| Full cap | 5/5, 37 FA/day | 0 FA/day |
| Full cap + ECG | 5/5, 30 FA/day | 0 FA/day |

For this patient, a personal ear-hook + ECG model matched the full cap. (One patient and 3.2 h: 1 false alarm = 7.5/day, and the best threshold is chosen after the fact, so this is optimistic.)

## 5. Test 3: TUSZ (pending access)

The **TUH EEG Seizure Corpus** (v2: 675 patients, 280+ with seizures, 3,500+ seizures; clinical scalp EEG including T3/T4/T5/T6) is the dataset that would fix the CNN's main problem: too few patients. It is free for research but **requires registration**; access takes about a day. Once credentials exist, it can be streamed the same way (features only). Deferred; the project moves to hardware (Prototype 1).

## 6. Engineering notes

### 6.1 Streaming Siena

- **Source:** PhysioNet's AWS open-data mirror (`physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0`), the same route as CHB-MIT.
- **Transfer:** each EDF (75 MB – 2.1 GB) is read into RAM, and only the 20 electrodes needed for the 18 bipolar channels plus EKG are decoded.
- **Storage:** features are saved atomically; **no EDF is written to disk**.
- **Memory:** channel pairs are decoded on demand rather than all at once, since a 13-h file × 20 channels would be about 2 GB of float32.
- **Throughput:** 2 workers; all 21.8 GB streamed in 46 min, with no failures.
- **Seizure timing:** onsets come from each EDF header's own start time, not the text file.

### 6.2 Typos in the dataset's seizure lists (all caught by parser checks)

| File | Problem | Handling |
|---|---|---|
| `Seizures-list-PN01.txt` | File name given once before both seizures, and as `PN01.edf` (the file is `PN01-1.edf`) | Parser carries the file name forward; a name that doesn't exist maps to the patient's only EDF |
| `Seizures-list-PN06.txt` | `PNO6-1.edf` (letter O instead of zero) | Normalised |
| `Seizures-list-PN00.txt` | Seizure 3: 18.28.29 – **19**.29.29 (61 min inside a 42-min recording) | Treated as an hour typo: end → 18.29.29 (1 min), with a logged note |

All 47 seizures were recovered and checked against each recording's length.

### 6.3 A scikit-learn 1.9 slowdown, found and fixed

**Symptom.** The PN00 test (5 recordings!) ran 8+ minutes without finishing.

**Diagnosis:**
1. **Profiling:** one model fit on 4,523 rows × 252 features took 37 s.
2. **Not contention:** it was not CPU contention (1 thread: 30 s; CPU load 26%), and not the data (random data of the same shape: 26 s).
3. **Where the time goes:** a profile showed **19.46 of 19.54 s** in `_BinMapper.fit`, the step that bins features before boosting.
4. **Weights are the trigger:** with sample weights (`class_weight="balanced"`), scikit-learn 1.9 computes each feature's 254 bin edges by calling `_weighted_percentile` **once per edge**. That function then loops over percentiles in Python through array-API wrappers.
   - Fit time with weights: 55 s; without weights: 5 s.
   - Above 200,000 rows scikit-learn subsamples and drops the weights, so **SeizeIT2 was never affected**. Siena and CHB-MIT-sized training sets are.

**Fix** (`detect/detect.py`, `_speed_up_weighted_binning()`):
- a vectorised twin of the same computation: the same sort call, the same cumulative weights, the same "averaged_inverted_cdf" rule, but one `searchsorted` for all edges;
- it is installed at import, so every script uses it.

**Verification:**
- **0 of 200** random features (plain, tie-heavy, discrete, with missing values) had any threshold difference;
- model predictions were **bit-identical**;
- fit time 33.4 s → 3.9 s.

The full Siena run (70 models) then took **3 minutes**.

## 7. Conclusions for Prototype 1

| Finding | Design consequence |
|---|---|
| Electrode **contact** drives false alarms (Test 1) | Measure contact quality continuously (impedance / amplitude stability); gate alarms on it; prompt "reseat the ear-hook" |
| Electrode **position** limits what is seen (Tests 1–2) | Place EEG contacts as high and as far back around the ear as comfort allows (towards T7/T8 and P7/P8, not only the mastoid); consider an extra contact per side |
| Two ears > one (CHB-MIT, Siena) | Bilateral device, as planned |
| ECG/heart helps a 2-channel device, not a full cap (Siena, SeizeIT2) | In-ear PPG is essential for the ear-hook |
| Ear-level EEG catches 88–96% of seizures in new users (CHB-MIT, Siena) | The ear site is sound; the work is false-alarm control |

## 8. Limitations

- **Test 1:**
  - compares patients **within SeizeIT2's range** of contact quality, so it cannot say what much better electrodes would do;
  - unstable amplitude may partly reflect patient movement, not only contact;
  - 71 patients.
- **Test 2:**
  - 14 patients and 47 seizures, so one seizure is about 2 percentage points; small differences (e.g. 17% vs 4% at ≤1/day) are noise;
  - mostly temporal-lobe epilepsy;
  - hospital recordings;
  - the "ear-hook" is still a proxy made from **scalp** electrodes at T7–P7 / T8–P8, not a real ear device;
  - best-along-the-curve numbers are slightly optimistic;
  - no personalisation was run on Siena.
- **No dataset with a real ear-hook exists yet.** Only our own hardware can answer that.

## 9. Reproduce

From `epilepsy_sim/`:

```bash
uv run detect/analysis_electrodes.py       # Test 1 (needs the cached outputs of seizeit2.py personal, k=2)
uv run detect/siena.py index               # parse the Siena seizure lists (prints all 47 seizures)
uv run detect/siena.py fetch --workers 2   # stream all 41 EDFs (21.8 GB through RAM, ~46 min); features only
uv run detect/siena.py run                 # leave-one-patient-out, 5 setups (~3 min) -> detect/results_siena/metrics.json
```

## 10. References

- Detti P, Vatti G, Zabalo Manrique de Lara G. EEG synchronization analysis for seizure prediction: a study on data of noninvasive recordings. *Processes* 8(7):846, 2020. Siena Scalp EEG Database, PhysioNet, doi:10.13026/5d4a-j060 (CC BY 4.0).
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220, 2000.
- Gu Y, Cleeren E, Dan J, et al. Comparison between scalp EEG and behind-the-ear EEG for development of a wearable seizure detection system for patients with focal epilepsy. *Sensors* 18(1):29, 2018. [PMC5795884](https://pmc.ncbi.nlm.nih.gov/articles/PMC5795884/)
- SeizeIT1, KU Leuven RDR, doi:10.48804/P5Q0OJ (no longer shared: ethical approval expired). [Link](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi%3A10.48804%2FP5Q0OJ)
- Bhagubai M et al. SeizeIT2: wearable dataset of patients with focal epilepsy. *Scientific Data*, 2025.
- Shah V et al. The Temple University Hospital Seizure Detection Corpus. *Front Neuroinform* 12:83, 2018. [arXiv:1801.08085](https://arxiv.org/pdf/1801.08085)

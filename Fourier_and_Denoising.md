# Fourier analysis and noise reduction: can cleaning the ear EEG cut false alarms?

*30 September 2026. Scripts: `epilepsy_sim/detect/fourier_denoise.py` (single patients), `epilepsy_sim/detect/seizeit2_spectral.py` (all 125 patients). Results: `detect/results_seizeit2/fourier_denoise.json`, `spectral_general.json`, `fourier_sub-103.png`, `fourier_sub-047.png`.*

> **Short answer.** Partly.
> - **Fourier analysis explains the false alarms.** They are mostly broadband muscle and movement bursts, with about twice the power of seizures and more energy above 30 Hz. Mains (50 Hz) noise matters only in some recordings.
> - **Cleaning helps an ear-EEG-only detector.** A 30 Hz low-pass or 50 Hz notch, plus five "artefact-aware" spectral features, cut its false alarms by up to 48% at high sensitivity and lift strict-budget detection by about 3 points across all 125 patients.
> - **The gain mostly overlaps with the heart and context features.** On the best multimodal model the gain shrinks to 6–15%, and to nothing at the strictest budgets.
> - **Adopt it anyway.** It is free, and it matters most when the heart sensor drops out. It is not the fix for false alarms.

---

## Contents

1. [Why](#1-why)
2. [What the pipeline already did](#2-what-the-pipeline-already-did)
3. [Method](#3-method)
4. [Fourier analysis of two patients](#4-fourier-analysis-of-two-patients)
5. [Noise reduction on two patients](#5-noise-reduction-on-two-patients)
6. [The proper test: all 125 patients](#6-the-proper-test-all-125-patients)
7. [What it means](#7-what-it-means)
8. [Engineering notes](#8-engineering-notes)
9. [Limitations](#9-limitations)
10. [Reproduce](#10-reproduce)

---

## 1. Why

The question was whether a Fourier look at the raw EEG, or simply removing noise, could lower false alarms in general.

## 2. What the pipeline already did

The detector was already Fourier-based. Every 4 s window of each ear-EEG channel is:
- band-passed at 0.5–45 Hz (4th-order Butterworth);
- turned into an FFT (Hann window);
- summarised by 5 log band powers (δ 0.5–4, θ 4–8, α 8–13, β 13–30, γ 30–45 Hz) and 5 relative band powers, plus line length, log-variance and Hjorth mobility/complexity.

The new questions were therefore:
1. What does the **spectrum of seizures vs false alarms** look like?
2. Does **extra cleaning** remove false alarms?
3. Does the **information the 45 Hz cut throws away** help?

## 3. Method

### 3.1 Streaming

Each patient's 2-channel behind-the-ear EEG (256 Hz) was streamed from OpenNeuro into RAM, run by run. Nothing was written to disk except features.

### 3.2 Cleaning variants

| Variant | What it does | Targets |
|---|---|---|
| baseline | 0.5–45 Hz band-pass (the existing pipeline) | – |
| notch | + 50 Hz and 100 Hz IIR notch (Q = 30) | Mains interference leaking through the gentle 45 Hz cut |
| lowpass30 | 0.5–30 Hz band-pass instead | Muscle (EMG) contamination above 30 Hz |
| clip | Samples beyond 6 × the running robust SD (MAD over 10 s) are clipped | Electrode pops, big movement spikes |
| lowpass30 + clip | Both | – |

### 3.3 Artefact-aware spectral features

These are new. Per channel and 4 s window, they are computed on the **unfiltered** signal, so the 45–100 Hz band that the pipeline normally discards is kept:

| Feature | Definition | Why |
|---|---|---|
| Muscle index | 45–100 Hz power / 0.5–100 Hz power (mains band 48–52 Hz excluded) | Muscle artefact lives here; brain activity mostly doesn't |
| Spectral flatness | Geometric / arithmetic mean of the 2–45 Hz spectrum | 1 = white, broadband burst (artefact); 0 = peaky rhythm (seizure) |
| Spectral entropy | Normalised Shannon entropy of the 2–45 Hz spectrum | Same idea, different estimator |
| 1/f slope | Linear fit of log power vs log frequency, 2–45 Hz | Muscle flattens the slope; slow seizure activity steepens it |
| Peak frequency | Frequency of the maximum in 1–30 Hz | Rhythmic seizure activity has a clear peak |

The variants "X + spectral" add these 10 features (5 per channel, each normalised against its recent baseline) to the cleaned EEG features.

### 3.4 Evaluation

- **Single patients:** patient-specific detector (HistGradientBoosting, EEG only). Runs split into 5 chronological folds; causal 10-min baseline normalisation; usual event scoring.
- **All patients:** the general model with patient-wise 5-fold cross-validation, **the same folds and the same training windows** as the main SeizeIT2 study (twin seeds). Only the EEG features change. Both the ear-EEG-only setup and the best multimodal setup (EEG + ECG + context + HR-excess) were tested.

## 4. Fourier analysis of two patients

The patients were chosen from the cached results:
- **sub-103:** 15 seizures, 90 h, 722 false alarms/day for the general model, only 33% of seizures caught.
- **sub-047:** 64 seizures, 68 h.

### 4.1 Relative band power (median spectra)

**sub-103:**

| | δ | θ | α | β | γ 30–45 | Muscle 45–100 | Mains 49–51 | Total power |
|---|---|---|---|---|---|---|---|---|
| Background | 34.6% | 8.9% | 9.2% | 16.8% | 11.5% | 19.0% | 2.0% | 506 µV² |
| Seizure | **40.1%** | 4.6% | 5.9% | 15.1% | 12.0% | 22.2% | 2.2% | 1,272 µV² |
| **False alarm** | 35.9% | 2.6% | 3.5% | 15.7% | **16.3%** | **25.8%** | 2.1% | **2,467 µV²** |

**sub-047:**

| | δ | θ | α | β | γ | Muscle | Mains | Total |
|---|---|---|---|---|---|---|---|---|
| Background | 72.7% | 8.7% | 1.9% | 2.3% | 1.1% | 13.4% | **11.9%** | 3,182 µV² |
| Seizure | **82.8%** | 4.2% | 2.1% | 2.5% | 0.5% | 8.0% | 7.4% | 4,718 µV² |
| **False alarm** | 71.5% | 9.2% | 3.9% | 3.7% | 1.3% | 10.4% | 9.2% | **7,035 µV²** |

### 4.2 What the spectra show

1. **False alarms carry the most power:** about 2× the seizures in both patients, and about 5× (sub-103) or 2× (sub-047) the background.
2. **They are broadband bursts.** In sub-103 the false-alarm spectrum lies above the seizure spectrum at every frequency from 10 to 100 Hz, with a larger γ and muscle share. In the spectrogram they appear as short, bright **vertical stripes** (all frequencies at once): the signature of muscle and movement artefacts. This matches the false-alarm taxonomy (`False_Alarm_Reduction.md` §7: muscle 12%, big movements 12%).
3. **Seizures differ:** more slow-wave (δ) share and a moderate, sustained power rise.
4. **Mains noise depends on the recording.** It is 2% of power for sub-103 but **7–12% for sub-047**. The pipeline's gentle 4th-order 45 Hz filter only partly attenuates strong 50 Hz interference, and the residue leaks into the γ and line-length features.

Figures: `detect/results_seizeit2/fourier_sub-103.png`, `fourier_sub-047.png`. Each shows the median spectra plus spectrograms around a seizure and around a false alarm.

## 5. Noise reduction on two patients

**sub-103** (15 seizures, 89.6 h, 7 runs):

| Variant | FA/day at 50 / 70% caught | Caught at ≤20 / ≤50 FA/day | At 0.5 |
|---|---|---|---|
| baseline | 292 / 462 | 0 / 7% | 7%, 44 FA/day |
| notch | 600 / – | 7 / 13% | 7%, 40 |
| lowpass30 | 244 / 477 | 13 / 20% | 20%, 41 |
| clip | 506 / 667 | 7 / 20% | 20%, 51 |
| lowpass30 + clip | 305 / – | 13 / 20% | 13%, 36 |
| baseline + spectral | 208 / 574 | 7 / 13% | 7%, 35 |
| notch + spectral | 178 / 536 | 7 / 13% | 7%, 32 |
| **lowpass30 + spectral** | **159** (−46%) / 613 | 13 / 13% | 13%, 33 |

**sub-047** (64 seizures, 68.3 h, 7 runs):

| Variant | FA/day at 50 / 70% caught | Caught at ≤1 / 5 / 20 / 50 FA/day | At 0.5 |
|---|---|---|---|
| baseline | 29.5 / 154.5 | 23 / 34 / 42 / 56% | 52%, 30 FA/day |
| **notch** | 9.5 (−68%) / 70.9 (−54%) | 31 / 48 / 52 / 61% | 56%, 33 |
| lowpass30 | 18.6 / 186.1 | 27 / 38 / 50 / 58% | 56%, 40 |
| clip | 20.4 / 294.7 | 22 / 39 / 48 / 59% | 58%, 27 |
| lowpass30 + clip | 27.0 / 128.2 | 22 / 39 / 47 / 56% | 55%, 40 |
| baseline + spectral | 7.7 (−74%) / 220 | 28 / 45 / 52 / 59% | 52%, 21 |
| notch + spectral | 6.3 (−79%) / 135 | **34** / 41 / **61** / 61% | **61%**, 23 |
| **lowpass30 + spectral** | **4.6 (−84%) / 108 (−30%)** | 27 / **50** / 59 / **64%** | 59%, 25 |

**Single-patient conclusions:**
- **Filtering alone gives mixed results.** The notch helps only where mains is present (sub-047), and clipping hurts.
- **Measuring the artefacts works better.** Low-pass or notch plus the spectral features gave the biggest cuts: **−46% (sub-103), −84% (sub-047)** at 50% caught.
- **sub-103's seizures stay nearly invisible** to behind-the-ear EEG, whatever the cleaning. That is the signal ceiling.

## 6. The proper test: all 125 patients

All 2,850 EEG files (~48 GB) were streamed through memory with 6 workers in about 2.2 h. Per run, the notch EEG features, low-pass EEG features and spectral features were stored on the main window grid. Twelve runs hit transient network timeouts and succeeded on the automatic retry. Evaluation used 125 patients, 883 seizures and 11,626 h, with patient-wise 5-fold CV and the same folds and training windows as the main study.

**Ear EEG only:**

| Setup | FA/day at 50 / 60 / 70 / 80% caught | Caught at ≤1 / 5 / 20 / 50 / 200 FA/day | At 0.5 |
|---|---|---|---|
| Baseline | 92.9 / 161.7 / 239.3 / 757.4 | 7 / 14 / 26 / 38 / 66% | 79%, 608 FA/day |
| + notch | 94.0 / 139.6 / 241.8 / 541.3 | 8 / 14 / 28 / 38 / 67% | 82%, 610 |
| + notch + spectral | **87.9 / 131.3 / 227.6** / 452.9 | 8 / 16 / 29 / **43** / 66% | 82%, 583 |
| **+ lowpass30 + spectral** | 90.2 / 133.9 / 231.1 / **396.9** | **10 / 17 / 30** / 41 / **68%** | 81%, 587 |

**Best model (EEG + ECG + context + HR-excess):**

| Setup | FA/day at 50 / 60 / 70 / 80% caught | Caught at ≤1 / 5 / 20 / 50 / 200 FA/day | At 0.5 |
|---|---|---|---|
| Baseline | 43.1 / 77.7 / 150.1 / 356.4 | 17 / 29 / 41 / 53 / 74% | 73%, 179 FA/day |
| + notch + spectral | 42.7 / 77.3 / **139.9** / 355.9 | 16 / 29 / 42 / 53 / 76% | 74%, 178 |
| **+ lowpass30 + spectral** | **36.8 / 71.2** / 150.1 / **335.7** | 16 / 29 / **43 / 54**% | 73%, 179 |

**Baselines reproduce the main study.** EEG only: 161.7 at 60% and 14% at ≤5 per day. Best model: 28–29% at ≤5 per day. Small differences come from the finer threshold grid used here.

## 7. What it means

1. **For an ear-EEG-only detector the gain is consistent but modest:**
   - −17 to −19% false alarms at 60% caught;
   - −40 to −48% at 80% caught;
   - +3 points at ≤1 and ≤5 FA/day with the 30 Hz low-pass;
   - about +3 points of seizures caught at the default threshold.
2. **With heart and context features it mostly overlaps:**
   - best model with low-pass + spectral: −15% at 50%, −8% at 60%, −6% at 80%;
   - no change at the strictest budgets.

   The heart signal and the minute-scale context already tell "a noisy burst with a calm heart and no build-up" apart from a seizure.
3. **The single-patient results overstated the effect.** They came from patients with unusually strong muscle or mains contamination. Averaged over everyone the effect is modest, which is why the full test mattered.
4. **For Prototype 1:**
   - Adopt a **30 Hz low-pass + the five spectral features** in the pipeline. They cost no hardware and little computation, and help most when the in-ear heart signal drops out.
   - Keep a **50 Hz notch** for recordings with mains interference. Detecting interference is itself easy: the 49–51 Hz share.
   - Use the muscle index as an input to the **contact-quality gate** as well.
   - The big false-alarm levers remain elsewhere: electrode position (ear + back of head), tonic-clonic alarm mode, the personal verifier, and the extra sensors (jaw EMG, head IMU, contact impedance).

## 8. Engineering notes

- **Spectral features** are computed on the *unfiltered* signal (median removed), so that the 45–100 Hz band is available.
- **Window alignment:** the window count is computed exactly as in the main feature extraction, so the new features line up row-for-row with the existing ones. This was verified on a test run: 1,784 windows in both.
- **Throughput:** the first all-patient attempt ran at about 2 min per run, with 3 workers and the CPU 44% busy. The bottleneck was computation (three filter passes plus FFTs), not the network (2.2 MB/s per stream). With 6 workers it ran at about 3.3 s per run, and cached runs were skipped on restart.
- **A self-matching kill command.** A PowerShell command meant to stop the old queue matched its own command line (it contained "queue14") and tried to terminate itself. The old processes were then confirmed stopped with a filter that excludes the current PID.

## 9. Limitations

- The single-patient analyses cover 2 patients with patient-specific models. The all-patient test is the result to rely on.
- The spectral features were tested with the general model; the personal verifier and tonic-clonic mode were not re-run with them.
- Hospital recordings with adhesive mastoid electrodes. A dry-electrode ear-hook in daily life will have more artefacts, so the spectral features may matter more there. This is to be measured with Prototype 1.
- Wavelet denoising was not tested (PyWavelets was not installed). ICA is not possible with 2 channels.

## 10. Reproduce

From `epilepsy_sim/`:

```bash
uv run detect/fourier_denoise.py sub-103 sub-047        # single-patient Fourier analysis + 5 cleaning variants (~15 min)
uv run detect/fourier_denoise.py sub-103 sub-047 --variants baseline,baseline+spectral,notch+spectral,lowpass30+spectral
uv run detect/seizeit2_spectral.py fetch --workers 6    # all 2,850 EEG runs streamed, features only (~2.2 h)
uv run detect/seizeit2_spectral.py run                  # 7 setups, patient-wise CV (~50 min)
```

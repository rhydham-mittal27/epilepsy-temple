# Temple Wearable — Technical Dossier

*As of 2026-09-27*

Temple is a head-worn optical wearable from Deepinder Goyal, Zomato's founder. It is **not** a Zomato or Eternal product. It sits on the side of the forehead and uses multi-wavelength PPG (photoplethysmography: light shone into skin, reflected light measured). From that signal it computes a proprietary Brain Flow index and an Entropy score. It has no regulatory clearance and is sold as a general-wellness device.

Each fact below is tagged by how much you can rely on it:

- **Confirmed**: from Temple's own preprints or website.
- **Reported**: from press coverage, which sometimes conflicts.
- **Inferred**: my engineering estimate, not stated by Temple.

---

## 1. Company and research setup

Temple started as a lab project and became a funded company in early 2026. It now owns a London clinic to run its clinical trials.

| Date | Event | Detail |
| --- | --- | --- |
| 27 Aug 2026 | Bought Longevous (London longevity clinic) | Dr Robert Mohr became Chief Medical Officer and Dr Avi Roy became Head of Science. The aim is trials that can pass peer review |
| Jul 2026 | Targeting a ~$500M valuation | Reported, not closed; first employee share buyback |
| Feb 2026 | Raised $54M at ~$190M valuation | Investors: Goyal (lead), Steadview, Peak XV, Info Edge Ventures, Dharana; angels include Kunal Shah, Vijay Shekhar Sharma, Nithin and Nikhil Kamath. 30+ employees invested at the same valuation |
| Jan 2026 | Goyal stepped down as Eternal CEO | He moved to "higher-risk exploration" |
| 2025 | Research under Continue Research | Goyal's personal lab, with ~$25M of his own money. The prototype went viral and Goyal published the "Gravity Ageing Hypothesis" (the idea that gravity reduces blood flow to the brain and speeds ageing) |

**Legal entities:**

- Temple Pvt. Ltd., Gurugram, India.
- Temple Wearables Inc., San Francisco, USA.
- Contacts: build@temple.com, sachin@temple.com.

**Outside scientists:**

- David A. Boas, a leading diffuse-optics researcher, sits on Temple's Scientific Advisory Board and receives research support from Temple.
- Satchidananda Panda (Salk Institute) is a paid consultant.
- The Transcranial Doppler study ran at CCS University and LLRM Medical College, Meerut, with Temple funding.

---

## 2. Hardware

The device is an adhesive-mounted optical PPG pod using green, red and infrared light. Temple has not published exact wavelengths, sensor geometry, chipset or battery specs.

| Item | Detail | Confidence |
| --- | --- | --- |
| Sensing principle | Multi-wavelength reflectance PPG | Confirmed |
| Wavelengths | Green (shallow skin vessels), red and infrared (deeper). Exact nm values not disclosed | Confirmed |
| Placement | Front of the temple, above the cheekbone arch and beside the outer edge of the eye socket. This is over the superficial temporal artery, a scalp artery that doesn't feed the brain | Confirmed |
| Attachment | Medical-grade double-sided adhesive; a lifetime supply of tape comes with purchase | Confirmed / Reported |
| Raw PPG sampling | 32 Hz, green channel, right temple (age study) | Confirmed |
| Metric output rate | Brain Flow index 1 Hz; Entropy updates every second | Confirmed |
| Heart-rate error vs reference | Median 3.6–3.8% during exercise and posture changes | Confirmed |
| Source–detector spacing and geometry | Withheld; the paper lists this as a limitation | Confirmed (withheld) |
| Form factor | Pebble-shaped metallic pod in gold or silver. v2 is less than half v1's volume, and Goyal calls it "the smallest wearable on the market" | Reported |
| Production | First batch of 100 devices went to testers, including PV Sindhu | Reported |
| Battery, charging, weight, chipset, radio, IP rating, materials | Not published | Unknown |

**Placement debate (Jan 2026):**

- Neurologist Dr Sudhir Kumar suggested measuring at the neck over the internal carotid artery, which does supply the brain.
- Goyal answered that PPG signal resolution is "significantly higher at the temple than at the neck."
- He said machine-learning models trained against MRI and Transcranial Doppler bridge the gap between scalp and brain signals.

---

## 3. Software, algorithms and metrics

Temple exposes two proprietary scores, the Brain Flow index and Entropy. Both are closed-source and computed from optical features by machine-learning models.

**App:** Temple's mobile app was at version 9.6 during the TCD study. It has a "workout mode" that was used to record study sessions (confirmed).

**Brain Flow index (BFI), confirmed:**

- A number with no units. It shows change relative to your own baseline, not an absolute blood flow.
- It combines three groups of optical features:
    - PPG-based perfusion (how much blood reaches the tissue)
    - Haemoglobin-related optical changes across wavelengths
    - Pulse-waveform shape
- Goyal says the models are calibrated against Transcranial Doppler and MRI.

**Entropy, announced June 2026 (reported):**

- A score from 1 to 250, updated every second, for "metabolic and sympathetic demand". Temple markets it as "the real-time cost of being alive".
- Two derived scores: **Entropy Maxima** (peak output capacity) and **Entropy Minima** (lowest resting state).
- Claimed correlation with a metabolic cart (a lab machine that measures energy use from breath): r = 0.93 over more than 100 cardio sessions. Heart rate alone scored r = 0.55.
- The data is unpublished: no methods, participant data or peer review.

**Signal pipeline from Temple's age study (confirmed):**

1. Filter the raw green PPG signal: 0.5–10 Hz band-pass, 4th-order Butterworth, applied without phase shift.
2. Resample to 75 Hz and cut into non-overlapping 25-second windows.
3. Extract 89 pulse-shape features:
    - Timing, amplitude and area measures
    - Ratios between those measures
    - Ratios from the 1st, 2nd and 3rd derivatives, e.g. (Ad−Ab)/Aa and AGImod
    - Frequency-harmonic ratios
4. Reduce with principal component analysis. Five components explained 85.3% of the variation.

This pipeline is very likely close to what the production feature extractor does.

---

## 4. Validation studies

Three preprints (early papers, none peer-reviewed yet) show the Brain Flow index tracks reference brain-blood-flow measures at r ≈ 0.8–0.9 in healthy young adults. Temple funded all three.

| Preprint | Setup | Reference method | Key result |
| --- | --- | --- | --- |
| Temple vs depth-resolved time-domain NIRS (bioRxiv, Sep 2026, Gulati et al.) | 44 young men, 60 sessions: 30° head-down tilt, standing to squatting, standing to lying down | Three-module time-domain NIRS (near-infrared light measured at different tissue depths); a two-layer model separates scalp from brain | Brain layer vs scalp layer: tilt 0.91 vs 0.49; squat 0.84 vs 0.70; lying down 0.84 vs 0.77. Direction matched in 59 of 60 sessions |
| Temple vs Transcranial Doppler (medRxiv, Sep 2026, Kumar et al.) | 23 adults aged 18–30, 20 analysed per protocol: cycling, and standing ⇄ lying down | Viasonix Dolphin IQ, 2 MHz ultrasound measuring blood speed in the middle cerebral artery (MCAv) | Same-time r ≈ 0.80. With a time shift allowed: r = 0.84 exercise (−19 s), 0.82 posture (+9 s). Tracked brain flow rising while heart rate fell |
| Temple vs wrist PPG as an age signal (bioRxiv, Aug 2026, Liu et al.) | Temple group of about 124; wrist group of 240 (Microsoft Aurora-BP, MAX30101 at 500 Hz) | Chronological age | 14 pulse-shape features tracked age at the temple vs 3 at the wrist. Different people at each site, so not a head-to-head test |

**TCD study, further details:**

- Participants: 15 men and 8 women, skin tones Monk scale 2–8.
- Before testing: no caffeine or alcohol for 12 hours.
- Analysis: z-scores, correlations with time shifts of −30 to +30 s, Wilcoxon tests with Bonferroni correction, and Bland-Altman agreement plots.
- Ethics approval: SC-1/2025/9370.

**Limitations the authors admit:**

- The signal is optical and comes from outside the skull. Brain-specific measurement is not proven.
- MCAv is blood speed in one artery, not total blood volume.
- Only healthy young adults were tested, with no day-to-day repeatability testing.
- The algorithm is closed, so its internals can't be inspected.
- The studies are industry-funded, and authors have conflicts of interest.

---

## 5. Product status

A limited launch edition is expected to ship before the end of 2026 at around $1,000. It has no regulatory clearance.

| Item | Detail |
| --- | --- |
| Price | ~$1,000 or ₹80,000, including lifetime platform access and adhesive tape. One outlet says ₹72,000. **Not officially announced** |
| Timeline | Pre-orders around Q3 2026, shipping in Q4 2026 |
| Regulatory | No CDSCO (Indian regulator) or FDA clearance. Positioned as general wellness, not a medical device |
| Target users | Elite athletes first, now widening to sleep, stress, food and recovery tracking. Competes with Whoop, Oura and Garmin |
| Website | temple.com: early-access sign-up, Science & Research page |

**Hiring roadmap (Feb 2026 job posts):**

- BCI engineers for real-time EEG/EMG capture (brain and muscle electrical signals)
- Embedded engineers for hardware bring-up, on-device signal and image processing, and on-device AI
- Advanced materials, neural decoding, computational neuroscience and physiological AI

Applicants were also asked to have body fat below 16% for men or 26% for women. EEG hardware is **not** confirmed in any shipping product.

---

## 6. Engineering inference (unconfirmed)

These are my estimates of the likely architecture, based on what devices this size usually contain. Temple has disclosed none of it.

| Subsystem | Likely implementation | Reasoning |
| --- | --- | --- |
| Optical front end (the PPG sensor chip) | Multi-LED, multi-photodiode chip in the Maxim MAX86141/86176 or ADI ADPD4100 class; at least 3 LED colours and 2+ photodiodes at different distances | Separating shallow and deep signals needs at least two light paths, which fits the "depth" framing |
| Main chip | Low-power Bluetooth LE processor, Nordic nRF52/nRF54 class | Standard for tiny wearables; the embedded-AI hiring points to on-device feature extraction |
| Motion sensor | 6-axis IMU (accelerometer and gyroscope) | Needed to reject motion noise and detect posture, which is central to every study |
| Power | Li-po cell of about 20–50 mAh; 1–3 days per charge; magnetic or pogo-pin charging | Limited by the device's volume |
| Compute split | Filtering and features on the device; BFI and Entropy models in the app or cloud | Metrics come out at only 1 Hz, and the models are closed |
| Entropy model | Regression on heart rate, heart-rate variability, pulse-shape features, red/IR perfusion and motion, trained against metabolic-cart data | Validated against energy use, not information entropy |

---

## 7. Relevance to epilepsy

Temple cannot detect seizures as shipped.

- It measures optical blood-flow and pulse signals, not the brain's electrical activity (EEG).
- It sits over a scalp artery, not over brain tissue.
- At most, it might pick up the heart-rate and blood-flow changes that happen around a seizure. Nobody has validated that.
- One lower-reliability article (founderpin) claims "EEG, epilepsy detection". None of Temple's papers or official material supports that claim.
- If Temple adds EEG hardware later, as its BCI hiring hints, this could change.

---

## Sources

- Temple — Science & Research: https://www.temple.com/science-and-research
- Temple — home: https://www.temple.com/
- medRxiv — Temple vs Transcranial Doppler: https://www.medrxiv.org/content/10.64898/2026.09.02.26362022v1
- bioRxiv — Temple vs time-domain NIRS: https://www.biorxiv.org/content/10.64898/2026.09.17.751664v1
- bioRxiv — Temple PPG as an age signal: https://www.biorxiv.org/content/10.64898/2026.08.19.745616v1
- TechCrunch — $54M round: https://techcrunch.com/2026/02/27/after-zomato-deepinder-goyal-returns-with-a-54m-brain-monitoring-bet/
- ThePrint — Temple study coverage: https://theprint.in/tech/temple-deepinder-goyal-wearable-study/3036721/
- LatestLY — time-domain NIRS study: https://www.latestly.com/technology/what-does-temple-device-actually-do-know-what-recent-study-shows-7620983.html
- Telangana Today — placement debate: https://telanganatoday.com/zomatos-deepinder-goyal-and-neurologist-dr-sudhir-kumar-weigh-in-on-temple-wearable-in-viral-x-exchange
- Digit — v2 design: https://www.digit.in/news/general/deepinder-goyal-teases-new-version-of-temple-pre-orders-design-and-more-details.html/amp/
- Healthcare.digital — Longevous acquisition: https://www.healthcare.digital/single-post/deepinder-goyal-s-temple-acquires-longevous-to-pivot-hardware-engineering-toward-clinical-validation
- StartupTalky — $500M target: https://startuptalky.com/news/temple-targets-500-million-valuation/
- BusinessToday — Entropy launch: https://www.businesstoday.in/technology/news/story/what-if-your-body-had-a-cost-of-being-alive-score-deepinder-goyal-reveals-temples-new-discovery-538154-2026-06-19
- BusinessToday — Gravity Ageing Hypothesis: https://www.businesstoday.in/latest/trends/story/after-viral-backlash-deepinder-goyal-reintroduces-gravity-aging-theory-with-scientific-clarity-503083-2025-11-20
- Neurofounders — Entropy pivot: https://www.neurofounders.co/articles/temples-entropy-moves-wearables-away-from-the-brain
- Substack — Entropy critique: https://goelh.substack.com/p/temples-entropy-biomarker-real-breakthrough
- Omnicuris — Entropy critique: https://www.omnicuris.com/medshots/daily_updates/temple-entropy-biomarker-metabolic-wearable
- LatestLY — hiring roles: https://www.latestly.com/socially/technology/deepinder-goyal-hiring-engineers-for-neuroscience-startup-temple-with-strict-fitness-criteria-for-applicants-check-name-of-roles-7331162.html
- Goyal on X — recruiting post: https://x.com/deepigoyal/status/2027256222660133375
- WearableXP — specs (mostly speculative): https://wearablexp.com/smart-wearables/temple-wearable-explained/

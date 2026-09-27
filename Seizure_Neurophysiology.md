# What Happens in the Brain During a Seizure

*As of 2026-09-27. Companion to `Epilepsy_Overview.md` and the `epilepsy_sim/` simulation.*

At the neurological level, a seizure happens when the brain's balance between excitation and inhibition breaks down. A group of neurons then fires in abnormal lock-step and pulls neighbouring areas into the same rhythm.

- **Part A** covers the whole life cycle: between seizures → build-up → seizure → termination → recovery → long-term change.
- **Part B** zooms in on the seizure itself (the ictal phase), roughly second by second.

---

# Part A — The full seizure life cycle

## A0. The normal state

- Each neuron sits at about **−70 mV**. It fires a brief spike when excitatory input pushes it over threshold.
- **Excitation** comes mainly from **glutamate** acting on AMPA and NMDA receptors, which let Na⁺ and Ca²⁺ into the cell.
- **Inhibition** comes mainly from **GABA** acting on GABA-A receptors, which let Cl⁻ into the cell and keep it below threshold. Inhibitory interneurons (e.g. parvalbumin cells) act like brakes spread through the network.
- Normally, neurons fire **asynchronously**, each at its own time. Healthy brain activity is a sparse, loosely coordinated conversation, not a chorus.

## A1. Between seizures: the epileptic focus

In an epileptic brain, a region has become hyperexcitable (from a scar, a mutation, an infection, and so on). At the cellular level, that shows up as:

- **Faulty ion channels.** For example, *SCN1A* loss in Dravet syndrome disables the sodium channels of inhibitory interneurons, so the brakes fail. *KCNQ2* loss weakens the potassium current that normally damps firing.
- **Rewired circuits.** In hippocampal sclerosis, inhibitory interneurons die, and surviving excitatory cells sprout new connections onto each other (mossy-fibre sprouting). The result is recurrent excitatory loops.
- **The paroxysmal depolarization shift (PDS).** In the focus, groups of neurons suddenly depolarise for 50–200 ms and fire a burst of spikes, then stop. Summed together, these bursts are the **interictal spikes** seen on EEG. They are the signature of epilepsy even between seizures.

At this stage, strong inhibitory "surround" activity still contains the bursts, so no seizure occurs.

## A2. The transition to a seizure (pre-ictal, seconds to minutes)

- **Potassium builds up outside the cells.** Each spike releases K⁺. When firing is intense, astrocytes (support cells) can't clear it fast enough, and extracellular K⁺ rises from about 3 mM to 8–12 mM. That depolarises every nearby neuron and makes them easier to fire. This is a positive feedback loop.
- **Inhibition turns into excitation.** Heavy GABA activity loads neurons with Cl⁻, and the chloride pump (KCC2) can't keep up. GABA then weakens, and can even become excitatory.
- **Interneurons exhaust themselves.** Inhibitory cells fire so fast they enter **depolarization block** and fall silent.
- **Synchrony spreads without synapses.** Gap junctions (direct electrical connections between cells) and electric fields (ephaptic coupling) pull neurons into phase.
- **NMDA receptors switch on** in the depolarised cells. Ca²⁺ floods in, which lengthens the bursts.

EEG at onset: often **low-voltage fast activity** (20–100+ Hz), or a "hypersynchronous onset" (a string of large spikes).

## A3. The ictal phase (overview)

- **The focus is recruited.** Thousands to millions of neurons fire in rhythmic lock-step, typically at 3–30 Hz.
- **Spread.** Activity travels along white-matter pathways, and symptoms depend on the region reached. An **ictal wavefront** advances slowly (about 1 mm/s), while fast discharges travel rapidly behind it.
- **Loss of consciousness** occurs when the seizure disrupts the thalamus and the networks spread across both hemispheres.
- **Absence seizures are a thalamocortical loop problem.**
  - Thalamic relay neurons have T-type Ca²⁺ channels that produce rebound bursts.
  - A loop between cortex, the reticular thalamic nucleus and the relay thalamus locks into a **3 Hz spike-and-wave** rhythm. Each spike is a burst of excitation, and each wave is a period of GABA-B inhibition.
  - This is why ethosuximide, which blocks T-type channels, stops absence seizures.

(Details in Part B.)

## A4. Termination

Seizures usually stop by themselves within 1–3 minutes, through energy depletion, adenosine, acidosis, overdriven Na⁺/K⁺ pumps and the whole network exhausting itself at once. (Details in Part B, section B7.)

## A5. The postictal phase (minutes to hours)

- **The brain is suppressed.** Neurons are depleted and hyperpolarised, and adenosine lingers. The person feels confused and sleepy, and may have Todd's paralysis (temporary weakness), which reflects local exhaustion.
- **Blood flow falls.** After the seizure, blood flow in the affected region drops by up to about 50% for roughly an hour. Low oxygen follows. Animal research (Farrell & Teskey) links this to vessel constriction driven by COX-2 and L-type Ca²⁺ channels.
- **Postictal generalised EEG suppression, plus brainstem suppression of breathing and heart control, is the leading mechanism behind SUDEP.**

## A6. Long-term: "seizures beget seizures"

Repeated seizures can drive further change:

- Neuronal loss, especially in the hippocampus.
- Reactive astrocytes (gliosis) that handle K⁺ and glutamate poorly.
- Leaks in the blood-brain barrier that let albumin in, which triggers TGF-β signalling and more hyperexcitability.
- Inflammation (IL-1β, HMGB1).
- Further synaptic rewiring.

This process, called **epileptogenesis**, is why some epilepsies get worse over time and why early control matters.

---

# Part B — Inside the seizure, second by second

## B1. Second 0: onset, the brakes give way

**Ions:**

- Extracellular **K⁺** has already climbed from about 3 mM to 8–12 mM, because intense firing released it faster than astrocytes can clear it. That makes every nearby neuron less negative inside, closer to its firing threshold.
- Inside neurons, **Cl⁻** has built up because the chloride pump (KCC2) can't keep up. When GABA opens its Cl⁻ channels, less Cl⁻ rushes in, or it flows out. Inhibition weakens and can even turn into excitation.

**Cells:**

- Fast-spiking inhibitory interneurons (parvalbumin cells) fire at 200–400 Hz, then enter **depolarization block**: they get stuck partly activated and can't fire at all.
- The "inhibitory surround" that held the focus in check disappears.

**On EEG:** often a sudden flattening followed by **low-voltage fast activity** (20–100+ Hz).

**In the simulation (`network_seizure.png`):** this is the moment the blue GABA line drops.

## B2. Seconds 1–10: recruitment, everything fires together

**Cells:**

- Each excitatory (pyramidal) neuron goes into a **paroxysmal depolarization shift**. Its voltage jumps about 20–30 mV and stays up for 50–200 ms, firing a volley of spikes.
- The drivers:
    - **AMPA receptors** let Na⁺ in: fast excitation.
    - **NMDA receptors** have their Mg²⁺ block removed by the depolarisation and let **Ca²⁺** flood in. This lengthens the bursts.
    - **Persistent sodium currents** and **T-type or L-type Ca²⁺ channels** add bursts of their own.

**Synapses:** each burst releases large amounts of **glutamate**. It spills out of the synapse and activates neighbouring neurons, and astrocytes can't take it back up fast enough.

**Synchrony:**

- Gap junctions, electric fields (ephaptic coupling) and shared K⁺ in the space around cells all pull neurons into phase.
- Thousands, then millions, of neurons fire **within the same few milliseconds**.
- **In the simulation:** the vertical black bands in the raster. Synchrony went from 0.02 to 0.98.

**On EEG:** a rhythmic discharge that builds in amplitude, usually starting fast (about 10–30 Hz) and gradually slowing.

## B3. Seconds 5–60: spread across the brain

**The wavefront:**

- The seizure territory expands slowly (about 1 mm/s) as neighbouring inhibition is overrun.
- Behind it, fast discharges race to connected regions along white-matter pathways, such as the corpus callosum, the thalamus, and fibre bundles between regions.

**What the person experiences depends on which region is captured:**

| Region reached | What happens |
|---|---|
| Motor cortex | Jerking that spreads in sequence, e.g. hand → arm → face (the "Jacksonian march") |
| Hippocampus and amygdala | Fear, déjà vu, memory not being recorded |
| Insula | Stomach sensation, **heart-rate surge** |
| Thalamus and cortex in both hemispheres | **Consciousness lost** |
| Brainstem | Breathing changes, then the tonic-clonic phase |

**Why consciousness goes:** the seizure hijacks the frontal and parietal association cortex and the thalamus. That shuts down the subcortical arousal systems, so the brain can't integrate information.

## B4. Tonic-clonic seizures specifically

**Tonic phase (10–20 s):**

- Neurons fire continuously at high rates, with no gaps.
- Motor neurons are driven constantly, so every muscle contracts. The person stiffens and air is forced out past the vocal cords (the "epileptic cry").

**Clonic phase (30–90 s):**

- Inhibition is partly restored and starts interrupting the firing. Bursts alternate with silent periods: each **burst is a jerk**, and each **silence is relaxation**.
- As the brain's energy runs out, the silent periods lengthen, so the jerks get slower and further apart.
- **In the simulation (`epileptor_seizures.png`):** the spikes spread out toward the end of each seizure.

## B5. Energy and blood flow (the whole time)

**Energy:**

- Brain glucose and oxygen use jump **2–3×**. The Na⁺/K⁺ pumps are working flat-out to restore ion gradients.
- ATP falls. Tissue switches to anaerobic metabolism, so **lactate** rises and **pH** falls.

**Blood flow:**

- Neurovascular coupling works overtime. Nitric oxide, prostaglandins and K⁺ from neurons and astrocytes dilate blood vessels, so **local blood flow and blood volume rise sharply** at the focus.
- NIRS detects this rise on the seizure side in about 96% of cases.
- Early on, oxygen can briefly dip before the extra flow arrives. In long seizures, supply never catches up.
- **This rise in blood flow is what a Temple-style optical sensor could in theory detect.**

## B6. The autonomic surge

The seizure spreads into the insula, amygdala and hypothalamus, which triggers a **sympathetic surge**:

- Heart rate +35 bpm or more (ictal tachycardia)
- Rising blood pressure
- Sweating (what wristbands measure as electrodermal activity)
- Dilated pupils
- Irregular breathing, sometimes pauses

This is what wrist wearables (Embrace2, EpiWatch, NightWatch) detect.

## B7. The end: why it stops so abruptly

Several brakes kick in at once:

1. **Energy crash.** Low ATP opens ATP-sensitive K⁺ channels, which silence neurons.
2. **Adenosine**, a breakdown product of ATP, floods the tissue and strongly inhibits neurons through A1 receptors.
3. **Acidosis.** Falling pH blocks NMDA receptors and activates acid-sensing channels on inhibitory cells.
4. **The Na⁺/K⁺ pumps overshoot** and leave neurons more negative than normal (hyperpolarised).
5. **The whole network exhausts itself at once.** Everything stops together, so the EEG goes **suddenly flat** (postictal suppression).

**In the Epileptor model:** the slow variable `z` (the ion and energy build-up) reaches its peak and forces the system back to rest. That's why each seizure ends sharply.

## B8. If it doesn't stop (status epilepticus)

- Within minutes, neurons **pull GABA-A receptors inside the cell**, while **NMDA receptors are moved to the surface**. Inhibition weakens and excitation strengthens: the seizure sustains itself.
- **Ca²⁺ overload** triggers excitotoxicity (enzyme activation, mitochondrial failure, cell death), particularly in the hippocampus.
- This is why emergency treatment starts at 5 minutes, and why benzodiazepines, which act on GABA-A receptors, stop working if given late.

---

## Summary chain

**K⁺ rises + Cl⁻ loads** → **interneurons go silent** → **glutamate + NMDA Ca²⁺ drive depolarization shifts** → **gap junctions and fields lock neurons into sync** → **spread along pathways** (symptoms and loss of consciousness) → **energy use ×2–3, blood flow ↑, sympathetic surge** → **ATP ↓, adenosine and acidosis ↑** → **the whole network shuts off at once** → **postictal suppression and reduced blood flow**

## What a sensor could detect

| Seizure event | Signal | Detectable by |
|---|---|---|
| Interictal spikes, ictal rhythm, postictal suppression | Electrical | EEG only (scalp, under-scalp or intracranial) |
| Rise in local blood flow and volume during the seizure | Haemodynamic | NIRS / multi-wavelength optical (a Temple-style sensor, in theory) |
| Heart-rate and blood-pressure surge | Autonomic | PPG, ECG |
| Sweating | Autonomic | Electrodermal sensor (wrist) |
| Convulsions | Motor | Accelerometer / IMU |
| Postictal drop in blood flow | Haemodynamic | NIRS / optical (in theory) |

---

## Notes

This document describes well-established seizure physiology from neuroscience textbooks and review literature, written up as an explanatory synthesis. Specific figures cited elsewhere in this project:

- Ictal tachycardia (+35 bpm / +55%, ~51% of paediatric seizures): https://www.sciencedirect.com/science/article/pii/S1059131124003182
- NIRS ictal hyperperfusion (96% vs 69% SPECT): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12196729/
- Glutamate/GABA imbalance in seizures: https://www.sciencedirect.com/science/article/pii/S1059131121002168
- Status epilepticus review: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9491413/
- AAN SUDEP guideline: https://www.neurology.org/doi/10.1212/WNL.0000000000003685

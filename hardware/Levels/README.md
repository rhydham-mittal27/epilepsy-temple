# Prototype 1 build levels: how 'from scratch' and how costly

Each level is a complete, buildable set. Inside each: `<Component>/<Model>/` with `INR_<price>.txt`, `image.png` (dimensioned drawing) or real photos, and `size.txt`, plus a README with the total.

| Level | Skills | EEG quality | Total (Rs, one device incl. one-time costs) |
|---|---|---|---|
| [Level 0 - Buy a kit (NPG Lite Beast Pack)](Level_0_Buy_Kit/README.md) | None: plug in, add sensors, run the Chords software | Medium: 12-bit, 6 channels, no electrode-contact check. Great for EMG/ECG; EEG must pass the noise and alpha tests. | **~27,950** |
| [Level 1 - Cheap ready-made modules on an ESP32](Level_1_Cheap_Modules/README.md) | Wiring on a breadboard/perfboard, Arduino code | Low for EEG (ECG-grade amplifiers + the ESP32's own noisy ADC); fine for jaw EMG, ECG, motion and PPG experiments. | **~6,640** |
| [Level 2 - Open-source ADS1299 board, factory-assembled + ESP32](Level_2_Open_Source_Board/README.md) | Ordering from JLCPCB (upload open design files), wiring, code | High: 24-bit ADS1299, 4 channels, built-in electrode-contact check. Best EEG per rupee. | **~14,990** |
| [Level 3 - Fully custom board (our design), factory-assembled](Level_3_Fully_Custom_Factory/README.md) | PCB design (KiCad), firmware (nRF Connect / Zephyr), 3D printing | High, in the real ear-hook form: 8 channels, contact check, in-ear PPG, head IMU, jaw EMG. | **~38,130** |
| [Level 4 - Fully custom board, hand-soldered by you](Level_4_Hand_Soldered_Custom/README.md) | Fine-pitch SMD soldering (0.5 mm TQFP, 2.5 mm LGA), hot air, inspection | High if soldered well; risk of bad joints on the ADS1299 / IMU (inspect under magnification). | **~30,630** |

Level 3 per-device cost after the first batch: ~Rs 8,000-13,000 (parts only). Level 2 is the cheapest route to high-quality (24-bit) EEG. All prices are estimates; confirm before ordering.

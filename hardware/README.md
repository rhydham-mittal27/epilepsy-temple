# Prototype 1 hardware (fully custom build)

Each component has a folder; inside it, one folder per model with `INR_<price>.txt` (price estimate, quantity, where to buy), `image.png` (dimensioned drawing from datasheet sizes; not a photo) and `size.txt`. Regenerate with `uv run python hardware/generate_hardware_folders.py` (from `epilepsy_sim/`, or any Python with matplotlib).

| Component | Model | Price / unit (Rs) | Qty per device | Line total (Rs) |
|---|---|---|---|---|
| EEG EMG Front End | `TI_ADS1299IPAGR` | 5,000 | 1 | 5,000 |
| EEG EMG Front End | `TI_ADS1299-6PAGR` | 3,000 | - | 0 |
| Microcontroller Bluetooth | `Raytac_MDBT50Q-1MV2_nRF52840` | 900 | 1 | 900 |
| Microcontroller Bluetooth | `Nordic_nRF5340_QKAA` | 550 | - | 0 |
| PPG SpO2 Sensor | `Maxim_MAX30101EFD` | 650 | 1 | 650 |
| PPG SpO2 Sensor | `ADI_MAX86141ENP` | 500 | - | 0 |
| Motion Sensor IMU | `Bosch_BMI270` | 280 | 1 | 280 |
| Temperature Sensor | `TI_TMP117AIDRVR` | 300 | 1 | 300 |
| Flash Memory | `Winbond_W25Q128JVSIQ` | 80 | 1 | 80 |
| Battery | `LiPo_402030_250mAh` | 350 | 1 | 350 |
| Battery Charger | `Microchip_MCP73831T-2ACI_OT` | 50 | 1 | 50 |
| Voltage Regulator Digital | `TI_TPS7A2033PDBVR` | 60 | 1 | 60 |
| Boost Converter Analog | `TI_TPS61023DRLR` | 60 | 1 | 60 |
| Voltage Regulator Analog | `TI_TPS7A2050PDBVR` | 60 | 1 | 60 |
| ESD Protection | `TI_TPD4E1U06DBVR` | 40 | 3 | 120 |
| Electrode Dry Comb | `Dry_EEG_Comb_Electrode_AgAgCl` | 600 | 4 | 2,400 |
| Electrode Ear Flat | `AgAgCl_Flat_Disc_10mm` | 250 | 6 | 1,500 |
| Connector | `JST_SH_SM04B-SRSS-TB` | 30 | 4 | 120 |
| Cable | `Shielded_32AWG_Micro_Coax` | 400 | 1 | 400 |
| PCB Main Board | `JLCPCB_4-layer_40x30mm` | 800 | 1 | 800 |
| PCB Ear Tip Flex | `JLCPCB_Flex_12x8mm` | 500 | 1 | 500 |
| Mechanical Housing | `3D_Printed_EarHook_and_Band` | 1,800 | 1 | 1,800 |
| Programmer Debugger | `SEGGER_J-Link_EDU_Mini` | 6,000 | - | 0 |
| Reference Heart Rate Strap | `Polar_H10` | 8,000 | - | 0 |

**Parts per device (primary models, qty > 0): about Rs 15,430.**
One-time tools (J-Link, Polar H10): about Rs 14,000. Alternatives (qty 0) are listed for comparison.
PCB fabrication + assembly is priced per board here; the first 5-board order is about Rs 12-25k with assembly.

Prices are estimates (Sep 2026, about Rs 88 per US$). Confirm with the distributor before ordering. Safety: battery-only while worn; never connect to a mains-powered charger or laptop while on the head.

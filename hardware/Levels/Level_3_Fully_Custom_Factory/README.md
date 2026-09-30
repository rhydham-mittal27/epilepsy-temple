# Level 3 - Fully custom board (our design), factory-assembled

**Skills:** PCB design (KiCad), firmware (nRF Connect / Zephyr), 3D printing

**EEG quality:** High, in the real ear-hook form: 8 channels, contact check, in-ear PPG, head IMU, jaw EMG.

| Component | Model | Price / unit (Rs) | Qty | Line total (Rs) |
|---|---|---|---|---|
| EEG EMG Front End | `TI_ADS1299IPAGR` | 5,000 | 1 | 5,000 |
| Microcontroller Bluetooth | `Raytac_MDBT50Q-1MV2_nRF52840` | 900 | 1 | 900 |
| PPG SpO2 Sensor | `Maxim_MAX30101EFD` | 650 | 1 | 650 |
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
| Mechanical Housing | `3D_Printed_EarHook_and_Band` | 1,800 | 1 | 1,800 |
| PCB Batch Order | `JLCPCB_First_Batch_5_boards_with_assembly` | 18,000 | 1 | 18,000 |
| Programmer Debugger | `SEGGER_J-Link_EDU_Mini` | 6,000 | 1 | 6,000 |

**Total: about Rs 38,130** (estimates, Sep 2026; confirm before ordering).

Safety: battery-only while worn; never connected to a mains charger or laptop on the head.

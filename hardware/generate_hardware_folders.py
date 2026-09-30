"""Builds the Prototype 1 (fully custom) hardware folder:

    hardware/<Component>/<Model_Name>/
        INR_<price>.txt   price estimate (per unit, quantity, line total, where to buy)
        image.png         dimensioned drawing of the part (top + side view), drawn from datasheet sizes
        size.txt          package, dimensions, notes

Prices are estimates (September 2026, about Rs 88 per US$, incl. a rough 20-40 % import margin where
relevant); confirm them when ordering. Images are technical drawings, not photos: product photos are
linked in each price file. Re-run after editing the PARTS table:  uv run python hardware/generate_hardware_folders.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent

# component folder, model folder, price per unit (INR, estimate), qty per device, price range text,
# size lines, drawing (kind, w, h, thickness, pins), role, where to buy / photo
PARTS = [
    dict(comp="EEG_EMG_Front_End", model="TI_ADS1299IPAGR", price=5000, qty=1, rng="Rs 4,500-5,500 (US$ 55-60 at qty 1)",
         size=["Package: TQFP-64 (PAG), 0.5 mm pin pitch", "Body: 10.0 x 10.0 mm; with leads 12.0 x 12.0 mm", "Height: 1.2 mm max",
               "Pins: 64 (16 per side)"],
         draw=("qfp", 10, 10, 1.2, 16, 12), role="8-channel 24-bit EEG/EMG converter with built-in lead-off (electrode contact) detection. "
         "Channels: 4 EEG (T7-P7, T8-P8, P7-O1, P8-O2) or 6 (+P3-O1, P4-O2), 2 jaw EMG.",
         buy=["https://www.ti.com/product/ADS1299", "https://www.lcsc.com/search?q=ADS1299IPAGR", "https://www.mouser.in/c/?q=ADS1299IPAGR"]),
    dict(comp="EEG_EMG_Front_End", model="TI_ADS1299-6PAGR", price=3000, qty=0, rng="Rs 2,800-3,500 (US$ 32-40)  [alternative, 6 channels]",
         size=["Package: TQFP-64 (PAG), 0.5 mm pin pitch", "Body: 10.0 x 10.0 mm; with leads 12.0 x 12.0 mm", "Height: 1.2 mm max"],
         draw=("qfp", 10, 10, 1.2, 16, 12), role="Cheaper 6-channel version (4 EEG + 2 EMG). Same footprint and firmware as ADS1299.",
         buy=["https://www.ti.com/product/ADS1299-6", "https://www.lcsc.com/search?q=ADS1299-6PAGR"]),
    dict(comp="Microcontroller_Bluetooth", model="Raytac_MDBT50Q-1MV2_nRF52840", price=900, qty=1, rng="Rs 800-1,200 (US$ 9-13)",
         size=["Module: 10.0 x 15.0 x 2.2 mm (with shield)", "Chip inside: Nordic nRF52840 (Arm Cortex-M4F, 1 MB flash, 256 KB RAM, BLE 5)",
               "Antenna: on-module chip antenna; pre-certified module (FCC/CE/TELEC)"],
         draw=("module", 10, 15, 2.2, 0, 0), role="Runs the firmware, reads ADS1299 (SPI) + PPG/IMU (I2C), streams to the phone over Bluetooth LE.",
         buy=["https://www.raytac.com/product/ins.php?index_id=31", "https://www.nordicsemi.com/products/nrf52840", "https://www.mouser.in/c/?q=MDBT50Q-1MV2"]),
    dict(comp="Microcontroller_Bluetooth", model="Nordic_nRF5340_QKAA", price=550, qty=0, rng="Rs 500-700 (US$ 5-6)  [alternative bare chip; needs RF design + certification]",
         size=["Package: aQFN-94, 7.0 x 7.0 mm", "Height: 0.85 mm"], draw=("lga", 7, 7, .85, 0, 0),
         role="Dual-core option if more processing is needed on the device; bare chip, so antenna/RF layout and certification are on you.",
         buy=["https://www.nordicsemi.com/products/nrf5340", "https://www.lcsc.com/product-detail/C3015611.html"]),
    dict(comp="PPG_SpO2_Sensor", model="Maxim_MAX30101EFD", price=650, qty=1, rng="Rs 550-800 (US$ 6-8)",
         size=["Package: 14-pin OESIP (optical module)", "Size: 5.6 x 3.3 x 1.55 mm", "Includes red, IR and green LEDs + photodetector + AFE"],
         draw=("optical", 5.6, 3.3, 1.55, 0, 0), role="In-ear heart rate + SpO2, on a small flex PCB in the ear tip.",
         buy=["https://www.analog.com/en/products/max30101.html", "https://www.lcsc.com/search?q=MAX30101EFD"]),
    dict(comp="PPG_SpO2_Sensor", model="ADI_MAX86141ENP", price=500, qty=0, rng="Rs 450-600 (US$ 5-6)  [alternative, needs external LEDs + photodiode]",
         size=["Package: WLP-20, 2.0 x 2.0 mm (approx.)", "Needs separate LEDs + photodiode"], draw=("lga", 2.0, 2.0, .6, 0, 0),
         role="Higher-performance PPG front end for a later version (lower power, better motion handling).",
         buy=["https://www.analog.com/en/products/max86141.html", "https://www.lcsc.com/product-detail/C5328762.html"]),
    dict(comp="Motion_Sensor_IMU", model="Bosch_BMI270", price=280, qty=1, rng="Rs 250-400 (US$ 3-4)",
         size=["Package: LGA-14", "Size: 2.5 x 3.0 x 0.8 mm", "6-axis: accelerometer + gyroscope"],
         draw=("lga", 3.0, 2.5, .8, 0, 0), role="Head motion: rhythmic jerking of tonic-clonic seizures, and movement artefacts.",
         buy=["https://www.bosch-sensortec.com/products/motion-sensors/imus/bmi270/", "https://www.lcsc.com/search?q=BMI270"]),
    dict(comp="Temperature_Sensor", model="TI_TMP117AIDRVR", price=300, qty=1, rng="Rs 250-350 (US$ 3-4)  [optional]",
         size=["Package: WSON-6 (DRV)", "Size: 2.0 x 2.0 x 0.8 mm", "Accuracy: +/-0.1 C (-20 to 50 C)"],
         draw=("lga", 2.0, 2.0, .8, 0, 0), role="Ear-canal temperature (optional).",
         buy=["https://www.ti.com/product/TMP117", "https://www.lcsc.com/search?q=TMP117AIDRVR"]),
    dict(comp="Flash_Memory", model="Winbond_W25Q128JVSIQ", price=80, qty=1, rng="Rs 60-120",
         size=["Package: SOIC-8, 208 mil", "Body: 5.3 x 5.3 mm; with leads 5.3 x 8.0 mm", "Height: ~2.0 mm", "Capacity: 16 MB (128 Mbit)"],
         draw=("soic", 5.3, 5.3, 2.0, 4, 8.0), role="Backup storage of raw signals when the Bluetooth link drops.",
         buy=["https://www.winbond.com/hq/product/code-storage-flash-memory/serial-nor-flash/", "https://www.lcsc.com/search?q=W25Q128JVSIQ"]),
    dict(comp="Battery", model="LiPo_402030_250mAh", price=350, qty=1, rng="Rs 250-500",
         size=["Cell: 4.0 x 20 x 30 mm (thickness x width x length)", "3.7 V nominal, ~250 mAh, with protection circuit", "Weight: ~5 g"],
         draw=("battery", 20, 30, 4.0, 0, 0), role="Powers the device; about a day of streaming. Charged only when OFF the head.",
         buy=["https://robu.in/?s=402030", "https://www.evelta.com/search?q=lipo+402030"]),
    dict(comp="Battery_Charger", model="Microchip_MCP73831T-2ACI_OT", price=50, qty=1, rng="Rs 35-70",
         size=["Package: SOT-23-5", "Body: 2.9 x 1.6 mm; with leads 2.9 x 2.8 mm", "Height: 1.45 mm max"],
         draw=("sot", 2.9, 1.6, 1.45, 3, 2.8), role="Single-cell Li-ion/LiPo charger (charge-only port, device off the head).",
         buy=["https://www.microchip.com/en-us/product/MCP73831", "https://www.lcsc.com/search?q=MCP73831T-2ACI%2FOT"]),
    dict(comp="Voltage_Regulator_Digital", model="TI_TPS7A2033PDBVR", price=60, qty=1, rng="Rs 40-80",
         size=["Package: SOT-23-5 (DBV)", "Body: 2.9 x 1.6 mm; with leads 2.9 x 2.8 mm", "Height: 1.45 mm max", "Output: 3.3 V, 300 mA, very low noise"],
         draw=("sot", 2.9, 1.6, 1.45, 3, 2.8), role="3.3 V rail for the module, sensors and ADS1299 digital side.",
         buy=["https://www.ti.com/product/TPS7A20", "https://www.lcsc.com/search?q=TPS7A2033PDBVR"]),
    dict(comp="Boost_Converter_Analog", model="TI_TPS61023DRLR", price=60, qty=1, rng="Rs 40-80",
         size=["Package: SOT-563 (DRL)", "Size: 1.6 x 1.6 x 0.6 mm"], draw=("sot", 1.6, 1.2, .6, 3, 1.6),
         role="Boosts the battery to ~5.5 V, feeding the low-noise 5 V analog regulator for the ADS1299.",
         buy=["https://www.ti.com/product/TPS61023", "https://www.lcsc.com/search?q=TPS61023DRLR"]),
    dict(comp="Voltage_Regulator_Analog", model="TI_TPS7A2050PDBVR", price=60, qty=1, rng="Rs 40-80",
         size=["Package: SOT-23-5 (DBV)", "Body: 2.9 x 1.6 mm; with leads 2.9 x 2.8 mm", "Output: 5.0 V, very low noise"],
         draw=("sot", 2.9, 1.6, 1.45, 3, 2.8), role="Clean 5 V analog supply (AVDD) for the ADS1299: clean power = clean EEG.",
         buy=["https://www.ti.com/product/TPS7A20", "https://www.lcsc.com/search?q=TPS7A2050PDBVR"]),
    dict(comp="ESD_Protection", model="TI_TPD4E1U06DBVR", price=40, qty=3, rng="Rs 25-60 each",
         size=["Package: SOT-23-6 (DBV)", "Body: 2.9 x 1.6 mm; with leads 2.9 x 2.8 mm", "Protects 4 lines per chip"],
         draw=("sot", 2.9, 1.6, 1.45, 3, 2.8), role="ESD protection on every electrode line (with series resistors) - wearer and chip safety.",
         buy=["https://www.ti.com/product/TPD4E1U06", "https://www.lcsc.com/search?q=TPD4E1U06DBVR"]),
    dict(comp="Electrode_Dry_Comb", model="Dry_EEG_Comb_Electrode_AgAgCl", price=600, qty=4, rng="Rs 500-1,300 each (AliExpress ~US$ 6; Neurospec ~US$ 14)",
         size=["Base disc: ~12-15 mm diameter (approx.)", "Pins: ~8-12 pins, ~3-5 mm long (approx.)", "Snap connector on the back"],
         draw=("electrode_comb", 14, 14, 5, 0, 0), role="Through-hair contact at P7/P8 and O1/O2 (and P3/P4 on the 6-channel version).",
         buy=["https://shop.openbci.com/products/5-mm-spike-electrode-pack-of-30", "https://shop.neurospec.com/products/electrodes/dry-electrodes",
              "https://www.aliexpress.com/item/1005003443803910.html"]),
    dict(comp="Electrode_Ear_Flat", model="AgAgCl_Flat_Disc_10mm", price=250, qty=6, rng="Rs 150-400 each",
         size=["Disc: ~10 mm diameter, ~2 mm thick (approx.)", "Sintered Ag/AgCl or conductive silicone", "Lead wire or snap"],
         draw=("electrode_flat", 10, 10, 2, 0, 0), role="Ear contacts (T7/T8 area), mastoid reference + bias, and 2 jaw (temporalis) EMG contacts.",
         buy=["https://shop.neurospec.com/products/electrodes/dry-electrodes", "https://www.fri-fl-shop.com/collections/electrodes-eeg-electrodes"]),
    dict(comp="Connector", model="JST_SH_SM04B-SRSS-TB", price=30, qty=4, rng="Rs 20-50 each",
         size=["1.0 mm pitch, 4-pin, side entry", "Size: ~6.0 x 4.3 x 2.9 mm (approx.)"], draw=("connector", 6.0, 4.3, 2.9, 4, 0),
         role="Detachable electrode leads / ear-tip flex to the main board.",
         buy=["https://www.jst-mfg.com/product/index.php?series=231", "https://www.lcsc.com/search?q=SM04B-SRSS-TB"]),
    dict(comp="Cable", model="Shielded_32AWG_Micro_Coax", price=400, qty=1, rng="Rs 300-600 for ~2 m",
         size=["Outer diameter: ~0.5-0.8 mm", "32 AWG conductor + shield"], draw=("cable", 30, 4, .7, 0, 0),
         role="Low-noise leads from electrodes to the board.", buy=["https://www.mouser.in/c/?q=32%20awg%20micro%20coax"]),
    dict(comp="PCB_Main_Board", model="JLCPCB_4-layer_40x30mm", price=800, qty=1, rng="Rs 300-800 per board (part of a 5-board order, Rs 12-25k with assembly)",
         size=["Board: ~40 x 30 mm, 4 layers, 1.0-1.6 mm thick", "Factory assembly (ADS1299 is 0.5 mm pitch)"],
         draw=("board", 40, 30, 1.2, 0, 0), role="Main board: ADS1299, module, power, protection, connectors.",
         buy=["https://jlcpcb.com", "https://www.pcbway.com", "https://www.lioncircuits.com"]),
    dict(comp="PCB_Ear_Tip_Flex", model="JLCPCB_Flex_12x8mm", price=500, qty=1, rng="Rs 300-700 per board (5-board order Rs 3-6k)",
         size=["Flex PCB: ~12 x 8 mm, 0.1-0.2 mm thick (approx.)"], draw=("board", 12, 8, .15, 0, 0),
         role="Carries the MAX30101 (+ TMP117) inside the silicone ear tip.", buy=["https://jlcpcb.com/flex-pcb"]),
    dict(comp="Mechanical_Housing", model="3D_Printed_EarHook_and_Band", price=1800, qty=1, rng="Rs 1,000-2,500 per set (TPU/resin; several iterations)",
         size=["Ear-hooks: ~60 x 45 x 8 mm each (approx.)", "Back band: ~300 mm spring-steel or TPU", "Silicone ear tips: standard earbud sizes S/M/L"],
         draw=("hook", 45, 60, 8, 0, 0), role="The '?' ear-hooks, the band around the back of the head, battery/board pod.",
         buy=["https://jlc3dp.com", "local 3D-printing shops"]),
    dict(comp="Programmer_Debugger", model="SEGGER_J-Link_EDU_Mini", price=6000, qty=0, rng="Rs 5,000-7,000 (one-time tool)",
         size=["Size: ~33 x 20 x 10 mm (approx.)", "10-pin 1.27 mm Cortex debug cable"], draw=("module", 20, 33, 10, 0, 0),
         role="Flashes and debugs the nRF firmware (one-time tool, not per device).",
         buy=["https://www.segger.com/products/debug-probes/j-link/models/j-link-edu-mini/", "https://www.mouser.in/c/?q=j-link%20edu%20mini"]),
    dict(comp="Reference_Heart_Rate_Strap", model="Polar_H10", price=8000, qty=0, rng="Rs 7,000-9,000 (one-time validation tool)",
         size=["Sensor: ~65 x 34 x 10 mm (approx.)", "Chest strap, ECG-based heart rate, Bluetooth"], draw=("module", 65, 34, 10, 0, 0),
         role="Ground truth for validating the in-ear PPG heart rate during self-wear tests.",
         buy=["https://www.polar.com/en/sensors/h10-heart-rate-sensor"]),
]


def dim(ax, x0, y0, x1, y1, text, off=0.0, vertical=False):
    ax.annotate("", (x0, y0), (x1, y1), arrowprops=dict(arrowstyle="<->", color="#3fd8ff", lw=1))
    if vertical:
        ax.text(x0 + off, (y0 + y1) / 2, text, color="#cfefff", fontsize=9, rotation=90, va="center", ha="center")
    else:
        ax.text((x0 + x1) / 2, y0 + off, text, color="#cfefff", fontsize=9, ha="center", va="center")


def drawing(p, out):
    kind, w, h, t, pins, lead = p["draw"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4.2), gridspec_kw={"width_ratios": [1.6, 1]}, facecolor="#050b14")
    for a in (a1, a2):
        a.set_facecolor("#050b14")
        a.set_aspect("equal")
        a.axis("off")
    body, edge = "#1b3550", "#3fd8ff"
    # ---- top view
    if kind == "qfp":
        L = lead
        for i in range(pins):
            s = -w / 2 + (i + .5) * w / pins
            for (x, y, ww, hh) in ((s - .12, h / 2, .24, (L - h) / 2), (s - .12, -L / 2, .24, (L - h) / 2),
                                   (w / 2, s - .12, (L - w) / 2, .24), (-L / 2, s - .12, (L - w) / 2, .24)):
                a1.add_patch(Rectangle((x, y), ww, hh, color="#c9d6e2"))
        a1.add_patch(Rectangle((-w / 2, -h / 2), w, h, fc=body, ec=edge, lw=1.5))
        a1.add_patch(Circle((-w / 2 + 1, h / 2 - 1), .35, fc="#0b1a2a", ec=edge))
        ext = L
    elif kind in ("sot", "soic"):
        L = lead
        for i in range(pins):
            x = -w / 2 + (i + .5) * w / pins
            a1.add_patch(Rectangle((x - .15, h / 2), .3, (L - h) / 2, color="#c9d6e2"))
            a1.add_patch(Rectangle((x - .15, -L / 2), .3, (L - h) / 2, color="#c9d6e2"))
        a1.add_patch(Rectangle((-w / 2, -h / 2), w, h, fc=body, ec=edge, lw=1.5))
        a1.add_patch(Circle((-w / 2 + .35 * min(w, h), -h / 2 + .35 * min(w, h)), .12 * min(w, h), fc="#0b1a2a", ec=edge))
        ext = max(w, L)
    elif kind in ("lga", "optical"):
        a1.add_patch(Rectangle((-w / 2, -h / 2), w, h, fc=body, ec=edge, lw=1.5))
        if kind == "optical":
            for i, c in enumerate(("#ff3d5a", "#b0003a", "#3dff8a")):
                a1.add_patch(Circle((-w / 2 + .9 + i * .9, 0), .32, fc=c, ec="none", alpha=.9))
            a1.add_patch(Rectangle((w / 2 - 1.9, -.7), 1.4, 1.4, fc="#0b1a2a", ec=edge))
        a1.add_patch(Circle((-w / 2 + .25 * min(w, h), h / 2 - .25 * min(w, h)), .08 * min(w, h), fc="#3fd8ff"))
        ext = max(w, h)
    elif kind == "module":
        a1.add_patch(FancyBboxPatch((-w / 2, -h / 2), w, h, boxstyle="round,pad=0,rounding_size=%g" % (min(w, h) * .05), fc=body, ec=edge, lw=1.5))
        a1.add_patch(Rectangle((-w / 2 + w * .1, -h / 2 + h * .1), w * .8, h * .55, fc="#23466a", ec=edge, lw=.8))
        ext = max(w, h)
    elif kind == "battery":
        a1.add_patch(FancyBboxPatch((-w / 2, -h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc="#3a3f47", ec="#c9d6e2", lw=1.5))
        a1.plot([-2, -2], [h / 2, h / 2 + 5], color="#ff3d5a", lw=2)
        a1.plot([2, 2], [h / 2, h / 2 + 5], color="#111", lw=2)
        a1.text(0, 0, "LiPo 3.7 V\n250 mAh", color="#e6eef5", ha="center", va="center", fontsize=10)
        ext = h + 6
    elif kind in ("electrode_comb", "electrode_flat"):
        a1.add_patch(Circle((0, 0), w / 2, fc="#b9c3cc", ec="#e6eef5", lw=1.5))
        if kind == "electrode_comb":
            import math
            for r_, n in ((0, 1), (w * .2, 6), (w * .36, 10)):
                for i in range(n):
                    a = 2 * math.pi * i / n
                    a1.add_patch(Circle((r_ * math.cos(a), r_ * math.sin(a)), .6, fc="#6f7b86", ec="#e6eef5"))
        else:
            a1.add_patch(Circle((0, 0), w * .3, fc="#8c969f", ec="none"))
        ext = w
    elif kind == "connector":
        a1.add_patch(Rectangle((-w / 2, -h / 2), w, h, fc="#e8e0cc", ec="#c9b98f", lw=1.5))
        for i in range(pins):
            a1.add_patch(Rectangle((-w / 2 + .9 + i * 1.0, -h / 2 + .6), .4, h * .5, fc="#b58a2a"))
        ext = max(w, h)
    elif kind == "cable":
        a1.plot([-w / 2, w / 2], [0, 0], color="#c9d6e2", lw=4)
        a1.plot([-w / 2, w / 2], [0, 0], color="#ff9f3f", lw=1.2)
        ext = w
    elif kind == "board":
        a1.add_patch(FancyBboxPatch((-w / 2, -h / 2), w, h, boxstyle="round,pad=0,rounding_size=1", fc="#0e4d2c" if t > .5 else "#b5711c", ec=edge, lw=1.5))
        ext = max(w, h)
    elif kind == "hook":
        import numpy as np
        th = np.linspace(-0.3, 3.6, 200)
        a1.plot(18 * np.cos(th), 20 * np.sin(th) + 8, color="#3fd8ff", lw=6, solid_capstyle="round")
        a1.plot([18 * np.cos(3.6), 10], [20 * np.sin(3.6) + 8, -26], color="#3fd8ff", lw=6, solid_capstyle="round")
        a1.add_patch(Circle((12, -27), 3, fc="#ff3d5a"))
        ext = h
    m = ext * .35
    a1.set_xlim(-ext / 2 - m, ext / 2 + m)
    a1.set_ylim(-ext / 2 - m, ext / 2 + m)
    if kind not in ("hook", "cable"):
        dim(a1, -w / 2, -ext / 2 - m * .5, w / 2, -ext / 2 - m * .5, f"{w:g} mm", off=-m * .25)
        dim(a1, -ext / 2 - m * .5, -h / 2, -ext / 2 - m * .5, h / 2, f"{h:g} mm", off=-m * .3, vertical=True)
    a1.set_title("TOP VIEW", color="#6f98b5", fontsize=9, family="monospace")
    # ---- side view
    sw = max(w, 1)
    a2.add_patch(Rectangle((-sw / 2, 0), sw, t, fc=body if kind not in ("battery",) else "#3a3f47", ec=edge, lw=1.5))
    a2.set_xlim(-sw / 2 - sw * .4, sw / 2 + sw * .4)
    a2.set_ylim(-sw * .45, max(t, sw * .3) + sw * .35)
    dim(a2, sw / 2 + sw * .15, 0, sw / 2 + sw * .15, t, f"{t:g} mm", off=sw * .1, vertical=True)
    a2.set_title("SIDE VIEW (height)", color="#6f98b5", fontsize=9, family="monospace")
    fig.suptitle(f"{p['comp'].replace('_', ' ')}  |  {p['model']}", color="#e6f6ff", fontsize=12, family="monospace")
    fig.text(.5, .02, "Dimensioned drawing from datasheet sizes (not a photo) - see price file for product links",
             color="#6f98b5", fontsize=7.5, ha="center", family="monospace")
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    rows = []
    for p in PARTS:
        d = HERE / p["comp"] / p["model"]
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("INR_*.txt"):
            old.unlink()
        per = p["price"] * max(p["qty"], 1)
        (d / f"INR_{p['price']}.txt").write_text(
            f"{p['model']}  ({p['comp'].replace('_', ' ')})\n\n"
            f"Estimated price per unit : Rs {p['price']:,}\n"
            f"Price range              : {p['rng']}\n"
            f"Quantity per device      : {p['qty'] if p['qty'] else '0 (alternative or one-time tool)'}\n"
            f"Line total per device    : Rs {p['price'] * p['qty']:,}\n\n"
            f"Role: {p['role']}\n\n"
            "Where to buy / see a photo:\n" + "\n".join(f"  - {u}" for u in p["buy"]) +
            "\n\nEstimate as of Sep 2026 (about Rs 88 per US$, import margin included where relevant). Confirm before ordering.\n",
            encoding="utf-8")
        (d / "size.txt").write_text(f"{p['model']}\n\n" + "\n".join(p["size"]) +
                                    "\n\nSource: manufacturer datasheet / product page (values marked approx. vary by supplier).\n", encoding="utf-8")
        drawing(p, d / "image.png")
        rows.append((p["comp"], p["model"], p["price"], p["qty"], p["price"] * p["qty"]))
        print(f"{p['comp']:28s} {p['model']:36s} Rs {p['price']:>6,} x {p['qty']}")
    per_device = sum(r[4] for r in rows)
    tools = sum(r[2] for r in rows if r[3] == 0 and r[0] in ("Programmer_Debugger", "Reference_Heart_Rate_Strap"))
    lines = ["# Prototype 1 hardware (fully custom build)", "",
             "Each component has a folder; inside it, one folder per model with `INR_<price>.txt` (price estimate, quantity, where to buy), "
             "`image.png` (dimensioned drawing from datasheet sizes; not a photo) and `size.txt`. Regenerate with "
             "`uv run python hardware/generate_hardware_folders.py` (from `epilepsy_sim/`, or any Python with matplotlib).", "",
             "| Component | Model | Price / unit (Rs) | Qty per device | Line total (Rs) |", "|---|---|---|---|---|"]
    for c, m, pr, q, tot in rows:
        lines.append(f"| {c.replace('_', ' ')} | `{m}` | {pr:,} | {q if q else '-'} | {tot:,} |")
    lines += ["", f"**Parts per device (primary models, qty > 0): about Rs {per_device:,}.**",
              f"One-time tools (J-Link, Polar H10): about Rs {tools:,}. Alternatives (qty 0) are listed for comparison.",
              "PCB fabrication + assembly is priced per board here; the first 5-board order is about Rs 12-25k with assembly.", "",
              "Prices are estimates (Sep 2026, about Rs 88 per US$). Confirm with the distributor before ordering. "
              "Safety: battery-only while worn; never connect to a mains-powered charger or laptop while on the head."]
    (HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nparts per device: Rs {per_device:,}; tools: Rs {tools:,}")


# ================================================================== build levels 0-4
def P(comp, model, price, qty, rng, size, draw, role, buy, photo=None):
    return dict(comp=comp, model=model, price=price, qty=qty, rng=rng, size=size, draw=draw, role=role, buy=buy, photo=photo)


BY_MODEL = {p["model"]: p for p in PARTS}
ref = lambda model, qty=None: {**BY_MODEL[model], **({"qty": qty} if qty is not None else {})}
NPG_DIR = HERE / "_assets" / "npg_lite"  # official NPG Lite photos (Upside Down Labs open-source docs)

# common cheap modules (India: Robu.in / Evelta / Amazon.in; prices approx., Sep 2026)
ESP32S3 = P("Microcontroller_Bluetooth_DevBoard", "Espressif_ESP32-S3-DevKitC-1", 900, 1, "Rs 700-1,200 (clones from ~Rs 600)",
            ["Board: approx. 63 x 25.4 mm (varies by clone)", "Chip: ESP32-S3 (dual-core 240 MHz, Wi-Fi + Bluetooth LE 5)",
             "USB programming - no separate programmer needed"], ("module", 25.4, 63, 8, 0, 0),
            "Reads the sensors and streams to phone/laptop over Bluetooth or Wi-Fi; programmed over USB (Arduino / ESP-IDF).",
            ["https://www.espressif.com/en/products/devkits", "https://robu.in/?s=esp32-s3"])
MPU6050 = P("Motion_Sensor_IMU_Breakout", "GY-521_MPU6050_Module", 150, 1, "Rs 100-250",
            ["Module: approx. 21 x 16 x 3 mm", "6-axis accelerometer + gyroscope, I2C"], ("module", 21, 16, 3, 0, 0),
            "Cheapest head-motion sensor (convulsion rhythm, movement artefacts).", ["https://robu.in/?s=mpu6050"])
BMI270B = P("Motion_Sensor_IMU_Breakout", "BMI270_Breakout_Board", 550, 1, "Rs 400-900",
            ["Module: approx. 18 x 18 mm (varies)", "Bosch BMI270, I2C/SPI, low power"], ("module", 18, 18, 3, 0, 0),
            "Better, lower-power IMU than MPU6050.", ["https://robu.in/?s=bmi270", "https://www.adafruit.com/?q=bmi270"])
MAX30102 = P("PPG_SpO2_Breakout", "MAX30102_Module", 250, 1, "Rs 150-400",
             ["Module: approx. 14 x 14 mm (purple board) / 20 x 15 mm (varies)", "Red + IR LEDs, photodetector, I2C"], ("optical", 14, 14, 3, 0, 0),
             "Heart rate + SpO2 from an ear clip / earlobe (test-grade).", ["https://robu.in/?s=max30102"])
TP4056 = P("Battery_Charger_Module", "TP4056_USB-C_Module_with_protection", 40, 1, "Rs 25-60",
           ["Module: approx. 26 x 17 mm", "1 A LiPo charger + DW01 protection"], ("module", 26, 17, 4, 0, 0),
           "Charges the LiPo (only while the device is OFF the head).", ["https://robu.in/?s=tp4056"])
GEL = P("Electrode_Gel_Disposable", "ECG_Snap_Gel_Electrodes_pack_of_50", 450, 1, "Rs 300-600 per 50 (~Rs 6-12 each)",
        ["Pad: approx. 40-50 mm, snap stud 3.9-4 mm (varies)", "Foam/cloth, Ag/AgCl + wet gel, single use"], ("electrode_flat", 45, 45, 1.5, 0, 0),
        "Test electrodes: best signal, single use, not through hair. Gold standard to compare dry electrodes against.",
        ["https://robu.in/?s=ecg+electrode", "https://www.amazon.in/s?k=ecg+electrodes+snap"])
SNAP = P("Cable_Electrode_Snap", "Snap_Electrode_Cables_set", 350, 1, "Rs 200-500 per set",
         ["3.5 mm jack or Dupont to snap, ~1 m (varies)"], ("cable", 30, 4, .7, 0, 0),
         "Connects snap electrodes to the amplifier inputs.", ["https://robu.in/?s=electrode+cable"])
HEADBAND = P("Mechanical_Headband", "Elastic_Headband_and_Clips", 400, 1, "Rs 200-800",
             ["Adjustable elastic band, ~50-62 cm head size", "Velcro / 3D-printed clips for board + electrodes"], ("board", 60, 20, 3, 0, 0),
             "Holds the board and electrodes on the head during tests.", ["local sports shop / Amazon.in"])
WIRES = P("Wiring", "Dupont_Wires_Headers_Perfboard", 300, 1, "Rs 200-500",
          ["Jumper wires, 2.54 mm headers, small perfboard"], ("board", 50, 35, 1.6, 0, 0), "Prototype wiring.", ["https://robu.in"])
DRY4, FLAT = ref("Dry_EEG_Comb_Electrode_AgAgCl", 4), ref("AgAgCl_Flat_Disc_10mm", 6)
LIPO = ref("LiPo_402030_250mAh", 1)

LEVELS = {
    "Level_0_Buy_Kit": dict(
        title="Level 0 - Buy a kit (NPG Lite Beast Pack)", skills="None: plug in, add sensors, run the Chords software",
        quality="Medium: 12-bit, 6 channels, no electrode-contact check. Great for EMG/ECG; EEG must pass the noise and alpha tests.",
        parts=[P("Biosignal_Board_Kit", "Upside_Down_Labs_NPG_Lite_Beast_Pack", 21100, 1,
                 "US$ 240 (~Rs 21,100) on Crowd Supply incl. shipping; check the Indian store for INR price",
                 ["Board: Adafruit Feather, approx. 22.9 mm wide x 51-65 mm long, ~7-8 mm tall", "6 channels with VibZ+ Playmate",
                  "Pack: NPG Lite, VibZ+ Playmate, 13 snap cables, 8 alligator cables, 24 gel electrodes, battery connector, USB-C cable, 3D-printed case"],
                 None, "Ready-made 6-channel wireless biosignal recorder (ESP32-C6, 12-bit).",
                 ["https://www.crowdsupply.com/upside-down-labs/neuro-playground-lite", "https://store.upsidedownlabs.tech/"], photo=NPG_DIR),
               LIPO, BMI270B, MAX30102, DRY4, FLAT,
               ref("3D_Printed_EarHook_and_Band")]),
    "Level_1_Cheap_Modules": dict(
        title="Level 1 - Cheap ready-made modules on an ESP32", skills="Wiring on a breadboard/perfboard, Arduino code",
        quality="Low for EEG (ECG-grade amplifiers + the ESP32's own noisy ADC); fine for jaw EMG, ECG, motion and PPG experiments.",
        parts=[ESP32S3,
               P("Amplifier_Module_ECG_EMG", "AD8232_Heart_Monitor_Module", 350, 3, "Rs 250-450 each",
                 ["Module: approx. 36 x 28 mm (red clone) / 35 x 28 mm (SparkFun)", "Single-channel analog biopotential amp (designed for ECG)"],
                 ("module", 36, 28, 4, 0, 0), "One channel each: jaw EMG, ECG, 1 test EEG channel. Output read by the ESP32 ADC.",
                 ["https://robu.in/?s=ad8232", "https://www.sparkfun.com/products/12650"]),
               P("Amplifier_Module_EXG", "Upside_Down_Labs_BioAmp_EXG_Pill", 1200, 2, "Rs 900-1,600 each (check the maker's store)",
                 ["Module: approx. 25 x 10 mm ('pill', approx.)", "Single-channel EXG (EEG/ECG/EMG/EOG) analog front end"],
                 ("module", 10, 25, 3, 0, 0), "Better-suited single-channel EEG/EMG amplifier than AD8232 for the ear channels.",
                 ["https://store.upsidedownlabs.tech/", "https://github.com/upsidedownlabs/BioAmp-EXG-Pill"]),
               MPU6050, MAX30102, TP4056, LIPO, GEL, SNAP, WIRES, HEADBAND]),
    "Level_2_Open_Source_Board": dict(
        title="Level 2 - Open-source ADS1299 board, factory-assembled + ESP32", skills="Ordering from JLCPCB (upload open design files), wiring, code",
        quality="High: 24-bit ADS1299, 4 channels, built-in electrode-contact check. Best EEG per rupee.",
        parts=[ref("TI_ADS1299-6PAGR", 1) | {"model": "TI_ADS1299-4PAGR", "price": 3000, "rng": "Rs 2,800-3,500 (US$ 32-36)",
                                           "role": "4-channel 24-bit EEG front end (ear + back of head). Placed by the factory."},
               P("PCB_Fabrication", "JLCPCB_Open-Source_ADS1299_Board_5pcs", 2000, 1, "Rs 1,500-3,000 per 5-board order (1 used, 4 spares)",
                 ["Board: ~40-60 mm (depends on the chosen open design), 4 layers"], ("board", 50, 35, 1.6, 0, 0),
                 "Bare boards made from an open-source ADS1299 design (e.g. OpenBCI Cyton or HackEEG files; check each licence).",
                 ["https://jlcpcb.com", "https://github.com/OpenBCI", "https://github.com/starcat-io/hackeeg-shield"]),
               P("PCB_Assembly", "JLCPCB_SMT_Assembly_setup_and_parts", 4000, 1, "Rs 3,000-5,000 (setup + passives + placing the ADS1299)",
                 ["Factory places the 0.5 mm-pitch ADS1299 and small parts"], ("board", 50, 35, 1.6, 0, 0),
                 "Avoids hand-soldering the fine-pitch chip.", ["https://jlcpcb.com/smt-assembly"]),
               ESP32S3, BMI270B, MAX30102, TP4056, LIPO, GEL, SNAP, DRY4, WIRES, HEADBAND]),
    "Level_3_Fully_Custom_Factory": dict(
        title="Level 3 - Fully custom board (our design), factory-assembled", skills="PCB design (KiCad), firmware (nRF Connect / Zephyr), 3D printing",
        quality="High, in the real ear-hook form: 8 channels, contact check, in-ear PPG, head IMU, jaw EMG.",
        parts=[p for p in PARTS if p["qty"] > 0 and not p["comp"].startswith("PCB_")] + [
               P("PCB_Batch_Order", "JLCPCB_First_Batch_5_boards_with_assembly", 18000, 1, "Rs 12,000-25,000 one-time (5 main boards + 5 flex, assembled)",
                 ["5 main boards ~40 x 30 mm + 5 ear-tip flex boards"], ("board", 40, 30, 1.2, 0, 0),
                 "First production batch (one-time).", ["https://jlcpcb.com"]),
               ref("SEGGER_J-Link_EDU_Mini", 1)]),
    "Level_4_Hand_Soldered_Custom": dict(
        title="Level 4 - Fully custom board, hand-soldered by you", skills="Fine-pitch SMD soldering (0.5 mm TQFP, 2.5 mm LGA), hot air, inspection",
        quality="High if soldered well; risk of bad joints on the ADS1299 / IMU (inspect under magnification).",
        parts=[p for p in PARTS if p["qty"] > 0 and not p["comp"].startswith("PCB_")] + [
               P("PCB_Fabrication", "JLCPCB_Bare_Boards_5pcs_plus_Stencil", 3000, 1, "Rs 2,000-4,000 (5 main + 5 flex boards, steel stencil)",
                 ["Bare boards only, no assembly"], ("board", 40, 30, 1.2, 0, 0), "You place and solder every part.", ["https://jlcpcb.com"]),
               P("Tool_Soldering_Station", "Hot_Air_plus_Iron_Station_858D_class", 4000, 1, "Rs 3,000-6,000 (one-time)",
                 ["Bench unit approx. 200 x 150 x 110 mm"], ("module", 150, 110, 200, 0, 0), "Hot air for LGA/QFN parts; fine-tip iron + drag soldering for TQFP.",
                 ["https://robu.in/?s=858d", "https://www.amazon.in/s?k=858d+hot+air"]),
               P("Tool_Consumables", "Solder_Paste_Flux_Wick_Tweezers", 1500, 1, "Rs 1,000-2,000 (one-time)",
                 ["Low-temp solder paste, no-clean flux, wick, ESD tweezers"], ("board", 40, 30, 5, 0, 0), "Needed for SMD work.", ["https://robu.in"]),
               P("Tool_Magnification", "USB_Microscope_or_Loupe", 2000, 1, "Rs 800-5,000 (one-time)",
                 ["USB microscope ~1000x, or 10x loupe"], ("module", 30, 100, 30, 0, 0), "Inspect 0.5 mm-pitch joints for bridges.", ["https://robu.in/?s=usb+microscope"]),
               ref("SEGGER_J-Link_EDU_Mini", 1)]),
}


def write_part(p, base):
    import shutil
    d = base / p["comp"] / p["model"]
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("INR_*.txt"):
        old.unlink()
    (d / f"INR_{p['price']}.txt").write_text(
        f"{p['model']}  ({p['comp'].replace('_', ' ')})\n\n"
        f"Estimated price per unit : Rs {p['price']:,}\nPrice range              : {p['rng']}\n"
        f"Quantity                 : {p['qty']}\nLine total               : Rs {p['price'] * p['qty']:,}\n\n"
        f"Role: {p['role']}\n\nWhere to buy / see a photo:\n" + "\n".join(f"  - {u}" for u in p["buy"]) +
        "\n\nEstimate as of Sep 2026 (about Rs 88 per US$). Confirm before ordering.\n", encoding="utf-8")
    (d / "size.txt").write_text(f"{p['model']}\n\n" + "\n".join(p["size"]) +
                                "\n\nSource: manufacturer / seller listing (values marked approx. vary by supplier).\n", encoding="utf-8")
    if p.get("photo"):
        for f in Path(p["photo"]).glob("*.*"):
            if f.suffix.lower() in (".jpg", ".png") and f.name != "npg-lite-front-back.jpg":  # that one becomes image.jpg
                shutil.copy2(f, d / f.name)
        shutil.copy2(Path(p["photo"]) / "npg-lite-front-back.jpg", d / "image.jpg")
    elif p.get("draw"):
        drawing(p, d / "image.png")
    return p["price"] * p["qty"]


def build_levels():
    import shutil
    root = HERE / "Levels"
    if root.exists():
        shutil.rmtree(root)
    summary = ["# Prototype 1 build levels: how 'from scratch' and how costly", "",
               "Each level is a complete, buildable set. Inside each: `<Component>/<Model>/` with `INR_<price>.txt`, "
               "`image.png` (dimensioned drawing) or real photos, and `size.txt`, plus a README with the total.", "",
               "| Level | Skills | EEG quality | Total (Rs, one device incl. one-time costs) |", "|---|---|---|---|"]
    for name, L in LEVELS.items():
        base = root / name
        base.mkdir(parents=True)
        rows, total = [], 0
        for p in L["parts"]:
            line = write_part(p, base)
            total += line
            rows.append(f"| {p['comp'].replace('_', ' ')} | `{p['model']}` | {p['price']:,} | {p['qty']} | {line:,} |")
        (base / "README.md").write_text("\n".join([f"# {L['title']}", "", f"**Skills:** {L['skills']}", "",
                                                   f"**EEG quality:** {L['quality']}", "",
                                                   "| Component | Model | Price / unit (Rs) | Qty | Line total (Rs) |", "|---|---|---|---|---|", *rows, "",
                                                   f"**Total: about Rs {total:,}** (estimates, Sep 2026; confirm before ordering).", "",
                                                   "Safety: battery-only while worn; never connected to a mains charger or laptop on the head."]) + "\n",
                                        encoding="utf-8")
        summary.append(f"| [{L['title']}]({name}/README.md) | {L['skills']} | {L['quality']} | **~{total:,}** |")
        print(f"{name:32s} Rs {total:>7,}")
    summary += ["", "Level 3 per-device cost after the first batch: ~Rs 8,000-13,000 (parts only). "
                "Level 2 is the cheapest route to high-quality (24-bit) EEG. All prices are estimates; confirm before ordering."]
    (root / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
    build_levels()

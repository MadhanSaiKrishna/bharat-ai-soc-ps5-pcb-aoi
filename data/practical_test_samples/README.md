# Real-World Industrial PCB Inspection Test Samples

This directory contains **authentic, real-world optical photos of populated PCBs and SMT assemblies** extracted directly from the held-out test split of the [PCB Solder & Assembly Defect Detection Dataset](https://huggingface.co/datasets/ParsaVictor/pcb-solder-defect-detection) (Parsa Victor, 2024).

These are **not synthetic or simulated images**; they are physical macro-optical captures of real PCB defects from production assembly lines.

---

## 1. Catalog of Real Test Samples

| File | Defect Category | Defect Type | Resolution | What to Observe in the Image |
| :--- | :--- | :--- | :--- | :--- |
| **`01_real_solder_bridge.jpg`** | `SMT_SPI` | `SOLDER_BRIDGE` | 640 × 640 | High-contrast solder bridging and tin whiskering spanning across adjacent IC lead pins. |
| **`02_real_insufficient_solder.jpg`** | `SMT_SPI` | `INSUFFICIENT_SOLDER` | 480 × 480 | Starved solder paste deposit on SMT pads; thin, incomplete solder meniscus failing IPC Class 3 fillet standards. |
| **`03_real_excess_solder.jpg`** | `SMT_SPI` | `EXCESS_SOLDER` | 480 × 480 | Bulbous solder balling over SMT passive terminal with convex joint profile exceeding maximum allowable solder volume. |
| **`04_real_missing_component.jpg`** | `SMT_PLACEMENT` | `MISSING_COMPONENT` | 480 × 480 | Complete pick-and-place component dropout; bare PCB copper pads with unpopulated solder paste. |
| **`05_real_shifted_component.jpg`** | `SMT_PLACEMENT` | `SHIFTED_COMPONENT` | 480 × 480 | SMT passive resistor translated laterally and rotated off its nominal CAD landing pad (thermal/nozzle imbalance). |
| **`06_real_clean_pcb_pass.jpg`** | `PASS` | `ZERO_DEFECTS` | 1280 × 1280 | High-resolution golden PCB with clean SMT traces, ideal solder fillets, and fully intact components (Pass test case). |

---

## 2. How to Test These Real Samples in the Live Cockpit UI

1. **Launch the Operator Cockpit UI:**
   ```bash
   python3 module_b_guidance/app_ui.py --gerber data/sample_gerbers/CUSTOMER_PCB_JOB_REV1.zip
   ```

2. **Load a Real Defect Image:**
   - In the top header bar, click the **Optical Source** dropdown.
   - Select **`Load Optical Image from Disk...`**.
   - Browse to `data/practical_test_samples/` and select any of the images (e.g., `01_real_solder_bridge.jpg` or `04_real_missing_component.jpg`).

3. **Inspect the Board:**
   - Press the **`Spacebar`** (or click **`🚀 Inspect Current Board`**).
   - Watch the live pipeline execute:
     - **Canvas:** Highlights the real defect.
     - **Stage Gate:** Dynamically evaluates Stage 1 (SPI) or Stage 2 (Placement) and updates to `✗ DEFECT DETECTED` or `✓ PASS`.
     - **Copilot Advisory:** Generates instant root-cause DFM and SMT calibration recommendations.
     - **Yield Tab:** Updates real-time First-Pass Yield (FPY %) metrics.

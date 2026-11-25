# Real-World Industrial PCB Inspection Test Samples

This directory contains **authentic, real-world optical photos of populated PCBs and SMT assemblies** extracted from the held-out test split of the [PCB Solder & Assembly Defect Detection Dataset](https://huggingface.co/datasets/ParsaVictor/pcb-solder-defect-detection) (Parsa Victor, 2024).

These are physical macro-optical captures of real PCB defects from production assembly lines.

---

## 1. Catalog of Real Test Samples

| File | Defect Category | Defect Type | Resolution | What to Observe |
| :--- | :--- | :--- | :--- | :--- |
| **`01_real_solder_bridge.jpg`** | `SMT_SPI` | `SOLDER_BRIDGE` | 640 × 640 | High-contrast solder bridging spanning across adjacent IC lead pins. |
| **`02_real_insufficient_solder.jpg`** | `SMT_SPI` | `INSUFFICIENT_SOLDER` | 480 × 480 | Starved solder paste deposit on SMT pads; incomplete solder meniscus. |
| **`03_real_excess_solder.jpg`** | `SMT_SPI` | `EXCESS_SOLDER` | 480 × 480 | Bulbous excess solder over SMT passive terminal exceeding target volume. |
| **`04_real_missing_component.jpg`** | `SMT_PLACEMENT` | `MISSING_COMPONENT` | 480 × 480 | Pick-and-place component dropout; bare copper pads without component. |
| **`05_real_shifted_component.jpg`** | `SMT_PLACEMENT` | `SHIFTED_COMPONENT` | 480 × 480 | SMT passive translated/rotated off its nominal landing pad. |

---

## 2. How to Test in the Operator Cockpit UI

1. **Launch the Cockpit:**
   ```bash
   python3 module_b_guidance/app_ui.py
   ```
2. **Load an Image:** Click **`Load PCB Image`** and select any sample from `data/practical_test_samples/`.
3. **Run Inspection:** Click **`Inspect Frame (Space)`** (or press `Spacebar`):
   - **Viewport:** YOLOv8s-OBB draws oriented bounding boxes immediately (<1 s).
   - **Verdict & Table:** Updates the `REWORK REQUIRED` / `PASSED` pill and **Detected Defects** table.
   - **AI Copilot Advisories:** ExecuTorch Llama 3.2 1B generates DFM + Assembler advisories asynchronously in the background.
   - **Export Audit Report:** Generates and opens the PDF/CSV inspection report.

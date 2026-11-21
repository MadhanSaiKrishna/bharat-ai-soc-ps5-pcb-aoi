# Tier 0 (Laptop/Workstation) Baseline & Edge SoC Migration Guide

> **Project:** Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot  
> **Challenge:** Bharat AI-SoC PS5 Challenge 2026–27 (Arm / MeitY C2S / IIT Delhi)  
> **Target Production Hardware:** AMD Xilinx ZCU104 Evaluation Kit (Zynq UltraScale+ XCZU7EV MPSoC)  
> **Current Active Branch:** `feat/tier0-baseline`  

---

## 1. Executive Summary & Purpose of Tier 0

To accelerate development and enable all 3 team members to verify their modules simultaneously without requiring access to physical hardware (the AMD Xilinx ZCU104 development board, an industrial macro camera rig, or high-speed NVMe daughtercards), we established the **Tier 0 End-to-End Baseline**.

In **Tier 0**, the entire AOI inspection pipeline—optical acquisition, CAD registration, oriented bounding box (OBB) defect detection, real-time closed-loop LLM advisory, and telemetry logging—executes directly on a standard developer workstation or laptop (x86_64 CPU/GPU or Linux/WSL2). 

This document serves as the **official reference manual** detailing:
1. Every design accommodation made in Tier 0.
2. The exact target mapping for Tier 1, 2, and 3 on the ZCU104 edge board.
3. Code locations, configuration switches, and migration procedures required when porting to physical hardware.

---

## 2. Comprehensive Architectural Mapping (Tier 0 vs. Target Edge SoC)

| Subsystem / Layer | Tier 0 Baseline (Developer Laptop / Workstation) | Target Tier 2 / 3 (Edge AMD Xilinx ZCU104 SoC) |
| :--- | :--- | :--- |
| **Compute Hardware** | Intel/AMD x86_64 Host CPU (with optional AVX2/CUDA) | Heterogeneous MPSoC: Quad-core Arm Cortex-A53 (PS) + DPU B4096 IP Core (PL) |
| **Operating System** | Ubuntu 24.04 LTS (Native Linux or Windows WSL2 with WSLg) | Embedded PetaLinux 2023.2 or Ubuntu 22.04 LTS aarch64 on Arm PS |
| **Bulk Storage & Telemetry** | Local Laptop Filesystem / NVMe (`data/telemetry_logs/`, `data/raw_defect_dumps/`) | Dedicated PCIe Gen2x1/x2 M.2 NVMe SSD partition (`/mnt/nvme/aoi_buffer`) |
| **Camera & Image Feed** | Laptop USB 2.0/3.0 UVC Webcam or Optical Image Disk Loader (`/dev/video*`) | Industrial 5–8 MP UVC / MIPI-CSI Global Shutter Camera with telecentric macro lens & ring light |
| **CAD Reference Data** | Procedural 5000x4000 Synthetic Gerber Compositor (`gerber_renderer.py`) | Customer RS-274X Gerber bundle (`.gtp`, `.gtl`, `.gts`, `.gko`) submitted by EMS assembler |
| **Vision Inference (OBB)** | PyTorch / Ultralytics CPU runtime with synthetic OBB generator (`infer_vision.py`) | AMD Vitis AI DPU B4096 engine running quantized INT8 `.xmodel` on FPGA fabric |
| **On-Device LLM Copilot** | Meta Llama 3.2 1B-Instruct (4-bit GGUF via `llama-cpp-python` with AVX2 CPU acceleration) | Meta ExecuTorch runtime with Arm KleidiAI / XNNPACK INT4 kernels (`.pte` format) |
| **Operator Cockpit UI** | PyQt6 Desktop Application rendered on laptop screen (WSLg / X11) | Touchscreen HMI panel connected via ZCU104 DisplayPort (DP) or web/gRPC stream |
| **Active Learning & OOD** | MobileNetV3 feature extractor saving cropped anomalies to `data/operator_ingest_queue/` | On-chip edge vector cache (Faiss-CPU / SQLite) with automated batch export |

---

## 3. Subsystem Breakdown: Tier 0 Modifications & Edge Migration

### 3.1. Storage Subsystem: Local Laptop Filesystem vs. Dedicated NVMe Partition

#### Tier 0 Implementation:
- **Location:** `data/telemetry_logs/`, `data/raw_defect_dumps/`, and `data/operator_ingest_queue/`.
- **Mechanism:** Files are written directly to the host laptop's internal disk via standard POSIX Python I/O (`storage_manager.py`).
- **Memory Footprint:** Keeps a rolling 500-defect memory buffer in Python heap memory for instant LLM ingestion.
- **Why it was modified:** Eliminates the requirement for external hardware drives or physical mounting privileges during local algorithm development.

#### Edge Target (ZCU104):
- **Target Location:** `/mnt/nvme/aoi_buffer` (high-capacity PCIe M.2 NVMe SSD attached to FMC or PCIe slot).
- **Architecture Rationale:** A high-speed inspection line produces multi-megapixel uncompressed RAW frames at 30–60 FPS. Storing these directly into the ZCU104's 2GB DDR4 PS memory would choke AXI bus bandwidth and starve the Cortex-A53 CPU and DPU. The NVMe buffer decouples real-time inference from bulk audit persistence.
- **Porting Step:**
  ```python
  # edge_llm_storage/storage_manager.py
  # In Tier 0:
  STORAGE_ROOT = Path("data/telemetry_logs")
  # When porting to ZCU104 (Tier 2/3):
  STORAGE_ROOT = Path("/mnt/nvme/aoi_buffer")
  ```

---

### 3.2. Optical Image Acquisition: Laptop USB / Disk Fallback vs. Industrial Camera Rig

#### Tier 0 Implementation:
- **Location:** `module_b_guidance/app_ui.py` (`CameraWorker` thread) and `test_camera.py`.
- **Input Channels:**
  1. Standard USB UVC camera plugged into the laptop's USB port (`/dev/video0`, `/dev/video1`, etc.).
  2. In WSL2 environments (where USB devices are initially claimed by Windows), support for attaching via `usbipd-win`.
  3. Built-in **"Load Optical Image from Disk..."** fallback button, allowing developers and evaluators to feed synthetic or reference PCB high-res photos without any camera plugged in.
- **Why it was modified:** Ensures testing is never blocked if a developer is working remotely, in an office without a camera, or inside a virtualized WSL2 sandbox.

#### Edge Target (ZCU104):
- **Target Hardware:** Industrial 5–8 MP Global Shutter Camera with C-Mount telecentric lens (to eliminate perspective distortion on tall BGA/QFP packages) and diffuse LED ring light.
- **Interface:** UVC USB 3.0 via ZCU104 USB3 host port or MIPI-CSI2 mezzanine card connected to FPGA I/O pins.
- **Triggering:** Hardware optocoupled GPIO trigger synced with conveyor belt indexer (via Linux `/dev/uio*` or PetaLinux GPIO sysfs).
- **Porting Step:**
  - Configure V4L2 pipeline in PetaLinux:
    ```bash
    v4l2-ctl -d /dev/video0 --set-fmt-video=width=3840,height=2160,pixelformat=YUYV
    ```
  - Lock camera capture exposure and white balance to manual in `app_ui.py` to prevent lighting drift under shop-floor factory lights.

---

### 3.3. CAD Gerber Reference & Homography Alignment

#### Tier 0 Implementation:
- **Location:** `module_b_guidance/gerber_renderer.py` and `module_b_guidance/homography_align.py`.
- **Mechanism:**
  - `gerber_renderer.py` features a dual mode:
    1. Direct RS-274X Gerber layer loading (`.gtp` solder paste, `.gtl` top copper, `.gts` solder mask) using `pygerber`.
    2. High-precision 5000x4000 procedural synthetic PCB CAD generator that renders BGA ball arrays (0.65 mm pitch), 0201 SMT passives, wire bond dies, and 4 corner optical fiducials.
  - `homography_align.py` automatically detects fiducial count:
    - 3 fiducials: Uses affine transformation (`cv2.getAffineTransform`) embedded into a 3x3 planar matrix.
    - 4+ fiducials: Uses RANSAC homography (`cv2.findHomography(..., cv2.RANSAC)`).
- **Why it was modified:** Provides complete visual verification and CAD overlay matching before receiving proprietary customer production Gerbers.

#### Edge Target (ZCU104):
- **Target Deployment:** When an EMS assembler loads a new production job, the customer Gerber package (RS-274X zip) is uploaded once into `/mnt/nvme/aoi_buffer/active_cad/`.
- **Pre-rendering:** The high-resolution reference layers are pre-rendered into raster layers once upon job load, and pad coordinate matrices are indexed in memory. During live inspection, only homography transformation matrices `H` are computed per board.

---

### 3.4. Vision AI Defect Detection: Host CPU Baseline vs. FPGA DPU Acceleration

#### Tier 0 Implementation:
- **Location:** `module_a_vision/infer_vision.py`.
- **Mechanism:** Runs PyTorch / Ultralytics YOLOv8-OBB model on host CPU (or CUDA GPU if available). If no custom trained `.pt` weights are present, it falls back to an integrated synthetic defect injection engine that generates valid Oriented Bounding Boxes `[xc, yc, w, h, angle_deg]` conforming to `schemas/inspection_event.json`.
- **Cycle Time:** ~0.003s (synthetic baseline) or ~0.150s (PyTorch CPU).

#### Edge Target (ZCU104):
- **Target Deployment:** AMD Xilinx Deep Learning Processing Unit (DPU) IP core instantiated on the XCZU7EV Zynq UltraScale+ FPGA fabric (Programmable Logic / PL).
- **Architecture:** DPUCZDX8G B4096 architecture running at 300 MHz.
- **Model Format:** Quantized INT8 `.xmodel` compiled via AMD Vitis AI 3.0/3.5 Compiler (`vai_c_xir`).
- **Data Path:** Live camera frames are DMA-transferred directly to PL BRAM/URAM via AXI4-Stream, and DPU results are returned to Arm PS via shared memory with zero CPU copy overhead.
- **Target Vision Latency:** < 400 ms at full resolution.
- **Porting Step:**
  - Execute quantization and compilation scripts in `edge_llm_storage/vitis_ai_dpu/`:
    ```bash
    vai_q_pytorch --model yolov8_obb.pt --input_dir ./calibration_dataset
    vai_c_xir -x quantized_yolov8_obb.xmodel -a /opt/vitis_ai/arch/ZCU104/ZCU104.json -o dpu_yolov8_obb.xmodel -n dpu_yolov8_obb
    ```

---

### 3.5. On-Device LLM Copilot: GGUF CPU vs. ExecuTorch Arm KleidiAI

#### Tier 0 Implementation:
- **Location:** `edge_llm_storage/dual_advisor.py`.
- **Model:** Meta Llama 3.2 1B-Instruct in 4-bit quantization (`Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 771 MB).
- **Runtime:** `llama-cpp-python` with AVX2 CPU acceleration.
- **Word Guard:** Deterministic post-processing filter enforcing the competition rule: $\le 40$ words per recommendation branch.
- **Why it was modified:** Allows the team to immediately test prompt engineering, context window management, and live advisory UI feedback without needing Arm cross-compilers or specialized toolchains.

#### Edge Target (ZCU104):
- **Target Deployment:** Meta ExecuTorch runtime executing on the Quad-core Arm Cortex-A53 Processing System (PS).
- **Optimization:** Utilizes **Arm KleidiAI** micro-kernels (specifically optimized for Armv8-A architecture) and XNNPACK INT4 matrix multiplication kernels.
- **Model Format:** Exported `.pte` package (`llama3_2_1b_kleidiai_int4.pte`).
- **Target Generation Time:** < 3.0 s under 1100 MB RAM budget.
- **Export Script:** Already provided in `edge_llm_storage/export_llama_pte.py`.
- **Porting Step:**
  - On the ZCU104 target board:
    ```bash
    python3 edge_llm_storage/export_llama_pte.py --output /mnt/nvme/models/llama3_2_1b_kleidiai_int4.pte --quantize int4
    ```
  - In `edge_llm_storage/dual_advisor.py`, enable the ExecuTorch backend switch:
    ```python
    ADVISOR_BACKEND = "executorch"  # Switched from "gguf"
    ```

---

### 3.6. Operator Cockpit UI & Homography Display

#### Tier 0 Implementation:
- **Location:** `module_b_guidance/app_ui.py`.
- **Display:** PyQt6 window rendering on developer monitor via native X11 or WSLg.
- **Controls:**
  - Top CAD Overlay Opacity slider (0% to 100%).
  - Live inspection canvas with color-coded OBB defect boxes (Red: BGA/Solder, Amber: Placement, Blue: Wire Bond).
  - 4-Point BGA Gate checklist with automated pass/block status.
  - Dual LLM Advisory display (DFM Designer panel + Assembler Calibration panel).
  - Anomaly Ingestion tool for operator-driven 1-click cropping of out-of-distribution defects.

#### Edge Target (ZCU104):
- **Display:** 10.1" or 15.6" industrial touch HMI monitor connected to the ZCU104 DisplayPort (DP) interface.
- **Execution:** Runs under a lightweight X11/Wayland kiosk session in PetaLinux with touch gesture calibration.

---

## 4. Porting Checklist (From Tier 0 to Tier 1/2/3)

When moving code from this `feat/tier0-baseline` branch onto the physical ZCU104 evaluation platform, execute the following steps:

```markdown
- [ ] 1. Target Linux System Bring-up
      - Boot ZCU104 using official PetaLinux BSP or Ubuntu 22.04 LTS aarch64 image.
      - Configure PCIe M.2 NVMe carrier card; format and mount to `/mnt/nvme/aoi_buffer`.

- [ ] 2. Environment & Runtime Installation
      - Install Meta ExecuTorch runtime with Arm KleidiAI support:
        pip install executorch --extra-index-url https://download.pytorch.org/whl/executorch
      - Install Vitis AI runtime packages (`vitis-ai-runtime`, `xir`, `vart`).
      - Install UI dependencies: `pip install -r module_b_guidance/requirements.txt`.

- [ ] 3. Vision Model Migration (PL DPU)
      - Quantize trained YOLOv8-OBB PyTorch weights using Vitis AI Quantizer.
      - Compile to `dpu_yolov8_obb.xmodel` for DPUCZDX8G (B4096).
      - In `module_a_vision/infer_vision.py`, replace PyTorch inference call with `vart.Runner` DPU runner.

- [ ] 4. LLM Migration (PS ExecuTorch)
      - Run `python3 edge_llm_storage/export_llama_pte.py` to produce `llama3_2_1b_kleidiai_int4.pte`.
      - In `edge_llm_storage/dual_advisor.py`, set backend flag to `executorch`.

- [ ] 5. Camera Rig Integration
      - Connect industrial UVC / MIPI camera to ZCU104.
      - Verify `/dev/video0` output via `v4l2-ctl`.
      - Test acquisition with `python3 test_camera.py`.

- [ ] 6. Storage Configuration
      - Update telemetry output directories in `edge_llm_storage/storage_manager.py` to point to `/mnt/nvme/aoi_buffer/`.

- [ ] 7. End-to-End Benchmark Validation
      - Run `python3 benchmark_tier0.py 50` on the ZCU104 board.
      - Verify cycle time meets Tier 3 target: Total Cycle Time ≤ 3.5 s, Peak RAM < 1100 MB.
```

---

## 5. How to Run & Verify the Tier 0 Baseline Today

### 5.1. Run Full Regression & Contract Verification
```bash
# Run all unit tests (schema contract, vision inference, guidance UI modules)
python3 -m pytest tests/
```

### 5.2. Run the Tier 0 Performance Benchmark
```bash
# Evaluates cycle time, RAM usage, and 40-word advisory compliance over 15 iterations
python3 benchmark_tier0.py 15
```

### 5.3. Launch the Interactive Operator Cockpit UI
```bash
# Launches the 5-stage PyQt6 operator UI
python3 module_b_guidance/app_ui.py
```
*(If running on WSL2 without a camera attached, simply click the **"Load Optical Image from Disk..."** button to feed any synthetic or benchmark PCB image).*

### 5.4. Test Standalone LLM Closed-Loop Generation
```bash
python3 edge_llm_storage/dual_advisor.py --mock-input schemas/sample_inspection_event.json
```

---

## 6. Living Changelog: Implementations, Tradeoffs & Edge Migration Ledger

> [!IMPORTANT]
> **Documentation Rule:** Any code change, mock adaptation, or hardware fallback introduced on the `feat/tier0-baseline` branch **must** be logged in this ledger with its associated tradeoffs and edge porting requirements.

| Date / Commit | Subsystem / File | What Was Implemented for Tier 0 | Tradeoff / Laptop Accommodation | Edge ZCU104 Migration Task |
| :--- | :--- | :--- | :--- | :--- |
| **2026-11-21** (`bfd21de`) | `module_b_guidance/app_ui.py` | Added "Load Optical Image from Disk..." button and WSL2 camera diagnostic fallbacks | Avoids failure when no USB camera is plugged into the laptop or when WSL2 has not attached USB via `usbipd`. | In production, lock input to `/dev/video0` (V4L2) with hardware GPIO trigger sync; disable file picker in production kiosk mode. |
| **2026-11-21** (`bfd21de`) | `module_b_guidance/homography_align.py` | Dual-mode homography (3-point affine transform via `cv2.getAffineTransform` vs 4+ point RANSAC via `cv2.findHomography`) | Prevents crash (`npoints >= 4 expected`) when fewer than 4 optical fiducials are visible on test boards. | On the factory conveyor rig, ensure all 4 fiducials are within the optical rig FOV; enforce RANSAC with strict thresholding. |
| **2026-11-21** (`bfd21de`) | `module_b_guidance/gerber_renderer.py` | Procedural 5000x4000 synthetic PCB CAD generator (BGA, 0201 passives, wire bonds, fiducials) | Allows instant visual testing and CAD overlay alignment without waiting for proprietary customer Gerbers. | In production, replace procedural generator with standard customer RS-274X zip ingestion (`.gtp`, `.gtl`, `.gts`, `.gko`). |
| **2026-11-21** (`bfd21de`) | `edge_llm_storage/dual_advisor.py` | Real Meta Llama 3.2 1B-Instruct GGUF (Q4_K_M) inference using `llama-cpp-python` with AVX2 CPU acceleration | Executes on x86_64 CPU without requiring Arm cross-compilation or specialized hardware accelerators. | Port to ExecuTorch runtime using Arm KleidiAI / XNNPACK INT4 kernels (`.pte` format) executing on the Cortex-A53 PS. |
| **2026-11-21** (`bfd21de`) | `edge_llm_storage/storage_manager.py` | Telemetry logs and defect crops saved to local repository paths (`data/telemetry_logs/`, `data/raw_defect_dumps/`) | Uses standard laptop SSD storage without requiring physical partition mounting or root privileges. | Repoint `STORAGE_ROOT` to dedicated high-speed PCIe M.2 NVMe partition (`/mnt/nvme/aoi_buffer`) on the ZCU104 board. |
| **2026-11-21** (`bfd21de`) | `module_a_vision/infer_vision.py` | Synthetic defect fallback injecting standardized OBB defects when `.pt` weights are absent | Allows full end-to-end pipeline, contract schema, and UI validation before training custom YOLO weights. | Replace synthetic fallback with AMD Vitis AI DPU runtime (`vart.Runner`) loading quantized `dpu_yolov8_obb.xmodel`. |
| **2026-11-21** (`bfd21de`) | `benchmark_tier0.py` | Master evaluation suite measuring cycle time, RAM consumption, and 40-word advisory compliance | Runs on developer laptop to establish performance baselines and verify contract compliance. | Run identically on ZCU104 PetaLinux to verify compliance against Tier 3 targets (Cycle Time $\le 3.5\text{ s}$, RAM $< 1100\text{ MB}$). |


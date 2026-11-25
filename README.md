# Bharat AI-SoC PS5 — Phase 1 Baseline & Competition Roadmap (`feat/tier0-baseline`)

> **Branch Scope:** This `README.md` documents the **Phase 1 / Tier 0 working baseline** on `feat/tier0-baseline` and the **roadmap for Phases 2 and 3**. When merging into `main`, retain `main`'s root `README.md` if preferred.

---

## 1. What Is Implemented in This Phase 1 Baseline

This branch delivers a **100% offline, end-to-end PC/Tier-0 prototype** of the Closed-Loop Edge PCB Automated Optical Inspection (AOI) & Quality-Control Copilot:

1. **Real-Time Oriented Defect Detection (`module_a_vision/`):**
   - **Model:** 12-class **YOLOv8s-OBB** (`smt_merged_v2_v8s.pt`, 23 MB; **0.893 mAP@0.5** held-out HF test) + exported ExecuTorch XNNPACK `.pte` (`smt_merged_v2_v8s_xnnpack.pte`, 44 MB).
   - **Classes:** `INSUFFICIENT_SOLDER`, `EXCESS_SOLDER`, `SOLDER_BRIDGE`, `SOLDER_VOID`, `SOLDER_SPIKE`, `MISSING_COMPONENT`, `SHIFTED_COMPONENT`, `BURN`, `PHYSICAL_DAMAGE`, `OXIDATION`, `TOMBSTONE`, `ORIENTATION_ERROR`.
2. **On-Device Dual AI Copilot (`edge_llm_storage/`):**
   - **Runtime:** **Meta ExecuTorch 1.6.0** (`TextLLMRunner`, XNNPACK `8da4w` INT4 quantized `model.pte`, 915 MB, ~1.3–1.76 GiB RSS, ~6–7 tok/s decode) running **Meta Llama 3.2 1B-Instruct** locally (with `llama-cpp-python` GGUF fallback).
   - **Dual Advisories (<40 words each):**
     - **Branch 1 (CAD / DFM):** Layout/pad-clearance root-cause advice for PCB designers.
     - **Branch 2 (Assembler):** Machine/process calibration action for SMT operators.
   - **Storage (`storage_manager.py`):** Logs schema-validated JSONL telemetry (`schemas/inspection_event.json`).
3. **Minimal Responsive Operator Cockpit (`module_b_guidance/app_ui.py`):**
   - Clean, non-blocking PyQt6 UI (`Load PCB Image`, `Load Gerber (.zip)`, `Start Live Camera`, `Inspect Frame (Space)`).
   - Renders YOLOv8-OBB rotated bounding boxes immediately (<1 s) while Llama 3.2 1B runs asynchronously in `AdvisorWorker`.
   - Clickable top `status_banner` for live progress and 1-click navigation to **AI Copilot Advisories**, plus **Export Audit Report (PDF / CSV)** (`report_generator.py`).

---

## 2. Quickstart & Verification

```bash
source venv/bin/activate
pip install -r requirements.txt

# 1. Verify all 12 contract, vision, and UI integration tests
python3 -m pytest tests/

# 2. Launch the Operator Cockpit UI
python3 module_b_guidance/app_ui.py
```

**Model Weights (GitHub Release `v1.0.0-phase1`):**
- `model.pte` & `tokenizer.model` → `edge_llm_storage/weights/`
- `smt_merged_v2_v8s.pt` & `smt_merged_v2_v8s_xnnpack.pte` → `module_a_vision/weights/`

---

## 3. Repository Structure & File Map

| Path | Role / What It Does |
| :--- | :--- |
| `README.md` | Branch documentation for Phase 1 baseline, file map, and Phase 2/3 roadmap |
| `AGENTS.md` | Multi-developer collaboration protocol, folder ownership, and schema lock rules |
| `requirements.txt` | Top-level Python dependencies |
| `.github/workflows/ci.yml` | Automated CI workflow running `pytest` |
| `configs/prompts.yaml` | Dual-branch (DFM & Assembler) prompt templates for Llama 3.2 1B |
| `configs/tolerances.yaml` | Alignment & placement tolerance thresholds |
| `schemas/inspection_event.json` | Single-source-of-truth JSON Schema contract across all 3 modules |
| `schemas/sample_inspection_event.json` | Sample schema-compliant inspection payload for contract tests |
| `data/practical_test_samples/` | Real populated PCB defect images (`01`–`05`) used for demo and UI tests |
| `docs/TIER0_BASELINE_MIGRATION_GUIDE.md` | Tier-0 PC to ZCU104 / Arm SoC migration reference |
| `edge_llm_storage/dual_advisor.py` | ExecuTorch 1.6.0 `TextLLMRunner` + GGUF fallback generating DFM & Assembler advisories |
| `edge_llm_storage/export_llama_pte.py` | `optimum-executorch` XNNPACK `8da4w` INT4 export script for Llama 3.2 1B |
| `edge_llm_storage/storage_manager.py` | Local JSONL telemetry & raw defect crop logger |
| `module_a_vision/infer_vision.py` | `VisionInferenceEngine`: loads YOLOv8s-OBB and emits schema-compliant defects |
| `module_a_vision/train_obb.py` | Ultralytics YOLOv8-OBB fine-tuning wrapper |
| `module_a_vision/download_weights.py` | Fetches published YOLOv8-OBB weights from GitHub Releases |
| `module_a_vision/embedding_extractor.py` | MobileNetV3 feature extractor for Phase 2 active learning |
| `module_a_vision/datasets/` | `convert_soldef.py` & `build_smt_merged.py` dataset preparation scripts |
| `module_a_vision/README.md` | Detailed vision training, dataset fusion, class taxonomy, and evaluation notes |
| `module_a_vision/HANDOFF_MEMBER2.md` | Vision-to-LLM class/severity handoff specification |
| `module_b_guidance/app_ui.py` | Minimal, responsive PyQt6 Operator Cockpit (`InspectionCanvas`, `CameraWorker`, `AdvisorWorker`) |
| `module_b_guidance/gerber_renderer.py` | PyGerber RS-274X Gerber layer parser |
| `module_b_guidance/homography_align.py` | Fiducial homography alignment engine (`ΔX`, `ΔY`, `Δθ`) *(Phase 2 UI)* |
| `module_b_guidance/operator_ingest_tool.py` | Operator OOD anomaly crop & ingest tool *(Phase 2 UI)* |
| `module_b_guidance/report_generator.py` | One-click PDF & CSV compliance audit report exporter |
| `module_b_guidance/requirements.txt` | Module B / UI-specific dependencies |
| `tests/` | `test_contract.py`, `test_module_b_guidance.py`, `test_vision_inference.py` (12 tests) |
| `tools/stream_camera_windows.py` | Low-latency Windows-to-WSL2 HTTP camera bridge |
| `tools/test_e2e_pipeline.py` | Headless CLI end-to-end pipeline verification script |

---

## 4. Planned Features for Next Competition Phases

| Feature / Subsystem | Phase 1 Status | Phase 2 Plan | Phase 3 Plan |
| :--- | :--- | :--- | :--- |
| **YOLOv8s-OBB 12-Class Vision** | Implemented (`0.893 mAP@0.5`, `.pt` + `.pte`) | Add wire-bond & SPI paste classes + See3CAM macro fine-tuning (`vision-v3`) | Compile to INT8 `.xmodel` on ZCU104 Vitis AI DPU (`< 400 ms`) |
| **ExecuTorch Llama 3.2 1B Copilot** | Implemented (`model.pte` 8da4w INT4 on x86_64) | Tune domain prompts & rolling 500-defect trend memory | Deploy on Arm Cortex-A53 with KleidiAI INT4 kernels (`< 3 s`) |
| **Gerber CAD & Homography** | Backend parsed (`gerber_renderer.py`, `homography_align.py`); `Load Gerber (.zip)` in UI | Surface live CAD overlay + fiducial `ΔX/ΔY/Δθ` tolerance gating in UI | Hardware-calibrated optical rig alignment |
| **5-Stage Gate & BGA Checklist** | Schema & tolerance configs ready | Integrate 5-stage pass/rework checklist into Cockpit | Automated conveyor pass/reject diverter |
| **Active Learning OOD Crop** | Backend ready (`operator_ingest_tool.py`, `embedding_extractor.py`) | Enable 1-click anomaly crop & edge vector indexing in UI | Closed-loop offline incremental retraining |

### Key Porting Details & Architectural Tradeoffs (Phase 2 & Phase 3)

1. **Vision Precision vs. DPU Operator Support (`YOLOv8s-OBB`):**
   - *Phase 1:* Runs FP32 `.pt` (`0.893 mAP@0.5`, ~228 ms warm CPU) and exported XNNPACK `.pte` (44 MB).
   - *Phase 2/3 Tradeoff:* Lowering YOLOv8s-OBB to INT8 `.xmodel` on the ZCU104 Vitis AI DPU targets `< 400 ms` latency, but rotated-box (`OBB` angle) decoding and NMS are kept on the Arm PS CPU to avoid custom DPU op failures, and INT8 PTQ requires calibration on macro See3CAM frames to preserve `SOLDEF` / `ORIENTATION_ERROR` recall.
2. **LLM Footprint vs. PS DRAM Budget (`Llama 3.2 1B`):**
   - *Phase 1:* Quantized via `optimum-executorch` (`--dtype float32 --qlinear 8da4w --qembedding 8w --max_seq_len 512`) into a 915 MB `model.pte` (~1.33–1.76 GiB RSS, ~6–7 tok/s decode).
   - *Phase 2/3 Tradeoff:* Capping `max_seq_len=512` and enforcing `< 40 words` per advisory keeps memory under the ZCU104 Cortex-A53 `< 1.8 GB` PS limit and avoids KV-cache bloat; running `AdvisorWorker` asynchronously prevents the 8–10 s LLM step from blocking frame acquisition.
3. **Optical-to-CAD Homography vs. Uncalibrated Inputs:**
   - *Phase 1:* `homography_align.py` and `operator_ingest_tool.py` are implemented and unit-tested in `module_b_guidance/`, while the Cockpit UI stays minimal for arbitrary optical photos.
   - *Phase 2/3 Tradeoff:* Live CAD overlay + fiducial `ΔX/ΔY/Δθ` gating is deferred to Phase 2 when a fixed-mount optical rig (`See3CAM_CU30`) provides consistent scale and lighting, avoiding false alignment failures on varied single-board test photos.

---

## 5. Model, Dataset, Training & Hardware Credits

- **Meta Llama 3.2 1B-Instruct** (`unsloth/Llama-3.2-1B-Instruct`, `bartowski/Llama-3.2-1B-Instruct-GGUF`) — Meta Llama 3.2 Community License; Dubey et al., arXiv:2407.21783 (2024).
- **Meta ExecuTorch 1.6.0 & `optimum-executorch`** (XNNPACK `8da4w` quantization) + **Arm KleidiAI**.
- **Ultralytics YOLOv8s-OBB** (AGPL-3.0).
- **Datasets:** `ParsaVictor/pcb-solder-defect-detection` (Hugging Face, 5,209 images), `SolDef_AI` (Kaggle, GPL-3.0, 200 joint-level macro images), `SMD Component Placement Defect 2` (Roboflow Universe, CC BY 4.0, 2,750 originals). Details in [`module_a_vision/README.md`](module_a_vision/README.md).
- **Libraries & Hardware:** PyGerber, PyQt6, OpenCV, Pillow, ReportLab, AMD Xilinx ZCU104, e-con Systems See3CAM_CU30.

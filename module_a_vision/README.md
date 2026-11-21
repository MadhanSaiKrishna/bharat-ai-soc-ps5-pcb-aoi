# Module A — Vision AI (Member 1)

YOLOv8-OBB defect detection for the PS5 AOI pipeline. This document records
what was built, the data decisions behind it, model specs, results and how to
reproduce them.

**Status (2026-11-22):** `vision-v1` released. SMT solder & assembly model
meets the Tier 0 accuracy target on held-out data (**0.882 test mAP@0.5**, target ≥ 0.88).
Wire-bond, SPI-paste and tombstone detection are not trained yet.

---

## 1. Quick start

```bash
# 1. fetch trained weights (not in git; published as a GitHub release)
python -m module_a_vision.download_weights          # -> module_a_vision/weights/smt_merged_v8s.pt

# 2. use from code
python - <<'EOF'
import cv2
from module_a_vision.infer_vision import VisionInferenceEngine
engine = VisionInferenceEngine()                     # auto-loads weights/smt_merged_v8s.pt
defects, latency_ms = engine.detect_defects(cv2.imread("board.jpg"))
event = engine.build_complete_event("BOARD_001", "CHECKPOINT_2_PLACEMENT",
                                    {"dx_mm": 0, "dy_mm": 0, "dtheta_deg": 0}, defects)
EOF
```

- If the weights file is missing (or `model_weights=None` is passed), the engine
  falls back to the Tier-0 **analytical stub**, which returns fixed placeholder
  defects and does not look at the image. Tests use this mode deliberately.
- Output defects conform to `schemas/inspection_event.json` (verified with
  `jsonschema` on real model output).

## 2. Files

| File | Purpose |
|---|---|
| `infer_vision.py` | `VisionInferenceEngine`: loads weights, runs OBB inference, maps classes → schema category/severity, builds the contract event |
| `train_obb.py` | Ultralytics fine-tune wrapper; copies best checkpoint to `weights/<name>.pt` and prints val metrics |
| `download_weights.py` | Downloads release assets into `weights/` |
| `datasets/convert_soldef.py` | SolDef_AI LabelMe polygons → YOLO-OBB (joint-level subset) |
| `datasets/build_smt_merged.py` | Merges HF solder dataset + SolDef into one 10-class OBB dataset |
| `embedding_extractor.py` | MobileNetV3 embedding for operator anomaly ingestion (projection head still untrained) |
| `weights/` | Local weights (git-ignored) |

## 3. Datasets

Raw and converted datasets live **outside the repo** (default
`../datasets/`), because of size. Images in converted sets are symlinks.

| Dataset | Source | Licence | Used? | Notes |
|---|---|---|---|---|
| **PCB Solder & Assembly Defect** | [HF: ParsaVictor/pcb-solder-defect-detection](https://huggingface.co/datasets/ParsaVictor/pcb-solder-defect-detection) | Research / educational use; contact author for commercial use or redistribution | ✅ main source | 5,209 real populated-PCB photos, 9 classes, axis-aligned YOLO boxes. ~70 % of its val/test images are dark background-only tiles from one source. |
| **SolDef_AI** | [Kaggle: mauriziocalabrese/soldef-ai-…](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) | GPL-3.0 | ✅ joint subset | 428 labelled macro photos (2560×1440) of 0805/1206 resistors, LabelMe polygons. Visually closest to our See3CAM rig. |
| PCB-AoI | [Kaggle: kubeedgeianvs/pcb-aoi](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi) | Apache-2.0 | ❌ downloaded, not used | 173 train + 60 test IC-lead images under RGB tri-colour AOI lighting (domain mismatch). Classes: bad fillet slope, lifted lead. Candidate for a separate lead-inspection model. |
| SMD Component Placement Defect 2 | [Roboflow Universe](https://universe.roboflow.com/muhammad-huda-ferdinansyah/smd-component-placement-defect-2) | CC BY 4.0 | ⏳ next | 2,750 images: Normal, Misalignment, Orientation Error, Tombstoning. |
| Wire-bond OBB (Wong et al., BEEI) | [paper](https://beei.org/index.php/EEI/article/view/11059) | On request | ⏳ to request | 1,063 images, native YOLOv8-OBB labels. |

### Data decisions

1. **SolDef_AI has two labelling protocols.** Images from 2022-03-29 label the
   *whole resistor* (`good` / `no_good`, 1 box per image); 2022-03-30 and
   2022-04-08 label *each solder joint* (`good`, `exc_solder`, `poor_solder`,
   `spike`). Mixing them gives missing-label conflicts, so only the 200
   joint-level images are used. The component-level images are reserved for a
   possible good/bad crop classifier.
2. **Burst-grouped split.** SolDef shots taken seconds apart show the same part,
   so the split groups shots with < 60 s gaps and splits by group (per day).
   Seed 15 chosen because it gives ~25 % of every class in val.
3. **`GOOD_JOINT` dropped in the merged model.** The HF source does not label
   good joints, so keeping the class would teach contradictory background. AOI
   only needs defects reported.
4. **Axis-aligned → OBB.** HF boxes are converted to 4-corner OBBs with angle 0.
   SolDef polygons use `cv2.minAreaRect` (true rotated boxes).
5. **HF splits kept as published**; its test split (686 images) is the held-out
   score and was never used for training or model selection.

## 4. Class taxonomy & schema mapping

The schema locks the `category` enum only; `type` is free text. Mapping lives in
`infer_vision.py` (`CLASS_CATEGORY`, `CLASS_SEVERITY`, `NORMAL_CLASSES`).

| id | Class (`type`) | Category | Severity |
|---|---|---|---|
| 0 | INSUFFICIENT_SOLDER | SMT_SPI | HIGH |
| 1 | EXCESS_SOLDER | SMT_SPI | MEDIUM |
| 2 | SOLDER_BRIDGE | SMT_SPI | CRITICAL |
| 3 | SOLDER_VOID | SMT_SPI | MEDIUM |
| 4 | MISSING_COMPONENT | SMT_PLACEMENT | CRITICAL |
| 5 | SHIFTED_COMPONENT | SMT_PLACEMENT | HIGH |
| 6 | BURN | SMT_PLACEMENT | CRITICAL |
| 7 | PHYSICAL_DAMAGE | SMT_PLACEMENT | HIGH |
| 8 | OXIDATION | SMT_PLACEMENT | LOW |
| 9 | SOLDER_SPIKE | SMT_SPI | MEDIUM |

Also mapped for future models: `WIRE_SWEEP_SHORT`, `LIFTED_HEEL`, `LOOP_SAG`
(WIRE_BOND); `STARVED_PASTE` (SMT_SPI); `SKEWED_PASSIVE`, `TOMBSTONE`
(SMT_PLACEMENT). `GOOD_JOINT` and `WIRE_NORMAL` are never emitted as defects.
Unknown classes default to `SMT_PLACEMENT` / `HIGH`.

## 5. Model specs — `smt_merged_v8s.pt` (release `vision-v1`)

| Item | Value |
|---|---|
| Architecture | YOLOv8s-OBB (Ultralytics 8.4.216), init from DOTA-pretrained `yolov8s-obb.pt` |
| Input size | 832 px |
| Classes | 10 (table above) |
| Training data | 3,980 train / 743 val / 686 test images (HF + SolDef joint subset) |
| Epochs | 100 (best checkpoint: epoch 84, by Ultralytics fitness = 0.1·mAP50 + 0.9·mAP50-95) |
| Optimiser | auto → AdamW, lr 0.000714, cosine LR, batch 8, seed 0 |
| Augmentation | mosaic 1.0 (off for last 20 epochs), rotation ±15°, flip LR/UD 0.5, HSV-V 0.4 |
| Hardware / time | RTX 4050 Laptop 6 GB, 2.39 h |
| File size | 23.2 MB (FP32 `.pt`) |

### Results

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| val (743) | 0.895 | 0.727 | – | – |
| **test (686, held out)** | **0.882** | **0.748** | 0.811 | 0.884 |

Per-class test AP@0.5:

| Class | AP@0.5 |
|---|---|
| SHIFTED_COMPONENT | 0.995 * |
| PHYSICAL_DAMAGE | 0.959 |
| MISSING_COMPONENT | 0.944 |
| EXCESS_SOLDER | 0.915 |
| SOLDER_BRIDGE | 0.902 |
| INSUFFICIENT_SOLDER | 0.891 |
| SOLDER_VOID | 0.770 |
| BURN | 0.684 |

\* Only 31 test instances; possibly near-duplicates across the HF splits, so treat as optimistic.
SOLDER_SPIKE and OXIDATION have no test instances.

Inference latency (PyTorch FP32, RTX 4050, 832 px): **13.6 ms/img** model
(+0.9 ms pre, 0.6 ms post); ~12 ms end-to-end via `detect_defects()`.

### Baseline — `soldef_joint_v8s.pt`
YOLOv8s-OBB, 1024 px, SolDef joint subset only (144 train / 56 val), 4 classes
incl. `GOOD_JOINT`. Val mAP@0.5 **0.752** (excess 0.89, spike 0.85,
insufficient 0.70, good 0.57). Kept for reference; superseded by the merged model.

## 6. Reproduce

Environment used: Python 3.14, torch 2.15 + CUDA 13.2, ultralytics 8.4.216,
opencv 5.1, kaggle CLI 2.2.5 (`pip install -r requirements.txt kaggle`).

```bash
D=../datasets                                         # outside the repo

# download
kaggle auth login
kaggle datasets download -d mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection -p $D/soldef_ai --unzip
mkdir -p $D/pcb_solder_hf && cd $D/pcb_solder_hf && \
  curl -LO https://huggingface.co/datasets/ParsaVictor/pcb-solder-defect-detection/resolve/main/images_and_labels.zip && \
  unzip -q images_and_labels.zip && cd -

# convert + merge
python -m module_a_vision.datasets.convert_soldef   --src $D/soldef_ai/SolDef_AI/Labeled --dst $D/yolo/soldef_joint_obb
python -m module_a_vision.datasets.build_smt_merged --hf $D/pcb_solder_hf/final_solder \
       --soldef $D/yolo/soldef_joint_obb --dst $D/yolo/smt_merged_obb

# train (~2.4 h on RTX 4050) and evaluate on held-out test split
python -m module_a_vision.train_obb --data $D/yolo/smt_merged_obb/data.yaml \
       --name smt_merged_v8s --imgsz 832 --batch 8 --epochs 100 --patience 25
yolo obb val model=module_a_vision/weights/smt_merged_v8s.pt \
       data=$D/yolo/smt_merged_obb/data.yaml split=test imgsz=832
```

Run outputs (curves, confusion matrices, sample predictions) go to `../runs/`.

## 7. Known limitations

- **Not validated on our rig.** All scores are on public photos; See3CAM frames
  have not been labelled or tested yet.
- **No tombstone, skew, SPI-paste or wire-bond training data yet.** Those
  classes come only from the analytical stub / synthetic generator.
- **Weak classes:** BURN (0.68) and SOLDER_VOID (0.77).
- **Precision 0.81** → roughly 1 in 5 alarms is a false call at the default
  0.50 confidence threshold; threshold not yet tuned.
- `component_ref` is a placeholder (`COMP_LOC_x_y`); mapping detections to
  Gerber reference designators needs the homography from Module B.
- Not yet exported to ONNX / INT8, and not checked on the ZCU104 DPU.

## 8. Roadmap

1. Add Roboflow SMD placement data (tombstone, misalignment, orientation).
2. Request Wong et al. wire-bond dataset; extend `synth_defects.py` to all classes and write YOLO-OBB output.
3. Label 100–300 See3CAM frames and fine-tune for our rig; measure escape rate and false-call rate.
4. Tune confidence threshold per class for recall-first operation.
5. ONNX export → Vitis AI INT8 PTQ for the ZCU104 DPU (Tier 2/3); OBB post-processing on the Arm PS.

# Module A — Vision AI (Member 1)

YOLOv8-OBB defect detection for the PS5 AOI pipeline. This document records
what was built, the data decisions behind it, model specs, results and how to
reproduce them.

**Status (2026-11-22):** `vision-v2` released. 12-class SMT solder, assembly & placement model
fine-tuned from v1 weights; HF held-out test **0.893 mAP@0.5** (target ≥ 0.88).
Tombstone and orientation-error classes are now covered. Close-up solder suppression regression — see §7.
`vision-v1` remains available as release [`vision-v1`](https://github.com/MadhanSaiKrishna/bharat-ai-soc-ps5-pcb-aoi/releases/tag/vision-v1).

---

## 1. Quick start

```bash
# 1. fetch trained weights (not in git; published as a GitHub release)
python -m module_a_vision.download_weights          # vision-v2 -> module_a_vision/weights/smt_merged_v2_v8s.pt
python -m module_a_vision.download_weights --tag vision-v1 --all   # previous models

# 2. use from code
python - <<'EOF'
import cv2
from module_a_vision.infer_vision import VisionInferenceEngine
engine = VisionInferenceEngine()                     # auto-loads smt_merged_v2_v8s.pt, else smt_merged_v8s.pt
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
| `datasets/build_smt_merged.py` | Merges HF solder dataset + SolDef + Roboflow SMD placement (3 sources) into one 12-class OBB dataset |
| `embedding_extractor.py` | MobileNetV3 embedding for operator anomaly ingestion (projection head still untrained) |
| `weights/` | Local weights (git-ignored) |
| `HANDOFF_MEMBER2.md` | Note for Member 2: new v2 defect types, severities and the advisor fallback rules added for them |

## 3. Datasets

Raw and converted datasets live **outside the repo** (default
`../datasets/`), because of size. Images in converted sets are symlinks.

| Dataset | Source | Licence | Used? | Notes |
|---|---|---|---|---|
| **PCB Solder & Assembly Defect** | [HF: ParsaVictor/pcb-solder-defect-detection](https://huggingface.co/datasets/ParsaVictor/pcb-solder-defect-detection) | Research / educational use; contact author for commercial use or redistribution | ✅ main source | 5,209 real populated-PCB photos, 9 classes, axis-aligned YOLO boxes. ~70 % of its val/test images are dark background-only tiles from one source. |
| **SolDef_AI** | [Kaggle: mauriziocalabrese/soldef-ai-…](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) | GPL-3.0 | ✅ joint subset | 428 labelled macro photos (2560×1440) of 0805/1206 resistors, LabelMe polygons. Visually closest to our See3CAM rig. 93 Roboflow images originate from SolDef_AI (GPL-3.0, 2022-03-29 component-level day, not used by our joint subset); attribute both. |
| PCB-AoI | [Kaggle: kubeedgeianvs/pcb-aoi](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi) | Apache-2.0 | ❌ downloaded, not used | 173 train + 60 test IC-lead images under RGB tri-colour AOI lighting (domain mismatch). Classes: bad fillet slope, lifted lead. Candidate for a separate lead-inspection model. |
| **SMD Component Placement Defect 2** | [Roboflow Universe](https://universe.roboflow.com/muhammad-huda-ferdinansyah/smd-component-placement-defect-2) | CC BY 4.0 | ✅ used | 2,750 originals (640×640, 3 augmented copies per train image in export), polygon labels, one labelled component per image. Classes mapped: Tombstoning→TOMBSTONE, Misalignment→SHIFTED_COMPONENT, Orientation Error→ORIENTATION_ERROR, Normal→background. |
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
6. **Keep one of the 3 Roboflow augmented copies per original** in the train
   split to reduce redundancy while retaining data diversity.
7. **RF published splits leak** (crops of the same phone photo appear across
   splits). Images are grouped by capture burst (MVIMG_/IMG_/WIN_ timestamps
   < 60 s apart) or by original filename; any group spanning splits is moved to
   train. Result: 635 of 1,100 RF val/test images moved to train, leaving
   241 RF val / 224 RF test images.
8. **No near-duplicates between RF and HF** (hash check confirmed). HF test
   labels are byte-identical to v1, so the HF test split (686 images) remains
   the primary held-out score.
9. **Label-protocol conflict:** RF labels only one component per image; other
   visible defects are unlabelled. v1 fired solder classes on ~25 % of RF images
   at conf 0.5. This causes close-up solder suppression in v2 — see §7.

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
| 10 | TOMBSTONE | SMT_PLACEMENT | HIGH |
| 11 | ORIENTATION_ERROR | SMT_PLACEMENT | CRITICAL |

Also mapped for future models: `WIRE_SWEEP_SHORT`, `LIFTED_HEEL`, `LOOP_SAG`
(WIRE_BOND); `STARVED_PASTE` (SMT_SPI); `SKEWED_PASSIVE`
(SMT_PLACEMENT). `GOOD_JOINT` and `WIRE_NORMAL` are never emitted as defects.
Unknown classes default to `SMT_PLACEMENT` / `HIGH`.

> [!NOTE]
> ids 0–9 are **unchanged from v1**. TOMBSTONE (id 10): component stands on end,
> solder surface tension failure (SMT_PLACEMENT, HIGH). ORIENTATION_ERROR (id 11):
> flipped chip or reversed polarity (SMT_PLACEMENT, CRITICAL). OBB angle carries
> skew, so SKEWED_PASSIVE is merged into SHIFTED_COMPONENT and not a separate class.

## 5. Model specs — `smt_merged_v2_v8s.pt` (release `vision-v2`)

| Item | Value |
|---|---|
| Architecture | YOLOv8s-OBB (Ultralytics 8.4.216), fine-tuned from v1 weights `smt_merged_v8s.pt` |
| Input size | 832 px |
| Classes | 12 (ids 0–9 unchanged from v1; + 10 TOMBSTONE, 11 ORIENTATION_ERROR) |
| Training data | 6,216 train / 984 val / 910 test (686 HF + 224 RF) |
| Epochs | 60 (best checkpoint: epoch 54, by Ultralytics fitness = 0.1·mAP50 + 0.9·mAP50-95) |
| Optimiser | AdamW lr 0.000625 (auto), cosine LR, batch 8, seed 0 |
| Augmentation | mosaic 1.0 (off last 20 epochs), rotation ±15°, flip LR/UD 0.5, HSV-V 0.4 (same as v1) |
| Hardware / time | RTX 4050 Laptop 6 GB, 2.23 h |
| File size | 23.2 MB (FP32 `.pt`) |
| New head | 12-class head (new TOMBSTONE + ORIENTATION_ERROR classes added) |

Dataset built to `../datasets/yolo/smt_merged_v2_obb` with `--rf $D/rf_smd_placement/v5`.
Builder writes `data_test_hf.yaml` / `data_test_rf.yaml` for per-source evaluation.

### Results

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| val (984) | 0.878 | 0.721 | – | – |
| **HF test (686, held out)** | **0.893** | **0.763** | 0.836 | 0.889 |
| RF clean test (224) | 0.888 | 0.811 | 0.832 | 0.825 |
| **Combined test (910)** | **0.882** | **0.756** | – | – |

#### Per-class AP@0.5 — HF test (v1 → v2)

| Class | v1 AP@0.5 | v2 AP@0.5 |
|---|---|---|
| SHIFTED_COMPONENT | 0.995 * | 0.994 * |
| PHYSICAL_DAMAGE | 0.959 | 0.986 |
| MISSING_COMPONENT | 0.944 | 0.954 |
| EXCESS_SOLDER | 0.915 | 0.902 |
| SOLDER_BRIDGE | 0.902 | 0.903 |
| INSUFFICIENT_SOLDER | 0.892 | 0.885 |
| SOLDER_VOID | 0.770 | 0.858 |
| BURN | 0.684 | 0.660 (precision 0.67, was 0.82) |
| TOMBSTONE | n/a | — (no HF test instances) |
| ORIENTATION_ERROR | n/a | — (no HF test instances) |

\* Only 31 test instances; possibly near-duplicates across the HF splits, so treat as optimistic.
SOLDER_SPIKE and OXIDATION have no test instances.

#### Per-class AP@0.5 — RF clean test (224 images, v2 only)

| Class | AP@0.5 | Recall | Precision | Instances |
|---|---|---|---|---|
| TOMBSTONE | 0.993 | 1.00 | 0.92 | 51 |
| SHIFTED_COMPONENT | 0.984 | 0.98 | 0.90 | 42 |
| ORIENTATION_ERROR | 0.688 | 0.50 | 0.68 | 24 (small sample, weak) |

Inference latency (PyTorch FP32, RTX 4050, 832 px): **13.5 ms/img** model
(+0.9 ms pre, 0.5 ms post); unchanged vs v1.

### Previous: vision-v1

`smt_merged_v8s.pt` — YOLOv8s-OBB, 10 classes, init from DOTA-pretrained `yolov8s-obb.pt`.
3,980 train / 743 val / 686 test (HF + SolDef joint subset). 100 epochs (best: epoch 84).
AdamW lr 0.000714, batch 8, 2.39 h. HF test: **mAP@0.5 0.882**, mAP@0.5:0.95 0.749, P 0.811, R 0.884.
Still available as release [`vision-v1`](https://github.com/MadhanSaiKrishna/bharat-ai-soc-ps5-pcb-aoi/releases/tag/vision-v1).

#### Baseline — `soldef_joint_v8s.pt`
YOLOv8s-OBB, 1024 px, SolDef joint subset only (144 train / 56 val), 4 classes
incl. `GOOD_JOINT`. Val mAP@0.5 **0.752** (excess 0.89, spike 0.85,
insufficient 0.70, good 0.57). Kept for reference; superseded by the merged model.

## 6. Reproduce

Environment used: Python 3.14, torch 2.15 + CUDA 13.2, ultralytics 8.4.216,
opencv 5.1, kaggle CLI 2.2.5 (`pip install -r requirements.txt kaggle`).

```bash
D=../datasets                                         # outside the repo

# download HF and SolDef (same as v1)
kaggle auth login
kaggle datasets download -d mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection -p $D/soldef_ai --unzip
mkdir -p $D/pcb_solder_hf && cd $D/pcb_solder_hf && \
  curl -LO https://huggingface.co/datasets/ParsaVictor/pcb-solder-defect-detection/resolve/main/images_and_labels.zip && \
  unzip -q images_and_labels.zip && cd -

# download Roboflow "SMD Component Placement Defect 2" v5
# Roboflow Universe > Export Dataset > YOLOv8 format > Version 5
# Download zip and extract into $D/rf_smd_placement/v5
# URL: https://universe.roboflow.com/muhammad-huda-ferdinansyah/smd-component-placement-defect-2

# convert + merge (3 sources → v2 dataset)
python -m module_a_vision.datasets.convert_soldef   --src $D/soldef_ai/SolDef_AI/Labeled --dst $D/yolo/soldef_joint_obb
python -m module_a_vision.datasets.build_smt_merged --hf $D/pcb_solder_hf/final_solder \
       --soldef $D/yolo/soldef_joint_obb \
       --rf $D/rf_smd_placement/v5 \
       --dst $D/yolo/smt_merged_v2_obb
# builder writes data_test_hf.yaml and data_test_rf.yaml inside smt_merged_v2_obb/

# train (~2.23 h on RTX 4050) — fine-tune from v1 weights
python -m module_a_vision.train_obb --data $D/yolo/smt_merged_v2_obb/data.yaml \
       --model module_a_vision/weights/smt_merged_v8s.pt \
       --name smt_merged_v2_v8s --imgsz 832 --batch 8 --epochs 60 --patience 20

# evaluate on HF held-out test (686 images)
yolo obb val model=module_a_vision/weights/smt_merged_v2_v8s.pt \
       data=$D/yolo/smt_merged_v2_obb/data_test_hf.yaml split=test imgsz=832

# evaluate on RF clean test (224 images)
yolo obb val model=module_a_vision/weights/smt_merged_v2_v8s.pt \
       data=$D/yolo/smt_merged_v2_obb/data_test_rf.yaml split=test imgsz=832
```

Run outputs (curves, confusion matrices, sample predictions) go to `../runs/`.

## 7. Known limitations

- **Not validated on our rig.** All scores are on public photos; See3CAM frames
  have not been labelled or tested yet.
- **Close-up solder suppression (regression, v2).** RF images have unlabelled solder
  defects. v2 fires solder classes on 0 of 465 clean RF val+test images at conf 0.5
  (v1: 107). On SolDef macro val (56 imgs): solder recall 0.70→0.66, mAP50
  0.738→0.728 (INSUFFICIENT_SOLDER 0.651→0.606, EXCESS_SOLDER 0.840→0.837,
  SOLDER_SPIKE 0.724→0.741). **Risk for our close-up See3CAM rig.**
  Mitigation candidates: pseudo-label RF solder defects, or relabel a subset.
- **ORIENTATION_ERROR weak (RF test AP@0.5 0.688, R 0.50).** Only 24 RF test
  instances; small sample. Needs more labelled data.
- **BURN still weakest HF class** (AP@0.5 0.660 in v2, precision dropped to 0.67).
- **Skew merged into SHIFTED_COMPONENT.** OBB angle carries skew information;
  SKEWED_PASSIVE is not a separate class.
- **Precision 0.836** → roughly 1 in 6 alarms is a false call at the default
  0.50 confidence threshold; threshold not yet tuned per class.
- `component_ref` is a placeholder (`COMP_LOC_x_y`); mapping detections to
  Gerber reference designators needs the homography from Module B.
- Not yet exported to ONNX / INT8, and not checked on the ZCU104 DPU.

## 8. Roadmap

1. ~~Add Roboflow SMD placement data (tombstone, misalignment, orientation).~~ **Done in v2.**
2. Fix close-up solder suppression (pseudo-label or relabel RF solder defects).
3. Request Wong et al. wire-bond dataset; extend `synth_defects.py` to all classes and write YOLO-OBB output.
4. Label 100–300 See3CAM frames and fine-tune for our rig; measure escape rate and false-call rate.
5. Tune confidence threshold per class for recall-first operation.
6. ONNX export → Vitis AI INT8 PTQ for the ZCU104 DPU (Tier 2/3); OBB post-processing on the Arm PS.

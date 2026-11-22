#!/usr/bin/env python3
"""
End-to-End Pipeline Smoke Test & Verification Utility
Bharat AI-SoC PS5 PCB AOI

Tests the complete end-to-end chain across all modules:
  1. Input: Practical optical PCB sample (or synthetic frame)
  2. Module A (Vision AI): Oriented bounding box defect detection
  3. Contract Validation: Validates payload against schemas/inspection_event.json
  4. Module B (Edge LLM & Advisor): Ingests event, executes dual closed-loop copilot
  5. Telemetry & Storage: Records event in rolling historical telemetry database
"""

import argparse
import json
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module_a_vision.infer_vision import VisionInferenceEngine
from edge_llm_storage.dual_advisor import DualAdvisorEngine, HistoricalDefectBuffer
from edge_llm_storage.storage_manager import DecoupledStorageManager


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Test for Bharat AI-SoC PS5 AOI")
    parser.add_argument(
        "--image",
        type=str,
        default=str(REPO_ROOT / "data" / "practical_test_samples" / "01_real_solder_bridge.jpg"),
        help="Path to optical PCB test image",
    )
    parser.add_argument(
        "--board-id",
        type=str,
        default="PS5_AOI_TEST_001",
        help="Board Serial ID",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="CHECKPOINT_WIRE_AND_PLACEMENT",
        choices=["CHECKPOINT_1_SPI", "CHECKPOINT_2_PLACEMENT", "CHECKPOINT_WIRE_AND_PLACEMENT", "CHECKPOINT_4_FINAL"],
        help="Inspection checkpoint stage",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("  BHARAT AI-SOC PS5 PCB AOI - COMPLETE END-TO-END PIPELINE TEST")
    print("=" * 70)
    print(f"[INPUT] Optical Test Image: {image_path.name}")
    print(f"[INPUT] Target Checkpoint : {args.checkpoint}")
    print(f"[INPUT] Board ID          : {args.board_id}")
    print("-" * 70)

    # Step 1: Load Optical Image
    t_start = time.perf_counter()
    img = Image.open(image_path).convert("RGB")
    print(f"  [1/4] Image Loaded: Resolution {img.width}x{img.height} px")

    # Step 2: Module A - Vision Defect Detection
    print("  [2/4] Running Module A: Vision Defect Detection...")
    vision_engine = VisionInferenceEngine()
    defects, t_vision = vision_engine.detect_defects(img, checkpoint=args.checkpoint)
    print(f"        -> Detected {len(defects)} defects in {t_vision:.2f} ms")

    # Build Complete Contract Event
    event = vision_engine.build_complete_event(
        board_id=args.board_id,
        checkpoint=args.checkpoint,
        alignment_deltas={"dx_mm": 0.04, "dy_mm": -0.02, "dtheta_deg": 0.08},
        defects=defects,
        bga_zone_detected=True,
        bga_checklist_passed=(len(defects) == 0),
    )

    # Step 3: Validate against Schema
    schema_path = REPO_ROOT / "schemas" / "inspection_event.json"
    if schema_path.exists():
        import jsonschema
        with open(schema_path, "r") as sf:
            schema = json.load(sf)
        jsonschema.validate(instance=event, schema=schema)
        print("        -> Contract Validation: PASS (conforms to schemas/inspection_event.json)")

    # Print Defect Table
    if defects:
        print("\n  Detected Defect Manifest:")
        for d in defects:
            obb = d.get("obb", {})
            print(f"    - ID #{d.get('defect_id')}: [{d.get('severity')}] {d.get('type')} "
                  f"@ {d.get('component_ref')} (conf: {d.get('confidence', 1.0)}) "
                  f"OBB: ({obb.get('xc')}, {obb.get('yc')}, {obb.get('width')}x{obb.get('height')}, {obb.get('angle_deg')}°)")
    else:
        print("\n  No defects detected (Clean Board PASS).")

    # Step 4: Module B - Edge LLM Dual Advisory Copilot
    print("\n  [3/4] Running Module B: Dual Closed-Loop Root-Cause Advisor...")
    advisor_engine = DualAdvisorEngine()
    advisories = advisor_engine.generate_advisory(event)
    dfm_advice = advisories.get("dfm_layout_advisory", "")
    asm_advice = advisories.get("assembler_calibration_advisory", "")

    print("\n  [OUTPUT 1: CAD / DFM Designer Advisory (<40 words)]")
    print(f"  {dfm_advice.strip()}")

    print("\n  [OUTPUT 2: SMT / Assembler Calibration Advisory (<40 words)]")
    print(f"  {asm_advice.strip()}")

    # Step 5: Telemetry Storage
    print("\n  [4/4] Logging to Historical Telemetry Database...")
    storage = DecoupledStorageManager()
    log_path = storage.log_inspection_event(event, advisories=advisories)
    print(f"        -> Record logged to: {Path(log_path).name}")

    total_time = (time.perf_counter() - t_start) * 1000.0
    print("-" * 70)
    print(f"  END-TO-END PIPELINE STATUS: 100% OPERATIONAL (Total: {total_time:.1f} ms)")
    print("=" * 70)


if __name__ == "__main__":
    main()

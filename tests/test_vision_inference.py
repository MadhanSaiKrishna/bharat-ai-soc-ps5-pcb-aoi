#!/usr/bin/env python3
"""
Unit Tests for Vision Inference Engine (Tier 0 Deliverable)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot
"""

import sys
from pathlib import Path
import pytest
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
from module_a_vision.infer_vision import VisionInferenceEngine


def test_vision_engine_detection():
    # Tests the actual neural YOLOv8-OBB model on an authentic optical test sample
    engine = VisionInferenceEngine(conf_threshold=0.20)
    test_image_path = ROOT_DIR / "data" / "practical_test_samples" / "02_real_insufficient_solder.jpg"
    test_frame = Image.open(test_image_path)

    defects, latency_ms = engine.detect_defects(test_frame, checkpoint="CHECKPOINT_WIRE_AND_PLACEMENT")
    assert len(defects) >= 1
    assert latency_ms > 0

    # Verify OBB keys conforming strictly to schema
    for d in defects:
        assert "defect_id" in d
        assert "category" in d
        assert "type" in d
        assert "obb" in d
        obb = d["obb"]
        for key in ["xc", "yc", "width", "height", "angle_deg"]:
            assert key in obb


def test_vision_engine_event_builder():
    engine = VisionInferenceEngine(conf_threshold=0.20)
    test_image_path = ROOT_DIR / "data" / "practical_test_samples" / "02_real_insufficient_solder.jpg"
    test_frame = Image.open(test_image_path)
    defects, _ = engine.detect_defects(test_frame)

    event = engine.build_complete_event(
        board_id="TEST_BOARD_101",
        checkpoint="CHECKPOINT_WIRE_AND_PLACEMENT",
        alignment_deltas={"dx_mm": 0.03, "dy_mm": -0.01, "dtheta_deg": 0.05},
        defects=defects,
    )

    assert event["board_id"] == "TEST_BOARD_101"
    assert event["alignment"]["status"] == "PASS"
    assert len(event["defects"]) == len(defects)
    assert "historical_defect_context" in event
    assert "operator_ingestion" in event

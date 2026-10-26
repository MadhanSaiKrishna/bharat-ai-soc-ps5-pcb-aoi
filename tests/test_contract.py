#!/usr/bin/env python3
"""
CI / Smoke Test: Inter-Module Contract & Schema Validator
Ensures that code written by any member's AI agent does not break the shared schema.
"""

import os
import sys
import json
from pathlib import Path
import pytest
from jsonschema import validate, ValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SCHEMA_PATH = ROOT_DIR / "schemas" / "inspection_event.json"
SAMPLE_PATH = ROOT_DIR / "schemas" / "sample_inspection_event.json"


def test_schema_validity():
    """Verify that the master JSON schema itself is valid JSON."""
    assert SCHEMA_PATH.exists(), f"Schema not found at {SCHEMA_PATH}"
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    assert schema.get("title") == "InspectionEvent"


def test_sample_event_conforms_to_schema():
    """Verify that sample_inspection_event.json satisfies all schema rules."""
    assert SAMPLE_PATH.exists(), f"Sample event not found at {SAMPLE_PATH}"
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    with open(SAMPLE_PATH, "r") as f:
        sample = json.load(f)

    # Validate against schema
    try:
        validate(instance=sample, schema=schema)
    except ValidationError as e:
        pytest.fail(f"sample_inspection_event.json violates schemas/inspection_event.json: {e.message}")


def test_dual_advisor_deterministic_contract():
    """Verify that DualAdvisor produces required keys and conforms to word limit."""
    from edge_llm_storage.dual_advisor import DualAdvisorEngine

    advisor = DualAdvisorEngine()
    with open(SAMPLE_PATH, "r") as f:
        sample = json.load(f)

    advisories = advisor.generate_advisory(sample)
    assert "dfm_layout_advisory" in advisories
    assert "assembler_calibration_advisory" in advisories

    # Ensure advisory messages adhere to the strict competition <40 words rule
    dfm_words = len(advisories["dfm_layout_advisory"].split())
    asm_words = len(advisories["assembler_calibration_advisory"].split())

    assert dfm_words <= 40, f"DFM advisory exceeded 40 words: {dfm_words} words"
    assert asm_words <= 40, f"Assembler advisory exceeded 40 words: {asm_words} words"


def test_homography_tolerance_evaluation():
    """Verify homography engine tolerance evaluation conforms to schema format."""
    from module_b_guidance.homography_align import FiducialHomographyEngine

    engine = FiducialHomographyEngine()
    eval_result = engine.evaluate_alignment_tolerance(
        {"dx_mm": 0.05, "dy_mm": -0.02, "dtheta_deg": 0.08},
        tier_key="bga_die"
    )
    assert eval_result["status"] in ["PASS", "FAIL", "WARNING"]
    assert "dx_mm" in eval_result
    assert "tolerance_dx_mm" in eval_result

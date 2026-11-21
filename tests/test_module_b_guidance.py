#!/usr/bin/env python3
"""
Unit and Integration Tests for Module B Guidance & Operator UI (Member 3)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot
"""

import json
import os
import sys
from pathlib import Path
import pytest
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from module_b_guidance.gerber_renderer import GerberRenderer
from module_b_guidance.homography_align import FiducialHomographyEngine
from module_b_guidance.operator_ingest_tool import OperatorAnomalyIngestor
from module_b_guidance.report_generator import AuditReportGenerator

SAMPLE_PATH = ROOT_DIR / "schemas" / "sample_inspection_event.json"


def test_gerber_renderer_synthetic_board():
    renderer = GerberRenderer(pixels_per_mm=50.0)
    cad_img, fids, comps = renderer.generate_synthetic_cad_board(width_mm=100.0, height_mm=80.0)

    assert cad_img.size == (5000, 4000)
    assert cad_img.mode == "RGBA"
    assert len(fids) == 3
    assert len(comps) >= 5

    # Test blending
    cam_mock = Image.new("RGBA", (5000, 4000), (30, 50, 40, 255))
    blended = renderer.composite_with_camera_frame(cam_mock, cad_img, alpha=0.5)
    assert blended.size == (5000, 4000)


def test_gerber_renderer_bundle_render():
    renderer = GerberRenderer()
    sample_zip = ROOT_DIR / "data" / "sample_gerbers" / "CUSTOMER_PCB_JOB_REV1.zip"
    if sample_zip.exists():
        cad_img, fids, meta = renderer.render_gerber_bundle(sample_zip, dpi=100)
        assert cad_img is not None
        assert cad_img.mode == "RGBA"
        assert len(fids) == 3
        assert "CUSTOMER_PCB_JOB_REV1" in meta["job_name"]
        assert len(meta["layers"]) >= 2


def test_homography_alignment_calculation():
    engine = FiducialHomographyEngine(pixels_per_mm=50.0)
    cad_fiducials = [(10.0, 10.0), (90.0, 10.0), (50.0, 70.0)]
    cam_fiducials = [
        (10.0 * 50.0 + 1.0, 10.0 * 50.0 - 0.5),
        (90.0 * 50.0 + 1.1, 10.0 * 50.0 - 0.4),
        (50.0 * 50.0 + 1.0, 70.0 * 50.0 - 0.5),
    ]

    H, deltas = engine.compute_homography(cam_fiducials, cad_fiducials)
    assert H is not None
    assert abs(deltas["dx_mm"] - 0.02) < 0.01
    assert abs(deltas["dy_mm"] - (-0.01)) < 0.01

    eval_result = engine.evaluate_alignment_tolerance(deltas, tier_key="bga_die")
    assert eval_result["status"] == "PASS"


def test_operator_anomaly_ingest_workflow(tmp_path):
    queue_dir = tmp_path / "operator_queue"
    ingestor = OperatorAnomalyIngestor(queue_dir=queue_dir)

    dummy_frame = Image.new("RGB", (640, 480), (100, 150, 200))
    res = ingestor.crop_and_ingest(
        frame=dummy_frame,
        bbox=(50, 50, 150, 150),
        operator_label="SOLDER_SPLASH",
        board_id="TEST_UNIT_99",
        severity="HIGH",
        notes="Automated test crop",
    )

    assert res["flagged_new_anomaly"] is True
    assert res["operator_label"] == "SOLDER_SPLASH"
    assert "anomaly_id" in res
    assert (ROOT_DIR / res["crop_path"]).exists()

    # Check JSONL log entry
    log_file = queue_dir / "ingest_log.jsonl"
    assert log_file.exists()
    with open(log_file) as f:
        line = f.readline()
        entry = json.loads(line)
        assert entry["operator_label"] == "SOLDER_SPLASH"
        assert entry["board_id"] == "TEST_UNIT_99"


def test_audit_report_generation(tmp_path):
    generator = AuditReportGenerator(output_dir=tmp_path)
    with open(SAMPLE_PATH) as f:
        event_data = json.load(f)

    csv_file = generator.export_csv(event_data)
    assert csv_file.exists()
    with open(csv_file) as f:
        content = f.read()
        assert "WIRE_SWEEP_SHORT" in content
        assert "SKEWED_PASSIVE" in content

    pdf_file = generator.export_pdf(
        event_data,
        dfm_advisory="Increase pad spacing to >= 0.20 mm",
        asm_advisory="Reduce capillary power by 6%",
    )
    assert pdf_file.exists()


def test_operator_cockpit_app_initialization():
    try:
        from PyQt6.QtWidgets import QApplication
        from module_b_guidance.app_ui import OperatorCockpitApp
    except ImportError:
        pytest.skip("PyQt6 not installed in current environment; skipping GUI test.")

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(["test_app", "--headless"])
    window = OperatorCockpitApp()
    assert window.board_id_edit.text() == "OPTIMUS_BGA_REV1_004"
    assert len(window.current_event.get("defects", [])) == 2
    assert "U1_DIE_PAD_B4_B5" in window.dfm_text or "clearance" in window.dfm_text
    window.close()

#!/usr/bin/env python3
"""
AOI Inspection Audit Report Generator (Member 3 Deliverable)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot

Generates standardized shop-floor PDF and CSV compliance reports
from inspection events and dual DFM/Assembler advisories.
"""

import csv
import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = REPO_ROOT / "data" / "telemetry_logs"


class AuditReportGenerator:
    """Generates PDF and CSV compliance reports from inspection event records."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or DEFAULT_REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(self, event_data: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
        board_id = event_data.get("board_id", "BOARD_UNKNOWN")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = filepath or (self.output_dir / f"audit_report_{board_id}_{timestamp}.csv")

        defects = event_data.get("defects", [])
        fieldnames = ["defect_id", "category", "type", "component_ref", "confidence", "severity", "xc", "yc", "angle_deg"]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in defects:
                obb = d.get("obb", {})
                writer.writerow({
                    "defect_id": d.get("defect_id"),
                    "category": d.get("category"),
                    "type": d.get("type"),
                    "component_ref": d.get("component_ref"),
                    "confidence": d.get("confidence"),
                    "severity": d.get("severity"),
                    "xc": obb.get("xc"),
                    "yc": obb.get("yc"),
                    "angle_deg": obb.get("angle_deg"),
                })
        return out_path

    def export_pdf(
        self,
        event_data: Dict[str, Any],
        dfm_advisory: str = "N/A",
        asm_advisory: str = "N/A",
        filepath: Optional[Path] = None,
    ) -> Path:
        board_id = event_data.get("board_id", "BOARD_UNKNOWN")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = filepath or (self.output_dir / f"audit_report_{board_id}_{timestamp}.pdf")

        if not HAS_REPORTLAB:
            # Fallback to plain text summary if reportlab is not yet installed
            txt_path = out_path.with_suffix(".txt")
            with open(txt_path, "w") as f:
                f.write(f"=== BHARAT AI-SOC PS5 PCB AOI AUDIT REPORT ===\n")
                f.write(f"Board ID: {board_id}\n")
                f.write(f"Checkpoint: {event_data.get('checkpoint')}\n")
                f.write(f"Alignment Status: {event_data.get('alignment', {}).get('status')}\n")
                f.write(f"Defects Count: {len(event_data.get('defects', []))}\n\n")
                f.write(f"[DFM Advisory]\n{dfm_advisory}\n\n")
                f.write(f"[Assembler Calibration]\n{asm_advisory}\n")
            return txt_path

        doc = SimpleDocTemplate(str(out_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=12,
        )
        elements.append(Paragraph("Bharat AI-SoC PS5 PCB AOI - Quality Audit Report", title_style))
        elements.append(Paragraph(f"<b>Board ID:</b> {board_id} | <b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | <b>Checkpoint:</b> {event_data.get('checkpoint')}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Alignment & Gate Summary
        align = event_data.get("alignment", {})
        align_status = align.get("status", "UNKNOWN")
        status_color = colors.HexColor("#228B22") if align_status == "PASS" else colors.HexColor("#DC143C")

        summary_data = [
            ["Metric", "Measured Value", "Tolerance", "Status"],
            ["Translation dX", f"{align.get('dx_mm', 0.0):.3f} mm", f"<= {align.get('tolerance_dx_mm', 0.15)} mm", align_status],
            ["Translation dY", f"{align.get('dy_mm', 0.0):.3f} mm", f"<= {align.get('tolerance_dx_mm', 0.15)} mm", align_status],
            ["Rotation dTheta", f"{align.get('dtheta_deg', 0.0):.3f} deg", f"<= {align.get('tolerance_dtheta_deg', 0.20)} deg", align_status],
            ["BGA Checklist Gate", "All 4 Gates Passed" if event_data.get("bga_checklist_passed") else "Failed", "100%", "PASS" if event_data.get("bga_checklist_passed") else "FAIL"],
        ]
        t_summary = Table(summary_data, colWidths=[140, 130, 130, 100])
        t_summary.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ]))
        elements.append(t_summary)
        elements.append(Spacer(1, 16))

        # Detected Defects Table
        elements.append(Paragraph("<b>Detected Oriented Micro-Defects</b>", styles["Heading2"]))
        defects = event_data.get("defects", [])
        if defects:
            defect_rows = [["ID", "Category", "Defect Type", "Component", "Conf", "Severity"]]
            for d in defects:
                defect_rows.append([
                    str(d.get("defect_id")),
                    d.get("category", ""),
                    d.get("type", ""),
                    d.get("component_ref", ""),
                    f"{d.get('confidence', 0.0):.2f}",
                    d.get("severity", ""),
                ])
            t_defects = Table(defect_rows, colWidths=[30, 110, 140, 120, 50, 70])
            t_defects.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A5568")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (4, 1), (-1, -1), "CENTER"),
            ]))
            elements.append(t_defects)
        else:
            elements.append(Paragraph("<i>No micro-defects detected. Board conforms to quality standard.</i>", styles["Normal"]))
        elements.append(Spacer(1, 16))

        # Closed-Loop Dual Advisories
        elements.append(Paragraph("<b>Closed-Loop Dual Root-Cause Advisories</b>", styles["Heading2"]))
        adv_data = [
            [Paragraph("<b>Branch 1: CAD / DFM Layout Fix</b>", styles["Normal"])],
            [Paragraph(dfm_advisory, styles["Normal"])],
            [Paragraph("<b>Branch 2: SMT / Wire Bonder Calibration Action</b>", styles["Normal"])],
            [Paragraph(asm_advisory, styles["Normal"])],
        ]
        t_adv = Table(adv_data, colWidths=[520])
        t_adv.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#E2E8F0")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#A0AEC0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t_adv)

        doc.build(elements)
        return out_path


if __name__ == "__main__":
    generator = AuditReportGenerator()
    with open(REPO_ROOT / "schemas" / "sample_inspection_event.json") as f:
        sample_data = json.load(f)
    csv_file = generator.export_csv(sample_data)
    pdf_file = generator.export_pdf(
        sample_data,
        dfm_advisory="Increase pad clearance to 0.20mm at U1 die pads.",
        asm_advisory="Reduce capillary ultrasonic power by 6%.",
    )
    print(f"[AuditReportGenerator] CSV exported to: {csv_file}")
    print(f"[AuditReportGenerator] PDF exported to: {pdf_file}")

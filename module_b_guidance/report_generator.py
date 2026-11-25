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

        defects = event_data.get("defects", [])
        verdict = "PASSED (ZERO DEFECTS)" if len(defects) == 0 else f"REWORK REQUIRED ({len(defects)} DEFECTS)"
        vision_ms = event_data.get("metrics", {}).get("vision_latency_ms")
        vision_str = f" | Vision: {vision_ms:.1f} ms" if vision_ms is not None else ""

        if not HAS_REPORTLAB:
            txt_path = out_path.with_suffix(".txt")
            with open(txt_path, "w") as f:
                f.write("=== BHARAT AI-SOC PS5 PCB AOI — INSPECTION REPORT ===\n")
                f.write(f"Board / Image: {board_id}\n")
                f.write(f"Verdict: {verdict}{vision_str}\n")
                f.write(f"Defects Count: {len(defects)}\n\n")
                f.write(f"[CAD / DFM Advisory]\n{dfm_advisory}\n\n")
                f.write(f"[Assembler Action]\n{asm_advisory}\n")
            return txt_path

        doc = SimpleDocTemplate(str(out_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        elements = []

        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=8,
        )
        elements.append(Paragraph("Bharat AI-SoC PS5 PCB AOI — Inspection Report", title_style))
        elements.append(
            Paragraph(
                f"<b>Board / Image:</b> {board_id} | "
                f"<b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>"
                f"<b>Verdict:</b> {verdict}{vision_str}",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 14))

        # Detected Defects Table
        elements.append(Paragraph("<b>Detected Defects (YOLOv8s-OBB)</b>", styles["Heading2"]))
        if defects:
            defect_rows = [["ID", "Category", "Defect Type", "Conf", "Severity", "Angle (deg)"]]
            for d in defects:
                obb = d.get("obb", {})
                defect_rows.append([
                    str(d.get("defect_id", "")),
                    d.get("category", ""),
                    d.get("type", ""),
                    f"{d.get('confidence', 0.0):.2f}",
                    d.get("severity", ""),
                    f"{obb.get('angle_deg', 0.0):.1f}",
                ])
            t_defects = Table(defect_rows, colWidths=[35, 110, 165, 55, 80, 75])
            t_defects.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 1), (0, -1), "CENTER"),
                ("ALIGN", (3, 1), (-1, -1), "CENTER"),
            ]))
            elements.append(t_defects)
        else:
            elements.append(Paragraph("<i>No defects detected. Board passed inspection.</i>", styles["Normal"]))
        elements.append(Spacer(1, 16))

        # AI Copilot Advisories
        elements.append(Paragraph("<b>AI Copilot Advisories (Llama 3.2 1B ExecuTorch)</b>", styles["Heading2"]))
        adv_data = [
            [Paragraph("<b>CAD / DFM Root-Cause Advisory</b>", styles["Normal"])],
            [Paragraph(dfm_advisory, styles["Normal"])],
            [Paragraph("<b>Assembler Corrective Action</b>", styles["Normal"])],
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

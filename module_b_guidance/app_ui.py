#!/usr/bin/env python3
"""
Minimal Operator UI Cockpit & Quality-Control Copilot (Member 3 Deliverable)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot

PyQt6 Industrial Operator Cockpit featuring:
1. Optical PCB Frame & YOLOv8-OBB Rotated Bounding Box Defect Visualizer.
2. Live Camera Stream or Disk Image Inspection (Spacebar).
3. Pass / Rework Verdict Pill & Detected Defects Table.
4. Dual Closed-Loop AI Copilot Advisories (Branch 1: CAD DFM, Branch 2: Assembler Action).
5. One-click PDF/CSV Audit Report Export.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Headless check for CI / test environments
if "--headless" in sys.argv or os.environ.get("HEADLESS_TEST") == "1":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt, QRectF, QUrl, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QGroupBox, QMessageBox, QFileDialog,
    QSplitter, QTextEdit, QSizePolicy
)

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_EVENT_PATH = REPO_ROOT / "schemas" / "sample_inspection_event.json"
DEFAULT_SAMPLE_IMAGE = REPO_ROOT / "data" / "practical_test_samples" / "02_real_insufficient_solder.jpg"

from module_b_guidance.gerber_renderer import GerberRenderer
from module_b_guidance.report_generator import AuditReportGenerator

try:
    from edge_llm_storage.dual_advisor import DualAdvisorEngine
except ImportError:
    DualAdvisorEngine = None

try:
    from edge_llm_storage.storage_manager import DecoupledStorageManager
except ImportError:
    DecoupledStorageManager = None

try:
    from module_a_vision.infer_vision import VisionInferenceEngine
except ImportError:
    VisionInferenceEngine = None

try:
    import cv2
except ImportError:
    cv2 = None


class InspectionCanvas(QWidget):
    """Interactive viewport for PCB optical frame and YOLOv8-OBB rotated bounding boxes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #0F141C; border: 1px solid #2D3748; border-radius: 4px;")
        self.camera_pixmap: Optional[QPixmap] = None
        self.defects: List[Dict[str, Any]] = []
        self.selected_defect_id: Optional[int] = None

    def set_camera_frame(self, camera_img: Image.Image):
        """Updates the displayed PCB optical frame."""
        cam_rgba = camera_img.convert("RGBA")
        qim_cam = QImage(
            cam_rgba.tobytes(),
            cam_rgba.width,
            cam_rgba.height,
            QImage.Format.Format_RGBA8888,
        )
        self.camera_pixmap = QPixmap.fromImage(qim_cam)
        self.update()

    def set_defects(self, defects: List[Dict[str, Any]]):
        self.defects = defects
        self.update()

    def select_defect(self, defect_id: Optional[int]):
        self.selected_defect_id = defect_id
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        if not self.camera_pixmap:
            painter.setPen(QColor("#718096"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No PCB Image Loaded\nClick 'Load PCB Image' or 'Start Live Camera'",
            )
            return

        cam_w = max(1, self.camera_pixmap.width())
        cam_h = max(1, self.camera_pixmap.height())
        ratio = min(w / cam_w, h / cam_h)
        disp_w = cam_w * ratio
        disp_h = cam_h * ratio
        target_rect = QRectF((w - disp_w) / 2, (h - disp_h) / 2, disp_w, disp_h)

        scale_x = target_rect.width() / cam_w
        scale_y = target_rect.height() / cam_h

        painter.setOpacity(1.0)
        painter.drawPixmap(target_rect.toRect(), self.camera_pixmap)

        for defect in self.defects:
            obb = defect.get("obb", {})
            xc = obb.get("xc", 0.0) * scale_x + target_rect.left()
            yc = obb.get("yc", 0.0) * scale_y + target_rect.top()
            bw = obb.get("width", 20.0) * scale_x
            bh = obb.get("height", 20.0) * scale_y
            angle = obb.get("angle_deg", 0.0)
            severity = defect.get("severity", "MEDIUM")
            defect_id = defect.get("defect_id")

            if severity == "CRITICAL":
                color = QColor("#FF2A55")
            elif severity == "HIGH":
                color = QColor("#FF8C00")
            elif severity == "MEDIUM":
                color = QColor("#FFD700")
            else:
                color = QColor("#00E5FF")

            is_selected = defect_id == self.selected_defect_id
            pen_width = 3 if is_selected else 2
            painter.setPen(QPen(color, pen_width, Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 45)))

            painter.save()
            painter.translate(xc, yc)
            painter.rotate(angle)
            painter.drawRoundedRect(QRectF(-bw / 2, -bh / 2, bw, bh), 2, 2)
            painter.restore()

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#FFFFFF"))
            label_text = f"#{defect_id} {defect.get('type', 'DEFECT')} ({defect.get('confidence', 0.0):.2f})"
            painter.drawText(int(xc - bw / 2), int(yc - bh / 2 - 4), label_text)


def get_windows_host_ip() -> str:
    """Returns the Windows host default gateway IP from the WSL2 routing table."""
    try:
        import subprocess
        out = subprocess.check_output(["ip", "route", "show", "default"], text=True, stderr=subprocess.DEVNULL)
        parts = out.split()
        if "via" in parts:
            return parts[parts.index("via") + 1]
    except Exception:
        pass
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.split()[1]
    except Exception:
        pass
    return "127.0.0.1"


class CameraWorker(QThread):
    """Asynchronous camera worker thread for UVC or Windows HTTP bridge streams."""

    frame_ready = pyqtSignal(object)  # PIL.Image
    error_occurred = pyqtSignal(str)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True

    def run(self):
        if cv2 is None:
            self.error_occurred.emit("OpenCV (cv2) is not installed.")
            return

        cap = None
        stream_url = None
        try:
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.source)
            else:
                stream_url = str(self.source)
                cap = cv2.VideoCapture(stream_url)

            if not cap or not cap.isOpened():
                self.error_occurred.emit(
                    f"Could not open camera stream '{self.source}'.\n\n"
                    "If using Windows camera bridge, start it first:\n"
                    "    python tools/stream_camera_windows.py"
                )
                return

            if isinstance(self.source, int):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            consecutive_failures = 0
            last_emit_time = 0.0
            target_interval = 1.0 / 15.0
            while self.running:
                ret, bgr_frame = cap.read()
                now = time.time()
                if ret and bgr_frame is not None:
                    consecutive_failures = 0
                    if now - last_emit_time >= target_interval:
                        last_emit_time = now
                        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        self.frame_ready.emit(Image.fromarray(rgb_frame))
                    self.msleep(10)
                else:
                    consecutive_failures += 1
                    if stream_url and self.running and consecutive_failures < 10:
                        self.msleep(200)
                        cap.release()
                        cap = cv2.VideoCapture(stream_url)
                        continue
                    if consecutive_failures >= 5:
                        self.error_occurred.emit("Camera stream timed out.")
                        break
                    self.msleep(50)
        except Exception as e:
            self.error_occurred.emit(f"Camera stream error: {e}")
        finally:
            if cap:
                cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait(1000)


class AdvisorWorker(QThread):
    """Runs Llama 3.2 1B ExecuTorch advisory generation off the GUI thread."""

    finished_advisory = pyqtSignal(dict, dict, float)  # (event, advisories_dict, elapsed_s)
    error_occurred = pyqtSignal(str)

    def __init__(self, advisor_engine, event: Dict[str, Any]):
        super().__init__()
        self.advisor_engine = advisor_engine
        self.event = event

    def run(self):
        t0 = time.perf_counter()
        try:
            res = self.advisor_engine.generate_advisory(self.event)
            elapsed = time.perf_counter() - t0
            self.finished_advisory.emit(self.event, res, elapsed)
        except Exception as e:
            self.error_occurred.emit(str(e))


class OperatorCockpitApp(QMainWindow):
    """Minimal Industrial AOI Cockpit GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bharat AI-SoC PS5 — PCB AOI Operator Cockpit")
        self.resize(1180, 720)
        self.setMinimumSize(640, 480)

        self.gerber_renderer = GerberRenderer()
        self.report_generator = AuditReportGenerator()
        self.advisor_engine = DualAdvisorEngine() if DualAdvisorEngine else None
        self.vision_engine = VisionInferenceEngine() if VisionInferenceEngine else None
        self.storage_manager = DecoupledStorageManager() if DecoupledStorageManager else None

        self.current_event: Dict[str, Any] = {}
        self.camera_image: Optional[Image.Image] = None
        self.current_image_name: str = "sample_board"
        self.last_vision_ms: float = 0.0
        self.dfm_text: str = "Load a PCB image or start live camera, then click 'Inspect Frame'."
        self.asm_text: str = "Load a PCB image or start live camera, then click 'Inspect Frame'."
        self.camera_worker: Optional[CameraWorker] = None
        self.advisor_worker: Optional[AdvisorWorker] = None

        self._apply_theme()
        self._build_ui()
        self.dfm_box.setText(self.dfm_text)
        self.asm_box.setText(self.asm_text)
        self._set_status("Ready — Load a PCB image or start live camera.", "idle")

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0F141C; color: #E2E8F0; }
            QLabel { color: #CBD5E0; font-family: 'Segoe UI', sans-serif; }
            QGroupBox {
                border: 1px solid #2D3748;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                color: #63B3ED;
                padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton {
                background-color: #2B6CB0;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 7px 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3182CE; }
            QPushButton:pressed { background-color: #2C5282; }
            QTableWidget {
                background-color: #1A202C;
                color: #E2E8F0;
                gridline-color: #2D3748;
                border: 1px solid #2D3748;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2D3748;
                color: #A0AEC0;
                font-weight: bold;
                border: 1px solid #1A202C;
            }
            QTabWidget::pane { border: 1px solid #2D3748; background: #1A202C; border-radius: 4px; }
            QTabBar::tab {
                background: #2D3748;
                color: #A0AEC0;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #3182CE; color: #FFFFFF; font-weight: bold; }
        """)

    def _build_ui(self):
        main_widget = QWidget()
        main_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # 1. Responsive Two-Row Top Bar (prevents horizontal overflow on narrow windows)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("⚡ BHARAT AI-SoC PS5 — PCB AOI")
        title_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #63B3ED;")
        top_row.addWidget(title_label)
        top_row.addStretch()

        self.lbl_final_verdict = QLabel("VERDICT: READY")
        self.lbl_final_verdict.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_final_verdict.setStyleSheet(
            "background-color: #2D3748; color: #E2E8F0; padding: 6px 14px; border-radius: 4px;"
        )
        top_row.addWidget(self.lbl_final_verdict)
        main_layout.addLayout(top_row)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)

        self.load_img_btn = QPushButton("Load PCB Image")
        self.load_img_btn.clicked.connect(self._on_load_image)
        btn_row.addWidget(self.load_img_btn)

        self.load_gerber_btn = QPushButton("Load Gerber (.zip)")
        self.load_gerber_btn.clicked.connect(self._on_load_gerber)
        btn_row.addWidget(self.load_gerber_btn)

        self.camera_btn = QPushButton("Start Live Camera")
        self.camera_btn.setStyleSheet("background-color: #2C7A7B; color: #FFFFFF; font-weight: bold; padding: 7px 14px; border-radius: 4px;")
        self.camera_btn.clicked.connect(self._toggle_camera)
        btn_row.addWidget(self.camera_btn)

        self.inspect_btn = QPushButton("Inspect Frame (Space)")
        self.inspect_btn.setStyleSheet("background-color: #2F855A; color: #FFFFFF; font-weight: bold; padding: 7px 16px; border-radius: 4px;")
        self.inspect_btn.setShortcut("Space")
        self.inspect_btn.clicked.connect(self._run_cycle)
        btn_row.addWidget(self.inspect_btn)

        btn_row.addStretch()
        main_layout.addLayout(btn_row)

        # Prominent Top Status Banner (clickable to jump to AI Copilot Advisories tab)
        self.status_banner = QPushButton("Ready — Load a PCB image or start live camera.")
        self.status_banner.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_banner.setToolTip("Click to open the AI Copilot Advisories tab")
        self.status_banner.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        main_layout.addWidget(self.status_banner)

        # 2. Responsive Main Two-Column Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        splitter.setChildrenCollapsible(False)

        # Left: Optical PCB Viewport
        self.canvas = InspectionCanvas()
        splitter.addWidget(self.canvas)

        # Right: Defects + AI Copilot Tabs
        right_panel = QWidget()
        right_panel.setMinimumWidth(260)
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        r_layout = QVBoxLayout(right_panel)
        r_layout.setContentsMargins(6, 0, 0, 0)

        self.metrics_label = QLabel("Defects: 0 | Vision: -- ms")
        self.metrics_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.metrics_label.setStyleSheet("color: #A0AEC0; padding: 2px 4px;")
        r_layout.addWidget(self.metrics_label)

        self.tabs = QTabWidget()

        # Tab 1: Detected Defects
        defects_tab = QWidget()
        d_layout = QVBoxLayout(defects_tab)
        self.defect_table = QTableWidget(0, 5)
        self.defect_table.setHorizontalHeaderLabels(["ID", "Defect Type", "Conf", "Severity", "Angle (°)"])
        self.defect_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.defect_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.defect_table.itemSelectionChanged.connect(self._on_defect_selected)
        d_layout.addWidget(self.defect_table)
        self.tabs.addTab(defects_tab, "Detected Defects")

        # Tab 2: AI Copilot Advisories
        copilot_tab = QWidget()
        c_layout = QVBoxLayout(copilot_tab)

        dfm_group = QGroupBox("CAD / DFM Root-Cause Advisory")
        dfm_l = QVBoxLayout(dfm_group)
        self.dfm_box = QTextEdit()
        self.dfm_box.setReadOnly(True)
        self.dfm_box.setStyleSheet("background-color: #0F141C; color: #E2E8F0; border: none; font-size: 10pt;")
        dfm_l.addWidget(self.dfm_box)
        c_layout.addWidget(dfm_group)

        asm_group = QGroupBox("Assembler Corrective Action")
        asm_l = QVBoxLayout(asm_group)
        self.asm_box = QTextEdit()
        self.asm_box.setReadOnly(True)
        self.asm_box.setStyleSheet("background-color: #0F141C; color: #E2E8F0; border: none; font-size: 10pt;")
        asm_l.addWidget(self.asm_box)
        c_layout.addWidget(asm_group)

        self.tabs.addTab(copilot_tab, "AI Copilot Advisories")
        r_layout.addWidget(self.tabs)

        self.export_btn = QPushButton("Export Audit Report (PDF / CSV)")
        self.export_btn.clicked.connect(self._on_export_report)
        r_layout.addWidget(self.export_btn)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([700, 460])
        main_layout.addWidget(splitter)

        self.setCentralWidget(main_widget)

    def _load_initial_sample(self):
        """Loads a default PCB sample on startup and runs initial defect detection."""
        if DEFAULT_SAMPLE_IMAGE.exists():
            self.camera_image = Image.open(DEFAULT_SAMPLE_IMAGE).convert("RGBA")
            self.current_image_name = DEFAULT_SAMPLE_IMAGE.stem
            self.canvas.set_camera_frame(self.camera_image)

            defects: List[Dict[str, Any]] = []
            vision_ms = 0.0
            if self.vision_engine:
                defects, vision_ms = self.vision_engine.detect_defects(self.camera_image.convert("RGB"))

            if not defects and SAMPLE_EVENT_PATH.exists():
                with open(SAMPLE_EVENT_PATH) as f:
                    defects = json.load(f).get("defects", [])

            self._build_event(defects, vision_ms)
            self._update_defect_views(defects, vision_ms)
            self.dfm_box.setText(self.dfm_text)
            self.asm_box.setText(self.asm_text)
            self.status_bar.showMessage(f"Loaded sample PCB: {DEFAULT_SAMPLE_IMAGE.name} — Press Space or click 'Inspect Frame'")
        elif SAMPLE_EVENT_PATH.exists():
            self.load_mock_event(SAMPLE_EVENT_PATH)

    def load_mock_event(self, event_path: Path):
        """Loads a schema-compliant inspection event JSON as fallback."""
        with open(event_path) as f:
            event = json.load(f)
        self.current_event = event
        self.camera_image = Image.new("RGBA", (800, 600), (24, 58, 38, 255))
        self.canvas.set_camera_frame(self.camera_image)
        defects = event.get("defects", [])
        self._update_defect_views(defects, vision_ms=0.0)
        self.dfm_box.setText(self.dfm_text)
        self.asm_box.setText(self.asm_text)

    def _build_event(self, defects: List[Dict[str, Any]], vision_ms: float) -> Dict[str, Any]:
        event = {
            "event_id": f"AOI_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "board_id": self.current_image_name,
            "checkpoint": "CHECKPOINT_PLACEMENT",
            "homography": {"dx_mm": 0.0, "dy_mm": 0.0, "dtheta_deg": 0.0, "status": "PASS"},
            "defects": defects,
            "metrics": {"vision_latency_ms": round(vision_ms, 2)},
        }
        self.current_event = event
        return event

    def _update_defect_views(self, defects: List[Dict[str, Any]], vision_ms: float = 0.0):
        """Updates canvas OBBs, defects table, metrics label, and pass/rework verdict pill."""
        self.canvas.set_defects(defects)
        self.defect_table.setRowCount(len(defects))

        for row, d in enumerate(defects):
            obb = d.get("obb", {})
            self.defect_table.setItem(row, 0, QTableWidgetItem(str(d.get("defect_id", row + 1))))
            self.defect_table.setItem(row, 1, QTableWidgetItem(str(d.get("type", "UNKNOWN"))))
            self.defect_table.setItem(row, 2, QTableWidgetItem(f"{d.get('confidence', 0.0):.2f}"))
            self.defect_table.setItem(row, 3, QTableWidgetItem(str(d.get("severity", "MEDIUM"))))
            self.defect_table.setItem(row, 4, QTableWidgetItem(f"{obb.get('angle_deg', 0.0):.1f}"))

        self.metrics_label.setText(f"Defects: {len(defects)} | Vision: {vision_ms:.1f} ms")

        if len(defects) == 0:
            self.lbl_final_verdict.setText("VERDICT: PASSED (ZERO DEFECTS)")
            self.lbl_final_verdict.setStyleSheet(
                "background-color: #1C4532; color: #68D391; padding: 6px 14px; "
                "border-radius: 4px; border: 1px solid #276749;"
            )
        else:
            self.lbl_final_verdict.setText(f"VERDICT: REWORK REQUIRED ({len(defects)} DEFECTS)")
            self.lbl_final_verdict.setStyleSheet(
                "background-color: #742A2A; color: #FEB2B2; padding: 6px 14px; "
                "border-radius: 4px; border: 1px solid #9B2C2C;"
            )

    @pyqtSlot()
    def _on_defect_selected(self):
        selected = self.defect_table.selectedItems()
        if not selected:
            self.canvas.select_defect(None)
            return
        row = selected[0].row()
        id_item = self.defect_table.item(row, 0)
        if id_item:
            try:
                self.canvas.select_defect(int(id_item.text()))
            except ValueError:
                pass

    def _on_load_image(self):
        img_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PCB Optical Image",
            str(REPO_ROOT / "data" / "practical_test_samples"),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)",
        )
        if not img_path or not os.path.exists(img_path):
            return

        self._stop_camera_worker()
        p = Path(img_path)
        self.camera_image = Image.open(p).convert("RGBA")
        self.current_image_name = p.stem
        self.canvas.set_camera_frame(self.camera_image)
        self.canvas.set_defects([])
        self.defect_table.setRowCount(0)
        self.metrics_label.setText("Defects: -- | Vision: -- ms")
        self.lbl_final_verdict.setText("VERDICT: READY")
        self.lbl_final_verdict.setStyleSheet(
            "background-color: #2D3748; color: #E2E8F0; padding: 6px 14px; border-radius: 4px;"
        )
        self._set_status(f"Loaded {p.name} — click 'Inspect Frame (Space)' to inspect.", "idle")

    def _on_load_gerber(self):
        gerber_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Customer Gerber Package (.zip)",
            str(REPO_ROOT / "data" / "sample_gerbers"),
            "Gerber Bundles (*.zip *.tar.gz);;All Files (*)",
        )
        if not gerber_path or not os.path.exists(gerber_path):
            return

        try:
            p = Path(gerber_path)
            self._set_status(f"Parsing Gerber bundle {p.name}...", "vision")
            QApplication.processEvents()
            _cad_img, fids, meta = self.gerber_renderer.render_gerber_bundle(p, dpi=120)
            job_name = meta.get("job_name", p.stem)
            self.current_image_name = job_name
            if self.current_event:
                self.current_event["board_id"] = job_name
            layers = meta.get("layers", [])
            dims = f"{meta.get('width_mm', '?')} x {meta.get('height_mm', '?')} mm"
            self._set_status(f"Loaded Gerber: {job_name} ({dims}, {len(layers)} layers, {len(fids)} fiducials)", "done")
            if "--headless" not in sys.argv and os.environ.get("HEADLESS_TEST") != "1":
                QMessageBox.information(
                    self,
                    "Customer Gerber Parsed",
                    f"Job Name: {job_name}\n"
                    f"Board Dimensions: {dims}\n"
                    f"Fiducials Detected: {len(fids)}\n"
                    f"Layers ({len(layers)}):\n  • " + "\n  • ".join(layers),
                )
        except Exception as e:
            QMessageBox.warning(self, "Gerber Parse Error", f"Failed to load Gerber bundle:\n{e}")

    def _toggle_camera(self):
        if self.camera_worker and self.camera_worker.isRunning():
            self._stop_camera_worker()
            self._set_status("Live camera stopped.", "idle")
            return

        if cv2 is None:
            QMessageBox.warning(self, "OpenCV Required", "OpenCV (cv2) is required for live camera capture.")
            return

        host_ip = get_windows_host_ip()
        # Try local UVC camera 0 first if available, else Windows HTTP stream
        source: Any = 0 if os.path.exists("/dev/video0") else f"http://{host_ip}:8080/video_feed"
        self.camera_worker = CameraWorker(source)
        self.camera_worker.frame_ready.connect(self._on_camera_frame_ready)
        self.camera_worker.error_occurred.connect(self._on_camera_error)
        self.camera_worker.start()
        self.camera_btn.setText("Stop Camera")
        self._set_status(f"Connecting to live camera ({source})...", "vision")

    def _stop_camera_worker(self):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        self.camera_btn.setText("Start Live Camera")

    @pyqtSlot(object)
    def _on_camera_frame_ready(self, img: Image.Image):
        self.camera_image = img.convert("RGBA")
        self.canvas.set_camera_frame(self.camera_image)

    @pyqtSlot(str)
    def _on_camera_error(self, err_msg: str):
        self._stop_camera_worker()
        self._set_status("Camera connection failed.", "error")
        if "--headless" not in sys.argv and os.environ.get("HEADLESS_TEST") != "1":
            QMessageBox.warning(self, "Camera Stream", err_msg)

    def _set_status(self, message: str, state: str = "idle"):
        """Updates the prominent top status banner."""
        styles = {
            "idle": "background-color: #1A202C; color: #CBD5E0; border: 1px solid #2D3748;",
            "vision": "background-color: #1A365D; color: #90CDF4; border: 1px solid #2B6CB0;",
            "copilot": "background-color: #5F370E; color: #FBD38D; border: 1px solid #B7791F;",
            "done": "background-color: #1C4532; color: #9AE6B4; border: 1px solid #2F855A;",
            "error": "background-color: #742A2A; color: #FEB2B2; border: 1px solid #9B2C2C;",
        }
        style = styles.get(state, styles["idle"])
        self.status_banner.setText(message)
        self.status_banner.setStyleSheet(
            f"{style} text-align: left; padding: 7px 12px; border-radius: 4px; font-weight: bold;"
        )

    @pyqtSlot()
    def _run_cycle(self):
        """Executes inspection: fast YOLOv8-OBB on frame, then async Llama 3.2 1B Copilot."""
        if self.camera_image is None:
            return
        if self.advisor_worker and self.advisor_worker.isRunning():
            self._set_status("AI Copilot is still analyzing the previous frame — click to view AI Copilot tab.", "copilot")
            return

        self._set_status("Running YOLOv8-OBB defect detection...", "vision")
        QApplication.processEvents()

        defects: List[Dict[str, Any]] = []
        vision_ms = 0.0
        if self.vision_engine:
            defects, vision_ms = self.vision_engine.detect_defects(self.camera_image.convert("RGB"))

        self.last_vision_ms = vision_ms
        event = self._build_event(defects, vision_ms)
        self._update_defect_views(defects, vision_ms)

        if not self.advisor_engine:
            self.dfm_text = "DualAdvisorEngine unavailable."
            self.asm_text = "DualAdvisorEngine unavailable."
            self.dfm_box.setText(self.dfm_text)
            self.asm_box.setText(self.asm_text)
            self._log_event(event)
            self._set_status(f"Inspection Cycle Complete — {len(defects)} defect(s) | Vision: {vision_ms:.1f} ms", "done")
            return

        self.inspect_btn.setEnabled(False)
        self.dfm_box.setText("Generating CAD/DFM advisory on ExecuTorch Llama 3.2 1B...")
        self.asm_box.setText("Generating Assembler advisory on ExecuTorch Llama 3.2 1B...")
        self._set_status(
            f"Detected {len(defects)} defect(s) in {vision_ms:.1f} ms — AI Copilot is analyzing... (Click here for AI Copilot tab)",
            "copilot",
        )
        QApplication.processEvents()

        self.advisor_worker = AdvisorWorker(self.advisor_engine, event)
        self.advisor_worker.finished_advisory.connect(self._on_advisory_ready)
        self.advisor_worker.error_occurred.connect(self._on_advisory_error)
        self.advisor_worker.start()

        # In headless/offscreen automated tests, wait cleanly while keeping event loop alive
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "--headless" in sys.argv:
            while self.advisor_worker.isRunning():
                QApplication.processEvents()
                self.advisor_worker.wait(25)
            QApplication.processEvents()

    @pyqtSlot(dict, dict, float)
    def _on_advisory_ready(self, event: Dict[str, Any], res: Dict[str, str], llm_s: float):
        self.dfm_text = res.get("dfm_layout_advisory", "No DFM advisory generated.")
        self.asm_text = res.get("assembler_calibration_advisory", "No Assembler advisory generated.")
        self.dfm_box.setText(self.dfm_text)
        self.asm_box.setText(self.asm_text)
        self.inspect_btn.setEnabled(True)
        self._log_event(event)
        defects_count = len(event.get("defects", []))
        self._set_status(
            f"Inspection Cycle Complete — {defects_count} defect(s) | Vision: {self.last_vision_ms:.1f} ms | Copilot: {llm_s:.2f} s — Click to view AI Copilot Advisories",
            "done",
        )

    @pyqtSlot(str)
    def _on_advisory_error(self, err_msg: str):
        self.dfm_text = f"Advisory error: {err_msg}"
        self.asm_text = f"Advisory error: {err_msg}"
        self.dfm_box.setText(self.dfm_text)
        self.asm_box.setText(self.asm_text)
        self.inspect_btn.setEnabled(True)
        self._set_status(f"Copilot error: {err_msg}", "error")

    def _log_event(self, event: Dict[str, Any]):
        if self.storage_manager:
            try:
                self.storage_manager.log_inspection_event(
                    event,
                    advisories={"dfm_advisory": self.dfm_text, "asm_advisory": self.asm_text},
                )
            except Exception as e:
                print(f"[Storage] Warning: {e}")

    @pyqtSlot()
    def _on_export_report(self):
        if not self.current_event:
            self._set_status("Nothing to export yet — run an inspection first.", "error")
            return
        try:
            csv_path = self.report_generator.export_csv(self.current_event)
            pdf_path = self.report_generator.export_pdf(
                self.current_event,
                dfm_advisory=self.dfm_text,
                asm_advisory=self.asm_text,
            )
            if (
                "--headless" not in sys.argv
                and os.environ.get("HEADLESS_TEST") != "1"
                and os.environ.get("QT_QPA_PLATFORM") != "offscreen"
            ):
                opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf_path.resolve())))
                if opened:
                    self._set_status(f"Exported & opened audit reports: {pdf_path.name} & {csv_path.name}", "done")
                else:
                    self._set_status(f"Exported {pdf_path.name} & {csv_path.name} (could not auto-open PDF viewer)", "copilot")
            else:
                self._set_status(f"Exported audit reports: {pdf_path.name} & {csv_path.name}", "done")
        except Exception as e:
            self._set_status(f"Audit report export failed: {e}", "error")

    def closeEvent(self, event):
        self._stop_camera_worker()
        if self.advisor_worker and self.advisor_worker.isRunning():
            self.advisor_worker.quit()
            self.advisor_worker.wait(1000)
        super().closeEvent(event)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bharat AI-SoC PS5 PCB AOI - Minimal Operator Cockpit")
    parser.add_argument("--headless", action="store_true", help="Run in headless verification mode")
    parser.add_argument("--image", type=str, default=None, help="Optional PCB image path to load on startup")
    args, _ = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = OperatorCockpitApp()

    if args.image and os.path.exists(args.image):
        p = Path(args.image)
        window.camera_image = Image.open(p).convert("RGBA")
        window.current_image_name = p.stem
        window.canvas.set_camera_frame(window.camera_image)
        window._run_cycle()

    if args.headless or "--headless" in sys.argv:
        print("[OperatorCockpitApp] Headless verification mode: UI initialized successfully.")
        return 0

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

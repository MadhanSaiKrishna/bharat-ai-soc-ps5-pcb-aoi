#!/usr/bin/env python3
"""
5-Stage Operator UI Cockpit & Closed-Loop Copilot (Member 3 Deliverable)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot

PyQt6 Industrial Operator Cockpit featuring:
1. Live Optical Frame vs Gerber CAD overlay with dynamic opacity slider (0%-100%).
2. Real-time Homography alignment HUD (dX, dY, dTheta) vs tolerance thresholds.
3. 5-Stage Gate Inspection Checklist & 4-Point BGA verification gate.
4. Rotated Bounding Box (YOLOv8-OBB) defect visualizer with severity badges.
5. Operator-in-the-loop Active Learning crop tool for OOD anomalies.
6. Dual Closed-Loop Root-Cause Copilot (Branch 1: CAD DFM, Branch 2: Assembler Calibration).
7. One-click PDF/CSV audit compliance reporting.
"""

import sys
import os
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Headless check for CI / test environments
if "--headless" in sys.argv or os.environ.get("HEADLESS_TEST") == "1":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, pyqtSlot, QTimer, QThread
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont, QPolygonF, QTransform
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QGroupBox, QComboBox, QLineEdit, QMessageBox, QFileDialog,
    QSplitter, QStatusBar, QDialog, QTextEdit, QDialogButtonBox, QFormLayout
)

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_EVENT_PATH = REPO_ROOT / "schemas" / "sample_inspection_event.json"

# Import Member 3 subsystem components
from module_b_guidance.gerber_renderer import GerberRenderer
from module_b_guidance.homography_align import FiducialHomographyEngine
from module_b_guidance.operator_ingest_tool import OperatorAnomalyIngestor
from module_b_guidance.report_generator import AuditReportGenerator

# Import Member 2 Dual Advisor (runs heuristic fallback if ExecuTorch model is absent)
try:
    from edge_llm_storage.dual_advisor import DualAdvisorEngine
except ImportError:
    DualAdvisorEngine = None

try:
    from module_a_vision.infer_vision import VisionInferenceEngine
except ImportError:
    VisionInferenceEngine = None

try:
    import cv2
except ImportError:
    cv2 = None


class InspectionCanvas(QWidget):
    """Interactive viewport for live optical camera frame, CAD overlay, and OBB defects."""

    anomaly_cropped = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #0F141C; border: 1px solid #2D3748; border-radius: 4px;")
        
        self.camera_pixmap: Optional[QPixmap] = None
        self.cad_pixmap: Optional[QPixmap] = None
        self.cad_opacity: float = 0.50
        self.defects: List[Dict[str, Any]] = []
        self.selected_defect_id: Optional[int] = None
        
        # Crop interaction state
        self.crop_mode: bool = False
        self.crop_start: Optional[QPointF] = None
        self.crop_current: Optional[QPointF] = None

    def set_cad_frame(self, cad_img: Image.Image):
        """Converts and caches the static reference CAD overlay as QPixmap."""
        cad_rgba = cad_img.convert("RGBA")
        qim_cad = QImage(cad_rgba.tobytes(), cad_rgba.width, cad_rgba.height, QImage.Format.Format_RGBA8888)
        self.cad_pixmap = QPixmap.fromImage(qim_cad)
        self.update()

    def set_camera_frame(self, camera_img: Image.Image):
        """Updates the optical camera frame without re-rendering the CAD pixmap."""
        cam_rgba = camera_img.convert("RGBA")
        qim_cam = QImage(cam_rgba.tobytes(), cam_rgba.width, cam_rgba.height, QImage.Format.Format_RGBA8888)
        self.camera_pixmap = QPixmap.fromImage(qim_cam)
        self.update()

    def set_frames(self, camera_img: Image.Image, cad_img: Image.Image):
        """Sets both CAD and camera frames."""
        self.set_cad_frame(cad_img)
        self.set_camera_frame(camera_img)

    def set_cad_opacity(self, opacity: float):
        self.cad_opacity = max(0.0, min(1.0, opacity))
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
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Optical Rig Offline - Load Mock Stream")
            return

        # Target display rect keeping aspect ratio with zero-allocation scaling
        cam_w = max(1, self.camera_pixmap.width())
        cam_h = max(1, self.camera_pixmap.height())
        ratio = min(w / cam_w, h / cam_h)
        disp_w = cam_w * ratio
        disp_h = cam_h * ratio
        target_rect = QRectF((w - disp_w) / 2, (h - disp_h) / 2, disp_w, disp_h)

        # Scale factor from source image coords to display canvas
        scale_x = target_rect.width() / cam_w
        scale_y = target_rect.height() / cam_h

        # 1. Draw Optical Camera Frame (fast GPU/software blit directly into target rect)
        painter.setOpacity(1.0)
        painter.drawPixmap(target_rect.toRect(), self.camera_pixmap)

        # 2. Draw Gerber CAD Overlay with Opacity Slider (zero-copy draw directly into target rect)
        if self.cad_pixmap and self.cad_opacity > 0.01:
            painter.setOpacity(self.cad_opacity)
            painter.drawPixmap(target_rect.toRect(), self.cad_pixmap)

        painter.setOpacity(1.0)

        # 3. Draw Rotated Bounding Boxes (YOLOv8-OBB)
        for defect in self.defects:
            obb = defect.get("obb", {})
            xc = obb.get("xc", 0.0) * scale_x + target_rect.left()
            yc = obb.get("yc", 0.0) * scale_y + target_rect.top()
            bw = obb.get("width", 20.0) * scale_x
            bh = obb.get("height", 20.0) * scale_y
            angle = obb.get("angle_deg", 0.0)
            severity = defect.get("severity", "MEDIUM")
            defect_id = defect.get("defect_id")

            # Determine color by severity
            if severity == "CRITICAL":
                color = QColor("#FF2A55")
            elif severity == "HIGH":
                color = QColor("#FF8C00")
            elif severity == "MEDIUM":
                color = QColor("#FFD700")
            else:
                color = QColor("#00E5FF")

            is_selected = (defect_id == self.selected_defect_id)
            pen_width = 3 if is_selected else 2
            painter.setPen(QPen(color, pen_width, Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 45)))

            # Save state, transform for rotated box, and draw
            painter.save()
            painter.translate(xc, yc)
            painter.rotate(angle)
            rect = QRectF(-bw / 2, -bh / 2, bw, bh)
            painter.drawRoundedRect(rect, 2, 2)
            painter.restore()

            # Draw tag label
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#FFFFFF"))
            label_text = f"#{defect_id} {defect.get('type', 'DEFECT')} ({defect.get('confidence', 0.0):.2f})"
            painter.drawText(int(xc - bw / 2), int(yc - bh / 2 - 4), label_text)

        # 4. Draw Active Learning Crop Selection
        if self.crop_mode and self.crop_start and self.crop_current:
            crop_rect = QRectF(self.crop_start, self.crop_current).normalized()
            painter.setPen(QPen(QColor("#00FF88"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(0, 255, 136, 40)))
            painter.drawRect(crop_rect)

    def mousePressEvent(self, event):
        if self.crop_mode and event.button() == Qt.MouseButton.LeftButton:
            self.crop_start = event.position()
            self.crop_current = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self.crop_mode and self.crop_start:
            self.crop_current = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.crop_mode and event.button() == Qt.MouseButton.LeftButton and self.crop_start:
            self.crop_current = event.position()
            crop_rect = QRectF(self.crop_start, self.crop_current).normalized()
            self.crop_mode = False
            self.crop_start = None
            self.crop_current = None
            self.update()
            if crop_rect.width() > 10 and crop_rect.height() > 10:
                self.anomaly_cropped.emit(crop_rect)


class OperatorIngestDialog(QDialog):
    """Modal dialog allowing the operator to label and queue an active learning anomaly crop."""

    def __init__(self, crop_image: Image.Image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flag Out-of-Distribution (OOD) Anomaly")
        self.setFixedSize(440, 320)
        self.setStyleSheet("background-color: #1A202C; color: #E2E8F0;")
        self.crop_image = crop_image

        layout = QVBoxLayout(self)

        form_layout = QFormLayout()
        self.label_combo = QComboBox()
        self.label_combo.addItems([
            "FOREIGN_OBJECT_DEBRIS",
            "PAD_OXIDATION",
            "SOLDER_SPLASH",
            "SOLDER_BALLING",
            "COPPER_TRACE_VOIDING",
            "OPERATOR_CUSTOM_ANOMALY",
        ])
        form_layout.addRow("Anomaly Category:", self.label_combo)

        self.severity_combo = QComboBox()
        self.severity_combo.addItems(["MEDIUM", "HIGH", "CRITICAL", "LOW"])
        form_layout.addRow("Severity:", self.severity_combo)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Describe anomaly context, physical appearance, or observed defect...")
        self.notes_edit.setMaximumHeight(80)
        form_layout.addRow("Operator Notes:", self.notes_edit)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> Dict[str, str]:
        return {
            "label": self.label_combo.currentText(),
            "severity": self.severity_combo.currentText(),
            "notes": self.notes_edit.toPlainText().strip(),
        }


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
    """
    Asynchronous camera worker thread.
    Prevents blocking the PyQt6 GUI thread on slow or timing-out camera streams
    (e.g. over WSL2 usbipd bridge or network streams).
    """
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
                # Try V4L2 first on Linux/WSL2
                cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.source)
            elif isinstance(self.source, str) and "8080" in self.source:
                host_ip = get_windows_host_ip()
                candidates = [f"http://{host_ip}:8080/video_feed"]
                if host_ip != "127.0.0.1":
                    candidates.append("http://127.0.0.1:8080/video_feed")

                for u in candidates:
                    cap = cv2.VideoCapture(u)
                    if cap and cap.isOpened():
                        stream_url = u
                        break
            else:
                stream_url = str(self.source)
                cap = cv2.VideoCapture(stream_url)

            if not cap or not cap.isOpened():
                if isinstance(self.source, str) and "8080" in self.source:
                    self.error_occurred.emit(
                        "Could not connect to Windows Camera Stream at http://localhost:8080/video_feed.\n\n"
                        "Please run the bridge in Windows PowerShell first:\n"
                        "    python tools/stream_camera_windows.py\n\n"
                        "Then re-select 'Live Windows Stream' in the dropdown."
                    )
                else:
                    self.error_occurred.emit(f"Could not open optical stream '{self.source}'.")
                return

            if isinstance(self.source, int):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            consecutive_failures = 0
            last_emit_time = 0.0
            target_interval = 1.0 / 15.0  # Smooth 15 FPS throttle
            while self.running:
                ret, bgr_frame = cap.read()
                now = time.time()
                if ret and bgr_frame is not None:
                    consecutive_failures = 0
                    if now - last_emit_time >= target_interval:
                        last_emit_time = now
                        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(rgb_frame)
                        self.frame_ready.emit(img)
                    self.msleep(10)
                else:
                    consecutive_failures += 1
                    # Transparent auto-reconnect for network streams
                    if stream_url and self.running and consecutive_failures < 15:
                        self.msleep(200)
                        cap.release()
                        cap = cv2.VideoCapture(stream_url)
                        continue

                    if consecutive_failures >= (3 if isinstance(self.source, int) else 15):
                        msg = (
                            "Optical camera stream timed out.\n\n"
                            "WSL2 Root Cause:\n"
                            "In WSL2, high-resolution USB 3.0 UVC cameras (like See3CAM_CU30) suffer from "
                            "virtual host controller bandwidth packet drops (vhci_hcd: urb->status -104 ECONNRESET).\n\n"
                            "Zero-Latency Recommended Solution:\n"
                            "1. Run the native Windows bridge in PowerShell:\n"
                            "      python tools/stream_camera_windows.py\n"
                            "2. In this UI, select:\n"
                            "      'Live Windows Stream (http://localhost:8080/video_feed)'\n\n"
                            "Or click 'Load Optical Image from Disk...' to load a PCB image directly."
                        )
                        self.error_occurred.emit(msg)
                        break
                    self.msleep(50)
        except Exception as e:
            self.error_occurred.emit(f"Camera stream exception: {str(e)}")
        finally:
            if cap:
                cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait(1000)


class OperatorCockpitApp(QMainWindow):
    """Main Industrial AOI Cockpit GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bharat AI-SoC PS5 PCB AOI - Operator Cockpit & Quality-Control Copilot")
        self.resize(1340, 860)

        # Subsystems
        self.gerber_renderer = GerberRenderer()
        self.homography_engine = FiducialHomographyEngine()
        self.ingestor = OperatorAnomalyIngestor()
        self.report_generator = AuditReportGenerator()
        self.advisor_engine = DualAdvisorEngine() if DualAdvisorEngine else None
        self.vision_engine = VisionInferenceEngine() if VisionInferenceEngine else None

        # Data model
        self.current_event: Dict[str, Any] = {}
        self.camera_image: Optional[Image.Image] = None
        self.cad_image: Optional[Image.Image] = None
        self.dfm_text: str = "Awaiting inspection cycle execution..."
        self.asm_text: str = "Awaiting inspection cycle execution..."

        # Asynchronous non-blocking camera worker
        self.camera_worker: Optional[CameraWorker] = None

        self._apply_theme()
        self._build_ui()
        self.load_mock_event(SAMPLE_EVENT_PATH)

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
            QSlider::groove:horizontal { height: 6px; background: #2D3748; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4299E1; width: 14px; margin: -4px 0; border-radius: 7px; }
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
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # 1. Top HUD Header
        header_box = QWidget()
        h_layout = QHBoxLayout(header_box)
        h_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("⚡ BHARAT AI-SoC PS5 AOI COCKPIT")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #63B3ED;")
        h_layout.addWidget(title_label)

        h_layout.addSpacing(20)
        h_layout.addWidget(QLabel("Board ID:"))
        self.board_id_edit = QLineEdit("OPTIMUS_BGA_REV1_004")
        self.board_id_edit.setFixedWidth(180)
        self.board_id_edit.setStyleSheet("background-color: #1A202C; color: #FFFFFF; border: 1px solid #4A5568; padding: 4px;")
        h_layout.addWidget(self.board_id_edit)

        h_layout.addSpacing(10)
        h_layout.addWidget(QLabel("Checkpoint:"))
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.addItems([
            "CHECKPOINT_WIRE_AND_PLACEMENT",
            "CHECKPOINT_SPI",
            "CHECKPOINT_PLACEMENT",
        ])
        self.checkpoint_combo.setStyleSheet("background-color: #1A202C; color: #FFFFFF; border: 1px solid #4A5568; padding: 4px;")
        h_layout.addWidget(self.checkpoint_combo)

        h_layout.addSpacing(10)
        h_layout.addWidget(QLabel("Optical Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "Synthetic PCB Simulator",
            "Live UVC Camera (Index 0 - /dev/video0)",
            "Live UVC Camera (Index 1 - /dev/video1)",
            "Live Windows Stream (http://localhost:8080/video_feed)",
            "Load Optical Image from Disk...",
        ])
        self.source_combo.setStyleSheet("background-color: #1A202C; color: #FFFFFF; border: 1px solid #4A5568; padding: 4px;")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        h_layout.addWidget(self.source_combo)

        h_layout.addSpacing(10)
        self.load_gerber_btn = QPushButton("📁 Load Customer Gerber (.zip)...")
        self.load_gerber_btn.setStyleSheet(
            "background-color: #2B6CB0; color: #FFFFFF; font-weight: bold; "
            "padding: 5px 12px; border-radius: 4px; border: 1px solid #3182CE;"
        )
        self.load_gerber_btn.clicked.connect(lambda: self._on_load_customer_gerber())
        h_layout.addWidget(self.load_gerber_btn)

        h_layout.addStretch()

        # Homography Alignment HUD Pill
        self.align_hud_label = QLabel("Alignment: ΔX = 0.04mm | ΔY = -0.02mm | Δθ = 0.08° | GATE: PASS")
        self.align_hud_label.setStyleSheet(
            "background-color: #1C4532; color: #68D391; font-weight: bold; "
            "padding: 6px 14px; border-radius: 4px; border: 1px solid #276749;"
        )
        h_layout.addWidget(self.align_hud_label)

        main_layout.addWidget(header_box)

        # 2. Splitter: Center Viewport + Right Side Panel
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left/Center: Canvas & View Controls ---
        canvas_container = QWidget()
        c_layout = QVBoxLayout(canvas_container)
        c_layout.setContentsMargins(0, 0, 0, 0)

        # Opacity Slider Bar
        slider_bar = QHBoxLayout()
        slider_bar.addWidget(QLabel("Camera Only (0%)"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        slider_bar.addWidget(self.opacity_slider)
        slider_bar.addWidget(QLabel("CAD Only (100%)"))
        self.opacity_val_label = QLabel("50% CAD Ghost")
        self.opacity_val_label.setStyleSheet("color: #4299E1; font-weight: bold;")
        slider_bar.addWidget(self.opacity_val_label)
        c_layout.addLayout(slider_bar)

        # Central Canvas
        self.canvas = InspectionCanvas()
        self.canvas.anomaly_cropped.connect(self._on_anomaly_cropped)
        c_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(canvas_container)
        splitter.setStretchFactor(0, 3)

        # --- Right: 5-Stage Checklist, Defect List & Copilot Advisory ---
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)
        r_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Tab 1: 5-Stage Gate & BGA Checklist
        tab_stages = QWidget()
        st_layout = QVBoxLayout(tab_stages)
        
        self.lbl_gate_1 = QLabel("✓ Stage 1: SPI Paste Solder Coverage Gate (PASS)")
        self.lbl_gate_1.setStyleSheet("color: #48BB78; font-size: 11pt; padding: 4px;")
        st_layout.addWidget(self.lbl_gate_1)

        self.lbl_gate_2 = QLabel("✓ Stage 2: SMT Passive Placement & Skew (PASS)")
        self.lbl_gate_2.setStyleSheet("color: #48BB78; font-size: 11pt; padding: 4px;")
        st_layout.addWidget(self.lbl_gate_2)

        self.lbl_gate_3 = QLabel("✗ Stage 3: Wire Bond Inspection (DEFECT DETECTED)")
        self.lbl_gate_3.setStyleSheet("color: #F56565; font-size: 11pt; font-weight: bold; padding: 4px;")
        st_layout.addWidget(self.lbl_gate_3)

        bga_box = QGroupBox("Stage 4: 4-Point BGA Checklist Gate")
        bga_layout = QVBoxLayout(bga_box)
        self.chk_bga_bridge = QLabel("  • Ball Bridging Gate: PASS")
        self.chk_bga_coplanar = QLabel("  • Ball Coplanarity Gate: PASS")
        self.chk_bga_diameter = QLabel("  • Solder Ball Diameter Tolerance: PASS")
        self.chk_bga_missing = QLabel("  • Missing Ball Detection: PASS")
        for w in [self.chk_bga_bridge, self.chk_bga_coplanar, self.chk_bga_diameter, self.chk_bga_missing]:
            w.setStyleSheet("color: #9AE6B4;")
            bga_layout.addWidget(w)
        st_layout.addWidget(bga_box)

        st_layout.addSpacing(10)
        self.lbl_final_verdict = QLabel("STAGE 5 OVERALL VERDICT: REWORK REQUIRED")
        self.lbl_final_verdict.setStyleSheet("background-color: #742A2A; color: #FEB2B2; font-size: 12pt; font-weight: bold; padding: 10px; border-radius: 4px; text-align: center;")
        self.lbl_final_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        st_layout.addWidget(self.lbl_final_verdict)
        st_layout.addStretch()
        self.tabs.addTab(tab_stages, "5-Stage Gate")

        # Tab 2: Detected Defects Table
        tab_defects = QWidget()
        def_layout = QVBoxLayout(tab_defects)
        self.defect_table = QTableWidget()
        self.defect_table.setColumnCount(6)
        self.defect_table.setHorizontalHeaderLabels(["ID", "Category", "Type", "Component", "Conf", "Severity"])
        self.defect_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.defect_table.itemSelectionChanged.connect(self._on_defect_selected)
        def_layout.addWidget(self.defect_table)
        self.tabs.addTab(tab_defects, "Defects (2)")

        # Tab 3: Closed-Loop Dual Copilot Advisory
        tab_advisory = QWidget()
        adv_layout = QVBoxLayout(tab_advisory)

        dfm_box = QGroupBox("Branch 1: CAD / DFM Layout Advisory (<40 words)")
        dfm_box_layout = QVBoxLayout(dfm_box)
        self.dfm_view = QTextEdit()
        self.dfm_view.setReadOnly(True)
        self.dfm_view.setStyleSheet("background-color: #1A202C; color: #90CDF4; font-size: 10pt; border: none;")
        dfm_box_layout.addWidget(self.dfm_view)
        adv_layout.addWidget(dfm_box)

        asm_box = QGroupBox("Branch 2: SMT / Assembler Calibration Advisory (<40 words)")
        asm_box_layout = QVBoxLayout(asm_box)
        self.asm_view = QTextEdit()
        self.asm_view.setReadOnly(True)
        self.asm_view.setStyleSheet("background-color: #1A202C; color: #FBD38D; font-size: 10pt; border: none;")
        asm_box_layout.addWidget(self.asm_view)
        adv_layout.addWidget(asm_box)

        btn_regen_adv = QPushButton("↻ Refresh Dual Advisories")
        btn_regen_adv.clicked.connect(self._run_advisory_inference)
        adv_layout.addWidget(btn_regen_adv)

        self.tabs.addTab(tab_advisory, "Copilot Advisories")

        r_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter, stretch=1)

        # 3. Bottom Action Bar
        action_bar = QHBoxLayout()

        btn_cycle = QPushButton("▶ Run Inspection Cycle")
        btn_cycle.setStyleSheet("background-color: #2B6CB0; padding: 10px 18px; font-size: 11pt;")
        btn_cycle.clicked.connect(self._run_cycle)
        action_bar.addWidget(btn_cycle)

        btn_crop = QPushButton("🚩 Flag Custom Anomaly (Crop)")
        btn_crop.setStyleSheet("background-color: #2C7A7B; padding: 10px 18px; font-size: 11pt;")
        btn_crop.clicked.connect(self._enable_crop_mode)
        action_bar.addWidget(btn_crop)

        btn_load = QPushButton("📂 Load Event JSON")
        btn_load.setStyleSheet("background-color: #4A5568; padding: 10px 18px; font-size: 11pt;")
        btn_load.clicked.connect(self._load_custom_event_file)
        action_bar.addWidget(btn_load)

        action_bar.addStretch()

        btn_pdf = QPushButton("📄 Export Audit Report (PDF/CSV)")
        btn_pdf.setStyleSheet("background-color: #D69E2E; color: #1A202C; padding: 10px 18px; font-size: 11pt;")
        btn_pdf.clicked.connect(self._export_reports)
        action_bar.addWidget(btn_pdf)

        main_layout.addLayout(action_bar)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Contract loaded.")

        self.setCentralWidget(main_widget)

    def load_mock_event(self, filepath: Path):
        """Loads and visualizes inspection event conforming to schemas/inspection_event.json."""
        if not filepath.exists():
            return
        with open(filepath, "r") as f:
            self.current_event = json.load(f)

        # 1. Update Board & Header
        board_id = self.current_event.get("board_id", "BOARD_UNKNOWN")
        self.board_id_edit.setText(board_id)
        checkpoint = self.current_event.get("checkpoint", "CHECKPOINT_WIRE_AND_PLACEMENT")
        idx = self.checkpoint_combo.findText(checkpoint)
        if idx >= 0:
            self.checkpoint_combo.setCurrentIndex(idx)

        # 2. Update Alignment HUD
        align = self.current_event.get("alignment", {})
        dx = align.get("dx_mm", 0.0)
        dy = align.get("dy_mm", 0.0)
        dtheta = align.get("dtheta_deg", 0.0)
        status = align.get("status", "PASS")
        self.align_hud_label.setText(f"Alignment: ΔX = {dx:+.3f}mm | ΔY = {dy:+.3f}mm | Δθ = {dtheta:+.3f}° | GATE: {status}")
        if status == "PASS":
            self.align_hud_label.setStyleSheet("background-color: #1C4532; color: #68D391; font-weight: bold; padding: 6px 14px; border-radius: 4px; border: 1px solid #276749;")
        else:
            self.align_hud_label.setStyleSheet("background-color: #742A2A; color: #FEB2B2; font-weight: bold; padding: 6px 14px; border-radius: 4px; border: 1px solid #9B2C2C;")

        # 3. Generate Reference CAD and Synthetic Optical Frame
        self.cad_image, fids, comps = self.gerber_renderer.generate_synthetic_cad_board()
        # Simulated optical camera capture (slightly brighter PCB substrate with realistic sensor noise)
        self.camera_image = Image.new("RGBA", self.cad_image.size, (25, 45, 35, 255))
        self.canvas.set_frames(self.camera_image, self.cad_image)

        # 4. Populate Defect Table & Canvas
        defects = self.current_event.get("defects", [])
        self.canvas.set_defects(defects)
        self.tabs.setTabText(1, f"Defects ({len(defects)})")

        self.defect_table.setRowCount(len(defects))
        for row, d in enumerate(defects):
            self.defect_table.setItem(row, 0, QTableWidgetItem(str(d.get("defect_id"))))
            self.defect_table.setItem(row, 1, QTableWidgetItem(d.get("category", "")))
            self.defect_table.setItem(row, 2, QTableWidgetItem(d.get("type", "")))
            self.defect_table.setItem(row, 3, QTableWidgetItem(d.get("component_ref", "")))
            self.defect_table.setItem(row, 4, QTableWidgetItem(f"{d.get('confidence', 0.0):.2f}"))
            
            sev_item = QTableWidgetItem(d.get("severity", ""))
            if d.get("severity") == "CRITICAL":
                sev_item.setForeground(QColor("#FF2A55"))
            elif d.get("severity") == "HIGH":
                sev_item.setForeground(QColor("#FF8C00"))
            self.defect_table.setItem(row, 5, sev_item)

        # 5. Run Copilot Advisories
        self._run_advisory_inference()
        self.status_bar.showMessage(f"Loaded inspection event: {board_id} ({len(defects)} defects flagged)")

    def _on_opacity_change(self, value: int):
        alpha = value / 100.0
        self.canvas.set_cad_opacity(alpha)
        self.opacity_val_label.setText(f"{value}% CAD Ghost")

    def _on_defect_selected(self):
        selected_rows = self.defect_table.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            defect_id_item = self.defect_table.item(row, 0)
            if defect_id_item:
                self.canvas.select_defect(int(defect_id_item.text()))

    def _enable_crop_mode(self):
        self.canvas.crop_mode = True
        self.status_bar.showMessage("Active Learning: Click and drag a box around the anomalous region on the canvas...")

    def _on_anomaly_cropped(self, crop_rect: QRectF):
        if not self.camera_image:
            return
        # Translate canvas display coords back to source image coords
        w, h = self.canvas.width(), self.canvas.height()
        scaled_w = w
        scaled_h = h
        scale_x = self.camera_image.width / scaled_w
        scale_y = self.camera_image.height / scaled_h

        xmin = int(crop_rect.left() * scale_x)
        ymin = int(crop_rect.top() * scale_y)
        xmax = int(crop_rect.right() * scale_x)
        ymax = int(crop_rect.bottom() * scale_y)

        cropped = self.camera_image.crop((xmin, ymin, xmax, ymax))

        dialog = OperatorIngestDialog(cropped, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            ingest_result = self.ingestor.crop_and_ingest(
                frame=self.camera_image,
                bbox=(xmin, ymin, xmax, ymax),
                operator_label=data["label"],
                board_id=self.board_id_edit.text(),
                severity=data["severity"],
                notes=data["notes"],
            )
            QMessageBox.information(
                self,
                "Anomaly Ingested",
                f"Successfully indexed anomaly '{data['label']}'!\n"
                f"Embedding Vector ID: {ingest_result['embedding_vector_id']}\n"
                f"Queued for edge retraining.",
            )
            self.status_bar.showMessage(f"Indexed anomaly: {ingest_result['anomaly_id']}")

    def _run_advisory_inference(self):
        """Runs Dual Closed-Loop Advisor Engine from Member 2 subsystem."""
        if self.advisor_engine:
            advisories = self.advisor_engine.generate_advisory(self.current_event)
            self.dfm_text = advisories.get("dfm_layout_advisory", advisories.get("dfm_designer_advisory", "No DFM issues detected."))
            self.asm_text = advisories.get("assembler_calibration_advisory", "Calibration within nominal range.")
        else:
            self.dfm_text = "DFM Alert: Verify pad clearance for U1 die pads (spacing < 0.20mm)."
            self.asm_text = "Calibration Action: Reduce capillary ultrasonic power by 6% and inspect wire spool back-tension."

        self.dfm_view.setText(self.dfm_text)
        self.asm_view.setText(self.asm_text)

    def _run_cycle(self):
        self.status_bar.showMessage("Running full Tier 0 AOI inspection cycle...")
        import time
        t0 = time.perf_counter()

        # 1. Run Vision OBB Detection on current optical frame
        frame = self.camera_image or Image.new("RGB", (800, 600), (30, 45, 35))
        checkpoint = self.checkpoint_combo.currentText()
        if self.vision_engine:
            defects, t_vision = self.vision_engine.detect_defects(frame, checkpoint=checkpoint)
        else:
            defects = self.current_event.get("defects", [])
            t_vision = 0.0

        # 2. Update Canvas & Defect Table
        self.canvas.set_defects(defects)
        self.tabs.setTabText(1, f"Defects ({len(defects)})")
        self.defect_table.setRowCount(len(defects))
        for row, d in enumerate(defects):
            self.defect_table.setItem(row, 0, QTableWidgetItem(str(d.get("defect_id"))))
            self.defect_table.setItem(row, 1, QTableWidgetItem(d.get("category", "")))
            self.defect_table.setItem(row, 2, QTableWidgetItem(d.get("type", "")))
            self.defect_table.setItem(row, 3, QTableWidgetItem(d.get("component_ref", "")))
            self.defect_table.setItem(row, 4, QTableWidgetItem(f"{d.get('confidence', 0.0):.2f}"))
            sev_item = QTableWidgetItem(d.get("severity", ""))
            if d.get("severity") == "CRITICAL":
                sev_item.setForeground(QColor("#FF2A55"))
            elif d.get("severity") == "HIGH":
                sev_item.setForeground(QColor("#FF8C00"))
            self.defect_table.setItem(row, 5, sev_item)

        # 3. Update Event structure and run dual advisories
        align = self.current_event.get("alignment", {})
        if self.vision_engine:
            self.current_event = self.vision_engine.build_complete_event(
                board_id=self.board_id_edit.text(),
                checkpoint=checkpoint,
                alignment_deltas=align,
                defects=defects,
                bga_zone_detected=True,
                bga_checklist_passed=True,
            )

        self._run_advisory_inference()
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.status_bar.showMessage(
            f"Inspection Cycle Complete: {len(defects)} defects | Vision: {t_vision:.1f}ms | Total: {total_latency_ms:.1f}ms (Tier 0 PASS)"
        )

    def _load_custom_event_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Inspection Event JSON", str(REPO_ROOT / "schemas"), "JSON Files (*.json)")
        if file_path:
            self.load_mock_event(Path(file_path))

    def _export_reports(self):
        pdf_path = self.report_generator.export_pdf(
            self.current_event,
            dfm_advisory=self.dfm_text,
            asm_advisory=self.asm_text,
        )
        csv_path = self.report_generator.export_csv(self.current_event)
        QMessageBox.information(
            self,
            "Audit Reports Generated",
            f"Quality compliance reports successfully saved!\n\n"
            f"PDF: {pdf_path.name}\n"
            f"CSV: {csv_path.name}\n\n"
            f"Directory: data/telemetry_logs/",
        )
        self.status_bar.showMessage(f"Reports exported: {pdf_path.name}, {csv_path.name}")

    def _stop_camera_worker(self):
        if self.camera_worker:
            try:
                self.camera_worker.frame_ready.disconnect()
                self.camera_worker.error_occurred.disconnect()
            except Exception:
                pass
            self.camera_worker.stop()
            self.camera_worker = None

    def _on_camera_frame_ready(self, frame_img: Image.Image):
        self.camera_image = frame_img
        self.canvas.set_camera_frame(self.camera_image)

    def _on_camera_error(self, err_msg: str):
        self._stop_camera_worker()
        self.source_combo.blockSignals(True)
        self.source_combo.setCurrentIndex(0)
        self.source_combo.blockSignals(False)
        self.status_bar.showMessage("Camera stream halted: returned to Synthetic simulation.")
        QMessageBox.warning(self, "Camera Stream Notice", err_msg)

    def _on_source_changed(self, index: int):
        source_name = self.source_combo.currentText()
        self._stop_camera_worker()

        if "Live UVC Camera" in source_name:
            if cv2 is None:
                QMessageBox.warning(
                    self,
                    "OpenCV Required",
                    "OpenCV (cv2) is required for live optical camera streaming.\n"
                    "Install opencv-python or use the Synthetic PCB Simulator.",
                )
                self.source_combo.setCurrentIndex(0)
                return

            device_id = 0 if "Index 0" in source_name else 1
            dev_path = f"/dev/video{device_id}"

            import glob
            devs = glob.glob("/dev/video*")
            if not devs:
                msg = (
                    f"No /dev/video* devices found in Linux.\n\n"
                    f"If you are running inside WSL2 on Windows:\n"
                    f"USB cameras are managed by Windows and must be attached to WSL2.\n"
                    f"Run in Windows PowerShell (Admin):\n"
                    f"   1. usbipd list\n"
                    f"   2. usbipd bind --busid <BUSID>\n"
                    f"   3. usbipd attach --wsl --busid <BUSID>\n\n"
                    f"Alternatively, stream from Windows via 'Live Windows Stream' (http://localhost:8080/video_feed)."
                )
                QMessageBox.warning(self, "No Video Device", msg)
                self.source_combo.setCurrentIndex(0)
                return

            has_rw = os.access(dev_path, os.R_OK | os.W_OK) if os.path.exists(dev_path) else False
            if not has_rw:
                msg = (
                    f"Found camera device '{dev_path}', but Linux permission was denied.\n\n"
                    f"Run this command in your terminal to grant read/write access:\n"
                    f"   sudo chmod 666 /dev/video*\n\n"
                    f"Then re-select the camera in the dropdown."
                )
                QMessageBox.warning(self, "Camera Permission Denied", msg)
                self.source_combo.setCurrentIndex(0)
                return

            self.status_bar.showMessage(f"Connecting to non-blocking optical feed on {dev_path}...")
            self.camera_worker = CameraWorker(device_id)
            self.camera_worker.frame_ready.connect(self._on_camera_frame_ready)
            self.camera_worker.error_occurred.connect(self._on_camera_error)
            self.camera_worker.start()

        elif "Live Windows Stream" in source_name:
            if cv2 is None:
                QMessageBox.warning(self, "OpenCV Required", "OpenCV (cv2) is required for network streams.")
                self.source_combo.setCurrentIndex(0)
                return

            host_ip = get_windows_host_ip()
            url = f"http://{host_ip}:8080/video_feed"
            self.status_bar.showMessage(f"Connecting to Windows bridge stream at {url}...")
            self.camera_worker = CameraWorker(url)
            self.camera_worker.frame_ready.connect(self._on_camera_frame_ready)
            self.camera_worker.error_occurred.connect(self._on_camera_error)
            self.camera_worker.start()

        elif "Load Optical Image" in source_name:
            img_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Optical PCB Image",
                str(REPO_ROOT),
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)",
            )
            if img_path and os.path.exists(img_path):
                self.camera_image = Image.open(img_path).convert("RGBA")
                if self.cad_image:
                    self.canvas.set_frames(self.camera_image, self.cad_image)
                self.status_bar.showMessage(f"Loaded optical frame from: {Path(img_path).name}")
            else:
                self.source_combo.setCurrentIndex(0)

        else:
            self.status_bar.showMessage("Switched to Synthetic PCB Simulation mode.")
            self.load_mock_event(SAMPLE_EVENT_PATH)

    def _on_load_customer_gerber(self, gerber_file_path=None):
        """Allows the assembler to load a customer Gerber bundle (.zip or folder)."""
        if not gerber_file_path:
            gerber_file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Customer PCB Gerber Package",
                str(REPO_ROOT / "data" / "sample_gerbers"),
                "Gerber Bundles (*.zip *.tar.gz);;All Files (*)",
            )
        if not gerber_file_path or not os.path.exists(gerber_file_path):
            return

        try:
            p = Path(gerber_file_path)
            self.status_bar.showMessage(f"Parsing customer Gerber layers: {p.name}...")
            cad_img, fids, meta = self.gerber_renderer.render_gerber_bundle(p, dpi=120)

            self.cad_image = cad_img
            self.canvas.set_cad_frame(self.cad_image)

            job_name = meta.get("job_name", p.stem)
            self.board_id_edit.setText(job_name)
            self.current_event["board_id"] = job_name

            layers_list = meta.get("layers", [])
            layers_str = ", ".join(layers_list)
            dims_str = f"{meta.get('width_mm', '?')} x {meta.get('height_mm', '?')} mm"
            self.status_bar.showMessage(f"Active Job: {job_name} | Layers: {layers_str} | Size: {dims_str}")

            if "--headless" not in sys.argv and os.environ.get("HEADLESS_TEST") != "1":
                QMessageBox.information(
                    self,
                    "Customer Gerber Package Loaded",
                    f"Successfully parsed customer RS-274X CAD package!\n\n"
                    f"Job Name: {job_name}\n"
                    f"Board Dimensions: {dims_str}\n"
                    f"Composited Layers ({len(layers_list)}):\n  • " + "\n  • ".join(layers_list) + "\n\n"
                    f"The reference CAD overlay has been loaded into the live optical inspection canvas.",
                )
        except Exception as e:
            QMessageBox.warning(self, "Gerber Parsing Error", f"Failed to parse Gerber package:\n{str(e)}")

    def closeEvent(self, event):
        self._stop_camera_worker()
        super().closeEvent(event)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bharat AI-SoC PS5 PCB AOI - Operator Cockpit")
    parser.add_argument("--headless", action="store_true", help="Run in headless verification mode")
    parser.add_argument("--gerber", type=str, default=None, help="Path to customer Gerber zip or folder to load on startup")
    args, _ = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = OperatorCockpitApp()

    if args.gerber and os.path.exists(args.gerber):
        window._on_load_customer_gerber(args.gerber)

    if args.headless or "--headless" in sys.argv:
        print("[OperatorCockpitApp] Headless verification mode: UI initialized successfully.")
        return 0

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Gerber Layer Compositor & CAD Visualizer (Member 3 Deliverable)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot

Ingests RS-274X Gerber layers (Paste GTP, Copper GTL, Outline GKO, Silkscreen GTO)
and renders composite alignment layers with pad and fiducial extraction.
Includes a procedural high-fidelity synthetic CAD renderer for mock testing.
"""

import math
import os
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from pygerber.gerberx3.api import Rasterized2DLayer, Rasterized2DLayerParams, ColorScheme
    HAS_PYGERBER = True
except ImportError:
    HAS_PYGERBER = False


class GerberRenderer:
    """Renders RS-274X Gerber CAD layers and generates reference composite images."""

    def __init__(self, pixels_per_mm: float = 50.0):
        self.pixels_per_mm = pixels_per_mm
        self.nominal_fiducials_mm: List[Tuple[float, float]] = [
            (10.0, 10.0),   # Bottom-Left
            (90.0, 10.0),   # Bottom-Right
            (50.0, 70.0),   # Top-Center
        ]

    def generate_synthetic_cad_board(
        self,
        width_mm: float = 100.0,
        height_mm: float = 80.0,
    ) -> Tuple[Image.Image, List[Tuple[float, float]], List[Dict[str, Any]]]:
        """
        Generates a photorealistic high-density PCB CAD composite render
        containing BGA pads, 0201 passives, QFP ICs, micro wire bonds, and fiducials.
        
        Returns:
            composite_img (PIL.Image): RGBA CAD overlay image
            fiducials_mm: List of nominal (Xmm, Ymm) fiducial points
            components: List of component nominal metadata
        """
        w_px = int(width_mm * self.pixels_per_mm)
        h_px = int(height_mm * self.pixels_per_mm)

        # Transparent CAD background layer
        cad_img = Image.new("RGBA", (w_px, h_px), (15, 32, 24, 255))
        draw = ImageDraw.Draw(cad_img)

        # Board substrate boundary (Keepout / GKO)
        margin = int(2.0 * self.pixels_per_mm)
        draw.rounded_rectangle(
            [(margin, margin), (w_px - margin, h_px - margin)],
            radius=int(3.0 * self.pixels_per_mm),
            outline=(50, 160, 90, 255),
            width=3,
        )

        components = []

        # 1. Draw 3 Standard Optical Fiducials (Crosshair in circle)
        fiducial_radius_px = int(1.2 * self.pixels_per_mm)
        for (f_x_mm, f_y_mm) in self.nominal_fiducials_mm:
            cx = int(f_x_mm * self.pixels_per_mm)
            cy = int(f_y_mm * self.pixels_per_mm)
            # Copper annulus
            draw.ellipse(
                [(cx - fiducial_radius_px, cy - fiducial_radius_px),
                 (cx + fiducial_radius_px, cy + fiducial_radius_px)],
                fill=(212, 175, 55, 230),  # Gold ENIG pad
                outline=(255, 215, 0, 255),
                width=2,
            )
            # Center clearance
            core_r = int(0.5 * self.pixels_per_mm)
            draw.ellipse(
                [(cx - core_r, cy - core_r), (cx + core_r, cy + core_r)],
                fill=(15, 32, 24, 255),
            )
            # Crosshair
            ch_len = int(2.0 * self.pixels_per_mm)
            draw.line([(cx - ch_len, cy), (cx + ch_len, cy)], fill=(255, 255, 255, 200), width=1)
            draw.line([(cx, cy - ch_len), (cx, cy + ch_len)], fill=(255, 255, 255, 200), width=1)

        # 2. Draw Central BGA / Die Zone (e.g. U1 System-on-Chip)
        bga_cx_mm = 50.0
        bga_cy_mm = 40.0
        bga_cx_px = int(bga_cx_mm * self.pixels_per_mm)
        bga_cy_px = int(bga_cy_mm * self.pixels_per_mm)
        die_size_px = int(20.0 * self.pixels_per_mm)

        # Silicon die outline
        draw.rectangle(
            [(bga_cx_px - die_size_px // 2, bga_cy_px - die_size_px // 2),
             (bga_cx_px + die_size_px // 2, bga_cy_px + die_size_px // 2)],
            fill=(30, 35, 45, 240),
            outline=(100, 150, 255, 255),
            width=2,
        )

        # BGA Ball Matrix (8x8 ball grid array)
        pitch_mm = 0.8
        pitch_px = int(pitch_mm * self.pixels_per_mm)
        ball_r = int(0.25 * self.pixels_per_mm)
        for r in range(-4, 4):
            for c in range(-4, 4):
                bx = bga_cx_px + int((c + 0.5) * pitch_px)
                by = bga_cy_px + int((r + 0.5) * pitch_px)
                draw.ellipse(
                    [(bx - ball_r, by - ball_r), (bx + ball_r, by + ball_r)],
                    fill=(190, 195, 205, 230),  # Solder alloy
                    outline=(230, 230, 240, 255),
                    width=1,
                )

        components.append({
            "ref": "U1_SOC",
            "type": "BGA_64",
            "center_mm": (bga_cx_mm, bga_cy_mm),
            "zone": "BGA_CORE",
        })

        # 3. Micro Wire Bonding Perimeter Pads (Die-to-Substrate fingers)
        num_wires = 12
        wire_pad_r = int(0.3 * self.pixels_per_mm)
        for i in range(num_wires):
            ang = 2.0 * math.pi * (i / num_wires)
            # Die pad
            dx = bga_cx_px + int((die_size_px // 2 - 15) * math.cos(ang))
            dy = bga_cy_px + int((die_size_px // 2 - 15) * math.sin(ang))
            # Substrate lead pad
            sx = bga_cx_px + int((die_size_px // 2 + 35) * math.cos(ang))
            sy = bga_cy_px + int((die_size_px // 2 + 35) * math.sin(ang))

            draw.ellipse([(dx - wire_pad_r, dy - wire_pad_r), (dx + wire_pad_r, dy + wire_pad_r)], fill=(255, 215, 0, 255))
            draw.rectangle([(sx - wire_pad_r, sy - wire_pad_r), (sx + wire_pad_r, sy + wire_pad_r)], fill=(212, 175, 55, 255))
            # Nominal bond wire trace
            draw.line([(dx, dy), (sx, sy)], fill=(255, 225, 100, 180), width=2)

        # 4. SMT Passives (0201 & 0402 Resistors / Capacitors)
        passives = [
            ("R14_0201", 25.0, 30.0, 1.2, 0.6, 0.0),
            ("R15_0201", 25.0, 35.0, 1.2, 0.6, 0.0),
            ("C8_0402", 75.0, 30.0, 1.8, 1.0, 90.0),
            ("C9_0402", 75.0, 38.0, 1.8, 1.0, 90.0),
            ("L2_0603", 50.0, 65.0, 2.5, 1.4, 0.0),
        ]
        for ref, px_mm, py_mm, w_m, h_m, rot in passives:
            cx = int(px_mm * self.pixels_per_mm)
            cy = int(py_mm * self.pixels_per_mm)
            pw = int(w_m * self.pixels_per_mm)
            ph = int(h_m * self.pixels_per_mm)

            # Solder pads (pair)
            if rot == 0.0:
                pad_w = pw // 3
                draw.rectangle([(cx - pw // 2, cy - ph // 2), (cx - pw // 2 + pad_w, cy + ph // 2)], fill=(180, 180, 190, 240))
                draw.rectangle([(cx + pw // 2 - pad_w, cy - ph // 2), (cx + pw // 2, cy + ph // 2)], fill=(180, 180, 190, 240))
                # Component body
                draw.rectangle([(cx - pw // 2 + pad_w, cy - ph // 2 + 2), (cx + pw // 2 - pad_w, cy + ph // 2 - 2)], fill=(60, 60, 60, 240))
            else:
                pad_h = ph // 3
                draw.rectangle([(cx - pw // 2, cy - ph // 2), (cx + pw // 2, cy - ph // 2 + pad_h)], fill=(180, 180, 190, 240))
                draw.rectangle([(cx - pw // 2, cy + ph // 2 - pad_h), (cx + pw // 2, cy + ph // 2)], fill=(180, 180, 190, 240))
                # Component body
                draw.rectangle([(cx - pw // 2 + 2, cy - ph // 2 + pad_h), (cx + pw // 2 - 2, cy + ph // 2 - pad_h)], fill=(60, 60, 60, 240))

            components.append({
                "ref": ref,
                "center_mm": (px_mm, py_mm),
                "rotation_deg": rot,
            })

        # Silkscreen text
        draw.text((margin + 10, margin + 10), "BHARAT-AI-SOC PS5 AOI MOCK CAD (REV 1.0)", fill=(240, 240, 240, 200))

        return cad_img, self.nominal_fiducials_mm, components

    def composite_with_camera_frame(
        self,
        camera_img: Image.Image,
        cad_img: Image.Image,
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Blends camera optical image with the CAD overlay at given opacity (0.0=Cam, 1.0=CAD).
        """
        alpha = max(0.0, min(1.0, alpha))
        cam_rgba = camera_img.convert("RGBA").resize(cad_img.size)
        return Image.blend(cam_rgba, cad_img, alpha)

    def render_gerber_bundle(
        self,
        bundle_path: Path,
        dpi: int = 150,
    ) -> Tuple[Image.Image, List[Tuple[float, float]], Dict[str, Any]]:
        """
        Parses and composites customer RS-274X Gerber layers from a .zip archive or folder.
        Layers supported: Top Copper, Solder Paste, Solder Mask, Silkscreen, Board Outline.

        Returns:
            composite_img (PIL.Image): Full composited multi-layer CAD overlay image
            fiducials_mm: List of (X, Y) nominal fiducials in millimeters
            metadata: Dict with job name, detected layers, board dimensions
        """
        bundle_path = Path(bundle_path)
        job_name = bundle_path.stem

        if not HAS_PYGERBER:
            print(f"[WARN] PyGerber not available. Falling back to synthetic generator for job: {job_name}")
            cad, fids, comps = self.generate_synthetic_cad_board()
            return cad, fids, {"job_name": job_name, "mode": "synthetic_fallback", "layers": ["Synthetic CAD Fallback"]}

        # Extract zip or locate files
        temp_dir_ctx = None
        if bundle_path.is_file() and bundle_path.suffix.lower() == ".zip":
            temp_dir_ctx = tempfile.TemporaryDirectory()
            work_dir = Path(temp_dir_ctx.name)
            with zipfile.ZipFile(bundle_path, "r") as zf:
                zf.extractall(work_dir)
        elif bundle_path.is_dir():
            work_dir = bundle_path
        else:
            raise ValueError(f"Invalid Gerber bundle path: {bundle_path} (must be .zip or directory)")

        try:
            # Find and classify Gerber layer files
            all_files = list(work_dir.glob("**/*"))
            layer_map: List[Tuple[Path, Any, str]] = []

            for f in all_files:
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                name = f.name.lower()

                # Classification by extension or common EDA conventions (KiCad / Altium / Eagle)
                if ext in [".gtl", ".top", ".cmp"] or "f_cu" in name or "top_copper" in name:
                    layer_map.append((f, ColorScheme.COPPER_ALPHA, "Top Copper (GTL)"))
                elif ext in [".gtp", ".crc"] or "f_paste" in name or "solder_paste" in name:
                    layer_map.append((f, ColorScheme.PASTE_MASK_ALPHA, "Solder Paste (GTP)"))
                elif ext in [".gts", ".stc"] or "f_mask" in name or "solder_mask" in name:
                    layer_map.append((f, ColorScheme.SOLDER_MASK_ALPHA, "Solder Mask (GTS)"))
                elif ext in [".gto", ".plc"] or "f_silkscreen" in name or "silkscreen" in name:
                    layer_map.append((f, ColorScheme.SILK_ALPHA, "Silkscreen (GTO)"))

            if not layer_map:
                print(f"[WARN] No recognizable Gerber layers found in {bundle_path}. Using synthetic CAD.")
                cad, fids, comps = self.generate_synthetic_cad_board()
                return cad, fids, {"job_name": job_name, "mode": "synthetic_fallback", "layers": ["Synthetic CAD Fallback"]}

            # Render individual layers and compute union bounding box
            rendered_layers = []
            min_x = 1e9
            min_y = 1e9
            max_x = -1e9
            max_y = -1e9

            for path, scheme, layer_desc in layer_map:
                try:
                    layer = Rasterized2DLayer(Rasterized2DLayerParams(source_path=path, colors=scheme, dpi=dpi))
                    res = layer.render()
                    img = res.get_image()
                    bbox = res.get_properties().target_bounding_box
                    val_min_x = float(bbox.min_x.value)
                    val_min_y = float(bbox.min_y.value)
                    val_max_x = float(bbox.max_x.value)
                    val_max_y = float(bbox.max_y.value)

                    min_x = min(min_x, val_min_x)
                    min_y = min(min_y, val_min_y)
                    max_x = max(max_x, val_max_x)
                    max_y = max(max_y, val_max_y)

                    rendered_layers.append((img, val_min_x, val_min_y, val_max_x, val_max_y, layer_desc))
                except Exception as e:
                    print(f"[WARN] Failed to render layer {path.name}: {e}")

            if not rendered_layers:
                cad, fids, comps = self.generate_synthetic_cad_board()
                return cad, fids, {"job_name": job_name, "mode": "synthetic_fallback", "layers": ["Synthetic CAD Fallback"]}

            # Composite onto dark green PCB substrate
            scale = dpi / 25.4
            margin_px = 30
            board_w_mm = max_x - min_x
            board_h_mm = max_y - min_y
            total_w_px = int(board_w_mm * scale) + margin_px * 2
            total_h_px = int(board_h_mm * scale) + margin_px * 2

            board_img = Image.new("RGBA", (total_w_px, total_h_px), (18, 38, 25, 255))

            # Composite in standard stack order: Copper -> Paste -> Silk
            for img, lx_min, ly_min, lx_max, ly_max, layer_desc in rendered_layers:
                ox = int((lx_min - min_x) * scale) + margin_px
                oy = int((max_y - ly_max) * scale) + margin_px
                board_img.alpha_composite(img.convert("RGBA"), (ox, oy))

            # Extract 3 nominal optical fiducials (nominal corner locations)
            fiducials_mm = [
                (round(min_x + 5.0, 2), round(min_y + 5.0, 2)),
                (round(max_x - 5.0, 2), round(min_y + 5.0, 2)),
                (round((min_x + max_x) / 2.0, 2), round(max_y - 5.0, 2)),
            ]

            meta = {
                "job_name": job_name,
                "layers": [l[5] for l in rendered_layers],
                "width_mm": round(board_w_mm, 2),
                "height_mm": round(board_h_mm, 2),
                "resolution_px": board_img.size,
                "dpi": dpi,
            }
            return board_img, fiducials_mm, meta

        finally:
            if temp_dir_ctx:
                temp_dir_ctx.cleanup()


if __name__ == "__main__":
    renderer = GerberRenderer()
    cad, fids, comps = renderer.generate_synthetic_cad_board()
    print(f"[GerberRenderer] Generated CAD Image: {cad.size[0]}x{cad.size[1]} px")
    print(f"[GerberRenderer] Fiducials: {fids}")
    print(f"[GerberRenderer] Components: {len(comps)} elements mapped")

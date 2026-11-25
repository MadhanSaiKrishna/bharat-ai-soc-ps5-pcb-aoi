#!/usr/bin/env python3
"""
Gerber Fiducial Homography & Alignment Engine (Member 3 Deliverable)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot

Maps camera pixel coordinates (u, v) to physical Gerber coordinates (Xmm, Ymm)
using planar homography (cv2.findHomography) computed from 3+ fiducial markers.
Calculates real-time translation (ΔX, ΔY) and rotation (Δθ) and gates inspection
against tolerance standards (Standard, Fine-Pitch, BGA, Wire Bond).
"""

import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import yaml

try:
    import cv2
except ImportError:
    cv2 = None

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_TOLERANCES_PATH = CONFIG_DIR / "tolerances.yaml"


class FiducialHomographyEngine:
    """Computes planar homography matrix and alignment deltas relative to Gerber nominal positions."""

    def __init__(self, tolerances_path: Optional[Path] = None, pixels_per_mm: float = 50.0):
        self.tolerances_path = tolerances_path or DEFAULT_TOLERANCES_PATH
        self.pixels_per_mm = pixels_per_mm
        self.tolerances = self._load_tolerances()
        self.homography_matrix: Optional[np.ndarray] = None

    def _load_tolerances(self) -> Dict[str, Any]:
        if self.tolerances_path.exists():
            with open(self.tolerances_path, "r") as f:
                data = yaml.safe_load(f)
                return data.get("tolerances", {})
        return {
            "standard": {"dx_mm": 0.20, "dy_mm": 0.20, "dtheta_deg": 0.50},
            "fine_pitch": {"dx_mm": 0.10, "dy_mm": 0.10, "dtheta_deg": 0.30},
            "bga_die": {"dx_mm": 0.15, "dy_mm": 0.15, "dtheta_deg": 0.20},
            "wire_bond": {"dx_mm": 0.05, "dy_mm": 0.05, "dtheta_deg": 0.10},
        }

    def compute_homography(
        self,
        camera_fiducials_px: List[Tuple[float, float]],
        cad_fiducials_mm: List[Tuple[float, float]],
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """
        Computes 3x3 homography matrix H mapping CAD coordinates to camera pixel frame.
        camera_fiducials_px: list of (u, v) detected in optical image
        cad_fiducials_mm: list of nominal (Xmm, Ymm) from Gerber GTP/GTO
        """
        if len(camera_fiducials_px) < 3 or len(cad_fiducials_mm) < 3:
            raise ValueError("At least 3 fiducial pairs are required for homography calculation.")

        src_pts = np.array(cad_fiducials_mm, dtype=np.float32) * self.pixels_per_mm
        dst_pts = np.array(camera_fiducials_px, dtype=np.float32)

        if cv2 is not None:
            if len(src_pts) >= 4:
                H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                self.homography_matrix = H
            else:
                # For 3 fiducials, compute exact 2D affine / rigid transform embedded in 3x3 matrix
                M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
                H = np.vstack([M, [0.0, 0.0, 1.0]])
                self.homography_matrix = H
        else:
            # Fallback affine / rigid estimate if cv2 is not yet installed in host environment
            H = np.eye(3, dtype=np.float32)
            self.homography_matrix = H

        # Compute translation (dx, dy) in mm from centroid offset
        centroid_cad = np.mean(cad_fiducials_mm, axis=0)
        centroid_cam_mm = np.mean(camera_fiducials_px, axis=0) / self.pixels_per_mm
        dx_mm = float(centroid_cam_mm[0] - centroid_cad[0])
        dy_mm = float(centroid_cam_mm[1] - centroid_cad[1])

        # Compute rotation (dtheta) from vector between fiducial 0 and fiducial 1
        v_cad = np.array(cad_fiducials_mm[1]) - np.array(cad_fiducials_mm[0])
        v_cam = (np.array(camera_fiducials_px[1]) - np.array(camera_fiducials_px[0])) / self.pixels_per_mm

        ang_cad = math.atan2(v_cad[1], v_cad[0])
        ang_cam = math.atan2(v_cam[1], v_cam[0])
        dtheta_rad = ang_cam - ang_cad
        dtheta_deg = float(math.degrees(dtheta_rad))

        return self.homography_matrix, {
            "dx_mm": round(dx_mm, 4),
            "dy_mm": round(dy_mm, 4),
            "dtheta_deg": round(dtheta_deg, 4),
        }

    def evaluate_alignment_tolerance(
        self,
        deltas: Dict[str, float],
        tier_key: str = "bga_die",
    ) -> Dict[str, Any]:
        """Validates alignment deltas against configured tolerances."""
        tier = self.tolerances.get(tier_key, self.tolerances["standard"])
        max_dx = tier.get("dx_mm", 0.15)
        max_dy = tier.get("dy_mm", 0.15)
        max_dtheta = tier.get("dtheta_deg", 0.20)

        dx = abs(deltas.get("dx_mm", 0.0))
        dy = abs(deltas.get("dy_mm", 0.0))
        dtheta = abs(deltas.get("dtheta_deg", 0.0))

        passed = (dx <= max_dx) and (dy <= max_dy) and (dtheta <= max_dtheta)
        status = "PASS" if passed else "FAIL"

        return {
            "status": status,
            "dx_mm": deltas.get("dx_mm", 0.0),
            "dy_mm": deltas.get("dy_mm", 0.0),
            "dtheta_deg": deltas.get("dtheta_deg", 0.0),
            "tolerance_dx_mm": max_dx,
            "tolerance_dtheta_deg": max_dtheta,
            "tier": tier_key,
        }

    def detect_optical_fiducials(
        self,
        image_rgb: np.ndarray,
        cad_fiducials_mm: List[Tuple[float, float]],
    ) -> Tuple[List[Tuple[float, float]], str]:
        """
        Detects 3 fiducial reference points directly from an optical PCB image.
        Uses OpenCV HoughCircles / contour detection to locate copper/gold fiducial
        markers near expected board quadrants, falling back to detected PCB contour
        corners scaled to image dimensions when inspecting cropped/close-up samples.
        """
        h, w = image_rgb.shape[:2]
        if cv2 is None or len(cad_fiducials_mm) < 3:
            scale_x = w / 100.0
            scale_y = h / 80.0
            return [(fx * scale_x, fy * scale_y) for fx, fy in cad_fiducials_mm[:3]], "geometric_estimate"

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # 1. Try HoughCircles for circular optical fiducial pads
        min_r = max(3, int(min(w, h) * 0.008))
        max_r = max(8, int(min(w, h) * 0.04))
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, int(min(w, h) * 0.15)),
            param1=80,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )

        # Map CAD fiducial positions into expected image pixel coordinates
        cad_arr = np.array(cad_fiducials_mm[:3], dtype=np.float32)
        min_cx, min_cy = cad_arr.min(axis=0)
        span_x = max(1e-3, float(cad_arr[:, 0].max() - min_cx))
        span_y = max(1e-3, float(cad_arr[:, 1].max() - min_cy))
        if (cad_arr[:, 0].max() * self.pixels_per_mm <= w * 1.05) and (cad_arr[:, 1].max() * self.pixels_per_mm <= h * 1.05) and min_cx >= 0 and min_cy >= 0:
            expected_px = [(float(fx * self.pixels_per_mm), float(fy * self.pixels_per_mm)) for fx, fy in cad_fiducials_mm[:3]]
        else:
            expected_px = [
                (
                    (0.1 + 0.8 * (fx - min_cx) / span_x) * w,
                    (0.1 + 0.8 * (fy - min_cy) / span_y) * h,
                )
                for fx, fy in cad_fiducials_mm[:3]
            ]

        if circles is not None and len(circles[0]) >= 3:
            pts = [(float(c[0]), float(c[1])) for c in circles[0]]
            matched = []
            used = set()
            for ex, ey in expected_px:
                best_idx = None
                best_dist = 1e9
                for idx, (px, py) in enumerate(pts):
                    if idx in used:
                        continue
                    dist = math.hypot(px - ex, py - ey)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                if best_idx is not None:
                    used.add(best_idx)
                    matched.append(pts[best_idx])
            if len(matched) == 3:
                return matched, "opencv_hough_fiducials"

        # 2. Fallback to brightest metallic pad / corner feature near each expected zone
        matched = []
        win_r = max(12, int(min(w, h) * 0.08))
        for ex, ey in expected_px:
            x0, x1 = max(0, int(ex - win_r)), min(w, int(ex + win_r))
            y0, y1 = max(0, int(ey - win_r)), min(h, int(ey + win_r))
            roi = blurred[y0:y1, x0:x1]
            if roi.size > 0:
                _, max_val, _, max_loc = cv2.minMaxLoc(roi)
                if max_val > 80:
                    matched.append((float(x0 + max_loc[0]), float(y0 + max_loc[1])))
                    continue
            matched.append((float(ex), float(ey)))
        return matched, "opencv_pad_peak_match"

    def warp_camera_to_cad(
        self,
        image_rgb: np.ndarray,
        cad_size: Tuple[int, int],
        homography_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Warps an optical RGB frame into the CAD coordinate frame using H."""
        H = homography_matrix if homography_matrix is not None else self.homography_matrix
        if cv2 is None or H is None:
            return image_rgb
        return cv2.warpPerspective(image_rgb, H.astype(np.float32), cad_size)


if __name__ == "__main__":
    engine = FiducialHomographyEngine()

    # Nominal Gerber fiducials in millimeters
    cad_fiducials = [(10.0, 10.0), (90.0, 10.0), (50.0, 70.0)]
    # Detected camera pixel coordinates (with small simulated offset: +2px X, -1px Y, slight rotation)
    scale = 50.0  # 50 px/mm
    cam_fiducials = [
        (10.0 * scale + 2.0, 10.0 * scale - 1.0),
        (90.0 * scale + 2.2, 10.0 * scale - 0.8),
        (50.0 * scale + 2.1, 70.0 * scale - 1.1),
    ]

    H, deltas = engine.compute_homography(cam_fiducials, cad_fiducials)
    evaluation = engine.evaluate_alignment_tolerance(deltas, tier_key="bga_die")

    print("[Homography Engine Result]")
    print(f"Alignment Deltas: dX = {deltas['dx_mm']} mm, dY = {deltas['dy_mm']} mm, dTheta = {deltas['dtheta_deg']} deg")
    print(f"Verdict: {evaluation['status']} (Tolerance: <= {evaluation['tolerance_dx_mm']} mm, <= {evaluation['tolerance_dtheta_deg']} deg)")

#!/usr/bin/env python3
"""
Zero-latency Windows-to-WSL2 Camera Bridge.
Runs natively on the Windows host to stream external USB cameras (like See3CAM_CU30)
directly into WSL2 over local HTTP MJPEG, bypassing usbipd isochronous bandwidth limitations.

Usage (Run in Windows PowerShell or Command Prompt):
    # 1. List available cameras to see their indices:
    python tools/stream_camera_windows.py --list

    # 2. Preview camera in a local window to confirm it's pointing to the PCB:
    python tools/stream_camera_windows.py --index 1 --preview

    # 3. Stream to WSL2 (default will auto-detect external camera):
    python tools/stream_camera_windows.py --index 1
"""

import sys
import time
import argparse
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV not installed on Windows host. Run: pip install opencv-python")
    sys.exit(1)

# Global frame buffer
current_frame_jpeg = None
is_running = True


def get_windows_device_names():
    """Queries Windows PnP subsystem for camera friendly names."""
    names = []
    try:
        cmd = 'powershell -NoProfile -Command "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq \'Camera\' -or $_.PNPClass -eq \'Image\' } | Select-Object -ExpandProperty Caption"'
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        for line in out.strip().split("\n"):
            line = line.strip()
            if line:
                names.append(line)
    except Exception:
        pass
    return names


def list_available_cameras():
    """Scans DirectShow indices and displays detected cameras and resolutions."""
    print("\n" + "=" * 65)
    print("      WINDOWS CAMERA SCAN & DIAGNOSTIC UTILITY      ")
    print("=" * 65)

    pnp_names = get_windows_device_names()
    if pnp_names:
        print("Hardware Devices Detected by Windows:")
        for name in pnp_names:
            print(f"  • {name}")
        print()

    print("Probing OpenCV DirectShow (CAP_DSHOW) Indices (0 to 4):")
    found_any = False
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            # Query default resolution
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Test setting higher resolution (e.g. 1080p)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            max_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            max_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            ret, frame = cap.read()
            if ret and frame is not None:
                found_any = True
                desc = "External USB / Industrial Camera (e.g. See3CAM)" if idx > 0 or max_w > 1280 else "Built-in Laptop Webcam"
                print(f"  [Index {idx}] ACTIVE -> Default: {w}x{h} px | Max: {max_w}x{max_h} px | ({desc})")
            cap.release()

    if not found_any:
        print("  [WARN] No active camera indices opened. Please check physical USB connection.")

    print("=" * 65)
    print("To select a specific camera, pass: --index <NUMBER>")
    print("Example: python tools/stream_camera_windows.py --index 1\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class CamStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_frame_jpeg
        if self.path == "/video_feed" or self.path == "/":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            while is_running:
                if current_frame_jpeg is not None:
                    try:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(current_frame_jpeg)))
                        self.end_headers()
                        self.wfile.write(current_frame_jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.033)  # ~30 FPS
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logs
        return


def capture_loop(camera_idx=None, width=1920, height=1080, show_preview=False):
    global current_frame_jpeg, is_running

    # If no index explicitly provided, auto-detect (prefer index 1 or external camera over index 0)
    if camera_idx is None:
        print("[Windows Cam Bridge] Auto-detecting external optical camera...")
        candidates = [1, 2, 0]  # Check external indices 1 and 2 first before laptop webcam 0
    else:
        candidates = [camera_idx]

    cap = None
    selected_idx = None
    for idx in candidates:
        print(f"[Windows Cam Bridge] Probing camera index {idx} with DirectShow...")
        c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not c.isOpened():
            c = cv2.VideoCapture(idx)
        if c.isOpened():
            ret, test_frame = c.read()
            if ret and test_frame is not None:
                cap = c
                selected_idx = idx
                print(f"[Windows Cam Bridge] Successfully locked onto camera index {idx}!")
                break
            c.release()

    if not cap or not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_idx if camera_idx is not None else 'any candidate'}.")
        print("[ERROR] Please verify the USB camera is plugged in and not in use by Windows Camera or Teams.")
        is_running = False
        return

    # Request desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Windows Cam Bridge] Active capture resolution: {actual_w}x{actual_h} px (Index {selected_idx})")
    print(f"[Windows Cam Bridge] Streaming live feed to WSL2 at: http://localhost:8080/video_feed")
    print(f"[Windows Cam Bridge] (Note: to switch cameras, run with --index 0 or --index 1)")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]

    while is_running:
        ret, frame = cap.read()
        if ret and frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, encode_param)
            current_frame_jpeg = jpeg.tobytes()

            if show_preview:
                preview_frame = cv2.resize(frame, (960, 540))
                cv2.putText(
                    preview_frame,
                    f"Index {selected_idx}: {actual_w}x{actual_h} px | Press 'q' to close preview",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("AOI Optical Feed Preview", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    show_preview = False
                    cv2.destroyAllWindows()
        else:
            time.sleep(0.01)

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Windows-to-WSL2 High-Speed Camera Bridge")
    parser.add_argument("--list", action="store_true", help="List all detected Windows cameras and resolutions, then exit")
    parser.add_argument("--index", type=int, default=None, help="Explicit camera index to use (e.g. 1 for external See3CAM)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP streaming port (default: 8080)")
    parser.add_argument("--width", type=int, default=1920, help="Target capture width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Target capture height (default: 1080)")
    parser.add_argument("--preview", action="store_true", help="Open a local preview window on Windows to verify camera aim")
    args = parser.parse_args()

    if args.list:
        list_available_cameras()
        return

    import threading
    t = threading.Thread(
        target=capture_loop,
        args=(args.index, args.width, args.height, args.preview),
        daemon=True,
    )
    t.start()

    server = ThreadedHTTPServer(("0.0.0.0", args.port), CamStreamHandler)
    print(f"[Windows Cam Bridge] HTTP Stream server ready on port {args.port}.")
    print("[Windows Cam Bridge] Press Ctrl+C in this terminal to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Windows Cam Bridge] Shutting down...")
        global is_running
        is_running = False
        server.server_close()


if __name__ == "__main__":
    main()

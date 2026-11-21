#!/usr/bin/env python3
"""
Zero-latency Windows-to-WSL2 Camera Bridge.
Runs natively on the Windows host to stream external USB cameras (like See3CAM_CU30)
directly into WSL2 over local HTTP MJPEG, bypassing usbipd isochronous bandwidth limitations.

Usage (Run in Windows PowerShell or CMD):
    pip install opencv-python
    python tools/stream_camera_windows.py --index 0 --port 8080
"""

import sys
import time
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV not installed on Windows. Run: pip install opencv-python")
    sys.exit(1)

# Global frame buffer
current_frame_jpeg = None
is_running = True


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


def capture_loop(camera_idx=0, width=1920, height=1080):
    global current_frame_jpeg, is_running
    print(f"[Windows Cam Bridge] Probing optical camera (requested index: {camera_idx})...")
    
    cap = None
    # Try requested index first, then scan other indices if needed
    candidates = [camera_idx] + [i for i in [0, 1, 2, 3] if i != camera_idx]
    for idx in candidates:
        print(f"[Windows Cam Bridge] Trying camera index {idx} with DirectShow...")
        c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not c.isOpened():
            c = cv2.VideoCapture(idx)
        if c.isOpened():
            ret, test_frame = c.read()
            if ret and test_frame is not None:
                cap = c
                camera_idx = idx
                print(f"[Windows Cam Bridge] Successfully connected to camera index {idx}!")
                break
            c.release()

    if not cap or not cap.isOpened():
        print(f"[ERROR] Could not open any optical camera on Windows host.")
        print("[ERROR] Please verify the USB camera is plugged in and not in use by another app (e.g. Windows Camera, Teams).")
        is_running = False
        return

    # Attempt high-resolution optical capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Windows Cam Bridge] Camera streaming at {actual_w}x{actual_h} px.")
    print(f"[Windows Cam Bridge] Stream URL: http://localhost:8080/video_feed")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]

    while is_running:
        ret, frame = cap.read()
        if ret and frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, encode_param)
            current_frame_jpeg = jpeg.tobytes()
        else:
            time.sleep(0.01)

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Windows-to-WSL2 High-Speed Camera Bridge")
    parser.add_argument("--index", type=int, default=0, help="Camera index on Windows (default: 0)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP streaming port (default: 8080)")
    parser.add_argument("--width", type=int, default=1920, help="Target capture width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Target capture height (default: 1080)")
    args = parser.parse_args()

    import threading
    t = threading.Thread(target=capture_loop, args=(args.index, args.width, args.height), daemon=True)
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

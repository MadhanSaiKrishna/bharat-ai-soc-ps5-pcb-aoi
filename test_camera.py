#!/usr/bin/env python3
"""
Diagnostic utility to detect and test cameras on Linux / WSL2.
"""

import sys
import glob

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def check_cameras():
    print("\n" + "="*60)
    print("        OPTICAL CAMERA DIAGNOSTIC UTILITY (WSL2 / LINUX)        ")
    print("="*60)

    if not HAS_CV2:
        print("[ERROR] OpenCV (cv2) is not installed in the current environment.")
        return

    # 1. Check for /dev/video* devices
    video_devices = sorted(glob.glob("/dev/video*"))
    print(f"Detected /dev/video devices: {video_devices if video_devices else 'NONE FOUND'}")

    if not video_devices:
        print("\n[DIAGNOSIS: WSL2 USB Passthrough Required]")
        print("You are running inside WSL2 (Windows Subsystem for Linux).")
        print("By default, Windows does not share USB webcams with WSL2.")
        print("\nTo share your external USB camera with WSL2:")
        print("  1. In Windows PowerShell (Run as Administrator):")
        print("       winget install dorssel.usbipd-win")
        print("       usbipd list")
        print("       usbipd bind --busid <BUSID_OF_YOUR_CAMERA>")
        print("       usbipd attach --wsl --busid <BUSID_OF_YOUR_CAMERA>")
        print("\nAlternatively, you can test with an IP camera or simulated video stream.")
        return

    # 2. Test opening each detected device
    for dev_path in video_devices:
        try:
            dev_idx = int(dev_path.replace("/dev/video", ""))
        except ValueError:
            continue

        # Check device hardware name from sysfs
        dev_name = "Unknown"
        try:
            with open(f"/sys/class/video4linux/video{dev_idx}/name", "r") as f:
                dev_name = f.read().strip()
        except Exception:
            pass

        print(f"\nDevice {dev_path}: '{dev_name}'")

        # Check permissions
        import os
        has_rw = os.access(dev_path, os.R_OK | os.W_OK)
        if not has_rw:
            print(f"  [ERROR: PERMISSION DENIED] Your user cannot read/write {dev_path}.")
            print("  Fix with this command in your terminal:")
            print(f"      sudo chmod 666 /dev/video*")
            continue

        cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  [SUCCESS] {dev_path} is ACTIVE! Capture resolution: {w}x{h} px")
            else:
                print(f"  [WARN] {dev_path} opened but returned no frame (may be a metadata node).")
            cap.release()
        else:
            print(f"  [INFO] {dev_path} could not be opened as a direct video stream.")

    print("="*60 + "\n")


if __name__ == "__main__":
    check_cameras()

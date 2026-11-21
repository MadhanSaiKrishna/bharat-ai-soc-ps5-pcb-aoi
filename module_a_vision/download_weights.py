#!/usr/bin/env python3
"""
Fetch trained YOLOv8-OBB weights from the GitHub release (Member 1 Deliverable)

Weights are not committed to git (.gitignore: *.pt, weights/). They are
published as GitHub Release assets instead. This script downloads them into
module_a_vision/weights/, where VisionInferenceEngine looks by default.

Usage:
    python -m module_a_vision.download_weights            # default release
    python -m module_a_vision.download_weights --tag vision-v1 --all
"""

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = "MadhanSaiKrishna/bharat-ai-soc-ps5-pcb-aoi"
DEFAULT_TAG = "vision-v1"
DEFAULT_ASSET = "smt_merged_v8s.pt"
ALL_ASSETS = ["smt_merged_v8s.pt", "soldef_joint_v8s.pt"]
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def download(tag: str, asset: str, force: bool = False) -> Path:
    WEIGHTS_DIR.mkdir(exist_ok=True)
    dst = WEIGHTS_DIR / asset
    if dst.exists() and not force:
        print(f"[download_weights] {dst.name} already present, skipping (use --force to refresh)")
        return dst
    url = f"https://github.com/{REPO}/releases/download/{tag}/{asset}"
    print(f"[download_weights] {url}")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    print(f"[download_weights] saved {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--asset", default=DEFAULT_ASSET)
    ap.add_argument("--all", action="store_true", help="Download every published model")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    assets = ALL_ASSETS if args.all else [args.asset]
    try:
        for a in assets:
            download(args.tag, a, args.force)
    except Exception as e:
        sys.exit(f"[download_weights] failed: {e}")


if __name__ == "__main__":
    main()

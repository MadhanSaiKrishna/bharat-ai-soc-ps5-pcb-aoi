#!/usr/bin/env python3
"""
YOLOv8-OBB training entry point (Member 1 Deliverable)

Fine-tunes a DOTA-pretrained YOLOv8-OBB model on a converted AOI dataset
(e.g. the SolDef_AI joint-level set produced by datasets/convert_soldef.py)
and copies the best checkpoint to module_a_vision/weights/<name>.pt, which is
where VisionInferenceEngine looks for weights.

Usage:
    python -m module_a_vision.train_obb --data ../datasets/yolo/soldef_joint_obb/data.yaml \
        --name soldef_joint_v8s --model yolov8s-obb.pt --imgsz 1024 --batch 8
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--name", required=True, help="Run name; also the exported weights filename")
    ap.add_argument("--model", default="yolov8s-obb.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default=str(Path(__file__).resolve().parents[2] / "runs"),
                    help="Where Ultralytics writes run folders (kept outside the repo)")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=0,
        # Small real-image dataset: lean on geometric + photometric augmentation
        degrees=15.0,      # small camera/board rotation
        flipud=0.5,
        fliplr=0.5,
        hsv_v=0.4,         # lighting variation between rigs
        mosaic=1.0,
        close_mosaic=20,
        cos_lr=True,
        plots=True,
    )

    best = Path(model.trainer.best)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    dst = WEIGHTS_DIR / f"{args.name}.pt"
    shutil.copy2(best, dst)
    print(f"[train_obb] best checkpoint -> {dst}")

    metrics = YOLO(str(dst)).val(data=args.data, imgsz=args.imgsz, device=args.device,
                                 project=args.project, name=f"{args.name}_val", exist_ok=True)
    print(f"[train_obb] val mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SolDef_AI -> YOLOv8-OBB converter (Member 1 Deliverable)

SolDef_AI (Kaggle: mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection)
ships LabelMe polygon JSON for 428 images captured on three days with two
different labelling protocols:

  * 2022-03-29: one polygon around the WHOLE resistor, label good / no_good
  * 2022-03-30, 2022-04-08: one polygon per SOLDER JOINT, labels
    good / exc_solder / poor_solder / spike

Mixing the two protocols in one detector creates missing-label conflicts, so
this script exports the joint-level subset only (default). Each polygon is
converted to its minimum-area rotated rectangle and written in Ultralytics OBB
format:  class x1 y1 x2 y2 x3 y3 x4 y4  (normalised to [0, 1]).

Images captured seconds apart show the same component, so the train/val split
is done on capture bursts (groups of shots with < BURST_GAP_S between them),
not on individual images, to keep near-duplicates out of validation.

Usage:
    python -m module_a_vision.datasets.convert_soldef \
        --src  ../datasets/soldef_ai/SolDef_AI/Labeled \
        --dst  ../datasets/yolo/soldef_joint_obb
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Joint-level class map: SolDef label -> AOI class name (index = YOLO class id)
JOINT_CLASSES = ["GOOD_JOINT", "EXCESS_SOLDER", "INSUFFICIENT_SOLDER", "SOLDER_SPIKE"]
JOINT_LABEL_MAP = {
    "good": "GOOD_JOINT",
    "exc_solder": "EXCESS_SOLDER",
    "poor_solder": "INSUFFICIENT_SOLDER",
    "spike": "SOLDER_SPIKE",
}
COMPONENT_DAYS = {"20220329"}  # whole-component protocol, excluded from joint set
BURST_GAP_S = 60


def capture_time(name: str) -> datetime:
    # WIN_20220330_10_12_33_Pro.jpg
    stem = name.split("WIN_")[1][:17]
    return datetime.strptime(stem, "%Y%m%d_%H_%M_%S")


def polygon_to_obb(points, img_w: int, img_h: int) -> list:
    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    box = cv2.boxPoints(rect)
    box[:, 0] = np.clip(box[:, 0] / img_w, 0.0, 1.0)
    box[:, 1] = np.clip(box[:, 1] / img_h, 0.0, 1.0)
    return box.flatten().tolist()


def group_bursts(jsons: list) -> list:
    """Groups images into capture bursts; returns list of lists of paths."""
    items = sorted(jsons, key=lambda p: capture_time(p.name))
    groups, cur, last = [], [], None
    for p in items:
        t = capture_time(p.name)
        if last is not None and (t - last).total_seconds() > BURST_GAP_S:
            groups.append(cur)
            cur = []
        cur.append(p)
        last = t
    if cur:
        groups.append(cur)
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="SolDef_AI/Labeled directory")
    ap.add_argument("--dst", type=Path, required=True, help="Output YOLO dataset root")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=15)  # chosen for per-class val balance (~25% each)
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    args = ap.parse_args()

    jsons = [p for p in sorted(args.src.glob("*.json"))
             if p.name.split("WIN_")[1][:8] not in COMPONENT_DAYS]
    if not jsons:
        raise SystemExit(f"No joint-level LabelMe JSON found in {args.src}")

    # Burst-grouped split, done per capture day so both days appear in val
    rng = random.Random(args.seed)
    split = {}
    by_day = defaultdict(list)
    for p in jsons:
        by_day[p.name.split("WIN_")[1][:8]].append(p)
    for day, files in sorted(by_day.items()):
        groups = group_bursts(files)
        rng.shuffle(groups)
        n_val = max(1, round(len(files) * args.val_frac))
        taken = 0
        for g in groups:
            subset = "val" if taken < n_val else "train"
            if subset == "val":
                taken += len(g)
            for p in g:
                split[p] = subset

    stats = {s: Counter() for s in ("train", "val")}
    n_img = Counter()
    for subset in ("train", "val"):
        (args.dst / "images" / subset).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / subset).mkdir(parents=True, exist_ok=True)

    for p, subset in split.items():
        d = json.loads(p.read_text())
        img_src = (p.parent / d["imagePath"]).resolve()
        w, h = d["imageWidth"], d["imageHeight"]
        lines = []
        for s in d["shapes"]:
            cls = JOINT_LABEL_MAP.get(s["label"])
            if cls is None or len(s["points"]) < 3:
                continue
            coords = polygon_to_obb(s["points"], w, h)
            lines.append(" ".join([str(JOINT_CLASSES.index(cls))] + [f"{c:.6f}" for c in coords]))
            stats[subset][cls] += 1

        img_dst = args.dst / "images" / subset / img_src.name
        if img_dst.exists() or img_dst.is_symlink():
            img_dst.unlink()
        if args.copy:
            img_dst.write_bytes(img_src.read_bytes())
        else:
            img_dst.symlink_to(img_src)
        (args.dst / "labels" / subset / (img_src.stem + ".txt")).write_text("\n".join(lines) + "\n")
        n_img[subset] += 1

    yaml_lines = [f"path: {args.dst.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {c}" for i, c in enumerate(JOINT_CLASSES)]
    (args.dst / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    for subset in ("train", "val"):
        print(f"[convert_soldef] {subset}: {n_img[subset]} images, {dict(stats[subset])}")
    print(f"[convert_soldef] wrote {args.dst / 'data.yaml'}")


if __name__ == "__main__":
    main()

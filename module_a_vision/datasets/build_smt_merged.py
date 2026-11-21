#!/usr/bin/env python3
"""
Merged SMT solder & assembly dataset builder (Member 1 Deliverable)

Combines two real-image sources into one YOLOv8-OBB dataset:

  1. ParsaVictor/pcb-solder-defect-detection (Hugging Face, research/educational
     use): 5,209 populated-PCB photos, 9 classes, axis-aligned YOLO boxes.
     Boxes are written as 4-corner OBBs with angle 0.
  2. SolDef_AI joint-level subset, already converted to OBB by convert_soldef.py.
     GOOD_JOINT is dropped: the HF source does not label good joints, so keeping
     it would teach the model contradictory background.

Images are symlinked (no copies). Filenames are prefixed by source to avoid
collisions. HF train/val/test splits are kept; SolDef train/val go to train/val.

Usage:
    python -m module_a_vision.datasets.build_smt_merged \
        --hf     ../datasets/pcb_solder_hf/final_solder \
        --soldef ../datasets/yolo/soldef_joint_obb \
        --dst    ../datasets/yolo/smt_merged_obb
"""

import argparse
from collections import Counter
from pathlib import Path

CLASSES = [
    "INSUFFICIENT_SOLDER",
    "EXCESS_SOLDER",
    "SOLDER_BRIDGE",
    "SOLDER_VOID",
    "MISSING_COMPONENT",
    "SHIFTED_COMPONENT",
    "BURN",
    "PHYSICAL_DAMAGE",
    "OXIDATION",
    "SOLDER_SPIKE",
]
# HF class order (from its data.yaml) -> merged id
HF_MAP = {i: i for i in range(9)}
# convert_soldef.JOINT_CLASSES order -> merged id (None = drop)
SOLDEF_MAP = {0: None, 1: 1, 2: 0, 3: 9}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def hbb_to_obb(xc, yc, w, h):
    x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return [min(max(v, 0.0), 1.0) for p in pts for v in p]


def link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True, help="HF final_solder root (images/, labels/)")
    ap.add_argument("--soldef", type=Path, required=True, help="convert_soldef.py output root")
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    stats = {s: Counter() for s in ("train", "val", "test")}
    n_img = Counter()
    for s in stats:
        (args.dst / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / s).mkdir(parents=True, exist_ok=True)

    # 1. HF source (axis-aligned -> OBB)
    for s in ("train", "val", "test"):
        for img in sorted((args.hf / "images" / s).iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lab = args.hf / "labels" / s / (img.stem + ".txt")
            lines = []
            if lab.exists():
                for ln in lab.read_text().splitlines():
                    v = ln.split()
                    if len(v) != 5:
                        continue
                    cid = HF_MAP[int(v[0])]
                    coords = hbb_to_obb(*map(float, v[1:]))
                    lines.append(" ".join([str(cid)] + [f"{c:.6f}" for c in coords]))
                    stats[s][CLASSES[cid]] += 1
            name = f"hf_{img.stem}"
            link(img, args.dst / "images" / s / f"{name}{img.suffix}")
            (args.dst / "labels" / s / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img[s] += 1

    # 2. SolDef joint-level (already OBB)
    for s in ("train", "val"):
        for img in sorted((args.soldef / "images" / s).iterdir()):
            lab = args.soldef / "labels" / s / (img.stem + ".txt")
            lines = []
            for ln in lab.read_text().splitlines():
                v = ln.split()
                cid = SOLDEF_MAP[int(v[0])]
                if cid is None:
                    continue
                lines.append(" ".join([str(cid)] + v[1:]))
                stats[s][CLASSES[cid]] += 1
            name = f"soldef_{img.stem}"
            link(img, args.dst / "images" / s / f"{name}{img.suffix}")
            (args.dst / "labels" / s / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img[s] += 1

    yaml = [f"path: {args.dst.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (args.dst / "data.yaml").write_text("\n".join(yaml) + "\n")
    for s in ("train", "val", "test"):
        print(f"[build_smt_merged] {s}: {n_img[s]} images, {dict(sorted(stats[s].items()))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Merged SMT solder & assembly dataset builder (Member 1 Deliverable)

Combines up to three real-image sources into one YOLOv8-OBB dataset:

  1. ParsaVictor/pcb-solder-defect-detection (Hugging Face, research/educational
     use): 5,209 populated-PCB photos, 9 classes, axis-aligned YOLO boxes.
     Boxes are written as 4-corner OBBs with angle 0.
  2. SolDef_AI joint-level subset, already converted to OBB by convert_soldef.py.
     GOOD_JOINT is dropped: the HF source does not label good joints, so keeping
     it would teach the model contradictory background.

  3. (optional, --rf) Roboflow "SMD Component Placement Defect 2" v5 export
     (CC BY 4.0), YOLOv8 polygon labels, one labelled component per image.
     Mapping: Tombstoning -> TOMBSTONE, Misalignment -> SHIFTED_COMPONENT,
     Orientation Error -> ORIENTATION_ERROR, Normal -> label dropped, image
     kept as a background negative. Polygons -> cv2.minAreaRect OBBs.
     The export's train split holds 3 Roboflow-augmented copies per original;
     only one copy per original is kept. Its published splits leak (crops of
     the same phone photo / burst land in different splits), so images are
     grouped by capture burst (MVIMG_/IMG_/WIN_ timestamps < 60 s apart) or by
     original filename, and any group spanning splits is moved to train.

Images are symlinked (no copies). Filenames are prefixed by source to avoid
collisions. HF train/val/test splits are kept (HF test = held-out score);
SolDef train/val go to train/val; RF val/test (after leak removal) go to
val/test. Per-source test lists test_hf.txt / test_rf.txt and matching
data_test_hf.yaml / data_test_rf.yaml are written for per-source evaluation.

Usage:
    python -m module_a_vision.datasets.build_smt_merged \
        --hf     ../datasets/pcb_solder_hf/final_solder \
        --soldef ../datasets/yolo/soldef_joint_obb \
        [--rf    ../datasets/rf_smd_placement/v5] \
        --dst    ../datasets/yolo/smt_merged_obb
"""

import argparse
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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
    "TOMBSTONE",
    "ORIENTATION_ERROR",
]
# HF class order (from its data.yaml) -> merged id
HF_MAP = {i: i for i in range(9)}
# convert_soldef.JOINT_CLASSES order -> merged id (None = drop)
SOLDEF_MAP = {0: None, 1: 1, 2: 0, 3: 9}
# Roboflow class order ['Misalignment', 'Normal', 'Orientation Error', 'Tombstoning'] -> merged id
RF_MAP = {0: 5, 1: None, 2: 11, 3: 10}
RF_SPLITS = {"train": "train", "valid": "val", "test": "test"}
BURST_GAP_S = 60
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def rf_base(stem: str) -> str:
    """Original upload name: strip Roboflow '_jpg.rf.<hash>' suffix and '-trashed-<n>-' prefix."""
    stem = re.sub(r"_(jpe?g|png|bmp)\.rf\.[0-9a-f]+$", "", stem, flags=re.I)
    return re.sub(r"^-trashed-\d+-", "", stem)


def rf_capture_time(base: str):
    """Capture time from phone (MVIMG_/IMG_YYYYMMDD_HHMMSS) or Windows camera (WIN_YYYYMMDD_HH_MM_SS) names."""
    m = re.search(r"(?:MV)?IMG_(\d{8})_(\d{6})", base) or re.search(r"WIN_(\d{8})_(\d{2})_?(\d{2})_?(\d{2})", base)
    if not m:
        return None
    d = "".join(m.groups())
    try:
        return datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]), int(d[10:12]), int(d[12:14]))
    except ValueError:  # malformed timestamp: fall back to name-only grouping
        return None


def rf_groups(bases):
    """Union bases into leak groups: same original name, or same capture burst."""
    group = {b: b for b in bases}
    timed = sorted((t, b) for b in bases if (t := rf_capture_time(b)) is not None)
    for (t0, b0), (t1, b1) in zip(timed, timed[1:]):
        if (t1 - t0).total_seconds() <= BURST_GAP_S:
            group[b1] = group[b0]
    return group


def polygon_to_obb(xy_norm, img_w: int, img_h: int) -> list:
    pts = (np.asarray(xy_norm, dtype=np.float64).reshape(-1, 2) * (img_w, img_h)).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    box[:, 0] = np.clip(box[:, 0] / img_w, 0.0, 1.0)
    box[:, 1] = np.clip(box[:, 1] / img_h, 0.0, 1.0)
    return box.flatten().tolist()


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
    ap.add_argument("--rf", type=Path, help="Roboflow SMD placement v5 YOLOv8 export root (train/, valid/, test/)")
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    stats = {s: Counter() for s in ("train", "val", "test")}
    n_img = Counter()
    test_lists = defaultdict(list)
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
            if s == "test":
                test_lists["hf"].append(f"./images/test/{name}{img.suffix}")

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

    # 3. Roboflow SMD placement (polygons -> OBB), leak-free regrouping
    if args.rf:
        items = []  # (rf_split, img, base)
        for rs in RF_SPLITS:
            for img in sorted((args.rf / rs / "images").iterdir()):
                if img.suffix.lower() in IMG_EXTS:
                    items.append((rs, img, rf_base(img.stem)))
        group = rf_groups({b for _, _, b in items})
        g_splits = defaultdict(set)
        for rs, _, b in items:
            g_splits[group[b]].add(RF_SPLITS[rs])
        seen_train, moved, dup = set(), 0, 0
        for rs, img, b in items:
            s = RF_SPLITS[rs]
            if rs == "train":
                if b in seen_train:  # Roboflow augmented copy of an original already kept
                    dup += 1
                    continue
                seen_train.add(b)
            elif len(g_splits[group[b]]) > 1:
                s, moved = "train", moved + 1
            lab = args.rf / rs / "labels" / (img.stem + ".txt")
            w, h = Image.open(img).size
            lines = []
            for ln in lab.read_text().splitlines():
                v = ln.split()
                if len(v) < 7:
                    continue
                cid = RF_MAP[int(v[0])]
                if cid is None:
                    continue
                coords = polygon_to_obb(list(map(float, v[1:])), w, h)
                lines.append(" ".join([str(cid)] + [f"{c:.6f}" for c in coords]))
                stats[s][CLASSES[cid]] += 1
            name = f"rf_{img.stem}"
            link(img, args.dst / "images" / s / f"{name}{img.suffix}")
            (args.dst / "labels" / s / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img[s] += 1
            if s == "test":
                test_lists["rf"].append(f"./images/test/{name}{img.suffix}")
        print(f"[build_smt_merged] rf: dropped {dup} augmented train copies, moved {moved} leaking val/test images to train")

    head = [f"path: {args.dst.resolve()}", "train: images/train", "val: images/val"]
    names = ["names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (args.dst / "data.yaml").write_text("\n".join(head + ["test: images/test"] + names) + "\n")
    for src, files in test_lists.items():
        (args.dst / f"test_{src}.txt").write_text("\n".join(files) + "\n")
        (args.dst / f"data_test_{src}.yaml").write_text("\n".join(head + [f"test: test_{src}.txt"] + names) + "\n")
    for s in ("train", "val", "test"):
        print(f"[build_smt_merged] {s}: {n_img[s]} images, {dict(sorted(stats[s].items()))}")


if __name__ == "__main__":
    main()

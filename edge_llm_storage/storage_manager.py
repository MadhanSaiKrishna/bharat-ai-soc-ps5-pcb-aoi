#!/usr/bin/env python3
"""
Decoupled Edge Storage Manager (Member 2 Deliverable)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot

Orchestrates decoupled high-capacity local storage (NVMe/SSD partition on ZCU104/Host)
to prevent DDR RAM choking during real-time vision and LLM inspection:
  1. RAW defect frame burst buffering (full-res crops).
  2. Append-only inspection event telemetry log (JSONL / SQLite).
  3. Operator-flagged active learning queue (images + embedding metadata).
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


class DecoupledStorageManager:
    """Manages local storage partitions for telemetry, defect images, and active learning."""

    def __init__(self, root_storage_dir: Optional[str] = None):
        if root_storage_dir:
            self.base_dir = Path(root_storage_dir)
        else:
            self.base_dir = Path(__file__).resolve().parent.parent / "data"

        self.telemetry_dir = self.base_dir / "telemetry_logs"
        self.raw_dumps_dir = self.base_dir / "raw_defect_dumps"
        self.active_learning_queue = self.base_dir / "operator_ingest_queue"

        # Create directory hierarchy
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dumps_dir.mkdir(parents=True, exist_ok=True)
        self.active_learning_queue.mkdir(parents=True, exist_ok=True)

        self.telemetry_log_file = self.telemetry_dir / "inspection_history.jsonl"

    def log_inspection_event(self, event: Dict[str, Any], advisories: Optional[Dict[str, str]] = None) -> str:
        """Appends complete inspection payload with timestamp and advisories."""
        record = {
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "advisories": advisories or {},
        }
        with open(self.telemetry_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        return str(self.telemetry_log_file)

    def save_raw_defect_crop(self, frame_patch, defect_id: int, board_id: str) -> str:
        """Saves a multi-megapixel RAW defect image crop to isolated storage."""
        filename = f"{board_id}_defect_{defect_id}_{int(time.time()*1000)}.png"
        filepath = self.raw_dumps_dir / filename
        if cv2 and frame_patch is not None:
            cv2.imwrite(str(filepath), frame_patch)
        else:
            with open(filepath, "wb") as f:
                f.write(b"MOCK_RAW_IMAGE_DATA")
        return str(filepath)

    def queue_operator_anomaly(
        self,
        crop_image,
        label: str,
        embedding_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Saves operator-flagged out-of-distribution anomaly for active learning."""
        sample_id = f"anomaly_{int(time.time()*1000)}"
        img_name = f"{sample_id}.png"
        meta_name = f"{sample_id}.json"

        img_path = self.active_learning_queue / img_name
        meta_path = self.active_learning_queue / meta_name

        if cv2 and crop_image is not None:
            cv2.imwrite(str(img_path), crop_image)
        else:
            with open(img_path, "wb") as f:
                f.write(b"MOCK_ANOMALY_IMAGE_DATA")

        meta = {
            "sample_id": sample_id,
            "operator_label": label,
            "embedding_id": embedding_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": metadata or {},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {"image_path": str(img_path), "metadata_path": str(meta_path)}


if __name__ == "__main__":
    storage = DecoupledStorageManager()
    sample_event = {"board_id": "BOARD_TEST_001", "checkpoint": "CHECKPOINT_1_SPI", "defects": []}
    log_path = storage.log_inspection_event(sample_event)
    print(f"[StorageManager] Logged sample event to: {log_path}")

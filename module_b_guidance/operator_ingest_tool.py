#!/usr/bin/env python3
"""
Operator Active-Learning Anomaly Ingest Tool (Member 3 Deliverable)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot

Allows shop-floor operators to crop out-of-distribution (OOD) anomalies,
tag them with labels/notes, extract feature embeddings, and queue them
for edge vector indexing and offline retraining.
Conforms strictly to schemas/inspection_event.json.
"""

import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
INGEST_QUEUE_DIR = REPO_ROOT / "data" / "operator_ingest_queue"
CROPS_DIR = INGEST_QUEUE_DIR / "crops"
INGEST_LOG_PATH = INGEST_QUEUE_DIR / "ingest_log.jsonl"


class OperatorAnomalyIngestor:
    """Handles operator manual crop extraction, metadata logging, and embedding queuing."""

    def __init__(self, queue_dir: Optional[Path] = None):
        self.queue_dir = queue_dir or INGEST_QUEUE_DIR
        self.crops_dir = self.queue_dir / "crops"
        self.log_path = self.queue_dir / "ingest_log.jsonl"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    def crop_and_ingest(
        self,
        frame: Union[np.ndarray, Image.Image],
        bbox: Tuple[int, int, int, int],  # (xmin, ymin, xmax, ymax)
        operator_label: str,
        board_id: str = "UNKNOWN_BOARD",
        severity: str = "MEDIUM",
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Crops region of interest, saves image artifact, generates unique ID,
        and records to the active learning queue.
        """
        # Convert to PIL Image
        if isinstance(frame, np.ndarray):
            # Check if OpenCV BGR or RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Assuming RGB or BGR - standard RGB PIL image
                img = Image.fromarray(frame)
            else:
                img = Image.fromarray(frame)
        else:
            img = frame

        xmin, ymin, xmax, ymax = bbox
        width, height = img.size
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(xmin + 1, min(xmax, width))
        ymax = max(ymin + 1, min(ymax, height))

        cropped = img.crop((xmin, ymin, xmax, ymax))

        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        anomaly_id = f"ANOM_{timestamp_str}_{uuid.uuid4().hex[:6]}"
        crop_filename = f"{anomaly_id}_{operator_label.replace(' ', '_')}.png"
        crop_filepath = self.crops_dir / crop_filename

        cropped.save(crop_filepath)

        # Attempt embedding extraction if module_a_vision is available
        embedding_id = f"EMB_{anomaly_id}"
        has_embedding = False
        try:
            from module_a_vision.embedding_extractor import LightweightEmbeddingExtractor
            extractor = LightweightEmbeddingExtractor()
            vector = extractor.extract_embedding(cropped)
            has_embedding = True
        except Exception:
            # Fallback when PyTorch or weights are not yet present on machine
            has_embedding = False

        try:
            rel_crop_path = str(crop_filepath.relative_to(REPO_ROOT))
        except ValueError:
            rel_crop_path = str(crop_filepath)

        record = {
            "anomaly_id": anomaly_id,
            "board_id": board_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operator_label": operator_label,
            "severity": severity,
            "notes": notes,
            "bbox": [xmin, ymin, xmax, ymax],
            "crop_path": rel_crop_path,
            "embedding_vector_id": embedding_id if has_embedding else None,
            "queued_for_retraining": True,
        }

        # Append record to JSONL queue
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Return standardized schema payload matching schemas/inspection_event.json
        contract_event_fragment = {
            "flagged_new_anomaly": True,
            "operator_label": operator_label,
            "embedding_vector_id": embedding_id if has_embedding else f"PENDING_{anomaly_id}",
            "anomaly_id": anomaly_id,
            "crop_path": rel_crop_path,
        }

        return contract_event_fragment


if __name__ == "__main__":
    ingestor = OperatorAnomalyIngestor()
    # Test with dummy test image
    test_img = Image.new("RGB", (640, 480), color=(50, 100, 150))
    res = ingestor.crop_and_ingest(
        frame=test_img,
        bbox=(100, 100, 250, 250),
        operator_label="FOREIGN_OBJECT_DEBRIS",
        board_id="TEST_BOARD_001",
        notes="Suspected lint particle across pad fingers",
    )
    print("[Operator Ingest Tool] Ingest Result:")
    print(json.dumps(res, indent=2))

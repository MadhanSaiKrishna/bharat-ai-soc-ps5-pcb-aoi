#!/usr/bin/env python3
"""
Operator Anomaly Feature Extractor (Member 1 Deliverable)
Closed-Loop Multi-Tier Edge AOI & Quality-Control Copilot

Extracts 128/256-D normalized visual embeddings from cropped defect patches
using a lightweight MobileNetV3 backbone for zero-downtime operator anomaly
ingestion (Active Learning / Few-Shot matching).
"""

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LightweightEmbeddingExtractor:
    """Extracts compact embeddings for shop-floor operator anomaly matching."""

    def __init__(self, embedding_dim: int = 128, device: str = "cpu"):
        self.embedding_dim = embedding_dim
        self.device = device
        self.model = None

        if HAS_TORCH:
            self._init_pytorch_backbone()
        else:
            print("[EmbeddingExtractor] PyTorch not loaded, operating in lightweight mock/deterministic mode.")

        # Local in-memory vector store: list of (embedding_id, label, vector)
        self.vector_store: List[Dict[str, Any]] = []

    def _init_pytorch_backbone(self):
        try:
            # Load pretrained MobileNetV3 Small for compact edge footprint (< 10 MB)
            weights = MobileNet_V3_Small_Weights.DEFAULT
            base_model = mobilenet_v3_small(weights=weights)
            in_features = base_model.classifier[0].in_features
            # Replace classifier with embedding projection head
            base_model.classifier = nn.Sequential(
                nn.Linear(in_features, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
            )
            base_model.eval()
            self.model = base_model.to(self.device)
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception as e:
            print(f"[EmbeddingExtractor] Error initializing MobileNetV3: {e}")
            self.model = None

    def extract_embedding(self, image_input) -> List[float]:
        """Generates an L2-normalized feature embedding vector."""
        if HAS_TORCH and self.model is not None:
            if isinstance(image_input, str) or isinstance(image_input, Path):
                img = Image.open(image_input).convert("RGB")
            elif hasattr(image_input, "shape"): # numpy array from cv2
                img = Image.fromarray(image_input).convert("RGB")
            else:
                img = image_input

            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(tensor)
                feat = nn.functional.normalize(feat, p=2, dim=1)
                return feat.squeeze(0).cpu().tolist()
        else:
            # Deterministic pseudo-embedding for testing without PyTorch weights downloaded
            vec = [math.sin(i * 0.1) for i in range(self.embedding_dim)]
            norm = math.sqrt(sum(x * x for x in vec))
            return [x / norm for x in vec]

    def index_sample(self, vector_id: str, label: str, embedding: List[float]):
        """Saves embedding into local vector store."""
        self.vector_store.append({
            "vector_id": vector_id,
            "label": label,
            "embedding": embedding,
        })

    def query_nearest(self, query_vector: List[float], top_k: int = 1) -> List[Tuple[str, float]]:
        """Finds nearest neighbor using cosine similarity."""
        results = []
        for item in self.vector_store:
            stored_vec = item["embedding"]
            cos_sim = sum(a * b for a, b in zip(query_vector, stored_vec))
            results.append((item["label"], cos_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Alias for backward compatibility across subsystems
MobileNetV3FeatureExtractor = LightweightEmbeddingExtractor


if __name__ == "__main__":
    extractor = LightweightEmbeddingExtractor(embedding_dim=128)
    sample_vec = extractor.extract_embedding(None)
    extractor.index_sample("v_001", "FOREIGN_PARTICULATE", sample_vec)
    nearest = extractor.query_nearest(sample_vec, top_k=1)
    print(f"[Embedding Extractor] Extracted vector length: {len(sample_vec)}")
    print(f"[Embedding Extractor] Top match: {nearest[0][0]} with cosine similarity = {nearest[0][1]:.4f}")

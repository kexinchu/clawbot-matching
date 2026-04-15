"""SBERT-based semantic encoder — drop-in replacement for SimpleEncoder.

Uses BAAI/bge-base-en-v1.5 by default (768-dim, MIT license, English).
For multilingual / Chinese-English mixed content, swap to BAAI/bge-m3.

Interface is identical to SimpleEncoder so generate_fixtures.py and
any caller only needs to change the import line.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class SBERTEncoder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        # Keep attribute name consistent with SimpleEncoder
        self.embedding_dim: int = self.model.get_embedding_dimension()

    def __call__(self, text: str) -> np.ndarray:
        """Encode a single text string → unit-normalised float64 vector."""
        emb = self.model.encode(text, normalize_embeddings=True)
        return emb.astype(np.float64)

    def batch(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings → (N, dim) float64 array (normalised)."""
        embs = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
        return embs.astype(np.float64)

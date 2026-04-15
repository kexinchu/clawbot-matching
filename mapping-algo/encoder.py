"""Lightweight text encoder for testing and prototyping.

Uses a deterministic bag-of-words approach: each unique word gets a fixed
random direction (seeded by hash), phrase embedding = normalised sum of
word vectors. This gives natural semantic similarity without requiring
a pretrained model download.

For production, replace with sentence-transformers / BGE / OpenAI embeddings.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np


class SimpleEncoder:
    """Deterministic bag-of-words encoder.

    >>> enc = SimpleEncoder(dim=64, seed=42)
    >>> v1 = enc("clinical research")
    >>> v2 = enc("clinical data analysis")
    >>> cos = np.dot(v1, v2)  # high — they share "clinical"
    """

    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self._cache: Dict[str, np.ndarray] = {}
        self._word_cache: Dict[str, np.ndarray] = {}

    def _word_vector(self, word: str) -> np.ndarray:
        if word not in self._word_cache:
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            rng = np.random.RandomState(h % (2**31) ^ self.seed)
            vec = rng.randn(self.dim)
            vec /= np.linalg.norm(vec) + 1e-12
            self._word_cache[word] = vec
        return self._word_cache[word]

    def __call__(self, text: str) -> np.ndarray:
        if text not in self._cache:
            words = text.lower().strip().split()
            if not words:
                self._cache[text] = np.zeros(self.dim)
            else:
                vec = sum(self._word_vector(w) for w in words)
                norm = np.linalg.norm(vec)
                self._cache[text] = vec / norm if norm > 1e-12 else vec
        return self._cache[text].copy()

    @property
    def embedding_dim(self) -> int:
        return self.dim

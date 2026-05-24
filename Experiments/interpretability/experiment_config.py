"""Shared encoder / MatchConfig factory for interpretability experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from config import MatchConfig
from encoder import SimpleEncoder


EncoderFn = Callable[[str], np.ndarray]


@dataclass
class ExperimentContext:
    encoder_name: str
    enc: EncoderFn
    embed_dim: int
    cfg: MatchConfig

    def encode(self, text: str) -> np.ndarray:
        return self.enc(text)


def build_encoder(
    name: str = "simple",
    simple_dim: int = 64,
    sbert_model: str = "BAAI/bge-base-en-v1.5",
) -> ExperimentContext:
    """Return an ExperimentContext for SimpleEncoder or SBERTEncoder."""
    name = name.lower().strip()
    if name in ("simple", "bow"):
        enc = SimpleEncoder(dim=simple_dim)
        return ExperimentContext(
            encoder_name=f"SimpleEncoder(dim={simple_dim})",
            enc=enc,
            embed_dim=simple_dim,
            cfg=MatchConfig(
                embedding_dim=simple_dim,
                tau_hard=0.3,
                temperature=0.1,
                tau_update=0.3,
            ),
        )
    if name in ("sbert", "bge", "bge-base"):
        try:
            from encoder_sbert import SBERTEncoder
        except ImportError as exc:
            raise ImportError(
                "SBERT encoder requires sentence-transformers. "
                "Install: pip install sentence-transformers"
            ) from exc
        sbert = SBERTEncoder(sbert_model)
        dim = int(sbert.embedding_dim)
        return ExperimentContext(
            encoder_name=f"SBERTEncoder({sbert_model})",
            enc=sbert,
            embed_dim=dim,
            cfg=MatchConfig(
                embedding_dim=dim,
                tau_hard=0.63,
                temperature=0.1,
                tau_update=0.63,
            ),
        )
    raise ValueError(f"Unknown encoder: {name!r} (use 'simple' or 'sbert')")

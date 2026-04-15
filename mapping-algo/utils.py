"""Vector math and attention — corresponds to §4A.2 of clawbot-nips.md"""

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 for zero vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 1e-12 else 0.0


def cosine_sim_batch(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query against multiple keys.

    query: (d,)   keys: (n, d)   → (n,)
    """
    qn = np.linalg.norm(query)
    if qn < 1e-12:
        return np.zeros(len(keys))
    kn = np.linalg.norm(keys, axis=1)
    denoms = np.where(qn * kn < 1e-12, 1.0, qn * kn)
    return (keys @ query) / denoms


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def attention_weighted_value(
    query_emb: np.ndarray,
    key_embs: np.ndarray,
    values: np.ndarray,
    temperature: float = 0.1,
) -> float:
    """Attention soft-matching — the fundamental building block.

    Computes:  p̃ = Σ_i α_i · v_i
    where α = softmax(sim(query, key_i) / τ)

    Returns 0.0 when key_embs is empty.
    """
    if len(key_embs) == 0:
        return 0.0
    sims = cosine_sim_batch(query_emb, key_embs)
    weights = softmax(sims / temperature)
    return float(np.dot(weights, values))

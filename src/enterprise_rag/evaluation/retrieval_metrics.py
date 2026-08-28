"""Deterministic information-retrieval metrics."""
from math import log2
from typing import Sequence

def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    items = list(retrieved[:k]); return sum(x in relevant for x in items) / k if k else 0.0

def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(x in relevant for x in retrieved[:k]) / len(relevant) if relevant else 0.0

def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant: return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: Sequence[str], judgments: dict[str, int], k: int) -> float:
    gains = [judgments.get(x, 0) for x in retrieved[:k]]
    dcg = sum((2 ** gain - 1) / log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum((2 ** gain - 1) / log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0

def compute_metrics(retrieved: Sequence[str], judgments: dict[str, int], k: int) -> dict[str, float]:
    relevant = {chunk_id for chunk_id, score in judgments.items() if score > 0}
    return {"precision_at_k": precision_at_k(retrieved, relevant, k), "recall_at_k": recall_at_k(retrieved, relevant, k), "mrr": reciprocal_rank(retrieved, relevant), "ndcg_at_k": ndcg_at_k(retrieved, judgments, k)}

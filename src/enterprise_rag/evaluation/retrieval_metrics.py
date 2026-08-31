"""Deterministic information-retrieval metrics."""
from math import log2
from typing import Sequence

def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0

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
    return {"hit_at_k": hit_at_k(retrieved, relevant, k), "precision_at_k": precision_at_k(retrieved, relevant, k), "recall_at_k": recall_at_k(retrieved, relevant, k), "mrr": reciprocal_rank(retrieved, relevant), "ndcg_at_k": ndcg_at_k(retrieved, judgments, k)}

def compute_retrieval_metrics(retrieved: Sequence[str], relevant: set[str], ks: Sequence[int] = (1, 3, 5)) -> dict[str, float | int | None]:
    first = next((rank for rank, item in enumerate(retrieved, 1) if item in relevant), None)
    values: dict[str, float | int | None] = {"first_relevant_rank": first, "reciprocal_rank": 1.0 / first if first else 0.0}
    for k in ks:
        values[f"hit_at_{k}"] = hit_at_k(retrieved, relevant, k)
        values[f"precision_at_{k}"] = precision_at_k(retrieved, relevant, k)
        values[f"recall_at_{k}"] = recall_at_k(retrieved, relevant, k)
    return values

def aggregate_metrics(rows: Sequence[dict[str, float | int | None]]) -> dict[str, float | int]:
    if not rows: return {"total_questions": 0, "successful_questions": 0}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    return {"total_questions": len(rows), "successful_questions": len(rows), **{key: sum(float(row[key]) for row in rows if isinstance(row.get(key), (int, float))) / len(rows) for key in keys}}

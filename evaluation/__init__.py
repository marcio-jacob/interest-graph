"""
evaluation/
===========
Offline evaluation pipeline for the hybrid recommendation system.

Metrics implemented (evaluation/metrics.py):
  Ranking accuracy : precision@k, recall@k, ndcg@k, mrr
  Diversity        : intra_list_diversity (ILD), catalog_entropy
  Novelty          : novelty_at_k (self-information)
  Coverage         : catalog coverage across all users

Usage
-----
    from evaluation.metrics import evaluate_recommendations

    results = evaluate_recommendations(
        ranked=["v1", "v3", "v7"],
        relevant={"v1", "v7"},
        topic_map={"v1": ["tech"], "v3": ["comedy"], "v7": ["tech"]},
        popularity={"v1": 0.05, "v3": 0.20, "v7": 0.03},
        k=3,
    )
    # {'precision@3': 0.667, 'recall@3': 1.0, 'ndcg@3': 0.865, ...}
"""

from .metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr,
    intra_list_diversity,
    novelty_at_k,
    coverage,
    catalog_entropy,
    evaluate_recommendations,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "intra_list_diversity",
    "novelty_at_k",
    "coverage",
    "catalog_entropy",
    "evaluate_recommendations",
]

"""
evaluation/metrics.py
=====================
Offline evaluation metrics for the hybrid recommendation pipeline.

Metrics
-------
Ranking accuracy
  precision_at_k    — fraction of top-k that are relevant
  recall_at_k       — fraction of all relevant items found in top-k
  ndcg_at_k         — normalised DCG (binary relevance)
  mrr               — mean reciprocal rank across multiple queries

Diversity
  intra_list_diversity — avg pairwise Jaccard distance on topics within top-k
  catalog_entropy      — Shannon entropy of a topic or creator distribution

Novelty
  novelty_at_k      — mean self-information: Σ -log₂(p(v)) / k

Coverage
  coverage          — fraction of catalog appearing across all recommendations

Aggregate helper
  evaluate_recommendations — runs all metrics for a single ranked list
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Ranking accuracy
# ---------------------------------------------------------------------------

def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for v in ranked[:k] if v in relevant)
    return hits / k


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of all relevant items found in the top-k recommendations."""
    if not relevant:
        return 0.0
    hits = sum(1 for v in ranked[:k] if v in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.
    Binary relevance: 1.0 if item is in `relevant`, 0.0 otherwise.
    """
    def _dcg(items: List[str]) -> float:
        return sum(
            (1.0 / math.log2(i + 2)) if v in relevant else 0.0
            for i, v in enumerate(items[:k])
        )

    dcg = _dcg(ranked)
    # Ideal DCG: top-k are all relevant (limited by |relevant|)
    ideal = _dcg(list(relevant)[:k])
    return dcg / ideal if ideal > 0 else 0.0


def mrr(
    ranked_lists: List[List[str]],
    relevant_sets: List[Set[str]],
) -> float:
    """
    Mean Reciprocal Rank across multiple queries.

    For each query, finds the rank (1-indexed) of the first relevant item.
    Returns the mean of 1/rank across all queries (0 if no hit).
    """
    if not ranked_lists:
        return 0.0
    rr_sum = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        for i, v in enumerate(ranked, start=1):
            if v in relevant:
                rr_sum += 1.0 / i
                break
    return rr_sum / len(ranked_lists)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def intra_list_diversity(
    ranked: List[str],
    topic_map: Dict[str, List[str]],
    k: int,
) -> float:
    """
    Intra-list diversity: average pairwise Jaccard distance between topic sets
    of all pairs of items in the top-k.

    Jaccard distance(A, B) = 1 - |topics(A) ∩ topics(B)| / |topics(A) ∪ topics(B)|

    Returns 0.0 if fewer than 2 items in top-k.
    """
    items = ranked[:k]
    if len(items) < 2:
        return 0.0
    pairs = 0
    total_dist = 0.0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            t_i = set(topic_map.get(items[i]) or [])
            t_j = set(topic_map.get(items[j]) or [])
            if not t_i and not t_j:
                dist = 0.0
            elif not t_i or not t_j:
                dist = 1.0
            else:
                union = t_i | t_j
                intersect = t_i & t_j
                dist = 1.0 - len(intersect) / len(union)
            total_dist += dist
            pairs += 1
    return total_dist / pairs if pairs > 0 else 0.0


def catalog_entropy(
    recommended_items: List[str],
    attribute_map: Dict[str, str],
) -> float:
    """
    Shannon entropy of a categorical attribute (topic or creator slug) across
    all recommended items.

    Higher entropy = more diverse recommendations.
    Returns 0.0 for an empty list.

    Parameters
    ----------
    recommended_items : flat list of video_ids
    attribute_map     : video_id → topic slug or creator username
    """
    counts = Counter(attribute_map.get(v, "unknown") for v in recommended_items)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------

def novelty_at_k(
    ranked: List[str],
    popularity: Dict[str, float],
    k: int,
) -> float:
    """
    Mean self-information of items in top-k.

    novelty(v) = -log₂(p(v))  where p(v) is the item's normalised popularity
    (fraction of users who watched it).  Items with lower popularity have
    higher novelty scores.

    Returns 0.0 for an empty list.
    """
    items = ranked[:k]
    if not items:
        return 0.0
    total = 0.0
    for v in items:
        p = max(popularity.get(v, 1e-6), 1e-10)
        total += -math.log2(p)
    return total / len(items)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage(all_recommended: List[str], catalog_size: int) -> float:
    """
    Fraction of the catalog covered by the union of all recommendations.

    Parameters
    ----------
    all_recommended : flat list of every video_id recommended across all users
    catalog_size    : total number of videos in the system
    """
    if catalog_size == 0:
        return 0.0
    return len(set(all_recommended)) / catalog_size


# ---------------------------------------------------------------------------
# Aggregate evaluator
# ---------------------------------------------------------------------------

def evaluate_recommendations(
    ranked: List[str],
    relevant: Set[str],
    topic_map: Dict[str, List[str]],
    popularity: Dict[str, float],
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute all pointwise and diversity metrics for a single ranked list.

    Parameters
    ----------
    ranked     : ordered list of recommended video_ids
    relevant   : ground-truth set of video_ids the user interacted with
    topic_map  : video_id → list of topic slugs
    popularity : video_id → fraction of users who watched it [0, 1]
    k          : ranking cutoff

    Returns
    -------
    dict with keys: precision@k, recall@k, ndcg@k, ild@k, novelty@k
    """
    return {
        f"precision@{k}": round(precision_at_k(ranked, relevant, k), 4),
        f"recall@{k}":    round(recall_at_k(ranked, relevant, k), 4),
        f"ndcg@{k}":      round(ndcg_at_k(ranked, relevant, k), 4),
        f"ild@{k}":       round(intra_list_diversity(ranked, topic_map, k), 4),
        f"novelty@{k}":   round(novelty_at_k(ranked, popularity, k), 4),
    }

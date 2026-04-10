"""
ranking/reranker.py
===================
Feature reranker that combines signals from all candidate generators.

The reranker uses a transparent, configurable feature blend that can be tuned
offline and replaced with a gradient-boosted tree or neural scorer later. Every
ranked item carries a full explanation trace showing the contribution of each
feature component.

Final score formula (per candidate):
    final = (normalised_raw × engine_weight)
          + (session_alignment × SESSION_WEIGHT)
          + exploration_bonus

Where:
  normalised_raw   — engine score rescaled to [0, 1] within its own engine pool
  engine_weight    — configurable per-engine bias knob
  session_alignment — topic_affinity(candidate_topics, session) ∈ [-1, 1]
  exploration_bonus — small additive reward for serendipitous cross-topic items
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .candidates import Candidate
from .session_encoder import SessionFeatures


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default engine weights — higher = more influence in the final blend.
# collab_filter gets a lift because peer-validated signals reduce false positives.
# trending gets the lowest weight because it has no personalisation.
DEFAULT_ENGINE_WEIGHTS: Dict[str, float] = {
    "content_based": 1.0,
    "collab_filter":  1.2,
    "embedding":      0.9,
    "trending":       0.6,
    "creator":        0.8,
}

# Contribution of session alignment to the final score
SESSION_WEIGHT = 0.3

# Additive bonus for candidates whose topics are all outside the user's top-3
EXPLORATION_BONUS = 0.05


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RankedCandidate:
    """A candidate that has been scored and ranked by the reranker."""
    video_id: str
    source_engine: str
    raw_score: float
    final_score: float
    rank: int
    explanation: str
    explanation_trace: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rank":          self.rank,
            "video_id":      self.video_id,
            "source_engine": self.source_engine,
            "raw_score":     round(self.raw_score, 4),
            "final_score":   round(self.final_score, 4),
            "explanation":   self.explanation,
            "trace":         self.explanation_trace,
            "metadata":      self.metadata,
        }


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class FeatureReranker:
    """
    Combines multi-engine candidates into a single ranked feed.

    Parameters
    ----------
    engine_weights :
        Dict mapping engine_name → multiplicative weight.
        Defaults to DEFAULT_ENGINE_WEIGHTS.
    exploration_bonus :
        Additive score given to candidates whose topic is NOT in the user's
        top-3 long-term interests (serendipity incentive).
    """

    def __init__(
        self,
        engine_weights: Dict[str, float] | None = None,
        exploration_bonus: float = EXPLORATION_BONUS,
    ) -> None:
        self.engine_weights = engine_weights or DEFAULT_ENGINE_WEIGHTS
        self.exploration_bonus = exploration_bonus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: List[Candidate],
        session: SessionFeatures,
    ) -> List[RankedCandidate]:
        """
        Deduplicate, score, and sort candidates.

        When the same video appears from multiple engines, the one with the
        highest raw_score is kept (its source_engine is preserved).

        Parameters
        ----------
        candidates : all Candidates from all generators (may have duplicates)
        session    : SessionFeatures for the target user

        Returns
        -------
        Deduplicated list of RankedCandidate sorted by final_score DESC.
        """
        # Deduplicate: keep highest raw_score per video
        best: dict[str, Candidate] = {}
        for c in candidates:
            if c.video_id not in best or c.raw_score > best[c.video_id].raw_score:
                best[c.video_id] = c

        # Build per-engine [min, max] for normalisation
        engine_pools: dict[str, list[float]] = {}
        for c in best.values():
            engine_pools.setdefault(c.source_engine, []).append(c.raw_score)

        engine_bounds: dict[str, tuple[float, float]] = {
            eng: (min(sc), max(sc))
            for eng, sc in engine_pools.items()
        }

        # Top-3 long-term topics for serendipity detection
        top_topics = set(
            sorted(
                session.long_term_topics,
                key=session.long_term_topics.get,
                reverse=True,
            )[:3]
        )

        # Score every unique candidate
        ranked: list[RankedCandidate] = []
        for c in best.values():
            lo, hi = engine_bounds.get(c.source_engine, (0.0, 1.0))
            norm_raw = self._normalise(c.raw_score, lo, hi)
            session_align = session.topic_affinity(c.metadata.get("topics") or [])
            engine_w = self.engine_weights.get(c.source_engine, 1.0)

            cand_topics = set(c.metadata.get("topics") or [])
            is_serendipitous = bool(cand_topics) and cand_topics.isdisjoint(top_topics)
            exploration = self.exploration_bonus if is_serendipitous else 0.0

            final = norm_raw * engine_w + session_align * SESSION_WEIGHT + exploration

            trace = {
                "norm_raw_score":    round(norm_raw, 4),
                "engine_weight":     engine_w,
                "session_alignment": round(session_align, 4),
                "serendipitous":     is_serendipitous,
                "exploration_bonus": round(exploration, 4),
            }

            ranked.append(RankedCandidate(
                video_id=c.video_id,
                source_engine=c.source_engine,
                raw_score=c.raw_score,
                final_score=final,
                rank=0,
                explanation=c.explanation,
                explanation_trace=trace,
                metadata=c.metadata,
            ))

        # Sort and assign ranks
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        for i, r in enumerate(ranked, start=1):
            r.rank = i

        return ranked

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.5
        return (value - lo) / (hi - lo)

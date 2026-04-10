"""
ranking/
========
Hybrid recommendation pipeline.

Exports the main entry point:
    from ranking.pipeline import RecommendationPipeline

Public API summary:
    Candidate        — structured output from any single engine
    CandidateGenerator — ABC that all engines implement
    RankedCandidate  — output of the reranker (final_score, rank, trace)
    FeatureReranker  — combines multi-engine candidates into a ranked feed
    SessionFeatures  — session-aware context (short + long-term topics)
    RecommendationPipeline — top-level orchestrator
"""

from .candidates import (
    Candidate,
    CandidateGenerator,
    ContentBasedGenerator,
    CollaborativeFilteringGenerator,
    EmbeddingRetrievalGenerator,
    TrendingGenerator,
    CreatorBasedGenerator,
    build_all_generators,
)
from .reranker import FeatureReranker, RankedCandidate
from .session_encoder import SessionFeatures, encode_session
from .pipeline import RecommendationPipeline

__all__ = [
    "Candidate",
    "CandidateGenerator",
    "ContentBasedGenerator",
    "CollaborativeFilteringGenerator",
    "EmbeddingRetrievalGenerator",
    "TrendingGenerator",
    "CreatorBasedGenerator",
    "build_all_generators",
    "FeatureReranker",
    "RankedCandidate",
    "SessionFeatures",
    "encode_session",
    "RecommendationPipeline",
]

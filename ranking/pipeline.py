"""
ranking/pipeline.py
===================
Top-level orchestrator for the hybrid recommendation pipeline.

Wires together:
  1. All 5 candidate generators (content-based, CF, embedding, trending, creator)
  2. Session-aware feature encoder (short-term + long-term user context)
  3. Feature reranker (normalised raw scores + session alignment + exploration)

Usage
-----
    from ranking.pipeline import RecommendationPipeline

    pipeline = RecommendationPipeline()
    feed = pipeline.recommend("6c0f0b07-ce08-4f2b-b9f6-e323acdefc7e", n=10)
    for item in feed:
        print(item.rank, item.source_engine, item.video_id, item.explanation)

    # Pretty-print with full traces:
    pipeline.explain("6c0f0b07-ce08-4f2b-b9f6-e323acdefc7e", n=10)
"""
from __future__ import annotations

import os
from typing import Dict, List

from dotenv import load_dotenv

from .candidates import CandidateGenerator, build_all_generators
from .reranker import FeatureReranker, RankedCandidate
from .session_encoder import SessionFeatures, encode_session

load_dotenv()


class RecommendationPipeline:
    """
    Hybrid recommendation pipeline.

    Parameters
    ----------
    uri, user, password :
        Neo4j connection credentials. Default: read from .env
        (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).
    candidates_per_engine :
        How many candidates each engine produces before reranking.
        More candidates = better coverage at the cost of extra queries.
    engine_weights :
        Override the reranker's default per-engine blend weights.
    exploration_bonus :
        Additive serendipity reward for cross-topic candidates.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        candidates_per_engine: int = 20,
        engine_weights: Dict[str, float] | None = None,
        exploration_bonus: float = 0.05,
    ) -> None:
        self._uri      = uri      or os.environ["NEO4J_URI"]
        self._user     = user     or os.environ["NEO4J_USER"]
        self._password = password or os.environ["NEO4J_PASSWORD"]
        self._candidates_per_engine = candidates_per_engine

        self._generators: list[CandidateGenerator] = build_all_generators(
            self._uri, self._user, self._password
        )
        self._reranker = FeatureReranker(
            engine_weights=engine_weights,
            exploration_bonus=exploration_bonus,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        verbose: bool = False,
    ) -> List[RankedCandidate]:
        """
        Generate a ranked feed of `n` recommendations for the given user.

        Returns a list of RankedCandidate sorted by final_score DESC.
        Each item carries full explanation text and a trace dict for debugging.
        """
        # 1. Encode session context
        session = encode_session(
            user_id,
            self._uri,
            self._user,
            self._password,
        )
        if verbose:
            print(
                f"Session encoded: {len(session.recent_video_ids)} recent videos,"
                f" avg_cr={session.avg_completion:.2f},"
                f" top_topics={list(session.long_term_topics)[:3]}"
            )

        # 2. Generate candidates from all engines
        all_candidates = []
        for gen in self._generators:
            try:
                candidates = gen.generate(user_id, limit=self._candidates_per_engine)
                all_candidates.extend(candidates)
                if verbose:
                    print(f"  {gen.engine_name}: {len(candidates)} candidates")
            except Exception as exc:
                print(f"  Warning: {gen.engine_name} failed — {exc}")

        if verbose:
            unique = len({c.video_id for c in all_candidates})
            print(f"Total candidates: {len(all_candidates)} ({unique} unique)")

        # 3. Rerank and return top-n
        ranked = self._reranker.rerank(all_candidates, session)
        return ranked[:n]

    def recommend_with_session(
        self,
        user_id: str,
        n: int = 10,
    ) -> tuple[List[RankedCandidate], SessionFeatures]:
        """Like recommend(), but also returns the encoded SessionFeatures."""
        session = encode_session(
            user_id, self._uri, self._user, self._password
        )
        all_candidates = []
        for gen in self._generators:
            try:
                all_candidates.extend(
                    gen.generate(user_id, limit=self._candidates_per_engine)
                )
            except Exception as exc:
                print(f"  Warning: {gen.engine_name} failed — {exc}")
        ranked = self._reranker.rerank(all_candidates, session)
        return ranked[:n], session

    def explain(self, user_id: str, n: int = 10) -> None:
        """Pretty-print a ranked feed with full explanation traces."""
        feed = self.recommend(user_id, n=n, verbose=True)
        print(f"\n{'='*72}")
        print(f"Top-{n} recommendations for user {user_id}")
        print(f"{'='*72}")
        for item in feed:
            print(f"\n#{item.rank:02d}  [{item.source_engine:<15}]  {item.video_id}")
            print(f"      Final {item.final_score:.4f}  |  raw {item.raw_score:.4f}")
            print(f"      {item.explanation}")
            topics = item.metadata.get("topics") or []
            if topics:
                print(f"      Topics : {', '.join(topics)}")
            creator = item.metadata.get("creator")
            if creator and creator != "unknown":
                print(f"      Creator: {creator}")
            desc = item.metadata.get("description", "")
            if desc:
                print(f"      '{desc[:68]}...'")
            t = item.explanation_trace
            print(
                f"      Trace  : norm={t.get('norm_raw_score', 0):.3f}"
                f" × w={t.get('engine_weight', 1):.1f}"
                f" + align={t.get('session_alignment', 0):.3f}"
                f" + explore={t.get('exploration_bonus', 0):.3f}"
            )

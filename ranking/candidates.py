"""
ranking/candidates.py
=====================
Defines the unified Candidate interface that all recommendation engines output.
Each engine wraps its Cypher queries and produces a list of Candidates with
structured explanations for downstream reranking and logging.

Engines
-------
1. ContentBasedGenerator      — topic × entity × hashtag relevance (graph path)
2. CollaborativeFilteringGenerator — peer SIMILAR_TO liked videos
3. EmbeddingRetrievalGenerator — Node2Vec cosine similarity (batched Python)
4. TrendingGenerator           — global engagement × completion × like boost
5. CreatorBasedGenerator       — creator topic-match + social proof + PageRank
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A single video candidate produced by one recommendation engine."""
    video_id: str
    source_engine: str        # identifier for the engine that produced this
    raw_score: float          # engine's native score (scale varies per engine)
    explanation: str          # human-readable text for UI/logging
    metadata: dict = field(default_factory=dict)   # engine-specific signals


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CandidateGenerator(ABC):
    """
    Abstract base for all recommendation engines.

    Each concrete generator connects to Neo4j, runs its query, and returns a
    list of Candidate objects with a consistent schema.  The caller (pipeline)
    deduplicates and reranks across all generators.
    """
    engine_name: ClassVar[str]

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password

    def _run_query(self, query: str, **params) -> list[dict]:
        """Execute a Cypher query and return all records as dicts."""
        with GraphDatabase.driver(self._uri, auth=(self._user, self._password)) as drv:
            with drv.session() as session:
                result = session.run(query, **params)
                return [dict(r) for r in result]

    @abstractmethod
    def generate(self, user_id: str, limit: int = 20) -> list[Candidate]:
        """Return up to `limit` candidates for the given user."""


# ---------------------------------------------------------------------------
# Engine 1 — Content-Based (Topic × Entity × Hashtag)
# ---------------------------------------------------------------------------

_CONTENT_BASED_QUERY = """
MATCH (u:User {user_id: $uid})-[ti:INTERESTED_IN_TOPIC]->(t:Topic)<-[:IS_ABOUT]-(v:Video)
WHERE NOT EXISTS {
    MATCH (u)-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(v)
}
WITH u, v, sum(ti.topic_score) AS ts

OPTIONAL MATCH (u)-[ei:INTERESTED_IN_ENTITY]->(e:Entity)<-[:MENTIONS]-(v)
WITH u, v, ts, sum(coalesce(ei.entity_score, 0)) AS er

OPTIONAL MATCH (u)-[hi:INTERESTED_IN_HASHTAG]->(hh:Hashtag)<-[:HAS_HASHTAG]-(v)
WITH v, ts AS ts2, er AS er2, sum(coalesce(hi.hashtag_score, 0)) AS hr

WITH v, ts2 + er2 * 0.6 + hr * 0.4 AS relevance_score
ORDER BY relevance_score DESC LIMIT $limit

OPTIONAL MATCH (v)-[:IS_ABOUT]->(t2:Topic)
OPTIONAL MATCH (v)-[:CREATED_BY]->(c:Creator)
RETURN v.video_id        AS video_id,
       v.description     AS description,
       relevance_score,
       collect(DISTINCT t2.slug) AS topics,
       c.username         AS creator
"""


class ContentBasedGenerator(CandidateGenerator):
    """Engine 1: Weighted topic × entity × hashtag relevance via graph traversal."""
    engine_name = "content_based"

    def generate(self, user_id: str, limit: int = 20) -> list[Candidate]:
        rows = self._run_query(_CONTENT_BASED_QUERY, uid=user_id, limit=limit)
        candidates = []
        for r in rows:
            topics = r.get("topics") or []
            creator = r.get("creator") or "unknown"
            topic_str = ", ".join(topics[:2]) if topics else "this topic"
            explanation = (
                f"Matches your interest in {topic_str}"
                f" (score {r['relevance_score']:.3f})"
            )
            candidates.append(Candidate(
                video_id=r["video_id"],
                source_engine=self.engine_name,
                raw_score=float(r["relevance_score"]),
                explanation=explanation,
                metadata={
                    "topics": topics,
                    "creator": creator,
                    "description": (r.get("description") or "")[:80],
                },
            ))
        return candidates


# ---------------------------------------------------------------------------
# Engine 2 — Collaborative Filtering (SIMILAR_TO peers)
# ---------------------------------------------------------------------------

_COLLAB_QUERY = """
MATCH (u:User {user_id: $uid})-[sim:SIMILAR_TO]->(peer:User)
WHERE sim.similarity >= $min_sim

MATCH (peer)-[:HAS_SESSION]->(:UserSession)-[:LIKED]->(v:Video)
WHERE NOT EXISTS {
    MATCH (u)-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(v)
}
WITH v,
     sum(sim.similarity)   AS collab_score,
     count(DISTINCT peer)  AS peer_count

ORDER BY collab_score DESC LIMIT $limit

OPTIONAL MATCH (v)-[:IS_ABOUT]->(t:Topic)
OPTIONAL MATCH (v)-[:CREATED_BY]->(c:Creator)
RETURN v.video_id       AS video_id,
       v.description    AS description,
       collab_score,
       peer_count,
       collect(DISTINCT t.slug) AS topics,
       c.username        AS creator
"""


class CollaborativeFilteringGenerator(CandidateGenerator):
    """Engine 2: Collaborative filtering via GDS-computed SIMILAR_TO peers."""
    engine_name = "collab_filter"

    def __init__(self, uri: str, user: str, password: str, min_sim: float = 0.3) -> None:
        super().__init__(uri, user, password)
        self._min_sim = min_sim

    def generate(self, user_id: str, limit: int = 20) -> list[Candidate]:
        rows = self._run_query(_COLLAB_QUERY, uid=user_id, limit=limit, min_sim=self._min_sim)
        candidates = []
        for r in rows:
            topics = r.get("topics") or []
            creator = r.get("creator") or "unknown"
            peer_count = r.get("peer_count", 0)
            explanation = (
                f"Liked by {peer_count} user{'s' if peer_count != 1 else ''} similar to you"
                f" (combined similarity {r['collab_score']:.3f})"
            )
            candidates.append(Candidate(
                video_id=r["video_id"],
                source_engine=self.engine_name,
                raw_score=float(r["collab_score"]),
                explanation=explanation,
                metadata={
                    "topics": topics,
                    "creator": creator,
                    "peer_count": peer_count,
                    "description": (r.get("description") or "")[:80],
                },
            ))
        return candidates


# ---------------------------------------------------------------------------
# Engine 3 — Embedding Retrieval (Node2Vec cosine similarity)
# ---------------------------------------------------------------------------

_USER_EMB_QUERY = """
MATCH (u:User {user_id: $uid})
RETURN u.node2vec_embedding AS embedding
"""

_UNSEEN_VIDEO_EMBS_QUERY = """
MATCH (v:Video)
WHERE v.node2vec_embedding IS NOT NULL
  AND NOT EXISTS {
      MATCH (:User {user_id: $uid})-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(v)
  }
RETURN v.video_id AS video_id, v.node2vec_embedding AS embedding
SKIP $skip LIMIT $batch_size
"""

_VIDEO_META_QUERY = """
UNWIND $video_ids AS vid
MATCH (v:Video {video_id: vid})
OPTIONAL MATCH (v)-[:IS_ABOUT]->(t:Topic)
OPTIONAL MATCH (v)-[:CREATED_BY]->(c:Creator)
RETURN v.video_id   AS video_id,
       v.description AS description,
       collect(DISTINCT t.slug) AS topics,
       c.username    AS creator
"""


class EmbeddingRetrievalGenerator(CandidateGenerator):
    """
    Engine 3: Node2Vec cosine similarity between user and video embeddings.

    Embeddings are fetched in batches of `batch_size` and cosine similarity is
    computed client-side with NumPy to stay within the Aura free-tier 278 MB
    transaction memory limit.
    """
    engine_name = "embedding"

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        batch_size: int = 500,
    ) -> None:
        super().__init__(uri, user, password)
        self._batch_size = batch_size

    def generate(self, user_id: str, limit: int = 20) -> list[Candidate]:
        # Fetch user embedding
        rows = self._run_query(_USER_EMB_QUERY, uid=user_id)
        if not rows or rows[0]["embedding"] is None:
            return []
        user_emb = np.array(rows[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(user_emb)
        if norm == 0:
            return []
        user_emb /= norm

        # Scan all unseen video embeddings in batches
        scores: dict[str, float] = {}
        skip = 0
        while True:
            batch = self._run_query(
                _UNSEEN_VIDEO_EMBS_QUERY,
                uid=user_id,
                skip=skip,
                batch_size=self._batch_size,
            )
            if not batch:
                break
            for r in batch:
                emb = r.get("embedding")
                if emb is None:
                    continue
                v = np.array(emb, dtype=np.float32)
                n = np.linalg.norm(v)
                if n == 0:
                    continue
                scores[r["video_id"]] = float(np.dot(user_emb, v / n))
            skip += self._batch_size

        if not scores:
            return []

        # Take top candidates by cosine similarity
        top_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]

        # Fetch metadata for top videos
        meta_rows = self._run_query(_VIDEO_META_QUERY, video_ids=top_ids)
        meta_by_id = {r["video_id"]: r for r in meta_rows}

        candidates = []
        for vid_id in top_ids:
            cosine = scores[vid_id]
            meta = meta_by_id.get(vid_id, {})
            topics = meta.get("topics") or []
            creator = meta.get("creator") or "unknown"
            explanation = (
                f"Structurally close to your viewing pattern"
                f" (cosine {cosine:.4f})"
            )
            candidates.append(Candidate(
                video_id=vid_id,
                source_engine=self.engine_name,
                raw_score=cosine,
                explanation=explanation,
                metadata={
                    "topics": topics,
                    "creator": creator,
                    "cosine": cosine,
                    "description": (meta.get("description") or "")[:80],
                },
            ))
        return candidates


# ---------------------------------------------------------------------------
# Engine 4 — Trending (global engagement signal)
# ---------------------------------------------------------------------------

_TRENDING_QUERY = """
MATCH (s:UserSession)-[vr:VIEWED]->(v:Video)
WITH v,
     count(vr)                AS view_count,
     avg(vr.completion_rate)  AS avg_cr

OPTIONAL MATCH (:UserSession)-[:LIKED]->(v)
WITH v, view_count, avg_cr, count(*) AS like_count

WITH v,
     toFloat(view_count) * avg_cr
     * (1.0 + 0.3 * toFloat(like_count) / (toFloat(view_count) + 1.0))
     AS trending_score

ORDER BY trending_score DESC LIMIT $limit

OPTIONAL MATCH (v)-[:IS_ABOUT]->(t:Topic)
OPTIONAL MATCH (v)-[:CREATED_BY]->(c:Creator)
RETURN v.video_id       AS video_id,
       v.description    AS description,
       trending_score,
       view_count,
       like_count,
       collect(DISTINCT t.slug) AS topics,
       c.username        AS creator
"""


class TrendingGenerator(CandidateGenerator):
    """Engine 4: Global trending — high engagement × completion × like boost."""
    engine_name = "trending"

    def generate(self, user_id: str, limit: int = 20) -> list[Candidate]:
        rows = self._run_query(_TRENDING_QUERY, limit=limit)
        candidates = []
        for r in rows:
            topics = r.get("topics") or []
            creator = r.get("creator") or "unknown"
            explanation = (
                f"Trending globally — {r['view_count']} views,"
                f" {r.get('like_count', 0)} likes"
            )
            candidates.append(Candidate(
                video_id=r["video_id"],
                source_engine=self.engine_name,
                raw_score=float(r["trending_score"]),
                explanation=explanation,
                metadata={
                    "topics": topics,
                    "creator": creator,
                    "view_count": r.get("view_count", 0),
                    "like_count": r.get("like_count", 0),
                    "description": (r.get("description") or "")[:80],
                },
            ))
        return candidates


# ---------------------------------------------------------------------------
# Engine 5 — Creator-Based
# ---------------------------------------------------------------------------

_CREATOR_QUERY = """
MATCH (u:User {user_id: $uid})

OPTIONAL MATCH (u)-[:FOLLOWS]->(already:Creator)
WITH u, collect(already.user_id) AS following_ids

MATCH (u)-[ti:INTERESTED_IN_TOPIC]->(t:Topic)<-[:IS_ABOUT]-(v:Video)-[:CREATED_BY]->(c:Creator)
WHERE NOT c.user_id IN following_ids
  AND c.user_id <> u.user_id

WITH u, c, sum(ti.topic_score) AS topic_match, following_ids

OPTIONAL MATCH (u)-[:FOLLOWS]->(friend:User)-[fol:FOLLOWS]->(c)
WITH c,
     topic_match,
     coalesce(sum(fol.engagement_score), 0) AS social_boost,
     coalesce(c.pagerank_score, 0)           AS centrality

WITH c,
     topic_match,
     social_boost,
     centrality,
     topic_match + social_boost * 0.5 + centrality * 0.3 AS creator_score

ORDER BY creator_score DESC LIMIT $limit

MATCH (v2:Video)-[:CREATED_BY]->(c)
WITH c, creator_score, topic_match, social_boost, centrality,
     v2 ORDER BY coalesce(v2.view_pagerank, 0) DESC
WITH c, creator_score, topic_match, social_boost, centrality,
     collect(v2)[0] AS v2

OPTIONAL MATCH (v2)-[:IS_ABOUT]->(t2:Topic)
RETURN c.user_id      AS creator_id,
       c.username     AS creator,
       creator_score,
       topic_match,
       social_boost,
       centrality,
       v2.video_id    AS video_id,
       v2.description AS description,
       collect(DISTINCT t2.slug) AS topics
"""


class CreatorBasedGenerator(CandidateGenerator):
    """Engine 5: Creator recommendation via topic overlap + social proof + PageRank."""
    engine_name = "creator"

    def generate(self, user_id: str, limit: int = 20) -> list[Candidate]:
        rows = self._run_query(_CREATOR_QUERY, uid=user_id, limit=limit)
        candidates = []
        for r in rows:
            if not r.get("video_id"):
                continue
            topics = r.get("topics") or []
            creator = r.get("creator") or "unknown"
            explanation = (
                f"Creator {creator} matches your interests"
                f" (topic {r['topic_match']:.2f}, pagerank {r['centrality']:.2f})"
            )
            candidates.append(Candidate(
                video_id=r["video_id"],
                source_engine=self.engine_name,
                raw_score=float(r["creator_score"]),
                explanation=explanation,
                metadata={
                    "topics": topics,
                    "creator": creator,
                    "creator_id": r.get("creator_id"),
                    "topic_match": r.get("topic_match", 0),
                    "social_boost": r.get("social_boost", 0),
                    "pagerank": r.get("centrality", 0),
                    "description": (r.get("description") or "")[:80],
                },
            ))
        return candidates


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_all_generators(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> list[CandidateGenerator]:
    """
    Instantiate all 5 candidate generators from environment variables
    (or explicit overrides).
    """
    uri      = uri      or os.environ["NEO4J_URI"]
    user     = user     or os.environ["NEO4J_USER"]
    password = password or os.environ["NEO4J_PASSWORD"]
    return [
        ContentBasedGenerator(uri, user, password),
        CollaborativeFilteringGenerator(uri, user, password),
        EmbeddingRetrievalGenerator(uri, user, password),
        TrendingGenerator(uri, user, password),
        CreatorBasedGenerator(uri, user, password),
    ]

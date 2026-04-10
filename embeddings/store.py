"""
embeddings/store.py
===================
Embedding fetch and cosine similarity utilities.

Provides a thin wrapper around Neo4j embedding reads with an optional
in-memory cache so the same user/video embedding is not re-fetched within a
single pipeline run.

Background
----------
The Neo4j Aura free tier enforces a 278 MB transaction memory limit.
Running gds.similarity.cosine() over 4,000 video embeddings in one Cypher
query exceeds this limit.  This module fetches embeddings in configurable
batches (default 500) and computes cosine similarity client-side with NumPy.

The EmbeddingRetrievalGenerator in ranking/candidates.py uses the same
batched approach; this store adds cross-request caching for use cases that
call the embedding engine multiple times per session.
"""
from __future__ import annotations

import numpy as np
from neo4j import GraphDatabase


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

_USER_EMB_Q = """
MATCH (u:User {user_id: $uid})
RETURN u.node2vec_embedding AS embedding
"""

_BATCH_UNSEEN_Q = """
MATCH (v:Video)
WHERE v.node2vec_embedding IS NOT NULL
  AND NOT EXISTS {
      MATCH (:User {user_id: $uid})-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(v)
  }
RETURN v.video_id AS video_id, v.node2vec_embedding AS embedding
SKIP $skip LIMIT $batch_size
"""

_VIDEO_EMB_Q = """
MATCH (v:Video {video_id: $vid})
RETURN v.node2vec_embedding AS embedding
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class EmbeddingStore:
    """
    In-memory cache for Node2Vec embeddings with cosine similarity helpers.

    Parameters
    ----------
    uri, user, password : Neo4j connection credentials
    batch_size          : number of video embeddings fetched per transaction
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        batch_size: int = 500,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._batch_size = batch_size
        self._user_cache: dict[str, np.ndarray | None] = {}
        self._video_cache: dict[str, np.ndarray | None] = {}

    # ------------------------------------------------------------------
    # Fetch & cache
    # ------------------------------------------------------------------

    def get_user_embedding(self, user_id: str) -> np.ndarray | None:
        """
        Return the L2-normalised Node2Vec embedding for a user.
        Result is cached for the lifetime of this store instance.
        """
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        with GraphDatabase.driver(self._uri, auth=(self._user, self._password)) as drv:
            with drv.session() as sess:
                row = sess.run(_USER_EMB_Q, uid=user_id).single()
                emb = None
                if row and row["embedding"] is not None:
                    raw = np.array(row["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(raw)
                    emb = raw / norm if norm > 0 else None
                self._user_cache[user_id] = emb
                return emb

    def get_video_embedding(self, video_id: str) -> np.ndarray | None:
        """
        Return the L2-normalised Node2Vec embedding for a video.
        Result is cached for the lifetime of this store instance.
        """
        if video_id in self._video_cache:
            return self._video_cache[video_id]

        with GraphDatabase.driver(self._uri, auth=(self._user, self._password)) as drv:
            with drv.session() as sess:
                row = sess.run(_VIDEO_EMB_Q, vid=video_id).single()
                emb = None
                if row and row["embedding"] is not None:
                    raw = np.array(row["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(raw)
                    emb = raw / norm if norm > 0 else None
                self._video_cache[video_id] = emb
                return emb

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    def cosine_score(self, user_id: str, video_id: str) -> float:
        """
        Cosine similarity between a user and a video embedding.
        Returns 0.0 if either embedding is missing.
        """
        u = self.get_user_embedding(user_id)
        v = self.get_video_embedding(video_id)
        if u is None or v is None:
            return 0.0
        return float(np.dot(u, v))

    def cosine_top_k_unseen(
        self,
        user_id: str,
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k unseen videos by cosine similarity to the user embedding.

        Videos are fetched in batches to avoid OOM on the Aura free tier.
        Caches video embeddings for reuse within the same store instance.

        Returns
        -------
        List of (video_id, cosine_score) sorted descending.
        """
        user_emb = self.get_user_embedding(user_id)
        if user_emb is None:
            return []

        scores: dict[str, float] = {}

        with GraphDatabase.driver(self._uri, auth=(self._user, self._password)) as drv:
            skip = 0
            while True:
                with drv.session() as sess:
                    rows = sess.run(
                        _BATCH_UNSEEN_Q,
                        uid=user_id,
                        skip=skip,
                        batch_size=self._batch_size,
                    ).data()
                if not rows:
                    break
                for r in rows:
                    emb = r.get("embedding")
                    if emb is None:
                        continue
                    v = np.array(emb, dtype=np.float32)
                    n = np.linalg.norm(v)
                    if n == 0:
                        continue
                    vid = r["video_id"]
                    unit_v = v / n
                    self._video_cache[vid] = unit_v
                    scores[vid] = float(np.dot(user_emb, unit_v))
                skip += self._batch_size

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def cache_stats(self) -> dict:
        """Return cache hit counts."""
        return {
            "users_cached": len(self._user_cache),
            "videos_cached": len(self._video_cache),
        }

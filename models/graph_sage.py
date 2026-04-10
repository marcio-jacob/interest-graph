"""
models/graph_sage.py
====================
Two-hop GraphSAGE mean aggregator for user–video link prediction.

Uses Node2Vec embeddings (stored as float[] on User and Video nodes) as
initial node features. The aggregation and scoring are implemented in NumPy,
making this runnable on any machine without a GPU or PyTorch installation.

Architecture
------------
Layer 0 (init):
    h_v⁰ = node2vec_embedding(v)       [64-dim]

Layer 1 (1-hop mean aggregation):
    h_v¹ = mean { h_u⁰ | u ∈ N(v) }   [64-dim]

Representation:
    h_v  = concat(h_v⁰, h_v¹)         [128-dim]

Link-prediction score:
    score(u, v) = σ(h_u · W · h_v)
    where W ∈ ℝ¹²⁸ˣ¹²⁸ and σ is the sigmoid function.

W is initialised to the identity matrix. Replace with trained weights by
saving a NumPy array to models/weights/graph_sage_W.npy:

    import numpy as np
    np.save("models/weights/graph_sage_W.npy", trained_W)

    scorer = GraphSAGEScorer(uri, user, pw, weight_path="models/weights/graph_sage_W.npy")

Neighbour queries
-----------------
Users : FOLLOWS and SIMILAR_TO neighbours (social graph)
Videos: IS_ABOUT, HAS_HASHTAG, CREATED_BY neighbours (content graph)
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

_VIDEO_EMB_Q = """
MATCH (v:Video {video_id: $vid})
RETURN v.node2vec_embedding AS embedding
"""

_USER_NEIGHBOURS_Q = """
MATCH (u:User {user_id: $uid})-[:FOLLOWS|SIMILAR_TO]-(nb)
WHERE nb.node2vec_embedding IS NOT NULL
RETURN nb.node2vec_embedding AS embedding
LIMIT 20
"""

_VIDEO_NEIGHBOURS_Q = """
MATCH (v:Video {video_id: $vid})-[:IS_ABOUT|HAS_HASHTAG|CREATED_BY]-(nb)
WHERE nb.node2vec_embedding IS NOT NULL
RETURN nb.node2vec_embedding AS embedding
LIMIT 20
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_emb(raw, dim: int = 64) -> np.ndarray:
    if raw is None:
        return np.zeros(dim, dtype=np.float32)
    return np.array(raw, dtype=np.float32)


def _mean_pool(embeddings: list, dim: int = 64) -> np.ndarray:
    if not embeddings:
        return np.zeros(dim, dtype=np.float32)
    arrays = [_safe_emb(e, dim) for e in embeddings]
    return np.mean(arrays, axis=0).astype(np.float32)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class GraphSAGEScorer:
    """
    Two-hop GraphSAGE mean aggregator for user–video link prediction.

    Parameters
    ----------
    uri, user, password : Neo4j connection credentials
    weight_path : path to a .npy file containing a trained W matrix (128×128).
                  If None, W is the identity matrix (untrained baseline).
    dim         : node embedding dimension (must match node2vec_embedding size)
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        weight_path: str | None = None,
        dim: int = 64,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._dim = dim
        full_dim = dim * 2  # concat of h0 and h1
        if weight_path:
            self._W: np.ndarray = np.load(weight_path).astype(np.float32)
        else:
            self._W = np.eye(full_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self, query: str, **params) -> list:
        with GraphDatabase.driver(self._uri, auth=(self._user, self._password)) as drv:
            with drv.session() as sess:
                return [r["embedding"] for r in sess.run(query, **params)]

    def _represent_user(self, user_id: str) -> np.ndarray:
        h0 = _safe_emb(
            next(iter(self._fetch(_USER_EMB_Q, uid=user_id)), None), self._dim
        )
        h1 = _mean_pool(self._fetch(_USER_NEIGHBOURS_Q, uid=user_id), self._dim)
        return np.concatenate([h0, h1])

    def _represent_video(self, video_id: str) -> np.ndarray:
        h0 = _safe_emb(
            next(iter(self._fetch(_VIDEO_EMB_Q, vid=video_id)), None), self._dim
        )
        h1 = _mean_pool(self._fetch(_VIDEO_NEIGHBOURS_Q, vid=video_id), self._dim)
        return np.concatenate([h0, h1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, user_id: str, video_id: str) -> float:
        """
        Return the link-prediction score ∈ [0, 1] for a user–video pair.

        Score is σ(h_user · W · h_video) where h is the 128-dim representation.
        """
        h_u = self._represent_user(user_id)
        h_v = self._represent_video(video_id)
        raw = float(h_u @ self._W @ h_v)
        return _sigmoid(raw)

    def batch_score(self, user_id: str, video_ids: list[str]) -> dict[str, float]:
        """
        Score multiple videos for a user, reusing the user representation.

        More efficient than calling score() in a loop because the user's
        neighbourhood is fetched only once.

        Returns dict mapping video_id → score ∈ [0, 1].
        """
        h_u = self._represent_user(user_id)
        scores: dict[str, float] = {}
        for vid in video_ids:
            h_v = self._represent_video(vid)
            raw = float(h_u @ self._W @ h_v)
            scores[vid] = _sigmoid(raw)
        return scores

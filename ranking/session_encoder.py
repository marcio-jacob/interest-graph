"""
ranking/session_encoder.py
==========================
Session-aware feature extraction for recommendation reranking.

Reads the user's most recent sessions from Neo4j and builds:
  - short_term_topics : weighted topic interest from the last N sessions
  - long_term_topics  : normalised INTERESTED_IN_TOPIC scores from the graph
  - recent_video_ids  : for recency/diversity context
  - skipped_topics    : topics the user skipped recently (negative signal)
  - avg_completion    : session quality signal

These features are consumed by the FeatureReranker to personalise final scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from neo4j import GraphDatabase


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

_RECENT_SESSIONS_QUERY = """
MATCH (u:User {user_id: $uid})-[:HAS_SESSION]->(s:UserSession)
WITH s ORDER BY s.start_date DESC LIMIT $n_sessions
MATCH (s)-[vr:VIEWED]->(v:Video)
OPTIONAL MATCH (v)-[:IS_ABOUT]->(t:Topic)
RETURN v.video_id          AS video_id,
       vr.completion_rate  AS completion_rate,
       s.session_id        AS session_id,
       collect(DISTINCT t.slug) AS topics
"""

_USER_LONG_TERM_QUERY = """
MATCH (u:User {user_id: $uid})-[ti:INTERESTED_IN_TOPIC]->(t:Topic)
RETURN t.slug AS topic, ti.topic_score AS score
ORDER BY score DESC
"""


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class SessionFeatures:
    """Extracted features from a user's recent sessions."""
    user_id: str

    # Short-term: aggregated from last N sessions, normalised to [0, 1]
    short_term_topics: dict[str, float] = field(default_factory=dict)

    # Long-term: normalised INTERESTED_IN_TOPIC scores from the graph
    long_term_topics: dict[str, float] = field(default_factory=dict)

    # Raw recency signal
    recent_video_ids: list[str] = field(default_factory=list)
    recent_completion_rates: list[float] = field(default_factory=list)
    avg_completion: float = 0.0

    # Topics the user skipped (completion < 0.15) in recent sessions
    skipped_topics: set[str] = field(default_factory=set)

    def topic_affinity(self, topics: list[str]) -> float:
        """
        Score a candidate's topic list against this user's session context.
        Blends short-term (0.6) and long-term (0.4) signals.
        Applies a penalty for recently-skipped topics.
        """
        if not topics:
            return 0.0
        short_w = 0.6
        long_w = 0.4
        score = 0.0
        for t in topics:
            short = self.short_term_topics.get(t, 0.0)
            long_ = self.long_term_topics.get(t, 0.0)
            penalty = -0.3 if t in self.skipped_topics else 0.0
            score += short_w * short + long_w * long_ + penalty
        return score / len(topics)

    def topic_diversity_entropy(self) -> float:
        """Shannon entropy of the long-term topic distribution [0, 1 normalised]."""
        weights = list(self.long_term_topics.values())
        if not weights:
            return 0.0
        total = sum(weights)
        if total == 0:
            return 0.0
        probs = [w / total for w in weights if w > 0]
        raw_entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        return raw_entropy / max_entropy if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_session(
    user_id: str,
    uri: str,
    user: str,
    password: str,
    n_sessions: int = 3,
) -> SessionFeatures:
    """
    Build SessionFeatures for a user by reading their recent sessions from Neo4j.

    Parameters
    ----------
    user_id    : target user
    uri        : Neo4j URI
    user       : Neo4j username
    password   : Neo4j password
    n_sessions : how many recent sessions to use for short-term context
    """
    features = SessionFeatures(user_id=user_id)

    with GraphDatabase.driver(uri, auth=(user, password)) as drv:
        with drv.session() as sess:

            # --- Long-term interests ---
            lt_rows = sess.run(_USER_LONG_TERM_QUERY, uid=user_id)
            for r in lt_rows:
                features.long_term_topics[r["topic"]] = float(r["score"])

            # --- Recent sessions (short-term) ---
            recent_rows = sess.run(
                _RECENT_SESSIONS_QUERY, uid=user_id, n_sessions=n_sessions
            )
            topic_weights: dict[str, float] = {}
            for r in recent_rows:
                cr = float(r["completion_rate"] or 0.0)
                vid = r["video_id"]
                topics = r["topics"] or []
                features.recent_video_ids.append(vid)
                features.recent_completion_rates.append(cr)
                for t in topics:
                    if cr < 0.15:
                        features.skipped_topics.add(t)
                    else:
                        weight = 0.5 * cr if cr >= 0.8 else 0.1
                        topic_weights[t] = topic_weights.get(t, 0.0) + weight

            # Normalise short-term topics to [0, 1]
            if topic_weights:
                max_w = max(topic_weights.values())
                if max_w > 0:
                    features.short_term_topics = {
                        t: w / max_w for t, w in topic_weights.items()
                    }

            features.avg_completion = (
                float(np.mean(features.recent_completion_rates))
                if features.recent_completion_rates
                else 0.0
            )

    return features

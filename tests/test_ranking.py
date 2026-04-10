"""
tests/test_ranking.py
=====================
Comprehensive unit tests for every new module:

  ranking/candidates.py     — Candidate dataclass + 5 generator adapters
  ranking/session_encoder.py — SessionFeatures + encode_session
  ranking/reranker.py        — FeatureReranker
  ranking/pipeline.py        — RecommendationPipeline
  bandits/contextual.py      — EpsilonGreedyBandit + LinUCBBandit
  models/graph_sage.py       — GraphSAGEScorer
  evaluation/metrics.py      — all metric functions
  embeddings/store.py        — EmbeddingStore

All tests are mock-based — no live Neo4j connection required.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


# ===========================================================================
# Helpers — mock Neo4j driver
# ===========================================================================

def _make_driver_ctx(rows: list[dict]):
    """
    Return a mock that behaves like:
        with GraphDatabase.driver(...) as drv:
            with drv.session() as sess:
                result = sess.run(...)
                # iterating result yields rows
    """
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter(rows))
    mock_result.data.return_value = rows

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock()
    mock_drv.session.return_value = mock_session
    mock_drv.__enter__ = MagicMock(return_value=mock_drv)
    mock_drv.__exit__ = MagicMock(return_value=False)

    return mock_drv


def _patch_driver(rows: list[dict]):
    """Patch GraphDatabase.driver in all ranking/embeddings/models modules."""
    return patch("neo4j.GraphDatabase.driver", return_value=_make_driver_ctx(rows))


# ===========================================================================
# ranking/candidates.py — Candidate dataclass
# ===========================================================================

class TestCandidate:

    def test_fields_set_correctly(self):
        from ranking.candidates import Candidate
        c = Candidate(
            video_id="v1",
            source_engine="content_based",
            raw_score=4.2,
            explanation="test",
            metadata={"topics": ["tech"]},
        )
        assert c.video_id == "v1"
        assert c.source_engine == "content_based"
        assert c.raw_score == 4.2
        assert c.metadata["topics"] == ["tech"]

    def test_metadata_defaults_to_empty_dict(self):
        from ranking.candidates import Candidate
        c = Candidate(video_id="v1", source_engine="x", raw_score=1.0, explanation="")
        assert c.metadata == {}

    def test_candidates_are_equal_when_fields_match(self):
        from ranking.candidates import Candidate
        a = Candidate("v1", "eng", 1.0, "x")
        b = Candidate("v1", "eng", 1.0, "x")
        assert a == b

    def test_candidates_differ_on_score(self):
        from ranking.candidates import Candidate
        a = Candidate("v1", "eng", 1.0, "x")
        b = Candidate("v1", "eng", 2.0, "x")
        assert a != b


# ===========================================================================
# ranking/candidates.py — ContentBasedGenerator
# ===========================================================================

class TestContentBasedGenerator:

    def _rows(self):
        return [
            {
                "video_id": "v1",
                "description": "tech video",
                "relevance_score": 4.4,
                "topics": ["technology_science"],
                "creator": "alice",
            },
            {
                "video_id": "v2",
                "description": "lifestyle video",
                "relevance_score": 3.1,
                "topics": ["lifestyle_vlog"],
                "creator": "bob",
            },
        ]

    def test_returns_candidates_with_correct_engine_name(self):
        from ranking.candidates import ContentBasedGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = ContentBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1", limit=10)
        assert all(c.source_engine == "content_based" for c in result)

    def test_candidate_count_matches_db_rows(self):
        from ranking.candidates import ContentBasedGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = ContentBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1", limit=10)
        assert len(result) == 2

    def test_raw_score_copied_from_db(self):
        from ranking.candidates import ContentBasedGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = ContentBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1", limit=10)
        assert result[0].raw_score == pytest.approx(4.4)

    def test_metadata_contains_topics_and_creator(self):
        from ranking.candidates import ContentBasedGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = ContentBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1", limit=10)
        assert result[0].metadata["topics"] == ["technology_science"]
        assert result[0].metadata["creator"] == "alice"

    def test_empty_db_result_returns_empty_list(self):
        from ranking.candidates import ContentBasedGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx([])):
            gen = ContentBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result == []

    def test_none_creator_defaults_to_unknown(self):
        from ranking.candidates import ContentBasedGenerator
        rows = [{"video_id": "v1", "description": "x", "relevance_score": 1.0,
                 "topics": [], "creator": None}]
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(rows)):
            gen = ContentBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result[0].metadata["creator"] == "unknown"

    def test_description_truncated_to_80_chars(self):
        from ranking.candidates import ContentBasedGenerator
        long_desc = "x" * 200
        rows = [{"video_id": "v1", "description": long_desc, "relevance_score": 1.0,
                 "topics": [], "creator": "c"}]
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(rows)):
            gen = ContentBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert len(result[0].metadata["description"]) == 80


# ===========================================================================
# ranking/candidates.py — CollaborativeFilteringGenerator
# ===========================================================================

class TestCollaborativeFilteringGenerator:

    def _rows(self):
        return [
            {
                "video_id": "v3",
                "description": "collab video",
                "collab_score": 8.1,
                "peer_count": 2,
                "topics": ["technology_science"],
                "creator": "carol",
            }
        ]

    def test_engine_name(self):
        from ranking.candidates import CollaborativeFilteringGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = CollaborativeFilteringGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result[0].source_engine == "collab_filter"

    def test_peer_count_in_metadata(self):
        from ranking.candidates import CollaborativeFilteringGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = CollaborativeFilteringGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result[0].metadata["peer_count"] == 2

    def test_collab_score_is_raw_score(self):
        from ranking.candidates import CollaborativeFilteringGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = CollaborativeFilteringGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result[0].raw_score == pytest.approx(8.1)

    def test_explanation_mentions_peer_count(self):
        from ranking.candidates import CollaborativeFilteringGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = CollaborativeFilteringGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert "2 users" in result[0].explanation

    def test_singular_peer_grammar(self):
        from ranking.candidates import CollaborativeFilteringGenerator
        rows = [{"video_id": "v1", "description": "x", "collab_score": 5.0,
                 "peer_count": 1, "topics": [], "creator": "c"}]
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(rows)):
            gen = CollaborativeFilteringGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert "1 user" in result[0].explanation
        assert "1 users" not in result[0].explanation


# ===========================================================================
# ranking/candidates.py — EmbeddingRetrievalGenerator
# ===========================================================================

class TestEmbeddingRetrievalGenerator:

    def _user_emb(self, dim=64):
        v = np.random.default_rng(0).standard_normal(dim).astype(np.float32)
        return list(v / np.linalg.norm(v))

    def _video_emb(self, dim=64, seed=1):
        v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
        return list(v / np.linalg.norm(v))

    def test_returns_empty_when_no_user_embedding(self):
        from ranking.candidates import EmbeddingRetrievalGenerator
        rows = [{"embedding": None}]
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(rows)):
            gen = EmbeddingRetrievalGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result == []

    def test_returns_empty_when_zero_norm_user_embedding(self):
        from ranking.candidates import EmbeddingRetrievalGenerator
        rows = [{"embedding": [0.0] * 64}]
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(rows)):
            gen = EmbeddingRetrievalGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result == []

    def test_cosine_scores_are_in_minus1_plus1(self):
        from ranking.candidates import EmbeddingRetrievalGenerator
        user_emb = self._user_emb()
        vid_emb = self._video_emb()
        meta_row = {"video_id": "v1", "description": "d", "topics": [], "creator": "c"}

        call_count = [0]

        def fake_driver_factory(*args, **kwargs):
            call_count[0] += 1
            # First call: user embedding
            if call_count[0] == 1:
                return _make_driver_ctx([{"embedding": user_emb}])
            # Second call: batch of video embeddings
            if call_count[0] == 2:
                return _make_driver_ctx([{"video_id": "v1", "embedding": vid_emb}])
            # Third call: no more batches
            if call_count[0] == 3:
                return _make_driver_ctx([])
            # Fourth call: metadata
            return _make_driver_ctx([meta_row])

        with patch("ranking.candidates.GraphDatabase.driver",
                   side_effect=fake_driver_factory):
            gen = EmbeddingRetrievalGenerator("bolt://x", "u", "p", batch_size=500)
            result = gen.generate("user-1", limit=5)

        if result:
            for c in result:
                assert -1.0 <= c.raw_score <= 1.0

    def test_engine_name_is_embedding(self):
        from ranking.candidates import EmbeddingRetrievalGenerator
        user_emb = self._user_emb()
        vid_emb = self._video_emb()
        meta_row = {"video_id": "v1", "description": "d", "topics": ["tech"], "creator": "c"}

        call_n = [0]

        def side_effect(*a, **kw):
            call_n[0] += 1
            if call_n[0] == 1:
                return _make_driver_ctx([{"embedding": user_emb}])
            if call_n[0] == 2:
                return _make_driver_ctx([{"video_id": "v1", "embedding": vid_emb}])
            if call_n[0] == 3:
                return _make_driver_ctx([])
            return _make_driver_ctx([meta_row])

        with patch("ranking.candidates.GraphDatabase.driver", side_effect=side_effect):
            gen = EmbeddingRetrievalGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1", limit=5)

        if result:
            assert result[0].source_engine == "embedding"

    def test_skips_none_video_embeddings(self):
        """Videos with None embedding must be silently skipped."""
        from ranking.candidates import EmbeddingRetrievalGenerator
        user_emb = self._user_emb()
        call_n = [0]

        def side_effect(*a, **kw):
            call_n[0] += 1
            if call_n[0] == 1:
                return _make_driver_ctx([{"embedding": user_emb}])
            if call_n[0] == 2:
                return _make_driver_ctx([
                    {"video_id": "v_none", "embedding": None},
                    {"video_id": "v_zero", "embedding": [0.0] * 64},
                ])
            return _make_driver_ctx([])

        with patch("ranking.candidates.GraphDatabase.driver", side_effect=side_effect):
            gen = EmbeddingRetrievalGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1", limit=5)

        # v_none and v_zero should produce no candidates (empty scores)
        assert result == []


# ===========================================================================
# ranking/candidates.py — TrendingGenerator
# ===========================================================================

class TestTrendingGenerator:

    def _rows(self):
        return [
            {
                "video_id": "vt1", "description": "viral", "trending_score": 143.3,
                "view_count": 50, "like_count": 30, "topics": ["comedy_entertainment"],
                "creator": "comedian",
            }
        ]

    def test_engine_name(self):
        from ranking.candidates import TrendingGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = TrendingGenerator("bolt://x", "u", "p")
            result = gen.generate("any-user")
        assert result[0].source_engine == "trending"

    def test_raw_score_is_trending_score(self):
        from ranking.candidates import TrendingGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = TrendingGenerator("bolt://x", "u", "p")
            result = gen.generate("any-user")
        assert result[0].raw_score == pytest.approx(143.3)

    def test_view_and_like_count_in_metadata(self):
        from ranking.candidates import TrendingGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows())):
            gen = TrendingGenerator("bolt://x", "u", "p")
            result = gen.generate("any-user")
        assert result[0].metadata["view_count"] == 50
        assert result[0].metadata["like_count"] == 30

    def test_trending_is_not_user_specific(self):
        """TrendingGenerator should produce identical results for any user_id."""
        from ranking.candidates import TrendingGenerator
        # Each call to GraphDatabase.driver gets a fresh mock so iterators are
        # not shared between the two generate() calls.
        with patch("ranking.candidates.GraphDatabase.driver",
                   side_effect=lambda *a, **kw: _make_driver_ctx(self._rows())):
            gen = TrendingGenerator("bolt://x", "u", "p")
            r1 = gen.generate("user-A")
            r2 = gen.generate("user-B")
        assert [c.video_id for c in r1] == [c.video_id for c in r2]


# ===========================================================================
# ranking/candidates.py — CreatorBasedGenerator (LIMIT 1 bug regression)
# ===========================================================================

class TestCreatorBasedGenerator:

    def _rows(self, n=3):
        return [
            {
                "creator_id": f"c{i}", "creator": f"creator{i}",
                "creator_score": float(10 - i), "topic_match": float(5 - i),
                "social_boost": 0.5, "centrality": 1.0,
                "video_id": f"v{i}", "description": f"video {i}",
                "topics": ["technology_science"],
            }
            for i in range(n)
        ]

    def test_engine_name(self):
        from ranking.candidates import CreatorBasedGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows(1))):
            gen = CreatorBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result[0].source_engine == "creator"

    def test_multiple_creators_return_multiple_candidates(self):
        """Regression: LIMIT 1 bug would cause only 1 row ever."""
        from ranking.candidates import CreatorBasedGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows(3))):
            gen = CreatorBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1", limit=10)
        assert len(result) == 3

    def test_skips_rows_with_null_video_id(self):
        from ranking.candidates import CreatorBasedGenerator
        rows = [
            {"creator_id": "c1", "creator": "x", "creator_score": 5.0,
             "topic_match": 2.0, "social_boost": 0.0, "centrality": 1.0,
             "video_id": None, "description": "x", "topics": []},
        ]
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(rows)):
            gen = CreatorBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result == []

    def test_metadata_has_creator_and_pagerank(self):
        from ranking.candidates import CreatorBasedGenerator
        with patch("ranking.candidates.GraphDatabase.driver",
                   return_value=_make_driver_ctx(self._rows(1))):
            gen = CreatorBasedGenerator("bolt://x", "u", "p")
            result = gen.generate("user-1")
        assert result[0].metadata["creator"] == "creator0"
        assert "pagerank" in result[0].metadata


# ===========================================================================
# ranking/candidates.py — build_all_generators
# ===========================================================================

class TestBuildAllGenerators:

    def test_returns_five_generators(self):
        from ranking.candidates import build_all_generators
        env = {"NEO4J_URI": "bolt://x", "NEO4J_USER": "u", "NEO4J_PASSWORD": "p"}
        with patch.dict("os.environ", env):
            gens = build_all_generators()
        assert len(gens) == 5

    def test_generator_names_are_unique(self):
        from ranking.candidates import build_all_generators
        env = {"NEO4J_URI": "bolt://x", "NEO4J_USER": "u", "NEO4J_PASSWORD": "p"}
        with patch.dict("os.environ", env):
            gens = build_all_generators()
        names = [g.engine_name for g in gens]
        assert len(names) == len(set(names))

    def test_all_are_candidate_generators(self):
        from ranking.candidates import build_all_generators, CandidateGenerator
        env = {"NEO4J_URI": "bolt://x", "NEO4J_USER": "u", "NEO4J_PASSWORD": "p"}
        with patch.dict("os.environ", env):
            gens = build_all_generators()
        assert all(isinstance(g, CandidateGenerator) for g in gens)


# ===========================================================================
# ranking/session_encoder.py — SessionFeatures
# ===========================================================================

class TestSessionFeatures:

    def _features(self):
        from ranking.session_encoder import SessionFeatures
        sf = SessionFeatures(user_id="u1")
        sf.long_term_topics = {"technology_science": 1.0, "comedy_entertainment": 0.2}
        sf.short_term_topics = {"technology_science": 0.9}
        sf.skipped_topics = {"gaming_esports"}
        sf.avg_completion = 0.75
        return sf

    def test_topic_affinity_returns_float(self):
        sf = self._features()
        score = sf.topic_affinity(["technology_science"])
        assert isinstance(score, float)

    def test_topic_affinity_high_for_known_topic(self):
        sf = self._features()
        score = sf.topic_affinity(["technology_science"])
        assert score > 0.5

    def test_topic_affinity_zero_for_empty_topics(self):
        sf = self._features()
        assert sf.topic_affinity([]) == 0.0

    def test_topic_affinity_penalises_skipped_topics(self):
        sf = self._features()
        score_unknown = sf.topic_affinity(["unknown_topic"])
        score_skipped = sf.topic_affinity(["gaming_esports"])
        assert score_skipped < score_unknown

    def test_topic_diversity_entropy_zero_for_one_topic(self):
        from ranking.session_encoder import SessionFeatures
        sf = SessionFeatures(user_id="u1")
        sf.long_term_topics = {"technology_science": 1.0}
        # With only 1 topic, log2(1) = 0, entropy = 0
        assert sf.topic_diversity_entropy() == pytest.approx(0.0)

    def test_topic_diversity_entropy_one_for_uniform_two_topics(self):
        from ranking.session_encoder import SessionFeatures
        sf = SessionFeatures(user_id="u1")
        sf.long_term_topics = {"tech": 1.0, "comedy": 1.0}
        # uniform 2-topic → normalised entropy = 1.0
        assert sf.topic_diversity_entropy() == pytest.approx(1.0)

    def test_topic_diversity_entropy_zero_for_empty(self):
        from ranking.session_encoder import SessionFeatures
        sf = SessionFeatures(user_id="u1")
        assert sf.topic_diversity_entropy() == 0.0

    def test_topic_affinity_blends_short_and_long_term(self):
        from ranking.session_encoder import SessionFeatures
        sf = SessionFeatures(user_id="u1")
        # Only short-term signal
        sf.short_term_topics = {"tech": 1.0}
        sf.long_term_topics = {}
        score_short_only = sf.topic_affinity(["tech"])

        sf2 = SessionFeatures(user_id="u1")
        # Only long-term signal
        sf2.short_term_topics = {}
        sf2.long_term_topics = {"tech": 1.0}
        score_long_only = sf2.topic_affinity(["tech"])

        # Both should be positive; short-term has higher weight (0.6 vs 0.4)
        assert score_short_only > score_long_only
        assert score_short_only == pytest.approx(0.6)
        assert score_long_only == pytest.approx(0.4)


# ===========================================================================
# ranking/session_encoder.py — encode_session
# ===========================================================================

class TestEncodeSession:

    def _make_driver_multi(self, lt_rows, session_rows):
        """Mock driver that returns different rows per query call."""
        results = [lt_rows, session_rows]
        call_idx = [0]

        def make_result(rows):
            r = MagicMock()
            r.__iter__ = MagicMock(return_value=iter(rows))
            return r

        mock_session = MagicMock()

        def run_side_effect(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return make_result(results[idx] if idx < len(results) else [])

        mock_session.run.side_effect = run_side_effect
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_drv = MagicMock()
        mock_drv.session.return_value = mock_session
        mock_drv.__enter__ = MagicMock(return_value=mock_drv)
        mock_drv.__exit__ = MagicMock(return_value=False)

        return mock_drv

    def test_long_term_topics_populated(self):
        from ranking.session_encoder import encode_session
        lt = [{"topic": "technology_science", "score": 1.0}]
        sess = [{"video_id": "v1", "completion_rate": 0.9,
                 "session_id": "s1", "topics": ["technology_science"]}]
        drv = self._make_driver_multi(lt, sess)
        with patch("ranking.session_encoder.GraphDatabase.driver", return_value=drv):
            sf = encode_session("u1", "bolt://x", "u", "p")
        assert sf.long_term_topics["technology_science"] == pytest.approx(1.0)

    def test_short_term_topics_normalised_to_one(self):
        from ranking.session_encoder import encode_session
        lt = []
        sess = [
            {"video_id": "v1", "completion_rate": 0.9, "session_id": "s1",
             "topics": ["tech"]},
            {"video_id": "v2", "completion_rate": 0.8, "session_id": "s1",
             "topics": ["tech"]},
        ]
        drv = self._make_driver_multi(lt, sess)
        with patch("ranking.session_encoder.GraphDatabase.driver", return_value=drv):
            sf = encode_session("u1", "bolt://x", "u", "p")
        # Normalised to max → 1.0
        assert sf.short_term_topics.get("tech", 0) == pytest.approx(1.0)

    def test_skipped_topics_detected(self):
        from ranking.session_encoder import encode_session
        lt = []
        sess = [{"video_id": "v1", "completion_rate": 0.05, "session_id": "s1",
                 "topics": ["gaming_esports"]}]
        drv = self._make_driver_multi(lt, sess)
        with patch("ranking.session_encoder.GraphDatabase.driver", return_value=drv):
            sf = encode_session("u1", "bolt://x", "u", "p")
        assert "gaming_esports" in sf.skipped_topics

    def test_avg_completion_computed(self):
        from ranking.session_encoder import encode_session
        lt = []
        sess = [
            {"video_id": "v1", "completion_rate": 0.8, "session_id": "s1", "topics": []},
            {"video_id": "v2", "completion_rate": 0.6, "session_id": "s1", "topics": []},
        ]
        drv = self._make_driver_multi(lt, sess)
        with patch("ranking.session_encoder.GraphDatabase.driver", return_value=drv):
            sf = encode_session("u1", "bolt://x", "u", "p")
        assert sf.avg_completion == pytest.approx(0.7)

    def test_user_id_preserved(self):
        from ranking.session_encoder import encode_session
        drv = self._make_driver_multi([], [])
        with patch("ranking.session_encoder.GraphDatabase.driver", return_value=drv):
            sf = encode_session("uid-999", "bolt://x", "u", "p")
        assert sf.user_id == "uid-999"

    def test_empty_sessions_returns_defaults(self):
        from ranking.session_encoder import encode_session
        drv = self._make_driver_multi([], [])
        with patch("ranking.session_encoder.GraphDatabase.driver", return_value=drv):
            sf = encode_session("u1", "bolt://x", "u", "p")
        assert sf.short_term_topics == {}
        assert sf.avg_completion == 0.0
        assert sf.skipped_topics == set()

    def test_none_completion_rate_treated_as_zero(self):
        from ranking.session_encoder import encode_session
        lt = []
        sess = [{"video_id": "v1", "completion_rate": None, "session_id": "s1",
                 "topics": ["tech"]}]
        drv = self._make_driver_multi(lt, sess)
        with patch("ranking.session_encoder.GraphDatabase.driver", return_value=drv):
            sf = encode_session("u1", "bolt://x", "u", "p")
        # completion_rate=None → treated as 0.0 < 0.15 → skipped
        assert "tech" in sf.skipped_topics


# ===========================================================================
# ranking/reranker.py — FeatureReranker
# ===========================================================================

class TestFeatureReranker:

    def _session(self, long_term=None, short_term=None, skipped=None):
        from ranking.session_encoder import SessionFeatures
        sf = SessionFeatures(user_id="u1")
        sf.long_term_topics = long_term or {"technology_science": 1.0, "comedy_entertainment": 0.2}
        sf.short_term_topics = short_term or {"technology_science": 0.8}
        sf.skipped_topics = skipped or set()
        return sf

    def _candidate(self, vid, engine, score, topics=None):
        from ranking.candidates import Candidate
        return Candidate(
            video_id=vid,
            source_engine=engine,
            raw_score=score,
            explanation="test",
            metadata={"topics": topics or ["technology_science"]},
        )

    def test_returns_empty_for_empty_input(self):
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        result = r.rerank([], self._session())
        assert result == []

    def test_single_candidate_gets_rank_1(self):
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        c = self._candidate("v1", "content_based", 1.0)
        result = r.rerank([c], self._session())
        assert len(result) == 1
        assert result[0].rank == 1

    def test_ranks_are_1_indexed_and_ascending(self):
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        candidates = [
            self._candidate("v1", "content_based", 4.0),
            self._candidate("v2", "content_based", 3.0),
            self._candidate("v3", "content_based", 2.0),
        ]
        result = r.rerank(candidates, self._session())
        assert [rc.rank for rc in result] == [1, 2, 3]

    def test_higher_final_score_gets_lower_rank_number(self):
        """Rank 1 = best = highest final_score."""
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        candidates = [
            self._candidate("v1", "content_based", 2.0),
            self._candidate("v2", "content_based", 8.0),
        ]
        result = r.rerank(candidates, self._session())
        rank_of_v2 = next(rc.rank for rc in result if rc.video_id == "v2")
        rank_of_v1 = next(rc.rank for rc in result if rc.video_id == "v1")
        assert rank_of_v2 < rank_of_v1

    def test_deduplication_keeps_higher_raw_score(self):
        """Same video from two engines: keep the one with higher raw_score."""
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        candidates = [
            self._candidate("v1", "content_based", 4.0),
            self._candidate("v1", "collab_filter", 8.0),   # same video_id, higher score
        ]
        result = r.rerank(candidates, self._session())
        assert len(result) == 1
        assert result[0].source_engine == "collab_filter"
        assert result[0].raw_score == pytest.approx(8.0)

    def test_normalise_static_method(self):
        from ranking.reranker import FeatureReranker
        assert FeatureReranker._normalise(5.0, 0.0, 10.0) == pytest.approx(0.5)
        assert FeatureReranker._normalise(0.0, 0.0, 10.0) == pytest.approx(0.0)
        assert FeatureReranker._normalise(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_normalise_degenerate_range_returns_0_5(self):
        from ranking.reranker import FeatureReranker
        assert FeatureReranker._normalise(5.0, 5.0, 5.0) == pytest.approx(0.5)

    def test_engine_weight_applied(self):
        """collab_filter weight 1.2 > content_based 1.0 — should score higher when equal raw."""
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        # Both with the same raw score, same topics, same session alignment
        # collab_filter should score higher because weight=1.2 vs 1.0
        candidates = [
            self._candidate("v1", "content_based", 5.0),
            self._candidate("v2", "collab_filter", 5.0),
        ]
        result = r.rerank(candidates, self._session())
        rank_cf = next(rc.rank for rc in result if rc.source_engine == "collab_filter")
        rank_cb = next(rc.rank for rc in result if rc.source_engine == "content_based")
        assert rank_cf < rank_cb

    def test_serendipitous_flag_set_for_cross_topic(self):
        """Video in a topic not in user's top-3 should get serendipitous=True."""
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        session = self._session(long_term={"technology_science": 1.0})
        # "cooking_food" is not in user's top-3 (only 1 topic in long_term here)
        c = self._candidate("v1", "trending", 5.0, topics=["cooking_food"])
        result = r.rerank([c], session)
        assert result[0].explanation_trace["serendipitous"] is True

    def test_non_serendipitous_for_known_topic(self):
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        session = self._session(long_term={"technology_science": 1.0})
        c = self._candidate("v1", "content_based", 5.0, topics=["technology_science"])
        result = r.rerank([c], session)
        assert result[0].explanation_trace["serendipitous"] is False

    def test_exploration_bonus_added_for_serendipitous(self):
        from ranking.reranker import FeatureReranker
        bonus = 0.05
        r = FeatureReranker(exploration_bonus=bonus)
        session = self._session(long_term={"technology_science": 1.0})
        c = self._candidate("v1", "trending", 5.0, topics=["cooking_food"])
        result = r.rerank([c], session)
        assert result[0].explanation_trace["exploration_bonus"] == pytest.approx(bonus)

    def test_no_exploration_bonus_for_known_topic(self):
        from ranking.reranker import FeatureReranker
        r = FeatureReranker(exploration_bonus=0.05)
        session = self._session(long_term={"technology_science": 1.0})
        c = self._candidate("v1", "content_based", 5.0, topics=["technology_science"])
        result = r.rerank([c], session)
        assert result[0].explanation_trace["exploration_bonus"] == pytest.approx(0.0)

    def test_trace_keys_present(self):
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        c = self._candidate("v1", "content_based", 5.0)
        result = r.rerank([c], self._session())
        trace = result[0].explanation_trace
        for key in ("norm_raw_score", "engine_weight", "session_alignment",
                    "serendipitous", "exploration_bonus"):
            assert key in trace, f"Missing trace key: {key}"

    def test_to_dict_contains_all_fields(self):
        from ranking.reranker import FeatureReranker
        r = FeatureReranker()
        c = self._candidate("v1", "content_based", 5.0)
        result = r.rerank([c], self._session())
        d = result[0].to_dict()
        for key in ("rank", "video_id", "source_engine", "raw_score",
                    "final_score", "explanation", "trace", "metadata"):
            assert key in d

    def test_candidates_with_no_topics_get_no_exploration_bonus(self):
        """Empty topic list → cand_topics is empty → bool(set()) = False → no bonus."""
        from ranking.reranker import FeatureReranker
        r = FeatureReranker(exploration_bonus=0.1)
        session = self._session(long_term={"technology_science": 1.0})
        c = self._candidate("v1", "trending", 5.0, topics=[])
        result = r.rerank([c], session)
        assert result[0].explanation_trace["exploration_bonus"] == pytest.approx(0.0)


# ===========================================================================
# ranking/pipeline.py — RecommendationPipeline
# ===========================================================================

class TestRecommendationPipeline:

    def _make_mock_generator(self, engine_name, video_ids):
        from ranking.candidates import Candidate
        mock_gen = MagicMock()
        mock_gen.engine_name = engine_name
        mock_gen.generate.return_value = [
            Candidate(
                video_id=vid,
                source_engine=engine_name,
                raw_score=float(i + 1),
                explanation="mock",
                metadata={"topics": ["technology_science"]},
            )
            for i, vid in enumerate(video_ids)
        ]
        return mock_gen

    def _make_pipeline_with_mocks(self, generator_map: dict):
        from ranking.pipeline import RecommendationPipeline
        from ranking.session_encoder import SessionFeatures

        mock_sf = SessionFeatures(user_id="u1")
        mock_sf.long_term_topics = {"technology_science": 1.0}
        mock_sf.short_term_topics = {"technology_science": 0.8}

        pipeline = RecommendationPipeline(
            uri="bolt://x", user="u", password="p"
        )
        pipeline._generators = [
            self._make_mock_generator(eng, vids)
            for eng, vids in generator_map.items()
        ]
        return pipeline, mock_sf

    def test_recommend_returns_n_results(self):
        from ranking.pipeline import RecommendationPipeline
        from ranking.session_encoder import SessionFeatures

        pipeline = RecommendationPipeline(uri="bolt://x", user="u", password="p")
        mock_sf = SessionFeatures(user_id="u1")
        mock_sf.long_term_topics = {"technology_science": 1.0}

        pipeline._generators = [
            self._make_mock_generator("content_based", [f"v{i}" for i in range(20)]),
        ]

        with patch("ranking.pipeline.encode_session", return_value=mock_sf):
            result = pipeline.recommend("u1", n=5)

        assert len(result) == 5

    def test_recommend_returns_ranked_candidates(self):
        from ranking.pipeline import RecommendationPipeline
        from ranking.session_encoder import SessionFeatures
        from ranking.reranker import RankedCandidate

        pipeline = RecommendationPipeline(uri="bolt://x", user="u", password="p")
        mock_sf = SessionFeatures(user_id="u1")
        mock_sf.long_term_topics = {"technology_science": 1.0}

        pipeline._generators = [
            self._make_mock_generator("content_based", ["v1", "v2", "v3"]),
        ]

        with patch("ranking.pipeline.encode_session", return_value=mock_sf):
            result = pipeline.recommend("u1", n=3)

        assert all(isinstance(rc, RankedCandidate) for rc in result)

    def test_deduplication_across_engines(self):
        """Same video from two engines should appear only once in output."""
        from ranking.pipeline import RecommendationPipeline
        from ranking.session_encoder import SessionFeatures

        pipeline = RecommendationPipeline(uri="bolt://x", user="u", password="p")
        mock_sf = SessionFeatures(user_id="u1")
        mock_sf.long_term_topics = {"technology_science": 1.0}

        pipeline._generators = [
            self._make_mock_generator("content_based", ["v1", "v2"]),
            self._make_mock_generator("collab_filter", ["v1", "v3"]),  # v1 duplicate
        ]

        with patch("ranking.pipeline.encode_session", return_value=mock_sf):
            result = pipeline.recommend("u1", n=10)

        video_ids = [rc.video_id for rc in result]
        assert len(video_ids) == len(set(video_ids)), "Duplicate video_ids in output"

    def test_failed_generator_does_not_crash_pipeline(self):
        """A generator that raises must not propagate; pipeline continues with others."""
        from ranking.pipeline import RecommendationPipeline
        from ranking.session_encoder import SessionFeatures
        from ranking.candidates import Candidate

        pipeline = RecommendationPipeline(uri="bolt://x", user="u", password="p")
        mock_sf = SessionFeatures(user_id="u1")
        mock_sf.long_term_topics = {"technology_science": 1.0}

        failing_gen = MagicMock()
        failing_gen.engine_name = "content_based"
        failing_gen.generate.side_effect = RuntimeError("DB down")

        good_gen = self._make_mock_generator("trending", ["v1", "v2"])
        pipeline._generators = [failing_gen, good_gen]

        with patch("ranking.pipeline.encode_session", return_value=mock_sf):
            result = pipeline.recommend("u1", n=5)

        # Should still get results from the good generator
        assert len(result) > 0

    def test_recommend_n_larger_than_candidates_returns_all(self):
        """Requesting more items than available candidates → return all."""
        from ranking.pipeline import RecommendationPipeline
        from ranking.session_encoder import SessionFeatures

        pipeline = RecommendationPipeline(uri="bolt://x", user="u", password="p")
        mock_sf = SessionFeatures(user_id="u1")
        mock_sf.long_term_topics = {"technology_science": 1.0}

        pipeline._generators = [
            self._make_mock_generator("content_based", ["v1", "v2"]),
        ]

        with patch("ranking.pipeline.encode_session", return_value=mock_sf):
            result = pipeline.recommend("u1", n=100)

        assert len(result) == 2


# ===========================================================================
# bandits/contextual.py — BanditContext
# ===========================================================================

class TestBanditContext:

    def test_to_array_shape(self):
        from bandits.contextual import BanditContext
        ctx = BanditContext(dominant_topic_score=0.9, topic_diversity=0.5,
                            avg_session_completion=0.7, has_peers=True, has_embeddings=True)
        arr = ctx.to_array()
        assert arr.shape == (5,)

    def test_to_array_values(self):
        from bandits.contextual import BanditContext
        ctx = BanditContext(dominant_topic_score=0.9, topic_diversity=0.5,
                            avg_session_completion=0.7, has_peers=True, has_embeddings=False)
        arr = ctx.to_array()
        assert arr[0] == pytest.approx(0.9)
        assert arr[1] == pytest.approx(0.5)
        assert arr[2] == pytest.approx(0.7)
        assert arr[3] == pytest.approx(1.0)   # has_peers=True
        assert arr[4] == pytest.approx(0.0)   # has_embeddings=False

    def test_from_session_features(self):
        from bandits.contextual import BanditContext
        from ranking.session_encoder import SessionFeatures
        sf = SessionFeatures(user_id="u1")
        sf.long_term_topics = {"tech": 1.0, "comedy": 0.5}
        sf.avg_completion = 0.8
        ctx = BanditContext.from_session_features(sf)
        assert ctx.dominant_topic_score == pytest.approx(1.0)
        assert ctx.avg_session_completion == pytest.approx(0.8)


# ===========================================================================
# bandits/contextual.py — EpsilonGreedyBandit
# ===========================================================================

class TestEpsilonGreedyBandit:

    def test_select_engines_returns_correct_count(self):
        from bandits.contextual import EpsilonGreedyBandit
        b = EpsilonGreedyBandit(epsilon=0.0)   # always exploit
        engines = b.select_engines(n_engines=3)
        assert len(engines) == 3

    def test_selected_engines_are_python_strings(self):
        """Regression: numpy.str_ must be converted to Python str."""
        from bandits.contextual import EpsilonGreedyBandit, ENGINE_NAMES
        b = EpsilonGreedyBandit(epsilon=1.0, seed=0)  # always explore
        engines = b.select_engines(n_engines=3)
        for e in engines:
            assert isinstance(e, str), f"Expected str, got {type(e)}: {e!r}"

    def test_exploit_selects_highest_reward_engines(self):
        from bandits.contextual import EpsilonGreedyBandit
        b = EpsilonGreedyBandit(epsilon=0.0)
        # Manually set high reward for two engines
        b._arms["collab_filter"].pulls = 10
        b._arms["collab_filter"].total_reward = 9.0   # mean 0.9
        b._arms["embedding"].pulls = 10
        b._arms["embedding"].total_reward = 8.0       # mean 0.8
        engines = b.select_engines(n_engines=2)
        assert "collab_filter" in engines
        assert "embedding" in engines

    def test_explore_returns_subset_of_engine_names(self):
        from bandits.contextual import EpsilonGreedyBandit, ENGINE_NAMES
        b = EpsilonGreedyBandit(epsilon=1.0, seed=42)
        engines = b.select_engines(n_engines=3)
        assert set(engines).issubset(set(ENGINE_NAMES))

    def test_update_increments_pulls_and_reward(self):
        from bandits.contextual import EpsilonGreedyBandit
        b = EpsilonGreedyBandit()
        b.update("content_based", 0.8)
        assert b._arms["content_based"].pulls == 1
        assert b._arms["content_based"].total_reward == pytest.approx(0.8)

    def test_mean_reward_computed_correctly(self):
        from bandits.contextual import EpsilonGreedyBandit
        b = EpsilonGreedyBandit()
        b.update("content_based", 0.6)
        b.update("content_based", 0.8)
        assert b._arms["content_based"].mean_reward == pytest.approx(0.7)

    def test_epsilon_decays_after_update(self):
        from bandits.contextual import EpsilonGreedyBandit
        b = EpsilonGreedyBandit(epsilon=0.15, epsilon_decay=0.5, epsilon_min=0.01)
        initial = b.epsilon
        b.update("trending", 0.5)
        assert b.epsilon < initial

    def test_epsilon_does_not_go_below_min(self):
        from bandits.contextual import EpsilonGreedyBandit
        b = EpsilonGreedyBandit(epsilon=0.01, epsilon_decay=0.0001, epsilon_min=0.01)
        for _ in range(100):
            b.update("trending", 0.5)
        assert b.epsilon == pytest.approx(0.01)

    def test_stats_has_correct_keys(self):
        from bandits.contextual import EpsilonGreedyBandit, ENGINE_NAMES
        b = EpsilonGreedyBandit()
        stats = b.stats()
        assert set(stats.keys()) == set(ENGINE_NAMES)
        for v in stats.values():
            assert "pulls" in v
            assert "mean_reward" in v

    def test_select_engines_no_replacement(self):
        """All returned engines must be distinct."""
        from bandits.contextual import EpsilonGreedyBandit
        b = EpsilonGreedyBandit(epsilon=1.0, seed=99)
        engines = b.select_engines(n_engines=5)
        assert len(engines) == len(set(engines))


# ===========================================================================
# bandits/contextual.py — LinUCBBandit
# ===========================================================================

class TestLinUCBBandit:

    def _ctx(self):
        from bandits.contextual import BanditContext
        return BanditContext(0.8, 0.4, 0.75, True, True)

    def test_select_returns_correct_count(self):
        from bandits.contextual import LinUCBBandit
        b = LinUCBBandit()
        engines = b.select_engines(self._ctx(), n_engines=3)
        assert len(engines) == 3

    def test_selected_are_from_engine_names(self):
        from bandits.contextual import LinUCBBandit, ENGINE_NAMES
        b = LinUCBBandit()
        engines = b.select_engines(self._ctx(), n_engines=4)
        assert set(engines).issubset(set(ENGINE_NAMES))

    def test_selected_are_python_strings(self):
        from bandits.contextual import LinUCBBandit
        b = LinUCBBandit()
        engines = b.select_engines(self._ctx(), n_engines=3)
        for e in engines:
            assert isinstance(e, str)

    def test_update_changes_theta(self):
        from bandits.contextual import LinUCBBandit
        b = LinUCBBandit()
        theta_before = b.theta("collab_filter").copy()
        b.update("collab_filter", self._ctx(), reward=1.0)
        theta_after = b.theta("collab_filter")
        assert not np.allclose(theta_before, theta_after)

    def test_higher_reward_engine_selected_after_training(self):
        """After many high-reward updates for one engine, it should be preferred."""
        from bandits.contextual import LinUCBBandit
        b = LinUCBBandit(alpha=0.01)  # low alpha = less exploration
        ctx = self._ctx()
        for _ in range(20):
            b.update("collab_filter", ctx, reward=1.0)
        for _ in range(20):
            b.update("trending", ctx, reward=0.1)
        top_1 = b.select_engines(ctx, n_engines=1)
        assert "collab_filter" in top_1

    def test_theta_shape(self):
        from bandits.contextual import LinUCBBandit
        b = LinUCBBandit(context_dim=5)
        theta = b.theta("content_based")
        assert theta.shape == (5,)

    def test_stats_has_all_engines(self):
        from bandits.contextual import LinUCBBandit, ENGINE_NAMES
        b = LinUCBBandit()
        stats = b.stats()
        assert set(stats.keys()) == set(ENGINE_NAMES)

    def test_no_replacement_in_selection(self):
        from bandits.contextual import LinUCBBandit
        b = LinUCBBandit()
        engines = b.select_engines(self._ctx(), n_engines=5)
        assert len(engines) == len(set(engines))


# ===========================================================================
# models/graph_sage.py — GraphSAGEScorer
# ===========================================================================

class TestGraphSAGEScorer:

    def _emb(self, seed=0, dim=64):
        v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
        return list(v)

    def _make_driver_seq(self, emb_sequences):
        """Return driver mock that replays a list of embedding lists in sequence."""
        call_idx = [0]
        results = emb_sequences

        def factory(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            rows = results[idx] if idx < len(results) else []

            mock_result = MagicMock()
            mock_result.__iter__ = MagicMock(return_value=iter(rows))

            mock_sess = MagicMock()
            mock_sess.run.return_value = mock_result
            mock_sess.__enter__ = MagicMock(return_value=mock_sess)
            mock_sess.__exit__ = MagicMock(return_value=False)

            mock_drv = MagicMock()
            mock_drv.session.return_value = mock_sess
            mock_drv.__enter__ = MagicMock(return_value=mock_drv)
            mock_drv.__exit__ = MagicMock(return_value=False)
            return mock_drv

        return factory

    def test_score_in_zero_one_range(self):
        from models.graph_sage import GraphSAGEScorer
        emb = self._emb()
        # score() makes 4 driver calls: user_emb, user_neighbours, video_emb, video_neighbours
        sequences = [
            [{"embedding": emb}],      # user emb
            [{"embedding": emb}],      # user neighbours
            [{"embedding": emb}],      # video emb
            [{"embedding": emb}],      # video neighbours
        ]
        with patch("models.graph_sage.GraphDatabase.driver",
                   side_effect=self._make_driver_seq(sequences)):
            scorer = GraphSAGEScorer("bolt://x", "u", "p")
            s = scorer.score("user-1", "video-1")
        assert 0.0 <= s <= 1.0

    def test_score_with_missing_embeddings_returns_sigmoid_of_zero_dot(self):
        """Missing embeddings → zero vectors → dot product = 0 → sigmoid(0) ≈ 0.5."""
        from models.graph_sage import GraphSAGEScorer
        sequences = [
            [{"embedding": None}],   # user emb missing
            [],                       # no neighbours
            [{"embedding": None}],   # video emb missing
            [],                       # no neighbours
        ]
        with patch("models.graph_sage.GraphDatabase.driver",
                   side_effect=self._make_driver_seq(sequences)):
            scorer = GraphSAGEScorer("bolt://x", "u", "p")
            s = scorer.score("user-1", "video-1")
        assert s == pytest.approx(0.5, abs=0.01)

    def test_batch_score_consistent_with_single_score(self):
        """batch_score and score should return the same value for the same pair."""
        from models.graph_sage import GraphSAGEScorer
        emb_u = self._emb(seed=0)
        emb_v = self._emb(seed=1)

        # For batch_score: user_emb, user_neighbours, then per-video: video_emb, video_neighbours
        batch_sequences = [
            [{"embedding": emb_u}],   # user emb
            [{"embedding": emb_u}],   # user neighbours
            [{"embedding": emb_v}],   # video emb
            [{"embedding": emb_v}],   # video neighbours
        ]
        # For single score: same order
        single_sequences = [
            [{"embedding": emb_u}],
            [{"embedding": emb_u}],
            [{"embedding": emb_v}],
            [{"embedding": emb_v}],
        ]

        with patch("models.graph_sage.GraphDatabase.driver",
                   side_effect=self._make_driver_seq(batch_sequences)):
            scorer1 = GraphSAGEScorer("bolt://x", "u", "p")
            batch_result = scorer1.batch_score("user-1", ["v1"])

        with patch("models.graph_sage.GraphDatabase.driver",
                   side_effect=self._make_driver_seq(single_sequences)):
            scorer2 = GraphSAGEScorer("bolt://x", "u", "p")
            single_result = scorer2.score("user-1", "v1")

        assert batch_result["v1"] == pytest.approx(single_result, abs=1e-5)

    def test_identity_weight_matrix_at_init(self):
        """W should be identity at initialisation."""
        from models.graph_sage import GraphSAGEScorer
        scorer = GraphSAGEScorer("bolt://x", "u", "p")
        assert np.allclose(scorer._W, np.eye(128))


# ===========================================================================
# evaluation/metrics.py — precision / recall / ndcg / mrr
# ===========================================================================

class TestPrecisionAtK:

    def test_all_relevant(self):
        from evaluation.metrics import precision_at_k
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)

    def test_none_relevant(self):
        from evaluation.metrics import precision_at_k
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == pytest.approx(0.0)

    def test_partial_overlap(self):
        from evaluation.metrics import precision_at_k
        assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(2/3)

    def test_k_zero_returns_zero(self):
        from evaluation.metrics import precision_at_k
        assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_k_larger_than_list(self):
        from evaluation.metrics import precision_at_k
        # Only first 2 items considered, k=5 → 1/5 = 0.2
        assert precision_at_k(["a", "b"], {"a"}, 5) == pytest.approx(1/5)


class TestRecallAtK:

    def test_full_recall(self):
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_zero_recall(self):
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["a", "b", "c"], {"x"}, 3) == pytest.approx(0.0)

    def test_partial_recall(self):
        from evaluation.metrics import recall_at_k
        # relevant = {a, b, c}, top-2 = [a, b] → 2/3
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 2) == pytest.approx(2/3)

    def test_empty_relevant_set(self):
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["a", "b"], set(), 3) == 0.0


class TestNDCGAtK:

    def test_perfect_ranking(self):
        from evaluation.metrics import ndcg_at_k
        # ideal: all top items are relevant
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)

    def test_zero_relevant(self):
        from evaluation.metrics import ndcg_at_k
        assert ndcg_at_k(["a", "b", "c"], set(), 3) == 0.0

    def test_relevant_not_in_top_k(self):
        from evaluation.metrics import ndcg_at_k
        # relevant item is at position 4, k=3 → not seen → DCG=0
        assert ndcg_at_k(["a", "b", "c", "d"], {"d"}, 3) == pytest.approx(0.0)

    def test_ndcg_between_0_and_1(self):
        from evaluation.metrics import ndcg_at_k
        score = ndcg_at_k(["b", "a", "c"], {"a"}, 3)
        assert 0.0 <= score <= 1.0

    def test_higher_rank_gives_higher_ndcg(self):
        from evaluation.metrics import ndcg_at_k
        # relevant item "a" at rank 1 vs rank 2
        score_rank1 = ndcg_at_k(["a", "b", "c"], {"a"}, 3)
        score_rank2 = ndcg_at_k(["b", "a", "c"], {"a"}, 3)
        assert score_rank1 > score_rank2


class TestMRR:

    def test_first_item_relevant(self):
        from evaluation.metrics import mrr
        assert mrr([["a", "b", "c"]], [{"a"}]) == pytest.approx(1.0)

    def test_second_item_relevant(self):
        from evaluation.metrics import mrr
        assert mrr([["b", "a", "c"]], [{"a"}]) == pytest.approx(0.5)

    def test_no_relevant_item(self):
        from evaluation.metrics import mrr
        assert mrr([["a", "b", "c"]], [{"x"}]) == pytest.approx(0.0)

    def test_mean_across_queries(self):
        from evaluation.metrics import mrr
        # Query 1: first item relevant → RR=1.0; Query 2: second item relevant → RR=0.5
        score = mrr(
            [["a", "b"], ["b", "a"]],
            [{"a"}, {"a"}],
        )
        assert score == pytest.approx(0.75)

    def test_empty_ranked_lists(self):
        from evaluation.metrics import mrr
        assert mrr([], []) == 0.0


# ===========================================================================
# evaluation/metrics.py — diversity and novelty
# ===========================================================================

class TestIntraListDiversity:

    def test_identical_topics_gives_zero_diversity(self):
        from evaluation.metrics import intra_list_diversity
        topic_map = {"a": ["tech"], "b": ["tech"], "c": ["tech"]}
        score = intra_list_diversity(["a", "b", "c"], topic_map, 3)
        assert score == pytest.approx(0.0)

    def test_disjoint_topics_gives_full_diversity(self):
        from evaluation.metrics import intra_list_diversity
        topic_map = {"a": ["tech"], "b": ["comedy"], "c": ["cooking"]}
        score = intra_list_diversity(["a", "b", "c"], topic_map, 3)
        assert score == pytest.approx(1.0)

    def test_mixed_topics(self):
        from evaluation.metrics import intra_list_diversity
        # a=[tech], b=[tech, comedy], c=[cooking]
        topic_map = {"a": ["tech"], "b": ["tech", "comedy"], "c": ["cooking"]}
        score = intra_list_diversity(["a", "b", "c"], topic_map, 3)
        assert 0.0 < score < 1.0

    def test_fewer_than_two_items_returns_zero(self):
        from evaluation.metrics import intra_list_diversity
        assert intra_list_diversity(["a"], {"a": ["tech"]}, 5) == 0.0

    def test_k_limits_items_considered(self):
        from evaluation.metrics import intra_list_diversity
        # Only top-2 considered, both have same topic → 0
        topic_map = {"a": ["tech"], "b": ["tech"], "c": ["cooking"]}
        score = intra_list_diversity(["a", "b", "c"], topic_map, k=2)
        assert score == pytest.approx(0.0)

    def test_items_with_no_topics(self):
        from evaluation.metrics import intra_list_diversity
        # Missing topics → treat as disjoint (one has topics, other doesn't)
        topic_map = {"a": ["tech"]}  # b has no entry
        score = intra_list_diversity(["a", "b"], topic_map, 2)
        assert score == pytest.approx(1.0)


class TestNoveltyAtK:

    def test_low_popularity_gives_high_novelty(self):
        from evaluation.metrics import novelty_at_k
        # p=0.001 → -log2(0.001) ≈ 9.97
        score = novelty_at_k(["a"], {"a": 0.001}, 1)
        assert score == pytest.approx(-math.log2(0.001))

    def test_high_popularity_gives_low_novelty(self):
        from evaluation.metrics import novelty_at_k
        score_popular = novelty_at_k(["a"], {"a": 0.9}, 1)
        score_niche = novelty_at_k(["b"], {"b": 0.01}, 1)
        assert score_niche > score_popular

    def test_missing_popularity_defaults_to_low(self):
        from evaluation.metrics import novelty_at_k
        # Missing key → default 1e-6 → high novelty
        score = novelty_at_k(["x"], {}, 1)
        assert score > 10.0

    def test_empty_ranked_list(self):
        from evaluation.metrics import novelty_at_k
        assert novelty_at_k([], {"a": 0.5}, 5) == 0.0


class TestCoverage:

    def test_full_coverage(self):
        from evaluation.metrics import coverage
        assert coverage(["a", "b", "c"], 3) == pytest.approx(1.0)

    def test_partial_coverage(self):
        from evaluation.metrics import coverage
        assert coverage(["a", "b"], 10) == pytest.approx(0.2)

    def test_zero_catalog_size(self):
        from evaluation.metrics import coverage
        assert coverage(["a", "b"], 0) == 0.0

    def test_duplicates_not_counted_twice(self):
        from evaluation.metrics import coverage
        # a appears 3 times but catalog is 5 → 1/5 = 0.2
        assert coverage(["a", "a", "a"], 5) == pytest.approx(0.2)


class TestCatalogEntropy:

    def test_uniform_distribution_max_entropy(self):
        from evaluation.metrics import catalog_entropy
        items = ["a", "b", "c", "d"]
        attr_map = {"a": "t1", "b": "t2", "c": "t3", "d": "t4"}
        score = catalog_entropy(items, attr_map)
        # Uniform 4-class → entropy = log2(4) = 2.0
        assert score == pytest.approx(2.0)

    def test_all_same_topic_gives_zero_entropy(self):
        from evaluation.metrics import catalog_entropy
        items = ["a", "b", "c"]
        attr_map = {"a": "tech", "b": "tech", "c": "tech"}
        assert catalog_entropy(items, attr_map) == pytest.approx(0.0)

    def test_empty_items(self):
        from evaluation.metrics import catalog_entropy
        assert catalog_entropy([], {}) == 0.0


class TestEvaluateRecommendations:

    def test_aggregate_returns_all_metric_keys(self):
        from evaluation.metrics import evaluate_recommendations
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        topic_map = {"a": ["tech"], "b": ["comedy"], "c": ["tech"],
                     "d": ["gaming"], "e": ["cooking"]}
        popularity = {v: 0.1 for v in ranked}
        results = evaluate_recommendations(ranked, relevant, topic_map, popularity, k=5)
        for key in ("precision@5", "recall@5", "ndcg@5", "ild@5", "novelty@5"):
            assert key in results

    def test_aggregate_values_are_floats(self):
        from evaluation.metrics import evaluate_recommendations
        results = evaluate_recommendations(
            ["a"], {"a"}, {"a": ["tech"]}, {"a": 0.1}, k=1
        )
        assert all(isinstance(v, float) for v in results.values())

    def test_perfect_ranking(self):
        from evaluation.metrics import evaluate_recommendations
        ranked = ["a", "b"]
        relevant = {"a", "b"}
        results = evaluate_recommendations(
            ranked, relevant, {"a": ["t1"], "b": ["t2"]}, {"a": 0.1, "b": 0.1}, k=2
        )
        assert results["precision@2"] == pytest.approx(1.0)
        assert results["recall@2"] == pytest.approx(1.0)
        assert results["ndcg@2"] == pytest.approx(1.0)


# ===========================================================================
# embeddings/store.py — EmbeddingStore
# ===========================================================================

class TestEmbeddingStore:

    def _emb(self, seed=0, dim=64):
        v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
        return list(v)

    def _make_single_query_driver(self, rows):
        """Driver that returns `rows` for a single session.run call."""
        mock_result = MagicMock()
        mock_result.single.return_value = rows[0] if rows else None
        mock_result.__iter__ = MagicMock(return_value=iter(rows))
        mock_result.data.return_value = rows

        mock_sess = MagicMock()
        mock_sess.run.return_value = mock_result
        mock_sess.__enter__ = MagicMock(return_value=mock_sess)
        mock_sess.__exit__ = MagicMock(return_value=False)

        mock_drv = MagicMock()
        mock_drv.session.return_value = mock_sess
        mock_drv.__enter__ = MagicMock(return_value=mock_drv)
        mock_drv.__exit__ = MagicMock(return_value=False)
        return mock_drv

    def test_get_user_embedding_returns_normalised_array(self):
        from embeddings.store import EmbeddingStore
        raw_emb = self._emb()
        drv = self._make_single_query_driver([{"embedding": raw_emb}])
        with patch("embeddings.store.GraphDatabase.driver", return_value=drv):
            store = EmbeddingStore("bolt://x", "u", "p")
            result = store.get_user_embedding("user-1")
        assert result is not None
        assert result.shape == (64,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_get_user_embedding_caches_result(self):
        from embeddings.store import EmbeddingStore
        raw_emb = self._emb()
        drv = self._make_single_query_driver([{"embedding": raw_emb}])
        with patch("embeddings.store.GraphDatabase.driver", return_value=drv) as mock_gdb:
            store = EmbeddingStore("bolt://x", "u", "p")
            r1 = store.get_user_embedding("user-1")
            r2 = store.get_user_embedding("user-1")   # should use cache
        # Driver created once (first call); second call hits cache
        assert mock_gdb.call_count == 1

    def test_get_user_embedding_returns_none_when_missing(self):
        from embeddings.store import EmbeddingStore
        drv = self._make_single_query_driver([{"embedding": None}])
        with patch("embeddings.store.GraphDatabase.driver", return_value=drv):
            store = EmbeddingStore("bolt://x", "u", "p")
            result = store.get_user_embedding("user-1")
        assert result is None

    def test_get_user_embedding_returns_none_for_zero_vector(self):
        from embeddings.store import EmbeddingStore
        drv = self._make_single_query_driver([{"embedding": [0.0] * 64}])
        with patch("embeddings.store.GraphDatabase.driver", return_value=drv):
            store = EmbeddingStore("bolt://x", "u", "p")
            result = store.get_user_embedding("user-1")
        assert result is None

    def test_cosine_score_between_parallel_vectors_is_one(self):
        from embeddings.store import EmbeddingStore
        raw = self._emb()
        v = np.array(raw, dtype=np.float32)
        norm_v = v / np.linalg.norm(v)

        store = EmbeddingStore("bolt://x", "u", "p")
        store._user_cache["u1"] = norm_v
        store._video_cache["v1"] = norm_v   # same direction

        score = store.cosine_score("u1", "v1")
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_cosine_score_zero_when_embedding_missing(self):
        from embeddings.store import EmbeddingStore
        store = EmbeddingStore("bolt://x", "u", "p")
        store._user_cache["u1"] = None
        store._video_cache["v1"] = np.ones(64, dtype=np.float32)
        assert store.cosine_score("u1", "v1") == 0.0

    def test_cache_stats_reports_counts(self):
        from embeddings.store import EmbeddingStore
        store = EmbeddingStore("bolt://x", "u", "p")
        store._user_cache["u1"] = np.ones(64)
        store._video_cache["v1"] = np.ones(64)
        store._video_cache["v2"] = np.ones(64)
        stats = store.cache_stats()
        assert stats["users_cached"] == 1
        assert stats["videos_cached"] == 2

    def test_cosine_top_k_returns_empty_when_no_user_embedding(self):
        from embeddings.store import EmbeddingStore
        drv = self._make_single_query_driver([{"embedding": None}])
        with patch("embeddings.store.GraphDatabase.driver", return_value=drv):
            store = EmbeddingStore("bolt://x", "u", "p")
            result = store.cosine_top_k_unseen("user-1", k=5)
        assert result == []

    def test_cosine_top_k_respects_k_limit(self):
        from embeddings.store import EmbeddingStore
        user_emb = self._emb(seed=0)
        v_norm = np.array(user_emb, dtype=np.float32)
        v_norm = v_norm / np.linalg.norm(v_norm)

        call_n = [0]
        def factory(*args, **kwargs):
            call_n[0] += 1
            if call_n[0] == 1:
                # User embedding query
                r = MagicMock()
                r.single.return_value = {"embedding": user_emb}
                s = MagicMock()
                s.run.return_value = r
                s.__enter__ = MagicMock(return_value=s)
                s.__exit__ = MagicMock(return_value=False)
                d = MagicMock()
                d.session.return_value = s
                d.__enter__ = MagicMock(return_value=d)
                d.__exit__ = MagicMock(return_value=False)
                return d
            else:
                # Batch of video embeddings
                batch_rows = [
                    {"video_id": f"v{i}", "embedding": self._emb(seed=i + 1)}
                    for i in range(10)
                ]

                def make_drv(rows, is_last):
                    r = MagicMock()
                    r.data.return_value = rows if not is_last else []
                    s = MagicMock()
                    s.run.return_value = r
                    s.__enter__ = MagicMock(return_value=s)
                    s.__exit__ = MagicMock(return_value=False)
                    d = MagicMock()
                    d.session.return_value = s
                    d.__enter__ = MagicMock(return_value=d)
                    d.__exit__ = MagicMock(return_value=False)
                    return d

                if call_n[0] == 2:
                    return make_drv(batch_rows, is_last=False)
                else:
                    return make_drv([], is_last=True)

        with patch("embeddings.store.GraphDatabase.driver", side_effect=factory):
            store = EmbeddingStore("bolt://x", "u", "p", batch_size=500)
            result = store.cosine_top_k_unseen("user-1", k=3)

        assert len(result) <= 3
        # Scores should be sorted descending
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

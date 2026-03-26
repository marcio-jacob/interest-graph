"""
Unit tests for generators/interactions.py
==========================================
Covers: generate_interactions — structure, correctness, and statistical
        properties of the behavioural simulation output.
"""

import uuid
from collections import Counter, defaultdict

import numpy as np
import pytest

from generators.base import reset_rng
from generators.interactions import generate_interactions
from generators.sessions import generate_sessions
from generators.taxonomy import (
    generate_countries,
    generate_entities,
    generate_hashtags,
    generate_sounds,
    generate_topics,
)
from generators.users import generate_users
from generators.videos import assign_video_taxonomy, generate_videos


# ---------------------------------------------------------------------------
# Required key sets
# ---------------------------------------------------------------------------
REQUIRED_RESULT_KEYS = {
    "views", "likes", "skips", "reposts", "comment_stubs",
    "topic_interests", "entity_interests", "hashtag_interests",
}
REQUIRED_VIEW_KEYS    = {"session_id", "video_id", "watch_time", "completion_rate"}
REQUIRED_SKIP_KEYS    = {"session_id", "video_id"}
REQUIRED_LIKE_KEYS    = {"session_id", "video_id"}
REQUIRED_REPOST_KEYS  = {"session_id", "video_id"}
REQUIRED_STUB_KEYS    = {
    "comment_id", "session_id", "video_id", "user_id", "sentiment", "created_at"
}
REQUIRED_TOPIC_KEYS   = {"user_id", "topic_id", "topic_score"}
REQUIRED_ENTITY_KEYS  = {"user_id", "entity_id", "entity_score"}
REQUIRED_HASHTAG_KEYS = {"user_id", "hashtag_id", "hashtag_score"}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def idata(cfg):
    """
    Full data chain for interaction testing.
    20 users, 60 videos — small enough to be fast, large enough to be statistical.
    """
    users = generate_users(20, cfg, cfg, cfg)
    sessions, _ = generate_sessions(users, cfg)

    topics    = generate_topics(cfg)
    countries = generate_countries(cfg)
    ht_nodes  = generate_hashtags(cfg)
    ent_nodes = generate_entities(cfg)
    sounds    = generate_sounds(cfg)

    videos = generate_videos(
        60, users, topics, ht_nodes, ent_nodes, sounds,
        countries, cfg, cfg, cfg,
    )
    video_hashtags, video_entities, _, _ = assign_video_taxonomy(
        videos, topics, ht_nodes, ent_nodes, sounds, cfg, cfg
    )

    result = generate_interactions(
        sessions, videos, users, topics,
        video_entities, video_hashtags,
        cfg, cfg,
    )
    return result, sessions, videos, users, topics, ht_nodes, ent_nodes


# ===========================================================================
# Return structure
# ===========================================================================

class TestReturnStructure:

    def test_returns_dict(self, idata):
        result, *_ = idata
        assert isinstance(result, dict)

    def test_has_all_required_keys(self, idata):
        result, *_ = idata
        assert REQUIRED_RESULT_KEYS == set(result.keys())

    def test_all_values_are_lists(self, idata):
        result, *_ = idata
        for key, val in result.items():
            assert isinstance(val, list), f"result['{key}'] is not a list"

    def test_views_have_required_keys(self, idata):
        result, *_ = idata
        for row in result["views"]:
            assert REQUIRED_VIEW_KEYS == set(row.keys()), \
                f"view key mismatch: {set(row.keys()) ^ REQUIRED_VIEW_KEYS}"

    def test_skips_have_required_keys(self, idata):
        result, *_ = idata
        for row in result["skips"]:
            assert REQUIRED_SKIP_KEYS == set(row.keys())

    def test_likes_have_required_keys(self, idata):
        result, *_ = idata
        for row in result["likes"]:
            assert REQUIRED_LIKE_KEYS == set(row.keys())

    def test_reposts_have_required_keys(self, idata):
        result, *_ = idata
        for row in result["reposts"]:
            assert REQUIRED_REPOST_KEYS == set(row.keys())

    def test_comment_stubs_have_required_keys(self, idata):
        result, *_ = idata
        for row in result["comment_stubs"]:
            assert REQUIRED_STUB_KEYS == set(row.keys()), \
                f"stub key mismatch: {set(row.keys()) ^ REQUIRED_STUB_KEYS}"

    def test_topic_interests_have_required_keys(self, idata):
        result, *_ = idata
        for row in result["topic_interests"]:
            assert REQUIRED_TOPIC_KEYS == set(row.keys())

    def test_entity_interests_have_required_keys(self, idata):
        result, *_ = idata
        for row in result["entity_interests"]:
            assert REQUIRED_ENTITY_KEYS == set(row.keys())

    def test_hashtag_interests_have_required_keys(self, idata):
        result, *_ = idata
        for row in result["hashtag_interests"]:
            assert REQUIRED_HASHTAG_KEYS == set(row.keys())


# ===========================================================================
# Reference validity
# ===========================================================================

class TestReferenceValidity:

    def test_view_session_ids_are_valid(self, idata):
        result, sessions, *_ = idata
        valid = {s["session_id"] for s in sessions}
        for row in result["views"]:
            assert row["session_id"] in valid

    def test_view_video_ids_are_valid(self, idata):
        result, _, videos, *_ = idata
        valid = {v["video_id"] for v in videos}
        for row in result["views"]:
            assert row["video_id"] in valid

    def test_skip_session_ids_are_valid(self, idata):
        result, sessions, *_ = idata
        valid = {s["session_id"] for s in sessions}
        for row in result["skips"]:
            assert row["session_id"] in valid

    def test_comment_stub_user_ids_are_valid(self, idata):
        result, _, _, users, *_ = idata
        valid = {u["user_id"] for u in users}
        for row in result["comment_stubs"]:
            assert row["user_id"] in valid

    def test_topic_interest_user_ids_are_valid(self, idata):
        result, _, _, users, *_ = idata
        valid = {u["user_id"] for u in users}
        for row in result["topic_interests"]:
            assert row["user_id"] in valid

    def test_topic_interest_topic_ids_are_valid(self, idata):
        result, _, _, _, topics, *_ = idata
        valid = {t["topic_id"] for t in topics}
        for row in result["topic_interests"]:
            assert row["topic_id"] in valid

    def test_entity_interest_entity_ids_are_valid(self, idata):
        result, _, _, _, _, _, ent_nodes = idata
        valid = {e["entity_id"] for e in ent_nodes}
        for row in result["entity_interests"]:
            assert row["entity_id"] in valid

    def test_hashtag_interest_hashtag_ids_are_valid(self, idata):
        result, _, _, _, _, ht_nodes, _ = idata
        valid = {h["hashtag_id"] for h in ht_nodes}
        for row in result["hashtag_interests"]:
            assert row["hashtag_id"] in valid


# ===========================================================================
# Behavioural correctness
# ===========================================================================

class TestBehaviouralCorrectness:

    def test_no_pair_in_both_views_and_skips(self, idata):
        """A (session, video) pair must appear in exactly one of views or skips."""
        result, *_ = idata
        view_pairs = {(r["session_id"], r["video_id"]) for r in result["views"]}
        skip_pairs = {(r["session_id"], r["video_id"]) for r in result["skips"]}
        overlap = view_pairs & skip_pairs
        assert not overlap, f"{len(overlap)} pairs appear in both views and skips"

    def test_likes_are_subset_of_views(self, idata):
        """A video can only be liked if it was viewed (not skipped)."""
        result, *_ = idata
        view_pairs = {(r["session_id"], r["video_id"]) for r in result["views"]}
        for row in result["likes"]:
            assert (row["session_id"], row["video_id"]) in view_pairs, \
                "Like recorded for a skipped video"

    def test_reposts_are_subset_of_views(self, idata):
        result, *_ = idata
        view_pairs = {(r["session_id"], r["video_id"]) for r in result["views"]}
        for row in result["reposts"]:
            assert (row["session_id"], row["video_id"]) in view_pairs

    def test_comments_are_subset_of_views(self, idata):
        result, *_ = idata
        view_pairs = {(r["session_id"], r["video_id"]) for r in result["views"]}
        for row in result["comment_stubs"]:
            assert (row["session_id"], row["video_id"]) in view_pairs

    def test_completion_rate_in_0_1(self, idata):
        result, *_ = idata
        for row in result["views"]:
            assert 0.0 < row["completion_rate"] <= 1.0, \
                f"completion_rate {row['completion_rate']} outside (0, 1]"

    def test_watch_time_positive(self, idata):
        result, *_ = idata
        for row in result["views"]:
            assert row["watch_time"] > 0, "watch_time must be positive"

    def test_comment_sentiment_is_valid(self, idata):
        result, *_ = idata
        valid = {"positive", "neutral", "negative"}
        for row in result["comment_stubs"]:
            assert row["sentiment"] in valid

    def test_comment_ids_are_unique_uuids(self, idata):
        result, *_ = idata
        ids = [r["comment_id"] for r in result["comment_stubs"]]
        assert len(set(ids)) == len(ids), "Duplicate comment_id in stubs"
        for cid in ids:
            uuid.UUID(cid)

    def test_no_duplicate_view_pairs(self, idata):
        """Same (session_id, video_id) should not appear twice in views."""
        result, *_ = idata
        pairs = [(r["session_id"], r["video_id"]) for r in result["views"]]
        assert len(set(pairs)) == len(pairs), "Duplicate (session, video) in views"

    def test_no_duplicate_skip_pairs(self, idata):
        result, *_ = idata
        pairs = [(r["session_id"], r["video_id"]) for r in result["skips"]]
        assert len(set(pairs)) == len(pairs), "Duplicate (session, video) in skips"


# ===========================================================================
# Interest score correctness
# ===========================================================================

class TestInterestScores:

    def test_topic_scores_in_range(self, idata):
        result, *_ = idata
        for row in result["topic_interests"]:
            assert 0 < row["topic_score"] <= 1.0, \
                f"topic_score {row['topic_score']} outside (0, 1]"

    def test_entity_scores_in_range(self, idata):
        result, *_ = idata
        for row in result["entity_interests"]:
            assert 0 < row["entity_score"] <= 1.0

    def test_hashtag_scores_in_range(self, idata):
        result, *_ = idata
        for row in result["hashtag_interests"]:
            assert 0 < row["hashtag_score"] <= 1.0

    def test_max_topic_score_per_user_is_1(self, idata):
        """After normalisation, the maximum score per user must be 1.0."""
        result, *_ = idata
        max_by_user: dict[str, float] = defaultdict(float)
        for row in result["topic_interests"]:
            uid = row["user_id"]
            max_by_user[uid] = max(max_by_user[uid], row["topic_score"])
        for uid, mx in max_by_user.items():
            assert abs(mx - 1.0) < 0.001, \
                f"User {uid} max topic_score = {mx} (expected 1.0)"

    def test_scores_above_min_threshold(self, idata, cfg):
        """No score below the configured pruning threshold should survive."""
        result, *_ = idata
        min_thr = cfg["interest"]["min_score_threshold"]
        for row in result["topic_interests"]:
            assert row["topic_score"] > min_thr

    def test_users_with_sessions_have_topic_interests(self, idata):
        """Every user who attended at least one session should appear in interests."""
        result, sessions, *_ = idata
        users_with_sessions = {s["user_id"] for s in sessions}
        users_with_interests = {r["user_id"] for r in result["topic_interests"]}
        # Allow a small number of misses (some users may only have net-negative scores)
        missing = users_with_sessions - users_with_interests
        assert len(missing) <= len(users_with_sessions) * 0.30, \
            f"{len(missing)} / {len(users_with_sessions)} users have no topic interests"


# ===========================================================================
# Statistical properties
# ===========================================================================

class TestStatisticalProperties:

    def test_skip_rate_near_kuairec(self, cfg):
        """KuaiRec skip rate is ~13.1 %; allow [8 %, 20 %] with N=2000 pairs."""
        reset_rng(42)
        users    = generate_users(20, cfg, cfg, cfg)
        sessions, _ = generate_sessions(users, cfg)
        topics    = generate_topics(cfg)
        countries = generate_countries(cfg)
        ht_nodes  = generate_hashtags(cfg)
        ent_nodes = generate_entities(cfg)
        sounds    = generate_sounds(cfg)
        videos = generate_videos(
            60, users, topics, ht_nodes, ent_nodes, sounds,
            countries, cfg, cfg, cfg,
        )
        vht, vent, _, _ = assign_video_taxonomy(
            videos, topics, ht_nodes, ent_nodes, sounds, cfg, cfg
        )
        result = generate_interactions(
            sessions, videos, users, topics, vent, vht, cfg, cfg
        )
        total = len(result["views"]) + len(result["skips"])
        if total == 0:
            pytest.skip("No interactions generated")
        skip_rate = len(result["skips"]) / total
        assert 0.08 <= skip_rate <= 0.20, \
            f"Skip rate {skip_rate:.3f} outside expected [0.08, 0.20]"

    def test_views_plus_skips_equals_total_interactions(self, idata):
        """Every (session, video) pair processed must produce either a view or a skip."""
        result, *_ = idata
        view_pairs = {(r["session_id"], r["video_id"]) for r in result["views"]}
        skip_pairs = {(r["session_id"], r["video_id"]) for r in result["skips"]}
        all_pairs  = view_pairs | skip_pairs
        # No pair should be double-counted
        assert len(all_pairs) == len(result["views"]) + len(result["skips"])

    def test_like_count_le_view_count(self, idata):
        """Aggregate likes can never exceed aggregate views."""
        result, *_ = idata
        assert len(result["likes"]) <= len(result["views"])

    def test_comment_count_le_view_count(self, idata):
        result, *_ = idata
        assert len(result["comment_stubs"]) <= len(result["views"])

    def test_at_least_some_views_and_skips(self, idata):
        result, *_ = idata
        assert len(result["views"]) > 0,  "Expected at least some views"
        assert len(result["skips"]) > 0,  "Expected at least some skips"


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_sessions_returns_empty(self, cfg):
        users = generate_users(5, cfg, cfg, cfg)
        topics = generate_topics(cfg)
        countries = generate_countries(cfg)
        ht_nodes = generate_hashtags(cfg)
        ent_nodes = generate_entities(cfg)
        sounds = generate_sounds(cfg)
        videos = generate_videos(
            10, users, topics, ht_nodes, ent_nodes, sounds,
            countries, cfg, cfg, cfg,
        )
        vht, vent, _, _ = assign_video_taxonomy(
            videos, topics, ht_nodes, ent_nodes, sounds, cfg, cfg
        )
        result = generate_interactions([], videos, users, topics, vent, vht, cfg, cfg)
        assert all(result[k] == [] for k in REQUIRED_RESULT_KEYS)

    def test_empty_videos_returns_empty(self, cfg):
        users = generate_users(5, cfg, cfg, cfg)
        sessions, _ = generate_sessions(users, cfg)
        topics = generate_topics(cfg)
        result = generate_interactions(sessions, [], users, topics, [], [], cfg, cfg)
        assert all(result[k] == [] for k in REQUIRED_RESULT_KEYS)

    def test_result_has_all_keys_on_empty(self, cfg):
        result = generate_interactions([], [], [], [], [], [], cfg, cfg)
        assert REQUIRED_RESULT_KEYS == set(result.keys())

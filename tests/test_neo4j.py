"""
Unit tests for neo4j/connection.py, neo4j/schema.py, neo4j/loader.py
=======================================================================
All tests use mock objects — no live Neo4j instance required.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, call, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_driver(session_cm=None):
    """Return a MagicMock that behaves like a neo4j Driver."""
    driver = MagicMock()
    mock_session = MagicMock()
    if session_cm is not None:
        mock_session.__enter__ = MagicMock(return_value=session_cm)
        mock_session.__exit__ = MagicMock(return_value=False)
    else:
        inner = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=inner)
        mock_session.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = mock_session
    return driver


# ===========================================================================
# neo4j/connection.py
# ===========================================================================

class TestGetDriver:

    def test_returns_driver_when_uri_set(self):
        from neo4j.connection import get_driver, reset_driver
        reset_driver()
        env = {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": "pass"}
        with patch.dict("os.environ", env):
            with patch("neo4j.connection.GraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = MagicMock()
                driver = get_driver()
                assert driver is not None
                mock_gdb.driver.assert_called_once_with(
                    "bolt://localhost:7687", auth=("neo4j", "pass")
                )
        reset_driver()

    def test_raises_when_uri_missing(self):
        from neo4j.connection import get_driver, reset_driver
        reset_driver()
        with patch.dict("os.environ", {}, clear=True):
            with patch("neo4j.connection.os.environ.get", return_value=""):
                with pytest.raises(EnvironmentError, match="NEO4J_URI"):
                    get_driver()
        reset_driver()

    def test_singleton_reuses_driver(self):
        from neo4j.connection import get_driver, reset_driver
        reset_driver()
        env = {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": "pass"}
        with patch.dict("os.environ", env):
            with patch("neo4j.connection.GraphDatabase") as mock_gdb:
                mock_driver = MagicMock()
                mock_gdb.driver.return_value = mock_driver
                d1 = get_driver()
                d2 = get_driver()
                assert d1 is d2
                assert mock_gdb.driver.call_count == 1
        reset_driver()


class TestCloseDriver:

    def test_close_calls_driver_close(self):
        from neo4j.connection import close_driver, reset_driver
        import neo4j.connection as conn_mod
        mock_drv = MagicMock()
        conn_mod._driver = mock_drv
        close_driver()
        mock_drv.close.assert_called_once()
        assert conn_mod._driver is None

    def test_close_when_already_none_is_noop(self):
        from neo4j.connection import close_driver, reset_driver
        reset_driver()
        close_driver()  # should not raise


class TestResetDriver:

    def test_reset_sets_driver_to_none(self):
        from neo4j.connection import reset_driver
        import neo4j.connection as conn_mod
        conn_mod._driver = MagicMock()
        reset_driver()
        assert conn_mod._driver is None


class TestTestConnection:

    def test_returns_true_on_successful_ping(self):
        from neo4j.connection import test_connection, reset_driver
        reset_driver()
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: 1
        mock_session.run.return_value.single.return_value = mock_record
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_drv = MagicMock()
        mock_drv.session.return_value = mock_session
        mock_drv.get_server_info.return_value = MagicMock(address="localhost:7687", agent="Neo4j/5.0")

        with patch("neo4j.connection.get_driver", return_value=mock_drv):
            result = test_connection()
        assert result is True

    def test_returns_false_on_exception(self):
        from neo4j.connection import test_connection, reset_driver
        reset_driver()
        mock_drv = MagicMock()
        mock_drv.session.side_effect = Exception("connection refused")
        with patch("neo4j.connection.get_driver", return_value=mock_drv):
            result = test_connection()
        assert result is False


# ===========================================================================
# neo4j/schema.py
# ===========================================================================

class TestApplySchema:

    def test_runs_all_constraints_and_indexes(self):
        from neo4j.schema import apply_schema, _CONSTRAINTS, _INDEXES
        inner = MagicMock()
        driver = _make_driver(inner)
        apply_schema(driver)
        total_stmts = len(_CONSTRAINTS) + len(_INDEXES)
        assert inner.run.call_count == total_stmts

    def test_all_constraints_are_create_constraint(self):
        from neo4j.schema import _CONSTRAINTS
        for stmt in _CONSTRAINTS:
            assert stmt.strip().upper().startswith("CREATE CONSTRAINT"), stmt

    def test_all_indexes_are_create_index(self):
        from neo4j.schema import _INDEXES
        for stmt in _INDEXES:
            assert stmt.strip().upper().startswith("CREATE INDEX"), stmt

    def test_nine_constraints(self):
        from neo4j.schema import _CONSTRAINTS
        assert len(_CONSTRAINTS) == 9

    def test_three_indexes(self):
        from neo4j.schema import _INDEXES
        assert len(_INDEXES) == 3

    def test_all_constraints_have_if_not_exists(self):
        from neo4j.schema import _CONSTRAINTS
        for stmt in _CONSTRAINTS:
            assert "IF NOT EXISTS" in stmt.upper(), stmt

    def test_all_indexes_have_if_not_exists(self):
        from neo4j.schema import _INDEXES
        for stmt in _INDEXES:
            assert "IF NOT EXISTS" in stmt.upper(), stmt

    def test_required_node_labels_have_constraints(self):
        from neo4j.schema import _CONSTRAINTS
        text = " ".join(_CONSTRAINTS).lower()
        for label in ("user", "usersession", "video", "hashtag", "entity", "sound", "country", "topic", "comment"):
            assert label in text, f"Missing constraint for label: {label}"


# ===========================================================================
# neo4j/loader.py — helper
# ===========================================================================

class TestRunBatches:

    def test_empty_data_returns_zero(self):
        from neo4j.loader import _run_batches
        driver = _make_driver()
        result = _run_batches(driver, [], "RETURN 1", 500, "test")
        assert result == 0
        driver.session.assert_not_called()

    def test_returns_len_of_data(self):
        from neo4j.loader import _run_batches
        inner = MagicMock()
        driver = _make_driver(inner)
        data = [{"x": i} for i in range(10)]
        result = _run_batches(driver, data, "UNWIND $batch AS row RETURN row", 500, "test")
        assert result == 10

    def test_batches_correctly(self):
        from neo4j.loader import _run_batches
        inner = MagicMock()
        driver = _make_driver(inner)
        data = [{"x": i} for i in range(1250)]
        _run_batches(driver, data, "UNWIND $batch AS row RETURN row", 500, "test")
        # 1250 items / 500 batch_size = 3 batches
        assert inner.run.call_count == 3

    def test_batch_sizes_are_correct(self):
        from neo4j.loader import _run_batches
        inner = MagicMock()
        driver = _make_driver(inner)
        data = [{"x": i} for i in range(1100)]
        _run_batches(driver, data, "UNWIND $batch AS row RETURN row", 500, "test")
        calls = inner.run.call_args_list
        batches = [c.kwargs.get("batch") or c.args[1] for c in calls]
        # Try keyword or positional
        batches = []
        for c in calls:
            if "batch" in c.kwargs:
                batches.append(c.kwargs["batch"])
            else:
                batches.append(c.args[1])
        assert len(batches[0]) == 500
        assert len(batches[1]) == 500
        assert len(batches[2]) == 100


# ===========================================================================
# neo4j/loader.py — node upload functions
# ===========================================================================

class TestNodeUploads:

    def _check_upload(self, fn, sample_row: dict):
        inner = MagicMock()
        driver = _make_driver(inner)
        data = [sample_row]
        result = fn(driver, data)
        assert result == 1
        inner.run.assert_called_once()
        cypher, kwargs = inner.run.call_args.args[0], inner.run.call_args.kwargs
        assert "UNWIND" in cypher
        assert "MERGE" in cypher
        batch_arg = kwargs.get("batch") or inner.run.call_args.args[1]
        assert batch_arg == data

    def test_upload_countries(self):
        from neo4j.loader import upload_countries
        self._check_upload(upload_countries, {"country_id": "C01", "name": "USA", "iso": "US"})

    def test_upload_topics(self):
        from neo4j.loader import upload_topics
        self._check_upload(upload_topics, {"topic_id": "T01", "name": "Cooking", "slug": "cooking_food"})

    def test_upload_sounds(self):
        from neo4j.loader import upload_sounds
        self._check_upload(upload_sounds, {"song_id": "S001", "song_name": "Dynamite", "singer": "BTS", "genre": "K-pop", "country": "KR"})

    def test_upload_hashtags(self):
        from neo4j.loader import upload_hashtags
        self._check_upload(upload_hashtags, {"hashtag_id": "HT001", "name": "#foodtok", "topic_slug": "cooking_food"})

    def test_upload_entities(self):
        from neo4j.loader import upload_entities
        self._check_upload(upload_entities, {"entity_id": "E01", "name": "Gordon Ramsay", "aliases": ["Chef Ramsay"], "topic_slug": "cooking_food"})

    def test_upload_sessions(self):
        from neo4j.loader import upload_sessions
        from datetime import datetime
        self._check_upload(upload_sessions, {
            "session_id": "sess-1",
            "user_id": "u-1",
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 1, 0, 30),
        })

    def test_upload_videos(self):
        from neo4j.loader import upload_videos
        from datetime import datetime
        self._check_upload(upload_videos, {
            "video_id": "vid-1",
            "author_id": "u-1",
            "video_duration": 30,
            "posted_at": datetime(2024, 1, 1),
            "description": "Test video",
            "topic_id": "T01",
            "likes": 10,
            "downloads": 2,
            "shares": 3,
            "reposts": 1,
            "comments": 4,
        })

    def test_upload_comments(self):
        from neo4j.loader import upload_comments
        from datetime import datetime
        self._check_upload(upload_comments, {
            "comment_id": "cmt-1",
            "video_id": "vid-1",
            "user_id": "u-1",
            "comment_text": "Great!",
            "comment_sentiment": "positive",
            "created_at": datetime(2024, 1, 1),
        })

    def test_upload_users_calls_two_queries(self):
        """upload_users runs a base SET query + a Creator label query per batch."""
        from neo4j.loader import upload_users
        inner = MagicMock()
        driver = _make_driver(inner)
        data = [{"user_id": "u-1", "username": "alice", "joined_at": None,
                 "followers": 100, "following": 50, "like_count": 200,
                 "average_watch_time": 15.0, "last_login": None,
                 "country_id": "C01", "is_creator": True}]
        result = upload_users(driver, data)
        assert result == 1
        # Two queries per batch: base properties + Creator label
        assert inner.run.call_count == 2

    def test_upload_users_returns_correct_count(self):
        from neo4j.loader import upload_users
        inner = MagicMock()
        driver = _make_driver(inner)
        data = [{"user_id": f"u-{i}", "username": f"user{i}", "joined_at": None,
                 "followers": 0, "following": 0, "like_count": 0,
                 "average_watch_time": 0.0, "last_login": None,
                 "country_id": "C01", "is_creator": False}
                for i in range(5)]
        assert upload_users(driver, data) == 5

    def test_upload_users_empty_returns_zero(self):
        from neo4j.loader import upload_users
        driver = _make_driver()
        assert upload_users(driver, []) == 0
        driver.session.assert_not_called()


# ===========================================================================
# neo4j/loader.py — relationship upload functions
# ===========================================================================

class TestRelUploads:

    def _check_rel(self, fn, sample_row: dict):
        inner = MagicMock()
        driver = _make_driver(inner)
        result = fn(driver, [sample_row])
        assert result == 1
        inner.run.assert_called_once()
        cypher = inner.run.call_args.args[0]
        assert "UNWIND" in cypher
        assert "MERGE" in cypher
        assert "MATCH" in cypher

    def test_upload_rel_has_session(self):
        from neo4j.loader import upload_rel_has_session
        self._check_rel(upload_rel_has_session, {"user_id": "u-1", "session_id": "s-1"})

    def test_upload_rel_last_session(self):
        from neo4j.loader import upload_rel_last_session
        self._check_rel(upload_rel_last_session, {"user_id": "u-1", "session_id": "s-1"})

    def test_upload_rel_prev_session(self):
        from neo4j.loader import upload_rel_prev_session
        self._check_rel(upload_rel_prev_session, {"session_id": "s-2", "prev_session_id": "s-1"})

    def test_upload_rel_created_by(self):
        from neo4j.loader import upload_rel_created_by
        self._check_rel(upload_rel_created_by, {"video_id": "v-1", "author_id": "u-1"})

    def test_upload_rel_viewed(self):
        from neo4j.loader import upload_rel_viewed
        self._check_rel(upload_rel_viewed, {
            "session_id": "s-1", "video_id": "v-1",
            "watch_time": 25.5, "completion_rate": 0.85,
        })

    def test_upload_rel_liked(self):
        from neo4j.loader import upload_rel_liked
        self._check_rel(upload_rel_liked, {"session_id": "s-1", "video_id": "v-1"})

    def test_upload_rel_skipped(self):
        from neo4j.loader import upload_rel_skipped
        self._check_rel(upload_rel_skipped, {"session_id": "s-1", "video_id": "v-1"})

    def test_upload_rel_reposted(self):
        from neo4j.loader import upload_rel_reposted
        self._check_rel(upload_rel_reposted, {"session_id": "s-1", "video_id": "v-1"})

    def test_upload_rel_commented(self):
        from neo4j.loader import upload_rel_commented
        self._check_rel(upload_rel_commented, {"session_id": "s-1", "comment_id": "c-1"})

    def test_upload_rel_comment_on_video(self):
        from neo4j.loader import upload_rel_comment_on_video
        self._check_rel(upload_rel_comment_on_video, {"comment_id": "c-1", "video_id": "v-1"})

    def test_upload_rel_video_hashtag(self):
        from neo4j.loader import upload_rel_video_hashtag
        self._check_rel(upload_rel_video_hashtag, {"video_id": "v-1", "hashtag_id": "HT001"})

    def test_upload_rel_video_entity(self):
        from neo4j.loader import upload_rel_video_entity
        self._check_rel(upload_rel_video_entity, {"video_id": "v-1", "entity_id": "E01"})

    def test_upload_rel_video_sound(self):
        from neo4j.loader import upload_rel_video_sound
        self._check_rel(upload_rel_video_sound, {"video_id": "v-1", "song_id": "S001"})

    def test_upload_rel_video_topic(self):
        from neo4j.loader import upload_rel_video_topic
        self._check_rel(upload_rel_video_topic, {"video_id": "v-1", "topic_id": "T01", "is_primary": True})

    def test_upload_rel_entity_topic(self):
        from neo4j.loader import upload_rel_entity_topic
        self._check_rel(upload_rel_entity_topic, {"entity_id": "E01", "topic_id": "T01", "is_primary": True})

    def test_upload_rel_user_country(self):
        from neo4j.loader import upload_rel_user_country
        self._check_rel(upload_rel_user_country, {"user_id": "u-1", "country_id": "C01"})

    def test_upload_rel_video_country(self):
        from neo4j.loader import upload_rel_video_country
        self._check_rel(upload_rel_video_country, {"video_id": "v-1", "country_id": "C01"})

    def test_upload_rel_sound_country(self):
        from neo4j.loader import upload_rel_sound_country
        self._check_rel(upload_rel_sound_country, {"song_id": "S001", "country_id": "C01"})

    def test_upload_rel_follows(self):
        from neo4j.loader import upload_rel_follows
        self._check_rel(upload_rel_follows, {"follower_id": "u-1", "followee_id": "u-2", "engagement_score": 0.7})

    def test_upload_rel_interested_topic(self):
        from neo4j.loader import upload_rel_interested_topic
        self._check_rel(upload_rel_interested_topic, {"user_id": "u-1", "topic_id": "T01", "topic_score": 0.9})

    def test_upload_rel_interested_entity(self):
        from neo4j.loader import upload_rel_interested_entity
        self._check_rel(upload_rel_interested_entity, {"user_id": "u-1", "entity_id": "E01", "entity_score": 0.7})

    def test_upload_rel_interested_hashtag(self):
        from neo4j.loader import upload_rel_interested_hashtag
        self._check_rel(upload_rel_interested_hashtag, {"user_id": "u-1", "hashtag_id": "HT001", "hashtag_score": 0.6})

    def test_upload_rel_viewed_sets_properties(self):
        """VIEWED edge must carry watch_time and completion_rate as SET properties."""
        from neo4j.loader import upload_rel_viewed
        inner = MagicMock()
        driver = _make_driver(inner)
        upload_rel_viewed(driver, [{"session_id": "s-1", "video_id": "v-1", "watch_time": 30.0, "completion_rate": 1.0}])
        cypher = inner.run.call_args.args[0]
        assert "watch_time" in cypher
        assert "completion_rate" in cypher

    def test_upload_rel_follows_sets_engagement_score(self):
        from neo4j.loader import upload_rel_follows
        inner = MagicMock()
        driver = _make_driver(inner)
        upload_rel_follows(driver, [{"follower_id": "a", "followee_id": "b", "engagement_score": 0.5}])
        cypher = inner.run.call_args.args[0]
        assert "engagement_score" in cypher

    def test_empty_data_returns_zero(self):
        from neo4j.loader import (
            upload_rel_has_session, upload_rel_liked, upload_rel_follows,
            upload_rel_interested_topic,
        )
        driver = _make_driver()
        for fn in (upload_rel_has_session, upload_rel_liked, upload_rel_follows, upload_rel_interested_topic):
            driver.reset_mock()
            assert fn(driver, []) == 0
            driver.session.assert_not_called()


# ===========================================================================
# Function inventory — verify all 32 functions exist in loader.py
# ===========================================================================

class TestLoaderFunctionInventory:

    _EXPECTED = [
        "upload_countries",
        "upload_topics",
        "upload_sounds",
        "upload_hashtags",
        "upload_entities",
        "upload_users",
        "upload_sessions",
        "upload_videos",
        "upload_comments",
        "upload_rel_has_session",
        "upload_rel_last_session",
        "upload_rel_prev_session",
        "upload_rel_created_by",
        "upload_rel_viewed",
        "upload_rel_liked",
        "upload_rel_skipped",
        "upload_rel_reposted",
        "upload_rel_commented",
        "upload_rel_comment_on_video",
        "upload_rel_video_hashtag",
        "upload_rel_video_entity",
        "upload_rel_video_sound",
        "upload_rel_video_topic",
        "upload_rel_entity_topic",
        "upload_rel_user_country",
        "upload_rel_video_country",
        "upload_rel_sound_country",
        "upload_rel_follows",
        "upload_rel_interested_topic",
        "upload_rel_interested_entity",
        "upload_rel_interested_hashtag",
    ]

    def test_all_functions_exist(self):
        import neo4j.loader as loader_mod
        for name in self._EXPECTED:
            assert hasattr(loader_mod, name), f"Missing function: {name}"

    def test_all_functions_are_callable(self):
        import neo4j.loader as loader_mod
        for name in self._EXPECTED:
            fn = getattr(loader_mod, name)
            assert callable(fn), f"{name} is not callable"

    def test_function_count(self):
        import neo4j.loader as loader_mod
        assert len(self._EXPECTED) == 31  # 9 nodes + 22 rels

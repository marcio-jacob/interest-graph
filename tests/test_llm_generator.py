"""
Unit tests for llm/generator.py
=================================
Uses a stub OllamaClient — no live Ollama instance required.
Tests both the LLM-active path (stub returns text) and the fallback path
(stub claims unavailable).
"""

import uuid
from datetime import datetime

import pytest

from generators.base import reset_rng
from generators.taxonomy import generate_topics
from generators.users import generate_users
from generators.videos import generate_videos, assign_video_taxonomy
from generators.taxonomy import (
    generate_countries, generate_hashtags, generate_entities, generate_sounds,
)
from llm.generator import fill_video_descriptions, generate_comments


# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------

class _StubClient:
    """Synchronous stub that always returns a canned string."""

    def __init__(self, available: bool = True, response: str = "Generated text."):
        self._available = available
        self._response = response

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt, **kwargs) -> str:
        return self._response


class _UnavailableClient(_StubClient):
    def __init__(self):
        super().__init__(available=False, response="")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_videos(cfg):
    """20 videos with PLACEHOLDER descriptions and their user list."""
    topics = generate_topics(cfg)
    countries = generate_countries(cfg)
    hashtags = generate_hashtags(cfg)
    entities = generate_entities(cfg)
    sounds = generate_sounds(cfg)
    users = generate_users(30, cfg, cfg, cfg)
    videos = generate_videos(
        20, users, topics, hashtags, entities, sounds,
        countries, cfg, cfg, cfg,
    )
    return videos, users


@pytest.fixture
def comment_stubs(cfg, sample_videos):
    """10 minimal comment stubs pointing at real videos."""
    videos, users = sample_videos
    rng = __import__("numpy").random.default_rng(42)
    stubs = []
    for i in range(10):
        video = videos[int(rng.integers(0, len(videos)))]
        user = users[int(rng.integers(0, len(users)))]
        stubs.append(
            {
                "video_id": video["video_id"],
                "user_id": user["user_id"],
            }
        )
    return stubs, videos


# ===========================================================================
# fill_video_descriptions
# ===========================================================================

class TestFillVideoDescriptions:

    def test_replaces_placeholder_with_llm_text(self, cfg, sample_videos):
        videos, _ = sample_videos
        fill_video_descriptions(videos, cfg, _StubClient(), cfg)
        for v in videos:
            assert v["description"] == "Generated text."

    def test_replaces_placeholder_with_fallback_text(self, cfg, sample_videos):
        videos, _ = sample_videos
        fill_video_descriptions(videos, cfg, _UnavailableClient(), cfg)
        for v in videos:
            assert v["description"] != "PLACEHOLDER"
            assert isinstance(v["description"], str)
            assert len(v["description"]) > 0

    def test_mutates_in_place(self, cfg, sample_videos):
        videos, _ = sample_videos
        original_list_id = id(videos)
        fill_video_descriptions(videos, cfg, _StubClient(), cfg)
        assert id(videos) == original_list_id

    def test_all_placeholders_replaced(self, cfg, sample_videos):
        videos, _ = sample_videos
        fill_video_descriptions(videos, cfg, _StubClient(), cfg)
        remaining = [v for v in videos if v["description"] == "PLACEHOLDER"]
        assert remaining == [], f"{len(remaining)} videos still have PLACEHOLDER"

    def test_empty_videos_is_noop(self, cfg):
        fill_video_descriptions([], cfg, _StubClient(), cfg)  # should not raise

    def test_skips_already_filled_descriptions(self, cfg, sample_videos):
        videos, _ = sample_videos
        videos[0]["description"] = "Already written."
        fill_video_descriptions(videos, cfg, _StubClient(), cfg)
        assert videos[0]["description"] == "Already written.", \
            "Pre-existing description should not be overwritten"

    def test_fallback_text_contains_topic_word(self, cfg, sample_videos):
        """Fallback descriptions should mention the topic area."""
        videos, _ = sample_videos
        fill_video_descriptions(videos, cfg, _UnavailableClient(), cfg)
        topics = generate_topics(cfg)
        tid_to_slug = {t["topic_id"]: t["slug"] for t in topics}
        for v in videos:
            slug = tid_to_slug.get(v["topic_id"], "")
            topic_word = slug.replace("_", " ").split()[0] if slug else ""
            if topic_word:
                assert topic_word in v["description"].lower(), \
                    f"Topic word '{topic_word}' missing from fallback: {v['description']!r}"

    def test_empty_llm_response_triggers_fallback(self, cfg, sample_videos):
        """If LLM returns empty string, fallback text should be used."""
        videos, _ = sample_videos
        client = _StubClient(available=True, response="")
        fill_video_descriptions(videos, cfg, client, cfg)
        for v in videos:
            assert v["description"] != "PLACEHOLDER"
            assert v["description"] != ""


# ===========================================================================
# generate_comments
# ===========================================================================

class TestGenerateComments:

    REQUIRED_KEYS = {
        "comment_id", "video_id", "user_id",
        "comment_text", "comment_sentiment", "created_at",
    }

    def test_returns_list(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        result = generate_comments(stubs, videos, cfg, _StubClient())
        assert isinstance(result, list)

    def test_correct_count(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        result = generate_comments(stubs, videos, cfg, _StubClient())
        assert len(result) == len(stubs)

    def test_required_keys_present(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        for comment in generate_comments(stubs, videos, cfg, _StubClient()):
            assert self.REQUIRED_KEYS == set(comment.keys()), \
                f"Key mismatch: {set(comment.keys()) ^ self.REQUIRED_KEYS}"

    def test_comment_ids_are_unique_uuids(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        comments = generate_comments(stubs, videos, cfg, _StubClient())
        ids = [c["comment_id"] for c in comments]
        assert len(set(ids)) == len(ids), "Duplicate comment_id"
        for cid in ids:
            uuid.UUID(cid)  # raises if invalid

    def test_sentiment_is_valid_value(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        valid = {"positive", "neutral", "negative"}
        for comment in generate_comments(stubs, videos, cfg, _StubClient()):
            assert comment["comment_sentiment"] in valid

    def test_video_id_matches_stub(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        comments = generate_comments(stubs, videos, cfg, _StubClient())
        for stub, comment in zip(stubs, comments):
            assert comment["video_id"] == stub["video_id"]

    def test_user_id_matches_stub(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        comments = generate_comments(stubs, videos, cfg, _StubClient())
        for stub, comment in zip(stubs, comments):
            assert comment["user_id"] == stub["user_id"]

    def test_created_at_is_datetime(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        for comment in generate_comments(stubs, videos, cfg, _StubClient()):
            assert isinstance(comment["created_at"], datetime)

    def test_comment_text_uses_llm_response(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        comments = generate_comments(stubs, videos, cfg, _StubClient(response="Test text."))
        for comment in comments:
            assert comment["comment_text"] == "Test text."

    def test_comment_text_non_empty_in_fallback(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        for comment in generate_comments(stubs, videos, cfg, _UnavailableClient()):
            assert comment["comment_text"], "Empty comment text in fallback"

    def test_stub_sentiment_is_respected(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        for stub in stubs:
            stub["sentiment"] = "positive"
        comments = generate_comments(stubs, videos, cfg, _StubClient())
        for comment in comments:
            assert comment["comment_sentiment"] == "positive"

    def test_stub_created_at_is_preserved(self, cfg, comment_stubs):
        stubs, videos = comment_stubs
        fixed_dt = datetime(2024, 6, 15, 12, 0, 0)
        for stub in stubs:
            stub["created_at"] = fixed_dt
        comments = generate_comments(stubs, videos, cfg, _StubClient())
        for comment in comments:
            assert comment["created_at"] == fixed_dt

    def test_empty_stubs_returns_empty(self, cfg, sample_videos):
        videos, _ = sample_videos
        assert generate_comments([], videos, cfg, _StubClient()) == []

    def test_stub_with_invalid_video_id_skipped(self, cfg, sample_videos):
        videos, users = sample_videos
        stubs = [{"video_id": "nonexistent-uuid", "user_id": users[0]["user_id"]}]
        result = generate_comments(stubs, videos, cfg, _StubClient())
        assert result == [], "Stub with invalid video_id should be skipped"

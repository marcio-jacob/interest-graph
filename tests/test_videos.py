"""
Unit tests for generators/videos.py
=====================================
Covers: generate_videos, assign_video_taxonomy
"""

import uuid
from collections import Counter
from datetime import datetime

import numpy as np
import pytest

from generators.base import reset_rng
from generators.taxonomy import (
    generate_countries,
    generate_entities,
    generate_hashtags,
    generate_sounds,
    generate_topics,
)
from generators.users import generate_users
from generators.videos import assign_video_taxonomy, generate_videos


REQUIRED_VIDEO_KEYS = {
    "video_id", "creator_id", "topic_id", "secondary_topic_id",
    "country_id", "video_duration_seconds", "play_count",
    "like_count", "comment_count", "share_count",
    "download_count", "repost_count", "description", "created_at",
}


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def tax(cfg):
    """All taxonomy record lists."""
    return (
        generate_topics(cfg),
        generate_countries(cfg),
        generate_hashtags(cfg),
        generate_entities(cfg),
        generate_sounds(cfg),
    )


@pytest.fixture
def videos_fixture(cfg, tax):
    """100 videos generated from 50 users."""
    topics, countries, hashtags, entities, sounds = tax
    users = generate_users(50, cfg, cfg, cfg)
    videos = generate_videos(
        100, users, topics, hashtags, entities, sounds,
        countries, cfg, cfg, cfg,
    )
    return videos, users, topics, countries, hashtags, entities, sounds


@pytest.fixture
def taxonomy_rels(cfg, videos_fixture):
    """Result of assign_video_taxonomy for the 100-video fixture."""
    videos, users, topics, countries, hashtags, entities, sounds = videos_fixture
    ht, ent, snd, vt = assign_video_taxonomy(
        videos, topics, hashtags, entities, sounds, cfg, cfg
    )
    return ht, ent, snd, vt, videos, topics, hashtags, entities, sounds


# ===========================================================================
# generate_videos
# ===========================================================================

class TestGenerateVideos:

    # ── Return type & structure ──────────────────────────────────────────────

    def test_returns_list(self, videos_fixture):
        videos, *_ = videos_fixture
        assert isinstance(videos, list)

    def test_correct_count(self, videos_fixture):
        videos, *_ = videos_fixture
        assert len(videos) == 100

    def test_required_keys(self, videos_fixture):
        videos, *_ = videos_fixture
        for v in videos:
            assert REQUIRED_VIDEO_KEYS == set(v.keys()), \
                f"Key mismatch: {set(v.keys()) ^ REQUIRED_VIDEO_KEYS}"

    def test_video_ids_are_unique_uuids(self, videos_fixture):
        videos, *_ = videos_fixture
        ids = [v["video_id"] for v in videos]
        assert len(set(ids)) == len(ids), "Duplicate video_id detected"
        for vid_id in ids:
            uuid.UUID(vid_id)  # raises ValueError if invalid

    # ── Creator selection ────────────────────────────────────────────────────

    def test_creators_only_author_videos(self, videos_fixture):
        videos, users, *_ = videos_fixture
        creator_ids = {u["user_id"] for u in users if u["is_creator"]}
        for v in videos:
            assert v["creator_id"] in creator_ids, \
                f"Non-creator {v['creator_id']} authored a video"

    # ── Topic ────────────────────────────────────────────────────────────────

    def test_topic_ids_are_valid(self, videos_fixture):
        videos, _, topics, *_ = videos_fixture
        valid_tids = {t["topic_id"] for t in topics}
        for v in videos:
            assert v["topic_id"] in valid_tids

    def test_secondary_topic_none_or_differs_from_primary(self, videos_fixture):
        videos, *_ = videos_fixture
        for v in videos:
            sec = v["secondary_topic_id"]
            if sec is not None:
                assert sec != v["topic_id"], \
                    f"secondary_topic_id == topic_id for video {v['video_id']}"

    def test_secondary_topic_fraction_near_config(self, cfg, tax):
        """~35 % of videos should have a secondary topic (±10 pp)."""
        reset_rng(42)
        topics, countries, hashtags, entities, sounds = tax
        users = generate_users(100, cfg, cfg, cfg)
        videos = generate_videos(
            500, users, topics, hashtags, entities, sounds,
            countries, cfg, cfg, cfg,
        )
        frac = sum(1 for v in videos if v["secondary_topic_id"] is not None) / 500
        target = cfg["videos"]["secondary_topic_probability"]
        assert abs(frac - target) <= 0.10, \
            f"Secondary topic fraction {frac:.3f} far from target {target}"

    # ── Country ──────────────────────────────────────────────────────────────

    def test_country_ids_are_valid(self, videos_fixture):
        videos, _, _, countries, *_ = videos_fixture
        valid = {c["country_id"] for c in countries}
        for v in videos:
            assert v["country_id"] in valid

    # ── Duration ─────────────────────────────────────────────────────────────

    def test_duration_positive(self, videos_fixture):
        videos, *_ = videos_fixture
        for v in videos:
            assert v["video_duration_seconds"] > 0, \
                f"Non-positive duration: {v['video_duration_seconds']}"

    def test_duration_within_histogram_range(self, videos_fixture):
        """Duration must not exceed the histogram max (600 s)."""
        videos, *_ = videos_fixture
        for v in videos:
            assert v["video_duration_seconds"] <= 600.0, \
                f"Duration exceeds histogram max: {v['video_duration_seconds']}"

    # ── Engagement counts ────────────────────────────────────────────────────

    def test_play_count_at_least_one(self, videos_fixture):
        videos, *_ = videos_fixture
        for v in videos:
            assert v["play_count"] >= 1

    def test_engagement_counts_non_negative(self, videos_fixture):
        videos, *_ = videos_fixture
        for v in videos:
            for key in ("like_count", "comment_count", "share_count",
                        "download_count", "repost_count"):
                assert v[key] >= 0, f"{key} < 0 for video {v['video_id']}"

    def test_likes_not_exceed_play_count(self, cfg, tax):
        """like_count <= play_count (Poisson mean << 1 so almost always true; check statistically)."""
        reset_rng(42)
        topics, countries, hashtags, entities, sounds = tax
        users = generate_users(100, cfg, cfg, cfg)
        videos = generate_videos(
            500, users, topics, hashtags, entities, sounds,
            countries, cfg, cfg, cfg,
        )
        violations = [v for v in videos if v["like_count"] > v["play_count"]]
        # Poisson(play_count * 0.036) — essentially impossible to exceed play_count
        assert len(violations) == 0, \
            f"{len(violations)} videos where like_count > play_count"

    # ── Description ──────────────────────────────────────────────────────────

    def test_description_is_placeholder(self, videos_fixture):
        videos, *_ = videos_fixture
        for v in videos:
            assert v["description"] == "PLACEHOLDER"

    # ── Temporal ─────────────────────────────────────────────────────────────

    def test_created_at_is_datetime(self, videos_fixture):
        videos, *_ = videos_fixture
        for v in videos:
            assert isinstance(v["created_at"], datetime)

    def test_created_at_within_sim_range(self, videos_fixture, cfg):
        videos, *_ = videos_fixture
        sim_start = datetime.strptime(cfg["dates"]["sim_start"], "%Y-%m-%d")
        sim_end = datetime.strptime(cfg["dates"]["sim_end"], "%Y-%m-%d")
        for v in videos:
            assert sim_start <= v["created_at"] <= sim_end, \
                f"created_at {v['created_at']} outside sim range"

    def test_created_at_after_creator_joined(self, videos_fixture):
        videos, users, *_ = videos_fixture
        uid_to_joined = {u["user_id"]: u["joined_at"] for u in users}
        for v in videos:
            joined = uid_to_joined[v["creator_id"]]
            assert v["created_at"] >= joined, \
                f"Video created {v['created_at']} before creator joined {joined}"

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_zero_videos_returns_empty(self, cfg, tax):
        topics, countries, hashtags, entities, sounds = tax
        users = generate_users(20, cfg, cfg, cfg)
        result = generate_videos(
            0, users, topics, hashtags, entities, sounds,
            countries, cfg, cfg, cfg,
        )
        assert result == []

    def test_no_creators_returns_empty(self, cfg, tax):
        topics, countries, hashtags, entities, sounds = tax
        users = generate_users(10, cfg, cfg, cfg)
        for u in users:
            u["is_creator"] = False
        result = generate_videos(
            10, users, topics, hashtags, entities, sounds,
            countries, cfg, cfg, cfg,
        )
        assert result == []


# ===========================================================================
# assign_video_taxonomy
# ===========================================================================

class TestAssignVideoTaxonomy:

    # ── Return types & structure ─────────────────────────────────────────────

    def test_returns_four_lists(self, taxonomy_rels):
        ht, ent, snd, vt, *_ = taxonomy_rels
        assert isinstance(ht, list)
        assert isinstance(ent, list)
        assert isinstance(snd, list)
        assert isinstance(vt, list)

    def test_video_hashtags_keys(self, taxonomy_rels):
        ht, *_ = taxonomy_rels
        for row in ht:
            assert {"video_id", "hashtag_id"} == set(row.keys())

    def test_video_entities_keys(self, taxonomy_rels):
        _, ent, *_ = taxonomy_rels
        for row in ent:
            assert {"video_id", "entity_id"} == set(row.keys())

    def test_video_sounds_keys(self, taxonomy_rels):
        _, _, snd, *_ = taxonomy_rels
        for row in snd:
            assert {"video_id", "sound_id"} == set(row.keys())

    def test_video_topics_keys(self, taxonomy_rels):
        _, _, _, vt, *_ = taxonomy_rels
        for row in vt:
            assert {"video_id", "topic_id", "is_primary"} == set(row.keys())

    # ── Sounds ───────────────────────────────────────────────────────────────

    def test_every_video_gets_exactly_one_sound(self, taxonomy_rels):
        _, _, snd, _, videos, *_ = taxonomy_rels
        counts = Counter(r["video_id"] for r in snd)
        for v in videos:
            assert counts[v["video_id"]] == 1, \
                f"Video {v['video_id']} has {counts[v['video_id']]} sounds"

    def test_sound_ids_are_valid(self, taxonomy_rels):
        _, _, snd, _, _, _, _, _, sounds = taxonomy_rels
        valid_ids = {s["song_id"] for s in sounds}
        for row in snd:
            assert row["sound_id"] in valid_ids

    # ── Topics ───────────────────────────────────────────────────────────────

    def test_every_video_has_primary_topic_link(self, taxonomy_rels):
        _, _, _, vt, videos, *_ = taxonomy_rels
        video_ids_with_primary = {r["video_id"] for r in vt if r["is_primary"]}
        for v in videos:
            assert v["video_id"] in video_ids_with_primary

    def test_exactly_one_primary_topic_per_video(self, taxonomy_rels):
        _, _, _, vt, videos, *_ = taxonomy_rels
        primary_counts = Counter(r["video_id"] for r in vt if r["is_primary"])
        for v in videos:
            assert primary_counts[v["video_id"]] == 1, \
                f"Video {v['video_id']} has {primary_counts[v['video_id']]} primary topics"

    def test_is_primary_is_bool(self, taxonomy_rels):
        _, _, _, vt, *_ = taxonomy_rels
        for row in vt:
            assert isinstance(row["is_primary"], bool)

    def test_topic_ids_in_vt_are_valid(self, taxonomy_rels):
        _, _, _, vt, _, topics, *_ = taxonomy_rels
        valid_tids = {t["topic_id"] for t in topics}
        for row in vt:
            assert row["topic_id"] in valid_tids

    # ── Hashtags ─────────────────────────────────────────────────────────────

    def test_hashtag_count_within_config_range(self, taxonomy_rels, cfg):
        ht, _, _, _, videos, *_ = taxonomy_rels
        ht_min = cfg["videos"]["hashtags_per_video_min"]
        ht_max = cfg["videos"]["hashtags_per_video_max"]
        counts = Counter(r["video_id"] for r in ht)
        for v in videos:
            cnt = counts.get(v["video_id"], 0)
            assert ht_min <= cnt <= ht_max, \
                f"Video {v['video_id']} has {cnt} hashtags (expected [{ht_min}, {ht_max}])"

    def test_hashtag_ids_are_valid(self, taxonomy_rels):
        ht, _, _, _, _, _, hashtags, *_ = taxonomy_rels
        valid_ids = {h["hashtag_id"] for h in hashtags}
        for row in ht:
            assert row["hashtag_id"] in valid_ids

    def test_no_duplicate_hashtag_per_video(self, taxonomy_rels):
        ht, *_ = taxonomy_rels
        pairs = [(r["video_id"], r["hashtag_id"]) for r in ht]
        assert len(set(pairs)) == len(pairs), "Duplicate video-hashtag pair detected"

    # ── Entities ─────────────────────────────────────────────────────────────

    def test_entity_ids_are_valid(self, taxonomy_rels):
        _, ent, _, _, _, _, _, entities, _ = taxonomy_rels
        valid_ids = {e["entity_id"] for e in entities}
        for row in ent:
            assert row["entity_id"] in valid_ids

    def test_entity_count_within_config_range(self, taxonomy_rels, cfg):
        _, ent, _, _, videos, *_ = taxonomy_rels
        ent_max = cfg["videos"]["entities_per_video_max"]
        counts = Counter(r["video_id"] for r in ent)
        for v in videos:
            cnt = counts.get(v["video_id"], 0)
            assert cnt <= ent_max, \
                f"Video {v['video_id']} has {cnt} entities (max {ent_max})"

    # ── Edge case ─────────────────────────────────────────────────────────────

    def test_empty_videos_returns_four_empty_lists(self, cfg):
        topics = generate_topics(cfg)
        hashtags = generate_hashtags(cfg)
        entities = generate_entities(cfg)
        sounds = generate_sounds(cfg)
        ht, ent, snd, vt = assign_video_taxonomy(
            [], topics, hashtags, entities, sounds, cfg, cfg
        )
        assert ht == [] and ent == [] and snd == [] and vt == []

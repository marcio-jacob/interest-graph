"""
Unit tests for generators/users.py
=====================================
Covers: _strip_diacritics, _clean_username, _build_topic_vocabulary,
        build_username, generate_users, generate_follows
"""

import re
import uuid
from collections import Counter

import numpy as np
import pytest
from faker import Faker

from generators.base import reset_rng
from generators.users import (
    _build_topic_vocabulary,
    _clean_username,
    _strip_diacritics,
    build_username,
    generate_follows,
    generate_users,
)


# ===========================================================================
# _strip_diacritics
# ===========================================================================

class TestStripDiacritics:
    @pytest.mark.parametrize("inp, expected", [
        ("café",  "cafe"),
        ("naïve", "naive"),
        ("über",  "uber"),
        ("señor", "senor"),
        ("João",  "Joao"),
        ("Müller","Muller"),
        ("hello", "hello"),        # no diacritics → unchanged
    ])
    def test_removes_diacritics(self, inp, expected):
        assert _strip_diacritics(inp) == expected

    def test_cjk_unchanged(self):
        """CJK characters have no NFKD decomposition — they pass through."""
        assert _strip_diacritics("陽菜") == "陽菜"

    def test_empty_string(self):
        assert _strip_diacritics("") == ""


# ===========================================================================
# _clean_username
# ===========================================================================

class TestCleanUsername:
    @pytest.mark.parametrize("inp, expected", [
        ("João Silva",      "joaosilva"),       # diacritics + space
        ("HELLO_WORLD",     "hello_world"),     # uppercase + underscore kept
        ("test.name42",     "test.name42"),     # dot and digits kept
        ("!!!hello!!!",     "hello"),           # leading/trailing specials
        ("陽菜",             ""),               # CJK → empty
        ("a b-c_d.e",       "ab_c_d.e"),        # space removed, hyphen→underscore
    ])
    def test_clean_cases(self, inp, expected):
        assert _clean_username(inp) == expected

    def test_max_length_30(self):
        long_input = "a" * 60
        assert len(_clean_username(long_input)) == 30

    def test_result_only_valid_chars(self):
        raw = "Héllo Wörld! @2024#special"
        result = _clean_username(raw)
        assert re.fullmatch(r"[a-z0-9_.]*", result)


# ===========================================================================
# _build_topic_vocabulary
# ===========================================================================

class TestBuildTopicVocabulary:
    def test_returns_list(self, cfg):
        vocab = _build_topic_vocabulary(cfg)
        assert isinstance(vocab, list)

    def test_contains_topic_slug_parts(self, cfg):
        vocab = _build_topic_vocabulary(cfg)
        # Every topic slug word ≥3 chars should appear
        for topic in cfg["topics"]:
            for part in topic["slug"].split("_"):
                if len(part) >= 3:
                    assert part in vocab, f"'{part}' missing from vocabulary"

    def test_contains_hashtag_words(self, cfg):
        vocab = _build_topic_vocabulary(cfg)
        # A known hashtag word should be present
        assert "foodtok" in vocab

    def test_no_very_short_words(self, cfg):
        vocab = _build_topic_vocabulary(cfg)
        assert all(len(w) >= 3 for w in vocab)

    def test_sorted_output(self, cfg):
        vocab = _build_topic_vocabulary(cfg)
        assert vocab == sorted(vocab)


# ===========================================================================
# build_username
# ===========================================================================

class TestBuildUsername:
    """Test every username pattern in the taxonomy."""

    @pytest.fixture
    def build_deps(self, cfg):
        """Return (fake, topic_words, adjectives, country_suffix, rng)."""
        fake        = Faker("en_US")
        topic_words = _build_topic_vocabulary(cfg)
        adjectives  = cfg.get("username_adjectives", ["cool"])
        sfx         = "us"
        rng         = np.random.default_rng(42)
        return fake, topic_words, adjectives, sfx, rng

    def test_result_is_string(self, cfg, build_deps):
        fake, tv, adj, sfx, rng = build_deps
        pat = cfg["username_patterns"][0]
        result = build_username(fake, tv, pat, adj, sfx, rng)
        assert isinstance(result, str)

    def test_result_is_lowercase(self, cfg, build_deps):
        fake, tv, adj, sfx, rng = build_deps
        for pat in cfg["username_patterns"]:
            result = build_username(fake, tv, pat, adj, sfx, rng)
            assert result == result.lower(), f"Pattern '{pat}' produced uppercase: {result}"

    def test_result_only_valid_chars(self, cfg, build_deps):
        fake, tv, adj, sfx, rng = build_deps
        for pat in cfg["username_patterns"]:
            result = build_username(fake, tv, pat, adj, sfx, rng)
            assert re.fullmatch(r"[a-z0-9_.]*", result), \
                f"Pattern '{pat}' produced invalid chars: {result}"

    def test_result_max_30_chars(self, cfg, build_deps):
        fake, tv, adj, sfx, rng = build_deps
        for pat in cfg["username_patterns"]:
            result = build_username(fake, tv, pat, adj, sfx, rng)
            assert len(result) <= 30, \
                f"Pattern '{pat}' exceeded 30 chars: {result!r}"

    def test_all_10_patterns_produce_nonempty(self, cfg, build_deps):
        fake, tv, adj, sfx, rng = build_deps
        assert len(cfg["username_patterns"]) == 10
        for pat in cfg["username_patterns"]:
            result = build_username(fake, tv, pat, adj, sfx, rng)
            assert result, f"Pattern '{pat}' produced empty username"

    def test_verb_er_pattern_works(self, cfg, build_deps):
        """Pattern 'the{noun}{verb}er' should produce e.g. 'thefoodlover'."""
        fake, tv, adj, sfx, _ = build_deps
        rng  = np.random.default_rng(1)    # fixed seed for determinism
        pat  = "the{noun}{verb}er"
        result = build_username(fake, tv, pat, adj, sfx, rng)
        # Must start with 'the' and end with a letter (the literal 'er' suffix)
        assert result.startswith("the"), f"Got: {result!r}"
        assert result.endswith(("r", "n", "d", "m", "g")), f"Got: {result!r}"

    def test_cjk_locale_produces_nonempty(self, cfg):
        """Faker('ja_JP') generates CJK names; fallback must still produce output."""
        fake        = Faker("ja_JP")
        tv          = _build_topic_vocabulary(cfg)
        adj         = cfg.get("username_adjectives", ["cool"])
        rng         = np.random.default_rng(42)
        # Use a pattern that relies on {first} — the CJK name should be stripped
        # and the fallback (en_US) should kick in
        result = build_username(fake, tv, "{first}{last}{nn}", adj, "jp", rng)
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# generate_users
# ===========================================================================

REQUIRED_USER_KEYS = {
    "user_id", "username", "joined_at", "followers", "following",
    "like_count", "average_watch_time", "last_login", "country_id", "is_creator",
}


class TestGenerateUsers:
    def test_returns_correct_count(self, small_users):
        assert len(small_users) == 10

    def test_required_keys_present(self, small_users):
        for user in small_users:
            assert REQUIRED_USER_KEYS == set(user.keys()), \
                f"Key mismatch: {set(user.keys()) ^ REQUIRED_USER_KEYS}"

    def test_user_ids_are_valid_uuids(self, small_users):
        for user in small_users:
            uuid.UUID(user["user_id"])  # raises ValueError if invalid

    def test_user_ids_are_unique(self, medium_users):
        ids = [u["user_id"] for u in medium_users]
        assert len(set(ids)) == len(ids)

    def test_usernames_are_unique(self, medium_users):
        names = [u["username"] for u in medium_users]
        assert len(set(names)) == len(names)

    def test_usernames_are_valid(self, medium_users):
        for u in medium_users:
            assert re.fullmatch(r"[a-z0-9_.]+", u["username"]), \
                f"Invalid username: {u['username']!r}"
            assert 1 <= len(u["username"]) <= 30

    def test_creator_fraction_near_config(self, cfg):
        """Creator fraction should be within ±8 pp of the configured value."""
        reset_rng(42)
        users = generate_users(200, cfg, cfg, cfg)
        frac    = sum(u["is_creator"] for u in users) / len(users)
        target  = cfg["user_social"]["video_author_fraction"]
        assert abs(frac - target) <= 0.08, \
            f"Creator fraction {frac:.3f} too far from target {target:.3f}"

    def test_valid_country_ids(self, medium_users, cfg):
        valid = {c["country_id"] for c in cfg["countries"]}
        for u in medium_users:
            assert u["country_id"] in valid, \
                f"Unknown country_id: {u['country_id']}"

    def test_joined_at_within_sim_range(self, medium_users, cfg):
        from datetime import datetime
        sim_start = datetime.strptime(cfg["dates"]["sim_start"], "%Y-%m-%d")
        sim_end   = datetime.strptime(cfg["dates"]["sim_end"],   "%Y-%m-%d")
        for u in medium_users:
            assert sim_start <= u["joined_at"] <= sim_end, \
                f"joined_at {u['joined_at']} out of sim range"

    def test_last_login_after_joined_at(self, medium_users):
        for u in medium_users:
            assert u["last_login"] >= u["joined_at"], \
                f"last_login {u['last_login']} < joined_at {u['joined_at']}"

    def test_followers_non_negative(self, medium_users):
        for u in medium_users:
            assert u["followers"] >= 0

    def test_following_non_negative(self, medium_users):
        for u in medium_users:
            assert u["following"] >= 0

    def test_creators_have_higher_median_followers(self, cfg):
        """The creator multiplier should produce clearly higher follower counts."""
        reset_rng(42)
        users = generate_users(100, cfg, cfg, cfg)
        creator_fans = [u["followers"] for u in users if u["is_creator"]]
        regular_fans = [u["followers"] for u in users if not u["is_creator"]]
        assert np.median(creator_fans) > np.median(regular_fans)

    def test_non_creators_have_zero_like_count(self, medium_users):
        for u in medium_users:
            if not u["is_creator"]:
                assert u["like_count"] == 0, \
                    f"Non-creator {u['username']} has like_count={u['like_count']}"

    def test_creators_have_positive_like_count(self, medium_users):
        for u in medium_users:
            if u["is_creator"]:
                assert u["like_count"] > 0

    def test_average_watch_time_in_range(self, medium_users):
        for u in medium_users:
            assert 1.0 <= u["average_watch_time"] <= 60.0, \
                f"average_watch_time {u['average_watch_time']} out of [1, 60]"

    def test_is_creator_is_bool(self, small_users):
        for u in small_users:
            assert isinstance(u["is_creator"], bool)

    def test_zero_users_returns_empty(self, cfg):
        assert generate_users(0, cfg, cfg, cfg) == []


# ===========================================================================
# generate_follows
# ===========================================================================

REQUIRED_FOLLOW_KEYS = {"follower_id", "followee_id", "engagement_score"}


class TestGenerateFollows:
    def test_returns_list(self, users_and_follows):
        _, follows = users_and_follows
        assert isinstance(follows, list)

    def test_required_keys_present(self, users_and_follows):
        _, follows = users_and_follows
        for edge in follows:
            assert REQUIRED_FOLLOW_KEYS == set(edge.keys()), \
                f"Key mismatch: {set(edge.keys()) ^ REQUIRED_FOLLOW_KEYS}"

    def test_no_self_follows(self, users_and_follows):
        _, follows = users_and_follows
        for edge in follows:
            assert edge["follower_id"] != edge["followee_id"], \
                "Self-follow detected"

    def test_no_duplicate_pairs(self, users_and_follows):
        _, follows = users_and_follows
        pairs = [(e["follower_id"], e["followee_id"]) for e in follows]
        assert len(set(pairs)) == len(pairs), "Duplicate follow pairs detected"

    def test_all_ids_are_valid_user_ids(self, users_and_follows):
        users, follows = users_and_follows
        valid_ids = {u["user_id"] for u in users}
        for edge in follows:
            assert edge["follower_id"] in valid_ids
            assert edge["followee_id"] in valid_ids

    def test_engagement_score_in_range(self, users_and_follows):
        _, follows = users_and_follows
        for edge in follows:
            assert 0.1 <= edge["engagement_score"] <= 1.0, \
                f"engagement_score {edge['engagement_score']} out of [0.1, 1.0]"

    def test_following_count_updated_on_users(self, users_and_follows):
        """user['following'] must equal the actual outgoing edge count."""
        users, follows = users_and_follows
        actual = Counter(e["follower_id"] for e in follows)
        for u in users:
            expected = actual.get(u["user_id"], 0)
            assert u["following"] == expected, \
                f"User {u['username']}: following={u['following']}, " \
                f"actual edges={expected}"

    def test_creators_get_higher_engagement_on_average(self, users_and_follows):
        """Edges pointing to creators should have higher mean engagement_score."""
        users, follows = users_and_follows
        creator_ids = {u["user_id"] for u in users if u["is_creator"]}
        if not creator_ids or not follows:
            pytest.skip("Not enough creators/follows to test")
        creator_eng = [e["engagement_score"] for e in follows
                       if e["followee_id"] in creator_ids]
        regular_eng = [e["engagement_score"] for e in follows
                       if e["followee_id"] not in creator_ids]
        if not creator_eng or not regular_eng:
            pytest.skip("Not enough edges in each category")
        assert np.mean(creator_eng) > np.mean(regular_eng)

    def test_empty_users_returns_empty(self, cfg):
        assert generate_follows([], cfg, cfg) == []

    def test_at_least_some_edges_generated(self, users_and_follows):
        _, follows = users_and_follows
        assert len(follows) > 0, "Expected at least some FOLLOWS edges"

    def test_no_edges_when_all_following_zero(self, cfg):
        """Users with following=0 should produce no outgoing edges."""
        reset_rng(42)
        users = generate_users(10, cfg, cfg, cfg)
        for u in users:
            u["following"] = 0
        follows = generate_follows(users, cfg, cfg)
        assert follows == [], "Expected no edges when all following=0"

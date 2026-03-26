"""
Unit tests for generators/base.py
==================================
Covers: load_config, get_rng/reset_rng, clamp, date_between,
        weighted_choice, weighted_choices,
        sample_lognormal, sample_from_histogram
"""

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from generators.base import (
    clamp,
    date_between,
    get_rng,
    load_config,
    reset_config_cache,
    reset_rng,
    sample_from_histogram,
    sample_lognormal,
    weighted_choice,
    weighted_choices,
)


# ===========================================================================
# load_config
# ===========================================================================

class TestLoadConfig:
    def test_returns_dict(self, cfg):
        assert isinstance(cfg, dict)

    @pytest.mark.parametrize("key", [
        "seed", "scale", "dates", "sessions", "interactions",
        "interest", "social", "topics", "countries", "sounds",
        "hashtags", "entities", "watch_behavior", "user_social",
        "video_engagement", "social_graph",
    ])
    def test_has_required_key(self, cfg, key):
        assert key in cfg, f"Missing top-level key: '{key}'"

    def test_topics_length(self, cfg):
        assert len(cfg["topics"]) == 12

    def test_countries_length(self, cfg):
        assert len(cfg["countries"]) == 12

    def test_sounds_length(self, cfg):
        assert len(cfg["sounds"]) == 30

    def test_hashtags_covers_all_topics(self, cfg):
        topic_slugs = {t["slug"] for t in cfg["topics"]}
        assert topic_slugs == set(cfg["hashtags"].keys())

    def test_each_topic_has_10_hashtags(self, cfg):
        for slug, tags in cfg["hashtags"].items():
            assert len(tags) == 10, f"Topic '{slug}' has {len(tags)} hashtags"

    def test_entities_has_60_entries(self, cfg):
        assert len(cfg["entities"]) == 60

    def test_cached_on_repeat_call(self, cfg):
        second = load_config()
        assert cfg is second  # same object in memory

    def test_reset_cache_forces_reload(self, cfg):
        first_id = id(cfg)
        reset_config_cache()
        reloaded = load_config()
        assert id(reloaded) != first_id

    def test_missing_file_raises_file_not_found(self, tmp_path):
        reset_config_cache()
        with pytest.raises(FileNotFoundError):
            load_config(
                params_path=tmp_path / "missing.yaml",
                taxonomy_path=tmp_path / "missing2.yaml",
                distributions_path=tmp_path / "missing3.yaml",
            )


# ===========================================================================
# RNG
# ===========================================================================

class TestRNG:
    def test_get_rng_returns_numpy_generator(self):
        rng = get_rng()
        assert isinstance(rng, np.random.Generator)

    def test_same_seed_produces_same_sequence(self):
        reset_rng(42)
        seq1 = [int(get_rng().integers(0, 1000)) for _ in range(20)]
        reset_rng(42)
        seq2 = [int(get_rng().integers(0, 1000)) for _ in range(20)]
        assert seq1 == seq2

    def test_different_seeds_produce_different_sequences(self):
        reset_rng(42)
        seq1 = [int(get_rng().integers(0, 1000)) for _ in range(10)]
        reset_rng(99)
        seq2 = [int(get_rng().integers(0, 1000)) for _ in range(10)]
        assert seq1 != seq2

    def test_get_rng_returns_singleton(self):
        rng1 = get_rng()
        rng2 = get_rng()
        assert rng1 is rng2

    def test_reset_rng_returns_new_object(self):
        rng_before = get_rng()
        reset_rng(7)
        rng_after = get_rng()
        # After reset, the global singleton is replaced
        assert rng_before is not rng_after


# ===========================================================================
# clamp
# ===========================================================================

class TestClamp:
    @pytest.mark.parametrize("val, lo, hi, expected", [
        (5.0,  0.0, 3.0, 3.0),   # above hi → hi
        (-1.0, 0.0, 3.0, 0.0),   # below lo → lo
        (1.5,  0.0, 3.0, 1.5),   # in range → unchanged
        (3.0,  0.0, 3.0, 3.0),   # at hi boundary
        (0.0,  0.0, 3.0, 0.0),   # at lo boundary
        (0.0,  0.0, 0.0, 0.0),   # lo == hi → lo
        (-99,  -10, 10,  -10),   # large negative
    ])
    def test_clamp_cases(self, val, lo, hi, expected):
        assert clamp(val, lo, hi) == expected


# ===========================================================================
# date_between
# ===========================================================================

class TestDateBetween:
    def test_returns_datetime(self):
        d = date_between("2023-01-01", "2024-12-31")
        assert isinstance(d, datetime)

    def test_result_within_range(self):
        start = datetime(2023, 1, 1)
        end   = datetime(2024, 12, 31)
        # Run several draws — all must land in range (seed is fixed)
        for _ in range(50):
            d = date_between("2023-01-01", "2024-12-31")
            assert start <= d <= end

    def test_single_day_range(self):
        d = date_between("2024-06-15", "2024-06-15")
        assert d.date() == datetime(2024, 6, 15).date()


# ===========================================================================
# weighted_choice
# ===========================================================================

class TestWeightedChoice:
    def test_returns_item_from_list(self):
        items = ["a", "b", "c"]
        result = weighted_choice(items, [1, 1, 1])
        assert result in items

    def test_uniform_weights_all_items_appear(self):
        """With enough draws, every item should be chosen at least once."""
        reset_rng(42)
        items = ["x", "y", "z"]
        seen = {weighted_choice(items, [1, 1, 1]) for _ in range(300)}
        assert seen == {"x", "y", "z"}

    def test_skewed_weights_favours_heavy_item(self):
        """Item with weight 10× others should appear most often."""
        reset_rng(42)
        items = ["rare", "common", "rare2"]
        counts = Counter(weighted_choice(items, [1, 10, 1]) for _ in range(2000))
        assert counts["common"] > counts["rare"]
        assert counts["common"] > counts["rare2"]
        # Roughly 83 % of draws should be "common"
        assert 0.75 <= counts["common"] / 2000 <= 0.92

    def test_single_item_always_returned(self):
        for _ in range(20):
            assert weighted_choice(["only"], [1]) == "only"

    def test_weights_need_not_sum_to_one(self):
        """Weights are normalised internally; raw values like [100, 200] are fine."""
        reset_rng(42)
        items = ["a", "b"]
        counts = Counter(weighted_choice(items, [100, 200]) for _ in range(900))
        # Expect ~33 % 'a', ~67 % 'b'
        assert 0.28 <= counts["a"] / 900 <= 0.40


# ===========================================================================
# weighted_choices
# ===========================================================================

class TestWeightedChoices:
    def test_returns_correct_length(self):
        result = weighted_choices(["a", "b", "c"], [1, 1, 1], k=7)
        assert len(result) == 7

    def test_all_results_from_items(self):
        items = ["x", "y"]
        result = weighted_choices(items, [1, 1], k=50)
        assert all(r in items for r in result)

    def test_zero_k_returns_empty(self):
        assert weighted_choices(["a", "b"], [1, 1], k=0) == []

    def test_heavily_weighted_item_dominates(self):
        reset_rng(42)
        counts = Counter(weighted_choices(["rare", "common"], [1, 99], k=1000))
        assert counts["common"] > 900


# ===========================================================================
# sample_lognormal
# ===========================================================================

class TestSampleLognormal:
    def test_returns_float(self):
        assert isinstance(sample_lognormal(3.0, 1.0), float)

    def test_respects_clip_lo(self):
        for _ in range(100):
            val = sample_lognormal(0.0, 5.0, clip_lo=10.0)
            assert val >= 10.0

    def test_respects_clip_hi(self):
        for _ in range(100):
            val = sample_lognormal(10.0, 2.0, clip_hi=50.0)
            assert val <= 50.0

    def test_default_clip_lo_is_zero(self):
        # With mu=0, sigma=5, some draws would be < 0 without clipping
        for _ in range(200):
            val = sample_lognormal(0.0, 5.0)
            assert val >= 0.0

    def test_median_close_to_exp_mu(self):
        """Median of lognormal ≈ exp(mu). With N=2000 draws, check within 25 %."""
        reset_rng(42)
        mu = 4.0
        samples = [sample_lognormal(mu, 0.3) for _ in range(2000)]
        median = float(np.median(samples))
        assert np.exp(mu) * 0.75 <= median <= np.exp(mu) * 1.25


# ===========================================================================
# sample_from_histogram
# ===========================================================================

class TestSampleFromHistogram:
    def test_returns_float(self, cfg):
        edges  = cfg["watch_behavior"]["watch_ratio"]["histogram"]["bin_edges"]
        counts = cfg["watch_behavior"]["watch_ratio"]["histogram"]["counts"]
        assert isinstance(sample_from_histogram(edges, counts), float)

    def test_result_within_histogram_range(self, cfg):
        edges  = cfg["watch_behavior"]["watch_ratio"]["histogram"]["bin_edges"]
        counts = cfg["watch_behavior"]["watch_ratio"]["histogram"]["counts"]
        lo, hi = edges[0], edges[-1]
        for _ in range(200):
            val = sample_from_histogram(edges, counts)
            assert lo <= val <= hi

    def test_reproduces_kuairec_skip_rate(self, cfg):
        """
        Samples from the KuaiRec watch_ratio histogram should reproduce the
        observed skip rate of ~13.1 % (watch_ratio < 0.15).
        """
        reset_rng(42)
        edges  = cfg["watch_behavior"]["watch_ratio"]["histogram"]["bin_edges"]
        counts = cfg["watch_behavior"]["watch_ratio"]["histogram"]["counts"]
        N      = 10_000
        samples = [sample_from_histogram(edges, counts) for _ in range(N)]
        skip_rate = sum(s < 0.15 for s in samples) / N
        # Real rate: 0.131 — allow ±3 percentage points
        assert 0.10 <= skip_rate <= 0.16

    def test_single_bin_always_in_that_bin(self):
        edges  = [0.0, 1.0, 2.0]
        counts = [0, 100]           # only second bin has weight
        for _ in range(30):
            val = sample_from_histogram(edges, counts)
            assert 1.0 <= val <= 2.0

"""
Unit tests for llm/prompts.py
================================
Verifies the structure and content of the prompt matrices.
No LLM calls — pure dict/string checks.
"""

import pytest

from llm.prompts import (
    ANGLE_TONE_PAIRS,
    COMMENT_PROMPTS,
    DESCRIPTION_PROMPTS,
    _ANGLE_TONE_TEMPLATE,
    _COMMENT_SUBJECT,
    _COMMENT_SENTIMENT_TEMPLATE,
    _TOPIC_CONTEXT,
)

_EXPECTED_TOPICS = {
    "cooking_food", "gaming_esports", "fashion_beauty", "fitness_wellness",
    "travel_adventure", "music_dance", "comedy_entertainment",
    "technology_science", "sports_athletics", "education_tutorials",
    "art_creativity", "lifestyle_vlog",
}
_EXPECTED_SENTIMENTS = {"positive", "neutral", "negative"}


# ===========================================================================
# DESCRIPTION_PROMPTS
# ===========================================================================

class TestDescriptionPrompts:
    def test_has_84_entries(self):
        """12 topics × 7 angle/tone pairs = 84."""
        assert len(DESCRIPTION_PROMPTS) == 84

    def test_keys_are_3_tuples(self):
        for key in DESCRIPTION_PROMPTS:
            assert isinstance(key, tuple) and len(key) == 3, \
                f"Key is not a 3-tuple: {key!r}"

    def test_all_topic_slugs_present(self):
        topics_in_keys = {k[0] for k in DESCRIPTION_PROMPTS}
        assert topics_in_keys == _EXPECTED_TOPICS

    def test_all_7_angle_tone_pairs_present_per_topic(self):
        for topic in _EXPECTED_TOPICS:
            pairs = {(k[1], k[2]) for k in DESCRIPTION_PROMPTS if k[0] == topic}
            assert len(pairs) == 7, \
                f"Topic '{topic}' has {len(pairs)} angle/tone pairs (expected 7)"

    def test_values_are_nonempty_strings(self):
        for key, val in DESCRIPTION_PROMPTS.items():
            assert isinstance(val, str) and val, \
                f"Empty or non-string value for key {key!r}"

    def test_no_hashtag_instruction_in_prompts(self):
        """Prompts instruct model NOT to produce hashtags."""
        for key, val in DESCRIPTION_PROMPTS.items():
            assert "no hashtag" in val.lower(), \
                f"Prompt for {key!r} does not mention 'no hashtags'"

    def test_prompt_under_300_words(self):
        """Each prompt should be short enough to fit within token budget."""
        for key, val in DESCRIPTION_PROMPTS.items():
            word_count = len(val.split())
            assert word_count < 300, \
                f"Prompt for {key!r} is {word_count} words (limit 300)"

    def test_angle_tone_pairs_match_constant(self):
        pairs_in_keys = set()
        for k in DESCRIPTION_PROMPTS:
            pairs_in_keys.add((k[1], k[2]))
        assert pairs_in_keys == set(ANGLE_TONE_PAIRS)

    def test_known_key_produces_topic_context(self):
        """Spot-check: cooking_food tutorial prompt should mention cooking context."""
        prompt = DESCRIPTION_PROMPTS[("cooking_food", "tutorial", "casual_genz")]
        assert "ingredient" in prompt.lower() or "dish" in prompt.lower() or "viral" in prompt.lower()

    def test_known_key_produces_tone_instruction(self):
        """Gen-Z tone prompt should contain Gen-Z vocabulary cues."""
        prompt = DESCRIPTION_PROMPTS[("gaming_esports", "tutorial", "casual_genz")]
        genz_cues = {"no cap", "lowkey", "hits different", "real talk"}
        assert any(cue in prompt.lower() for cue in genz_cues), \
            f"No Gen-Z cues found in prompt: {prompt!r}"


# ===========================================================================
# COMMENT_PROMPTS
# ===========================================================================

class TestCommentPrompts:
    def test_has_36_entries(self):
        """12 topics × 3 sentiments = 36."""
        assert len(COMMENT_PROMPTS) == 36

    def test_keys_are_2_tuples(self):
        for key in COMMENT_PROMPTS:
            assert isinstance(key, tuple) and len(key) == 2, \
                f"Key is not a 2-tuple: {key!r}"

    def test_all_topic_slugs_present(self):
        topics_in_keys = {k[0] for k in COMMENT_PROMPTS}
        assert topics_in_keys == _EXPECTED_TOPICS

    def test_all_3_sentiments_present_per_topic(self):
        for topic in _EXPECTED_TOPICS:
            sentiments = {k[1] for k in COMMENT_PROMPTS if k[0] == topic}
            assert sentiments == _EXPECTED_SENTIMENTS, \
                f"Topic '{topic}' missing sentiments: {_EXPECTED_SENTIMENTS - sentiments}"

    def test_values_are_nonempty_strings(self):
        for key, val in COMMENT_PROMPTS.items():
            assert isinstance(val, str) and val, \
                f"Empty or non-string value for key {key!r}"

    def test_positive_prompt_asks_for_enthusiasm(self):
        for topic in _EXPECTED_TOPICS:
            val = COMMENT_PROMPTS[(topic, "positive")]
            assert "enthusiastic" in val.lower() or "loved" in val.lower(), \
                f"Positive prompt for '{topic}' lacks enthusiasm cue"

    def test_negative_prompt_excludes_hate_speech(self):
        """Negative prompts must contain an explicit no-hate-speech constraint."""
        for topic in _EXPECTED_TOPICS:
            val = COMMENT_PROMPTS[(topic, "negative")]
            assert "hate" in val.lower() or "respectful" in val.lower(), \
                f"Negative prompt for '{topic}' lacks safety constraint"

    def test_single_sentence_instruction(self):
        """All comment prompts should instruct the model to produce one sentence."""
        for key, val in COMMENT_PROMPTS.items():
            assert "single sentence" in val.lower() or "one sentence" in val.lower(), \
                f"Comment prompt for {key!r} does not specify single sentence"


# ===========================================================================
# ANGLE_TONE_PAIRS
# ===========================================================================

class TestAngleTonePairs:
    def test_has_7_pairs(self):
        assert len(ANGLE_TONE_PAIRS) == 7

    def test_all_pairs_are_2_tuples_of_strings(self):
        for pair in ANGLE_TONE_PAIRS:
            assert isinstance(pair, tuple) and len(pair) == 2
            assert all(isinstance(p, str) for p in pair)

    def test_no_duplicate_pairs(self):
        assert len(set(ANGLE_TONE_PAIRS)) == len(ANGLE_TONE_PAIRS)

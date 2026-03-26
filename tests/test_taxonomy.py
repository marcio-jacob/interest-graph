"""
Unit tests for generators/taxonomy.py
======================================
Covers: generate_topics, generate_countries, generate_hashtags,
        generate_entities, generate_sounds, entity_topic_links
"""

from collections import Counter

import pytest

from generators.taxonomy import (
    entity_topic_links,
    generate_countries,
    generate_entities,
    generate_hashtags,
    generate_sounds,
    generate_topics,
)


# ===========================================================================
# generate_topics
# ===========================================================================

class TestGenerateTopics:
    def test_returns_list(self, cfg):
        assert isinstance(generate_topics(cfg), list)

    def test_count_is_12(self, cfg):
        assert len(generate_topics(cfg)) == 12

    def test_required_keys(self, cfg):
        for t in generate_topics(cfg):
            assert {"topic_id", "name", "slug"} == set(t.keys()), \
                f"Key mismatch: {set(t.keys())}"

    def test_ids_start_with_T(self, cfg):
        for t in generate_topics(cfg):
            assert t["topic_id"].startswith("T"), f"Bad topic_id: {t['topic_id']}"

    def test_unique_topic_ids(self, cfg):
        ids = [t["topic_id"] for t in generate_topics(cfg)]
        assert len(set(ids)) == len(ids)

    def test_slugs_use_underscores_not_spaces(self, cfg):
        for t in generate_topics(cfg):
            assert " " not in t["slug"], f"Slug has spaces: {t['slug']}"

    def test_names_are_nonempty(self, cfg):
        for t in generate_topics(cfg):
            assert t["name"], f"Empty name for {t['topic_id']}"


# ===========================================================================
# generate_countries
# ===========================================================================

class TestGenerateCountries:
    def test_returns_list(self, cfg):
        assert isinstance(generate_countries(cfg), list)

    def test_count_is_12(self, cfg):
        assert len(generate_countries(cfg)) == 12

    def test_required_keys(self, cfg):
        for c in generate_countries(cfg):
            assert {"country_id", "name", "iso"} == set(c.keys()), \
                f"Key mismatch: {set(c.keys())}"

    def test_iso_is_2_chars(self, cfg):
        for c in generate_countries(cfg):
            assert len(c["iso"]) == 2, f"ISO not 2 chars: {c['iso']}"

    def test_country_id_equals_iso(self, cfg):
        for c in generate_countries(cfg):
            assert c["country_id"] == c["iso"]

    def test_no_duplicate_country_ids(self, cfg):
        ids = [c["country_id"] for c in generate_countries(cfg)]
        assert len(set(ids)) == len(ids)


# ===========================================================================
# generate_hashtags
# ===========================================================================

class TestGenerateHashtags:
    def test_returns_list(self, cfg):
        assert isinstance(generate_hashtags(cfg), list)

    def test_count_is_120(self, cfg):
        assert len(generate_hashtags(cfg)) == 120

    def test_required_keys(self, cfg):
        for ht in generate_hashtags(cfg):
            assert {"hashtag_id", "name", "topic_slug"} == set(ht.keys()), \
                f"Key mismatch: {set(ht.keys())}"

    def test_ids_are_sequential_ht_format(self, cfg):
        hashtags = generate_hashtags(cfg)
        for i, ht in enumerate(hashtags, start=1):
            assert ht["hashtag_id"] == f"HT{i:03d}", \
                f"Expected HT{i:03d}, got {ht['hashtag_id']}"

    def test_names_start_with_hash(self, cfg):
        for ht in generate_hashtags(cfg):
            assert ht["name"].startswith("#"), f"No # prefix: {ht['name']}"

    def test_unique_hashtag_ids(self, cfg):
        ids = [ht["hashtag_id"] for ht in generate_hashtags(cfg)]
        assert len(set(ids)) == len(ids)

    def test_topic_slugs_are_valid(self, cfg):
        valid_slugs = {t["slug"] for t in generate_topics(cfg)}
        for ht in generate_hashtags(cfg):
            assert ht["topic_slug"] in valid_slugs, \
                f"Unknown topic_slug: {ht['topic_slug']}"

    def test_exactly_10_per_topic(self, cfg):
        counts = Counter(ht["topic_slug"] for ht in generate_hashtags(cfg))
        for slug, cnt in counts.items():
            assert cnt == 10, f"Topic '{slug}' has {cnt} hashtags (expected 10)"


# ===========================================================================
# generate_entities
# ===========================================================================

class TestGenerateEntities:
    def test_returns_list(self, cfg):
        assert isinstance(generate_entities(cfg), list)

    def test_count_is_60(self, cfg):
        assert len(generate_entities(cfg)) == 60

    def test_required_keys(self, cfg):
        for e in generate_entities(cfg):
            assert {
                "entity_id", "name", "aliases", "primary_topic", "secondary_topics"
            } == set(e.keys()), f"Key mismatch: {set(e.keys())}"

    def test_ids_start_with_E(self, cfg):
        for e in generate_entities(cfg):
            assert e["entity_id"].startswith("E"), f"Bad entity_id: {e['entity_id']}"

    def test_unique_entity_ids(self, cfg):
        ids = [e["entity_id"] for e in generate_entities(cfg)]
        assert len(set(ids)) == len(ids)

    def test_aliases_is_list(self, cfg):
        for e in generate_entities(cfg):
            assert isinstance(e["aliases"], list)

    def test_secondary_topics_is_list(self, cfg):
        for e in generate_entities(cfg):
            assert isinstance(e["secondary_topics"], list)

    def test_primary_topic_is_valid_slug(self, cfg):
        valid_slugs = {t["slug"] for t in generate_topics(cfg)}
        for e in generate_entities(cfg):
            assert e["primary_topic"] in valid_slugs, \
                f"Invalid primary_topic: {e['primary_topic']}"

    def test_5_entities_per_topic(self, cfg):
        counts = Counter(e["primary_topic"] for e in generate_entities(cfg))
        for slug, cnt in counts.items():
            assert cnt == 5, f"Topic '{slug}' has {cnt} entities (expected 5)"


# ===========================================================================
# generate_sounds
# ===========================================================================

class TestGenerateSounds:
    def test_returns_list(self, cfg):
        assert isinstance(generate_sounds(cfg), list)

    def test_count_is_30(self, cfg):
        assert len(generate_sounds(cfg)) == 30

    def test_required_keys(self, cfg):
        for s in generate_sounds(cfg):
            assert {"song_id", "song_name", "singer", "genre", "country_id"} \
                == set(s.keys()), f"Key mismatch: {set(s.keys())}"

    def test_ids_start_with_S(self, cfg):
        for s in generate_sounds(cfg):
            assert s["song_id"].startswith("S"), f"Bad song_id: {s['song_id']}"

    def test_unique_song_ids(self, cfg):
        ids = [s["song_id"] for s in generate_sounds(cfg)]
        assert len(set(ids)) == len(ids)

    def test_country_ids_are_valid(self, cfg):
        valid = {c["country_id"] for c in generate_countries(cfg)}
        for s in generate_sounds(cfg):
            assert s["country_id"] in valid, \
                f"Unknown country_id in sound {s['song_id']}: {s['country_id']}"


# ===========================================================================
# entity_topic_links
# ===========================================================================

class TestEntityTopicLinks:
    def test_returns_list(self, cfg):
        topics = generate_topics(cfg)
        entities = generate_entities(cfg)
        assert isinstance(entity_topic_links(entities, topics), list)

    def test_required_keys(self, cfg):
        topics = generate_topics(cfg)
        entities = generate_entities(cfg)
        for link in entity_topic_links(entities, topics):
            assert {"entity_id", "topic_id", "is_primary"} == set(link.keys()), \
                f"Key mismatch: {set(link.keys())}"

    def test_every_entity_has_at_least_one_link(self, cfg):
        topics = generate_topics(cfg)
        entities = generate_entities(cfg)
        links = entity_topic_links(entities, topics)
        entity_ids_in_links = {l["entity_id"] for l in links}
        for e in entities:
            assert e["entity_id"] in entity_ids_in_links, \
                f"Entity {e['entity_id']} has no links"

    def test_exactly_one_primary_per_entity(self, cfg):
        topics = generate_topics(cfg)
        entities = generate_entities(cfg)
        links = entity_topic_links(entities, topics)
        primary_counts = Counter(l["entity_id"] for l in links if l["is_primary"])
        for e in entities:
            assert primary_counts[e["entity_id"]] == 1, \
                f"Entity {e['entity_id']} has {primary_counts[e['entity_id']]} primary links"

    def test_all_topic_ids_valid(self, cfg):
        topics = generate_topics(cfg)
        entities = generate_entities(cfg)
        valid_tids = {t["topic_id"] for t in topics}
        for link in entity_topic_links(entities, topics):
            assert link["topic_id"] in valid_tids, \
                f"Unknown topic_id in link: {link['topic_id']}"

    def test_is_primary_is_bool(self, cfg):
        topics = generate_topics(cfg)
        entities = generate_entities(cfg)
        for link in entity_topic_links(entities, topics):
            assert isinstance(link["is_primary"], bool)

    def test_no_duplicate_entity_topic_pairs(self, cfg):
        topics = generate_topics(cfg)
        entities = generate_entities(cfg)
        links = entity_topic_links(entities, topics)
        pairs = [(l["entity_id"], l["topic_id"]) for l in links]
        assert len(set(pairs)) == len(pairs), "Duplicate entity-topic links detected"

    def test_secondary_links_have_is_primary_false(self, cfg):
        topics = generate_topics(cfg)
        entities = generate_entities(cfg)
        links = entity_topic_links(entities, topics)
        # All non-primary links must have is_primary=False
        for l in links:
            if not l["is_primary"]:
                assert l["is_primary"] is False

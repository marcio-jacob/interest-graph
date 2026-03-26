"""
generators/taxonomy.py
======================
Thin extractors that transform raw taxonomy YAML dicts into flat record lists
suitable for Neo4j loading.  No RNG needed — these are deterministic lookups.
"""

from __future__ import annotations


def generate_topics(taxonomy: dict) -> list[dict]:
    """Return topic records: topic_id, name, slug."""
    return [
        {"topic_id": t["topic_id"], "name": t["name"], "slug": t["slug"]}
        for t in taxonomy["topics"]
    ]


def generate_countries(taxonomy: dict) -> list[dict]:
    """Return country records: country_id, name, iso."""
    return [
        {"country_id": c["country_id"], "name": c["name"], "iso": c["iso"]}
        for c in taxonomy["countries"]
    ]


def generate_hashtags(taxonomy: dict) -> list[dict]:
    """
    Return a flat list of hashtag records with sequential IDs (HT001…HT120).
    Each record: hashtag_id, name (with # prefix), topic_slug.
    """
    records: list[dict] = []
    counter = 1
    for topic_slug, tags in taxonomy["hashtags"].items():
        for name in tags:
            records.append(
                {
                    "hashtag_id": f"HT{counter:03d}",
                    "name": name,
                    "topic_slug": topic_slug,
                }
            )
            counter += 1
    return records


def generate_entities(taxonomy: dict) -> list[dict]:
    """
    Return entity records preserving all taxonomy fields:
    entity_id, name, aliases, primary_topic, secondary_topics.
    """
    return [
        {
            "entity_id": e["entity_id"],
            "name": e["name"],
            "aliases": list(e.get("aliases", [])),
            "primary_topic": e["primary_topic"],
            "secondary_topics": list(e.get("secondary_topics", [])),
        }
        for e in taxonomy["entities"]
    ]


def generate_sounds(taxonomy: dict) -> list[dict]:
    """Return sound records straight from taxonomy.yaml."""
    return [
        {
            "song_id": s["song_id"],
            "song_name": s["song_name"],
            "singer": s["singer"],
            "genre": s["genre"],
            "country_id": s["country_id"],
        }
        for s in taxonomy["sounds"]
    ]


def entity_topic_links(entities: list[dict], topics: list[dict]) -> list[dict]:
    """
    Build (entity_id, topic_id) link records for primary AND secondary topics.

    Returns list of dicts: {entity_id, topic_id, is_primary}.
    """
    slug_to_id = {t["slug"]: t["topic_id"] for t in topics}
    links: list[dict] = []
    for e in entities:
        primary_tid = slug_to_id.get(e["primary_topic"])
        if primary_tid:
            links.append(
                {"entity_id": e["entity_id"], "topic_id": primary_tid, "is_primary": True}
            )
        for slug in e.get("secondary_topics", []):
            tid = slug_to_id.get(slug)
            if tid:
                links.append(
                    {"entity_id": e["entity_id"], "topic_id": tid, "is_primary": False}
                )
    return links

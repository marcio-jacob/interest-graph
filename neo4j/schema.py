"""
neo4j/schema.py
===============
DDL helpers — constraints, indexes, and label utilities.

apply_schema(driver)  — idempotent; safe to call on every run.
"""

from __future__ import annotations

from neo4j import Driver

# ---------------------------------------------------------------------------
# Constraint statements (IF NOT EXISTS → idempotent)
# ---------------------------------------------------------------------------

_CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:UserSession) REQUIRE s.session_id IS UNIQUE",
    "CREATE CONSTRAINT video_id IF NOT EXISTS FOR (v:Video) REQUIRE v.video_id IS UNIQUE",
    "CREATE CONSTRAINT hashtag_id IF NOT EXISTS FOR (h:Hashtag) REQUIRE h.hashtag_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT sound_id IF NOT EXISTS FOR (s:Sound) REQUIRE s.song_id IS UNIQUE",
    "CREATE CONSTRAINT country_id IF NOT EXISTS FOR (c:Country) REQUIRE c.country_id IS UNIQUE",
    "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",
    "CREATE CONSTRAINT comment_id IF NOT EXISTS FOR (c:Comment) REQUIRE c.comment_id IS UNIQUE",
]

# ---------------------------------------------------------------------------
# Index statements
# ---------------------------------------------------------------------------

_INDEXES: list[str] = [
    "CREATE INDEX user_country IF NOT EXISTS FOR (u:User) ON (u.country_id)",
    "CREATE INDEX video_posted_at IF NOT EXISTS FOR (v:Video) ON (v.posted_at)",
    "CREATE INDEX session_date IF NOT EXISTS FOR (s:UserSession) ON (s.start_date)",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_schema(driver: Driver) -> None:
    """
    Apply all constraints and indexes to the database.

    Idempotent — uses IF NOT EXISTS so re-running is safe.
    Prints each statement as it runs.
    """
    with driver.session() as session:
        for stmt in _CONSTRAINTS + _INDEXES:
            print(f"  schema: {stmt[:60]}…")
            session.run(stmt)
    print("Schema applied.")

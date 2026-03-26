"""
neo4j/loader.py
===============
Batched UNWIND Cypher upload for all node and relationship types.

Every upload_* function:
  - Accepts driver, data: list[dict], batch_size=500
  - Uses UNWIND $batch AS row + MERGE (idempotent)
  - Shows tqdm progress per batch
  - Returns the total number of records processed
"""

from __future__ import annotations

import math
from typing import Callable

from neo4j import Driver
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run_batches(
    driver: Driver,
    data: list[dict],
    cypher: str,
    batch_size: int,
    desc: str,
) -> int:
    """Split data into batches and execute cypher against each."""
    if not data:
        return 0
    n_batches = math.ceil(len(data) / batch_size)
    with driver.session() as session:
        for i in tqdm(range(n_batches), desc=desc, unit="batch"):
            batch = data[i * batch_size : (i + 1) * batch_size]
            session.run(cypher, batch=batch)
    return len(data)


# ---------------------------------------------------------------------------
# Node uploads
# ---------------------------------------------------------------------------

def upload_countries(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Upload Country nodes.

    Expected keys: country_id, name, iso
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (c:Country {country_id: row.country_id})
    SET c.name = row.name,
        c.iso  = row.iso
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading Country nodes")


def upload_topics(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Upload Topic nodes.

    Expected keys: topic_id, name, slug
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (t:Topic {topic_id: row.topic_id})
    SET t.name = row.name,
        t.slug = row.slug
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading Topic nodes")


def upload_sounds(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Upload Sound nodes.

    Expected keys: song_id, song_name, singer, genre, country
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (s:Sound {song_id: row.song_id})
    SET s.song_name = row.song_name,
        s.singer    = row.singer,
        s.genre     = row.genre,
        s.country   = row.country
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading Sound nodes")


def upload_hashtags(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Upload Hashtag nodes.

    Expected keys: hashtag_id, name, topic_slug
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (h:Hashtag {hashtag_id: row.hashtag_id})
    SET h.name       = row.name,
        h.topic_slug = row.topic_slug
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading Hashtag nodes")


def upload_entities(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Upload Entity nodes.

    Expected keys: entity_id, name, aliases (list), topic_slug
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (e:Entity {entity_id: row.entity_id})
    SET e.name       = row.name,
        e.aliases    = row.aliases,
        e.topic_slug = row.topic_slug
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading Entity nodes")


def upload_users(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Upload User nodes; add :Creator label where is_creator=True.

    Expected keys: user_id, username, joined_at, followers, following,
                   like_count, average_watch_time, last_login, country_id,
                   is_creator
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (u:User {user_id: row.user_id})
    SET u.username           = row.username,
        u.joined_at          = row.joined_at,
        u.followers          = row.followers,
        u.following          = row.following,
        u.like_count         = row.like_count,
        u.average_watch_time = row.average_watch_time,
        u.last_login         = row.last_login,
        u.country_id         = row.country_id,
        u.is_creator         = row.is_creator
    WITH u, row
    CALL apoc.do.when(
        row.is_creator,
        'SET u:Creator RETURN u',
        'RETURN u',
        {u: u}
    ) YIELD value
    RETURN count(*)
    """
    # Fall back to plain Cypher if APOC is not available
    cypher_no_apoc = """
    UNWIND $batch AS row
    MERGE (u:User {user_id: row.user_id})
    SET u.username           = row.username,
        u.joined_at          = row.joined_at,
        u.followers          = row.followers,
        u.following          = row.following,
        u.like_count         = row.like_count,
        u.average_watch_time = row.average_watch_time,
        u.last_login         = row.last_login,
        u.country_id         = row.country_id,
        u.is_creator         = row.is_creator
    """
    cypher_creator_label = """
    UNWIND $batch AS row
    MATCH (u:User {user_id: row.user_id})
    WHERE row.is_creator = true
    SET u:Creator
    """
    if not data:
        return 0
    n_batches = math.ceil(len(data) / batch_size)
    with driver.session() as session:
        for i in tqdm(range(n_batches), desc="Uploading User nodes", unit="batch"):
            batch = data[i * batch_size : (i + 1) * batch_size]
            session.run(cypher_no_apoc, batch=batch)
            # Apply Creator label in a second pass (avoids APOC dependency)
            session.run(cypher_creator_label, batch=batch)
    return len(data)


def upload_sessions(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Upload UserSession nodes.

    Expected keys: session_id, user_id, start_date, end_date
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (s:UserSession {session_id: row.session_id})
    SET s.user_id    = row.user_id,
        s.start_date = row.start_date,
        s.end_date   = row.end_date
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading UserSession nodes")


def upload_videos(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Upload Video nodes.

    Expected keys: video_id, creator_id, topic_id, video_duration_seconds,
                   created_at, description, play_count, like_count,
                   comment_count, share_count, download_count, repost_count
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (v:Video {video_id: row.video_id})
    SET v.creator_id            = row.creator_id,
        v.video_duration        = row.video_duration_seconds,
        v.posted_at             = row.created_at,
        v.description           = row.description,
        v.topic_id              = row.topic_id,
        v.play_count            = row.play_count,
        v.likes                 = row.like_count,
        v.downloads             = row.download_count,
        v.shares                = row.share_count,
        v.reposts               = row.repost_count,
        v.comments              = row.comment_count
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading Video nodes")


def upload_comments(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Upload Comment nodes.

    Expected keys: comment_id, video_id, user_id, comment_text,
                   comment_sentiment, created_at
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (c:Comment {comment_id: row.comment_id})
    SET c.video_id          = row.video_id,
        c.user_id           = row.user_id,
        c.comment_text      = row.comment_text,
        c.comment_sentiment = row.comment_sentiment,
        c.created_at        = row.created_at
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading Comment nodes")


# ---------------------------------------------------------------------------
# Relationship uploads
# ---------------------------------------------------------------------------

def upload_rel_has_session(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """User-[:HAS_SESSION]->UserSession.

    Expected keys: user_id, session_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (u:User {user_id: row.user_id})
    MATCH (s:UserSession {session_id: row.session_id})
    MERGE (u)-[:HAS_SESSION]->(s)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading HAS_SESSION rels")


def upload_rel_last_session(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """User-[:LAST_SESSION]->UserSession.

    Expected keys: user_id, session_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (u:User {user_id: row.user_id})
    MATCH (s:UserSession {session_id: row.session_id})
    MERGE (u)-[:LAST_SESSION]->(s)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading LAST_SESSION rels")


def upload_rel_prev_session(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """UserSession-[:PREVIOUS_SESSION]->UserSession.

    Expected keys: session_id, prev_session_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (s:UserSession {session_id: row.session_id})
    MATCH (p:UserSession {session_id: row.prev_session_id})
    MERGE (s)-[:PREVIOUS_SESSION]->(p)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading PREVIOUS_SESSION rels")


def upload_rel_created_by(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Video-[:CREATED_BY]->Creator (User with :Creator label).

    Expected keys: video_id, author_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (v:Video {video_id: row.video_id})
    MATCH (c:Creator {user_id: row.author_id})
    MERGE (v)-[:CREATED_BY]->(c)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading CREATED_BY rels")


def upload_rel_viewed(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """UserSession-[:VIEWED {watch_time, completion_rate}]->Video.

    Expected keys: session_id, video_id, watch_time, completion_rate
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (s:UserSession {session_id: row.session_id})
    MATCH (v:Video {video_id: row.video_id})
    MERGE (s)-[r:VIEWED]->(v)
    SET r.watch_time      = row.watch_time,
        r.completion_rate = row.completion_rate
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading VIEWED rels")


def upload_rel_liked(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """UserSession-[:LIKED]->Video.

    Expected keys: session_id, video_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (s:UserSession {session_id: row.session_id})
    MATCH (v:Video {video_id: row.video_id})
    MERGE (s)-[:LIKED]->(v)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading LIKED rels")


def upload_rel_skipped(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """UserSession-[:SKIPPED]->Video.

    Expected keys: session_id, video_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (s:UserSession {session_id: row.session_id})
    MATCH (v:Video {video_id: row.video_id})
    MERGE (s)-[:SKIPPED]->(v)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading SKIPPED rels")


def upload_rel_reposted(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """UserSession-[:REPOSTED]->Video.

    Expected keys: session_id, video_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (s:UserSession {session_id: row.session_id})
    MATCH (v:Video {video_id: row.video_id})
    MERGE (s)-[:REPOSTED]->(v)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading REPOSTED rels")


def upload_rel_commented(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """UserSession-[:COMMENTED]->Comment.

    Expected keys: session_id, comment_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (s:UserSession {session_id: row.session_id})
    MATCH (c:Comment {comment_id: row.comment_id})
    MERGE (s)-[:COMMENTED]->(c)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading COMMENTED rels")


def upload_rel_comment_on_video(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Comment-[:ON_VIDEO]->Video.

    Expected keys: comment_id, video_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (c:Comment {comment_id: row.comment_id})
    MATCH (v:Video {video_id: row.video_id})
    MERGE (c)-[:ON_VIDEO]->(v)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading ON_VIDEO rels")


def upload_rel_written_by(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Comment-[:WRITTEN_BY]->User.

    Expected keys: comment_id, user_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (c:Comment {comment_id: row.comment_id})
    MATCH (u:User {user_id: row.user_id})
    MERGE (c)-[:WRITTEN_BY]->(u)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading WRITTEN_BY rels")


def upload_rel_video_hashtag(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Video-[:HAS_HASHTAG]->Hashtag.

    Expected keys: video_id, hashtag_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (v:Video {video_id: row.video_id})
    MATCH (h:Hashtag {hashtag_id: row.hashtag_id})
    MERGE (v)-[:HAS_HASHTAG]->(h)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading HAS_HASHTAG rels")


def upload_rel_video_entity(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Video-[:MENTIONS]->Entity.

    Expected keys: video_id, entity_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (v:Video {video_id: row.video_id})
    MATCH (e:Entity {entity_id: row.entity_id})
    MERGE (v)-[:MENTIONS]->(e)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading MENTIONS rels")


def upload_rel_video_sound(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Video-[:USES_SOUND]->Sound.

    Expected keys: video_id, sound_id  (sound_id holds the song_id value)
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (v:Video {video_id: row.video_id})
    MATCH (s:Sound {song_id: row.sound_id})
    MERGE (v)-[:USES_SOUND]->(s)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading USES_SOUND rels")


def upload_rel_video_topic(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Video-[:IS_ABOUT {is_primary}]->Topic.

    Expected keys: video_id, topic_id, is_primary (bool)
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (v:Video {video_id: row.video_id})
    MATCH (t:Topic {topic_id: row.topic_id})
    MERGE (v)-[r:IS_ABOUT]->(t)
    SET r.is_primary = row.is_primary
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading IS_ABOUT rels")


def upload_rel_entity_topic(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Entity-[:RELATED_TO {is_primary}]->Topic.

    Expected keys: entity_id, topic_id, is_primary (bool)
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (e:Entity {entity_id: row.entity_id})
    MATCH (t:Topic {topic_id: row.topic_id})
    MERGE (e)-[r:RELATED_TO]->(t)
    SET r.is_primary = row.is_primary
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading RELATED_TO (entity→topic) rels")


def upload_rel_user_country(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """User-[:FROM_COUNTRY]->Country.

    Expected keys: user_id, country_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (u:User {user_id: row.user_id})
    MATCH (c:Country {country_id: row.country_id})
    MERGE (u)-[:FROM_COUNTRY]->(c)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading FROM_COUNTRY (user) rels")


def upload_rel_video_country(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Video-[:ORIGINATED_IN]->Country.

    Expected keys: video_id, country_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (v:Video {video_id: row.video_id})
    MATCH (c:Country {country_id: row.country_id})
    MERGE (v)-[:ORIGINATED_IN]->(c)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading ORIGINATED_IN (video) rels")


def upload_rel_sound_country(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """Sound-[:FROM_COUNTRY]->Country.

    Expected keys: song_id, country_id
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (s:Sound {song_id: row.song_id})
    MATCH (c:Country {country_id: row.country_id})
    MERGE (s)-[:FROM_COUNTRY]->(c)
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading FROM_COUNTRY (sound) rels")


def upload_rel_follows(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """User-[:FOLLOWS {engagement_score}]->User.

    Expected keys: follower_id, followee_id, engagement_score
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (a:User {user_id: row.follower_id})
    MATCH (b:User {user_id: row.followee_id})
    MERGE (a)-[r:FOLLOWS]->(b)
    SET r.engagement_score = row.engagement_score
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading FOLLOWS rels")


def upload_rel_interested_topic(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """User-[:INTERESTED_IN_TOPIC {topic_score}]->Topic.

    Expected keys: user_id, topic_id, topic_score
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (u:User {user_id: row.user_id})
    MATCH (t:Topic {topic_id: row.topic_id})
    MERGE (u)-[r:INTERESTED_IN_TOPIC]->(t)
    SET r.topic_score = row.topic_score
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading INTERESTED_IN_TOPIC rels")


def upload_rel_interested_entity(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """User-[:INTERESTED_IN_ENTITY {entity_score}]->Entity.

    Expected keys: user_id, entity_id, entity_score
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (u:User {user_id: row.user_id})
    MATCH (e:Entity {entity_id: row.entity_id})
    MERGE (u)-[r:INTERESTED_IN_ENTITY]->(e)
    SET r.entity_score = row.entity_score
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading INTERESTED_IN_ENTITY rels")


def upload_rel_interested_hashtag(driver: Driver, data: list[dict], batch_size: int = 500) -> int:
    """User-[:INTERESTED_IN_HASHTAG {hashtag_score}]->Hashtag.

    Expected keys: user_id, hashtag_id, hashtag_score
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (u:User {user_id: row.user_id})
    MATCH (h:Hashtag {hashtag_id: row.hashtag_id})
    MERGE (u)-[r:INTERESTED_IN_HASHTAG]->(h)
    SET r.hashtag_score = row.hashtag_score
    """
    return _run_batches(driver, data, cypher, batch_size, "Uploading INTERESTED_IN_HASHTAG rels")

"""
main.py
=======
Full pipeline orchestrator for the TikTok Interest Graph.

Usage examples
--------------
  # Validate generation only (fast, no Neo4j, no LLM)
  python main.py --scale small --skip-llm --skip-upload

  # Regenerate even if data/ cache exists
  python main.py --scale large --skip-llm --skip-upload --regen

  # Full generation + upload, faker fallback for text
  python main.py --scale medium --skip-llm

  # Full run (requires Ollama + Neo4j credentials in .env)
  python main.py --scale large
"""

from __future__ import annotations

import argparse
import time

# ---------------------------------------------------------------------------
# Config + generators
# ---------------------------------------------------------------------------
from generators.base import load_config, reset_config_cache, reset_rng
from generators.taxonomy import (
    entity_topic_links,
    generate_countries,
    generate_entities,
    generate_hashtags,
    generate_sounds,
    generate_topics,
)
from generators.users import generate_follows, generate_users
from generators.sessions import generate_sessions
from generators.videos import assign_video_taxonomy, generate_videos
from generators.interactions import generate_interactions
from generators.persistence import (
    dataset_exists,
    comments_exist,
    descriptions_filled,
    save_dataset,
    save_videos,
    save_comments,
    load_dataset,
    load_comments,
)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
from llm.hf_client import HuggingFaceClient
from llm.generator import fill_video_descriptions, generate_comments

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
from neo4j.connection import get_driver, close_driver
from neo4j.schema import apply_schema
from neo4j.loader import (
    upload_countries,
    upload_topics,
    upload_sounds,
    upload_hashtags,
    upload_entities,
    upload_users,
    upload_sessions,
    upload_videos,
    upload_comments,
    upload_rel_has_session,
    upload_rel_last_session,
    upload_rel_prev_session,
    upload_rel_created_by,
    upload_rel_viewed,
    upload_rel_liked,
    upload_rel_skipped,
    upload_rel_reposted,
    upload_rel_commented,
    upload_rel_comment_on_video,
    upload_rel_written_by,
    upload_rel_video_hashtag,
    upload_rel_video_entity,
    upload_rel_video_sound,
    upload_rel_video_topic,
    upload_rel_entity_topic,
    upload_rel_user_country,
    upload_rel_video_country,
    upload_rel_sound_country,
    upload_rel_follows,
    upload_rel_interested_topic,
    upload_rel_interested_entity,
    upload_rel_interested_hashtag,
)

# ---------------------------------------------------------------------------
# Scale presets  (override scale.num_users / scale.num_videos in params)
# ---------------------------------------------------------------------------
_SCALE_PRESETS: dict[str, dict] = {
    "small":  {"num_users": 100,  "num_videos": 400},
    "medium": {"num_users": 500,  "num_videos": 2000},
    "large":  {"num_users": 1000, "num_videos": 4000},
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Generate a synthetic TikTok interest graph and upload it to Neo4j Aura.",
    )
    p.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="medium",
        help="Scale preset (overrides num_users / num_videos in params.yaml). Default: medium",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Ollama — use faker fallback for descriptions and comments.",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Generate data only — do not connect to Neo4j.",
    )
    p.add_argument(
        "--regen",
        action="store_true",
        help="Force regeneration even if data/{scale}/ cache already exists.",
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a custom params.yaml (default: config/params.yaml).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(step: int, title: str) -> None:
    print(f"\n── Step {step}: {title} ──")


def _fmt(n: int) -> str:
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Generation (Steps 1-5)
# ---------------------------------------------------------------------------

def _generate(cfg: dict, num_users: int, num_videos: int, scale: str) -> dict:
    """Run the full generation pipeline and save to data/{scale}/."""

    _banner(1, "Generate taxonomy nodes")
    topics          = generate_topics(cfg)
    countries       = generate_countries(cfg)
    sounds          = generate_sounds(cfg)
    hashtags        = generate_hashtags(cfg)
    entities        = generate_entities(cfg)
    ent_topic_links = entity_topic_links(entities, topics)
    print(f"  topics={len(topics)}  countries={len(countries)}  sounds={len(sounds)}"
          f"  hashtags={len(hashtags)}  entities={len(entities)}")

    _banner(2, "Generate users + follow graph")
    users    = generate_users(num_users, cfg, cfg, cfg)
    follows  = generate_follows(users, cfg, cfg)
    creators = [u for u in users if u["is_creator"]]
    print(f"  users={len(users)}  creators={len(creators)}  follows={len(follows)}")

    _banner(3, "Generate sessions")
    sessions, last_session_map = generate_sessions(users, cfg)
    print(f"  sessions={len(sessions)}")

    _banner(4, "Generate videos + assign taxonomy")
    videos = generate_videos(
        num_videos, users, topics, hashtags, entities,
        sounds, countries, cfg, cfg, cfg,
    )
    video_hashtags, video_entities, video_sounds, video_topics = assign_video_taxonomy(
        videos, topics, hashtags, entities, sounds, cfg, cfg,
    )
    print(f"  videos={len(videos)}  video-hashtags={len(video_hashtags)}"
          f"  video-entities={len(video_entities)}  video-sounds={len(video_sounds)}")

    _banner(5, "Generate interactions + interest scores")
    interactions = generate_interactions(
        sessions, videos, users, topics,
        video_entities, video_hashtags,
        cfg, cfg,
    )
    views             = interactions["views"]
    likes             = interactions["likes"]
    skips             = interactions["skips"]
    reposts           = interactions["reposts"]
    comment_stubs     = interactions["comment_stubs"]
    topic_interests   = interactions["topic_interests"]
    entity_interests  = interactions["entity_interests"]
    hashtag_interests = interactions["hashtag_interests"]
    print(
        f"  views={len(views)}  likes={len(likes)}  skips={len(skips)}"
        f"  reposts={len(reposts)}  comment_stubs={len(comment_stubs)}"
    )
    print(
        f"  topic_interests={len(topic_interests)}"
        f"  entity_interests={len(entity_interests)}"
        f"  hashtag_interests={len(hashtag_interests)}"
    )

    # Build last_session_map as a list for persistence
    last_sessions = [
        {"user_id": uid, "session_id": sid}
        for uid, sid in last_session_map.items()
    ]

    data = {
        "topics": topics, "countries": countries, "sounds": sounds,
        "hashtags": hashtags, "entities": entities,
        "entity_topic_links": ent_topic_links,
        "users": users, "follows": follows,
        "sessions": sessions,
        "last_sessions": last_sessions,
        "videos": videos,
        "video_hashtags": video_hashtags,
        "video_entities": video_entities,
        "video_sounds": video_sounds,
        "video_topics": video_topics,
        "views": views, "likes": likes, "skips": skips, "reposts": reposts,
        "comment_stubs": comment_stubs,
        "topic_interests": topic_interests,
        "entity_interests": entity_interests,
        "hashtag_interests": hashtag_interests,
    }

    print(f"\n  Saving to data/{scale}/...")
    save_dataset(scale, data)
    return data


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    t_start = time.perf_counter()

    # ── Config ───────────────────────────────────────────────────────────────
    reset_config_cache()
    cfg = load_config(params_path=args.config) if args.config else load_config()

    preset = _SCALE_PRESETS[args.scale]
    cfg.setdefault("scale", {})
    cfg["scale"]["num_users"]  = preset["num_users"]
    cfg["scale"]["num_videos"] = preset["num_videos"]

    num_users  = cfg["scale"]["num_users"]
    num_videos = cfg["scale"]["num_videos"]
    print(f"\nScale: {args.scale}  |  users={num_users}  videos={num_videos}")
    print(f"skip-llm={args.skip_llm}  skip-upload={args.skip_upload}  regen={args.regen}")

    reset_rng(cfg.get("seed", 42))

    # ── Steps 1-5: Generate or load from cache ────────────────────────────────
    if not args.regen and dataset_exists(args.scale):
        print(f"\n── Loading cached data from data/{args.scale}/ ──")
        data = load_dataset(args.scale)
        topics            = data["topics"]
        countries         = data["countries"]
        sounds            = data["sounds"]
        hashtags          = data["hashtags"]
        entities          = data["entities"]
        ent_topic_links   = data["entity_topic_links"]
        users             = data["users"]
        follows           = data["follows"]
        sessions          = data["sessions"]
        last_sessions     = data.get("last_sessions", [])
        videos            = data["videos"]
        video_hashtags    = data["video_hashtags"]
        video_entities    = data["video_entities"]
        video_sounds      = data["video_sounds"]
        video_topics      = data["video_topics"]
        views             = data["views"]
        likes             = data["likes"]
        skips             = data["skips"]
        reposts           = data["reposts"]
        comment_stubs     = data["comment_stubs"]
        topic_interests   = data["topic_interests"]
        entity_interests  = data["entity_interests"]
        hashtag_interests = data["hashtag_interests"]
        print(f"  users={len(users)}  sessions={len(sessions)}  videos={len(videos)}"
              f"  views={len(views)}  follows={len(follows)}")
    else:
        data          = _generate(cfg, num_users, num_videos, args.scale)
        topics        = data["topics"]
        countries     = data["countries"]
        sounds        = data["sounds"]
        hashtags      = data["hashtags"]
        entities      = data["entities"]
        ent_topic_links = data["entity_topic_links"]
        users         = data["users"]
        follows       = data["follows"]
        sessions      = data["sessions"]
        last_sessions = data["last_sessions"]
        videos        = data["videos"]
        video_hashtags  = data["video_hashtags"]
        video_entities  = data["video_entities"]
        video_sounds    = data["video_sounds"]
        video_topics    = data["video_topics"]
        views           = data["views"]
        likes           = data["likes"]
        skips           = data["skips"]
        reposts         = data["reposts"]
        comment_stubs   = data["comment_stubs"]
        topic_interests   = data["topic_interests"]
        entity_interests  = data["entity_interests"]
        hashtag_interests = data["hashtag_interests"]

    # ── Step 6: LLM text fill ─────────────────────────────────────────────────
    _banner(6, "LLM text fill (descriptions + comments)")

    need_descriptions = not descriptions_filled(args.scale) or args.regen
    need_comments     = not comments_exist(args.scale) or args.regen

    if not need_descriptions and not need_comments:
        print("  Already filled — loading comments from cache")
        comments = load_comments(args.scale)
    else:
        client = HuggingFaceClient(cfg)
        if args.skip_llm:
            print("  Skipping LLM — using faker fallback")
            client.is_available = lambda: False

        if need_descriptions:
            fill_video_descriptions(videos, cfg, client, cfg)
            save_videos(args.scale, videos)

        if need_comments:
            comments = generate_comments(comment_stubs, videos, cfg, client)
            save_comments(args.scale, comments)
        else:
            comments = load_comments(args.scale)

    print(f"  descriptions filled={len(videos)}  comments={len(comments)}")

    # ── Step 7–9: Neo4j upload ────────────────────────────────────────────────
    node_counts: dict[str, int] = {}
    rel_counts:  dict[str, int] = {}

    if args.skip_upload:
        print("\n── Steps 7–9: Skipped (--skip-upload) ──")
    else:
        driver = get_driver()

        _banner(7, "Neo4j schema setup")
        apply_schema(driver)

        _banner(8, "Upload nodes")
        node_counts["Country"]  = upload_countries(driver, countries)
        node_counts["Topic"]    = upload_topics(driver, topics)
        node_counts["Sound"]    = upload_sounds(driver, sounds)
        node_counts["Hashtag"]  = upload_hashtags(driver, hashtags)
        node_counts["Entity"]   = upload_entities(driver, entities)
        node_counts["User"]     = upload_users(driver, users)
        node_counts["Session"]  = upload_sessions(driver, sessions)
        node_counts["Video"]    = upload_videos(driver, videos)
        node_counts["Comment"]  = upload_comments(driver, comments)

        _banner(9, "Upload relationships")

        has_session_data = [
            {"user_id": s["user_id"], "session_id": s["session_id"]}
            for s in sessions
        ]
        rel_counts["HAS_SESSION"]      = upload_rel_has_session(driver, has_session_data)
        rel_counts["LAST_SESSION"]     = upload_rel_last_session(driver, last_sessions)

        prev_session_data = [
            {"session_id": s["session_id"], "prev_session_id": s["prev_session_id"]}
            for s in sessions if s.get("prev_session_id")
        ]
        rel_counts["PREVIOUS_SESSION"] = upload_rel_prev_session(driver, prev_session_data)

        created_by_data = [
            {"video_id": v["video_id"], "author_id": v["creator_id"]}
            for v in videos
        ]
        rel_counts["CREATED_BY"]       = upload_rel_created_by(driver, created_by_data)

        rel_counts["VIEWED"]           = upload_rel_viewed(driver, views)
        rel_counts["LIKED"]            = upload_rel_liked(driver, likes)
        rel_counts["SKIPPED"]          = upload_rel_skipped(driver, skips)
        rel_counts["REPOSTED"]         = upload_rel_reposted(driver, reposts)

        commented_data = [
            {"session_id": stub["session_id"], "comment_id": stub["comment_id"]}
            for stub in comment_stubs
        ]
        rel_counts["COMMENTED"]        = upload_rel_commented(driver, commented_data)

        on_video_data = [
            {"comment_id": c["comment_id"], "video_id": c["video_id"]}
            for c in comments
        ]
        rel_counts["ON_VIDEO"]         = upload_rel_comment_on_video(driver, on_video_data)

        written_by_data = [
            {"comment_id": c["comment_id"], "user_id": c["user_id"]}
            for c in comments
        ]
        rel_counts["WRITTEN_BY"]       = upload_rel_written_by(driver, written_by_data)

        rel_counts["HAS_HASHTAG"]      = upload_rel_video_hashtag(driver, video_hashtags)
        rel_counts["MENTIONS"]         = upload_rel_video_entity(driver, video_entities)
        rel_counts["USES_SOUND"]       = upload_rel_video_sound(driver, video_sounds)
        rel_counts["IS_ABOUT"]         = upload_rel_video_topic(driver, video_topics)
        rel_counts["RELATED_TO"]       = upload_rel_entity_topic(driver, ent_topic_links)

        user_country_data = [
            {"user_id": u["user_id"], "country_id": u["country_id"]}
            for u in users
        ]
        rel_counts["FROM_COUNTRY(U)"]  = upload_rel_user_country(driver, user_country_data)

        video_country_data = [
            {"video_id": v["video_id"], "country_id": v["country_id"]}
            for v in videos
        ]
        rel_counts["ORIGINATED_IN"]    = upload_rel_video_country(driver, video_country_data)

        sound_country_data = [
            {"song_id": s["song_id"], "country_id": s["country_id"]}
            for s in sounds
        ]
        rel_counts["FROM_COUNTRY(S)"]  = upload_rel_sound_country(driver, sound_country_data)

        rel_counts["FOLLOWS"]                = upload_rel_follows(driver, follows)
        rel_counts["INTERESTED_IN_TOPIC"]    = upload_rel_interested_topic(driver, topic_interests)
        rel_counts["INTERESTED_IN_ENTITY"]   = upload_rel_interested_entity(driver, entity_interests)
        rel_counts["INTERESTED_IN_HASHTAG"]  = upload_rel_interested_hashtag(driver, hashtag_interests)

        close_driver()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if node_counts:
        print("Nodes uploaded:")
        for label, n in node_counts.items():
            print(f"  {label:<12} {_fmt(n)}")
    else:
        print("Nodes (cached):")
        for label, n in [
            ("Country", len(countries)), ("Topic", len(topics)), ("Sound", len(sounds)),
            ("Hashtag", len(hashtags)), ("Entity", len(entities)), ("User", len(users)),
            ("Session", len(sessions)), ("Video", len(videos)), ("Comment", len(comments)),
        ]:
            print(f"  {label:<12} {_fmt(n)}")

    if rel_counts:
        print("Rels uploaded:")
        for rel_type, n in rel_counts.items():
            print(f"  {rel_type:<24} {_fmt(n)}")
    else:
        print("Rels (cached):")
        for rel_type, n in [
            ("VIEWED",          len(views)),
            ("LIKED",           len(likes)),
            ("SKIPPED",         len(skips)),
            ("REPOSTED",        len(reposts)),
            ("COMMENTED",       len(comment_stubs)),
            ("FOLLOWS",         len(follows)),
            ("INTERESTED_IN_*", len(topic_interests) + len(entity_interests) + len(hashtag_interests)),
        ]:
            print(f"  {rel_type:<24} {_fmt(n)}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    run(args)

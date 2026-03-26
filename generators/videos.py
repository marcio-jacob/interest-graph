"""
generators/videos.py
====================
Video node generation and taxonomy assignment.

generate_videos       — Video records with KuaiRec-calibrated engagement stats
assign_video_taxonomy — relationship lists: video_hashtags, video_entities,
                        video_sounds, video_topics
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from generators.base import (
    get_rng,
    sample_from_histogram,
    sample_lognormal,
    weighted_choice,
)

# KuaiRec distinct users in big_matrix.csv — used as scale denominator
_KUAIREC_USER_COUNT = 7176


def generate_videos(
    num_videos: int,
    users: list[dict],
    topics: list[dict],
    hashtags: list[dict],
    entities: list[dict],
    sounds: list[dict],
    countries: list[dict],
    taxonomy: dict,
    params: dict,
    distributions: dict,
) -> list[dict]:
    """
    Generate *num_videos* Video records.

    Creator selection uses power-law weights (followers^0.3).
    Duration is drawn from the KuaiRec histogram (histogram_clipped_600s).
    Engagement counts are Poisson-sampled from KuaiRec rates scaled by
    len(users) / 7176.

    Returns list of dicts with keys:
        video_id, creator_id, topic_id, secondary_topic_id, country_id,
        video_duration_seconds, play_count, like_count, comment_count,
        share_count, download_count, repost_count, description, created_at
    """
    if num_videos <= 0:
        return []

    creators = [u for u in users if u["is_creator"]]
    if not creators:
        return []

    rng = get_rng()

    # Power-law creator selection weights
    creator_weights = [max(u["followers"], 1) ** 0.3 for u in creators]

    # Topic index
    topic_ids = [t["topic_id"] for t in topics]
    slug_to_tid = {t["slug"]: t["topic_id"] for t in topics}

    # Country index
    country_ids = [c["country_id"] for c in countries]
    country_weights = [c.get("user_weight", 1.0) for c in countries]

    # Duration histogram
    dur_cfg = distributions["watch_behavior"]["video_duration_seconds"][
        "histogram_clipped_600s"
    ]
    dur_edges = dur_cfg["bin_edges"]
    dur_counts = dur_cfg["counts"]

    # Params
    video_cfg = params["videos"]
    play_mu = video_cfg["play_count_lognormal_mu"]
    play_sigma = video_cfg["play_count_lognormal_sigma"]
    scale_factor = len(users) / _KUAIREC_USER_COUNT
    secondary_prob = video_cfg["secondary_topic_probability"]
    country_match_prob = video_cfg["video_country_matches_creator"]

    intr = params["interactions"]
    like_rate = intr["like_rate"]
    comment_rate = intr["comment_rate"]
    share_rate = intr["share_rate"]
    repost_rate = intr["repost_rate"]
    download_rate = share_rate  # downloads ≈ shares in KuaiRec

    sim_start = datetime.strptime(params["dates"]["sim_start"], "%Y-%m-%d")
    sim_end = datetime.strptime(params["dates"]["sim_end"], "%Y-%m-%d")

    # Country → affinity topic slugs
    country_affinity: dict[str, list[str]] = {
        c["country_id"]: c.get("affinity_topics", []) for c in taxonomy["countries"]
    }

    videos: list[dict] = []
    for _ in range(num_videos):
        # ── Creator ────────────────────────────────────────────────────────
        creator = weighted_choice(creators, creator_weights)
        creator_country = creator["country_id"]

        # ── Primary topic (60 % country-affinity, 40 % uniform) ────────────
        affinity_slugs = country_affinity.get(creator_country, [])
        if affinity_slugs and rng.random() < 0.60:
            slug = affinity_slugs[int(rng.integers(0, len(affinity_slugs)))]
            primary_tid = slug_to_tid.get(slug, topic_ids[0])
        else:
            primary_tid = topic_ids[int(rng.integers(0, len(topic_ids)))]

        # ── Secondary topic ─────────────────────────────────────────────────
        secondary_tid: str | None = None
        if rng.random() < secondary_prob:
            others = [tid for tid in topic_ids if tid != primary_tid]
            secondary_tid = others[int(rng.integers(0, len(others)))]

        # ── Country ─────────────────────────────────────────────────────────
        if rng.random() < country_match_prob:
            video_country = creator_country
        else:
            video_country = weighted_choice(country_ids, country_weights)

        # ── Duration ────────────────────────────────────────────────────────
        dur = max(1.0, sample_from_histogram(dur_edges, dur_counts))

        # ── Play count (scaled to simulation size) ───────────────────────────
        raw_plays = sample_lognormal(play_mu, play_sigma, clip_lo=1.0)
        play_count = max(1, int(round(raw_plays * scale_factor)))

        # ── Poisson engagement ───────────────────────────────────────────────
        like_count = int(rng.poisson(play_count * like_rate))
        comment_count = int(rng.poisson(play_count * comment_rate))
        share_count = int(rng.poisson(play_count * share_rate))
        download_count = int(rng.poisson(play_count * download_rate))
        repost_count = int(rng.poisson(play_count * repost_rate))

        # ── Created at (after creator joined, within sim range) ──────────────
        earliest = max(creator["joined_at"], sim_start)
        if earliest >= sim_end:
            created_at = sim_end
        else:
            offset = int(rng.integers(0, (sim_end - earliest).days + 1))
            created_at = earliest + timedelta(days=offset)

        videos.append(
            {
                "video_id": str(uuid.uuid4()),
                "creator_id": creator["user_id"],
                "topic_id": primary_tid,
                "secondary_topic_id": secondary_tid,
                "country_id": video_country,
                "video_duration_seconds": round(dur, 3),
                "play_count": play_count,
                "like_count": like_count,
                "comment_count": comment_count,
                "share_count": share_count,
                "download_count": download_count,
                "repost_count": repost_count,
                "description": "PLACEHOLDER",
                "created_at": created_at,
            }
        )

    return videos


def assign_video_taxonomy(
    videos: list[dict],
    topics: list[dict],
    hashtags: list[dict],
    entities: list[dict],
    sounds: list[dict],
    taxonomy: dict,
    params: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Assign hashtags, entities, sounds, and topic relationships to each video.

    Returns:
        video_hashtags : [{video_id, hashtag_id}]
        video_entities : [{video_id, entity_id}]
        video_sounds   : [{video_id, sound_id}]
        video_topics   : [{video_id, topic_id, is_primary}]
    """
    if not videos:
        return [], [], [], []

    rng = get_rng()
    video_cfg = params["videos"]
    ht_min = video_cfg["hashtags_per_video_min"]
    ht_max = video_cfg["hashtags_per_video_max"]
    ent_min = video_cfg["entities_per_video_min"]
    ent_max = video_cfg["entities_per_video_max"]

    # Build lookup indexes
    tid_to_slug = {t["topic_id"]: t["slug"] for t in topics}

    # Hashtags grouped by topic slug
    ht_by_topic: dict[str, list[dict]] = defaultdict(list)
    for ht in hashtags:
        ht_by_topic[ht["topic_slug"]].append(ht)

    # Entities grouped by primary_topic slug
    ent_by_topic: dict[str, list[dict]] = defaultdict(list)
    for ent in entities:
        ent_by_topic[ent["primary_topic"]].append(ent)

    # Sound affinity weights per topic slug: {slug: [weight, ...]} indexed by sound_ids order
    sound_ids = [s["song_id"] for s in sounds]
    affinity_raw = taxonomy.get("sound_topic_affinity", {})
    sound_topic_weights: dict[str, list[float]] = {}
    for t in topics:
        slug = t["slug"]
        base = {sid: 1.0 for sid in sound_ids}
        for entry in affinity_raw.get(slug, []):
            base[entry["song_id"]] = float(entry["weight"])
        sound_topic_weights[slug] = [base[sid] for sid in sound_ids]

    video_hashtags: list[dict] = []
    video_entities: list[dict] = []
    video_sounds: list[dict] = []
    video_topics: list[dict] = []

    for v in videos:
        vid = v["video_id"]
        primary_tid = v["topic_id"]
        secondary_tid = v.get("secondary_topic_id")
        primary_slug = tid_to_slug.get(primary_tid, "")
        secondary_slug = tid_to_slug.get(secondary_tid, "") if secondary_tid else ""

        # ── Topic links ──────────────────────────────────────────────────────
        video_topics.append({"video_id": vid, "topic_id": primary_tid, "is_primary": True})
        if secondary_tid:
            video_topics.append(
                {"video_id": vid, "topic_id": secondary_tid, "is_primary": False}
            )

        # ── Hashtags ─────────────────────────────────────────────────────────
        n_ht = int(rng.integers(ht_min, ht_max + 1))
        pool: list[dict] = list(ht_by_topic.get(primary_slug, []))
        if secondary_slug:
            pool += ht_by_topic.get(secondary_slug, [])
        if not pool:
            pool = hashtags  # fallback
        chosen_ht_idx = rng.choice(len(pool), size=min(n_ht, len(pool)), replace=False)
        for idx in chosen_ht_idx:
            video_hashtags.append({"video_id": vid, "hashtag_id": pool[idx]["hashtag_id"]})

        # ── Entities ─────────────────────────────────────────────────────────
        n_ent = int(rng.integers(ent_min, ent_max + 1))
        if n_ent > 0:
            ent_pool: list[dict] = list(ent_by_topic.get(primary_slug, []))
            if secondary_slug:
                ent_pool += ent_by_topic.get(secondary_slug, [])
            if ent_pool:
                chosen_ent_idx = rng.choice(
                    len(ent_pool), size=min(n_ent, len(ent_pool)), replace=False
                )
                for idx in chosen_ent_idx:
                    video_entities.append(
                        {"video_id": vid, "entity_id": ent_pool[idx]["entity_id"]}
                    )

        # ── Sound (exactly one per video) ────────────────────────────────────
        weights = sound_topic_weights.get(primary_slug, [1.0] * len(sound_ids))
        chosen_sound = weighted_choice(sound_ids, weights)
        video_sounds.append({"video_id": vid, "sound_id": chosen_sound})

    return video_hashtags, video_entities, video_sounds, video_topics

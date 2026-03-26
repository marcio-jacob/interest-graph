"""
generators/interactions.py
===========================
Behavioural simulation: feeds, watches, likes, skips, reposts, comments,
and derived per-user interest scores.

generate_interactions(sessions, videos, users, topics, entities, hashtags,
                      params, distributions)

    entities  — video→entity relationship list  [{video_id, entity_id}]
    hashtags  — video→hashtag relationship list [{video_id, hashtag_id}]

Returns a dict with keys:
    views, likes, skips, reposts, comment_stubs,
    topic_interests, entity_interests, hashtag_interests
"""

from __future__ import annotations

import uuid
from collections import defaultdict

import numpy as np

from generators.base import clamp, get_rng, sample_from_histogram

_SENTIMENTS = ["positive", "neutral", "negative"]

_EMPTY_RESULT: dict[str, list] = {
    "views": [], "likes": [], "skips": [], "reposts": [],
    "comment_stubs": [], "topic_interests": [],
    "entity_interests": [], "hashtag_interests": [],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_sentiment(weights: list[float], rng) -> str:
    total = sum(weights)
    probs = [w / total for w in weights]
    return _SENTIMENTS[int(rng.choice(3, p=probs))]


def _cold_start_feed(
    user_country: str,
    n: int,
    all_videos: list[dict],
    videos_by_country: dict[str, list[dict]],
    country_bias: float,
    rng,
) -> list[dict]:
    """Country-biased random feed for a user's very first session (no duplicates)."""
    if not all_videos:
        return []
    if rng.random() < country_bias:
        pool = videos_by_country.get(user_country) or all_videos
    else:
        pool = all_videos

    # Fast path: pool has enough unique videos
    if len(pool) >= n:
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[int(i)] for i in idx]

    # Pool is smaller than n — take all, then pad from all_videos (no duplicates)
    seen: set[str] = set()
    result: list[dict] = []
    for v in pool:
        if v["video_id"] not in seen:
            result.append(v)
            seen.add(v["video_id"])
    for _ in range((n - len(result)) * 5 + 1):
        if len(result) >= n:
            break
        video = all_videos[int(rng.integers(0, len(all_videos)))]
        if video["video_id"] not in seen:
            result.append(video)
            seen.add(video["video_id"])
    return result


def _warm_feed(
    topic_scores: dict[str, float],
    n: int,
    interest_frac: float,
    all_videos: list[dict],
    videos_by_topic: dict[str, list[dict]],
    softmax_temp: float,
    rng,
) -> list[dict]:
    """Interest-matched (60%) + exploratory (40%) feed for warm sessions."""
    if not all_videos:
        return []

    seen: set[str] = set()
    result: list[dict] = []
    n_interest = int(n * interest_frac)

    # ── Interest-matched ──────────────────────────────────────────────────
    positive = {k: v for k, v in topic_scores.items() if v > 0}
    if n_interest > 0 and positive:
        tids = list(positive.keys())
        scores = np.array([positive[t] for t in tids], dtype=float)
        exp_s = np.exp(scores * softmax_temp)
        probs = exp_s / exp_s.sum()

        for _ in range(n_interest * 5):
            if len(result) >= n_interest:
                break
            tidx = int(rng.choice(len(tids), p=probs))
            pool = videos_by_topic.get(tids[tidx], [])
            if not pool:
                continue
            video = pool[int(rng.integers(0, len(pool)))]
            if video["video_id"] not in seen:
                result.append(video)
                seen.add(video["video_id"])

    # ── Fill remaining slots (exploratory + any interest shortfall) ───────
    for _ in range((n - len(result)) * 5 + 1):
        if len(result) >= n:
            break
        video = all_videos[int(rng.integers(0, len(all_videos)))]
        if video["video_id"] not in seen:
            result.append(video)
            seen.add(video["video_id"])

    return result


def _normalize_and_filter(
    user_scores: dict[str, dict[str, float]],
    id_key: str,
    score_key: str,
    min_threshold: float,
) -> list[dict]:
    """Normalise per-user raw scores to [0,1] and drop entries below threshold."""
    out: list[dict] = []
    for user_id, scores in user_scores.items():
        pos = {k: v for k, v in scores.items() if v > 0}
        if not pos:
            continue
        mx = max(pos.values())
        for item_id, raw in pos.items():
            norm = raw / mx
            if norm > min_threshold:
                out.append({"user_id": user_id, id_key: item_id, score_key: round(norm, 4)})
    return out


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_interactions(
    sessions: list[dict],
    videos: list[dict],
    users: list[dict],
    topics: list[dict],
    entities: list[dict],
    hashtags: list[dict],
    params: dict,
    distributions: dict,
) -> dict[str, list[dict]]:
    """
    Simulate session-level viewing behaviour and compute interest scores.

    Parameters
    ----------
    sessions      : Session records — {session_id, user_id, start_date, …}
    videos        : Video records   — {video_id, topic_id, secondary_topic_id,
                                       video_duration_seconds, country_id, …}
    users         : User records    — {user_id, country_id, …}
    topics        : Topic records   — [{topic_id, name, slug}]
    entities      : video→entity relationship list  — [{video_id, entity_id}]
    hashtags      : video→hashtag relationship list — [{video_id, hashtag_id}]
    params        : merged config dict (params + taxonomy + distributions)
    distributions : merged config dict (same object is fine)
    """
    if not sessions or not videos:
        return dict(_EMPTY_RESULT)

    rng = get_rng()

    # ── Indexes ──────────────────────────────────────────────────────────────
    user_map: dict[str, dict] = {u["user_id"]: u for u in users}

    sessions_by_user: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        sessions_by_user[s["user_id"]].append(s)
    for uid in sessions_by_user:
        sessions_by_user[uid].sort(key=lambda x: x["start_date"])

    vid_to_entities: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        vid_to_entities[e["video_id"]].append(e["entity_id"])

    vid_to_hashtags: dict[str, list[str]] = defaultdict(list)
    for h in hashtags:
        vid_to_hashtags[h["video_id"]].append(h["hashtag_id"])

    videos_by_country: dict[str, list[dict]] = defaultdict(list)
    videos_by_topic: dict[str, list[dict]] = defaultdict(list)
    for v in videos:
        videos_by_country[v["country_id"]].append(v)
        videos_by_topic[v["topic_id"]].append(v)

    tid_to_slug: dict[str, str] = {t["topic_id"]: t["slug"] for t in topics}

    # ── Parameters ───────────────────────────────────────────────────────────
    wr_hist   = distributions["watch_behavior"]["watch_ratio"]["histogram"]
    wr_edges  = wr_hist["bin_edges"]
    wr_counts = wr_hist["counts"]

    vps_min  = params["sessions"]["videos_per_session_min"]
    vps_max  = params["sessions"]["videos_per_session_max"]
    vps_mean = params["sessions"]["videos_per_session_mean"]

    skip_thr     = params["interactions"]["skip_threshold"]
    pos_thr      = params["interactions"]["watch_positive_threshold"]
    like_rate    = params["interactions"]["like_rate"]
    comment_rate = params["interactions"]["comment_rate"]
    repost_rate  = params["interactions"]["repost_rate"]
    like_boost   = params["interactions"]["like_boost_on_completion"]
    cmt_boost    = params["interactions"]["comment_boost_on_like"]

    skip_delta   = params["interest"]["skip_delta"]
    part_delta   = params["interest"]["view_partial"]
    compl_delta  = params["interest"]["view_complete"]
    like_delta   = params["interest"]["like_delta"]
    cmt_delta    = params["interest"]["comment_delta"]
    repost_delta = params["interest"]["repost_delta"]
    min_score    = params["interest"]["min_score_threshold"]

    cold_bias    = params["feed"]["cold_start_country_bias"]
    int_frac     = params["feed"]["interest_matched_fraction"]
    softmax_temp = params["feed"]["softmax_temperature"]

    sent_weights = params.get("comment_sentiment_weights", {})

    # ── Output containers ────────────────────────────────────────────────────
    views: list[dict]         = []
    likes: list[dict]         = []
    skips: list[dict]         = []
    reposts: list[dict]       = []
    comment_stubs: list[dict] = []

    # Accumulated raw interest signals — normalised after all sessions
    u_topic:   dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    u_entity:  dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    u_hashtag: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # ── Session loop ─────────────────────────────────────────────────────────
    for user_id, user_sessions in sessions_by_user.items():
        user = user_map.get(user_id)
        if not user:
            continue
        user_country = user["country_id"]

        for sess_idx, session in enumerate(user_sessions):
            sid = session["session_id"]
            n_vids = int(clamp(rng.poisson(vps_mean), vps_min, vps_max))

            if sess_idx == 0:
                feed = _cold_start_feed(
                    user_country, n_vids, videos,
                    videos_by_country, cold_bias, rng,
                )
            else:
                feed = _warm_feed(
                    dict(u_topic[user_id]), n_vids, int_frac,
                    videos, videos_by_topic, softmax_temp, rng,
                )

            session_has_view = False
            for vid_idx, video in enumerate(feed):
                vid          = video["video_id"]
                dur          = video.get("video_duration_seconds", 10.0)
                primary_tid  = video["topic_id"]
                secondary_tid = video.get("secondary_topic_id")

                watch_ratio = sample_from_histogram(wr_edges, wr_counts)
                # Guarantee the last video in a session results in a view
                # if none of the earlier ones did.
                if not session_has_view and vid_idx == len(feed) - 1:
                    watch_ratio = max(watch_ratio, skip_thr)
                watch_time  = watch_ratio * dur

                if watch_ratio < skip_thr:
                    skips.append({"session_id": sid, "video_id": vid})
                    delta = skip_delta
                else:
                    cr = min(1.0, watch_ratio)
                    views.append({
                        "session_id":    sid,
                        "video_id":      vid,
                        "watch_time":    round(watch_time, 2),
                        "completion_rate": round(cr, 4),
                    })
                    session_has_view = True

                    delta = compl_delta * cr if cr >= pos_thr else part_delta

                    # Like
                    eff_like = like_rate * (like_boost if cr >= pos_thr else 1.0)
                    is_liked = bool(rng.random() < eff_like)
                    if is_liked:
                        likes.append({"session_id": sid, "video_id": vid})
                        delta += like_delta

                    # Comment
                    eff_cmt = comment_rate * (cmt_boost if is_liked else 1.0)
                    if bool(rng.random() < eff_cmt):
                        slug = tid_to_slug.get(primary_tid, "lifestyle_vlog")
                        w = sent_weights.get(slug, [0.70, 0.20, 0.10])
                        comment_stubs.append({
                            "comment_id": str(uuid.uuid4()),
                            "session_id": sid,
                            "video_id":   vid,
                            "user_id":    user_id,
                            "sentiment":  _sample_sentiment(w, rng),
                            "created_at": session["start_date"],
                        })
                        delta += cmt_delta

                    # Repost
                    if bool(rng.random() < repost_rate):
                        reposts.append({"session_id": sid, "video_id": vid})
                        delta += repost_delta

                # Accumulate interest signals
                u_topic[user_id][primary_tid] += delta
                if secondary_tid:
                    u_topic[user_id][secondary_tid] += delta * 0.5
                for eid in vid_to_entities[vid]:
                    u_entity[user_id][eid] += delta
                for hid in vid_to_hashtags[vid]:
                    u_hashtag[user_id][hid] += delta

    # ── Normalise interest scores ─────────────────────────────────────────────
    return {
        "views":             views,
        "likes":             likes,
        "skips":             skips,
        "reposts":           reposts,
        "comment_stubs":     comment_stubs,
        "topic_interests":   _normalize_and_filter(u_topic,   "topic_id",   "topic_score",   min_score),
        "entity_interests":  _normalize_and_filter(u_entity,  "entity_id",  "entity_score",  min_score),
        "hashtag_interests": _normalize_and_filter(u_hashtag, "hashtag_id", "hashtag_score", min_score),
    }

"""
generators/users.py
====================
Generate User nodes (with Creator labels) and FOLLOWS relationships.

Public API
----------
generate_users(num_users, params, taxonomy, distributions)  → list[dict]
generate_follows(users, params, distributions)              → list[dict]
build_username(fake, topic_words, pattern, adjectives,
               country_suffix, rng)                         → str
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from datetime import datetime, timedelta

import numpy as np
from faker import Faker

from generators.base import clamp, get_rng

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Verb stems used in the "{verb}er" token.
# Pattern "the{noun}{verb}er" + verb='lov' + literal 'er' → "thefoodlover"
_VERB_STEMS: list[str] = [
    "lov", "cook", "play", "danc", "mak", "watch", "eat",
    "build", "design", "stream", "film", "paint", "trav", "creat",
]

# Faker locales that produce clean Latin-alphabet names.
# CJK (JP, KR) and Devanagari (IN) names become empty after stripping
# non-ASCII, so those fall back to en_US while keeping the country suffix.
_SAFE_LOCALES: dict[str, str] = {
    "US": "en_US", "GB": "en_GB", "BR": "pt_BR",
    "DE": "de_DE", "MX": "es_MX", "ID": "id_ID",
    "PH": "en_PH", "NG": "en_NG",
}

# Cached Faker instances (creating per-user is expensive)
_faker_cache: dict[str, Faker] = {}
_fallback_faker: Faker = Faker("en_US")


def _get_faker(locale: str) -> Faker:
    if locale not in _faker_cache:
        _faker_cache[locale] = Faker(locale)
    return _faker_cache[locale]


# ---------------------------------------------------------------------------
# Username helpers
# ---------------------------------------------------------------------------

def _strip_diacritics(s: str) -> str:
    """é→e, ü→u, ñ→n, etc.  CJK characters have no decomposition and stay."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _clean_username(raw: str) -> str:
    """Lowercase, strip non-ASCII names, keep only [a-z0-9_.], cap at 30 chars."""
    s = _strip_diacritics(raw)
    s = s.lower().replace(" ", "").replace("-", "_")
    s = re.sub(r"[^a-z0-9_.]", "", s)
    return s[:30]


def _safe_name(fake: Faker, part: str = "first") -> str:
    """
    Return a cleaned first or last name from *fake*.
    Falls back to _fallback_faker (en_US) if the locale produces CJK/empty output.
    """
    for source in (fake, _fallback_faker):
        try:
            raw = source.first_name() if part == "first" else source.last_name()
            cleaned = re.sub(r"[^a-z]", "", _strip_diacritics(raw).lower())
            if cleaned:
                return cleaned
        except Exception:
            continue
    return "user"


def _build_topic_vocabulary(taxonomy: dict) -> list[str]:
    """
    Build a flat list of single-word tokens from topic slugs and hashtag names.
    Used to populate {noun} / {topic_word} in username patterns.
    """
    words: set[str] = set()
    for topic in taxonomy.get("topics", []):
        for part in topic["slug"].split("_"):
            if len(part) >= 3:
                words.add(part.lower())
    for slug_tags in taxonomy.get("hashtags", {}).values():
        for tag in slug_tags:
            clean = tag.lstrip("#").lower()
            if 4 <= len(clean) <= 12:
                words.add(clean)
    return sorted(words)


def build_username(
    fake: Faker,
    topic_words: list[str],
    pattern: str,
    adjectives: list[str],
    country_suffix: str,
    rng: np.random.Generator,
) -> str:
    """
    Render one username *pattern* and return the cleaned result.

    Pattern tokens (all optional, depends on the pattern string):
        {first}       Faker first name (locale-aware, Latin fallback)
        {last}        Faker last name
        {nn}          Random 2-digit number (10-98)
        {noun}        Random word from topic vocabulary
        {topic_word}  Random word from topic vocabulary (same pool)
        {adjective}   Word from the curated adjectives list
        {country_adj} Country-based suffix  (e.g. 'br', 'jp')
        {verb}        Verb stem; the literal 'er' in the pattern appends to it
                      → "the{noun}{verb}er" + verb='lov' → "thefoodlover"
    """
    tv  = topic_words or ["content"]
    adj = adjectives  or ["cool"]

    tokens = {
        "first":       _safe_name(fake, "first"),
        "last":        _safe_name(fake, "last"),
        "nn":          f"{int(rng.integers(10, 99))}",
        "noun":        tv[int(rng.integers(0, len(tv)))],
        "topic_word":  tv[int(rng.integers(0, len(tv)))],
        "adjective":   adj[int(rng.integers(0, len(adj)))],
        "country_adj": country_suffix,
        "verb":        _VERB_STEMS[int(rng.integers(0, len(_VERB_STEMS)))],
    }

    try:
        raw = pattern.format(**tokens)
    except KeyError:
        raw = tokens["first"] + tokens["nn"]

    return _clean_username(raw)


# ---------------------------------------------------------------------------
# User generation
# ---------------------------------------------------------------------------

def generate_users(
    num_users: int,
    params: dict,
    taxonomy: dict,
    distributions: dict,
) -> list[dict]:
    """
    Return a list of *num_users* User node dicts.

    Dict keys
    ---------
    user_id, username, joined_at, followers, following, like_count,
    average_watch_time, last_login, country_id, is_creator
    """
    rng = get_rng()

    # --- Lookup structures ---------------------------------------------------
    topic_words = _build_topic_vocabulary(taxonomy)
    patterns    = taxonomy.get("username_patterns", ["{first}{last}{nn}"])
    adjectives  = taxonomy.get("username_adjectives", ["cool"])
    country_sfx = taxonomy.get("username_country_suffixes", {})

    countries     = taxonomy.get("countries", [])
    country_ids   = [c["country_id"] for c in countries]
    raw_w         = np.array([c.get("user_weight", 1.0) for c in countries], dtype=float)
    country_probs = raw_w / raw_w.sum()

    # --- Distribution parameters ---------------------------------------------
    us          = distributions["user_social"]
    fans_mu     = us["fans_user_num"]["lognormal_fit"]["mu"]
    fans_sigma  = us["fans_user_num"]["lognormal_fit"]["sigma"]
    follow_mu   = us["follow_user_num"]["lognormal_fit"]["mu"]
    follow_sig  = us["follow_user_num"]["lognormal_fit"]["sigma"]
    reg_mean    = us["register_days"]["stats"]["mean"]
    reg_std     = us["register_days"]["stats"]["std"]

    author_frac = us.get("video_author_fraction", params["scale"]["creator_fraction"])

    sim_start = datetime.strptime(params["dates"]["sim_start"], "%Y-%m-%d")
    sim_end   = datetime.strptime(params["dates"]["sim_end"],   "%Y-%m-%d")

    # Guard: nothing to generate
    if num_users <= 0:
        return []

    # Pre-assign creator flags uniformly at random
    n_creators   = max(1, int(round(num_users * author_frac)))
    creator_mask = np.zeros(num_users, dtype=bool)
    creator_mask[rng.choice(num_users, size=n_creators, replace=False)] = True

    # --- Generate users ------------------------------------------------------
    used_usernames: set[str] = set()
    users: list[dict] = []

    for i in range(num_users):
        is_creator = bool(creator_mask[i])

        # Country
        country_id = country_ids[rng.choice(len(country_ids), p=country_probs)]
        locale     = _SAFE_LOCALES.get(country_id, "en_US")
        fake       = _get_faker(locale)
        sfx        = country_sfx.get(country_id, "xx")

        # Username — retry up to 15 times for uniqueness
        username: str | None = None
        for _ in range(15):
            pat       = patterns[int(rng.integers(0, len(patterns)))]
            candidate = build_username(fake, topic_words, pat, adjectives, sfx, rng)
            if candidate and candidate not in used_usernames:
                username = candidate
                used_usernames.add(username)
                break
        if username is None:
            username = f"user_{i:05d}"
            used_usernames.add(username)

        # Followers — creators get a 5-20× boost on top of the base draw
        base_fans = float(rng.lognormal(fans_mu, fans_sigma))
        if is_creator:
            multiplier = float(rng.uniform(5.0, 20.0))
            followers  = int(clamp(base_fans * multiplier, 0, 10_000_000))
        else:
            followers  = int(clamp(base_fans, 0, 10_000_000))

        following = int(clamp(float(rng.lognormal(follow_mu, follow_sig)), 0, 10_000))

        # Account age → joined_at
        reg_days  = int(clamp(float(rng.normal(reg_mean, reg_std)), 7.0, 2000.0))
        joined_at = sim_end - timedelta(days=reg_days)
        if joined_at < sim_start:
            joined_at = sim_start

        # like_count — total likes received (creators only; updated precisely after videos)
        like_count = (
            int(clamp(float(rng.lognormal(9.0, 1.5)), 0, 50_000_000))
            if is_creator else 0
        )

        # average_watch_time in seconds (KuaiRec median ≈ 7 s)
        average_watch_time = round(
            float(clamp(float(rng.normal(7.0, 3.0)), 1.0, 60.0)), 2
        )

        # last_login — placeholder; overwritten by generate_sessions().
        # Must be >= joined_at even before sessions are generated.
        last_login = max(joined_at, sim_end - timedelta(days=int(rng.integers(0, 30))))

        users.append({
            "user_id":            str(uuid.uuid4()),
            "username":           username,
            "joined_at":          joined_at,
            "followers":          followers,
            "following":          following,
            "like_count":         like_count,
            "average_watch_time": average_watch_time,
            "last_login":         last_login,
            "country_id":         country_id,
            "is_creator":         is_creator,
        })

    return users


# ---------------------------------------------------------------------------
# Follow graph generation
# ---------------------------------------------------------------------------

def generate_follows(
    users: list[dict],
    params: dict,
    distributions: dict,  # kept for future engagement score calibration
) -> list[dict]:
    """
    Build directed FOLLOWS edges using preferential attachment.

    Algorithm
    ---------
    For each user *u* (ordered by index):
      1. Determine target_following = min(u.following, n-1, max_following)
      2. Reserve ≥ min_creator_follow_fraction of slots for creators
         (selected with weight ∝ sqrt(followers+1))
      3. Fill remainder from all non-self users
         (also weighted by sqrt(followers+1))
      4. Assign engagement_score ∈ [0.35, 1.0] for creator follows,
         [0.1, 0.65] for non-creator follows.

    Also mutates each user dict's 'following' field to equal the actual
    outgoing edge count (capped by network size).

    Returns list of {follower_id, followee_id, engagement_score}.
    """
    rng = get_rng()
    n   = len(users)

    user_ids   = np.array([u["user_id"]   for u in users])
    followings = np.array([u["following"] for u in users], dtype=float)
    followers  = np.array([u["followers"] for u in users], dtype=float)
    is_creator = np.array([u["is_creator"] for u in users], dtype=bool)

    max_following    = params["social"]["max_following"]
    min_creator_frac = params["social"]["min_creator_follow_fraction"]
    eng_min          = params["social"]["engagement_score_min"]
    eng_max          = params["social"]["engagement_score_max"]

    # Preferential attachment weights — sqrt dampens billion-follower outliers
    attach_w    = np.sqrt(followers + 1.0)
    creator_idx = np.where(is_creator)[0]

    actual_following = np.zeros(n, dtype=int)
    follows: list[dict] = []

    for i in range(n):
        target = int(clamp(followings[i], 0.0, float(min(max_following, n - 1))))
        if target == 0:
            continue

        # All candidate indices except self
        all_except_self = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        creator_cands   = creator_idx[creator_idx != i]

        chosen: list[int] = []

        # Step 1 — creator quota
        n_creator_req = int(min(
            max(1, int(target * min_creator_frac)),
            len(creator_cands),
        ))
        if n_creator_req > 0 and len(creator_cands) > 0:
            cw   = attach_w[creator_cands]
            cw   = cw / cw.sum()
            picks = rng.choice(creator_cands, size=n_creator_req, replace=False, p=cw)
            chosen.extend(picks.tolist())

        # Step 2 — fill remainder from everyone not yet chosen
        remaining = target - len(chosen)
        if remaining > 0:
            chosen_set = set(chosen)
            avail = all_except_self[~np.isin(all_except_self, list(chosen_set))]
            k     = min(remaining, len(avail))
            if k > 0:
                aw   = attach_w[avail]
                aw   = aw / aw.sum()
                rest = rng.choice(avail, size=k, replace=False, p=aw)
                chosen.extend(rest.tolist())

        # Emit edges
        for j in chosen:
            followee_is_creator = bool(is_creator[j])
            eng = round(float(
                rng.uniform(0.35, eng_max) if followee_is_creator
                else rng.uniform(eng_min, 0.65)
            ), 3)
            follows.append({
                "follower_id":      str(user_ids[i]),
                "followee_id":      str(user_ids[j]),
                "engagement_score": eng,
            })
            actual_following[i] += 1

    # Sync 'following' on user dicts to actual edge count
    for i, user in enumerate(users):
        user["following"] = int(actual_following[i])

    return follows


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from generators.base import load_config, reset_rng

    cfg = load_config()
    reset_rng(cfg["seed"])

    params        = cfg
    taxonomy      = cfg
    distributions = cfg

    N = 100
    print(f"Generating {N} users…")
    users = generate_users(N, params, taxonomy, distributions)

    # Username uniqueness
    unames = [u["username"] for u in users]
    print(f"  Unique usernames : {len(set(unames))} / {N}")

    # Creator fraction
    n_creators = sum(u["is_creator"] for u in users)
    print(f"  Creators         : {n_creators} / {N}  ({n_creators/N:.1%})")

    # Country distribution
    from collections import Counter
    country_dist = Counter(u["country_id"] for u in users)
    print("  Country dist     :", dict(sorted(country_dist.items())))

    # Follower stats
    all_followers = [u["followers"] for u in users]
    creator_fans  = [u["followers"] for u in users if u["is_creator"]]
    regular_fans  = [u["followers"] for u in users if not u["is_creator"]]
    print(f"  Followers (all)  : median={int(np.median(all_followers)):,}  "
          f"max={max(all_followers):,}")
    if creator_fans:
        print(f"  Followers (creators): median={int(np.median(creator_fans)):,}  "
              f"mean={int(np.mean(creator_fans)):,}")
    if regular_fans:
        print(f"  Followers (regular): median={int(np.median(regular_fans)):,}  "
              f"mean={int(np.mean(regular_fans)):,}")

    # Sample usernames
    print("\n  Sample usernames:")
    for u in users[:12]:
        tag = "🎬" if u["is_creator"] else "👤"
        print(f"    {tag} @{u['username']:<28s} [{u['country_id']}]  "
              f"followers={u['followers']:>8,}")

    # Follow graph
    print(f"\nGenerating follow graph…")
    follows = generate_follows(users, params, distributions)
    actual_followings = [u["following"] for u in users]
    in_degrees = Counter(f["followee_id"] for f in follows)

    print(f"  Total FOLLOWS edges  : {len(follows):,}")
    print(f"  Avg following        : {np.mean(actual_followings):.1f}")
    print(f"  Max following        : {max(actual_followings)}")
    print(f"  Avg followers (graph): {np.mean(list(in_degrees.values())):.1f}")
    print(f"  Max followers (graph): {max(in_degrees.values()) if in_degrees else 0}")

    creator_ids = {u["user_id"] for u in users if u["is_creator"]}
    creator_follows = sum(1 for f in follows if f["followee_id"] in creator_ids)
    print(f"  Creator-targeted follows : {creator_follows/len(follows):.1%}  "
          f"(min required: {params['social']['min_creator_follow_fraction']:.0%})")

    eng_scores = [f["engagement_score"] for f in follows]
    print(f"  Engagement score range   : "
          f"[{min(eng_scores):.3f}, {max(eng_scores):.3f}]  "
          f"mean={np.mean(eng_scores):.3f}")

    print("\nAll users checks passed.")

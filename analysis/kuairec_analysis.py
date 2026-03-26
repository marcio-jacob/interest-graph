"""
KuaiRec 2.0 Distribution Analysis
===================================
Reads the 4 KuaiRec CSV files and computes all behavioral distributions
needed to generate realistic synthetic TikTok data.

Outputs:  config/distributions.yaml
"""

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = Path(
    "/home/spike/.cache/kagglehub/datasets"
    "/arashnic/kuairec-recommendation-system-data-density-100"
    "/versions/1/KuaiRec 2.0/data"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
OUT_FILE = CONFIG_DIR / "distributions.yaml"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pct(arr: np.ndarray, qs=(5, 10, 25, 50, 75, 90, 95, 99)) -> dict:
    """Return a dict of percentile_label → rounded float value."""
    vals = np.nanpercentile(arr, qs)
    return {f"p{q}": round(float(v), 6) for q, v in zip(qs, vals)}


def basic_stats(arr: np.ndarray) -> dict:
    """Return mean, std, min, max plus standard percentiles."""
    clean = arr[np.isfinite(arr)]
    return {
        "mean": round(float(np.mean(clean)), 6),
        "std": round(float(np.std(clean)), 6),
        "min": round(float(np.min(clean)), 6),
        "max": round(float(np.max(clean)), 6),
        **pct(clean),
    }


def histogram_as_lists(arr: np.ndarray, bins: int, lo: float, hi: float) -> dict:
    """
    Compute a fixed-range histogram and return edges + counts as plain lists
    (safe for YAML serialisation).
    """
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    return {
        "bin_edges": [round(float(e), 6) for e in edges],
        "counts": [int(c) for c in counts],
    }


# ---------------------------------------------------------------------------
# 1. big_matrix.csv  (12.5 M rows — read in chunks)
# ---------------------------------------------------------------------------
print("=" * 60)
print("1/4  Reading big_matrix.csv (chunked, 12.5 M rows) …")

CHUNK = 1_000_000
BM_COLS = ["user_id", "video_id", "play_duration", "video_duration", "watch_ratio"]
BM_DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "play_duration": "float32",
    "video_duration": "float32",
    "watch_ratio": "float32",
}

watch_ratios: list[np.ndarray] = []
video_dur_ms: list[np.ndarray] = []
user_view_counts: dict[int, int] = {}
total_rows = 0

for chunk in pd.read_csv(
    DATASET_DIR / "big_matrix.csv",
    usecols=BM_COLS,
    dtype=BM_DTYPES,
    chunksize=CHUNK,
):
    total_rows += len(chunk)

    # watch_ratio: clip to [0, 3], drop NaN
    wr = chunk["watch_ratio"].values.astype("float32")
    wr = wr[np.isfinite(wr)]
    wr = np.clip(wr, 0.0, 3.0)
    watch_ratios.append(wr)

    # video_duration: already in milliseconds → convert to seconds
    vd = chunk["video_duration"].values.astype("float32")
    vd = vd[np.isfinite(vd) & (vd > 0)]
    video_dur_ms.append(vd / 1000.0)  # now in seconds

    # count views per user for mean-videos-per-user
    for uid, cnt in chunk["user_id"].value_counts().items():
        user_view_counts[int(uid)] = user_view_counts.get(int(uid), 0) + int(cnt)

    print(f"  … {total_rows:,} rows processed", end="\r")

print(f"\n  Total rows: {total_rows:,}")

watch_ratio_arr = np.concatenate(watch_ratios)
video_dur_arr = np.concatenate(video_dur_ms)

print(f"  watch_ratio samples: {len(watch_ratio_arr):,}")
print(f"  video_duration samples: {len(video_dur_arr):,}")

mean_videos_per_user = total_rows / len(user_view_counts)
print(f"  Distinct users: {len(user_view_counts):,}")
print(f"  Mean videos viewed per user: {mean_videos_per_user:.1f}")

# Statistics
watch_ratio_stats = basic_stats(watch_ratio_arr)
watch_ratio_hist = histogram_as_lists(watch_ratio_arr, bins=20, lo=0.0, hi=3.0)

# Video duration: clip at 600 s (10 min) for the histogram — long tail otherwise
video_dur_stats = basic_stats(video_dur_arr)
video_dur_clipped = np.clip(video_dur_arr, 1.0, 600.0)
video_dur_hist = histogram_as_lists(video_dur_clipped, bins=20, lo=0.0, hi=600.0)

# Fraction of all views where watch_ratio >= 0.8 (positive signal)
positive_signal_rate = float(np.mean(watch_ratio_arr >= 0.8))
skip_rate = float(np.mean(watch_ratio_arr < 0.15))

print(f"  Positive signal rate (≥0.8): {positive_signal_rate:.3f}")
print(f"  Skip rate (<0.15):            {skip_rate:.3f}")

# ---------------------------------------------------------------------------
# 2. user_features.csv  (7,176 rows)
# ---------------------------------------------------------------------------
print("\n2/4  Reading user_features.csv …")

UF_COLS = [
    "user_id", "follow_user_num", "fans_user_num",
    "friend_user_num", "register_days", "is_video_author",
]
UF_DTYPES = {
    "user_id": "int32",
    "follow_user_num": "float32",
    "fans_user_num": "float32",
    "friend_user_num": "float32",
    "register_days": "float32",
    "is_video_author": "float32",
}

uf = pd.read_csv(DATASET_DIR / "user_features.csv", usecols=UF_COLS, dtype=UF_DTYPES)
print(f"  Rows: {len(uf):,}")

follow_stats = basic_stats(uf["follow_user_num"].dropna().values)
fans_stats   = basic_stats(uf["fans_user_num"].dropna().values)
friend_stats = basic_stats(uf["friend_user_num"].dropna().values)
regdays_stats = basic_stats(uf["register_days"].dropna().values)

author_fraction = float(uf["is_video_author"].mean())
print(f"  Author fraction: {author_fraction:.3f}")
print(f"  follow_user_num  — mean: {follow_stats['mean']:.1f}  p90: {follow_stats['p90']:.1f}")
print(f"  fans_user_num    — mean: {fans_stats['mean']:.1f}  p90: {fans_stats['p90']:.1f}")

# Log-normal fit parameters (μ, σ of log distribution) — for sampling
def lognorm_params(arr: np.ndarray) -> dict:
    clean = arr[arr > 0]           # log requires positive values
    if len(clean) < 10:
        return {"mu": 0.0, "sigma": 1.0}
    log_vals = np.log(clean)
    return {
        "mu": round(float(np.mean(log_vals)), 6),
        "sigma": round(float(np.std(log_vals)), 6),
    }

follow_lognorm = lognorm_params(uf["follow_user_num"].dropna().values)
fans_lognorm   = lognorm_params(uf["fans_user_num"].dropna().values)

# ---------------------------------------------------------------------------
# 3. item_daily_features.csv  (343,341 rows — aggregate per video_id)
# ---------------------------------------------------------------------------
print("\n3/4  Reading item_daily_features.csv …")

IDF_COLS = [
    "video_id", "video_duration",
    "like_cnt", "play_cnt", "comment_cnt",
    "share_cnt", "download_cnt", "follow_cnt", "complete_play_cnt",
]
IDF_DTYPES = {col: "float32" for col in IDF_COLS}
IDF_DTYPES["video_id"] = "int32"

idf = pd.read_csv(DATASET_DIR / "item_daily_features.csv", usecols=IDF_COLS, dtype=IDF_DTYPES)
print(f"  Rows (daily): {len(idf):,}")

# Aggregate all daily rows per video
agg = idf.groupby("video_id")[
    ["like_cnt", "play_cnt", "comment_cnt", "share_cnt",
     "download_cnt", "follow_cnt", "complete_play_cnt", "video_duration"]
].sum().reset_index()

print(f"  Unique videos: {len(agg):,}")

# Filter: keep only videos with enough plays for reliable rate estimation
agg_filtered = agg[agg["play_cnt"] >= 10].copy()
print(f"  Videos with play_cnt ≥ 10: {len(agg_filtered):,}")

# Compute per-video rates
agg_filtered["like_rate"]     = agg_filtered["like_cnt"]          / agg_filtered["play_cnt"]
agg_filtered["comment_rate"]  = agg_filtered["comment_cnt"]       / agg_filtered["play_cnt"]
agg_filtered["share_rate"]    = agg_filtered["share_cnt"]         / agg_filtered["play_cnt"]
agg_filtered["download_rate"] = agg_filtered["download_cnt"]      / agg_filtered["play_cnt"]
agg_filtered["follow_rate"]   = agg_filtered["follow_cnt"]        / agg_filtered["play_cnt"]
agg_filtered["complete_rate"] = agg_filtered["complete_play_cnt"] / agg_filtered["play_cnt"]

def rate_stats(series: pd.Series) -> dict:
    """Stats for a rate column, clipped to [0, 1]."""
    arr = series.dropna().values.astype("float64")
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr[np.isfinite(arr)]
    return {
        "mean": round(float(np.mean(arr)), 6),
        "std":  round(float(np.std(arr)), 6),
        **{f"p{q}": round(float(np.percentile(arr, q)), 6)
           for q in (25, 50, 75, 90, 95)},
    }

rates = {
    "like_rate":     rate_stats(agg_filtered["like_rate"]),
    "comment_rate":  rate_stats(agg_filtered["comment_rate"]),
    "share_rate":    rate_stats(agg_filtered["share_rate"]),
    "download_rate": rate_stats(agg_filtered["download_rate"]),
    "follow_rate":   rate_stats(agg_filtered["follow_rate"]),
    "complete_rate": rate_stats(agg_filtered["complete_rate"]),
}

# Play count distribution (proxy for video popularity / view count)
play_cnt_stats = basic_stats(agg_filtered["play_cnt"].values)
play_lognorm   = lognorm_params(agg_filtered["play_cnt"].values)

for name, r in rates.items():
    print(f"  {name:16s} mean={r['mean']:.4f}  p50={r['p50']:.4f}  p90={r['p90']:.4f}")

# ---------------------------------------------------------------------------
# 4. social_network.csv  (472 rows)
# ---------------------------------------------------------------------------
print("\n4/4  Reading social_network.csv …")

sn = pd.read_csv(DATASET_DIR / "social_network.csv")
print(f"  Rows: {len(sn):,}")

def parse_friend_list(raw) -> list[int]:
    if pd.isna(raw):
        return []
    try:
        result = ast.literal_eval(str(raw))
        return result if isinstance(result, list) else []
    except Exception:
        return []

sn["friends"] = sn["friend_list"].apply(parse_friend_list)
sn["friend_count"] = sn["friends"].apply(len)

friends_arr = sn["friend_count"].values.astype("float64")
friends_stats = {
    "mean":    round(float(np.mean(friends_arr)), 6),
    "std":     round(float(np.std(friends_arr)), 6),
    "p50":     round(float(np.percentile(friends_arr, 50)), 6),
    "p90":     round(float(np.percentile(friends_arr, 90)), 6),
    "max":     round(float(np.max(friends_arr)), 6),
}
has_friend_fraction = float(np.mean(friends_arr > 0))

print(f"  friends_per_user — mean: {friends_stats['mean']:.2f}  p90: {friends_stats['p90']:.2f}")
print(f"  Fraction with ≥1 friend: {has_friend_fraction:.3f}")

# ---------------------------------------------------------------------------
# Build distributions.yaml
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Writing config/distributions.yaml …")

distributions = {
    # ------------------------------------------------------------------
    "watch_behavior": {
        "description": (
            "Derived from big_matrix.csv watch_ratio field. "
            "watch_ratio = play_duration / video_duration, clipped to [0, 3]. "
            "Values > 1 indicate replay/looping."
        ),
        "watch_ratio": {
            "stats": watch_ratio_stats,
            "histogram": watch_ratio_hist,
            "positive_signal_rate": round(positive_signal_rate, 6),
            "skip_rate": round(skip_rate, 6),
        },
        "video_duration_seconds": {
            "description": "Video duration in seconds (raw field was milliseconds).",
            "stats": video_dur_stats,
            "histogram_clipped_600s": video_dur_hist,
        },
        "mean_videos_viewed_per_user": round(mean_videos_per_user, 2),
        "distinct_users_in_big_matrix": len(user_view_counts),
    },

    # ------------------------------------------------------------------
    "user_social": {
        "description": "Derived from user_features.csv.",
        "follow_user_num": {
            "description": "How many users this user follows.",
            "stats": follow_stats,
            "lognormal_fit": follow_lognorm,
        },
        "fans_user_num": {
            "description": "How many followers this user has.",
            "stats": fans_stats,
            "lognormal_fit": fans_lognorm,
        },
        "friend_user_num": {
            "description": "Mutual follows (friends).",
            "stats": friend_stats,
        },
        "register_days": {
            "description": "Days since account registration.",
            "stats": regdays_stats,
        },
        "video_author_fraction": round(author_fraction, 6),
        "total_users_in_user_features": len(uf),
    },

    # ------------------------------------------------------------------
    "video_engagement": {
        "description": (
            "Derived from item_daily_features.csv. "
            "Aggregated per video (sum of all daily rows), "
            "filtered to videos with play_cnt >= 10."
        ),
        "filtered_video_count": int(len(agg_filtered)),
        "like_rate":     rates["like_rate"],
        "comment_rate":  rates["comment_rate"],
        "share_rate":    rates["share_rate"],
        "download_rate": rates["download_rate"],
        "follow_rate":   rates["follow_rate"],
        "complete_rate": rates["complete_rate"],
        "play_count": {
            "description": "Total plays per video (all-time, aggregated).",
            "stats": play_cnt_stats,
            "lognormal_fit": play_lognorm,
        },
    },

    # ------------------------------------------------------------------
    "social_graph": {
        "description": "Derived from social_network.csv (mutual follow / friend graph).",
        "friends_per_user": friends_stats,
        "has_friend_fraction": round(has_friend_fraction, 6),
        "total_users_in_social_network": len(sn),
    },
}

with open(OUT_FILE, "w") as f:
    yaml.dump(
        distributions,
        f,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )

print(f"  Written → {OUT_FILE}")

# ---------------------------------------------------------------------------
# Quick sanity print
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  watch_ratio   p50={watch_ratio_stats['p50']:.3f}  "
      f"p90={watch_ratio_stats['p90']:.3f}  "
      f"skip_rate={skip_rate:.3f}  positive_rate={positive_signal_rate:.3f}")
print(f"  video_dur     p50={video_dur_stats['p50']:.1f}s  "
      f"p90={video_dur_stats['p90']:.1f}s  "
      f"mean={video_dur_stats['mean']:.1f}s")
print(f"  follow_count  mean={follow_stats['mean']:.1f}  "
      f"p90={follow_stats['p90']:.1f}")
print(f"  fans_count    mean={fans_stats['mean']:.1f}  "
      f"p90={fans_stats['p90']:.1f}")
print(f"  like_rate     mean={rates['like_rate']['mean']:.4f}  "
      f"p50={rates['like_rate']['p50']:.4f}")
print(f"  comment_rate  mean={rates['comment_rate']['mean']:.4f}  "
      f"p50={rates['comment_rate']['p50']:.4f}")
print(f"  complete_rate mean={rates['complete_rate']['mean']:.4f}  "
      f"p50={rates['complete_rate']['p50']:.4f}")
print(f"  friends/user  mean={friends_stats['mean']:.2f}  "
      f"p90={friends_stats['p90']:.2f}")
print()
print(f"  config/distributions.yaml  ({OUT_FILE.stat().st_size:,} bytes)")
print("Done.")

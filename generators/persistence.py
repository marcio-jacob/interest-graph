"""
generators/persistence.py
=========================
Save and load generated data as Parquet files in data/{scale}/.

Usage
-----
    from generators.persistence import save_dataset, load_dataset, dataset_exists

    if dataset_exists("large"):
        data = load_dataset("large")
    else:
        data = generate_everything(...)
        save_dataset("large", data)

Dataset layout
--------------
    data/{scale}/
        topics.parquet
        countries.parquet
        sounds.parquet
        hashtags.parquet
        entities.parquet
        entity_topic_links.parquet
        users.parquet
        follows.parquet
        sessions.parquet
        videos.parquet            ← descriptions may still be PLACEHOLDER
        video_hashtags.parquet
        video_entities.parquet
        video_sounds.parquet
        video_topics.parquet
        views.parquet
        likes.parquet
        skips.parquet
        reposts.parquet
        comment_stubs.parquet
        topic_interests.parquet
        entity_interests.parquet
        hashtag_interests.parquet
        comments.parquet          ← only exists after LLM/faker step
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from generators.base import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Keys that form the "core" dataset (Steps 1-5).
# comments.parquet is written separately after Step 6.
# ---------------------------------------------------------------------------
_CORE_FILES = [
    "topics", "countries", "sounds", "hashtags", "entities",
    "entity_topic_links",
    "users", "follows",
    "sessions",
    "videos",
    "video_hashtags", "video_entities", "video_sounds", "video_topics",
    "views", "likes", "skips", "reposts",
    "comment_stubs",
    "topic_interests", "entity_interests", "hashtag_interests",
]

# Columns that contain Python datetime objects and must be round-tripped
# through pandas Timestamp → datetime conversion.
_DATETIME_COLS: dict[str, list[str]] = {
    "users":         ["joined_at", "last_login"],
    "sessions":      ["start_date", "end_date"],
    "videos":        ["posted_at"],
    "comment_stubs": ["created_at"],
    "comments":      ["created_at"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _data_dir(scale: str) -> Path:
    return DATA_ROOT / scale


def _to_df(records: list[dict]) -> pd.DataFrame:
    """Convert list-of-dicts → DataFrame (empty list → empty DataFrame)."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _from_df(df: pd.DataFrame, name: str) -> list[dict]:
    """Convert DataFrame → list-of-dicts, restoring datetime columns."""
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    dt_cols = _DATETIME_COLS.get(name, [])
    for row in records:
        for col in dt_cols:
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, pd.Timestamp):
                row[col] = val.to_pydatetime()
            elif hasattr(val, "item"):          # numpy scalar fallback
                row[col] = pd.Timestamp(val).to_pydatetime()
    return records


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, converting object columns safely."""
    if df.empty:
        # Write an empty parquet so we know the file was intentionally empty
        df.to_parquet(path, index=False, engine="pyarrow")
        return
    df.to_parquet(path, index=False, engine="pyarrow")


def _read(path: Path, name: str) -> list[dict]:
    """Read a parquet file and return list-of-dicts."""
    df = pd.read_parquet(path, engine="pyarrow")
    return _from_df(df, name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dataset_exists(scale: str) -> bool:
    """Return True if all core parquet files exist for this scale."""
    d = _data_dir(scale)
    return all((d / f"{name}.parquet").exists() for name in _CORE_FILES)


def comments_exist(scale: str) -> bool:
    """Return True if the comments parquet (post-LLM) exists."""
    return (_data_dir(scale) / "comments.parquet").exists()


def descriptions_filled(scale: str) -> bool:
    """Return True if videos.parquet has no remaining PLACEHOLDER descriptions."""
    p = _data_dir(scale) / "videos.parquet"
    if not p.exists():
        return False
    df = pd.read_parquet(p, columns=["description"], engine="pyarrow")
    if df.empty:
        return True
    return not df["description"].str.startswith("PLACEHOLDER").any()


def save_dataset(scale: str, data: dict[str, list[dict]]) -> None:
    """
    Persist core generation output (Steps 1-5) to data/{scale}/.

    Parameters
    ----------
    scale : "small" | "medium" | "large"
    data  : dict with keys matching _CORE_FILES
    """
    d = _data_dir(scale)
    d.mkdir(parents=True, exist_ok=True)
    for name in _CORE_FILES:
        records = data.get(name, [])
        path = d / f"{name}.parquet"
        _write(_to_df(records), path)
    print(f"  Saved {len(_CORE_FILES)} parquet files → {d}")


def save_videos(scale: str, videos: list[dict]) -> None:
    """Overwrite videos.parquet after descriptions have been filled."""
    d = _data_dir(scale)
    d.mkdir(parents=True, exist_ok=True)
    _write(_to_df(videos), d / "videos.parquet")


def save_comments(scale: str, comments: list[dict]) -> None:
    """Write comments.parquet after LLM / faker generation (Step 6)."""
    d = _data_dir(scale)
    d.mkdir(parents=True, exist_ok=True)
    _write(_to_df(comments), d / "comments.parquet")


def load_dataset(scale: str) -> dict[str, list[dict]]:
    """
    Load core generation output from data/{scale}/.

    Returns the same dict shape that main.py expects.
    Raises FileNotFoundError if any core file is missing.
    """
    d = _data_dir(scale)
    data: dict[str, list[dict]] = {}
    for name in _CORE_FILES:
        path = d / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet file: {path}")
        data[name] = _read(path, name)
    return data


def load_comments(scale: str) -> list[dict]:
    """Load comments.parquet; returns [] if it doesn't exist yet."""
    p = _data_dir(scale) / "comments.parquet"
    if not p.exists():
        return []
    return _read(p, "comments")

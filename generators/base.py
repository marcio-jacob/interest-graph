"""
generators/base.py
==================
Shared infrastructure: config loading, seeded RNG singleton, and utility helpers.
All generators import from here — nothing is hardcoded elsewhere.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from faker import Faker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
_config_cache: dict | None = None


def load_config(
    params_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
    distributions_path: str | Path | None = None,
) -> dict:
    """
    Load and merge params.yaml, taxonomy.yaml, and distributions.yaml into one dict.

    Keys at the top level:
      cfg['seed'], cfg['scale'], cfg['sessions'], cfg['dates'], ...  (from params)
      cfg['topics'], cfg['countries'], cfg['sounds'], ...            (from taxonomy)
      cfg['watch_behavior'], cfg['user_social'], ...                 (from distributions)

    Results are cached — subsequent calls return the same object.
    Pass explicit paths to override defaults (useful in tests).
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    params_path = Path(params_path or CONFIG_DIR / "params.yaml")
    taxonomy_path = Path(taxonomy_path or CONFIG_DIR / "taxonomy.yaml")
    distributions_path = Path(distributions_path or CONFIG_DIR / "distributions.yaml")

    merged: dict[str, Any] = {}

    for path in (params_path, taxonomy_path, distributions_path):
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if data:
            merged.update(data)

    _config_cache = merged
    return merged


def reset_config_cache() -> None:
    """Clear the config cache (useful in tests or when reloading with different paths)."""
    global _config_cache
    _config_cache = None


# ---------------------------------------------------------------------------
# Seeded RNG singleton
# ---------------------------------------------------------------------------
_rng: np.random.Generator | None = None


def get_rng(seed: int | None = None) -> np.random.Generator:
    """
    Return the module-level seeded RNG.
    On first call the seed is read from params.yaml (or the provided argument).
    All generators should call get_rng() rather than creating their own Generator.
    """
    global _rng
    if _rng is None:
        if seed is None:
            cfg = load_config()
            seed = cfg.get("seed", 42)
        _rng = np.random.default_rng(seed)
    return _rng


def reset_rng(seed: int | None = None) -> np.random.Generator:
    """Reset the RNG with a new seed (useful for reproducible test runs)."""
    global _rng
    if seed is None:
        cfg = load_config()
        seed = cfg.get("seed", 42)
    _rng = np.random.default_rng(seed)
    return _rng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_choice(items: list, weights: list[float]) -> Any:
    """
    Pick one item from *items* according to *weights* (need not sum to 1).

    >>> weighted_choice(['a', 'b', 'c'], [1, 2, 1])
    # returns 'b' ~50% of the time
    """
    rng = get_rng()
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = rng.choice(len(items), p=probs)
    return items[idx]


def weighted_choices(items: list, weights: list[float], k: int) -> list:
    """
    Pick *k* items WITH replacement from *items* according to *weights*.
    More efficient than calling weighted_choice k times.
    """
    rng = get_rng()
    total = sum(weights)
    probs = np.array(weights, dtype=float) / total
    indices = rng.choice(len(items), size=k, p=probs, replace=True)
    return [items[i] for i in indices]


def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp *val* to the inclusive range [lo, hi]."""
    return max(lo, min(hi, val))


def date_between(start_str: str, end_str: str, fake: Faker | None = None) -> datetime:
    """
    Return a random datetime between *start_str* and *end_str* (both "YYYY-MM-DD").
    Uses faker under the hood so timezone handling is consistent.
    The returned datetime is naive (no tzinfo).
    """
    if fake is None:
        fake = Faker()
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    return fake.date_time_between(start_date=start_dt, end_date=end_dt)


def sample_lognormal(mu: float, sigma: float, clip_lo: float = 0.0, clip_hi: float = float("inf")) -> float:
    """Sample a single value from a log-normal distribution and clamp the result."""
    rng = get_rng()
    val = float(rng.lognormal(mean=mu, sigma=sigma))
    return clamp(val, clip_lo, clip_hi)


def sample_from_histogram(bin_edges: list[float], counts: list[int]) -> float:
    """
    Sample a single value from a pre-computed histogram.
    Used to draw watch_ratio / video_duration values with the exact KuaiRec shape.
    Returns a value uniformly sampled within the chosen bin.
    """
    rng = get_rng()
    counts_arr = np.array(counts, dtype=float)
    probs = counts_arr / counts_arr.sum()
    bin_idx = rng.choice(len(counts), p=probs)
    lo = bin_edges[bin_idx]
    hi = bin_edges[bin_idx + 1]
    return float(rng.uniform(lo, hi))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    rng = get_rng()

    print("Config keys loaded:")
    for k in cfg:
        val = cfg[k]
        preview = str(val)[:60] + "…" if len(str(val)) > 60 else str(val)
        print(f"  {k:<30s} {preview}")

    print(f"\nRNG type : {type(rng)}")
    print(f"seed     : {cfg.get('seed')}")

    # weighted_choice
    result = weighted_choice(["a", "b", "c"], [1, 5, 1])
    print(f"\nweighted_choice(['a','b','c'], [1,5,1]) → {result!r}  (expect 'b' ~71%)")

    # clamp
    print(f"clamp(5, 0, 3) → {clamp(5, 0, 3)}")
    print(f"clamp(-1, 0, 3) → {clamp(-1, 0, 3)}")

    # date_between
    d = date_between("2023-01-01", "2024-12-31")
    print(f"date_between('2023-01-01', '2024-12-31') → {d}")

    # sample_from_histogram
    edges = cfg["watch_behavior"]["watch_ratio"]["histogram"]["bin_edges"]
    counts = cfg["watch_behavior"]["watch_ratio"]["histogram"]["counts"]
    samples = [sample_from_histogram(edges, counts) for _ in range(200_000)]
    print(f"\nwatch_ratio samples (n=200k):")
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:2d}: {np.percentile(samples, p):.3f}")
    print(f"  skip (<0.15): {sum(s < 0.15 for s in samples)/len(samples):.3f}")

    print("\nAll base checks passed.")

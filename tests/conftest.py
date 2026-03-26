"""
Shared fixtures for all test modules.

Every test function starts with a clean state:
  - config cache cleared
  - RNG reset to seed 42
This makes every test deterministic and order-independent.
"""

import pytest

from generators.base import load_config, reset_config_cache, reset_rng
from generators.users import generate_follows, generate_users
from generators.sessions import generate_sessions


# ---------------------------------------------------------------------------
# Autouse reset — runs before every test function
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """Ensure a clean, seeded environment for every test."""
    reset_config_cache()
    reset_rng(42)           # explicit seed — does NOT call load_config()
    yield
    reset_config_cache()    # leave a clean cache after the test too


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    """Merged config dict (params + taxonomy + distributions)."""
    return load_config()


# ---------------------------------------------------------------------------
# User fixtures  (independent: each creates its own users from a fresh RNG)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_users(cfg):
    """10 users — structural / type checks."""
    return generate_users(10, cfg, cfg, cfg)


@pytest.fixture
def medium_users(cfg):
    """50 users — statistical checks that need more data points."""
    return generate_users(50, cfg, cfg, cfg)


# ---------------------------------------------------------------------------
# Combined fixtures (avoid mutation ambiguity by owning their own user list)
# ---------------------------------------------------------------------------

@pytest.fixture
def users_and_follows(cfg):
    """
    50 fresh users + their FOLLOWS edges.
    generate_follows mutates user['following'] in place, so users and edges
    are always consistent inside this fixture.
    """
    users = generate_users(50, cfg, cfg, cfg)
    follows = generate_follows(users, cfg, cfg)
    return users, follows


@pytest.fixture
def sessions_and_users(cfg):
    """
    50 fresh users + (sessions, user_last_session_map).
    generate_sessions mutates user['last_login'] in place.
    """
    users = generate_users(50, cfg, cfg, cfg)
    sessions, last_map = generate_sessions(users, cfg)
    return sessions, last_map, users

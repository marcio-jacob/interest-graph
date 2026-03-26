"""
Unit tests for generators/sessions.py
========================================
Covers: generate_sessions — structure, temporal ordering,
        PREVIOUS_SESSION chaining, date constraints, weekend bias,
        last_login mutation, edge cases.
"""

from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pytest

from generators.base import reset_rng
from generators.sessions import generate_sessions
from generators.users import generate_users


REQUIRED_SESSION_KEYS = {"session_id", "user_id", "start_date", "end_date", "prev_session_id"}


# ===========================================================================
# generate_sessions
# ===========================================================================

class TestGenerateSessions:

    # ── Return types & structure ────────────────────────────────────────────

    def test_returns_tuple_of_list_and_dict(self, sessions_and_users):
        sessions, last_map, users = sessions_and_users
        assert isinstance(sessions, list)
        assert isinstance(last_map, dict)

    def test_required_keys_present(self, sessions_and_users):
        sessions, _, _ = sessions_and_users
        for s in sessions:
            assert REQUIRED_SESSION_KEYS == set(s.keys()), \
                f"Key mismatch: {set(s.keys()) ^ REQUIRED_SESSION_KEYS}"

    def test_session_ids_are_unique(self, sessions_and_users):
        sessions, _, _ = sessions_and_users
        ids = [s["session_id"] for s in sessions]
        assert len(set(ids)) == len(ids), "Duplicate session_id detected"

    def test_all_users_covered(self, sessions_and_users):
        sessions, last_map, users = sessions_and_users
        assert len(last_map) == len(users), \
            "Some users are missing from user_last_session_map"

    # ── Temporal ordering ───────────────────────────────────────────────────

    def test_sessions_ordered_by_start_date_per_user(self, sessions_and_users):
        sessions, _, _ = sessions_and_users
        by_user: dict[str, list] = defaultdict(list)
        for s in sessions:
            by_user[s["user_id"]].append(s)
        for uid, sess_list in by_user.items():
            starts = [s["start_date"] for s in sess_list]
            assert starts == sorted(starts), \
                f"Sessions not time-ordered for user {uid}"

    def test_end_date_after_start_date(self, sessions_and_users):
        sessions, _, _ = sessions_and_users
        for s in sessions:
            assert s["end_date"] > s["start_date"], \
                f"end_date <= start_date for session {s['session_id']}"

    def test_sessions_within_sim_range(self, sessions_and_users, cfg):
        sessions, _, _ = sessions_and_users
        sim_start = datetime.strptime(cfg["dates"]["sim_start"], "%Y-%m-%d")
        sim_end   = datetime.strptime(cfg["dates"]["sim_end"],   "%Y-%m-%d")
        for s in sessions:
            assert s["start_date"] >= sim_start, \
                f"start_date {s['start_date']} before sim_start"
            assert s["end_date"]   <= sim_end, \
                f"end_date {s['end_date']} after sim_end"

    def test_sessions_not_before_user_joined(self, sessions_and_users):
        sessions, _, users = sessions_and_users
        uid_to_joined = {u["user_id"]: u["joined_at"] for u in users}
        for s in sessions:
            joined = uid_to_joined[s["user_id"]]
            assert s["start_date"] >= joined, \
                f"Session starts before user joined: {s['start_date']} < {joined}"

    # ── PREVIOUS_SESSION chain ──────────────────────────────────────────────

    def test_first_session_has_no_prev(self, sessions_and_users):
        sessions, _, _ = sessions_and_users
        by_user: dict[str, list] = defaultdict(list)
        for s in sessions:
            by_user[s["user_id"]].append(s)
        for uid, sess_list in by_user.items():
            earliest = min(sess_list, key=lambda x: x["start_date"])
            assert earliest["prev_session_id"] is None, \
                f"First session of user {uid} has a prev_session_id"

    def test_non_first_sessions_have_valid_prev(self, sessions_and_users):
        sessions, _, _ = sessions_and_users
        session_ids = {s["session_id"] for s in sessions}
        by_user: dict[str, list] = defaultdict(list)
        for s in sessions:
            by_user[s["user_id"]].append(s)
        for uid, sess_list in by_user.items():
            sorted_sess = sorted(sess_list, key=lambda x: x["start_date"])
            for s in sorted_sess[1:]:               # skip the first
                assert s["prev_session_id"] is not None, \
                    f"Non-first session {s['session_id']} has no prev"
                assert s["prev_session_id"] in session_ids, \
                    f"prev_session_id {s['prev_session_id']} not in session set"

    def test_chain_points_to_immediate_predecessor(self, sessions_and_users):
        sessions, _, _ = sessions_and_users
        by_user: dict[str, list] = defaultdict(list)
        for s in sessions:
            by_user[s["user_id"]].append(s)
        id_map = {s["session_id"]: s for s in sessions}
        for uid, sess_list in by_user.items():
            sorted_sess = sorted(sess_list, key=lambda x: x["start_date"])
            for i, s in enumerate(sorted_sess[1:], start=1):
                prev = id_map[s["prev_session_id"]]
                assert prev["session_id"] == sorted_sess[i - 1]["session_id"], \
                    f"Chain broken for user {uid}"

    # ── last_session_map & last_login mutation ──────────────────────────────

    def test_last_session_map_keys_are_user_ids(self, sessions_and_users):
        _, last_map, users = sessions_and_users
        user_ids = {u["user_id"] for u in users}
        for uid in last_map:
            assert uid in user_ids

    def test_last_session_map_values_are_actual_last_sessions(self, sessions_and_users):
        sessions, last_map, _ = sessions_and_users
        by_user: dict[str, list] = defaultdict(list)
        for s in sessions:
            by_user[s["user_id"]].append(s)
        for uid, sess_id in last_map.items():
            last_actual = max(by_user[uid], key=lambda x: x["start_date"])
            assert last_actual["session_id"] == sess_id, \
                f"last_session_map for {uid} points to wrong session"

    def test_user_last_login_updated_to_last_session_end(self, sessions_and_users):
        sessions, last_map, users = sessions_and_users
        sess_by_id = {s["session_id"]: s for s in sessions}
        for user in users:
            uid = user["user_id"]
            if uid in last_map:
                expected_login = sess_by_id[last_map[uid]]["end_date"]
                assert user["last_login"] == expected_login, \
                    f"last_login mismatch for {user['username']}"

    # ── Statistical properties ──────────────────────────────────────────────

    def test_session_count_per_user_within_configured_range(self, sessions_and_users, cfg):
        """
        Each user gets at least 1 and at most s_max sessions.
        The floor is 1 (not s_min) because the exponential gap model can exit
        early when a user joined close to sim_end, producing fewer sessions
        than the target drawn from the log-normal before the loop ran out of time.
        """
        sessions, _, users = sessions_and_users
        s_max = cfg["sessions"]["sessions_per_user_max"]
        counts = Counter(s["user_id"] for s in sessions)
        for uid, cnt in counts.items():
            assert 1 <= cnt <= s_max, \
                f"User {uid} has {cnt} sessions, expected [1, {s_max}]"

    def test_weekend_sessions_longer_than_weekday(self, cfg):
        """Weekend sessions should have a higher mean duration than weekday."""
        reset_rng(42)
        users = generate_users(100, cfg, cfg, cfg)
        sessions, _ = generate_sessions(users, cfg)

        weekend_mins, weekday_mins = [], []
        for s in sessions:
            dur = (s["end_date"] - s["start_date"]).total_seconds() / 60.0
            if s["start_date"].weekday() >= 5:
                weekend_mins.append(dur)
            else:
                weekday_mins.append(dur)

        assert weekend_mins, "No weekend sessions found"
        assert weekday_mins, "No weekday sessions found"
        assert np.mean(weekend_mins) > np.mean(weekday_mins), (
            f"Weekend mean {np.mean(weekend_mins):.1f} min should be > "
            f"weekday mean {np.mean(weekday_mins):.1f} min"
        )

    def test_session_duration_within_configured_bounds(self, sessions_and_users, cfg):
        sessions, _, _ = sessions_and_users
        dur_min = cfg["sessions"]["session_duration_minutes_min"]
        dur_max = cfg["sessions"]["session_duration_minutes_max"]
        for s in sessions:
            dur = (s["end_date"] - s["start_date"]).total_seconds() / 60.0
            # allow 30 % over dur_max (weekend bonus) and small epsilon under dur_min
            assert dur_min * 0.9 <= dur <= dur_max * 1.35, \
                f"Duration {dur:.1f} min outside expected range"

    def test_mean_sessions_per_user_reasonable(self, cfg):
        """Mean sessions per user should land in the log-normal bulk [3, 25]."""
        reset_rng(42)
        users = generate_users(100, cfg, cfg, cfg)
        sessions, _ = generate_sessions(users, cfg)
        counts = Counter(s["user_id"] for s in sessions)
        mean_count = np.mean(list(counts.values()))
        assert 3 <= mean_count <= 25, \
            f"Mean sessions/user = {mean_count:.1f}, expected [3, 25]"

    # ── Edge cases ──────────────────────────────────────────────────────────

    def test_empty_users_returns_empty(self, cfg):
        sessions, last_map = generate_sessions([], cfg)
        assert sessions == []
        assert last_map == {}

    def test_single_user_gets_at_least_one_session(self, cfg):
        reset_rng(42)
        users = generate_users(1, cfg, cfg, cfg)
        sessions, last_map = generate_sessions(users, cfg)
        assert len(sessions) >= 1
        assert users[0]["user_id"] in last_map

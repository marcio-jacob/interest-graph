"""
generators/sessions.py
=======================
Generate UserSession nodes with temporal ordering and PREVIOUS_SESSION chaining.

Public API
----------
generate_sessions(users, params) → (list[dict], dict[str, str])
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from statistics import mean

import numpy as np

from generators.base import clamp, get_rng


def generate_sessions(
    users: list[dict],
    params: dict,
) -> tuple[list[dict], dict[str, str]]:
    """
    Generate session nodes for every user.

    Session timing model
    --------------------
    Sessions arrive as a Poisson process (exponential inter-arrival times)
    starting from the user's joined_at date through sim_end.
    Weekend sessions draw from a wider duration distribution.

    Parameters
    ----------
    users  : list of user dicts (must have 'user_id', 'joined_at')
    params : merged config dict (reads params['sessions'] and params['dates'])

    Returns
    -------
    sessions          : list of session dicts
                        keys: session_id, user_id, start_date, end_date,
                              prev_session_id (None for first session)
    user_last_session : {user_id → session_id} mapping for LAST_SESSION rels
                        Also mutates user['last_login'] to last session's end_date.
    """
    rng = get_rng()

    s_cfg    = params["sessions"]
    mu       = s_cfg["sessions_per_user_lognormal_mu"]
    sigma    = s_cfg["sessions_per_user_lognormal_sigma"]
    s_min    = s_cfg["sessions_per_user_min"]
    s_max    = s_cfg["sessions_per_user_max"]
    dur_min  = float(s_cfg["session_duration_minutes_min"])
    dur_max  = float(s_cfg["session_duration_minutes_max"])

    sim_end = datetime.strptime(params["dates"]["sim_end"], "%Y-%m-%d")

    all_sessions: list[dict]       = []
    user_last_session: dict[str, str] = {}

    for user in users:
        joined_at      = user["joined_at"]
        available_days = max(1.0, (sim_end - joined_at).total_seconds() / 86_400.0)

        # ── Number of sessions ──────────────────────────────────────────────
        n_sessions = int(clamp(float(rng.lognormal(mu, sigma)), s_min, s_max))

        # ── Gap model ───────────────────────────────────────────────────────
        # Mean days between sessions, floored at 0.5 day to avoid bunching.
        gap_mean = max(0.5, available_days / (n_sessions + 1))

        # ── First session start ─────────────────────────────────────────────
        first_offset = float(rng.exponential(max(0.5, gap_mean * 0.5)))
        current_dt   = joined_at + timedelta(days=first_offset)

        user_sessions: list[dict] = []

        for _ in range(n_sessions):
            if current_dt >= sim_end:
                break

            start_date = current_dt
            is_weekend = start_date.weekday() >= 5  # 5=Saturday, 6=Sunday

            # Weekend sessions skew longer
            if is_weekend:
                duration_minutes = float(rng.uniform(dur_min * 1.3, dur_max))
            else:
                duration_minutes = float(rng.uniform(dur_min, dur_max * 0.85))

            end_date = min(
                start_date + timedelta(minutes=duration_minutes),
                sim_end,
            )

            session_id   = str(uuid.uuid4())
            prev_sess_id = user_sessions[-1]["session_id"] if user_sessions else None

            user_sessions.append({
                "session_id":      session_id,
                "user_id":         user["user_id"],
                "start_date":      start_date,
                "end_date":        end_date,
                "prev_session_id": prev_sess_id,
            })

            # Advance cursor: session ends, then exponential gap before next
            gap_days   = float(rng.exponential(gap_mean))
            current_dt = end_date + timedelta(days=gap_days)

        # ── Update user metadata ────────────────────────────────────────────
        if user_sessions:
            last = user_sessions[-1]
            user_last_session[user["user_id"]] = last["session_id"]
            user["last_login"] = last["end_date"]   # overwrite placeholder

        all_sessions.extend(user_sessions)

    return all_sessions, user_last_session


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from collections import Counter, defaultdict

    from generators.base import load_config, reset_rng
    from generators.users import generate_users

    cfg = load_config()
    reset_rng(cfg["seed"])

    N = 100
    print(f"Generating {N} users + sessions…")
    users    = generate_users(N, cfg, cfg, cfg)
    sessions, user_last_map = generate_sessions(users, cfg)

    print(f"  Total sessions          : {len(sessions):,}")
    print(f"  Users with sessions     : {len(user_last_map)} / {N}")

    # Sessions per user
    per_user = Counter(s["user_id"] for s in sessions)
    counts   = list(per_user.values())
    print(f"  Sessions/user  min={min(counts)}  max={max(counts)}  "
          f"mean={mean(counts):.1f}  "
          f"median={sorted(counts)[len(counts)//2]}")

    # Verify temporal ordering per user (start_date strictly non-decreasing)
    by_user: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        by_user[s["user_id"]].append(s)
    for uid, sess_list in by_user.items():
        ordered = sorted(sess_list, key=lambda x: x["start_date"])
        assert sess_list == ordered, f"Sessions out of order for user {uid}"
    print("  Temporal ordering       : OK (all users)")

    # Verify PREVIOUS_SESSION chain
    for uid, sess_list in by_user.items():
        # Build id→session map
        id_map = {s["session_id"]: s for s in sess_list}
        for s in sess_list:
            prev_id = s["prev_session_id"]
            if prev_id is None:
                # Must be the first session (index 0 after sorting)
                sorted_ids = [x["session_id"] for x in
                              sorted(sess_list, key=lambda x: x["start_date"])]
                assert s["session_id"] == sorted_ids[0], \
                    f"Non-first session has no prev_session_id: {uid}"
            else:
                assert prev_id in id_map, f"prev_session_id not found: {prev_id}"
    print("  PREVIOUS_SESSION chain  : OK")

    # Weekend vs weekday duration
    weekend_durs  = []
    weekday_durs  = []
    for s in sessions:
        mins = (s["end_date"] - s["start_date"]).total_seconds() / 60.0
        if s["start_date"].weekday() >= 5:
            weekend_durs.append(mins)
        else:
            weekday_durs.append(mins)
    wd_mean  = mean(weekend_durs)  if weekend_durs  else 0
    wkd_mean = mean(weekday_durs)  if weekday_durs  else 0
    print(f"  Weekend avg duration    : {wd_mean:.1f} min  "
          f"({len(weekend_durs)} sessions)")
    print(f"  Weekday avg duration    : {wkd_mean:.1f} min  "
          f"({len(weekday_durs)} sessions)")
    assert wd_mean >= wkd_mean * 0.9, \
        "Expected weekend sessions to be at least as long as weekday"
    print("  Weekend bias            : OK")

    # last_login updated on user
    for user in users:
        uid = user["user_id"]
        if uid in user_last_map:
            last_sess = next(s for s in sessions if s["session_id"] == user_last_map[uid])
            assert user["last_login"] == last_sess["end_date"], \
                f"last_login mismatch for {uid}"
    print("  last_login sync         : OK")

    # Show sample sessions for one user
    sample_uid   = users[0]["user_id"]
    sample_sess  = sorted(
        [s for s in sessions if s["user_id"] == sample_uid],
        key=lambda x: x["start_date"],
    )
    print(f"\n  Sessions for user[0] (@{users[0]['username']}):")
    for s in sample_sess[:6]:
        dur = (s["end_date"] - s["start_date"]).total_seconds() / 60
        day = s["start_date"].strftime("%Y-%m-%d %a")
        print(f"    {day}  {dur:5.1f} min  prev={'✓' if s['prev_session_id'] else '—'}")
    if len(sample_sess) > 6:
        print(f"    … and {len(sample_sess)-6} more")

    print("\nAll session checks passed.")

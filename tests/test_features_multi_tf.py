"""Regression tests for ``core/features/multi_tf.py::_w1_close_slope_sign``.

Guards against the within-period W1 lookahead bug discovered by Arc 8 +
Arc 10 v3.0.2 audits:

The W1 panel is left-labelled (``label='left', closed='left'`` on ``W-MON``
freq), so the W1 bar for week N has its index timestamp at the Monday
00:00 UTC of week N, BUT its ``close_bid`` / ``close_ask`` columns hold
the END-OF-WEEK close (Sunday 23:59 UTC of week N). The pre-fix producer
used ``pd.merge_asof(direction='backward', allow_exact_matches=False)``
which, at any H4 timestamp strictly after Monday 00:00 of week N, picks
the SAME week's W1 bar — leaking the eventual Sunday close every
mid-week H4 bar. Convention-independent (UTC and EET both affected;
that's why the 2026-05 audit's State A classification missed it).

The fix replaces ``merge_asof`` with the canonical
``core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)``
utility — semantics: at LTF timestamp t, pick the most recent W1 bar
whose END (next W1 bar's start) is ``<= t``. Mid-week N, week N's bar
is NOT fully closed; the lookup returns week N-1's W1 instead.

Tests:
  1. ``test_w1_close_slope_sign_no_lookahead_within_week`` — at a
     mid-week H4 timestamp, the slope sign reflects only PRIOR weeks,
     never the current week's eventual Sunday close. Pre-fix this test
     fails; post-fix passes.
  2. ``test_w1_close_slope_sign_lag_correctness`` — hand-computed
     expected values at multiple H4 timestamps match the producer.
  3. ``test_w1_close_slope_sign_canonical_alignment`` — static source
     guard: the producer must use ``get_htf_value_at`` and must not use
     ``merge_asof``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from core.features.multi_tf import _w1_close_slope_sign
from core.sim.panel import Panel

# ── Synthetic W1 / H4 panel construction ─────────────────────────────


def _build_w1_panel(weekly_closes: list[float], start_monday: str = "2024-01-01") -> Panel:
    """Build a synthetic W1 Panel keyed at Monday 00:00 UTC.

    ``weekly_closes[i]`` is the END-OF-WEEK close for week ``i`` — this
    mirrors the production aggregator's ``label='left'`` schema where
    the W1 bar at index ``Monday[i]`` carries week ``i``'s closing OHLC
    in its columns.
    """
    monday = pd.Timestamp(start_monday, tz="UTC")
    idx = pd.date_range(monday, periods=len(weekly_closes), freq="W-MON", tz="UTC")
    idx.name = "timestamp_utc"
    spread = 0.00010  # 1 pip half-width
    closes = np.array(weekly_closes, dtype=float)
    df = pd.DataFrame(
        {
            "open_bid": closes - spread,
            "high_bid": closes - spread,
            "low_bid": closes - spread,
            "close_bid": closes - spread,
            "open_ask": closes + spread,
            "high_ask": closes + spread,
            "low_ask": closes + spread,
            "close_ask": closes + spread,
            "volume": np.ones(len(closes), dtype=np.int64),
            "spread_close": np.full(len(closes), 2 * spread, dtype=float),
            "bid_ask_data_quality": ["ok"] * len(closes),
        },
        index=idx,
    )
    return Panel(pair_dfs={"EURUSD": df}, tf="W1")


def _build_h4_frame(start_monday: str, n_weeks: int) -> pd.DataFrame:
    """Build an H4 ``pair_df`` index spanning ``n_weeks`` from ``start_monday``."""
    monday = pd.Timestamp(start_monday, tz="UTC")
    n_bars = n_weeks * 7 * 6  # 7 days × 6 H4 bars/day
    idx = pd.date_range(monday, periods=n_bars, freq="4h", tz="UTC")
    idx.name = "timestamp_utc"
    df = pd.DataFrame(index=idx)
    df.attrs["pair"] = "EURUSD"
    return df


def _attach_w1_aux(h4_frame: pd.DataFrame, w1_panel: Panel) -> Panel:
    """Build an H4 Panel and attach W1 to ``.aux`` (the convention used by
    the engine — see ``scripts/arc_7/run_arc_7.py``).
    """
    h4_panel = Panel(pair_dfs={"EURUSD": _h4_bidask(h4_frame.index)}, tf="H4")
    # Panel is a frozen dataclass; the engine uses object.__setattr__ to
    # bolt on an ``aux`` dict. Mirror that pattern in tests.
    object.__setattr__(h4_panel, "aux", {"w1": w1_panel})
    return h4_panel


def _h4_bidask(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Trivial H4 OHLC frame matching the schema Panel expects."""
    n = len(idx)
    closes = np.full(n, 1.10, dtype=float)
    spread = 0.00010
    df = pd.DataFrame(
        {
            "open_bid": closes - spread,
            "high_bid": closes - spread,
            "low_bid": closes - spread,
            "close_bid": closes - spread,
            "open_ask": closes + spread,
            "high_ask": closes + spread,
            "low_ask": closes + spread,
            "close_ask": closes + spread,
            "volume": np.ones(n, dtype=np.int64),
            "spread_close": np.full(n, 2 * spread, dtype=float),
            "bid_ask_data_quality": ["ok"] * n,
        },
        index=idx,
    )
    return df


# ── Test 1: no within-week lookahead ─────────────────────────────────


def test_w1_close_slope_sign_no_lookahead_within_week() -> None:
    """At any H4 timestamp mid-week N, w1_close_slope_sign must use only
    fully-closed W1 bars (week N-1 + week N-2), never the eventual close
    of week N itself.

    Setup: weekly closes are chosen so the prior-vs-prior-prior slope
    has the OPPOSITE sign to the current-vs-prior slope. Pre-fix
    ``merge_asof(direction='backward', allow_exact_matches=False)`` at
    a mid-week-N H4 timestamp picks week N's W1 bar (with its eventual
    Sunday close), producing the wrong sign. Post-fix
    ``get_htf_value_at(..., require_fully_closed=True)`` picks week
    N-1's W1 bar, producing the correct sign.

    Weekly closes:
        week 0 = 1.10
        week 1 = 1.20  → slope(1→2): +1, slope(0→1): +1
        week 2 = 1.00  → slope(2→3): +1, slope(1→2): -1  ← divergent
        week 3 = 1.25  → slope(3→4): -1, slope(2→3): +1  ← divergent
        week 4 = 1.05  → slope(4→5): +1, slope(3→4): -1  ← divergent
        week 5 = 1.30
    """
    weekly_closes = [1.10, 1.20, 1.00, 1.25, 1.05, 1.30]
    w1_panel = _build_w1_panel(weekly_closes, start_monday="2024-01-01")
    h4_frame = _build_h4_frame("2024-01-01", n_weeks=6)
    panel = _attach_w1_aux(h4_frame, w1_panel)

    out = _w1_close_slope_sign(h4_frame, panel=panel)
    assert isinstance(out, pd.Series)
    assert len(out) == len(h4_frame)

    # H4 bars at Wednesday 12:00 UTC of weeks 2, 3, 4 — each mid-week,
    # where the buggy merge_asof picks the same week's eventual close
    # and the canonical lookup picks the prior week's close.
    # Week N Monday-00:00 + 2 days 12 hours = Wednesday 12:00.
    monday0 = pd.Timestamp("2024-01-01", tz="UTC")
    wednesdays = {
        n: monday0 + pd.Timedelta(weeks=n, days=2, hours=12) for n in range(2, 5)
    }

    # Expected (canonical) sign at mid-week N = sign(close[N-1] - close[N-2]):
    #   wed week 2 → sign(1.20 - 1.10) = +1
    #   wed week 3 → sign(1.00 - 1.20) = -1
    #   wed week 4 → sign(1.25 - 1.00) = +1
    expected_canonical = {2: +1.0, 3: -1.0, 4: +1.0}
    # The buggy (pre-fix) sign at mid-week N = sign(close[N] - close[N-1]),
    # i.e. leaking the current week's eventual close. Asserted to NOT
    # equal the canonical sign — proves the test is non-vacuous.
    buggy_signs = {2: -1.0, 3: +1.0, 4: -1.0}

    for n, ts in wednesdays.items():
        # ts must exist in the H4 index
        assert ts in out.index, f"week {n} Wednesday {ts} missing from H4 index"
        v = out.loc[ts]
        assert v == expected_canonical[n], (
            f"Week {n} Wed: expected sign {expected_canonical[n]:+.1f} "
            f"(close[{n-1}]-close[{n-2}]); got {v:+.1f}. Within-week "
            f"lookahead would have produced {buggy_signs[n]:+.1f}."
        )
        # Sanity: the buggy and canonical answers really do diverge for
        # these timestamps. Otherwise the test would be vacuous.
        assert expected_canonical[n] != buggy_signs[n]


# ── Test 2: lag correctness at hand-computed timestamps ──────────────


def test_w1_close_slope_sign_lag_correctness() -> None:
    """Hand-computed expected values for multiple H4 timestamps match the
    producer output. Covers Monday boundary, mid-week, and Friday close
    timestamps to pin down the strict-prior semantics at edges.

    Weekly closes (monotone-up to make slope sign trivial):
        week 0 = 1.00 (warmup; lag-2 NaN at week 0/1)
        week 1 = 1.10
        week 2 = 1.20
        week 3 = 1.30
        week 4 = 1.40
    """
    weekly_closes = [1.00, 1.10, 1.20, 1.30, 1.40]
    w1_panel = _build_w1_panel(weekly_closes, start_monday="2024-01-01")
    h4_frame = _build_h4_frame("2024-01-01", n_weeks=5)
    panel = _attach_w1_aux(h4_frame, w1_panel)
    out = _w1_close_slope_sign(h4_frame, panel=panel)

    monday0 = pd.Timestamp("2024-01-01", tz="UTC")

    # During week N (N ≥ 2), the canonical fully-closed prior W1 is week
    # N-1, whose pre-shifted slope value is close[N-1] - close[N-2].
    # For monotone-up closes, all signs are +1.
    cases = [
        # (timestamp, expected_value, description)
        # ── Monday 00:00 of week 1: only week 0 is fully closed; its
        #    slope (close[0] - close[-1]) is NaN → np.sign(NaN) = NaN.
        (monday0 + pd.Timedelta(weeks=1), float("nan"), "Mon 00 week 1 (warmup)"),
        # ── Mid-week N: slope sign at that H4 reflects close[N-1] -
        #    close[N-2]. All monotone-up → +1.
        (monday0 + pd.Timedelta(weeks=2, days=2, hours=12), +1.0, "Wed 12 week 2"),
        (monday0 + pd.Timedelta(weeks=3, days=4, hours=16), +1.0, "Fri 16 week 3"),
        # ── Monday 00:00 of week 4 (exact boundary): the bar that
        #    closes at this instant is week 3's W1. Per "fully closed at
        #    ts iff bar_end[k] <= ts", week 3 (ending at Mon 00 of wk 4)
        #    IS fully closed. Slope value = sign(close[3] - close[2]).
        (monday0 + pd.Timedelta(weeks=4), +1.0, "Mon 00 week 4 (boundary)"),
    ]

    for ts, expected, desc in cases:
        assert ts in out.index, f"{desc}: ts {ts} missing from H4 index"
        actual = out.loc[ts]
        if np.isnan(expected):
            assert np.isnan(actual), (
                f"{desc}: expected NaN (warmup), got {actual!r}"
            )
        else:
            assert actual == expected, (
                f"{desc}: expected sign {expected:+.1f}, got {actual:+.1f}"
            )


# ── Test 3: canonical alignment — source-level guard ─────────────────


def test_w1_close_slope_sign_canonical_alignment() -> None:
    """Static guard: the producer must use the canonical HTF utility
    and must not use ``merge_asof`` (the pre-fix idiom that introduced
    the within-week lookahead).
    """
    source = inspect.getsource(_w1_close_slope_sign)
    assert "merge_asof" not in source, (
        "Lookahead-prone merge_asof must not be used; canonical "
        "get_htf_value_at(..., require_fully_closed=True) is the only "
        "approved HTF lookup pattern in this module."
    )
    assert "get_htf_value_at" in source, (
        "Producer must use the canonical core.signals.htf_alignment "
        "utility for HTF lookups."
    )

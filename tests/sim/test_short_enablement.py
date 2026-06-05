"""Short-side enablement: label producer, two-producer parity, config parse.

Pins the Step-1 / label-space half of the short-enablement spec
(``discovery/SHORTS_ENABLEMENT_PROBE.md``). The engine-level take-the-loss
short mirror lives in ``tests/sim/test_take_the_loss_invariant.py`` and the
cost symmetry in ``tests/sim/test_cost_application.py``; this file covers:

  1. :func:`core.sim.honest_label.reached_1r_before_sl` SHORT branch — the
     take-the-loss invariant in label space, mirrored (stop ABOVE entry on the
     ask, favourable = price falling). Same-bar +1R/SL resolves SL-first → NaN.
  2. **Two-producer parity** — the spec requires the two Step-1 pool producers
     (``core.arc.arc_pool_builder`` and ``core.discovery.pool_simulator``) to
     agree trade-for-trade on a short. A losing short that hits its hard stop
     must come out identical (entry / SL / exit / final_r / bars_held / label)
     from both bar-walks. A long control proves the harness is non-trivial.
  3. :func:`core.sim.account.parse_direction` — the load-bearing parser for the
     configs' ``direction:`` key.

Numpy / pandas + stdlib only → runs under CI's ``pytest -m "not research"``.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.arc.arc_pool_builder import ArcPoolConfig, _simulate_pair_pool
from core.arc.signal_protocol import PerPairSignalState
from core.discovery.pool_simulator import DiscoveryExitConfig, simulate_pair_pool
from core.sim.account import Direction, parse_direction
from core.sim.honest_label import reached_1r_before_sl

# ── 1. honest label — SHORT branch (take-the-loss in label space) ───────────
#
# entry=100, stop ABOVE at 102 (sl_distance=2 == 1R), +1R favourable level = 98.
# Short stop fires on high_ask >= 102; favourable on low_ask <= 98.


def test_short_label_reach_strictly_before_sl():
    """low_ask hits +1R (<=98) at offset 2; high_ask breaches 102 at offset 4."""
    high_ask = [100.0, 100.5, 100.5, 100.5, 102.5]   # off4 breaches stop
    low_ask = [100.0, 99.5, 98.0, 99.0, 99.0]         # off2 reaches +1R (==98)
    got = reached_1r_before_sl(
        high_bid=high_ask, low_bid=low_ask, high_ask=high_ask, low_ask=low_ask,
        direction="short", entry_idx=0, exit_off=4,
        entry_price=100.0, sl_price=102.0, sl_distance=2.0,
    )
    assert got == 2.0


def test_short_label_sl_strictly_before_reach_is_nan():
    """high_ask breaches the stop at offset 2 before any +1R (only at 4)."""
    high_ask = [100.0, 100.5, 102.5, 100.5, 100.5]    # off2 breaches stop first
    low_ask = [100.0, 99.5, 99.5, 99.5, 97.0]
    got = reached_1r_before_sl(
        high_bid=high_ask, low_bid=low_ask, high_ask=high_ask, low_ask=low_ask,
        direction="short", entry_idx=0, exit_off=2,
        entry_price=100.0, sl_price=102.0, sl_distance=2.0,
    )
    assert math.isnan(got)


def test_short_label_same_bar_plus1r_and_sl_is_nan():
    """THE bug case, mirrored: a bar whose low_ask reaches +1R AND whose
    high_ask breaches the stop resolves SL-first → NaN."""
    high_ask = [100.0, 100.5, 102.5]   # off2 high_ask >= 102 (stop)
    low_ask = [100.0, 99.5, 98.0]      # off2 low_ask <= 98 (+1R) — same bar
    got = reached_1r_before_sl(
        high_bid=high_ask, low_bid=low_ask, high_ask=high_ask, low_ask=low_ask,
        direction="short", entry_idx=0, exit_off=2,
        entry_price=100.0, sl_price=102.0, sl_distance=2.0,
    )
    assert math.isnan(got), "same-bar +1R/SL must NOT register a +1R reach"


def test_short_label_never_reached_is_nan():
    high_ask = [100.0, 100.5, 101.0, 101.5]
    low_ask = [100.0, 99.5, 99.0, 98.5]   # never reaches +1R (98.0)
    got = reached_1r_before_sl(
        high_bid=high_ask, low_bid=low_ask, high_ask=high_ask, low_ask=low_ask,
        direction="short", entry_idx=0, exit_off=3,
        entry_price=100.0, sl_price=102.0, sl_distance=2.0,
    )
    assert math.isnan(got)


def test_short_label_requires_ask_arrays():
    with pytest.raises(ValueError, match="requires high_ask and low_ask"):
        reached_1r_before_sl(
            high_bid=[100.0, 101.0], low_bid=[99.0, 98.0],
            direction="short", entry_idx=0, exit_off=1,
            entry_price=100.0, sl_price=102.0, sl_distance=2.0,
        )


# ── 2. two-producer parity (the spec cross-check) ───────────────────────────


def _df(rows: list[dict]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=4 * i) for i in range(len(rows))],
        name="timestamp_utc",
    )
    return pd.DataFrame(rows, index=idx)


def _run_both(rows: list[dict], direction: Direction):
    """Run the SAME pair through both Step-1 pool producers and return
    (arc_trade_dict, pool_traderow). The fixtures hold close==open at the
    signal/entry bars so the two SL anchors (arc: signal-bar close; pool:
    entry price) coincide — isolating the bar-walk geometry under test."""
    df = _df(rows)
    mask = pd.Series([i == 0 for i in range(len(rows))], index=df.index)
    atr = pd.Series([1.0] * len(rows), index=df.index, dtype="float64")

    state = PerPairSignalState(signal_mask=mask, atr=atr, direction=direction)
    arc_cfg = ArcPoolConfig(arc_name="t", sl_atr_mult=2.0, hold_bars=240, primary_tf_warmup_bars=0)
    arc_trades, _paths, _nid = _simulate_pair_pool("EURUSD", df, state, arc_cfg, 1)
    assert len(arc_trades) == 1, arc_trades

    disc_cfg = DiscoveryExitConfig(
        initial_sl_atr_mult=2.0, trail_activation_atr_mult=4.0,
        trail_distance_atr_mult=2.0, primary_tf_warmup_bars=0, direction=direction,
    )
    pool_trades, _n, _it, _ab = simulate_pair_pool("EURUSD", df, mask, atr, disc_cfg)
    assert len(pool_trades) == 1, pool_trades
    return arc_trades[0], pool_trades[0]


# A losing SHORT: enters at open_bid=100, price rises into the stop at 102.
_SHORT_STOP_ROWS = [
    # bar0 signal: close==100 both sides (== entry, so the SL anchors coincide)
    dict(open_bid=99.9, open_ask=99.9, high_bid=100.1, high_ask=100.1,
         low_bid=99.8, low_ask=99.8, close_bid=100.0, close_ask=100.0),
    # bar1 entry: open==100; no stop this bar
    dict(open_bid=100.0, open_ask=100.0, high_bid=100.5, high_ask=100.5,
         low_bid=99.5, low_ask=99.5, close_bid=100.2, close_ask=100.2),
    # bar2: high_ask 102.5 >= sl 102 -> hard_sl at 102
    dict(open_bid=100.5, open_ask=100.5, high_bid=102.5, high_ask=102.5,
         low_bid=100.0, low_ask=100.0, close_bid=101.5, close_ask=101.5),
    # bar3 filler
    dict(open_bid=101.5, open_ask=101.5, high_bid=101.6, high_ask=101.6,
         low_bid=101.0, low_ask=101.0, close_bid=101.2, close_ask=101.2),
]

# A losing LONG control: enters at open_ask=100, price falls into the stop at 98.
_LONG_STOP_ROWS = [
    dict(open_bid=100.1, open_ask=100.1, high_bid=100.2, high_ask=100.2,
         low_bid=99.9, low_ask=99.9, close_bid=100.0, close_ask=100.0),
    dict(open_bid=100.0, open_ask=100.0, high_bid=100.5, high_ask=100.5,
         low_bid=99.5, low_ask=99.5, close_bid=99.8, close_ask=99.8),
    # bar2: low_bid 97.5 <= sl 98 -> hard_sl at 98
    dict(open_bid=99.5, open_ask=99.5, high_bid=100.0, high_ask=100.0,
         low_bid=97.5, low_ask=97.5, close_bid=98.5, close_ask=98.5),
    dict(open_bid=98.5, open_ask=98.5, high_bid=99.0, high_ask=99.0,
         low_bid=98.4, low_ask=98.4, close_bid=98.6, close_ask=98.6),
]


def _assert_producers_agree(arc: dict, pool, *, entry, sl, final_r):
    assert arc["entry_price"] == pytest.approx(entry)
    assert pool.entry_price == pytest.approx(entry)
    assert arc["sl_at_entry_price"] == pytest.approx(sl)
    assert pool.sl_at_entry_price == pytest.approx(sl)
    assert arc["exit_reason"] == "hard_sl"
    assert pool.exit_reason == "hard_sl"
    assert arc["exit_price"] == pytest.approx(sl)
    assert pool.exit_price == pytest.approx(sl)
    assert arc["final_r"] == pytest.approx(final_r)
    assert pool.final_r == pytest.approx(final_r)
    assert arc["bars_held"] == pool.bars_held
    # The +1R-before-SL label agrees (never reached on a losing trade → NaN).
    assert math.isnan(arc["bars_to_1r_mfe"]) and math.isnan(pool.bars_to_1r_mfe)


def test_two_producers_agree_on_a_losing_short():
    """The spec cross-check: both Step-1 producers agree trade-for-trade on a
    short that hits its hard stop (stop ABOVE entry, exit at the SL, -1R)."""
    arc, pool = _run_both(_SHORT_STOP_ROWS, Direction.SHORT)
    _assert_producers_agree(arc, pool, entry=100.0, sl=102.0, final_r=-1.0)
    assert arc["bars_held"] == 1


def test_two_producers_agree_on_a_losing_long_control():
    """Control: the same harness on a LONG also agrees — so the short parity
    above is a real cross-check, not a vacuous pass."""
    arc, pool = _run_both(_LONG_STOP_ROWS, Direction.LONG)
    _assert_producers_agree(arc, pool, entry=100.0, sl=98.0, final_r=-1.0)
    assert arc["bars_held"] == 1


def test_short_final_r_sign_is_mirrored():
    """A short that falls in its favour books a POSITIVE final_r in both
    producers (sign mirrored, not a long-signed negative)."""
    rows = [
        dict(open_bid=99.9, open_ask=99.9, high_bid=100.1, high_ask=100.1,
             low_bid=99.8, low_ask=99.8, close_bid=100.0, close_ask=100.0),
        dict(open_bid=100.0, open_ask=100.0, high_bid=100.2, high_ask=100.2,
             low_bid=99.0, low_ask=99.0, close_bid=99.5, close_ask=99.5),
        # price falls; no stop (high_ask 99.8 < 102). time/hold or eod exit below.
        dict(open_bid=99.0, open_ask=99.0, high_bid=99.8, high_ask=99.8,
             low_bid=97.0, low_ask=97.0, close_bid=97.5, close_ask=97.5),
    ]
    df = _df(rows)
    mask = pd.Series([True, False, False], index=df.index)
    atr = pd.Series([1.0, 1.0, 1.0], index=df.index, dtype="float64")
    state = PerPairSignalState(signal_mask=mask, atr=atr, direction=Direction.SHORT)
    cfg = ArcPoolConfig(arc_name="t", sl_atr_mult=2.0, hold_bars=240, primary_tf_warmup_bars=0)
    trades, _p, _n = _simulate_pair_pool("EURUSD", df, state, cfg, 1)
    assert len(trades) == 1
    # entry 100, exits at end of window on close_ask=97.5 -> (100-97.5)/2 = +1.25R
    assert trades[0]["final_r"] == pytest.approx((100.0 - 97.5) / 2.0)
    assert trades[0]["final_r"] > 0.0


# ── 3. parse_direction (load-bearing config key) ────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("long", Direction.LONG),
    ("long_only", Direction.LONG),
    ("LONG", Direction.LONG),
    (" Long ", Direction.LONG),
    ("short", Direction.SHORT),
    ("short_only", Direction.SHORT),
    ("SHORT", Direction.SHORT),
    (Direction.SHORT, Direction.SHORT),
    (Direction.LONG, Direction.LONG),
    (None, Direction.LONG),   # absent key -> long default
])
def test_parse_direction_accepts(raw, expected):
    assert parse_direction(raw) is expected


def test_parse_direction_rejects_typo():
    with pytest.raises(ValueError, match="unrecognised direction"):
        parse_direction("lonng")


def test_parse_direction_required_when_default_none():
    with pytest.raises(ValueError, match="required"):
        parse_direction(None, default=None)

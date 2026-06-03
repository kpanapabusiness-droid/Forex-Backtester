"""CI-gated regression: the take-the-loss invariant in LABEL space.

The engine pins take-the-loss at the trade level
(``tests/sim/test_take_the_loss_invariant.py``): a stop breach at or before
the +1R partial bar is −1R, and a same-bar +1R/SL trade resolves SL-first —
ambiguity never resolves to a win. This file pins the SAME invariant for the
heavy_ml training label, closing ``HONEST_ENGINE_SWEEP.md`` Part D:

  * **FLAG-D1 (provenance).** ``bars_to_1r_mfe`` had no in-tree producer.
    It now has one — :func:`core.sim.honest_label.reached_1r_before_sl` —
    exercised here directly AND end-to-end through the real discovery pool
    simulator, so the label's bar indices come from the SL-honest forward
    walk and nothing else.
  * **FLAG-D2 (tie-break bug).** The label compared ``exit_reason`` against
    ``"sl"`` while the simulators emit ``"hard_sl"``, so a same-bar
    (+1R-high AND SL-low) trade was labelled a WIN. It is now a LOSS.

Design constraint: this module imports ONLY numpy / pandas (+ the
standard-library producer and the sklearn-free
``core.heavy_ml_probe.labels``), so it runs under CI's minimal
``pytest -m "not research"`` env exactly like the engine fixture — unlike
``tests/heavy_ml_probe/*`` which ``importorskip("flaml")`` and are skipped
there.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.discovery.pool_simulator import DiscoveryExitConfig, simulate_pair_pool
from core.heavy_ml_probe.labels import (
    build_meta_label_target,
    build_survival_target,
)
from core.sim.honest_label import (
    ONE_R,
    SL_EXIT_REASONS,
    is_stop_loss_exit,
    reached_1r_before_sl,
)

# ── synthetic-OHLC helpers (mirror tests/discovery/test_pool_simulator) ──


def _ohlc(rows: list[dict]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [pd.Timestamp(r["ts"], tz="UTC") for r in rows], name="timestamp_utc"
    )
    return pd.DataFrame(
        {
            "open_bid": [r.get("open_bid", r["open_ask"]) for r in rows],
            "high_bid": [r["high_bid"] for r in rows],
            "low_bid": [r["low_bid"] for r in rows],
            "close_bid": [r["close_bid"] for r in rows],
            "open_ask": [r["open_ask"] for r in rows],
            "high_ask": [r.get("high_ask", r["high_bid"]) for r in rows],
            "low_ask": [r.get("low_ask", r["low_bid"]) for r in rows],
            "close_ask": [r.get("close_ask", r["close_bid"]) for r in rows],
            "volume": [1.0] * len(rows),
            "spread_close": [
                r.get("close_ask", r["close_bid"]) - r["close_bid"] for r in rows
            ],
            "bid_ask_data_quality": ["ok"] * len(rows),
        },
        index=idx,
    )


def _cfg() -> DiscoveryExitConfig:
    # sl_mult=2, atr=1 → sl_distance = 2.0 = 1R; +1R level = entry + 2.0.
    # trail arms at close >= entry + 4xATR (deliberately out of reach in
    # these fixtures so only SL / +1R drive the outcome).
    return DiscoveryExitConfig(
        initial_sl_atr_mult=2.0,
        trail_activation_atr_mult=4.0,
        trail_distance_atr_mult=2.0,
        primary_tf_warmup_bars=0,
    )


def _sig(idx: pd.DatetimeIndex) -> pd.Series:
    arr = np.zeros(len(idx), dtype=bool)
    arr[0] = True  # signal at bar 0 → entry at bar 1 (offset 0)
    return pd.Series(arr, index=idx)


def _atr(idx: pd.DatetimeIndex, value: float = 1.0) -> pd.Series:
    return pd.Series([value] * len(idx), index=idx, dtype="float64")


def _one_trade(df: pd.DataFrame):
    trades, _, _, _ = simulate_pair_pool("EURUSD", df, _sig(df.index), _atr(df.index), _cfg())
    assert len(trades) == 1
    return trades[0]


def _meta_label(bars_to_1r_mfe, bars_held, exit_reason, final_r=0.0) -> int:
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": bars_to_1r_mfe,
        "bars_held": float(bars_held),
        "exit_reason": exit_reason,
        "final_r": final_r,
    }])
    return int(build_meta_label_target(pool)[0])


def _survival_event(bars_to_1r_mfe, bars_held, exit_reason) -> int:
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": bars_to_1r_mfe,
        "bars_held": float(bars_held),
        "exit_reason": exit_reason,
    }])
    _, event = build_survival_target(pool)
    return int(event[0])


# ── 1. Producer unit (the honest bars_to_1r_mfe walk) ───────────────────


def test_producer_reaches_strictly_before_sl():
    """High reaches +1R at offset 2; stop breaches at offset 4 → offset 2."""
    high = [100.0, 101.0, 102.0, 101.0, 101.0]   # off2 high == entry+1R
    low = [99.5, 99.5, 99.5, 99.5, 97.5]          # off4 low <= sl
    got = reached_1r_before_sl(
        high_bid=high, low_bid=low, entry_idx=0, exit_off=4,
        entry_price=100.0, sl_price=98.0, sl_distance=2.0,
    )
    assert got == 2.0


def test_producer_sl_strictly_before_reach_is_nan():
    """Stop breaches at offset 2 before any +1R (which only comes at 4)."""
    high = [100.0, 100.5, 100.5, 100.5, 103.0]
    low = [99.5, 99.5, 97.5, 99.5, 99.5]          # off2 low <= sl (first)
    got = reached_1r_before_sl(
        high_bid=high, low_bid=low, entry_idx=0, exit_off=2,
        entry_price=100.0, sl_price=98.0, sl_distance=2.0,
    )
    assert math.isnan(got)


def test_producer_same_bar_plus1r_and_sl_is_nan():
    """THE BUG CASE — a bar whose high reaches +1R AND whose low breaches
    the stop resolves SL-first: +1R was NOT reached first → NaN."""
    high = [100.0, 100.5, 102.5]   # off2 high >= entry+1R (102.0)
    low = [99.5, 99.5, 97.5]       # off2 low <= sl (98.0) — same bar
    got = reached_1r_before_sl(
        high_bid=high, low_bid=low, entry_idx=0, exit_off=2,
        entry_price=100.0, sl_price=98.0, sl_distance=2.0,
    )
    assert math.isnan(got), "same-bar +1R/SL must NOT register a +1R reach"


def test_producer_never_reached_is_nan():
    high = [100.0, 100.5, 101.0, 101.5]   # never hits +1R (102.0)
    low = [99.5, 99.5, 99.5, 99.5]        # never breaches sl
    got = reached_1r_before_sl(
        high_bid=high, low_bid=low, entry_idx=0, exit_off=3,
        entry_price=100.0, sl_price=98.0, sl_distance=2.0,
    )
    assert math.isnan(got)


def test_producer_reach_on_at_close_exit_bar_is_included():
    """+1R first reached ON the exit bar; the trade is open through it
    (at-close / intrabar exit) → the exit bar counts."""
    high = [100.0, 101.0, 102.0]   # off2 == exit bar reaches +1R
    low = [99.5, 99.5, 99.5]
    got = reached_1r_before_sl(
        high_bid=high, low_bid=low, entry_idx=0, exit_off=2,
        entry_price=100.0, sl_price=98.0, sl_distance=2.0,
        exit_at_bar_open=False,
    )
    assert got == 2.0


def test_producer_reach_on_open_fill_exit_bar_is_excluded():
    """Same bars, but the trade left at the OPEN of the exit bar (queued
    trail/time fill) → that bar's high is unreachable → NaN."""
    high = [100.0, 101.0, 102.0]
    low = [99.5, 99.5, 99.5]
    got = reached_1r_before_sl(
        high_bid=high, low_bid=low, entry_idx=0, exit_off=2,
        entry_price=100.0, sl_price=98.0, sl_distance=2.0,
        exit_at_bar_open=True,
    )
    assert math.isnan(got)


def test_producer_one_r_constant_is_unit():
    assert ONE_R == 1.0


# ── 2. Producer through the REAL simulator (engine bar indices) ─────────


def test_sim_same_bar_tie_nan_despite_raw_mfe_touching_1r():
    """End-to-end: a same-bar (+1R-high AND SL-low) trade exits hard_sl with
    raw mfe_r >= 1.0 (the high DID touch +1R intrabar) yet honest
    bars_to_1r_mfe is NaN — exactly the divergence the fix introduces."""
    df = _ohlc([
        {"ts": "2020-01-01T00", "open_ask": 99.0, "high_bid": 100.0, "low_bid": 98.5, "close_bid": 99.0},
        {"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 100.5, "low_bid": 99.5, "close_bid": 100.2},
        # off1: high 102.5 (>= +1R 102.0) AND low 97.5 (<= sl 98.0) — tie.
        {"ts": "2020-01-01T02", "open_ask": 100.1, "high_bid": 102.5, "low_bid": 97.5, "close_bid": 98.6},
    ])
    t = _one_trade(df)
    assert t.exit_reason == "hard_sl"
    assert t.mfe_r >= 1.0          # raw excursion touched +1R on the SL bar
    assert math.isnan(t.bars_to_1r_mfe)   # honest label says: not reached first
    # And the label built from this trade is a LOSS.
    assert _meta_label(t.bars_to_1r_mfe, t.bars_held, t.exit_reason, t.final_r) == 0


def test_sim_reach_before_sl_records_offset():
    """+1R at forward offset 1, stop at offset 2 → bars_to_1r_mfe == 1."""
    df = _ohlc([
        {"ts": "2020-01-01T00", "open_ask": 99.0, "high_bid": 100.0, "low_bid": 98.5, "close_bid": 99.0},
        {"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 100.5, "low_bid": 99.5, "close_bid": 100.2},
        # off1: high 102.5 reaches +1R, low 99.5 (no stop)
        {"ts": "2020-01-01T02", "open_ask": 100.1, "high_bid": 102.5, "low_bid": 99.5, "close_bid": 101.0},
        # off2: low 97.5 breaches stop
        {"ts": "2020-01-01T03", "open_ask": 101.0, "high_bid": 101.5, "low_bid": 97.5, "close_bid": 98.0},
    ])
    t = _one_trade(df)
    assert t.exit_reason == "hard_sl"
    assert t.bars_held == 2
    assert t.bars_to_1r_mfe == 1.0
    # Reached +1R strictly before the stop → meta label = win.
    assert _meta_label(t.bars_to_1r_mfe, t.bars_held, t.exit_reason, t.final_r) == 1


def test_sim_sl_before_reach_is_nan():
    """Stop at offset 1 before any +1R → NaN → loss."""
    df = _ohlc([
        {"ts": "2020-01-01T00", "open_ask": 99.0, "high_bid": 100.0, "low_bid": 98.5, "close_bid": 99.0},
        {"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 100.5, "low_bid": 99.5, "close_bid": 100.2},
        # off1: low 97.5 breaches stop; high 101.0 never reached +1R
        {"ts": "2020-01-01T02", "open_ask": 100.1, "high_bid": 101.0, "low_bid": 97.5, "close_bid": 98.0},
        # later +1R must NOT resurrect the trade
        {"ts": "2020-01-01T03", "open_ask": 98.0, "high_bid": 103.0, "low_bid": 97.9, "close_bid": 102.5},
    ])
    t = _one_trade(df)
    assert t.exit_reason == "hard_sl"
    assert t.bars_held == 1
    assert math.isnan(t.bars_to_1r_mfe)
    assert _meta_label(t.bars_to_1r_mfe, t.bars_held, t.exit_reason, t.final_r) == 0


# ── 3. The REAL label functions (sklearn-free import) ───────────────────


def test_meta_label_reach_before_sl_is_win():
    assert _meta_label(2.0, 10.0, "hard_sl", final_r=-1.0) == 1


def test_meta_label_sl_before_reach_is_loss():
    assert _meta_label(float("nan"), 10.0, "hard_sl", final_r=-1.0) == 0


def test_meta_label_same_bar_hard_sl_is_loss():
    """The dispatch bug case: bars_to_1r_mfe == bars_held with a hard_sl
    exit MUST label 0 (LOSS), not 1."""
    assert _meta_label(5.0, 5.0, "hard_sl", final_r=-1.0) == 0


def test_meta_label_same_bar_non_adverse_is_win():
    """+1R on the same bar as a benign time exit is still a +1R reach."""
    assert _meta_label(5.0, 5.0, "time_exit", final_r=1.0) == 1


def test_survival_event_mirrors_meta_label():
    assert _survival_event(2.0, 10.0, "hard_sl") == 1     # reached before
    assert _survival_event(float("nan"), 10.0, "hard_sl") == 0  # never
    assert _survival_event(5.0, 5.0, "hard_sl") == 0      # same-bar → censored
    assert _survival_event(5.0, 5.0, "time_exit") == 1    # same-bar benign


# ── 4. Old-vs-new: prove the bug is fixed at the contract boundary ──────


def test_same_bar_hard_sl_old_vs_new_label():
    """OLD logic compared exit_reason against the literal "sl"; the
    simulators emit "hard_sl", so the same-bar tie slipped through as a WIN.
    NEW logic normalises via is_stop_loss_exit → LOSS."""
    old_is_sl = ("hard_sl".lower() == "sl")          # the buggy comparison
    new_is_sl = is_stop_loss_exit("hard_sl")          # the fix
    assert old_is_sl is False, "documents the old bug: hard_sl != sl"
    assert new_is_sl is True, "fix: hard_sl is recognised as a stop"

    # Old label on the same-bar tie would have been 1 (win); the real
    # builder now returns 0 (loss).
    old_label = 0 if old_is_sl else 1
    new_label = _meta_label(5.0, 5.0, "hard_sl", final_r=-1.0)
    assert old_label == 1          # bug: win
    assert new_label == 0          # fixed: loss
    assert old_label != new_label


def test_is_stop_loss_exit_recognises_all_stop_spellings():
    for r in ("sl", "SL", "Sl", "hard_sl", "HARD_SL", " hard_sl ", "stop_loss", "stop"):
        assert is_stop_loss_exit(r) is True, r
    for r in ("time_exit", "trail", "tp", "end_of_data", "", None):
        assert is_stop_loss_exit(r) is False, r
    # Every simulator stop spelling is covered by the canonical set.
    assert {"sl", "hard_sl"} <= SL_EXIT_REASONS


# ── 5. Full chain: simulator → pool → real label (no hand-set columns) ──


def test_chain_same_bar_tie_simulator_to_label_is_loss():
    """The complete provenance path: a same-bar tie trade produced by the
    real simulator, fed (verbatim) into the real meta-label builder, is a
    LOSS — no hand-authored bars_to_1r_mfe anywhere."""
    df = _ohlc([
        {"ts": "2020-01-01T00", "open_ask": 99.0, "high_bid": 100.0, "low_bid": 98.5, "close_bid": 99.0},
        {"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 100.5, "low_bid": 99.5, "close_bid": 100.2},
        {"ts": "2020-01-01T02", "open_ask": 100.1, "high_bid": 102.5, "low_bid": 97.5, "close_bid": 98.6},
    ])
    t = _one_trade(df)
    row = t.to_dict()
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": row["bars_to_1r_mfe"],
        "bars_held": float(row["bars_held"]),
        "exit_reason": row["exit_reason"],
        "final_r": row["final_r"],
    }])
    assert build_meta_label_target(pool).tolist() == [0]
    _, event = build_survival_target(pool)
    assert event.tolist() == [0]

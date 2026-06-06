"""Gate tests for the co-simulated portfolio equity curve (``core/wfo/cosim_book.py``).

These are CI-pinning, pure-function / synthetic-engine tests (no data corpus),
so they gate in CI (unmarked = not ``research``). They lock the four properties
the Item-E PR rests on:

  1. **Byte-identity to scoring alone** — a single-component book reproduces the
     component's own canonical FoldStats (engine run on a synthetic panel).
  2. **No-interaction == linear** — with caps off and disjoint currencies, the
     co-sim per-fold ROI equals the linear combiner's weighted ROI exactly. This
     is the anchor that proves the co-sim adds no return source.
  3. **Can only tighten** — the book exposure cap can only DROP entries (never
     add one), so when it drops a winner the co-sim ROI is < linear; and the
     shared daily-DD cap can only FAIL a fold the linear ROI judge passes.
  4. **Determinism** — two runs on identical inputs are byte-identical.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from core.runners._fold_stats_helpers import build_fold_stats_from_run
from core.sim.account import ClosedTrade, Direction
from core.sim.costs.model import CostModel
from core.sim.multipair_backtester import MultiPairBacktester, Order
from core.sim.panel import Panel
from core.wfo.cosim_book import (
    CoSimComponent,
    cosim_book_fold,
    mark_prices_from_panel,
)
from core.wfo.folds import Fold

# A fold whose OOS window is calendar-2010 (all synthetic trades live here).
FOLD_2010 = Fold(
    fold_id=1,
    is_start=date(2010, 1, 1),
    is_end=date(2009, 12, 31),  # empty IS — A1-style no-training
    oos_start=date(2010, 1, 1),
    oos_end=date(2010, 12, 31),
)

SB = 100_000.0
IDX = pd.date_range("2010-01-04 00:00", periods=48, freq="h", tz="UTC")  # 2 days, hourly


# ── synthetic ledger helpers ──────────────────────────────────────────────
def _ct(
    pid: int,
    pair: str,
    entry_i: int,
    exit_i: int,
    pnl: float,
    *,
    direction: Direction = Direction.LONG,
    entry_px: float = 1.0,
    size: float = 100_000.0,
    idx: pd.DatetimeIndex = IDX,
) -> ClosedTrade:
    """One closed trade with ``pnl`` realised at ``exit_i`` (zero spread → only
    commission/slippage cost). Prices are consistent: pnl = sign·(exit−entry)·size."""
    exit_px = entry_px + direction.sign * pnl / size
    return ClosedTrade(
        position_id=pid,
        pair=pair,
        direction=direction,
        entry_time=idx[entry_i],
        entry_price=entry_px,
        exit_time=idx[exit_i],
        exit_price=exit_px,
        size=size,
        pnl=pnl,
        exit_reason="take_profit",
        parent_position_id=None,
        entry_bid=entry_px,
        entry_ask=entry_px,
        exit_bid=exit_px,
        exit_ask=exit_px,
        sl_price=entry_px - 0.02,
    )


def _flat_marks(pairs, *, idx: pd.DatetimeIndex = IDX, px: float = 1.0) -> dict:
    return {p: pd.Series(px, index=idx, name="close_mid", dtype=float) for p in pairs}


def _component_own_roi(comp: CoSimComponent, weight: float = 1.0) -> float:
    """Score a component on its own through the co-sim (weight=1 book)."""
    return cosim_book_fold([comp], FOLD_2010, weights=[1.0], starting_balance=SB).roi_pct


# ── 1. byte-identity: single-component book == scoring it alone (real engine) ──
def _panel_one_sl_trade() -> Panel:
    """A hand-built EURUSD panel where one long position opens then stops out —
    a real ``MultiPairBacktester`` close, no data corpus."""
    idx = pd.date_range("2010-01-04 00:00", periods=12, freq="h", tz="UTC")
    # bid closes: flat 1.10, then a down bar that breaches a 100-pip stop.
    o = [1.10] * 6 + [1.08] * 6
    c = [1.10] * 5 + [1.08] * 7
    h = [1.10] * 6 + [1.08] * 6
    lo = [1.10] * 5 + [1.08] * 7  # bar 5 low dips to 1.08
    spread = 0.0001
    df = pd.DataFrame(
        {
            "open_bid": o,
            "high_bid": h,
            "low_bid": lo,
            "close_bid": c,
            "open_ask": [x + spread for x in o],
            "high_ask": [x + spread for x in h],
            "low_ask": [x + spread for x in lo],
            "close_ask": [x + spread for x in c],
            "volume": [1000] * 12,
            "spread_close": [spread] * 12,
            "bid_ask_data_quality": ["ok"] * 12,
        },
        index=idx,
    )
    return Panel.from_frames({"EURUSD": df}, tf="H4", boundary_convention="utc")


def _entry_once_strategy():
    fired = {"done": False}

    def strat(t, snapshot, account):
        if fired["done"]:
            return []
        bar = snapshot.get("EURUSD")
        if bar is None:
            return []
        fired["done"] = True
        entry_proxy = float(bar["close_ask"])
        return [
            Order(
                pair="EURUSD",
                direction=Direction.LONG,
                size=100_000.0,
                sl_price=entry_proxy - 0.01,  # 100-pip stop → breached at bar 5
            )
        ]

    return strat


def test_single_component_book_equals_scoring_alone():
    """A 1-component co-sim book reproduces the component's canonical FoldStats
    bar-for-bar — the co-sim re-marking does not perturb a single book."""
    panel = _panel_one_sl_trade()
    from core.sim.account import Account

    acct = Account(starting_balance=SB)
    bt = MultiPairBacktester(panel=panel, account=acct, strategy=_entry_once_strategy())
    rr = bt.run()
    assert rr.n_trades == 1, "fixture must produce exactly one closed trade"

    reference = build_fold_stats_from_run(fold=FOLD_2010, run_result=rr, starting_balance=SB)

    comp = CoSimComponent(
        name="solo",
        closed_trades=rr.closed_trades,
        mark_prices=mark_prices_from_panel(panel),
        starting_balance=SB,
    )
    book = cosim_book_fold([comp], FOLD_2010, weights=[1.0], starting_balance=SB)

    # Gross equity reproduced exactly (same marks, same trade). check_freq=False:
    # the co-sim clock is a set-union index (no freq), the engine's is a
    # date_range (freq=H) — the VALUES are identical, only the freq attr differs.
    pd.testing.assert_series_equal(
        book.gross_equity, rr.equity_curve, check_names=False, check_freq=False
    )
    # And therefore the cost-netted FoldStats are identical.
    assert book.fold_stats.roi_pct == pytest.approx(reference.roi_pct, abs=1e-12)
    assert book.fold_stats.max_dd_pct == pytest.approx(reference.max_dd_pct, abs=1e-12)
    assert book.fold_stats.n_trades == reference.n_trades
    assert book.fold_stats.days_breaching_daily_5pct == reference.days_breaching_daily_5pct
    assert book.n_admitted == 1 and book.n_dropped == 0


# ── 2. no-interaction == linear combiner ──────────────────────────────────
@pytest.mark.parametrize("weights", [None, [0.5, 0.3, 0.2], [0.6, 0.4, 0.0]])
def test_no_interaction_equals_linear(weights):
    """With disjoint currencies (cap never binds) the co-sim per-fold ROI equals
    the linear combiner's weighted ROI Σ w_k·ROI_k — exactly. This is the anchor
    proving the co-sim introduces NO return source."""
    comps = [
        CoSimComponent("A", (_ct(1, "EURUSD", 2, 30, +600.0),), _flat_marks(["EURUSD"]), SB),
        CoSimComponent("B", (_ct(1, "GBPJPY", 4, 20, -200.0),), _flat_marks(["GBPJPY"]), SB),
        CoSimComponent("C", (_ct(1, "AUDCAD", 6, 28, +350.0),), _flat_marks(["AUDCAD"]), SB),
    ]
    n = len(comps)
    w = [1.0 / n] * n if weights is None else weights

    own = [_component_own_roi(c) for c in comps]
    linear = sum(wk * rk for wk, rk in zip(w, own))

    book = cosim_book_fold(
        comps, FOLD_2010, weights=weights, starting_balance=SB, apply_exposure_cap=False
    )
    assert book.n_dropped == 0
    assert book.roi_pct == pytest.approx(linear, abs=1e-12)


def test_no_interaction_equals_linear_even_with_cap_on_when_disjoint():
    """Cap ON but disjoint currencies → cap never binds → still == linear."""
    comps = [
        CoSimComponent("A", (_ct(1, "EURUSD", 2, 30, +600.0),), _flat_marks(["EURUSD"]), SB),
        CoSimComponent("B", (_ct(1, "GBPJPY", 4, 20, +200.0),), _flat_marks(["GBPJPY"]), SB),
    ]
    own = [_component_own_roi(c) for c in comps]
    linear = 0.5 * own[0] + 0.5 * own[1]
    book = cosim_book_fold(comps, FOLD_2010, starting_balance=SB)  # cap on, equal weights
    assert book.n_dropped == 0
    assert book.roi_pct == pytest.approx(linear, abs=1e-12)


# ── 3a. can only tighten: the exposure cap drops an over-currency WINNER ──
# This is the dispatch's literal "can-only-tighten" gate: a constructed
# multi-component case where the co-sim per-fold ROI ≤ the linear combiner's.
# (Pairs are all non-JPY so per-pip value is uniform — see the cost-model note
# in the module/PR: an active exposure cap is faithful but NOT strictly
# ROI-monotone, because dropping a net-LOSING entry would raise ROI. The
# strictly-monotone guarantee is the DD-interaction + daily-cap, anchored by
# the cap-off == linear tests. For the 4-way book the cap never binds — its
# components have disjoint event timing — so cap-on == cap-off there.)
def test_exposure_cap_drops_over_currency_position_and_tightens():
    """Three components all hold an EUR-touching pair simultaneously. The
    2-per-currency book cap admits two and DROPS the third. The third is a
    same-pip-scale WINNER, so dropping it makes the co-sim ROI strictly LOWER
    than the linear combiner (which counts all three) — the safety invariant,
    direction encoded."""
    # All three open at the same bar and overlap → EUR is touched 3× at once.
    comps = [
        CoSimComponent("A_eurusd", (_ct(1, "EURUSD", 2, 40, +100.0),), _flat_marks(["EURUSD"]), SB),
        CoSimComponent("B_eurgbp", (_ct(1, "EURGBP", 2, 40, +100.0),), _flat_marks(["EURGBP"]), SB),
        CoSimComponent("C_euraud", (_ct(1, "EURAUD", 2, 40, +500.0),), _flat_marks(["EURAUD"]), SB),
    ]
    own = [_component_own_roi(c) for c in comps]
    linear = sum(r / 3.0 for r in own)  # equal-weight linear combiner

    book = cosim_book_fold(comps, FOLD_2010, starting_balance=SB)  # cap on
    # EUR is touched by all three at once → exactly one is dropped.
    assert book.n_dropped == 1
    # The dropped one is the last by (component_index, position_id) tie-break — C.
    assert book.dropped == (("C_euraud", 1),)
    # Dropping a winner tightens: co-sim ROI strictly below the linear combiner.
    assert book.roi_pct < linear
    # And the cap-off book equals the linear combiner (no drop) — isolates the
    # delta as purely the (faithful) dropped winner.
    book_off = cosim_book_fold(
        comps, FOLD_2010, starting_balance=SB, apply_exposure_cap=False
    )
    assert book_off.roi_pct == pytest.approx(linear, abs=1e-12)
    assert book.roi_pct < book_off.roi_pct


def test_exposure_cap_admits_when_under_two_per_currency():
    """Two EUR-touchers is allowed (cap is 2); nothing dropped."""
    comps = [
        CoSimComponent("A", (_ct(1, "EURUSD", 2, 40, +300.0),), _flat_marks(["EURUSD"]), SB),
        CoSimComponent("B", (_ct(1, "EURGBP", 2, 40, +300.0),), _flat_marks(["EURGBP"]), SB),
    ]
    book = cosim_book_fold(comps, FOLD_2010, starting_balance=SB)
    assert book.n_dropped == 0 and book.n_admitted == 2


# ── 3b. can only tighten: shared daily-DD cap fails a fold linear ROI passes ──
def _dip_marks(pair: str, dip_at: int, *, entry_px: float = 1.0, dip_px: float = 0.90,
               idx: pd.DatetimeIndex = IDX) -> dict:
    """A close-mid series that sits at ``entry_px`` except a single deep dip
    bar (used to manufacture an intraday drawdown while a position is open)."""
    s = pd.Series(entry_px, index=idx, name="close_mid", dtype=float)
    s.iloc[dip_at] = dip_px
    return {pair: s}


def test_shared_daily_cap_fails_a_fold_the_roi_judge_passes():
    """A book that ends the day green (ROI > 0, passes the linear ROI judge) but
    dipped >5% intraday is FAILED by the shared 5% daily cap — a real 5ers daily
    breach = account dead. The linear combiner is blind to this; the co-sim is
    strictly harder."""
    # One position open across a deep midday dip, exiting slightly green.
    # entry 1.0, dip to 0.90 (−10% on full size) midday, exit +tiny.
    comp = CoSimComponent(
        "dipper",
        (_ct(1, "EURUSD", 1, 40, +50.0, entry_px=1.0, size=SB),),
        _dip_marks("EURUSD", dip_at=10, entry_px=1.0, dip_px=0.90),
        SB,
    )
    book = cosim_book_fold([comp], FOLD_2010, weights=[1.0], starting_balance=SB)
    assert book.passes_roi is True          # ends green → linear ROI judge passes
    assert book.daily_dd_breached is True   # ...but breached 5% intraday
    assert book.passes_with_daily_cap is False  # the shared cap fails the fold


def test_coincident_drawdowns_deepen_book_dd_vs_disjoint():
    """Real cross-component DD interaction the linear combiner cannot see: two
    components dipping on the SAME bar produce a deeper book max-DD than the
    same two dipping on DIFFERENT bars, holding ROI fixed."""
    # Each: long, open across the window, dip −6% at its dip bar, exit flat (pnl≈0).
    def comp(name, pair, dip_at):
        return CoSimComponent(
            name,
            (_ct(1, pair, 1, 40, +1.0, entry_px=1.0, size=SB),),  # ~breakeven
            _dip_marks(pair, dip_at=dip_at, entry_px=1.0, dip_px=0.94),  # −6% dip
            SB,
        )

    coincident = cosim_book_fold(
        [comp("A", "EURUSD", 10), comp("B", "GBPJPY", 10)],
        FOLD_2010, starting_balance=SB, apply_exposure_cap=False,
    )
    disjoint = cosim_book_fold(
        [comp("A", "EURUSD", 10), comp("B", "GBPJPY", 20)],
        FOLD_2010, starting_balance=SB, apply_exposure_cap=False,
    )
    # Same ROI (same trades), but coincident dips stack → deeper book DD.
    assert coincident.fold_stats.roi_pct == pytest.approx(disjoint.fold_stats.roi_pct, abs=1e-9)
    assert coincident.fold_stats.max_dd_pct > disjoint.fold_stats.max_dd_pct + 1e-6


# ── 4. determinism ────────────────────────────────────────────────────────
def test_determinism_two_run_byte_identity():
    comps = [
        CoSimComponent("A", (_ct(1, "EURUSD", 2, 30, +600.0), _ct(2, "EURUSD", 5, 33, -120.0)),
                       _flat_marks(["EURUSD"]), SB),
        CoSimComponent("B", (_ct(1, "GBPJPY", 4, 20, +200.0),), _flat_marks(["GBPJPY"]), SB),
        CoSimComponent("C", (_ct(1, "AUDCAD", 6, 28, +350.0),), _flat_marks(["AUDCAD"]), SB),
    ]
    r1 = cosim_book_fold(comps, FOLD_2010, starting_balance=SB)
    r2 = cosim_book_fold(comps, FOLD_2010, starting_balance=SB)
    pd.testing.assert_series_equal(r1.gross_equity, r2.gross_equity)
    pd.testing.assert_series_equal(r1.net_equity, r2.net_equity)
    assert r1.fold_stats == r2.fold_stats
    assert r1.dropped == r2.dropped
    assert r1.days_breaching_daily_5pct_eet == r2.days_breaching_daily_5pct_eet


def test_zero_weight_component_is_absent():
    """A 0-weight component contributes nothing (matches the combiner dropping it)."""
    comps = [
        CoSimComponent("keep", (_ct(1, "EURUSD", 2, 30, +600.0),), _flat_marks(["EURUSD"]), SB),
        CoSimComponent("drop", (_ct(1, "EURUSD", 3, 31, +900.0),), _flat_marks(["EURUSD"]), SB),
    ]
    book = cosim_book_fold(comps, FOLD_2010, weights=[1.0, 0.0], starting_balance=SB,
                           apply_exposure_cap=False)
    solo = cosim_book_fold([comps[0]], FOLD_2010, weights=[1.0], starting_balance=SB)
    assert book.roi_pct == pytest.approx(solo.roi_pct, abs=1e-12)


def test_costs_are_per_leg_not_netted():
    """Per-leg cost discipline: the book's total cost equals the sum of each
    component's standalone cost (no cross-instrument netting / Arc-10 defect)."""
    a = CoSimComponent("A", (_ct(1, "EURUSD", 2, 30, +600.0),), _flat_marks(["EURUSD"]), SB)
    b = CoSimComponent("B", (_ct(1, "GBPJPY", 4, 20, +200.0),), _flat_marks(["GBPJPY"]), SB)
    model = CostModel.fundednext()
    # Standalone net-vs-gross gap = standalone cost (equal weight → ×0.5 each).
    book = cosim_book_fold([a, b], FOLD_2010, starting_balance=SB, cost_model=model,
                           apply_exposure_cap=False)
    gross_a = cosim_book_fold([a], FOLD_2010, weights=[1.0], starting_balance=SB,
                              cost_model=CostModel.zero())
    net_a = cosim_book_fold([a], FOLD_2010, weights=[1.0], starting_balance=SB, cost_model=model)
    # The book applied a strictly positive cost (commission+slippage) per leg.
    assert net_a.roi_pct < gross_a.roi_pct
    # Book net ROI is below the cost-free linear combine (costs were applied, not waived).
    book_costfree = cosim_book_fold([a, b], FOLD_2010, starting_balance=SB,
                                    cost_model=CostModel.zero(), apply_exposure_cap=False)
    assert book.roi_pct < book_costfree.roi_pct

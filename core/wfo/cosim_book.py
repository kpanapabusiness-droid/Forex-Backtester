"""Co-simulated portfolio EQUITY CURVE — the single-book discovery gate (Item E).

This is the CANONICAL-CORE scoring path that replaces (or sits beside) the
per-fold **LINEAR** ROI combiner (``discovery/tools/combine_fold_roi.py``) for
the discovery all-folds-positive judgement on a multi-component BOOK. The linear
combiner sums each component's *independently simulated* per-fold ROI; it is
faithful only when the components trade disjoint universes and never hold
simultaneous positions. It cannot see the book's ACTUAL risk geometry: a shared
5% daily-DD cap and real cross-component drawdown interaction.

``cosim_book_fold`` marks **all components to ONE equity line on a shared clock**
under a **shared 2-per-currency exposure cap applied to the book's net exposure**
and a **shared 5% daily-DD cap**, then scores the single co-simulated curve
through the canonical cost + FoldStats chokepoint.

────────────────────────────────────────────────────────────────────────────
THE INVARIANT THAT MAKES THIS SAFE — "can only tighten"
────────────────────────────────────────────────────────────────────────────
A shared daily-DD cap + real cross-component drawdown interaction is *strictly
harder* than the per-fold linear sum, so the co-simulated curve can only make a
fold's verdict **equal or worse, never better**. There is ZERO fabrication
surface. Concretely, the co-sim adds only CONSTRAINTS and removes only the
linear sum's idealizations — it never adds a return source:

  1. Per-trade P&L is taken VERBATIM from each component's own canonical
     simulation (same entries, exits, fill prices, SL-first take-the-loss). The
     co-sim re-marks those trades onto one equity line; it never re-fills or
     re-prices a trade.
  2. Per-leg costs stay per-leg: each component position pays its own full
     FundedNext round-turn cost (``apply_cost_model`` runs per-position on the
     union ledger). The co-sim nets *exposure and risk on one equity line*, it
     does NOT net broker billing — modelling a cross-instrument cost saving here
     would be a textbook Arc-10 defect.
  3. With capital weights ``w`` (Σw = 1, frozen on IS — the SAME weights the
     linear combiner uses) and NO interaction (caps never bind, drawdowns never
     coincide), the co-sim per-fold ROI EQUALS the linear combiner's weighted
     ROI exactly: ``Σ_k w_k · ROI_k``. See
     ``tests/wfo/test_cosim_book.py::test_no_interaction_equals_linear``.
  4. The shared 5% daily cap can only FAIL a fold the book would otherwise pass
     (a real 5ers/FundedNext daily breach = account dead), never rescue one —
     and the discovery ROI judge is unchanged when caps are off, so the daily
     cap is a STRICT extra gate (``passes_with_daily_cap`` ⟹ ``passes_roi``).
  5. Real cross-component drawdown interaction can only DEEPEN the book's
     max-DD (coincident drawdowns stack), never flatten it below the linear
     view (which sees no book DD at all). Monotone.

The strictly-monotone "can only tighten" guarantee is points 3–5 (no return
source + daily cap + DD interaction). The shared 2-per-currency cap (point-2
companion) can only DROP entries the linear sum counted — never add one — but it
is NOT strictly ROI-monotone trade-by-trade: dropping a net-LOSING entry raises
ROI, and no lookahead-free cap can avoid that (you cannot decide to skip a trade
by its unknown outcome). This matters only when components hold simultaneous
same-currency positions; the 4-way book this gate validates has disjoint event
timing, so its cap NEVER binds (cap-on == cap-off) and the strict guarantee
holds in practice. The driver therefore reports the exposure drop explicitly
(``n_dropped`` / ``dropped``); ``apply_exposure_cap=False`` gives the
strictly-monotone bound. If a co-sim book ever scores *better* than the linear
combiner with the cap OFF, that is a BUG (optimism introduced — e.g. accidental
cost-netting or double-counted equity); see the "can only tighten" property
tests. The review anchor is: with the cap off this can only tighten the verdict;
with the cap on it additionally drops trades the real book could not take.

────────────────────────────────────────────────────────────────────────────
Capital convention (apples-to-apples with the linear combiner)
────────────────────────────────────────────────────────────────────────────
The book holds ONE capital pool of ``starting_balance``; component ``k`` is
allocated fraction ``w_k`` of it (Σ w_k = 1). Each component's already-scored
trades are linearly scaled by ``w_k`` (size → size·w_k, hence P&L and cost →
·w_k). The combined gross equity is therefore
``starting_balance + Σ_k w_k · pnl_k(t)`` and the combined ROI is
``Σ_k w_k · ROI_k`` BEFORE the (only-tightening) cap interactions — identical in
scale and convention to ``combine_fold_rois`` with the same weights. Equal
weights ``w_k = 1/N`` reproduce the combiner's ``"equal"`` mode.

────────────────────────────────────────────────────────────────────────────
Reuse, not re-author
────────────────────────────────────────────────────────────────────────────
The only NEW scoring code here is the co-simulation loop (shared-clock advance,
one equity line, the two book-level caps). Everything load-bearing is the
canonical engine, CALLED not re-rolled:

  - ``core.sim.account.Account`` (sign-based ``Direction``/``Position`` +
    ``parent_position_id`` multi-leg primitive, shorts probe Q2.6) for the
    2-per-currency exposure admission — ``exposure_check`` is the canonical cap.
  - ``core.sim.costs.model.apply_cost_model`` (FundedNext per-leg costs).
  - ``core.runners._fold_stats_helpers.build_fold_stats_from_run`` (the gate
    cost+FoldStats chokepoint) and ``compute_per_day_max_dd`` (EET daily-DD).

Determinism: components are processed in caller order; positions and same-clock
events are tie-broken by ``(component_index, position_id)``; the clock is the
sorted union of every component's mark index. No RNG, no wall-clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from core.runners._fold_stats_helpers import (
    build_fold_stats_from_run,
    compute_per_day_max_dd,
    slice_equity_to_oos,
)
from core.sim.account import (
    Account,
    ClosedTrade,
    Direction,
    ExposureRules,
)
from core.sim.costs.model import CostModel, apply_cost_model
from core.sim.multipair_backtester import RunResult
from core.wfo.folds import Fold
from core.wfo.gates import FoldStats

# The book-level exposure cap: 2 positions per currency applied to the WHOLE
# book's net exposure, uncapped per-pair and per-total (the dispatch specifies
# only the per-currency cap). Reuses the canonical ``Account.exposure_check``.
DEFAULT_BOOK_EXPOSURE: ExposureRules = ExposureRules(
    max_concurrent_total=None,
    max_concurrent_per_pair=None,
    max_concurrent_per_currency=2,
)

# 5ers / FundedNext daily-DD breach threshold (fraction of day-start equity).
DAILY_DD_CAP: float = 0.05

# Stable per-component id offset so position ids never collide across
# components when the union ledger is grouped by position_id for costing.
_COMPONENT_ID_OFFSET: int = 10_000_000


@dataclass(frozen=True)
class CoSimComponent:
    """One book component, ALREADY scored by the canonical engine on its own.

    ``closed_trades`` is the component's gross ``RunResult.closed_trades`` (each
    carrying entry/exit time+price, direction, size, sl_price and the entry/exit
    bid+ask the cost model reads). ``mark_prices`` is the per-pair close-mid
    series on the component's PRIMARY-TF panel (the engine's mark axis), used to
    mark this component's open positions on the shared clock — exactly the marks
    its own ``Account.mark_to_market`` used, so an un-dropped component
    reproduces its own equity contribution bar-for-bar.
    """

    name: str
    closed_trades: tuple[ClosedTrade, ...]
    mark_prices: Mapping[str, pd.Series]
    starting_balance: float = 100_000.0


@dataclass(frozen=True)
class CoSimBookResult:
    """Result of co-simulating a multi-component book over ONE fold."""

    fold_stats: FoldStats                # cost-netted, OOS-restricted (canonical chokepoint)
    gross_equity: pd.Series              # the single co-simulated equity line (gross), full clock
    net_equity: pd.Series                # cost-netted equity line, full clock
    weights: tuple[float, ...]           # capital fractions used, component order
    component_names: tuple[str, ...]
    n_admitted: int                      # positions admitted under the book exposure cap
    n_dropped: int                       # positions dropped by the book exposure cap
    dropped: tuple[tuple[str, int], ...]  # (component_name, position_id) dropped, sorted
    days_breaching_daily_5pct_eet: int   # EET-convention book daily-DD breaches over OOS
    per_day_max_dd_eet: pd.DataFrame     # compute_per_day_max_dd on the OOS net equity (EET)
    daily_dd_breached: bool              # True iff the book breached the 5% daily cap in OOS

    @property
    def roi_pct(self) -> float:
        return self.fold_stats.roi_pct

    @property
    def passes_roi(self) -> bool:
        """The linear-combiner-comparable judge: book ROI strictly positive."""
        return self.fold_stats.roi_pct > 0.0

    @property
    def passes_with_daily_cap(self) -> bool:
        """The full book judge: ROI > 0 AND no 5% daily-DD breach (account alive)."""
        return self.passes_roi and not self.daily_dd_breached


# ── internal: weight-scaled, globally-unique re-id of a component's trades ──
def _reid_scaled_trades(
    trades: Sequence[ClosedTrade], component_index: int, weight: float
) -> list[ClosedTrade]:
    """Return ``trades`` with size/pnl scaled by ``weight`` and position ids
    offset into a component-unique range (so the union ledger groups legs of
    the same original position together, and never merges two components'
    ``position_id == 1`` into one position at costing time).
    """
    base = (component_index + 1) * _COMPONENT_ID_OFFSET
    out: list[ClosedTrade] = []
    for t in trades:
        new_pid = base + int(t.position_id)
        new_parent = (
            None if t.parent_position_id is None else base + int(t.parent_position_id)
        )
        out.append(
            ClosedTrade(
                position_id=new_pid,
                pair=t.pair,
                direction=t.direction,
                entry_time=t.entry_time,
                entry_price=t.entry_price,
                exit_time=t.exit_time,
                exit_price=t.exit_price,
                size=float(t.size) * weight,
                pnl=float(t.pnl) * weight,
                exit_reason=t.exit_reason,
                parent_position_id=new_parent,
                entry_bid=t.entry_bid,
                entry_ask=t.entry_ask,
                exit_bid=t.exit_bid,
                exit_ask=t.exit_ask,
                sl_price=t.sl_price,
            )
        )
    return out


@dataclass
class _Position:
    """One co-sim position: an original position's leg group, on one component."""

    key: tuple[str, int]          # (component_name, original position_id)
    component_index: int
    pair: str
    direction: Direction
    entry_time: pd.Timestamp
    entry_price: float
    total_size: float             # weight-scaled
    legs: list[tuple[pd.Timestamp, float, float]]  # (exit_time, exit_price, leg_size), sorted by exit_time
    mark: pd.Series               # this component's close-mid for ``pair``

    @property
    def final_exit_time(self) -> pd.Timestamp:
        return self.legs[-1][0]


def _group_positions(
    component_index: int,
    component_name: str,
    reid_trades: Sequence[ClosedTrade],
    mark_prices: Mapping[str, pd.Series],
) -> list[_Position]:
    """Group a component's (re-id'd, scaled) closed trades into positions.

    Legs sharing a ``position_id`` are one position (partial close + runner);
    a standalone full close is a 1-leg position. The position's entry geometry
    is taken from any leg (all legs of a position share entry_time/price/dir).
    """
    by_pid: dict[int, list[ClosedTrade]] = {}
    for t in reid_trades:
        by_pid.setdefault(int(t.position_id), []).append(t)
    positions: list[_Position] = []
    for pid in sorted(by_pid):
        legs = sorted(by_pid[pid], key=lambda x: (pd.Timestamp(x.exit_time), float(x.size)))
        head = legs[0]
        pair = head.pair
        mark = mark_prices.get(pair)
        if mark is None:
            raise KeyError(
                f"component {component_name!r} position {pid} trades {pair!r} but "
                f"no mark_prices series was supplied for it"
            )
        orig_pid = pid - (component_index + 1) * _COMPONENT_ID_OFFSET
        positions.append(
            _Position(
                key=(component_name, orig_pid),
                component_index=component_index,
                pair=pair,
                direction=head.direction,
                entry_time=pd.Timestamp(head.entry_time),
                entry_price=float(head.entry_price),
                total_size=float(sum(float(leg.size) for leg in legs)),
                legs=[
                    (pd.Timestamp(leg.exit_time), float(leg.exit_price), float(leg.size))
                    for leg in legs
                ],
                mark=mark,
            )
        )
    return positions


def _admit_under_exposure_cap(
    positions: Sequence[_Position], exposure: ExposureRules
) -> set[tuple[str, int]]:
    """Walk a shared clock and admit/drop each position under the book cap.

    Reuses the canonical ``Account.exposure_check`` (the 2-per-currency cap
    applied to the whole book's open positions). At each timestamp OPEN events
    are processed before CLOSE events — matching the engine's intra-bar order
    (``_fill_pending_entries`` before ``_check_exits``), so a new entry is gated
    against positions still open on that bar. Ties at a timestamp break on
    ``(component_index, original position_id)`` for determinism.

    Returns the set of admitted position keys. A dropped position is one the
    book could not open without breaching its 2-per-currency cap; it never
    enters the equity curve or the cost ledger (honest skip).
    """
    account = Account(starting_balance=1.0, exposure=exposure)
    # Event kinds: 0 = OPEN (processed first at a timestamp), 1 = CLOSE.
    events: list[tuple[pd.Timestamp, int, int, int, _Position]] = []
    for p in positions:
        events.append((p.entry_time, 0, p.component_index, p.key[1], p))
        events.append((p.final_exit_time, 1, p.component_index, p.key[1], p))
    events.sort(key=lambda e: (e[0], e[1], e[2], e[3]))

    admitted: set[tuple[str, int]] = set()
    open_acct_id: dict[tuple[str, int], int] = {}
    for ts, kind, _ci, _oid, p in events:
        if kind == 1:  # CLOSE
            acct_id = open_acct_id.pop(p.key, None)
            if acct_id is not None:
                account.close(acct_id, ts, p.entry_price, "cosim_close")
        else:  # OPEN
            if account.exposure_check(p.pair):
                pos = account.open(
                    pair=p.pair,
                    direction=p.direction,
                    entry_time=ts,
                    entry_price=p.entry_price,
                    size=max(p.total_size, 1.0),  # size irrelevant to exposure; keep > 0
                )
                open_acct_id[p.key] = pos.position_id
                admitted.add(p.key)
    return admitted


def _position_contribution(pos: _Position, clock: pd.DatetimeIndex) -> np.ndarray:
    """Per-clock-bar gross P&L contribution of one position (realized + marked).

    For t < entry: 0. While open over a segment [prev_exit, leg_exit): the
    still-open size is marked at this component's close-mid (ffilled to the
    shared clock — a flat mark between the component's own bars, matching its
    own ``Account.mark_to_market``). Each leg's realized P&L is added forward
    from its exit. After the final exit the contribution is the position's
    total realized P&L (constant). Equivalent bar-for-bar to the component's
    own equity contribution when the position is admitted.
    """
    mark_arr = pos.mark.reindex(clock).ffill().to_numpy(dtype=float)
    clock_vals = clock.values
    contrib = np.zeros(len(clock), dtype=float)
    sign = pos.direction.sign
    entry = pos.entry_price
    size_remaining = pos.total_size
    prev = np.datetime64(pos.entry_time.to_datetime64())
    for exit_ts, exit_price, leg_size in pos.legs:
        exit_v = np.datetime64(pd.Timestamp(exit_ts).to_datetime64())
        seg = (clock_vals >= prev) & (clock_vals < exit_v)
        if seg.any():
            contrib[seg] += sign * (mark_arr[seg] - entry) * size_remaining
        after = clock_vals >= exit_v
        if after.any():
            contrib[after] += sign * (exit_price - entry) * leg_size
        size_remaining -= leg_size
        prev = exit_v
    # Defensive: a NaN mark inside an open segment (should not happen — the
    # segment lies within the component's own panel) must not poison equity.
    return np.nan_to_num(contrib, nan=0.0)


def cosim_book_fold(
    components: Sequence[CoSimComponent],
    fold: Fold,
    *,
    weights: Sequence[float] | None = None,
    starting_balance: float = 100_000.0,
    exposure: ExposureRules = DEFAULT_BOOK_EXPOSURE,
    cost_model: CostModel | None = None,
    apply_exposure_cap: bool = True,
) -> CoSimBookResult:
    """Co-simulate a multi-component book over ONE fold → one curve → FoldStats.

    Parameters
    ----------
    components
        The already-scored components (one per book leg). Order is the weight
        order and the determinism tie-break order.
    fold
        The fold whose OOS window the resulting FoldStats restrict to (same
        ``Fold`` the components were scored on).
    weights
        Capital fractions, summed to 1, frozen on IS (the SAME weights the
        linear combiner uses). ``None`` → equal weights ``1/N``. A 0-weight
        component contributes nothing (matches the combiner dropping it).
    starting_balance
        The book's single capital pool.
    exposure
        Book-level exposure cap. Default: 2-per-currency, uncapped otherwise.
    cost_model
        FundedNext by default (per-leg, the gate default). Never silently zero.
    apply_exposure_cap
        When False, no entry is dropped (pure equity superposition). Used by the
        ``test_no_interaction_equals_linear`` anchor; production gating leaves
        this True.

    Returns
    -------
    CoSimBookResult
        The co-simulated book's FoldStats + the single equity line + the
        exposure-drop and daily-DD-breach diagnostics.
    """
    if not components:
        raise ValueError("cosim_book_fold requires ≥1 component")
    n = len(components)
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} components")
    w = [float(x) for x in weights]
    if any(x < 0.0 for x in w):
        raise ValueError(f"weights must be ≥ 0; got {w}")
    tot = sum(w)
    if tot <= 0.0:
        raise ValueError("weights sum to 0")
    # Tolerate a non-normalised weight vector by normalising (frozen IS weights
    # from ``fit_weights`` already sum to 1; this is just defensive).
    if not np.isclose(tot, 1.0):
        w = [x / tot for x in w]

    model = cost_model if cost_model is not None else CostModel.fundednext()

    # 1. Weight-scale + globally re-id each component's trades; group into
    #    positions; collect the per-component mark series.
    all_positions: list[_Position] = []
    all_reid_trades: list[ClosedTrade] = []
    for ci, (comp, wk) in enumerate(zip(components, w)):
        if wk == 0.0:
            continue  # a 0-weight component is absent from the book
        reid = _reid_scaled_trades(comp.closed_trades, ci, wk)
        all_reid_trades.extend(reid)
        all_positions.extend(
            _group_positions(ci, comp.name, reid, comp.mark_prices)
        )

    # 2. Shared clock = sorted union of every component's mark index (the bars
    #    where the book's open positions are marked). Every trade entry/exit
    #    time is a bar on its pair's panel, so it is already in this union.
    clock = pd.DatetimeIndex([], tz="UTC")
    for comp in components:
        for series in comp.mark_prices.values():
            clock = clock.union(series.index)
    if len(clock) == 0:
        # No marks anywhere — degenerate empty book.
        return _empty_result(fold, tuple(w), tuple(c.name for c in components))

    # 3. Book-level exposure admission (the canonical 2-per-currency cap).
    if apply_exposure_cap:
        admitted_keys = _admit_under_exposure_cap(all_positions, exposure)
    else:
        admitted_keys = {p.key for p in all_positions}
    dropped = tuple(sorted(p.key for p in all_positions if p.key not in admitted_keys))

    # 4. One gross equity line: Σ admitted position contributions on the clock.
    pnl_path = np.zeros(len(clock), dtype=float)
    admitted_positions = [p for p in all_positions if p.key in admitted_keys]
    for pos in admitted_positions:
        pnl_path += _position_contribution(pos, clock)
    gross_equity = pd.Series(
        starting_balance + pnl_path, index=clock, name="equity"
    )

    # 5. Admitted union ledger (for per-leg cost netting). Drop any leg whose
    #    position was not admitted.
    admitted_pid = {
        (p.component_index + 1) * _COMPONENT_ID_OFFSET + p.key[1]
        for p in admitted_positions
    }
    admitted_trades = tuple(
        t for t in all_reid_trades if int(t.position_id) in admitted_pid
    )

    # 6. Canonical cost + FoldStats chokepoint (FundedNext per-leg, OOS-sliced).
    run_result = RunResult(
        final_balance=float(gross_equity.iloc[-1]),
        n_trades=len(admitted_trades),
        n_open_at_end=0,
        equity_curve=gross_equity,
        max_drawdown_pct=0.0,  # recomputed inside build_fold_stats_from_run
        closed_trades=admitted_trades,
    )
    fold_stats = build_fold_stats_from_run(
        fold=fold,
        run_result=run_result,
        starting_balance=starting_balance,
        cost_model=model,
    )

    # 7. EET daily-DD diagnostic on the OOS net equity (the canonical bucketing
    #    named in CLAUDE.md). FoldStats.days_breaching_daily_5pct already
    #    carries the UTC-convention count from the chokepoint.
    costed = apply_cost_model(run_result, model)
    net_equity = costed.net_equity
    net_oos = slice_equity_to_oos(net_equity, fold)
    per_day = compute_per_day_max_dd(net_oos, pair_set="cosim_book")
    if len(per_day):
        eet_breaches = int((per_day["day_max_dd_base_pct"] >= DAILY_DD_CAP).sum())
    else:
        eet_breaches = 0
    # The book breached the daily cap if EITHER convention saw a ≥5% day.
    daily_breached = bool(
        eet_breaches > 0 or fold_stats.days_breaching_daily_5pct > 0
    )

    return CoSimBookResult(
        fold_stats=fold_stats,
        gross_equity=gross_equity,
        net_equity=net_equity,
        weights=tuple(w),
        component_names=tuple(c.name for c in components),
        n_admitted=len(admitted_positions),
        n_dropped=len(dropped),
        dropped=dropped,
        days_breaching_daily_5pct_eet=eet_breaches,
        per_day_max_dd_eet=per_day,
        daily_dd_breached=daily_breached,
    )


def _empty_result(
    fold: Fold, weights: tuple[float, ...], names: tuple[str, ...]
) -> CoSimBookResult:
    empty = pd.Series([], dtype="float64", name="equity")
    return CoSimBookResult(
        fold_stats=FoldStats(
            fold_id=fold.fold_id,
            n_trades=0,
            roi_pct=0.0,
            max_dd_pct=0.0,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=0.0,
        ),
        gross_equity=empty,
        net_equity=empty,
        weights=weights,
        component_names=names,
        n_admitted=0,
        n_dropped=0,
        dropped=(),
        days_breaching_daily_5pct_eet=0,
        per_day_max_dd_eet=compute_per_day_max_dd(empty),
        daily_dd_breached=False,
    )


def mark_prices_from_panel(panel) -> dict[str, pd.Series]:
    """Extract per-pair close-mid ``(close_bid + close_ask) / 2`` from a Panel.

    This is the engine's mark axis (``MultiPairBacktester._close_mid``), so an
    admitted component reproduces its own equity contribution exactly. ``panel``
    is duck-typed (anything exposing ``.pair_dfs``) to avoid importing Panel.
    """
    out: dict[str, pd.Series] = {}
    for pair, df in panel.pair_dfs.items():
        out[pair] = ((df["close_bid"] + df["close_ask"]) / 2.0).rename("close_mid")
    return out


__all__ = (
    "CoSimComponent",
    "CoSimBookResult",
    "cosim_book_fold",
    "mark_prices_from_panel",
    "DEFAULT_BOOK_EXPOSURE",
    "DAILY_DD_CAP",
)

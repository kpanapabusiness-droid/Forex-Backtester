"""Gate-layer cost model — nets FundedNext broker costs onto a gross RunResult.

This is the wiring that resolves HONEST_ENGINE_SWEEP.md Part C (FAIL): the
per-trade primitives in this package (commission / slippage / spread_multiplier)
were orphaned — called on no gate path — so every gate verdict scored on raw
HistData bid/ask (1.0x spread, zero commission, zero slippage).

Design (per the approved plan, locked by the operator):

  - The engine (`MultiPairBacktester` / `Account`) stays **100% gross and
    untouched**. `ClosedTrade.pnl`, `Account.balance`, and the equity curve are
    never modified, so the take-the-loss invariant and determinism fixtures
    stay green and Step 6's spread-decomposition diagnostic (which reads gross
    ``pnl``) is untouched — no double-count.
  - Cost is a **separate explicit quantity**, computed here from the gross
    closed-trade ledger using the EXISTING primitives (this module does NOT
    reimplement the cost math). The netting ``net = gross - cost`` happens at
    the gate-scoring layer (`core.runners._fold_stats_helpers.build_fold_stats_from_run`),
    which nets the equity curve before computing `FoldStats`.

FundedNext profile (the gate default — the higher cost, as the conservative
bound): 1.5x spread, $5/lot round-turn commission, 0.5 pip/fill slippage x
n_fills, swaps OFF.

Conservative bias (dispatch: "over-apply, never flatter"):
  - slippage uses full original size x n_fills (over-applies on partial legs);
  - spread uses ``compute_extra_spread_price`` (the larger, full-spread extra);
  - when fill count is ambiguous we never under-count below the SL-honest
    leg structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from core.sim.costs.commission import compute_commission_usd
from core.sim.costs.slippage import compute_slippage_pips
from core.sim.costs.spread_multiplier import compute_extra_spread_price
from core.utils import LOT_SIZE, get_pip_size, pips_to_price

if TYPE_CHECKING:  # avoid runtime import cost / cycles; only used for typing
    from core.sim.multipair_backtester import RunResult


# FundedNext gate defaults (the higher-cost, conservative bound).
_FN_SPREAD_MULT: float = 1.5
_FN_COMMISSION_PER_LOT_RT: float = 5.0
_FN_SLIPPAGE_PIPS_PER_FILL: float = 0.5


_BREAKDOWN_COLUMNS: tuple[str, ...] = (
    "position_id",
    "pair",
    "original_size",
    "n_legs",
    "n_fills",
    "commission",
    "slippage",
    "spread",
    "total_cost",
    "gross_pnl",
    "net_pnl",
    "final_exit_time",
)


@dataclass(frozen=True)
class CostModel:
    """Broker cost profile applied at the gate-scoring layer.

    Defaults are the FundedNext profile (the gate default). Use
    :meth:`zero` for an EXPLICIT cost-free run (e.g. diagnostics / the
    KH-24 determinism anchor) — a cost-free gate number must never be the
    silent default.

    All cost math is delegated to the existing primitives in this package;
    this dataclass only carries the parameters and the unit conversions.
    """

    spread_mult: float = _FN_SPREAD_MULT
    commission_per_lot_rt: float = _FN_COMMISSION_PER_LOT_RT
    slippage_pips_per_fill: float = _FN_SLIPPAGE_PIPS_PER_FILL
    swaps_enabled: bool = False  # FundedNext: swaps OFF (confirmed unchanged)
    lot_size: float = float(LOT_SIZE)

    @classmethod
    def fundednext(cls) -> "CostModel":
        """The canonical gate default: 1.5x spread, $5/lot RT, 0.5 pip/fill, swaps off."""
        return cls(
            spread_mult=_FN_SPREAD_MULT,
            commission_per_lot_rt=_FN_COMMISSION_PER_LOT_RT,
            slippage_pips_per_fill=_FN_SLIPPAGE_PIPS_PER_FILL,
            swaps_enabled=False,
            lot_size=float(LOT_SIZE),
        )

    @classmethod
    def zero(cls) -> "CostModel":
        """An EXPLICIT cost-free profile (every primitive evaluates to 0)."""
        return cls(
            spread_mult=1.0,            # compute_extra_spread_price -> 0 at mult<=1
            commission_per_lot_rt=0.0,
            slippage_pips_per_fill=0.0,
            swaps_enabled=False,
            lot_size=float(LOT_SIZE),
        )


@dataclass(frozen=True)
class CostedRunResult:
    """Result of netting a gross :class:`RunResult` through a :class:`CostModel`.

    ``net_equity`` is the gross equity curve with each position's total cost
    debited at its final exit bar (cumulative, forward) — this reproduces an
    at-close balance numerically, so net ROI / max-DD / daily-5% breach are
    faithful. ``breakdown`` is the explicit, auditable per-position cost
    ledger (one row per closed position).
    """

    net_equity: pd.Series
    breakdown: pd.DataFrame
    gross_pnl_total: float
    cost_total: float
    net_pnl_total: float


def _safe_spread(ask: float, bid: float) -> float:
    """Recorded spread (ask - bid) in price units, NaN/negative coerced to 0.

    Synthetic / legacy panels carry NaN bid/ask (see
    ``core.sim.multipair_backtester._bar_field``); a raw ``ask - bid`` would
    yield NaN and poison the equity curve (and the determinism hash). Zero
    spread also matches the data-gap convention (no widening cost on a
    zero/absent spread — ``compute_extra_spread_price`` already floors at 0).
    """
    try:
        s = float(ask) - float(bid)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(s) or s <= 0.0:
        return 0.0
    return s


def _position_cost(
    legs: list, model: CostModel
) -> tuple[float, float, float, int, float]:
    """Return (commission, slippage, spread, n_fills, original_size) for one position.

    ``legs`` are all :class:`core.sim.account.ClosedTrade` rows sharing a
    ``position_id`` (a single full close => 1 leg; a +1R partial + runner =>
    2 legs). Cost is denominated in quote currency (== account currency under
    the engine's existing non-USD-quote simplification; see
    ``core/sim/risk/live_balance.py``).
    """
    original_size = float(sum(float(leg.size) for leg in legs))
    pair = legs[0].pair
    # n_fills tracks the ACTUAL SL-honest leg structure: a partial actually
    # executed iff the position closed in >1 leg (entry + N exits => 1 + N
    # fills; the canonical sl_partial_close_1r_runner_trail is <=2 legs, so
    # >1 leg == the +1R partial fired => 3 fills, else 2).
    partial_fired = len(legs) >= 2

    commission = compute_commission_usd(
        original_size / model.lot_size, model.commission_per_lot_rt
    )

    total_slip_pips, n_fills = compute_slippage_pips(
        model.slippage_pips_per_fill, tp1_hit=partial_fired
    )
    slippage = pips_to_price(pair, total_slip_pips) * original_size

    # Spread: per-leg extra at the multiplier, weighted by the leg's closed
    # size. Each leg inherits the position's entry spread; the exit spread is
    # the leg's own. Weighting by leg.size makes the entry-spread contribution
    # total correctly on original_size (sum of leg sizes) with no double-count.
    spread = 0.0
    entry_spread = _safe_spread(legs[0].entry_ask, legs[0].entry_bid)
    for leg in legs:
        exit_spread = _safe_spread(leg.exit_ask, leg.exit_bid)
        extra_price = compute_extra_spread_price(
            entry_spread, exit_spread, model.spread_mult
        )
        spread += extra_price * float(leg.size)

    return commission, slippage, spread, n_fills, original_size


def apply_cost_model(
    run_result: "RunResult", cost_model: CostModel | None = None
) -> CostedRunResult:
    """Net a gross :class:`RunResult` through ``cost_model`` (default FundedNext).

    The engine output is GROSS; this computes each closed position's cost from
    the gross ledger (using the existing primitives) and rebuilds a net equity
    curve by debiting each position's total cost at its final exit bar. Passing
    ``cost_model=None`` applies the FundedNext default — a cost-free result
    requires an EXPLICIT ``CostModel.zero()``.
    """
    model = cost_model if cost_model is not None else CostModel.fundednext()
    if model.swaps_enabled:
        raise NotImplementedError(
            "swaps are intentionally OFF for the gate cost model; wiring deferred"
        )

    equity = run_result.equity_curve
    trades = tuple(run_result.closed_trades)

    # Group legs by position (a partial-closed position shares one position_id
    # across its legs).
    groups: dict[int, list] = {}
    for t in trades:
        groups.setdefault(int(t.position_id), []).append(t)

    rows: list[dict] = []
    for pos_id in sorted(groups):
        legs = sorted(groups[pos_id], key=lambda x: (x.exit_time, float(x.size)))
        commission, slippage, spread, n_fills, original_size = _position_cost(legs, model)
        total_cost = commission + slippage + spread
        gross_pnl = float(sum(float(leg.pnl) for leg in legs))
        final_exit_time = max(leg.exit_time for leg in legs)
        rows.append(
            {
                "position_id": pos_id,
                "pair": legs[0].pair,
                "original_size": original_size,
                "n_legs": len(legs),
                "n_fills": n_fills,
                "commission": commission,
                "slippage": slippage,
                "spread": spread,
                "total_cost": total_cost,
                "gross_pnl": gross_pnl,
                "net_pnl": gross_pnl - total_cost,
                "final_exit_time": final_exit_time,
            }
        )

    breakdown = pd.DataFrame(rows, columns=list(_BREAKDOWN_COLUMNS))

    # Rebuild the net equity curve: debit each position's total cost at the
    # equity bar at/after its final exit, then carry forward (cumsum).
    if equity is None or len(equity) == 0:
        net_equity = equity if equity is not None else pd.Series(
            [], dtype="float64", name="equity"
        )
    else:
        cost_at_time = pd.Series(0.0, index=equity.index)
        for row in rows:  # rows are in sorted(position_id) order -> deterministic
            ts = row["final_exit_time"]
            idx = int(equity.index.searchsorted(ts))
            if idx >= len(equity.index):
                idx = len(equity.index) - 1
            cost_at_time.iloc[idx] += float(row["total_cost"])
        net_equity = (equity - cost_at_time.cumsum())
        net_equity.name = equity.name

    cost_total = float(breakdown["total_cost"].sum()) if len(breakdown) else 0.0
    gross_pnl_total = float(breakdown["gross_pnl"].sum()) if len(breakdown) else 0.0

    return CostedRunResult(
        net_equity=net_equity,
        breakdown=breakdown,
        gross_pnl_total=gross_pnl_total,
        cost_total=cost_total,
        net_pnl_total=gross_pnl_total - cost_total,
    )


__all__ = ("CostModel", "CostedRunResult", "apply_cost_model")

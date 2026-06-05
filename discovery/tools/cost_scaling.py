"""FundedNext cost-MULTIPLIER profile — discovery EXPERIMENT tool (BUILT).

A one-call constructor for a ``CostModel`` that is the canonical FundedNext gate
cost profile scaled by a single multiplier ``kappa`` — used to STRESS-TEST a
result's robustness to the cost assumption (the gate's one un-swept axis). It
does NOT reimplement any cost math: it only builds the parameter vector and
hands it to the canonical ``CostModel``; the netting stays canonical
(``apply_cost_model`` / ``build_fold_stats_from_run``).

The FundedNext profile (the gate default, conservative bound) is:
  - spread:     1.5x recorded spread  -> the EXTRA charged is 0.5x base spread
  - commission: $5 / lot round-turn
  - slippage:   0.5 pip / fill x n_fills
  - swaps:      OFF

A ``kappa``-multiple of that COST VECTOR scales each additive component linearly:
  - spread EXTRA   -> kappa * 0.5x base spread  => spread_mult = 1 + 0.5*kappa
  - commission     -> kappa * $5 / lot RT
  - slippage       -> kappa * 0.5 pip / fill
so kappa=1.0 reproduces FundedNext EXACTLY (spread_mult 1.5, $5, 0.5 pip),
kappa=0.0 is cost-free (== CostModel.zero(): spread_mult 1.0, $0, 0 pip), and
kappa=2.0 is double FundedNext cost (spread_mult 2.0, $10, 1.0 pip). The mapping
is exact because ``compute_extra_spread_price`` charges ``(spread_mult - 1) *
spread`` and floors at mult<=1 (so kappa=0 -> zero spread extra, matching
``CostModel.zero``).

GEOMETRY/ACCOUNTING ONLY — never realizes P&L. Feed the returned model to the
canonical ``build_fold_stats_from_run(..., cost_model=scaled_fundednext(kappa))``
to re-net an ALREADY-COMPUTED gross ``RunResult`` at a different cost level (no
engine re-run needed: the engine output is gross; only the netting changes).

Usage (cost-cushion sweep of an already-run fold):
    rr = runner.last_result.run_result            # gross RunResult for the fold
    for kappa in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        fs = build_fold_stats_from_run(
            fold=fold, run_result=rr, starting_balance=1e5,
            cost_model=scaled_fundednext(kappa),
        )
        # fs.roi_pct is the fold OOS ROI at kappa x FundedNext cost

Created by: arc 3022 (chat 3000s).
"""
from __future__ import annotations

from core.sim.costs.model import (
    _FN_COMMISSION_PER_LOT_RT,
    _FN_SLIPPAGE_PIPS_PER_FILL,
    _FN_SPREAD_MULT,
    CostModel,
)
from core.utils import LOT_SIZE


def scaled_fundednext(kappa: float) -> CostModel:
    """Return a ``CostModel`` equal to ``kappa`` x the FundedNext cost vector.

    kappa=1.0 -> FundedNext exactly; kappa=0.0 -> cost-free (== CostModel.zero);
    kappa>1   -> harsher than the gate default (the conservative-stress side).
    Raises on kappa<0 (a negative cost is meaningless).
    """
    if kappa < 0:
        raise ValueError(f"kappa must be >= 0; got {kappa}")
    # FundedNext spread EXTRA is (1.5 - 1) = 0.5x base spread; scale the EXTRA.
    spread_extra = (_FN_SPREAD_MULT - 1.0) * kappa  # 0.5 * kappa
    return CostModel(
        spread_mult=1.0 + spread_extra,
        commission_per_lot_rt=_FN_COMMISSION_PER_LOT_RT * kappa,
        slippage_pips_per_fill=_FN_SLIPPAGE_PIPS_PER_FILL * kappa,
        swaps_enabled=False,
        lot_size=float(LOT_SIZE),
    )


__all__ = ("scaled_fundednext",)

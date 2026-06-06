"""Prop-firm-challenge deployability of the 4-way book (EXPERIMENT / analysis).

GEOMETRY ONLY on the ALREADY-COMPUTED contiguous co-sim equity curve — it never
realizes P&L, never scores a trade, never touches the gate, never spends OOS. It
extends arc 1033 (`equity_risk_profile.py`, which gave the raw contiguous risk
geometry: max-DD / Calmar / time-underwater) by answering the question the corpus
characterized everywhere EXCEPT the operator's ACTUAL deployment vehicle:

  Can the path-A-gated 4-way book be deployed on a FundedNext / 5ers PROP-FIRM
  account, whose pass/fail rule is a PROFIT-TARGET vs MAX-DRAWDOWN hurdle
  (not the academic all-folds-positive gate)?

Two additions over arc 1033:

1. **Sharpe / Sortino** — the standard deployment risk-adjusted metric. The corpus
   has Calmar (arc 1033) and the mean t-stat (arc 1023, t=2.66) but never the
   annualized Sharpe. Reported at the daily-resampled frequency (deployment-relevant,
   what a continuous equity curve is judged on) AND at the annual-fold frequency
   (reconciles with arc 1023's t=2.66 → Sharpe = t/sqrt(n)).

2. **Prop-firm-challenge feasibility theorem.** Arc 1024 proved ROI and max-DD both
   scale LINEARLY with per-trade risk_pct (Calmar risk-invariant). So leverage `f`
   maps (ann_ret, maxDD) -> (f*ann_ret, f*maxDD). To reach a profit target `P` over
   `T` years needs `f = P/(ann_ret*T)`; the resulting max-DD is `f*maxDD =
   P/(Calmar*T)`. The challenge's max-DD limit `D` is cleared only if
       T  >=  (P/D) / Calmar              [the feasibility horizon]
   i.e. a book of Calmar C can pass a (P target, D max-DD) challenge by leverage
   ONLY by waiting `(P/D)/C` years AT the DD limit (zero margin). This is a clean,
   parameter-robust statement: the exact published numbers barely matter because the
   binding quantity is the ratio P/D against the book's Calmar.

EXPERIMENT tool: pure arithmetic on a canonical curve; reuses arc 1033's validated
`_build_4way_contiguous` (canonical A1 + MultiPairBacktester reproduction) and
`compute_risk_profile`. IS-only; OOS (2021+) NEVER touched.

Reproduce:  PYTHONPATH=. py discovery/tools/propfirm_feasibility.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from discovery.tools.equity_risk_profile import (
    _build_4way_contiguous,
    compute_risk_profile,
)


@dataclass(frozen=True)
class SharpeProfile:
    """Vol-adjusted risk-adjusted stats for one contiguous equity curve."""

    daily_sharpe_ann: float      # annualized Sharpe from business-day returns (x sqrt(252))
    daily_sortino_ann: float     # annualized Sortino (downside-deviation denominator)
    daily_vol_ann_pct: float     # annualized volatility of daily returns, %
    n_days: int

    def as_row(self) -> str:
        return (
            f"Sharpe(daily,ann) {self.daily_sharpe_ann:+.3f} | "
            f"Sortino {self.daily_sortino_ann:+.3f} | "
            f"vol {self.daily_vol_ann_pct:.3f}%/yr (n={self.n_days}d)"
        )


def compute_sharpe(equity: pd.Series, *, lo=None, hi=None) -> SharpeProfile:
    """Annualized Sharpe/Sortino from business-day-resampled returns.

    The book is event-driven (equity changes only on trade-close events); resampling
    to business days forward-fills the curve so flat days contribute legitimate zero
    returns (no position change). Annualize with sqrt(252). Risk-free = 0 (the book is
    measured net of costs; FundedNext capital carries no financing offset here).
    """
    eq = equity.dropna().sort_index()
    if lo is not None:
        eq = eq.loc[eq.index >= lo]
    if hi is not None:
        eq = eq.loc[eq.index <= hi]
    # to business-day frequency (UTC dates), forward-fill the step curve
    daily = eq.resample("1D").last().ffill().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2:
        raise ValueError("need >= 2 daily returns for Sharpe")
    mu = float(rets.mean())
    sd = float(rets.std(ddof=1))
    downside = rets[rets < 0.0]
    dd_sd = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    ann = 252.0
    sharpe = (mu / sd) * np.sqrt(ann) if sd > 1e-15 else float("nan")
    sortino = (mu / dd_sd) * np.sqrt(ann) if dd_sd and dd_sd > 1e-15 else float("nan")
    return SharpeProfile(
        daily_sharpe_ann=float(sharpe),
        daily_sortino_ann=float(sortino),
        daily_vol_ann_pct=float(sd * np.sqrt(ann) * 100.0),
        n_days=int(len(rets)),
    )


def annual_sharpe_from_fold_rois(fold_rois_pct) -> tuple[float, float, float]:
    """Annual Sharpe straight from the per-year book ROIs (reconciles arc 1023 t-stat).

    Returns (annual_sharpe, mean_pct, sd_pct). annual_sharpe = mean/sd over the years;
    t-stat = annual_sharpe * sqrt(n) (so n=10, t=2.66 => Sharpe ~ 0.84).
    """
    a = np.asarray(fold_rois_pct, dtype=float)
    mu, sd = float(a.mean()), float(a.std(ddof=1))
    return (mu / sd if sd > 1e-15 else float("nan"), mu, sd)


def feasibility_horizon_years(calmar: float, target_pct: float, max_dd_limit_pct: float) -> float:
    """Min years to pass a (target, max-DD-limit) challenge by linear leverage.

    T_min = (P/D)/Calmar  — at T_min the account sits exactly AT the DD limit
    (zero margin); safe operation needs a multiple of this.
    """
    if calmar <= 1e-12:
        return float("inf")
    return (target_pct / max_dd_limit_pct) / calmar


# Representative PUBLISHED prop-firm challenge structures (verify current terms before
# any deploy — these are standard-tier configs as of the corpus era, used here only to
# span the realistic P/D space; the Calmar bound is what actually decides).
_CHALLENGES = [
    # label, profit_target_pct, daily_dd_limit_pct, max_dd_limit_pct
    ("FundedNext Stellar 2-step  P1", 8.0, 5.0, 10.0),
    ("FundedNext Stellar 2-step  P2", 5.0, 5.0, 10.0),
    ("FundedNext Stellar 1-step    ", 10.0, 3.0, 6.0),
    ("5ers Hyper-growth (rep.)     ", 8.0, 5.0, 5.0),
    ("Generic lenient 8/10         ", 8.0, 5.0, 10.0),
]


def main():  # pragma: no cover - analysis driver
    lo = pd.Timestamp("2011-01-01", tz="UTC")
    hi = pd.Timestamp("2020-12-31 23:59:59", tz="UTC")
    names, w_eq, w_rp, cosim = _build_4way_contiguous()

    # per-year book ROIs at RP weights for the annual-Sharpe reconciliation
    book_rp_capoff = cosim(w_rp, False)

    print("\n" + "=" * 78)
    print("4-WAY BOOK -- PROP-FIRM DEPLOYABILITY (contiguous 2011-2020 IS curve)")
    print("arc 2033: Sharpe/Sortino + profit-target-vs-maxDD challenge feasibility")
    print("=" * 78)

    profiles = {}
    for label, w in [("equal      ", w_eq), ("risk-parity", w_rp)]:
        for cap_label, cap in [("cap-OFF", False), ("cap-ON ", True)]:
            book = cosim(w, cap)
            prof = compute_risk_profile(book.net_equity, lo=lo, hi=hi)
            shp = compute_sharpe(book.net_equity, lo=lo, hi=hi)
            profiles[(label.strip(), cap_label.strip())] = (prof, shp)
            print(f"\n[{label} | {cap_label}]")
            print(f"  ret {prof.ann_return_pct:+.3f}%/yr | maxDD {prof.max_dd_pct:.3f}% | "
                  f"Calmar {prof.calmar:.3f}")
            print("  " + shp.as_row())

    # annual-fold Sharpe (reconciles arc 1023 t=2.66) at RP weights
    # rebuild per-year book ROIs: cosim returns one curve; use the equity per calendar yr
    eq = book_rp_capoff.net_equity.dropna().sort_index()
    eq = eq.loc[(eq.index >= lo) & (eq.index <= hi)]
    yr_end = eq.resample("1YE").last()
    yr_start = eq.resample("1YE").first()
    fold_rois = ((yr_end.values / yr_start.values) - 1.0) * 100.0
    a_sharpe, a_mu, a_sd = annual_sharpe_from_fold_rois(fold_rois)
    print("\n" + "-" * 78)
    print("ANNUAL-FOLD Sharpe (RP cap-OFF, reconciles arc 1023 t-stat):")
    print(f"  per-year book ROI mean {a_mu:+.3f}% | sd {a_sd:.3f}% | "
          f"annual Sharpe {a_sharpe:+.3f} | implied t = Sharpe*sqrt(10) = {a_sharpe*np.sqrt(10):+.3f}")

    # ---- prop-firm-challenge feasibility ----
    print("\n" + "=" * 78)
    print("PROP-FIRM-CHALLENGE FEASIBILITY  (T_min = (target/maxDD)/Calmar years AT the DD limit)")
    print("via linear leverage; arc 1024: ROI & DD scale linearly => Calmar fixed)")
    print("=" * 78)
    for clabel, P, daily_lim, D in _CHALLENGES:
        print(f"\n{clabel}  [target {P:.0f}% | daily {daily_lim:.0f}% | maxDD {D:.0f}%]  P/D={P/D:.2f}")
        for label, cap_label in [("risk-parity", "cap-OFF"), ("risk-parity", "cap-ON"),
                                 ("equal", "cap-OFF")]:
            prof, _ = profiles[(label, cap_label)]
            C = prof.calmar
            T = feasibility_horizon_years(C, P, D)
            # leverage f to reach target at T_min; daily-cap headroom check (worst-day scales by f)
            f_at_Tmin = P / (prof.ann_return_pct * T) if (prof.ann_return_pct * T) > 1e-12 else float("inf")
            print(f"    {label:11s} {cap_label} Calmar {C:.3f} -> T_min {T:5.1f} yr "
                  f"(at DD limit; safe ~2x = {2*T:4.1f} yr); leverage x{f_at_Tmin:.1f}")


if __name__ == "__main__":  # pragma: no cover
    main()

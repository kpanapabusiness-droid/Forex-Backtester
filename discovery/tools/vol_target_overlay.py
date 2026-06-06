"""Ex-ante volatility-targeting overlay on an already-computed book equity curve.

EXPERIMENT / analysis tool — GEOMETRY ONLY on an ALREADY-SCORED net-equity
`pd.Series` (the canonical co-sim book curve, `core/wfo/cosim_book.py`). It NEVER
realizes P&L, never scores a trade, never touches the gate. It applies a CAUSAL
(no-lookahead) inverse-volatility leverage to the book's realized per-bar P&L and
recomputes (a) the per-calendar-year ROI sign pattern (the all-folds-positive check)
and (b) the deployment risk geometry (Calmar / max-DD / underwater, via the BUILT
`equity_risk_profile.compute_risk_profile`).

WHY this exists (arc 1036). Two open questions the corpus left UN-MEASURED:
  1. The pre-emptive note (`discovery/DISCOVERY_DIRECTION.md`, `NEEDS_ENABLEMENT.md`)
     dismisses a vol-target overlay as "ex-ante it merely scales a fold, it cannot
     turn a negative-expectancy fold positive." That reasoning treats vol-target as a
     PER-FOLD SCALAR. Real vol-targeting scales at SUB-YEAR (daily) resolution — IF a
     losing year's losses concentrate in high-vol stretches, causal inverse-vol sizing
     could flip the fold WITHOUT lookahead. Never tested at sub-year resolution.
  2. Arc 1033 proved CONSTANT leverage cannot improve the book's weak Calmar
     (0.24-0.36) — ROI and DD scale together. TIME-VARYING leverage (vol-target) was
     explicitly left open, and for REVERSION edges (which may live in high vol) the
     sign of the Calmar effect is genuinely uncertain.

DESIGN — neutral, leverage-conserving, causal:
  - Work on the contiguous constant-notional book curve: dP_t = net_equity.diff() is
    per-bar $ P&L (cosim is additive, never re-sizes off a running balance), so a
    linear per-year sum of dP is the faithful non-compounding ROI convention.
  - Estimate realized vol on a DAILY-resampled P&L series (rolling std), LAGGED one
    day (causal — vol at t uses only data through t-1).
  - Leverage L_t = clip(IS-median-vol / vol_{t-1}, 0, L_max), then MEAN-NORMALIZED to
    mean(L) == 1.0 exactly — this REDISTRIBUTES a fixed average exposure across time by
    vol state WITHOUT any net leverage change, isolating the vol-TIMING effect (not a
    leverage bet; the clip-above alone leaves mean>1, so normalization is load-bearing).
    L_max caps the low-vol blow-up. A time-shuffled L (same marginal) is the random null.
  - Scaled P&L dP'_t = L_t * dP_t; scaled equity = SB + cumsum(dP'); per-year ROI' =
    Σ_{t∈y} dP'_t / SB.

CAVEAT (documented, not hidden): this is a FIRST-ORDER overlay on the already-netted
return series — a faithful vol-targeted book would re-run sizing through the engine
(the 5%-daily-DD cap would re-bind), which is a code change (human-gated). We use the
cap-OFF curve as the clean monotone bound (no cap-interaction confound; matches arc
1033's convention) and report the cap caveat.

Reproduce:  PYTHONPATH=. py discovery/tools/vol_target_overlay.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from discovery.tools.equity_risk_profile import compute_risk_profile


@dataclass(frozen=True)
class OverlayResult:
    label: str
    lookback_days: int
    l_max: float
    mean_leverage: float
    corr_lev_pnl: float          # corr(L_t, daily P&L) — the WHY: does leverage rise when the book earns?
    per_year_roi: dict           # year -> scaled ROI %
    n_pos: int
    worst_fold_pct: float
    ann_return_pct: float
    max_dd_pct: float
    calmar: float
    longest_underwater_days: int


def _per_year_roi_pct(pnl_by_day: pd.Series, sb: float) -> dict:
    """Linear (non-compounding, constant-notional) per-calendar-year ROI %."""
    out = {}
    for yr, grp in pnl_by_day.groupby(pnl_by_day.index.year):
        out[int(yr)] = float(grp.sum() / sb * 100.0)
    return out


_IS_LO = pd.Timestamp("2011-01-01", tz="UTC")
_IS_HI = pd.Timestamp("2020-12-31 23:59:59", tz="UTC")


def _leverage_series(dP_day: pd.Series, lookback_days: int, l_max: float) -> pd.Series:
    """Causal inverse-vol leverage, MEAN-NORMALIZED to 1.0 (leverage-conserving).

    Mean-normalization is what isolates the vol-TIMING effect from any net leverage
    change: the clip-above makes the raw inverse-vol mean>1, so without this the
    overlay would secretly be a leverage bet. Dividing by its own mean makes it a pure
    redistribution of a fixed average exposure across time by vol state.
    """
    vol = dP_day.rolling(lookback_days, min_periods=max(5, lookback_days // 3)).std().shift(1)
    target = float(vol.median())
    lev = (target / vol).clip(upper=l_max).fillna(1.0)
    return lev / float(lev.mean())  # mean-normalize -> mean leverage == 1.0 exactly


def vol_target_overlay(
    net_equity: pd.Series,
    *,
    starting_balance: float = 100_000.0,
    lookback_days: int = 60,
    l_max: float = 3.0,
    label: str = "",
    leverage: pd.Series | None = None,
) -> OverlayResult:
    """Apply a causal, leverage-conserving inverse-vol overlay to a book curve.

    `net_equity` is the contiguous `SB + Σ w_k·pnl_k(t)` curve from `cosim_book_fold`.
    Returns the scaled book's per-year ROI sign pattern + deployment risk geometry
    (risk geometry on the 2011-2020 IS slice only — matching arc 1033). Pass
    `leverage` to inject a custom (e.g. random-null) mean-1 leverage series.
    """
    eq = net_equity.dropna().sort_index()
    # daily-resampled P&L (sum of per-bar increments within each UTC day)
    dP_bar = eq.diff().dropna()
    dP_day = dP_bar.resample("1D").sum()
    dP_day = dP_day[dP_day.index >= dP_bar.index[0]]

    lev = leverage if leverage is not None else _leverage_series(dP_day, lookback_days, l_max)
    lev = lev.reindex(dP_day.index).fillna(1.0)

    scaled_day = lev * dP_day
    # contiguous scaled equity, sliced to the IS window for the risk geometry
    scaled_eq = pd.Series(starting_balance + scaled_day.cumsum().to_numpy(),
                          index=scaled_day.index, name="equity")
    scaled_eq = scaled_eq[(scaled_eq.index >= _IS_LO) & (scaled_eq.index <= _IS_HI)]

    per_year = _per_year_roi_pct(scaled_day[(scaled_day.index >= _IS_LO) & (scaled_day.index <= _IS_HI)],
                                 starting_balance)
    n_pos = sum(1 for v in per_year.values() if v > 0)
    worst = min(per_year.values())

    prof = compute_risk_profile(scaled_eq)
    # corr(leverage, daily pnl): >0 => levers UP when earning (good); ~0 => no information
    valid = (~lev.isna()) & (~dP_day.isna())
    corr = float(np.corrcoef(lev[valid], dP_day[valid])[0, 1]) if valid.sum() > 3 else float("nan")

    return OverlayResult(
        label=label, lookback_days=lookback_days, l_max=l_max,
        mean_leverage=float(lev.mean()), corr_lev_pnl=corr,
        per_year_roi=per_year, n_pos=n_pos, worst_fold_pct=worst,
        ann_return_pct=prof.ann_return_pct, max_dd_pct=prof.max_dd_pct,
        calmar=prof.calmar, longest_underwater_days=prof.longest_underwater_days,
    )


def baseline_profile(net_equity: pd.Series, starting_balance: float = 100_000.0):
    """Unscaled (leverage=1) per-year ROI + risk geometry, on the SAME daily grid,
    IS-sliced (2011-2020) to match arc 1033."""
    eq = net_equity.dropna().sort_index()
    dP_day = eq.diff().dropna().resample("1D").sum()
    eq_day = pd.Series(starting_balance + dP_day.cumsum().to_numpy(), index=dP_day.index, name="equity")
    eq_day = eq_day[(eq_day.index >= _IS_LO) & (eq_day.index <= _IS_HI)]
    pys = dP_day[(dP_day.index >= _IS_LO) & (dP_day.index <= _IS_HI)]
    per_year = _per_year_roi_pct(pys, starting_balance)
    prof = compute_risk_profile(eq_day)
    n_pos = sum(1 for v in per_year.values() if v > 0)
    return per_year, n_pos, min(per_year.values()), prof


# ───────────────────────── __main__: overlay on the 4-way book ─────────────────────────
def main():  # pragma: no cover - analysis driver
    from discovery.tools.equity_risk_profile import _build_4way_contiguous

    names, w_eq, w_rp, cosim = _build_4way_contiguous()
    book = cosim(w_rp, False)  # risk-parity, cap-OFF (the operative monotone bound)
    sb = 100_000.0
    eq = book.net_equity

    print("\n" + "=" * 84)
    print("ARC 1036 -- VOL-TARGET OVERLAY ON THE 4-WAY BOOK (risk-parity, cap-OFF)")
    print("causal inverse-vol, leverage-conserving (target = IS-median trailing vol)")
    print("=" * 84)

    by, bnpos, bworst, bprof = baseline_profile(eq, sb)
    print("\n[BASELINE  leverage=1, IS 2011-2020 daily grid]")
    print(f"  per-year ROI%: " + " ".join(f"{y}:{v:+.2f}" for y, v in sorted(by.items())))
    print(f"  {bnpos}/10 pos | worst-fold {bworst:+.3f}% | ret {bprof.ann_return_pct:+.3f}%/yr | "
          f"Calmar {bprof.calmar:.3f} | maxDD {bprof.max_dd_pct:.3f}% | underwater {bprof.longest_underwater_days}d")

    print("\n[VOL-TARGET OVERLAYS — leverage-conserving (mean lev == 1.0), IS 2011-2020]")
    for lookback in (20, 40, 60, 120):
        for l_max in (2.0, 3.0):
            r = vol_target_overlay(eq, starting_balance=sb, lookback_days=lookback, l_max=l_max,
                                   label=f"lb{lookback}_Lmax{l_max:g}")
            print(f"\n  [{r.label}]  mean_lev={r.mean_leverage:.2f}  corr(lev,pnl)={r.corr_lev_pnl:+.3f}")
            print(f"    per-year ROI%: " + " ".join(f"{y}:{v:+.2f}" for y, v in sorted(r.per_year_roi.items())))
            print(f"    {r.n_pos}/10 pos | worst-fold {r.worst_fold_pct:+.3f}% | ret {r.ann_return_pct:+.3f}%/yr | "
                  f"Calmar {r.calmar:.3f} | maxDD {r.max_dd_pct:.3f}% | underwater {r.longest_underwater_days}d")

    # ── RANDOM-LEVERAGE NULL (arc-1021 discipline): same mean-1 leverage MARGINAL,
    #    shuffled in time, deterministic seed — is the vol-target overlay distinguishable
    #    from random reweighting of the same average exposure?
    print("\n[RANDOM-LEVERAGE NULL — vol-target leverage MARGINAL, time-shuffled (seed 42, 200 draws)]")
    dP_bar = eq.dropna().sort_index().diff().dropna()
    dP_day = dP_bar.resample("1D").sum()
    dP_day = dP_day[dP_day.index >= dP_bar.index[0]]
    lev60 = _leverage_series(dP_day, 60, 3.0).reindex(dP_day.index).fillna(1.0)
    for lb in (20, 60):
        lev_m = _leverage_series(dP_day, lb, 3.0).reindex(dP_day.index).fillna(1.0)
        rng = np.random.default_rng(42)
        marg = lev_m.to_numpy()
        calmars, worsts = [], []
        for _ in range(200):
            shuf = pd.Series(rng.permutation(marg), index=dP_day.index)
            rr = vol_target_overlay(eq, starting_balance=sb, leverage=shuf, label="null")
            calmars.append(rr.calmar)
            worsts.append(rr.worst_fold_pct)
        real = vol_target_overlay(eq, starting_balance=sb, lookback_days=lb, l_max=3.0, label=f"lb{lb}")
        calmars, worsts = np.array(calmars), np.array(worsts)
        print(f"\n  REAL vol-target lb{lb}_Lmax3: Calmar {real.calmar:.3f} | worst-fold {real.worst_fold_pct:+.3f}%")
        print(f"    NULL Calmar: mean {calmars.mean():.3f}  p5 {np.percentile(calmars,5):.3f}  "
              f"p95 {np.percentile(calmars,95):.3f}  | P(null Calmar >= real) = {(calmars>=real.calmar).mean():.3f}")
        print(f"    NULL worst-fold: mean {worsts.mean():+.3f}%  p5 {np.percentile(worsts,5):+.3f}  "
              f"p95 {np.percentile(worsts,95):+.3f}  | P(null worst >= real) = {(worsts>=real.worst_fold_pct).mean():.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""judge_power_audit — statistical POWER characterization of the discovery judge.

EXPERIMENT tool (arc 1063, chat 1000s). Built per the discovery council's
convergent recommendation: before trusting the ~90-arc "frontier exhausted"
verdict, audit whether the SOLE judge — ``all-folds-positive per calendar year``
(``core.wfo.discovery_measure.judge_all_folds_positive``) — has the statistical
POWER to certify a genuinely positive-EV edge of the survivors' thinness, or
whether it rejects real edges by per-fold sign-test sampling noise (Type-II).

This tool does NOT loosen, replace, or re-implement the gate. It CALLS the
canonical ``judge_all_folds_positive`` on SYNTHETIC per-fold ``FoldStats`` of
KNOWN expected value to map the judge's pass-probability as a function of the
true edge's per-year Sharpe and the number of folds. It touches no price data,
no engine, and no OOS holdout (§4 untouched — there is no real data here at all).

Two levels:
  (1) per-YEAR analytic/MC backbone — pass-rate vs true per-year Sharpe S=mu/sd
      and fold-count K (the clean, distribution-light result), plus placement of
      the real survivors using their PUBLISHED realized per-year ROI series.
  (2) per-TRADE convex-R Monte-Carlo — a stylized SL-honest take-the-loss
      distribution (capped -1R downside, +1R partials, fat right runner tail) to
      show the fat-tail penalty on top of the Gaussian backbone (ties the
      corpus's "tail-fragile" lessons, arcs 1057/2059).

Determinism: numpy default_rng(seed=42).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from core.wfo.discovery_measure import judge_all_folds_positive
from core.wfo.gates import FoldStats


# --------------------------------------------------------------------------- #
# canonical-judge adapter: build synthetic FoldStats and ask the REAL judge
# --------------------------------------------------------------------------- #
def _fold_stats_from_rois(rois, n_per_fold: int) -> list[FoldStats]:
    """Wrap a sequence of per-fold ROIs as FoldStats (judge reads roi_pct only)."""
    return [
        FoldStats(
            fold_id=i,
            n_trades=int(n_per_fold),
            roi_pct=float(r),
            max_dd_pct=1.0,            # inert for the judge
            days_breaching_daily_5pct=0,
            roi_dd_ratio=0.0,
        )
        for i, r in enumerate(rois)
    ]


def judge_passes(rois, n_per_fold: int = 100) -> bool:
    """True iff the CANONICAL discovery judge passes this per-fold ROI series."""
    return judge_all_folds_positive(_fold_stats_from_rois(rois, n_per_fold)).all_folds_positive


def _phi(x: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# --------------------------------------------------------------------------- #
# (1) per-YEAR backbone — Gaussian analytic + MC cross-check via the real judge
# --------------------------------------------------------------------------- #
def gaussian_all_folds_prob(per_year_sharpe: float, k_folds: int) -> float:
    """Analytic P(all K independent Gaussian years > 0) = Phi(S)^K.

    A year's ROI ~ N(mu, sd); P(year>0)=Phi(mu/sd)=Phi(S). Folds independent ⇒
    product. (Per-year Sharpe S = mu_year / sd_year — the realized per-fold
    mean-over-sd, NOT an annualized ratio.)
    """
    return _phi(per_year_sharpe) ** k_folds


def mc_all_folds_prob_gaussian(
    per_year_sharpe: float, k_folds: int, n_sims: int = 200_000, seed: int = 42
) -> float:
    """MC cross-check of the analytic backbone, routed through the CANONICAL judge.

    Draws K Gaussian years with mean=S, sd=1 (so per-year Sharpe = S), feeds each
    synthetic series to ``judge_all_folds_positive``, returns the pass fraction.
    Confirms the adapter + judge reproduce Phi(S)^K (sanity that we are auditing
    the real rule, not a hand re-impl).
    """
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=per_year_sharpe, scale=1.0, size=(n_sims, k_folds))
    passes = sum(judge_passes(row) for row in draws[:2000])  # judge-call sample
    judge_rate = passes / 2000
    vector_rate = float(np.mean(np.all(draws > 0.0, axis=1)))  # full-MC vectorized
    return judge_rate, vector_rate


# --------------------------------------------------------------------------- #
# (2) per-TRADE convex SL-honest sampler (stylized take-the-loss)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConvexRProfile:
    """Stylized per-trade R distribution for an SL-honest partial/runner fade.

    p_reach1r : P(reach +1R partial before stop) — the honest capture.
    runner_mean, runner_tail : the half-position runner's R after the +1R partial
        is taken (modeled lognormal-ish via exponential tail); the other half
        books exactly +1.000R at the partial (take-the-loss invariant).
    On a stop (prob 1-p_reach1r): R = -1.0 (capped).
    Position R when partial reached = 0.5*(+1.0) + 0.5*(runner_R).
    """

    p_reach1r: float = 0.55
    runner_mean: float = 0.9
    runner_tail: float = 1.4  # exponential scale of the right tail

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        reach = rng.random(size) < self.p_reach1r
        runner = rng.exponential(scale=self.runner_tail, size=size) + (
            self.runner_mean - self.runner_tail
        )
        runner = np.clip(runner, -1.0, None)  # runner can give back to ~-1R (BE-ish)
        pos_r = np.where(reach, 0.5 * 1.0 + 0.5 * runner, -1.0)
        return pos_r

    def per_trade_mean(self, rng: np.random.Generator) -> float:
        return float(np.mean(self.sample(2_000_000, rng)))


def mc_all_folds_prob_convex(
    target_per_trade_mean_R: float,
    n_per_year: int,
    k_folds: int,
    profile: ConvexRProfile,
    n_sims: int = 100_000,
    seed: int = 42,
) -> float:
    """MC pass-rate for a CONVEX per-trade edge shifted to a target per-trade mean.

    Draws n_per_year*k_folds trades from ``profile`` per sim, additively shifts so
    the per-trade mean == target (preserving the convex shape/skew), sums each
    year, and asks the canonical judge whether all K years > 0. Risk scale cancels
    for the sign test. Returns pass fraction over n_sims.
    """
    rng = np.random.default_rng(seed)
    base = profile.sample(4_000_000, rng)
    shift = target_per_trade_mean_R - float(np.mean(base))
    total = n_sims * k_folds * n_per_year
    trades = profile.sample(total, rng) + shift
    year_sums = trades.reshape(n_sims, k_folds, n_per_year).sum(axis=2)
    # vectorized all-folds rule (validated == canonical judge in driver)
    return float(np.mean(np.all(year_sums > 0.0, axis=1)))


# --------------------------------------------------------------------------- #
# (1b) real-survivor placement from PUBLISHED per-year ROI series
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SurvivorSeries:
    name: str
    is_years: tuple[float, ...]   # realized per-year IS ROI %, one per fold
    oos_years: tuple[float, ...]  # realized per-year OOS ROI %

    def stats(self):
        a = np.asarray(self.is_years, dtype=float)
        mu, sd = float(a.mean()), float(a.std(ddof=1))
        sharpe = mu / sd if sd > 0 else float("nan")
        return {
            "name": self.name,
            "n_is_folds": len(self.is_years),
            "is_mean": mu,
            "is_sd": sd,
            "is_per_year_sharpe": sharpe,
            "is_n_pos": int((a > 0).sum()),
            "p_all_is_gauss": gaussian_all_folds_prob(sharpe, len(self.is_years)),
            "judge_is_pass": judge_passes(self.is_years),
        }

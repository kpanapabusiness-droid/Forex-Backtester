"""arc 1063 (chat 1000s) — POWER AUDIT of the all-folds-positive discovery judge.

Council-recommended (arc-1063 generative council, convergent reviewer finding):
characterize the SOLE judge's statistical power before trusting the ~90-arc
"frontier exhausted" verdict. Does NOT loosen the gate — quantifies it (cf. arc
1023, which "quantifies why, does NOT loosen it"). No price/engine/OOS touched.

Run:  python discovery/_disco1_work/arc1063_judge_power_audit.py
"""

from __future__ import annotations

import numpy as np

from discovery.tools.judge_power_audit import (
    ConvexRProfile,
    SurvivorSeries,
    gaussian_all_folds_prob,
    judge_passes,
    mc_all_folds_prob_convex,
    mc_all_folds_prob_gaussian,
)
from core.wfo.discovery_measure import build_oos_year_folds
from core.wfo.folds import build_v3_folds

SEP = "=" * 78


def _sharpe_for_target(k_folds: int, target_pass: float) -> float:
    """Per-year Sharpe S such that Phi(S)^k == target_pass (analytic inverse)."""
    from math import sqrt
    from statistics import NormalDist

    nd = NormalDist()
    phi_needed = target_pass ** (1.0 / k_folds)
    return nd.inv_cdf(phi_needed)


def main() -> None:
    # ----- 0. canonical fold COUNTS (call the canonical builders) ------------
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    oos_folds = build_oos_year_folds(start_year=2021)
    K_IS, K_OOS = len(is_folds), len(oos_folds)
    K_JOINT = K_IS + K_OOS
    print(SEP)
    print("CANONICAL judged fold-set sizes (build_v3_folds + build_oos_year_folds):")
    print(f"  IS annual folds   K_IS   = {K_IS}")
    print(f"  OOS annual folds  K_OOS  = {K_OOS}")
    print(f"  joint             K_JOINT= {K_JOINT}")

    # ----- 1. adapter sanity: canonical judge reproduces Phi(S)^K ------------
    print(SEP)
    print("ADAPTER SANITY — canonical judge vs analytic Phi(S)^K (S=0.84, K=10):")
    jr, vr = mc_all_folds_prob_gaussian(0.84, 10)
    print(f"  canonical-judge MC = {jr:.4f}   vectorized MC = {vr:.4f}   "
          f"analytic = {gaussian_all_folds_prob(0.84, 10):.4f}")
    assert abs(jr - vr) < 0.03, "judge adapter disagrees with vectorized rule"

    # ----- 2. power backbone: Sharpe needed to pass at K folds ---------------
    print(SEP)
    print("JUDGE POWER — per-year Sharpe S (=mean/sd of the per-year ROI series)")
    print("needed for a given all-folds-positive PASS probability:")
    print(f"{'target P(pass)':>16} | {'K_IS='+str(K_IS):>10} | {'K_OOS='+str(K_OOS):>10} | {'joint='+str(K_JOINT):>11}")
    for tp in (0.50, 0.80, 0.95):
        s_is = _sharpe_for_target(K_IS, tp)
        s_oos = _sharpe_for_target(K_OOS, tp)
        s_joint = _sharpe_for_target(K_JOINT, tp)
        print(f"{tp:>16.2f} | {s_is:>10.2f} | {s_oos:>10.2f} | {s_joint:>11.2f}")

    print("\nP(all-folds-positive) at representative per-year Sharpe levels:")
    print(f"{'per-year Sharpe':>16} | {'P(IS '+str(K_IS)+')':>10} | {'P(OOS '+str(K_OOS)+')':>11} | {'P(joint '+str(K_JOINT)+')':>13}")
    for s in (0.50, 0.84, 1.00, 1.50, 2.00):
        print(f"{s:>16.2f} | {gaussian_all_folds_prob(s, K_IS):>10.3f} | "
              f"{gaussian_all_folds_prob(s, K_OOS):>11.3f} | "
              f"{gaussian_all_folds_prob(s, K_JOINT):>13.3f}")

    # ----- 3. real-survivor placement (PUBLISHED per-year ROI series) --------
    print(SEP)
    print("REAL SURVIVORS placed on the power curve (published per-year ROI %):")
    # me_long: published IS 8-fold + OOS 6-fold series (arcs 1056/1058).
    me_long = SurvivorSeries(
        name="me_long",
        is_years=(0.96, -0.23, -1.14, -0.51, 0.34, 0.90, 1.16, 0.15),
        oos_years=(-0.63, 0.55, 0.18, 0.54, 2.26, 0.02),
    )
    for s in (me_long,):
        st = s.stats()
        print(f"  {st['name']:8} IS: mean {st['is_mean']:+.3f}% sd {st['is_sd']:.3f}% "
              f"Sharpe {st['is_per_year_sharpe']:+.3f}  {st['is_n_pos']}/{st['n_is_folds']} pos  "
              f"P(all-IS)={st['p_all_is_gauss']:.3f}  judge_pass={st['judge_is_pass']}")
        oa = np.asarray(s.oos_years)
        s_oos = oa.mean() / oa.std(ddof=1)
        print(f"           OOS: mean {oa.mean():+.3f}% sd {oa.std(ddof=1):.3f}% "
              f"Sharpe {s_oos:+.3f}  {(oa>0).sum()}/{len(oa)} pos  "
              f"P(all-OOS)={gaussian_all_folds_prob(s_oos, len(oa)):.3f}  "
              f"judge_pass={judge_passes(s.oos_years)}")

    # 4-way book: arc 1023 published mean/sd across 10 folds (independent anchor)
    book_mu, book_sd, book_K = 0.624, 0.741, 10
    book_S = book_mu / book_sd
    print(f"\n  4-way BOOK (arc 1023: mean {book_mu:+.3f}% / across-fold sd {book_sd:.3f}% / K={book_K}):")
    print(f"     per-year Sharpe = {book_S:.3f}")
    print(f"     my backbone  P(all-{book_K}-folds) = {gaussian_all_folds_prob(book_S, book_K):.4f}")
    print(f"     arc-1023 independent bootstrap     = 0.1130  (P(>=1 neg)=0.887)")
    print("     ^ CROSS-CHECK: analytic backbone reproduces arc-1023's independent number.")

    # ----- 4. fat-tail penalty: convex SL-honest per-trade MC ---------------
    print(SEP)
    print("FAT-TAIL PENALTY — convex SL-honest take-the-loss edge vs Gaussian,")
    print("matched per-YEAR mean, at survivor thinness (n/yr trades, K_IS folds):")
    profile = ConvexRProfile(p_reach1r=0.55, runner_mean=0.9, runner_tail=1.4)
    rng = np.random.default_rng(42)
    base_mean = profile.per_trade_mean(rng)
    print(f"  convex profile base per-trade mean R = {base_mean:+.4f} "
          f"(p_reach1r=0.55, runner tail to ~+8R)")
    for n_yr, edge_R in ((12, 0.05), (20, 0.05), (20, 0.10), (25, 0.10)):
        p_convex = mc_all_folds_prob_convex(edge_R, n_yr, K_IS, profile, n_sims=60_000)
        # matched Gaussian: same per-year mean = n_yr*edge_R, per-trade sd of profile
        sd_trade = float(np.std(profile.sample(2_000_000, rng)))
        per_year_S = (n_yr * edge_R) / (np.sqrt(n_yr) * sd_trade)
        p_gauss = gaussian_all_folds_prob(per_year_S, K_IS)
        print(f"  n/yr={n_yr:3d} edge={edge_R:+.2f}R/trade  per-yr Sharpe~{per_year_S:.2f}  "
              f"P(all-IS) convex={p_convex:.3f}  Gaussian={p_gauss:.3f}  "
              f"penalty x{(p_gauss/p_convex if p_convex>0 else float('inf')):.2f}")

    print(SEP)
    print("DONE.")


if __name__ == "__main__":
    main()

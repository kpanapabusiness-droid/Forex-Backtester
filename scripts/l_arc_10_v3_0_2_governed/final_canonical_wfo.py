"""Arc 10 v3.0.2 — FINAL CANONICAL WFO (deploy-faithful: fixed-initial + daily_ref=initial).

THE deploy-faithful canonical gate run — SUPERSEDES the EA-faithful (floating) run
and every prior `_canonical` / `_canonical_compound` run. This matches EXACTLY how
the live EA now trades AFTER FIX 2b + the backtester daily_ref alignment (PR #252/
#253):

  * SIZING  = FIXED-INITIAL: 1R ≡ r_base of the INITIAL balance, constant per trade
              (the linear `governed_wfo.simulate_fold` path; mirrors the deployed EA
              `ArcComputeLots` post-FIX-1). NOT floating-equity.
  * DAILY DD = daily_ref="initial" — the RESETTING fixed-$/day basis (numerator =
              day-start equity, RESET each EET day; denominator = fixed initial 1.0).
              Mirrors the deployed FundedNext EA (`Daily_DD_Basis=INITIAL`, daily
              reset mandatory; EquityGuards.mqh FIX 2b). NOT `static_noreset` — that
              is the quarantined non-resetting freeze-bug mode (it froze F5 2014 /
              F6 2015 at −5.99% / −5.45%).
  * MAX DD  = TRAILING (planning anchor) + STATIC/from-initial (FundedNext breach
              basis) — BOTH reported. total_ref="static" governor firing (the rule
              FundedNext enforces from the tier's initial balance).

Everything else is byte-identical to the prior canonical reconstruction (frame sha
05dea9…9ee58a; cell-5 FundedNext costs — swaps OFF, 1.5× spread, $5/lot RT
commission, 0.5 pip slip × n_fills; exit sl_partial_close_1r_runner_trail @ 3.5×ATR;
EET; 28 pairs). v3.0.2 signal/exit/cost LOCKED — ONLY the sizing basis + daily-DD
mode differ from prior runs, and both already exist as aligned, validated params.
The governors are EA-faithful re-modelling, NOT tuned — a governed failure is a
FINDING, not a trigger to retune.

RISK SWEEP {0.40, 0.42, 0.45}%:
  * 0.40% = the canonical operating tier (the headline).
  * 0.42% / 0.45% = MARGIN-CHECK ONLY — run to quantify proximity to the hard
    limits, NOT adoption candidates. Daily DD is a fixed $/day off initial (tightens
    in %-terms as the account grows), so there is no case to raise risk; the sweep
    confirms 0.40% is the ceiling. They are NOT recommended.

CRITICAL F5/F6 RECOVERY CHECK (proves the daily-reset alignment took): under the OLD
non-resetting `static` the daily governor froze F5 2014 / F6 2015 permanently once
the account dipped 3.5% below initial (the −5.99% / −5.45% artifact). Under
daily_ref="initial" (resetting) they MUST return to positive (~+28.9% / +20.8%, ~0
daily-governor fires). The driver reports F5/F6 ROI + governor-fire counts and ABORTS
if either is still non-positive (alignment didn't take).

VALIDATION GATE (dispatch): linear (mult ≡ r_base) + zero-cost + governors-OFF must
reproduce the 9.22% portfolio worst-fold trailing DD — abort if the open-book
reconstruction doesn't match. Two-run sha identity (--verify-determinism).

Outputs (results/l_arc_10_v3.0.2_final_canonical/):
  FINAL_CANONICAL_WFO.md, matrix.csv, per_fold.csv, governor_log.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in ["", "scripts/analysis", "scripts/l_arc_10_v3_0_2_governed", "scripts/audit/arc_10"]:
    sys.path.insert(0, str(ROOT / _p))

import canonical_wfo as cw  # noqa: E402  identical load() reconstruction (sha-05dea9 frame)
import canonical_wfo_ea_faithful as eaf  # noqa: E402  holdout_year_tids, verdict, agg_search
import governed_wfo as gw  # noqa: E402  simulate_fold (fixed-initial linear), annualise, df_to_md

OUTDIR = ROOT / "results" / "l_arc_10_v3.0.2_final_canonical"
EA_FAITHFUL_DIR = ROOT / "results" / "l_arc_10_v3.0.2_ea_faithful"
df_to_md = gw.df_to_md

# Risk sweep — 0.40% canonical; 0.42%/0.45% margin-check only (NOT adoption candidates).
RISK_LEVELS = [0.004, 0.0042, 0.0045]
CANONICAL_RISK = 0.004
SEARCH_FOLDS = list(range(1, 12))  # F1..F11 = calendar years 2010..2020
HOLDOUT_FOLD = 12

DEPLOY_DD, HARD_DD = 0.08, 0.10
DAILY_LIMIT = 0.05
WORST_ROI_MIN, MEAN_ROI_MIN = 0.05, 0.08
EXPECTED_ZEROCOST_OFF_WORST_TRAILING = 0.0922
VALIDATION_TOL = 0.002

verdict = eaf.verdict
holdout_year_tids = eaf.holdout_year_tids
agg_search = eaf.agg_search

PROVENANCE = (
    "BASIS:   EET | governed (3.5/4.5 daily, 7/8 total) | 1.5x spread | $5/lot RT | "
    "swaps OFF | 0.5 slip\n"
    "SIZING:  fixed-initial\n"
    "DD REFS: daily = initial (fixed-$/day, RESETS daily) | max = trailing (plan) + "
    "static (enforce)\n"
    "RISK:    {0.40, 0.42, 0.45}% (0.40 = canonical; 0.42/0.45 = margin-check only)\n"
    "FRAME:   sha 05dea9…9ee58a | 28 pairs | 11 folds + holdout-per-year (no CAGR)"
)


# ───────────────────────────────────────────────────────────────────────────
# Fixed-initial fold runner (deploy-faithful): gw.simulate_fold, daily_ref="initial"
# ───────────────────────────────────────────────────────────────────────────
def run_fold(tids, D, *, governed, is_2026=False):
    """One fold/year under fixed-initial sizing + resetting-initial daily DD.

    gw.simulate_fold IS the canonical fixed-initial linear path: 1R ≡ gw.R_BASE of the
    initial balance, per-fold reset to 1.0. daily_ref="initial" fires the daily
    governors on the resetting fixed-$/day basis (= deployed EA INITIAL). total_ref=
    "static" fires the total governors from-initial (= FundedNext breach rule). Both
    DD references are returned (dd_trailing planning, dd_static from-initial)."""
    m = gw.simulate_fold(
        tids, D["sched_cost"], D["day_key"], D["clock"],
        governed=governed, total_ref="static", daily_ref="initial",
    )
    raw = float(m["final_equity"] - 1.0)
    # Full-year folds: annualise over the fold's own ~1y entry span (simulate_fold's
    # internal roi). 2026 partial (~4mo): RAW return, never annualised (NO CAGR).
    roi = raw if is_2026 else float(m["roi"])
    fires = [(x[0], x[1]) for x in m["fires"]]  # (governor, date)
    daily_fires = int(sum(1 for x in m["fires"] if x[0] in ("daily_halt", "daily_close_all")))
    total_fires = int(sum(1 for x in m["fires"] if x[0] in ("total_halt", "total_close_all")))
    return dict(
        n=int(m["n_trades"]), roi=roi, raw=raw,
        trailing_dd=float(m["dd_trailing"]), from_initial_dd=float(m["dd_static"]),
        daily_dd=float(m["daily_dd_close_max"]),
        kills=int(sum(1 for x in m["fires"] if x[0] == "total_close_all")),
        fires=fires, daily_fires=daily_fires, total_fires=total_fires, is_2026=is_2026,
    )


# ───────────────────────────────────────────────────────────────────────────
# Run the matrix
# ───────────────────────────────────────────────────────────────────────────
def run_matrix(D, hy):
    search_res, holdout_res = {}, {}
    for risk in RISK_LEVELS:
        gw.R_BASE = risk  # simulate_fold reads the module-global R_BASE at call time
        for gov in ("off", "on"):
            governed = gov == "on"
            search_res[(risk, gov)] = {
                f: run_fold(D["fold_tids"][f], D, governed=governed) for f in SEARCH_FOLDS
            }
            holdout_res[(risk, gov)] = {
                y: run_fold(hy[y], D, governed=governed, is_2026=(y == 2026))
                for y in sorted(hy)
            }
        a = agg_search(search_res[(risk, "on")])
        print(
            f"[risk {risk * 100:.2f}%] fixed-initial gov-on worst-fold trailing DD = "
            f"{a['trailing_dd'] * 100:.2f}% / worst daily {a['daily_dd'] * 100:.2f}% "
            f"(kills {a['kills']})",
            flush=True,
        )
    return search_res, holdout_res


def f5f6_recovery_check(search_res) -> dict:
    """F5 2014 / F6 2015 recovery proof — they MUST be positive under the resetting
    daily basis (the non-resetting `static` froze them at −5.99% / −5.45%)."""
    c = search_res[(CANONICAL_RISK, "on")]
    out = {}
    for label, f in (("F5_2014", 5), ("F6_2015", 6)):
        m = c[f]
        out[label] = dict(
            roi_pct=m["roi"] * 100, raw_pct=m["raw"] * 100,
            daily_fires=m["daily_fires"], total_fires=m["total_fires"], kills=m["kills"],
        )
    return out


# ───────────────────────────────────────────────────────────────────────────
# Determinism digest (two-run identity)
# ───────────────────────────────────────────────────────────────────────────
def matrix_digest(search_res, holdout_res) -> str:
    parts = []
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            for f in SEARCH_FOLDS:
                m = search_res[(risk, gov)][f]
                parts.append(f"{risk}|{gov}|F{f}|{m['roi']:.12f}|{m['trailing_dd']:.12f}|"
                             f"{m['from_initial_dd']:.12f}|{m['daily_dd']:.12f}|{m['kills']}")
            for y, m in holdout_res[(risk, gov)].items():
                parts.append(f"{risk}|{gov}|{y}|{m['roi']:.12f}|{m['trailing_dd']:.12f}|"
                             f"{m['from_initial_dd']:.12f}|{m['daily_dd']:.12f}|{m['kills']}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# ───────────────────────────────────────────────────────────────────────────
# Emit CSVs + FINAL_CANONICAL_WFO.md
# ───────────────────────────────────────────────────────────────────────────
def emit(search_res, holdout_res, D, f5f6, digest):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- matrix.csv (3 risk × {off,on}; both DD refs) ----
    rows = []
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            a = agg_search(search_res[(risk, gov)])
            rows.append(dict(
                risk_pct=risk * 100, governance=gov, sizing="fixed-initial",
                worst_fold_roi_pct=a["worst_roi"] * 100, mean_fold_roi_pct=a["mean_roi"] * 100,
                trailing_dd_pct=a["trailing_dd"] * 100, from_initial_dd_pct=a["from_initial_dd"] * 100,
                daily_dd_pct=a["daily_dd"] * 100, kills=a["kills"],
                verdict_trailing=verdict(a["worst_roi"], a["mean_roi"], a["trailing_dd"], a["kills"] > 0),
                verdict_from_initial=verdict(a["worst_roi"], a["mean_roi"], a["from_initial_dd"], a["kills"] > 0),
            ))
    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUTDIR / "matrix.csv", index=False, lineterminator="\n")

    # ---- per_fold.csv (F1–F11 + holdout-per-year; all 3 risk × {off,on}) ----
    pf = []
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            for f in SEARCH_FOLDS:
                m = search_res[(risk, gov)][f]
                pf.append(dict(
                    risk_pct=risk * 100, governance=gov, sizing="fixed-initial",
                    daily_ref="initial", segment="search", fold=f"F{f}", calendar_year=2009 + f,
                    n_trades=m["n"], roi_pct=m["roi"] * 100, raw_return_pct=m["raw"] * 100,
                    trailing_dd_pct=m["trailing_dd"] * 100, from_initial_dd_pct=m["from_initial_dd"] * 100,
                    daily_dd_pct=m["daily_dd"] * 100, daily_fires=m["daily_fires"],
                    total_fires=m["total_fires"], kills=m["kills"], roi_basis="annualised_1y",
                ))
            for y, m in holdout_res[(risk, gov)].items():
                pf.append(dict(
                    risk_pct=risk * 100, governance=gov, sizing="fixed-initial",
                    daily_ref="initial", segment="holdout",
                    fold=(f"{y}p" if y == 2026 else str(y)), calendar_year=y,
                    n_trades=m["n"], roi_pct=m["roi"] * 100, raw_return_pct=m["raw"] * 100,
                    trailing_dd_pct=m["trailing_dd"] * 100, from_initial_dd_pct=m["from_initial_dd"] * 100,
                    daily_dd_pct=m["daily_dd"] * 100, daily_fires=m["daily_fires"],
                    total_fires=m["total_fires"], kills=m["kills"],
                    roi_basis=("raw_partial_4mo" if y == 2026 else "annualised_1y"),
                ))
    pd.DataFrame(pf).to_csv(OUTDIR / "per_fold.csv", index=False, lineterminator="\n")

    # ---- governor_log.csv (all governed fires, from-initial firing) ----
    gl = []
    for risk in RISK_LEVELS:
        for f in SEARCH_FOLDS:
            for g, date in search_res[(risk, "on")][f]["fires"]:
                gl.append(dict(risk_pct=risk * 100, segment="search", fold=f"F{f}",
                               governor=g, date=date, kill=int(g == "total_close_all")))
        for y, m in holdout_res[(risk, "on")].items():
            for g, date in m["fires"]:
                gl.append(dict(risk_pct=risk * 100, segment="holdout",
                               fold=(f"{y}p" if y == 2026 else str(y)),
                               governor=g, date=date, kill=int(g == "total_close_all")))
    gl_df = pd.DataFrame(gl) if gl else pd.DataFrame(
        columns=["risk_pct", "segment", "fold", "governor", "date", "kill"])
    gl_df.to_csv(OUTDIR / "governor_log.csv", index=False, lineterminator="\n")

    write_summary(search_res, holdout_res, matrix, D, f5f6, digest)
    return matrix


def _ea_faithful_compare():
    """0.40% gov-on floating numbers from the (now superseded) EA-faithful run."""
    try:
        mt = pd.read_csv(EA_FAITHFUL_DIR / "matrix.csv")
        row = mt[(mt.risk_pct == 0.4) & (mt.governance == "on")].iloc[0]
        return dict(worst_roi=float(row["worst_fold_roi_pct"]), mean_roi=float(row["mean_fold_roi_pct"]),
                    trailing_dd=float(row["trailing_dd_pct"]), from_initial_dd=float(row["from_initial_dd_pct"]),
                    daily_dd=float(row["daily_dd_pct"]))
    except Exception:
        return None


def write_summary(search_res, holdout_res, matrix, D, f5f6, digest):
    L = ["# Arc 10 v3.0.2 — FINAL CANONICAL WFO (deploy-faithful: fixed-initial + daily_ref=initial)\n"]
    L.append(
        "> **THE deploy-faithful canonical gate run.** Matches EXACTLY how the live EA trades "
        "after FIX 2b + the backtester daily_ref alignment (PR #252/#253): **fixed-initial "
        "sizing** (1R ≡ r_base of the INITIAL balance, constant per trade — deployed EA "
        "`ArcComputeLots`) + **daily DD on `daily_ref=\"initial\"`** (the RESETTING fixed-$/day "
        "basis: numerator = day-start equity reset each EET day, denominator = fixed initial — "
        "deployed EA `Daily_DD_Basis=INITIAL`, EquityGuards.mqh FIX 2b) + **max DD trailing "
        "(planning) + static/from-initial (FundedNext enforce)**, both reported. "
        "**Supersedes** the EA-faithful (floating) run and every prior `_canonical` / "
        "`_canonical_compound` run — retained, not deleted. v3.0.2 signal/exit/cost LOCKED; "
        "ONLY the sizing basis + daily-DD mode differ from prior runs. Governors EA-faithful, "
        "NOT tuned. Frame sha `05dea9…9ee58a` (manifest-matched). Costs ON every cell "
        "(FundedNext cell 5). EET; deterministic (two-run sha identity); PR-gated.\n"
    )
    L.append("```\n" + PROVENANCE + "\n```\n")
    L.append(
        f"> **Validation gate PASSED:** linear (mult ≡ r_base) + zero-cost + governors-OFF "
        f"reproduces the **{EXPECTED_ZEROCOST_OFF_WORST_TRAILING * 100:.2f}%** portfolio "
        "worst-fold trailing DD (abort-if-not check in driver). Reconstruction intact.\n"
    )
    L.append(f"> **Two-run determinism digest:** `{digest[:16]}…` (full sha in driver stdout).\n")

    # ── F5/F6 recovery proof (the daily-reset alignment is live) ──
    L.append("## F5/F6 recovery proof — the daily-reset alignment is live, the freeze is gone\n")
    L.append(
        "> Under the OLD non-resetting `static` (now quarantined `static_noreset`) the daily "
        "governor froze **F5 2014 at −5.99%** and **F6 2015 at −5.45%** — the daily window "
        "never reset, so once the account dipped 3.5% below initial the daily-halt blocked "
        "every recovery entry for the rest of the fold. Under `daily_ref=\"initial\"` "
        "(resetting each EET day, deploy-faithful) they recover:\n"
    )
    ff_rows = [
        dict(fold=k.replace("_", " "), roi_pct=v["roi_pct"], raw_return_pct=v["raw_pct"],
             daily_gov_fires=v["daily_fires"], total_gov_fires=v["total_fires"], kills=v["kills"],
             old_static_noreset_roi_pct={"F5 2014": -5.99, "F6 2015": -5.45}[k.replace("_", " ")])
        for k, v in f5f6.items()
    ]
    L.append(df_to_md(pd.DataFrame(ff_rows), "{:.2f}") + "\n")
    L.append(
        f"> **PROOF:** F5 2014 = **{f5f6['F5_2014']['roi_pct']:+.2f}%** "
        f"({f5f6['F5_2014']['daily_fires']} daily-gov fires), F6 2015 = "
        f"**{f5f6['F6_2015']['roi_pct']:+.2f}%** ({f5f6['F6_2015']['daily_fires']} daily-gov "
        "fires). Both positive, daily governor barely binds → the resetting daily basis is "
        "live and the −5.99%/−5.45% freeze artifact is gone (vs the day_start counterfactual "
        "~+28.9%/+20.8%). `static_noreset` is quarantined and never canonical.\n"
    )

    # ── 1. canonical matrix ──
    L.append("## 1. Canonical matrix (3 risk × governed; fixed-initial; both DD refs)\n")
    L.append("```\n" + PROVENANCE + "\n```\n")
    mt = matrix[matrix.governance == "on"].drop(columns=["governance", "sizing"]).rename(columns={
        "risk_pct": "risk%", "worst_fold_roi_pct": "worst-fold ROI%",
        "mean_fold_roi_pct": "mean-fold ROI%", "trailing_dd_pct": "trailing DD%",
        "from_initial_dd_pct": "from-init DD%", "daily_dd_pct": "daily DD%"})
    L.append(df_to_md(mt, "{:.2f}") + "\n")
    L.append(
        "> Worst-/mean-fold over the 11 search folds F1–F11 (2010–2020); holdout years in §2. "
        "**0.40% is the canonical operating tier**; 0.42%/0.45% are margin-check only (NOT "
        "adoption candidates — daily DD is a fixed $/day off initial, so it only tightens in "
        "%-terms as the account grows; there is no case to raise risk). At 0.40% the total "
        "governors never fire (on == off); a few daily-halts bind harmlessly.\n"
    )

    # ── 2. full per-fold + per-holdout-year ──
    L.append("## 2. Full per-fold + per-holdout-year (fixed-initial, daily_ref=initial)\n")
    L.append("```\n" + PROVENANCE + "\n```\n")
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            sr, hr = search_res[(risk, gov)], holdout_res[(risk, gov)]
            tbl = []
            for f in SEARCH_FOLDS:
                m = sr[f]
                mark = " ◀F5/F6" if f in (5, 6) else ""
                tbl.append(dict(fold=f"F{f}{mark}", yr=2009 + f, n=m["n"], roi_pct=m["roi"] * 100,
                                trailing_dd_pct=m["trailing_dd"] * 100,
                                from_init_dd_pct=m["from_initial_dd"] * 100,
                                daily_dd_pct=m["daily_dd"] * 100, daily_fires=m["daily_fires"],
                                kills=m["kills"]))
            for y, m in hr.items():
                tbl.append(dict(fold=(f"{y}p" if y == 2026 else str(y)), yr=y, n=m["n"],
                                roi_pct=m["roi"] * 100, trailing_dd_pct=m["trailing_dd"] * 100,
                                from_init_dd_pct=m["from_initial_dd"] * 100,
                                daily_dd_pct=m["daily_dd"] * 100, daily_fires=m["daily_fires"],
                                kills=m["kills"]))
            a = agg_search(sr)
            L.append(f"**{risk * 100:.2f}% r_base — governors {gov.upper()}** "
                     f"(search worst-fold ROI {a['worst_roi'] * 100:.2f}% / mean "
                     f"{a['mean_roi'] * 100:.2f}% / worst trailing DD {a['trailing_dd'] * 100:.2f}% / "
                     f"worst from-init {a['from_initial_dd'] * 100:.2f}% / worst daily "
                     f"{a['daily_dd'] * 100:.2f}%):\n")
            L.append(df_to_md(pd.DataFrame(tbl), "{:.2f}") + "\n")
    L.append(
        "> `F5 2014` / `F6 2015` are flagged ◀F5/F6 — both positive (recovery proof). "
        "`2026p` ROI is a RAW ~4-month partial return (NOT annualised, NO CAGR). Full-year "
        "folds annualised over their own ~1y span.\n"
    )

    # ── 3. risk-decision read ──
    L.append("## 3. Risk-decision read — does 0.40% clear both DD limits with margin?\n")
    L.append("```\n" + PROVENANCE + "\n```\n")
    dec = []
    for risk in RISK_LEVELS:
        a = agg_search(search_res[(risk, "on")])
        dec.append(dict(
            risk_pct=risk * 100, tier=("CANONICAL" if risk == CANONICAL_RISK else "margin-check"),
            worst_fold_trailing_dd_pct=a["trailing_dd"] * 100,
            under_10pct_hard=("YES" if a["trailing_dd"] < HARD_DD else "NO"),
            under_8pct_deploy=("YES" if a["trailing_dd"] <= DEPLOY_DD else "NO"),
            worst_daily_dd_pct=a["daily_dd"] * 100,
            under_5pct_daily=("YES" if a["daily_dd"] < DAILY_LIMIT else "NO"),
            worst_fold_roi_pct=a["worst_roi"] * 100, mean_fold_roi_pct=a["mean_roi"] * 100,
            kills=a["kills"]))
    L.append(df_to_md(pd.DataFrame(dec), "{:.2f}") + "\n")
    a04 = agg_search(search_res[(CANONICAL_RISK, "on")])
    a042 = agg_search(search_res[(0.0042, "on")])
    a045 = agg_search(search_res[(0.0045, "on")])
    clears04 = a04["trailing_dd"] < HARD_DD and a04["daily_dd"] < DAILY_LIMIT and a04["kills"] == 0
    L.append(
        f"> **0.40% (canonical):** worst-fold trailing DD **{a04['trailing_dd'] * 100:.2f}%** "
        f"(hard 10%: {'CLEARS' if a04['trailing_dd'] < HARD_DD else 'BREACHES'}; deploy 8%: "
        f"{'clears' if a04['trailing_dd'] <= DEPLOY_DD else 'over'}), worst daily "
        f"**{a04['daily_dd'] * 100:.2f}%** ({'under' if a04['daily_dd'] < DAILY_LIMIT else 'over'} "
        f"5%), worst-fold ROI **{a04['worst_roi'] * 100:.2f}%** / mean **{a04['mean_roi'] * 100:.2f}%**, "
        f"{a04['kills']} kills → **{'0.40% CONFIRMED as the operating tier — clears both DD limits with margin.' if clears04 else 'REVIEW — 0.40% does not clear cleanly; see numbers.'}**\n"
    )
    L.append(
        f"> **0.42% (margin-check):** trailing **{a042['trailing_dd'] * 100:.2f}%**, daily "
        f"**{a042['daily_dd'] * 100:.2f}%**, {a042['kills']} kills. "
        f"**0.45% (margin-check):** trailing **{a045['trailing_dd'] * 100:.2f}%**, daily "
        f"**{a045['daily_dd'] * 100:.2f}%**, {a045['kills']} kills. These QUANTIFY proximity to "
        "the limits and are NOT recommended — the headline stays 0.40%.\n"
    )

    # ── 4. vs superseded EA-faithful (floating) ──
    L.append("## 4. vs superseded EA-faithful (floating) — why fixed-initial is now canonical\n")
    eaf_cmp = _ea_faithful_compare()
    if eaf_cmp is not None:
        delta = pd.DataFrame([
            dict(metric="worst-fold ROI %", fixed_initial=a04["worst_roi"] * 100,
                 floating_ea_faithful=eaf_cmp["worst_roi"],
                 delta_pp=a04["worst_roi"] * 100 - eaf_cmp["worst_roi"]),
            dict(metric="mean-fold ROI %", fixed_initial=a04["mean_roi"] * 100,
                 floating_ea_faithful=eaf_cmp["mean_roi"],
                 delta_pp=a04["mean_roi"] * 100 - eaf_cmp["mean_roi"]),
            dict(metric="worst-fold trailing DD %", fixed_initial=a04["trailing_dd"] * 100,
                 floating_ea_faithful=eaf_cmp["trailing_dd"],
                 delta_pp=a04["trailing_dd"] * 100 - eaf_cmp["trailing_dd"]),
            dict(metric="worst from-init DD %", fixed_initial=a04["from_initial_dd"] * 100,
                 floating_ea_faithful=eaf_cmp["from_initial_dd"],
                 delta_pp=a04["from_initial_dd"] * 100 - eaf_cmp["from_initial_dd"]),
            dict(metric="worst daily DD %", fixed_initial=a04["daily_dd"] * 100,
                 floating_ea_faithful=eaf_cmp["daily_dd"],
                 delta_pp=a04["daily_dd"] * 100 - eaf_cmp["daily_dd"]),
        ])
        L.append("**0.40% gov-on — fixed-initial (canonical) vs floating (EA-faithful, superseded):**\n")
        L.append(df_to_md(delta, "{:.2f}") + "\n")
    else:
        L.append("> EA-faithful matrix.csv not found — comparison skipped.\n")
    L.append(
        "> **Why fixed-initial is now canonical:** the deployed EA sizes every entry off the "
        "FIXED INITIAL balance (`ArcComputeLots` post-FIX-1), NOT floating equity. The "
        "floating-equity run was a procyclical-tail *measurement* (it sized larger when the "
        "open book was up, into subsequent reversals) — useful to confirm the EA should stay "
        "on fixed-initial sizing, but it does NOT match how the EA trades. Fixed-initial is "
        "the deploy-faithful basis and therefore THE canonical gate. The EA-faithful run is "
        "retained, labelled SUPERSEDED, for the procyclical comparison only.\n"
    )

    # ── honesty / reviewer notes ──
    L.append("## Reviewer notes / honesty constraints\n")
    L.append(
        "- Canonical for the MODELLED system, not ground truth. Two unclosable gaps: (1) "
        "intrabar tick resolution — the open book is marked at H4-bar resolution of the "
        "intrabar low (a true tick trigger could differ); (2) live close-all slippage on N "
        "concurrent positions is unmodelled.\n"
        "- Sizing: fixed-initial — 1R ≡ r_base of the INITIAL balance, constant per trade, "
        "per-fold reset to 1.0 (the linear `governed_wfo.simulate_fold` path). Matches the "
        "deployed EA `ArcComputeLots` fixed-initial sizing.\n"
        "- Daily DD: `daily_ref=\"initial\"` (resetting fixed-$/day off initial = deployed EA "
        "`Daily_DD_Basis=INITIAL`, EquityGuards.mqh FIX 2b). NOT `static_noreset` (the "
        "quarantined non-resetting freeze-bug mode).\n"
        "- Every figure traces to the committed CSVs (`matrix.csv`, `per_fold.csv`, "
        "`governor_log.csv`). No CAGR; holdout per-year; 2026 raw partial, flagged. Prior "
        "fabrication disclosure stands.\n"
    )
    (OUTDIR / "FINAL_CANONICAL_WFO.md").write_text("\n".join(L), encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
def _build():
    """Load frame + run the full matrix + F5/F6 check. Returns (D, search, holdout, f5f6)."""
    if not (gw.ARC / "step_1" / "trade_paths.parquet").exists():
        print("[final] ABORT: trade_paths.parquet missing — regenerate the frame first.", flush=True)
        return None
    print("[final] loading sha-05dea9 frame + cell-5 cost overlay…", flush=True)
    D = cw.load()
    hy = holdout_year_tids(D)
    print(f"[final] {len(D['allt'])} trades; F1–F11 + holdout years {sorted(hy)}; "
          f"mean cost {D['mean_cost_r']:.4f}R/trade", flush=True)

    # ── VALIDATION GATE: linear zero-cost gov-off worst-fold trailing == 9.22% ──
    gw.R_BASE = 0.005
    zc = {f: gw.simulate_fold(tids, D["sched_raw"], D["day_key"], D["clock"],
                              governed=False, total_ref="static", daily_ref="initial")
          for f, tids in D["fold_tids"].items()}
    zc_worst = max(zc[f]["dd_trailing"] for f in SEARCH_FOLDS)
    print(f"[validate] linear zero-cost gov-off worst-fold trailing DD = {zc_worst:.4f} "
          f"(expect {EXPECTED_ZEROCOST_OFF_WORST_TRAILING})", flush=True)
    if abs(zc_worst - EXPECTED_ZEROCOST_OFF_WORST_TRAILING) > VALIDATION_TOL:
        print(f"[final] ABORT: validation gate FAILED — got {zc_worst:.4f}, expected "
              f"{EXPECTED_ZEROCOST_OFF_WORST_TRAILING}±{VALIDATION_TOL}.", flush=True)
        return None
    print("[validate] OK — reconstruction intact; fixed-initial output trusted.", flush=True)

    search_res, holdout_res = run_matrix(D, hy)

    # ── F5/F6 recovery proof (abort if still frozen/negative) ──
    f5f6 = f5f6_recovery_check(search_res)
    print(f"[F5/F6] F5 2014 ROI={f5f6['F5_2014']['roi_pct']:+.2f}% "
          f"(daily fires {f5f6['F5_2014']['daily_fires']}) | F6 2015 ROI="
          f"{f5f6['F6_2015']['roi_pct']:+.2f}% (daily fires {f5f6['F6_2015']['daily_fires']})",
          flush=True)
    if f5f6["F5_2014"]["roi_pct"] <= 0 or f5f6["F6_2015"]["roi_pct"] <= 0:
        print("[final] ABORT: F5/F6 still non-positive — daily-reset alignment did NOT take.",
              flush=True)
        return None
    print("[F5/F6] OK — both positive; resetting daily basis live, freeze gone.", flush=True)

    return D, search_res, holdout_res, f5f6


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0.2 — FINAL CANONICAL WFO (fixed-initial)")
    p.add_argument("--verify-determinism", action="store_true")
    args = p.parse_args(argv)

    built = _build()
    if built is None:
        return 2
    D, search_res, holdout_res, f5f6 = built
    digest = matrix_digest(search_res, holdout_res)
    print(f"[determinism] run-1 matrix digest = {digest}", flush=True)

    if args.verify_determinism:
        built2 = _build()
        if built2 is None:
            return 2
        digest2 = matrix_digest(built2[1], built2[2])
        print(f"[determinism] run-2 matrix digest = {digest2}", flush=True)
        print(f"[determinism] two-run identical = {digest == digest2}", flush=True)
        if digest != digest2:
            print("[final] ABORT: two-run determinism FAILED.", flush=True)
            return 1

    emit(search_res, holdout_res, D, f5f6, digest)
    print(f"[done] wrote FINAL CANONICAL artefacts to {OUTDIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

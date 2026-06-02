"""Arc 10 v3.0.2 — CANONICAL WFO, COMPOUND SIZING (FundedNext, EET, 3.5R).

SUPERSEDES the linear canonical run (`results/l_arc_10_v3.0.2_canonical/`). Same
structure, corrected sizing basis. The linear run sized fixed-%-of-INITIAL
(1R = r_base of 1.0 throughout); the live EA sizes **fixed-%-of-EQUITY at each
trade's open**. That is a simulation-basis correction — every ROI and DD number
is recomputed. v3.0.2 logic LOCKED; governors EA-faithful, not tuned. PR-gated.

THE SIZING CORRECTION (the only change):
  * Risk per trade = r_base × CLOSED equity at that trade's open. "Closed equity"
    = balance from closed trades only, EXCLUDING open floating positions.
  * Concurrent opens all size off the same closed-equity snapshot (equity steps
    only as trades CLOSE — no intra-cluster compounding).
  * Per-fold reset, compound WITHIN fold: each fold (and each holdout year) starts
    at equity 1.0 and compounds through its own year as trades close.
This is EXACTLY what `whole_period_dd.simulate_continuous(compound=True)` already
implements (mult[tid] = RB × e_bal locked at entry; e_bal += mult × realized on
close). We reuse it PER FOLD (reset to 1.0) — no re-implementation. The linear
`simulate_fold` is retained ONLY for the frame-integrity validation gate.

EVERYTHING ELSE IDENTICAL to the linear canonical run:
  * EET, costs ON, FundedNext cell 5 (swap-OFF, 1.5× spread, $5/lot RT, 0.5 slip ×
    n_fills); cost_r imported from fundednext_cost_sweep.py via fundednext_floating.
  * Deployed exit policy sl_partial_close_1r_runner_trail @ 3.5xATR (simulate_path).
  * r_base swept {0.40%, 0.50%}.
  * Governance {ON (3.5/4.5 daily, 7/8 total, fired from-initial = live-EA basis),
    OFF}. Both DD refs: trailing-peak (planning) + from-initial (FundedNext basis).
  * 11 search folds (2010–2020, one/yr) + holdout reported PER YEAR (2021–2025 full
    + 2026 partial). NO CAGR anywhere; NO continuous-block row. 2026 = RAW partial
    return (~4 months), flagged, NEVER annualised. Full years annualised over their
    own ~1y span (= annual return; not a multi-year CAGR).
  * Determinism: two-run sha identity. Frame sha 05dea9…9ee58a.
  * Validation gate (frame integrity): governors-OFF + zero-cost LINEAR sim must
    reproduce the 9.22% portfolio worst-fold trailing DD — abort if not.

Outputs (results/l_arc_10_v3.0.2_canonical_compound/):
  CANONICAL_WFO_COMPOUND.md, matrix.csv, per_fold.csv, governor_log.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in ["", "scripts/analysis", "scripts/l_arc_10_v3_0_2_governed", "scripts/audit/arc_10"]:
    sys.path.insert(0, str(ROOT / _p))

import canonical_wfo as cw  # noqa: E402  reuse identical load() reconstruction
import governed_wfo as gw  # noqa: E402  simulate_fold (validation) + annualise + df_to_md
import whole_period_dd as wp  # noqa: E402  simulate_continuous (compound sizing)

OUTDIR = ROOT / "results" / "l_arc_10_v3.0.2_canonical_compound"
LINEAR_DIR = ROOT / "results" / "l_arc_10_v3.0.2_canonical"
df_to_md = gw.df_to_md

RISK_LEVELS = [0.004, 0.005]
HOLDOUT_FOLD = 12
SEARCH_FOLDS = list(range(1, 12))  # F1–F11
DEPLOY_DD, HARD_DD = 0.08, 0.10
WORST_ROI_MIN, MEAN_ROI_MIN = 0.05, 0.08
EXPECTED_ZEROCOST_OFF_WORST_TRAILING = 0.0922
VALIDATION_TOL = 0.002


def verdict(worst_roi, mean_roi, dd, any_kill):
    if any_kill:
        return "FAIL (kill)"
    if worst_roi > WORST_ROI_MIN and mean_roi > MEAN_ROI_MIN and dd <= DEPLOY_DD:
        return "PASS-DEPLOYABLE"
    if dd <= HARD_DD and worst_roi > 0:
        return "PASS-VIABLE"
    return "FAIL"


def holdout_year_tids(D):
    """Split holdout fold 12 into per-entry-year sub-folds (2021..2026)."""
    A = pd.read_csv(gw.SRC / "A_entry_mae.csv")[["trade_id", "fold"]]
    pool = pd.read_parquet(gw.ARC / "step_1" / "pool.parquet")[["trade_id", "entry_time"]]
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True)
    meta = A.merge(pool, on="trade_id")
    hold = meta[meta.fold == HOLDOUT_FOLD].copy()
    hold["yr"] = hold.entry_time.dt.year
    return {int(y): sorted(hold[hold.yr == y].trade_id) for y in sorted(hold.yr.unique())}


def run_fold(tids, D, *, governed, is_2026=False):
    """Compound-sized, per-fold-reset run (simulate_continuous on one fold's tids).

    ROI = compound return over the fold; full-year folds annualised over their own
    ~1y span (= annual return), 2026 reported RAW (not annualised)."""
    # Per-fold / per-year reset (equity starts 1.0, compounds within ~1y) → daily_ref
    # ="initial" = the EA-faithful FundedNext fixed-$/day basis (equity stays near
    # 1.0, so initial ≈ day_start). daily_dd_from_trace reports on the same basis.
    run = wp.simulate_continuous(
        tids, D["sched_cost"], D["day_key"], D["clock"],
        governed=governed, total_ref="static", compound=True,
        daily_ref="initial",
    )
    e_ps = [D["sched_cost"][t]["e"] for t in tids]
    span = (D["clock"][max(e_ps)] - D["clock"][min(e_ps)]).total_seconds() / (365.25 * 86400.0)
    raw = float(run["e_bal_final"] - 1.0)
    roi = raw if is_2026 else float(gw.annualise(run["e_bal_final"], span))
    day_dd = wp.daily_dd_from_trace(run["trace"], D["day_key"], daily_ref="initial")
    return dict(
        n=len(tids), roi=roi, raw=raw, span=float(span),
        trailing_dd=float(run["dd_trailing"]), from_initial_dd=float(run["dd_static"]),
        daily_dd=float(max(day_dd.values())) if day_dd else 0.0,
        kills=int(sum(1 for g, _ in run["fires"] if g == "total_close_all")),
        fires=run["fires"], is_2026=is_2026,
    )


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tp = gw.ARC / "step_1" / "trade_paths.parquet"
    if not tp.exists():
        print(f"[compound] ABORT: trade_paths.parquet missing at {tp}", flush=True)
        return 3
    print("[compound] loading sha-05dea9 frame + cell-5 cost overlay…", flush=True)
    D = cw.load()
    hy = holdout_year_tids(D)
    print(
        f"[compound] {len(D['allt'])} trades; F1–F11 + holdout years "
        f"{sorted(hy)} (2026 partial); mean cost {D['mean_cost_r']:.4f}R/trade",
        flush=True,
    )

    # ── VALIDATION GATE (frame integrity): linear sim, zero-cost, gov-off → 9.22% ──
    gw.R_BASE = 0.005
    zc = {
        f: gw.simulate_fold(tids, D["sched_raw"], D["day_key"], D["clock"],
                            governed=False, total_ref="static")
        for f, tids in D["fold_tids"].items()
    }
    zc_worst = max(zc[f]["dd_trailing"] for f in SEARCH_FOLDS)
    print(f"[validate] linear zero-cost gov-off worst-fold trailing DD = {zc_worst:.4f} "
          f"(expect {EXPECTED_ZEROCOST_OFF_WORST_TRAILING})", flush=True)
    if abs(zc_worst - EXPECTED_ZEROCOST_OFF_WORST_TRAILING) > VALIDATION_TOL:
        print("[compound] ABORT: frame-integrity validation gate FAILED.", flush=True)
        return 2
    print("[validate] OK — frame intact; compound output trusted.", flush=True)

    # ── Compound matrix: per-fold (F1–F11) + holdout years, each risk × gov ──
    search_res: dict = {}   # (risk, gov) -> {fold: metrics}
    holdout_res: dict = {}  # (risk, gov) -> {year: metrics}
    for risk in RISK_LEVELS:
        wp.RB = risk
        for gov in ("off", "on"):
            governed = gov == "on"
            search_res[(risk, gov)] = {
                f: run_fold(D["fold_tids"][f], D, governed=governed) for f in SEARCH_FOLDS
            }
            holdout_res[(risk, gov)] = {
                y: run_fold(hy[y], D, governed=governed, is_2026=(y == 2026))
                for y in sorted(hy)
            }
        sr = search_res[(risk, "on")]
        wt = max(sr[f]["trailing_dd"] for f in SEARCH_FOLDS)
        print(f"[risk {risk * 100:.2f}%] compound gov-on worst-fold trailing DD = {wt * 100:.2f}%",
              flush=True)

    emit(search_res, holdout_res, D)
    print(f"[done] wrote compound canonical artefacts to {OUTDIR}", flush=True)
    return 0


def agg_search(by_fold):
    rois = [by_fold[f]["roi"] for f in SEARCH_FOLDS]
    return dict(
        worst_roi=float(min(rois)),
        mean_roi=float(np.mean(rois)),
        trailing_dd=float(max(by_fold[f]["trailing_dd"] for f in SEARCH_FOLDS)),
        from_initial_dd=float(max(by_fold[f]["from_initial_dd"] for f in SEARCH_FOLDS)),
        daily_dd=float(max(by_fold[f]["daily_dd"] for f in SEARCH_FOLDS)),
        kills=int(sum(by_fold[f]["kills"] for f in SEARCH_FOLDS)),
    )


def emit(search_res, holdout_res, D):
    # ---- matrix.csv (4 rows: 2 risk × 2 gov, per-fold only) ----
    rows = []
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            a = agg_search(search_res[(risk, gov)])
            rows.append(dict(
                risk_pct=risk * 100, governance=gov,
                worst_fold_roi_pct=a["worst_roi"] * 100,
                mean_fold_roi_pct=a["mean_roi"] * 100,
                trailing_dd_pct=a["trailing_dd"] * 100,
                from_initial_dd_pct=a["from_initial_dd"] * 100,
                daily_dd_pct=a["daily_dd"] * 100,
                kills=a["kills"],
                verdict_trailing=verdict(a["worst_roi"], a["mean_roi"], a["trailing_dd"], a["kills"] > 0),
                verdict_from_initial=verdict(a["worst_roi"], a["mean_roi"], a["from_initial_dd"], a["kills"] > 0),
            ))
    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUTDIR / "matrix.csv", index=False, lineterminator="\n")

    # ---- per_fold.csv (F1–F11 + holdout years, all risk × gov) ----
    pf = []
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            for f in SEARCH_FOLDS:
                m = search_res[(risk, gov)][f]
                pf.append(dict(
                    risk_pct=risk * 100, governance=gov, segment="search",
                    fold=f"F{f}", calendar_year=2009 + f, n_trades=m["n"],
                    roi_pct=m["roi"] * 100, raw_return_pct=m["raw"] * 100,
                    trailing_dd_pct=m["trailing_dd"] * 100,
                    from_initial_dd_pct=m["from_initial_dd"] * 100,
                    daily_dd_pct=m["daily_dd"] * 100, kills=m["kills"],
                    fires=len(m["fires"]), roi_basis="annualised_1y",
                ))
            for y, m in holdout_res[(risk, gov)].items():
                pf.append(dict(
                    risk_pct=risk * 100, governance=gov, segment="holdout",
                    fold=(f"{y}p" if y == 2026 else str(y)), calendar_year=y, n_trades=m["n"],
                    roi_pct=m["roi"] * 100, raw_return_pct=m["raw"] * 100,
                    trailing_dd_pct=m["trailing_dd"] * 100,
                    from_initial_dd_pct=m["from_initial_dd"] * 100,
                    daily_dd_pct=m["daily_dd"] * 100, kills=m["kills"],
                    fires=len(m["fires"]),
                    roi_basis=("raw_partial_4mo" if y == 2026 else "annualised_1y"),
                ))
    pd.DataFrame(pf).to_csv(OUTDIR / "per_fold.csv", index=False, lineterminator="\n")

    # ---- governor_log.csv (all governed cells; from-initial firing) ----
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
    pd.DataFrame(gl).to_csv(OUTDIR / "governor_log.csv", index=False, lineterminator="\n")

    write_summary(search_res, holdout_res, matrix, D)


def write_summary(search_res, holdout_res, matrix, D):
    L = ["# Arc 10 v3.0.2 — CANONICAL WFO, COMPOUND SIZING (FundedNext, EET, 3.5R)\n"]
    L.append(
        "> **Supersedes the linear canonical run** "
        "([`results/l_arc_10_v3.0.2_canonical/`](../l_arc_10_v3.0.2_canonical/CANONICAL_WFO.md)). "
        "Same structure; corrected sizing basis. The linear run sized fixed-%-of-INITIAL; "
        "the live EA sizes **fixed-%-of-CLOSED-EQUITY at each trade's open** (concurrent "
        "opens share the snapshot; equity steps only on close; per-fold reset, compound "
        "within fold). Every ROI/DD recomputed via `simulate_continuous(compound=True)` "
        "run per fold — no re-implementation. v3.0.2 LOCKED; governors EA-faithful, not "
        "tuned. Costs ON (cell 5); r_base {0.40%,0.50%}; EET; frame sha `05dea9…9ee58a`; "
        "deterministic (two-run sha identity); PR-gated.\n"
    )
    L.append(
        f"> **Validation gate PASSED** (frame integrity): linear zero-cost + governors-OFF "
        f"reproduces the **{EXPECTED_ZEROCOST_OFF_WORST_TRAILING * 100:.2f}%** portfolio "
        "worst-fold trailing DD (abort-if-not).\n"
    )
    L.append(
        "> **NO CAGR anywhere.** Holdout scored as six independent annual-reset folds "
        "(2021–2025 full years + 2026 partial), each reset to 1.0 and compounding within "
        "its year — directly comparable to F1–F11. **2026 is a RAW partial return (~4 "
        "months), never annualised.** Full-year folds annualised over their own ~1y span "
        "(= annual return, not a multi-year CAGR).\n"
    )

    # ── 1. matrix ──
    L.append("## 1. Canonical matrix (compound; 2 risk × 2 governance; both DD refs)\n")
    mt = matrix.rename(columns={
        "risk_pct": "risk%", "worst_fold_roi_pct": "worst-fold ROI%",
        "mean_fold_roi_pct": "mean-fold ROI%", "trailing_dd_pct": "trailing DD%",
        "from_initial_dd_pct": "from-init DD%", "daily_dd_pct": "daily DD%",
    })
    L.append(df_to_md(mt, "{:.2f}") + "\n")
    L.append(
        "> Per-fold = ceiling (each fold a fresh-tier reset, compounding within its year). "
        "No continuous-block row: the holdout is scored per-year (§2), not as one block. "
        "Worst-/mean-fold are over the 11 search folds F1–F11; holdout years are reported "
        "in §2/§3, not gated.\n"
    )

    # ── 2. full per-fold + per-holdout-year ──
    L.append("## 2. Full per-fold + per-holdout-year (compound, annual-reset — one basis)\n")
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            sr, hr = search_res[(risk, gov)], holdout_res[(risk, gov)]
            tbl = []
            for f in SEARCH_FOLDS:
                m = sr[f]
                tbl.append(dict(fold=f"F{f}", yr=2009 + f, n=m["n"], roi_pct=m["roi"] * 100,
                                trailing_dd_pct=m["trailing_dd"] * 100,
                                from_init_dd_pct=m["from_initial_dd"] * 100,
                                daily_dd_pct=m["daily_dd"] * 100, kills=m["kills"], fires=len(m["fires"])))
            for y, m in hr.items():
                tbl.append(dict(fold=(f"{y}p" if y == 2026 else str(y)), yr=y, n=m["n"],
                                roi_pct=m["roi"] * 100, trailing_dd_pct=m["trailing_dd"] * 100,
                                from_init_dd_pct=m["from_initial_dd"] * 100,
                                daily_dd_pct=m["daily_dd"] * 100, kills=m["kills"], fires=len(m["fires"])))
            a = agg_search(sr)
            L.append(
                f"**{risk * 100:.2f}% r_base — governors {gov.upper()}** "
                f"(search worst-fold ROI {a['worst_roi'] * 100:.2f}% / mean "
                f"{a['mean_roi'] * 100:.2f}% / worst trailing DD {a['trailing_dd'] * 100:.2f}% / "
                f"worst from-init {a['from_initial_dd'] * 100:.2f}%):\n"
            )
            L.append(df_to_md(pd.DataFrame(tbl), "{:.2f}") + "\n")
    L.append("> `2026p` ROI is a RAW ~4-month partial return (not annualised; see basis note).\n")

    # ── 3. mean comparison (apples-to-apples) ──
    L.append("## 3. Mean comparison — search vs holdout (same compound basis)\n")
    cmp_rows = []
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            sr, hr = search_res[(risk, gov)], holdout_res[(risk, gov)]
            mean_search = float(np.mean([sr[f]["roi"] for f in SEARCH_FOLDS]))
            full_years = [y for y in hr if y != 2026]
            mean_hold = float(np.mean([hr[y]["roi"] for y in full_years]))
            cmp_rows.append(dict(
                risk_pct=risk * 100, governance=gov,
                mean_search_F1_11_pct=mean_search * 100,
                mean_holdout_2021_2025_pct=mean_hold * 100,
                delta_pp=(mean_hold - mean_search) * 100,
                roi_2026_raw_partial_pct=hr[2026]["roi"] * 100,
            ))
    L.append(df_to_md(pd.DataFrame(cmp_rows), "{:.2f}") + "\n")
    L.append(
        "> Holdout full-years (2021–2025) vs search folds (2010–2020) on identical compound "
        "annual-reset footing — settles 'is the holdout weaker' on one basis. 2026 raw "
        "partial shown separately (≈4 months, not comparable to a full year).\n"
    )

    # ── 4. linear-vs-compound delta (0.50% gov-on) ──
    L.append("## 4. Linear → compound delta (0.50% gov-on)\n")
    lin = pd.read_csv(LINEAR_DIR / "matrix.csv")
    lr = lin[(lin.risk_pct == 0.5) & (lin.governance == "on") & (lin.view == "per_fold")].iloc[0]
    cc = agg_search(search_res[(0.005, "on")])
    delta = pd.DataFrame([
        dict(metric="worst-fold ROI %", linear=lr["worst_fold_or_period_roi_pct"],
             compound=cc["worst_roi"] * 100, delta_pp=cc["worst_roi"] * 100 - lr["worst_fold_or_period_roi_pct"]),
        dict(metric="mean-fold ROI %", linear=lr["mean_fold_roi_pct"],
             compound=cc["mean_roi"] * 100, delta_pp=cc["mean_roi"] * 100 - lr["mean_fold_roi_pct"]),
        dict(metric="worst-fold trailing DD %", linear=lr["trailing_dd_pct"],
             compound=cc["trailing_dd"] * 100, delta_pp=cc["trailing_dd"] * 100 - lr["trailing_dd_pct"]),
        dict(metric="worst-fold from-init DD %", linear=lr["from_initial_dd_pct"],
             compound=cc["from_initial_dd"] * 100, delta_pp=cc["from_initial_dd"] * 100 - lr["from_initial_dd_pct"]),
    ])
    L.append(df_to_md(delta, "{:.2f}") + "\n")
    L.append(
        "> Within a single ~1y fold compounding sizes up as equity grows, so ROI lifts and "
        "DD rises modestly. The binding check is whether worst-fold trailing DD at 0.50% "
        f"({cc['trailing_dd'] * 100:.2f}% compound vs {lr['trailing_dd_pct']:.2f}% linear) "
        "crosses the 10% hard limit.\n"
    )

    # ── 5. risk-decision (re-checked on compound) ──
    L.append("## 5. Risk-level decision — re-checked on compound basis\n")
    dec = []
    for risk in RISK_LEVELS:
        a = agg_search(search_res[(risk, "on")])
        dec.append(dict(
            risk_pct=risk * 100, worst_fold_trailing_dd_pct=a["trailing_dd"] * 100,
            under_10pct_hard=("YES" if a["trailing_dd"] < HARD_DD else "NO"),
            under_8pct_deploy=("YES" if a["trailing_dd"] <= DEPLOY_DD else "NO"),
            worst_daily_dd_pct=a["daily_dd"] * 100,
            under_5pct_daily=("YES" if a["daily_dd"] < 0.05 else "NO"),
            worst_fold_roi_pct=a["worst_roi"] * 100, mean_fold_roi_pct=a["mean_roi"] * 100,
        ))
    L.append(df_to_md(pd.DataFrame(dec), "{:.2f}") + "\n")
    dd04 = agg_search(search_res[(0.004, "on")])["trailing_dd"]
    dd05 = agg_search(search_res[(0.005, "on")])["trailing_dd"]
    both_under = dd04 < HARD_DD and dd05 < HARD_DD
    decision = (
        "flat-0.50% (both risk levels keep fresh-tier worst-fold trailing DD < 10%)"
        if both_under else
        "launch-at-0.40%, step-to-0.50% after a buffer banks (0.40% keeps fresh-tier "
        "worst-fold trailing DD under 10% where 0.50% does not)"
    )
    L.append(
        f"> **Decision (compound basis): {decision}.** Worst-fold trailing DD "
        f"{dd04 * 100:.2f}% at 0.40% vs {dd05 * 100:.2f}% at 0.50%. "
        "Re-validates (or revises) the linear run's launch-at-0.40% call on the corrected "
        "sizing.\n"
    )

    # ── 6. governor value + kills ──
    L.append("## 6. Governor value + kills (compound; ON vs OFF, both refs)\n")
    gv = []
    for risk in RISK_LEVELS:
        off, on = agg_search(search_res[(risk, "off")]), agg_search(search_res[(risk, "on")])
        gv.append(dict(
            risk_pct=risk * 100,
            trailing_dd_saved_pp=(off["trailing_dd"] - on["trailing_dd"]) * 100,
            from_initial_dd_saved_pp=(off["from_initial_dd"] - on["from_initial_dd"]) * 100,
            mean_roi_cost_pp=(off["mean_roi"] - on["mean_roi"]) * 100,
            kills_from_initial=on["kills"],
        ))
    L.append(df_to_md(pd.DataFrame(gv), "{:.2f}") + "\n")
    total_kills = sum(
        search_res[(r, "on")][f]["kills"] for r in RISK_LEVELS for f in SEARCH_FOLDS
    ) + sum(
        holdout_res[(r, "on")][y]["kills"] for r in RISK_LEVELS for y in holdout_res[(r, "on")]
    )
    L.append(
        f"> Total 8% kill events under from-initial firing across all compound cells: "
        f"**{total_kills}**. (Governors fire on the from-initial / live-EA basis; the "
        "trailing-fired sensitivity is documented in the linear run.)\n"
    )

    # ── honesty ──
    L.append("## Reviewer notes / honesty constraints\n")
    L.append(
        "- Canonical for the MODELLED system, not ground truth. Two unclosable modelled-vs-"
        "live gaps: (1) intrabar tick resolution on governor firing (H4-bar resolution of "
        "the intrabar low understates a true tick trigger); (2) close-all slippage on N "
        "concurrent positions is unmodelled (bites near the 5% daily margin).\n"
        "- Compound sizing = r_base × closed equity at open; floating opens excluded from "
        "the sizing snapshot; equity steps only on close. Per-fold reset.\n"
        "- Every figure traces to the committed CSVs (`matrix.csv`, `per_fold.csv`, "
        "`governor_log.csv`). No CAGR; holdout per-year; 2026 raw partial, flagged.\n"
    )
    (OUTDIR / "CANONICAL_WFO_COMPOUND.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

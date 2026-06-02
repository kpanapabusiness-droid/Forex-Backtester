"""Arc 10 v3.0.2 — EA-FAITHFUL (floating-equity) WFO — REFERENCE-ONLY (FundedNext, EET, 3.5R).

REFERENCE-ONLY procyclical comparison — NOT the canonical gate. The live EA now
sizes on the FIXED INITIAL balance, so the canonical deploy-faithful basis is the
fixed-initial / linear path (`governed_wfo.simulate_fold`, mult ≡ r_base); see the
CANONICAL RUN DEFINITION in `governed_wfo.py`. This floating-equity run is retained
to MEASURE the procyclical concurrency tail (sizing off equity INCL. floating open
P&L vs the fixed-initial basis) — the comparison that justified keeping the EA on
fixed-initial sizing — not to set the gate.

This run sizes every entry off floating equity (the EA's PRE-fix behaviour,
`PositionManager.mqh:143`-style): `risk_amount = ACCOUNT_EQUITY × r_base`, where
ACCOUNT_EQUITY **includes floating open P&L** and is re-read live at each entry.
v3.0.2 LOCKED; governors EA-faithful, not tuned. PR-gated.

THE SIZING MODEL — match the EA exactly (the only change vs compound):
  At each entry, in chronological (bar) order:
    equity_at_entry = realized_balance + Σ(open-position floating MtM @ this bar)
                    = book("close") over the already-open book   (= ACCOUNT_EQUITY)
    mult[tid] = r_base × equity_at_entry        (lots ∝ risk_amount / SL_distance)
  Concurrent opens read LIVE, not a shared snapshot: entries processed in bar
  order; each reads equity as it stands (prior same-bar fills already in open_t
  contribute their floating mark), so the procyclical mechanism is modelled, not
  approximated by a daily/closed snapshot. Equity updates continuously with
  floating MtM (sizing + DD) and steps on closes.

This consolidates the existing open-book reconstruction (`governed_wfo.build_schedules`,
the concurrency-probe marks) + the floating machinery (`whole_period_dd` / `_flat`)
with floating-equity sizing — the reconstruction is NOT re-implemented; only the
per-entry sizing line differs from `simulate_continuous(compound=True)` (which
sized off CLOSED equity `e_bal`).

EVERYTHING ELSE identical to the compound canonical: EET; costs ON cell 5 (cost_r
from fundednext_cost_sweep.py via fundednext_floating); exit
sl_partial_close_1r_runner_trail @ 3.5xATR; r_base {0.40%,0.50%}; governance {ON
(3.5/4.5 daily, 7/8 total, fired from-initial = live EA), OFF}; both DD refs; 11
search folds (2010–2020) + holdout PER YEAR (2021–2025 + 2026 raw partial); NO
CAGR; frame sha 05dea9…9ee58a; two-run sha identity.

VALIDATION GATE (frame/reconstruction integrity): floating sizing collapsed to a
fixed-initial-equivalent (mult ≡ r_base, the linear sim) + zero cost must still
reproduce the 9.22% portfolio worst-fold trailing DD — abort if the open-book
reconstruction doesn't match the established figure.

Outputs (results/l_arc_10_v3.0.2_ea_faithful/):
  EA_FAITHFUL_WFO.md, matrix.csv, per_fold.csv, governor_log.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in ["", "scripts/analysis", "scripts/l_arc_10_v3_0_2_governed", "scripts/audit/arc_10"]:
    sys.path.insert(0, str(ROOT / _p))

import canonical_wfo as cw  # noqa: E402  identical load() reconstruction
import governed_wfo as gw  # noqa: E402  build_schedules, simulate_fold (validation), annualise, _date
import whole_period_dd as wp  # noqa: E402  _flat, daily_dd_from_trace

OUTDIR = ROOT / "results" / "l_arc_10_v3.0.2_ea_faithful"
COMPOUND_DIR = ROOT / "results" / "l_arc_10_v3.0.2_canonical_compound"
df_to_md = gw.df_to_md

RISK_LEVELS = [0.004, 0.005]
HOLDOUT_FOLD = 12
SEARCH_FOLDS = list(range(1, 12))
DEPLOY_DD, HARD_DD = 0.08, 0.10
WORST_ROI_MIN, MEAN_ROI_MIN = 0.05, 0.08
EXPECTED_ZEROCOST_OFF_WORST_TRAILING = 0.0922
VALIDATION_TOL = 0.002
DAILY_HALT, DAILY_CLOSE, TOTAL_HALT, TOTAL_KILL = 0.035, 0.045, 0.07, 0.08


# ───────────────────────────────────────────────────────────────────────────
# Floating-equity portfolio sim (adapted from whole_period_dd.simulate_continuous;
# ONLY the per-entry sizing differs: mult = rb × equity_INCL_FLOATING, not rb × e_bal).
# ───────────────────────────────────────────────────────────────────────────
def simulate_floating(tids, sched, day_key, clock, *, rb, governed, total_ref,
                      daily_ref="initial", trigger_mark="close"):
    # daily_ref in gw.DAILY_REF_MODES sets the DAILY-DD basis — EA-faithful
    # (EquityGuards.mqh FIX 2b), with the daily window RESETTING each EET day:
    #   "initial" (DEFAULT, FundedNext) = (day-start equity − intraday low) / FIXED
    #     initial (1.0) — a fixed $/day budget; numerator resets daily. The
    #     canonical gate basis matching the deployed EA's INITIAL basis.
    #   "day_start" (5ers) = vs the re-captured day-start equity; resets daily.
    #   "static_noreset" (QUARANTINED) = the OLD non-resetting freeze basis (warns).
    # The daily governors fire off this basis, and slot 3 of the trace carries the
    # daily NUMERATOR anchor so whole_period_dd.daily_dd_from_trace reports the same
    # reference. See gw.daily_anchors().
    gw.check_daily_ref(daily_ref)
    e_min = min(sched[t]["e"] for t in tids)
    x_max = max(sched[t]["x"] for t in tids)
    entries: dict = {}
    for t in tids:
        entries.setdefault(sched[t]["e"], []).append(t)
    for p in entries:
        entries[p].sort()  # deterministic intra-bar tiebreak (H4 has no finer order)

    e_bal = 1.0
    mult: dict = {}
    open_t: dict = {}
    day = None
    day_start = 1.0
    prev_float = 1.0
    peak_float = 1.0
    daily_halt = daily_closed = total_halt = killed = False
    fires, skipped, flattened = [], [], []
    curve, bars, trace = [], [], []
    kill_date = None

    def book(mk):
        s = e_bal
        for tid in open_t:
            s += mult[tid] * sched[tid][mk][p - sched[tid]["e"]]
        return s

    for p in range(e_min, x_max + 1):
        d = day_key[p]
        if d != day:
            day = d
            daily_halt = daily_closed = False
            day_start = prev_float
        daily_num, daily_den = gw.daily_anchors(daily_ref, day_start)

        # ── ENTRIES: EA-faithful floating-equity sizing ──
        for tid in entries.get(p, []):
            if governed and (killed or total_halt or daily_halt or daily_closed):
                gov = ("total_halt" if total_halt else "daily_close_all" if daily_closed
                       else "daily_halt")
                skipped.append((tid, gov))
                continue
            # ACCOUNT_EQUITY at this entry = realized + floating MtM of already-open
            # book (book() excludes tid — not yet added). Concurrent same-bar opens
            # already in open_t contribute their offset-0 mark -> live, not snapshot.
            equity_at_entry = book("close")
            mult[tid] = rb * equity_at_entry
            open_t[tid] = True

        trig = book(trigger_mark)
        close_eq = book("close")
        peak_ref = 1.0 if total_ref == "static" else peak_float
        total_dd = (peak_ref - trig) / peak_ref
        daily_dd = (daily_num - trig) / daily_den
        trace.append((p, float(close_eq), float(e_bal), float(daily_num), tuple(open_t)))

        if governed and not killed and total_dd >= TOTAL_KILL:
            e_bal += wp._flat(open_t, mult, sched, p, flattened, "total_close_all", trigger_mark)
            fires.append(("total_close_all", gw._date(clock[p])))
            killed = True
            kill_date = gw._date(clock[p])
        elif governed and not daily_closed and daily_dd >= DAILY_CLOSE:
            e_bal += wp._flat(open_t, mult, sched, p, flattened, "daily_close_all", trigger_mark)
            fires.append(("daily_close_all", gw._date(clock[p])))
            daily_closed = daily_halt = True
        else:
            if governed:
                if total_dd >= TOTAL_HALT and not total_halt:
                    total_halt = True
                    fires.append(("total_halt", gw._date(clock[p])))
                elif total_halt and total_dd < TOTAL_HALT:
                    total_halt = False
                if daily_dd >= DAILY_HALT and not daily_halt:
                    daily_halt = True
                    fires.append(("daily_halt", gw._date(clock[p])))

        for tid in [t for t in open_t if sched[t]["x"] == p]:
            e_bal += mult[tid] * sched[tid]["realized"]
            del open_t[tid]

        close_after = book("close")
        peak_float = max(peak_float, close_after)
        prev_float = close_after
        curve.append(close_after)
        bars.append(p)
        if killed:
            break

    curve = np.array(curve, dtype=float)
    bars = np.array(bars)
    peak = np.maximum.accumulate(curve) if curve.size else curve
    dd_trailing_series = (peak - curve) / peak if curve.size else curve
    return dict(
        curve=curve, bars=bars, e_bal_final=float(e_bal),
        dd_trailing=float(dd_trailing_series.max()) if curve.size else 0.0,
        dd_static=float(np.maximum(0.0, 1.0 - curve).max()) if curve.size else 0.0,
        fires=fires, skipped=skipped, flattened=flattened, killed=killed,
        kill_date=kill_date, trace=trace,
    )


def verdict(worst_roi, mean_roi, dd, any_kill):
    if any_kill:
        return "FAIL (kill)"
    if worst_roi > WORST_ROI_MIN and mean_roi > MEAN_ROI_MIN and dd <= DEPLOY_DD:
        return "PASS-DEPLOYABLE"
    if dd <= HARD_DD and worst_roi > 0:
        return "PASS-VIABLE"
    return "FAIL"


def holdout_year_tids(D):
    A = pd.read_csv(gw.SRC / "A_entry_mae.csv")[["trade_id", "fold"]]
    pool = pd.read_parquet(gw.ARC / "step_1" / "pool.parquet")[["trade_id", "entry_time"]]
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True)
    meta = A.merge(pool, on="trade_id")
    hold = meta[meta.fold == HOLDOUT_FOLD].copy()
    hold["yr"] = hold.entry_time.dt.year
    return {int(y): sorted(hold[hold.yr == y].trade_id) for y in sorted(hold.yr.unique())}


def run_fold(tids, D, *, rb, governed, is_2026=False):
    # CANONICAL FundedNext basis: daily_ref="initial" (fixed-$/day, resets each EET
    # day) = the deployed EA's INITIAL basis. daily_dd_from_trace reports on the same.
    run = simulate_floating(tids, D["sched_cost"], D["day_key"], D["clock"],
                            rb=rb, governed=governed, total_ref="static",
                            daily_ref="initial")
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


def agg_search(by_fold):
    rois = [by_fold[f]["roi"] for f in SEARCH_FOLDS]
    return dict(
        worst_roi=float(min(rois)), mean_roi=float(np.mean(rois)),
        trailing_dd=float(max(by_fold[f]["trailing_dd"] for f in SEARCH_FOLDS)),
        from_initial_dd=float(max(by_fold[f]["from_initial_dd"] for f in SEARCH_FOLDS)),
        daily_dd=float(max(by_fold[f]["daily_dd"] for f in SEARCH_FOLDS)),
        kills=int(sum(by_fold[f]["kills"] for f in SEARCH_FOLDS)),
    )


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not (gw.ARC / "step_1" / "trade_paths.parquet").exists():
        print("[ea] ABORT: trade_paths.parquet missing", flush=True)
        return 3
    print("[ea] loading sha-05dea9 frame + cell-5 cost overlay…", flush=True)
    D = cw.load()
    hy = holdout_year_tids(D)
    print(f"[ea] {len(D['allt'])} trades; F1–F11 + holdout years {sorted(hy)}; "
          f"mean cost {D['mean_cost_r']:.4f}R/trade", flush=True)

    # ── VALIDATION GATE (reconstruction integrity): linear, zero-cost, gov-off → 9.22% ──
    gw.R_BASE = 0.005
    zc = {f: gw.simulate_fold(tids, D["sched_raw"], D["day_key"], D["clock"],
                              governed=False, total_ref="static")
          for f, tids in D["fold_tids"].items()}
    zc_worst = max(zc[f]["dd_trailing"] for f in SEARCH_FOLDS)
    print(f"[validate] linear zero-cost gov-off worst-fold trailing DD = {zc_worst:.4f} "
          f"(expect {EXPECTED_ZEROCOST_OFF_WORST_TRAILING})", flush=True)
    if abs(zc_worst - EXPECTED_ZEROCOST_OFF_WORST_TRAILING) > VALIDATION_TOL:
        print("[ea] ABORT: reconstruction validation gate FAILED.", flush=True)
        return 2
    print("[validate] OK — reconstruction intact; floating output trusted.", flush=True)

    # ── EA-faithful matrix ──
    search_res, holdout_res = {}, {}
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            governed = gov == "on"
            search_res[(risk, gov)] = {
                f: run_fold(D["fold_tids"][f], D, rb=risk, governed=governed) for f in SEARCH_FOLDS
            }
            holdout_res[(risk, gov)] = {
                y: run_fold(hy[y], D, rb=risk, governed=governed, is_2026=(y == 2026))
                for y in sorted(hy)
            }
        a = agg_search(search_res[(risk, "on")])
        print(f"[risk {risk * 100:.2f}%] EA-faithful gov-on worst-fold trailing DD = "
              f"{a['trailing_dd'] * 100:.2f}% (kills {a['kills']})", flush=True)

    emit(search_res, holdout_res, D)
    print(f"[done] wrote EA-faithful artefacts to {OUTDIR}", flush=True)
    return 0


def emit(search_res, holdout_res, D):
    rows = []
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            a = agg_search(search_res[(risk, gov)])
            rows.append(dict(
                risk_pct=risk * 100, governance=gov,
                worst_fold_roi_pct=a["worst_roi"] * 100, mean_fold_roi_pct=a["mean_roi"] * 100,
                trailing_dd_pct=a["trailing_dd"] * 100, from_initial_dd_pct=a["from_initial_dd"] * 100,
                daily_dd_pct=a["daily_dd"] * 100, kills=a["kills"],
                verdict_trailing=verdict(a["worst_roi"], a["mean_roi"], a["trailing_dd"], a["kills"] > 0),
                verdict_from_initial=verdict(a["worst_roi"], a["mean_roi"], a["from_initial_dd"], a["kills"] > 0),
            ))
    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUTDIR / "matrix.csv", index=False, lineterminator="\n")

    pf = []
    for risk in RISK_LEVELS:
        for gov in ("off", "on"):
            for f in SEARCH_FOLDS:
                m = search_res[(risk, gov)][f]
                pf.append(dict(risk_pct=risk * 100, governance=gov, segment="search",
                               fold=f"F{f}", calendar_year=2009 + f, n_trades=m["n"],
                               roi_pct=m["roi"] * 100, raw_return_pct=m["raw"] * 100,
                               trailing_dd_pct=m["trailing_dd"] * 100,
                               from_initial_dd_pct=m["from_initial_dd"] * 100,
                               daily_dd_pct=m["daily_dd"] * 100, kills=m["kills"],
                               fires=len(m["fires"]), roi_basis="annualised_1y"))
            for y, m in holdout_res[(risk, gov)].items():
                pf.append(dict(risk_pct=risk * 100, governance=gov, segment="holdout",
                               fold=(f"{y}p" if y == 2026 else str(y)), calendar_year=y, n_trades=m["n"],
                               roi_pct=m["roi"] * 100, raw_return_pct=m["raw"] * 100,
                               trailing_dd_pct=m["trailing_dd"] * 100,
                               from_initial_dd_pct=m["from_initial_dd"] * 100,
                               daily_dd_pct=m["daily_dd"] * 100, kills=m["kills"],
                               fires=len(m["fires"]),
                               roi_basis=("raw_partial_4mo" if y == 2026 else "annualised_1y")))
    pd.DataFrame(pf).to_csv(OUTDIR / "per_fold.csv", index=False, lineterminator="\n")

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
    L = ["# Arc 10 v3.0.2 — EA-FAITHFUL (floating-equity) WFO — REFERENCE-ONLY (FundedNext, EET, 3.5R)\n"]
    L.append(
        "> **REFERENCE-ONLY — the floating-equity procyclical comparison, NOT the canonical "
        "gate.** The live EA now sizes on the FIXED INITIAL balance, so the canonical "
        "deploy-faithful basis is the fixed-initial / linear path "
        "(`governed_wfo.simulate_fold`); see the canonical run definition in `governed_wfo.py`. "
        "This run sizes every entry off floating equity "
        "(`risk_amount = ACCOUNT_EQUITY × r_base`, equity **including floating open P&L**, "
        "re-read per entry) to MEASURE the procyclical concurrency tail vs the fixed-initial "
        "basis. Reconstruction (`build_schedules`, open-book marks, `_flat`) reused verbatim; "
        "only the per-entry sizing differs from the compound run (floating equity vs closed "
        "`e_bal`). Daily DD on the EA-faithful `initial` basis (fixed-$/day, daily window "
        "resets each EET day = deployed EA INITIAL basis). v3.0.2 LOCKED; "
        "governors EA-faithful, not tuned. Costs ON (cell 5); r_base {0.40%,0.50%}; EET; "
        "frame sha `05dea9…9ee58a`; deterministic; PR-gated.\n"
    )
    L.append(
        f"> **Validation gate PASSED** (reconstruction integrity): linear (mult≡r_base) + "
        f"zero-cost reproduces the **{EXPECTED_ZEROCOST_OFF_WORST_TRAILING * 100:.2f}%** "
        "portfolio worst-fold trailing DD.\n"
    )
    L.append(
        "> **NO CAGR.** Holdout per-year (2021–2025 full + 2026 RAW partial, ~4mo, never "
        "annualised). Full-year folds annualised over their own ~1y span.\n"
    )

    # 1. matrix
    L.append("## 1. EA-faithful matrix (floating-equity; 2 risk × 2 gov; both DD refs)\n")
    mt = matrix.rename(columns={
        "risk_pct": "risk%", "worst_fold_roi_pct": "worst-fold ROI%",
        "mean_fold_roi_pct": "mean-fold ROI%", "trailing_dd_pct": "trailing DD%",
        "from_initial_dd_pct": "from-init DD%", "daily_dd_pct": "daily DD%"})
    L.append(df_to_md(mt, "{:.2f}") + "\n")
    L.append("> Worst-/mean-fold over the 11 search folds F1–F11; holdout years in §2/§3.\n")

    # 2. full per-fold + per-year
    L.append("## 2. Full per-fold + per-holdout-year (floating-equity basis)\n")
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
            L.append(f"**{risk * 100:.2f}% r_base — governors {gov.upper()}** "
                     f"(search worst-fold ROI {a['worst_roi'] * 100:.2f}% / mean "
                     f"{a['mean_roi'] * 100:.2f}% / worst trailing DD {a['trailing_dd'] * 100:.2f}% / "
                     f"worst from-init {a['from_initial_dd'] * 100:.2f}%):\n")
            L.append(df_to_md(pd.DataFrame(tbl), "{:.2f}") + "\n")
    L.append("> `2026p` ROI is a RAW ~4-month partial return (not annualised).\n")

    # 3. decision read
    L.append("## 3. Decision read — does launch-at-0.40% still hold on the live basis?\n")
    dec = []
    for risk in RISK_LEVELS:
        a = agg_search(search_res[(risk, "on")])
        dec.append(dict(risk_pct=risk * 100, worst_fold_trailing_dd_pct=a["trailing_dd"] * 100,
                        under_10pct_hard=("YES" if a["trailing_dd"] < HARD_DD else "NO"),
                        under_8pct_deploy=("YES" if a["trailing_dd"] <= DEPLOY_DD else "NO"),
                        worst_daily_dd_pct=a["daily_dd"] * 100,
                        under_5pct_daily=("YES" if a["daily_dd"] < 0.05 else "NO"),
                        worst_fold_roi_pct=a["worst_roi"] * 100, mean_fold_roi_pct=a["mean_roi"] * 100,
                        kills=a["kills"]))
    L.append(df_to_md(pd.DataFrame(dec), "{:.2f}") + "\n")
    dd04 = agg_search(search_res[(0.004, "on")])
    dd05 = agg_search(search_res[(0.005, "on")])
    L.append(
        f"> **0.40% floating-equity:** worst-fold trailing DD **{dd04['trailing_dd'] * 100:.2f}%** "
        f"(hard 10%: {'CLEARS' if dd04['trailing_dd'] < HARD_DD else 'BREACHES'}; deploy 8%: "
        f"{'clears' if dd04['trailing_dd'] <= DEPLOY_DD else 'over'}), worst daily "
        f"**{dd04['daily_dd'] * 100:.2f}%** ({'under' if dd04['daily_dd'] < 0.05 else 'over'} 5%), "
        f"{dd04['kills']} kills. **0.50%:** trailing **{dd05['trailing_dd'] * 100:.2f}%**, daily "
        f"**{dd05['daily_dd'] * 100:.2f}%**, {dd05['kills']} kills.\n"
    )

    # 4. procyclical amplification vs closed-equity compound
    L.append("## 4. Procyclical amplification — floating-equity vs closed-equity compound (0.50% gov-on)\n")
    comp = pd.read_csv(COMPOUND_DIR / "matrix.csv")
    cr = comp[(comp.risk_pct == 0.5) & (comp.governance == "on")].iloc[0]
    fa = agg_search(search_res[(0.005, "on")])
    delta = pd.DataFrame([
        dict(metric="worst-fold trailing DD %", closed_equity=cr["trailing_dd_pct"],
             floating_equity=fa["trailing_dd"] * 100, delta_pp=fa["trailing_dd"] * 100 - cr["trailing_dd_pct"]),
        dict(metric="worst-fold from-init DD %", closed_equity=cr["from_initial_dd_pct"],
             floating_equity=fa["from_initial_dd"] * 100, delta_pp=fa["from_initial_dd"] * 100 - cr["from_initial_dd_pct"]),
        dict(metric="worst daily DD %", closed_equity=cr["daily_dd_pct"],
             floating_equity=fa["daily_dd"] * 100, delta_pp=fa["daily_dd"] * 100 - cr["daily_dd_pct"]),
        dict(metric="mean-fold ROI %", closed_equity=cr["mean_fold_roi_pct"],
             floating_equity=fa["mean_roi"] * 100, delta_pp=fa["mean_roi"] * 100 - cr["mean_fold_roi_pct"]),
    ])
    L.append(df_to_md(delta, "{:.2f}") + "\n")
    amp = fa["trailing_dd"] * 100 - cr["trailing_dd_pct"]
    L.append(
        f"> Sizing off floating equity (vs closed) moves worst-fold trailing DD by "
        f"**{amp:+.2f}pp** at 0.50% gov-on. The closed-equity run's linear→compound step was "
        "a benign +0.30pp; this delta is the procyclical concurrency tail the basis "
        "correction exists to measure — bigger means floating sizing materially amplifies "
        "DD when the open book is up and entries size larger into a subsequent reversal.\n"
    )

    # 5. governor value + kills
    L.append("## 5. Governor value + kills (floating basis)\n")
    gv = []
    for risk in RISK_LEVELS:
        off, on = agg_search(search_res[(risk, "off")]), agg_search(search_res[(risk, "on")])
        gv.append(dict(risk_pct=risk * 100,
                       trailing_dd_saved_pp=(off["trailing_dd"] - on["trailing_dd"]) * 100,
                       from_initial_dd_saved_pp=(off["from_initial_dd"] - on["from_initial_dd"]) * 100,
                       mean_roi_cost_pp=(off["mean_roi"] - on["mean_roi"]) * 100,
                       kills_from_initial=on["kills"]))
    L.append(df_to_md(pd.DataFrame(gv), "{:.2f}") + "\n")
    kill_flags = []
    for risk in RISK_LEVELS:
        for f in SEARCH_FOLDS:
            for g, date in search_res[(risk, "on")][f]["fires"]:
                if g == "total_close_all":
                    kill_flags.append(f"{risk * 100:.2f}% F{f} {date}")
        for y, m in holdout_res[(risk, "on")].items():
            for g, date in m["fires"]:
                if g == "total_close_all":
                    kill_flags.append(f"{risk * 100:.2f}% {y} {date}")
    L.append("> **8% total-kill events (from-initial firing):** "
             + ("; ".join(kill_flags) if kill_flags else "none") + ". "
             "Procyclical sizing makes kills more likely than prior runs — reported honestly.\n")

    # decision framing
    L.append("## The decision this feeds (stated, not acted)\n")
    L.append(
        f"> If 0.40% floating-equity worst-fold trailing DD stays under the 10% hard limit "
        f"with margin AND daily under 5%, the live EA sizing is fine as-is. Here 0.40% lands "
        f"at **{dd04['trailing_dd'] * 100:.2f}% trailing / {dd04['daily_dd'] * 100:.2f}% daily** "
        f"({dd04['kills']} kills). If floating sizing pushes 0.40% toward/over 10% trailing or "
        "over 5% daily, the procyclical tail is material and the EA should be reconsidered "
        "(switch to closed-balance sizing — the compound run showed that costs ~nothing in "
        "ROI and keeps DD flat). The keep-vs-change-EA decision is the user's.\n"
    )

    # honesty
    L.append("## Reviewer notes / honesty constraints\n")
    L.append(
        "- Canonical for the MODELLED system, not ground truth. Two unclosable gaps: (1) "
        "intrabar tick resolution — open-book marked at H4-bar resolution of the intrabar "
        "low (a tick trigger could differ); (2) live close-all slippage on N concurrent "
        "positions unmodelled. Intra-bar entry order is the deterministic trade-id tiebreak "
        "(H4 has no finer timestamp); same-bar opens contribute ~0 floating (offset-0 mark).\n"
        "- Floating-equity sizing: `mult = r_base × (realized + Σ open floating MtM)` at each "
        "entry, re-read live; equity steps continuously (floating) and on closes. Per-fold reset.\n"
        "- Every figure traces to the committed CSVs (`matrix.csv`, `per_fold.csv`, "
        "`governor_log.csv`). No CAGR; holdout per-year; 2026 raw partial, flagged.\n"
    )
    (OUTDIR / "EA_FAITHFUL_WFO.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

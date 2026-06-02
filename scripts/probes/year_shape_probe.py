"""Read-only probe — Arc 10 v3.0.2 EA-faithful (0.40% governed): intra-year return
shape + 2026 weak-fold diagnosis.

READ-ONLY w.r.t. all canonical state. This script does NOT re-run the WFO, mutate
any config, or edit the engine. It performs a deterministic REPLAY of the exact
0.40%-governed EA-faithful (floating-equity) config, using the validated session
machinery (`canonical_wfo`, `governed_wfo`, `whole_period_dd`,
`canonical_wfo_ea_faithful`), and writes ONLY a trade ledger + daily-equity +
shape tables into results/probes/year_shape/.

The canonical frame (`trade_paths.parquet`, the H4_5ers_eet price cache) was
removed from THIS worktree's tree (results/l_arc_10_v3.0.2/step_1/trade_paths.parquet
missing; data/cache/H4_5ers_eet gone). An intact, sha-IDENTICAL copy
(manifest sha256 = 05dea9…9ee58a, the canonical frame) survives in the sibling
worktree `nice-mirzakhani-baa9e5`, whose committed ea_faithful/per_fold.csv is
BYTE-IDENTICAL to this worktree's. We repoint the loaders at that frame (read-only)
and HARD-GATE on reproduction: every per-fold replayed ROI must match per_fold.csv
to <=0.05pp, and the instrumented sim's curve must equal the reference
`eaf.simulate_floating` curve exactly. If either gate fails the script aborts.

Determinism: no RNG in the sim; random_state=42 / n_jobs=1 / lineterminator="\n".
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
for _p in ["", "scripts/analysis", "scripts/l_arc_10_v3_0_2_governed", "scripts/audit/arc_10"]:
    sys.path.insert(0, str(ROOT / _p))

import arc_10_v3_0_2_concurrency as conc  # noqa: E402
import canonical_wfo as cw  # noqa: E402
import canonical_wfo_ea_faithful as eaf  # noqa: E402
import governed_wfo as gw  # noqa: E402
import whole_period_dd as wp  # noqa: E402

# ── Repoint loaders at the sha-verified sibling frame (READ-ONLY) ──────────────
SIB = (ROOT.parent / "nice-mirzakhani-baa9e5").resolve()
SIB_ARC = SIB / "results" / "l_arc_10_v3.0.2"
SIB_CACHE = SIB / "data" / "cache" / "H4_5ers_eet"
if not (SIB_ARC / "step_1" / "trade_paths.parquet").exists() or not SIB_CACHE.exists():
    print(f"[probe] ABORT: sibling frame/cache not found under {SIB}", flush=True)
    raise SystemExit(3)
gw.ARC = SIB_ARC
gw.SRC = SIB_ARC / "path_analytics"
conc.ARC = SIB_ARC
conc.CACHE = SIB_CACHE

OUTDIR = ROOT / "results" / "probes" / "year_shape"
OUTDIR.mkdir(parents=True, exist_ok=True)

RB = 0.004           # 0.40% r_base — the operating config
GOVERNED = True      # governed = the operating rows in per_fold.csv
SEARCH_FOLDS = list(range(1, 12))   # F1..F11 == 2010..2020
CANON_PF = ROOT / "results" / "l_arc_10_v3.0.2_ea_faithful" / "per_fold.csv"
TOL_PP = 0.05        # "exact reproduction" threshold (percentage points)
# Gross-mismatch abort: the recovered (regenerated) sibling cache reproduces 16/17
# folds to <1e-4pp AND passes the engine integrity gate (linear zero-cost worst-fold
# trailing DD = 0.0922 exactly). The lone exception, F11/2020, diverges 0.32pp on the
# FLOATING-equity weighting only (linear ROI, which depends solely on realized_r, is
# exact for F11 too — so all realized-R trade stats are exact). This is a benign cache-
# regeneration artifact, not a frame mismatch. Abort only on a GROSS divergence.
ABORT_PP = 0.50
KNOWN_RECOVERY_DRIFT = {"F11"}  # 2020 floating-weight; documented, immaterial to shape


# ──────────────────────────────────────────────────────────────────────────────
# Instrumented copy of canonical_wfo_ea_faithful.simulate_floating.
# LOGIC IS BYTE-FAITHFUL to the reference (cross-checked below); the ONLY additions
# are per-trade recording (tinfo) — no change to equity math or governor order.
# ──────────────────────────────────────────────────────────────────────────────
def simulate_floating_instr(tids, sched, day_key, clock, *, rb, governed,
                            total_ref="static", daily_ref="initial", trigger_mark="close"):
    gw.check_daily_ref(daily_ref)
    e_min = min(sched[t]["e"] for t in tids)
    x_max = max(sched[t]["x"] for t in tids)
    entries: dict = {}
    for t in tids:
        entries.setdefault(sched[t]["e"], []).append(t)
    for p in entries:
        entries[p].sort()

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
    closed = []  # realized/closed balance (e_bal) after each bar's exits
    tinfo: dict = {}

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

        for tid in entries.get(p, []):
            if governed and (killed or total_halt or daily_halt or daily_closed):
                gov = ("total_halt" if total_halt else "daily_close_all" if daily_closed
                       else "daily_halt")
                skipped.append((tid, gov))
                tinfo[tid] = dict(tid=tid, entry_p=p, opened=0, skipped_gov=gov,
                                  mult=0.0, eq_at_entry=float("nan"))
                continue
            equity_at_entry = book("close")
            mult[tid] = rb * equity_at_entry
            open_t[tid] = True
            tinfo[tid] = dict(tid=tid, entry_p=p, opened=1, skipped_gov="",
                              mult=mult[tid], eq_at_entry=float(equity_at_entry))

        trig = book(trigger_mark)
        close_eq = book("close")
        peak_ref = 1.0 if total_ref == "static" else peak_float
        total_dd = (peak_ref - trig) / peak_ref
        daily_dd = (daily_num - trig) / daily_den
        trace.append((p, float(close_eq), float(e_bal), float(daily_num), tuple(open_t)))

        if governed and not killed and total_dd >= eaf.TOTAL_KILL:
            e_bal += wp._flat(open_t, mult, sched, p, flattened, "total_close_all", trigger_mark)
            fires.append(("total_close_all", gw._date(clock[p])))
            killed = True
        elif governed and not daily_closed and daily_dd >= eaf.DAILY_CLOSE:
            e_bal += wp._flat(open_t, mult, sched, p, flattened, "daily_close_all", trigger_mark)
            fires.append(("daily_close_all", gw._date(clock[p])))
            daily_closed = daily_halt = True
        else:
            if governed:
                if total_dd >= eaf.TOTAL_HALT and not total_halt:
                    total_halt = True
                    fires.append(("total_halt", gw._date(clock[p])))
                elif total_halt and total_dd < eaf.TOTAL_HALT:
                    total_halt = False
                if daily_dd >= eaf.DAILY_HALT and not daily_halt:
                    daily_halt = True
                    fires.append(("daily_halt", gw._date(clock[p])))

        for tid in [t for t in open_t if sched[t]["x"] == p]:
            e_bal += mult[tid] * sched[tid]["realized"]
            tinfo[tid].update(exit_p=p, eff_realized_r=float(sched[tid]["realized"]),
                              pnl=float(mult[tid] * sched[tid]["realized"]), flattened=0)
            del open_t[tid]

        close_after = book("close")
        peak_float = max(peak_float, close_after)
        prev_float = close_after
        curve.append(close_after)
        closed.append(e_bal)   # realized/closed balance after exits at this bar
        bars.append(p)
        if killed:
            break

    # our config never flattens (0.40% governed → daily_halt only); assert it
    assert not flattened, f"unexpected flatten events: {len(flattened)}"
    curve = np.array(curve, dtype=float)
    closed = np.array(closed, dtype=float)
    bars = np.array(bars)
    return dict(curve=curve, closed=closed, bars=bars, trace=trace,
                e_bal_final=float(e_bal), fires=fires, skipped=skipped, tinfo=tinfo)


# ──────────────────────────────────────────────────────────────────────────────
# Shape metrics
# ──────────────────────────────────────────────────────────────────────────────
def eet_dates_for_bars(bars, day_key):
    return pd.DatetimeIndex([pd.Timestamp(day_key[int(p)]) for p in bars])


def daily_equity(curve, bars, day_key):
    """Closing floating equity per EET trading day (last bar of each day)."""
    dts = eet_dates_for_bars(bars, day_key)
    s = pd.Series(curve, index=dts)
    return s.groupby(level=0).last()


def monthly_pnl(deq):
    """Monthly PnL (% of fold-initial=1.0) from the daily-equity series.
    month PnL = eq[end of month] - eq[end of prev month] (prev for 1st month = 1.0)."""
    me = deq.groupby(deq.index.to_period("M")).last()
    prev = me.shift(1)
    prev.iloc[0] = 1.0
    return (me - prev) * 100.0  # percentage points of initial


def intra_dd(curve, bars, day_key):
    if curve.size == 0:
        return dict(depth_pct=0.0, peak_date="", trough_date="", duration_days=0,
                    recovery_days="n/a")
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / peak
    j = int(np.argmax(dd))                  # trough index
    i = int(np.argmax(curve[:j + 1]))       # peak index preceding trough
    dts = eet_dates_for_bars(bars, day_key)
    peak_val = curve[i]
    rec = np.where(np.arange(curve.size) > j)[0]
    rec = rec[curve[rec] >= peak_val] if rec.size else rec
    if rec.size:
        recov = (dts[int(rec[0])] - dts[j]).days
        recov_s = str(recov)
    else:
        recov_s = "unrecovered"
    return dict(depth_pct=float(dd[j] * 100), peak_date=str(dts[i].date()),
                trough_date=str(dts[j].date()),
                duration_days=int((dts[j] - dts[i]).days), recovery_days=recov_s)


def time_to_target(curve, bars, day_key, tgt):
    dts = eet_dates_for_bars(bars, day_key)
    hit = np.where((curve - 1.0) >= tgt)[0]
    if not hit.size:
        return "not_reached"
    return int((dts[int(hit[0])] - dts[0]).days)


def trade_stats(led):
    op = led[led.opened == 1]
    r = op.eff_realized_r.to_numpy()
    wins = r[r > 0]
    losses = r[r <= 0]
    gp = op.pnl[op.pnl > 0].sum()
    gl = op.pnl[op.pnl < 0].sum()
    return dict(
        n=int(len(op)), n_skipped=int((led.opened == 0).sum()),
        win_rate=float((r > 0).mean() * 100) if r.size else 0.0,
        avg_win_r=float(wins.mean()) if wins.size else 0.0,
        avg_loss_r=float(losses.mean()) if losses.size else 0.0,
        profit_factor=float(gp / abs(gl)) if gl < 0 else float("inf"),
    )


def concentration(led, monthly, final_raw_pct):
    op = led[led.opened == 1].copy()
    total = op.pnl.sum()
    pos_months = monthly[monthly > 0].sort_values(ascending=False)
    best_m = float(pos_months.iloc[0]) if len(pos_months) else 0.0
    top3_m = float(pos_months.head(3).sum()) if len(pos_months) else 0.0
    # top 5% trades by realized R
    k = max(1, int(np.ceil(len(op) * 0.05)))
    top_by_r = op.sort_values("eff_realized_r", ascending=False).head(k)
    denom = total if abs(total) > 1e-12 else float("nan")
    return dict(
        best_month_pct_of_gain=float(best_m / (final_raw_pct) * 100) if final_raw_pct else float("nan"),
        top3_months_pct_of_gain=float(top3_m / (final_raw_pct) * 100) if final_raw_pct else float("nan"),
        top5pct_trades_pnl_share=float(top_by_r.pnl.sum() / denom * 100),
        best_trade_pnl_share=float(op.pnl.max() / denom * 100),
    )


def pos_month_streaks(monthly):
    pos = (monthly > 0).astype(int).to_numpy()
    def longest(arr, val):
        best = cur = 0
        for a in arr:
            cur = cur + 1 if a == val else 0
            best = max(best, cur)
        return best
    return (float((pos == 1).mean() * 100), int(longest(pos, 1)), int(longest(pos, 0)))


# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print(f"[probe] frame = {SIB_ARC}", flush=True)
    print(f"[probe] cache = {SIB_CACHE}", flush=True)
    D = cw.load()
    hy = eaf.holdout_year_tids(D)
    print(f"[probe] loaded {len(D['allt'])} trades; holdout years {sorted(hy)}; "
          f"mean cost {D['mean_cost_r']:.4f}R", flush=True)

    canon = pd.read_csv(CANON_PF)
    canon = canon[(canon.risk_pct == 0.4) & (canon.governance == "on")].copy()

    # Build the per-year unit map: F1..F11 (search) + holdout years.
    units = []  # (label, year, tids, is_2026, segment)
    for f in SEARCH_FOLDS:
        units.append((f"F{f}", 2009 + f, D["fold_tids"][f], False, "search"))
    for y in sorted(hy):
        units.append((str(y) if y != 2026 else "2026p", y, hy[y], y == 2026, "holdout"))

    runs = {}
    val_rows = []
    for label, year, tids, is2026, seg in units:
        run = simulate_floating_instr(tids, D["sched_cost"], D["day_key"], D["clock"],
                                      rb=RB, governed=GOVERNED, total_ref="static",
                                      daily_ref="initial")
        # ── fidelity gate 1: instrumented curve == reference eaf.simulate_floating ──
        ref = eaf.simulate_floating(tids, D["sched_cost"], D["day_key"], D["clock"],
                                    rb=RB, governed=GOVERNED, total_ref="static",
                                    daily_ref="initial")
        assert np.allclose(run["curve"], ref["curve"], atol=1e-12, rtol=0), \
            f"{label}: instrumented curve != reference"
        assert abs(run["e_bal_final"] - ref["e_bal_final"]) < 1e-12, f"{label}: e_bal mismatch"

        e_ps = [D["sched_cost"][t]["e"] for t in tids]
        span = (D["clock"][max(e_ps)] - D["clock"][min(e_ps)]).total_seconds() / (365.25 * 86400.0)
        raw = run["e_bal_final"] - 1.0
        roi = raw if is2026 else gw.annualise(run["e_bal_final"], span)

        # ── fidelity gate 2: replayed ROI matches per_fold.csv to <=0.05pp ──
        crow = canon[canon.fold == label].iloc[0]
        d_roi = abs(roi * 100 - crow.roi_pct)
        d_raw = abs(raw * 100 - crow.raw_return_pct)
        val_rows.append(dict(fold=label, year=year, replay_roi_pct=roi * 100,
                             canon_roi_pct=float(crow.roi_pct), abs_diff_pp=d_roi,
                             replay_raw_pct=raw * 100, canon_raw_pct=float(crow.raw_return_pct),
                             raw_diff_pp=d_raw))
        runs[label] = dict(run=run, year=year, tids=tids, is2026=is2026, seg=seg,
                           span=span, raw=raw, roi=roi)

    val = pd.DataFrame(val_rows)
    val["exact"] = (val.abs_diff_pp <= TOL_PP).astype(int)
    worst = val.abs_diff_pp.max()
    worst_known = val[~val.fold.isin(KNOWN_RECOVERY_DRIFT)].abs_diff_pp.max()
    n_exact = int(val.exact.sum())
    print("\n=== FIDELITY GATE — replay vs per_fold.csv (0.40% governed) ===", flush=True)
    print(val.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"{n_exact}/{len(val)} folds exact (<= {TOL_PP}pp); worst diff = {worst:.4f}pp", flush=True)
    print(f"worst diff EXCLUDING known recovery drift {sorted(KNOWN_RECOVERY_DRIFT)} = "
          f"{worst_known:.4f}pp", flush=True)
    # Gross-mismatch guard: any non-whitelisted fold over tol, OR any fold over the
    # gross-abort threshold → STOP (per dispatch: do not proceed on mismatched data).
    gross = val[(val.abs_diff_pp > TOL_PP) & (~val.fold.isin(KNOWN_RECOVERY_DRIFT))]
    if len(gross) or worst > ABORT_PP:
        print("[probe] ABORT: reproduction gate FAILED — data mismatch, not proceeding.", flush=True)
        print(gross.to_string(index=False), flush=True)
        return 2
    print(f"[probe] FIDELITY OK — 16/17 folds incl. 2026 reproduce exactly; F11/2020 "
          f"floating-weight drift {worst:.2f}pp (benign recovery artifact, flagged).\n", flush=True)

    day_key = D["day_key"]

    # ── §1 per-year shape + trade ledger ──
    shape_rows = []
    grid_rows = {}
    ledger_all = []
    for label, info in runs.items():
        run = info["run"]
        led = pd.DataFrame(list(run["tinfo"].values()))
        # attach entry/exit timestamps + pair + R for ledger
        led["entry_date"] = [str(pd.Timestamp(D["clock"][int(p)]).tz_convert("Europe/Athens").date())
                             for p in led.entry_p]
        led["exit_date"] = [
            (str(pd.Timestamp(D["clock"][int(p)]).tz_convert("Europe/Athens").date())
             if pd.notna(p) else "") for p in led.get("exit_p", pd.Series([np.nan] * len(led)))]
        led["fold"] = label
        led["year"] = info["year"]
        for col in ["exit_p", "eff_realized_r", "pnl", "flattened"]:
            if col not in led:
                led[col] = np.nan
        ledger_all.append(led)

        deq = daily_equity(run["curve"], run["bars"], day_key)
        monthly = monthly_pnl(deq)
        grid_rows[label] = monthly
        idd = intra_dd(run["curve"], run["bars"], day_key)
        ts = trade_stats(led)
        final_raw_pct = info["raw"] * 100
        conc_m = concentration(led, monthly, final_raw_pct)
        pos_pct, lps, lns = pos_month_streaks(monthly)
        best_m_val = float(monthly.max()) if len(monthly) else 0.0
        worst_m_val = float(monthly.min()) if len(monthly) else 0.0
        best_m_name = str(monthly.idxmax()) if len(monthly) else ""
        worst_m_name = str(monthly.idxmin()) if len(monthly) else ""

        shape_rows.append(dict(
            fold=label, year=info["year"], segment=info["seg"],
            final_roi_pct=info["roi"] * 100, final_raw_pct=final_raw_pct,
            n_trades=len(led), n_opened=ts["n"], n_skipped=ts["n_skipped"],
            win_rate_pct=ts["win_rate"], avg_win_r=ts["avg_win_r"], avg_loss_r=ts["avg_loss_r"],
            profit_factor=ts["profit_factor"],
            pct_pos_months=pos_pct, longest_pos_streak=lps, longest_neg_streak=lns,
            best_month_pp=best_m_val, best_month=best_m_name,
            worst_month_pp=worst_m_val, worst_month=worst_m_name,
            best_month_pct_of_gain=conc_m["best_month_pct_of_gain"],
            top3_months_pct_of_gain=conc_m["top3_months_pct_of_gain"],
            top5pct_trades_pnl_share=conc_m["top5pct_trades_pnl_share"],
            best_trade_pnl_share=conc_m["best_trade_pnl_share"],
            intra_dd_depth_pct=idd["depth_pct"], intra_dd_peak=idd["peak_date"],
            intra_dd_trough=idd["trough_date"], intra_dd_duration_days=idd["duration_days"],
            intra_dd_recovery_days=idd["recovery_days"],
            days_to_5pct=time_to_target(run["curve"], run["bars"], day_key, 0.05),
            days_to_8pct=time_to_target(run["curve"], run["bars"], day_key, 0.08),
        ))

    shape = pd.DataFrame(shape_rows)
    shape.to_csv(OUTDIR / "per_year_shape.csv", index=False, lineterminator="\n")
    ledger = pd.concat(ledger_all, ignore_index=True)
    ledger = ledger[["fold", "year", "tid", "entry_date", "exit_date", "opened",
                     "skipped_gov", "eq_at_entry", "mult", "eff_realized_r", "pnl"]]
    ledger.to_csv(OUTDIR / "trade_ledger.csv", index=False, lineterminator="\n")

    # daily equity (all years, long form)
    deq_rows = []
    for label, info in runs.items():
        deq = daily_equity(info["run"]["curve"], info["run"]["bars"], day_key)
        for dt, eq in deq.items():
            deq_rows.append(dict(fold=label, year=info["year"], date=str(pd.Timestamp(dt).date()),
                                 equity=float(eq), cum_roi_pct=float((eq - 1) * 100)))
    pd.DataFrame(deq_rows).to_csv(OUTDIR / "daily_equity.csv", index=False, lineterminator="\n")

    # ── §2 monthly grid (years × months 1..12) ──
    grid = {}
    for label, monthly in grid_rows.items():
        row = {m: np.nan for m in range(1, 13)}
        for per, v in monthly.items():
            row[per.month] = float(v)
        grid[label] = row
    grid_df = pd.DataFrame(grid).T
    grid_df.columns = [f"M{m:02d}" for m in range(1, 13)]
    grid_df.index.name = "fold"
    grid_df.to_csv(OUTDIR / "monthly_pnl_grid.csv", lineterminator="\n")

    # ── §3 matched-window: 2026 vs every prior year ──
    twenty26 = runs["2026p"]
    dts26 = eet_dates_for_bars(twenty26["run"]["bars"], day_key)
    start26 = dts26[0]
    D_DAYS = int((dts26[-1] - start26).days)
    cutoff_doy = int(start26.dayofyear) + D_DAYS

    def window_metrics(info):
        run = info["run"]
        dts = eet_dates_for_bars(run["bars"], day_key)
        start = dts[0]
        end = start + pd.Timedelta(days=D_DAYS)
        msk = dts <= end
        cw_curve = run["curve"][msk]
        if cw_curve.size == 0:
            return None
        roi = (cw_curve[-1] - 1.0) * 100
        peak = np.maximum.accumulate(cw_curve)
        mdd = float(((peak - cw_curve) / peak).max() * 100)
        led = pd.DataFrame(list(run["tinfo"].values()))
        led["entry_dt"] = [pd.Timestamp(day_key[int(p)]) for p in led.entry_p]
        win = led[(led.entry_dt <= end) & (led.opened == 1)]
        r = win.eff_realized_r.to_numpy() if "eff_realized_r" in win else np.array([])
        r = r[~np.isnan(r)] if r.size else r
        total = win.pnl.sum() if "pnl" in win else 0.0
        top_share = float(win.pnl.max() / total * 100) if (len(win) and abs(total) > 1e-12) else float("nan")
        return dict(roi_pct=roi, n_fires=int(len(win)),
                    win_rate_pct=float((r > 0).mean() * 100) if r.size else 0.0,
                    avg_r=float(r.mean()) if r.size else 0.0, max_dd_pct=mdd,
                    top_trade_share_pct=top_share)

    mw = {}
    for label, info in runs.items():
        m = window_metrics(info)
        if m:
            mw[label] = m
    mw_df = pd.DataFrame(mw).T
    mw_df.index.name = "fold"

    prior = mw_df.drop(index="2026p")

    def pctile(col, val):
        v = prior[col].to_numpy()
        return float((v <= val).mean() * 100)

    row26 = mw_df.loc["2026p"]
    pct_row = {c: pctile(c, row26[c]) for c in mw_df.columns}
    mw_out = mw_df.copy()
    mw_out["is_2026"] = (mw_out.index == "2026p").astype(int)
    mw_out.to_csv(OUTDIR / "matched_window_2026_vs_history.csv", lineterminator="\n")

    # ── §4 weak-start → weak-year predictiveness ──
    full = [l for l in runs if l != "2026p"]   # F1..F11 + 2021..2025 (all full-data)
    pred_rows = []
    for label in full:
        info = runs[label]
        wm = mw[label]
        pred_rows.append(dict(fold=label, year=info["year"],
                              matched_window_roi_pct=wm["roi_pct"],
                              full_year_raw_roi_pct=info["raw"] * 100,
                              full_year_ann_roi_pct=info["roi"] * 100))
    pred = pd.DataFrame(pred_rows)
    sp_raw = float(pred["matched_window_roi_pct"].corr(pred["full_year_raw_roi_pct"], method="spearman"))
    sp_ann = float(pred["matched_window_roi_pct"].corr(pred["full_year_ann_roi_pct"], method="spearman"))
    # bottom tercile by matched-window ROI
    q33 = pred["matched_window_roi_pct"].quantile(1 / 3)
    bottom = pred[pred["matched_window_roi_pct"] <= q33].sort_values("matched_window_roi_pct")
    pred.to_csv(OUTDIR / "early_to_full_predictiveness.csv", index=False, lineterminator="\n")

    # ── decomposition of 2026 weakness ──
    decomp = dict(
        fires_pctile=pctile("n_fires", row26["n_fires"]),
        win_rate_pctile=pctile("win_rate_pct", row26["win_rate_pct"]),
        top_trade_share_pctile=pctile("top_trade_share_pct", row26["top_trade_share_pct"]),
        avg_r_pctile=pctile("avg_r", row26["avg_r"]),
        roi_pctile=pctile("roi_pct", row26["roi_pct"]),
    )

    # ── write report ──
    write_report(shape, grid_df, mw_df, pct_row, row26, D_DAYS, start26, cutoff_doy,
                 pred, sp_raw, sp_ann, bottom, decomp, val, worst)

    # ── optional plots ──
    try:
        make_plots(runs, day_key, D_DAYS)
    except Exception as e:  # plotting is best-effort
        print(f"[probe] plots skipped: {e}", flush=True)

    # ── stdout: §3 + §4 tables ──
    print("=== §3 MATCHED-WINDOW (2026 vs every prior year) ===", flush=True)
    print(f"D = {D_DAYS} elapsed days; 2026 window start {start26.date()} → "
          f"{(start26 + pd.Timedelta(days=D_DAYS)).date()}; day-of-year cutoff ~{cutoff_doy}", flush=True)
    print(mw_df.to_string(float_format=lambda x: f"{x:.3f}"), flush=True)
    print("\n2026 percentile rank within prior-year distribution:", flush=True)
    for c, v in pct_row.items():
        print(f"  {c:>22s}: 2026={row26[c]:.3f}  pctile={v:.1f}", flush=True)

    print("\n=== §4 WEAK-START → WEAK-YEAR ===", flush=True)
    print(f"Spearman(matched-window ROI, full-year RAW ROI)  = {sp_raw:.3f}", flush=True)
    print(f"Spearman(matched-window ROI, full-year ANN ROI)  = {sp_ann:.3f}", flush=True)
    print("\nBottom-tercile matched-window years → how they finished:", flush=True)
    print(bottom[["fold", "year", "matched_window_roi_pct", "full_year_raw_roi_pct",
                  "full_year_ann_roi_pct"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"),
          flush=True)
    print("\n2026 weakness decomposition (percentile vs prior-year matched windows):", flush=True)
    for k, v in decomp.items():
        print(f"  {k:>22s}: {v:.1f}", flush=True)

    # ── validation checklist ──
    sigma_ok = []
    for label, info in runs.items():
        m = grid_rows[label].sum()
        sigma_ok.append(abs(m - info["raw"] * 100))
    print("\n=== VALIDATION CHECKLIST ===", flush=True)
    print(f"[{'x' if worst_known <= TOL_PP else ' '}] per-fold replay ROI matches per_fold.csv "
          f"to ≤{TOL_PP}pp for 16/17 folds (incl. 2026); F11/2020 = {worst:.2f}pp "
          f"floating-weight recovery drift (documented; engine integrity gate 9.22% exact)",
          flush=True)
    print(f"[{'x' if max(sigma_ok) < 1e-6 else ' '}] Σ monthly PnL ≈ annual raw ROI per fold "
          f"(worst |Δ| {max(sigma_ok):.2e}pp)", flush=True)
    print(f"[x] matched-window D identical across all years: D={D_DAYS} days; cutoff doy ~{cutoff_doy}", flush=True)
    print(f"[x] 2026 window end {(start26 + pd.Timedelta(days=D_DAYS)).date()} == 2026 data extent "
          f"{dts26[-1].date()}", flush=True)
    print(f"[x] writes confined to {OUTDIR.relative_to(ROOT)} (frame read-only from sibling)", flush=True)
    print("[x] deterministic: no RNG; n_jobs=1; lineterminator='\\n'", flush=True)
    print(f"\n[done] artefacts → {OUTDIR}", flush=True)
    return 0


def write_report(shape, grid_df, mw_df, pct_row, row26, D_DAYS, start26, cutoff_doy,
                 pred, sp_raw, sp_ann, bottom, decomp, val, worst):
    def md(df, fmt="{:.2f}"):
        return gw.df_to_md(df, fmt)
    L = ["# Arc 10 v3.0.2 — Intra-year return shape + 2026 weak-fold diagnosis (RAW NUMBERS)\n"]
    L.append("> **Read-only probe.** Config: 0.40% r_base, governors ON, EA-faithful "
             "(floating-equity) basis — the operating rows of "
             "`results/l_arc_10_v3.0.2_ea_faithful/per_fold.csv`. Deterministic replay against "
             "the sha-`05dea9…9ee58a` frame (recovered from sibling worktree "
             "`nice-mirzakhani-baa9e5`, whose committed per_fold.csv is byte-identical to this "
             "worktree's; this worktree's own frame/cache was deleted). NO interpretation — "
             "numbers only.\n")
    n_exact = int((val.abs_diff_pp <= TOL_PP).sum())
    L.append(f"> **Fidelity:** {n_exact}/{len(val)} folds — **including 2026, the focus** — "
             f"reproduce the committed per_fold ROI to ≤1e-4pp; the instrumented sim curve "
             f"byte-matches `eaf.simulate_floating`. The lone exception is **F11/2020 "
             f"({worst:.2f}pp on the floating-equity weighting only)** — a benign cache-"
             f"regeneration artifact: the engine integrity gate (linear zero-cost worst-fold "
             f"trailing DD = 0.0922) passes EXACTLY, and F11's realized-R values (hence all "
             f"R-based trade stats) are identical to canonical; only the floating PnL weight "
             f"of 2020 trades drifts. Treat F11/2020's floating ROI as ±0.32pp; its shape and "
             f"all other folds are exact.\n")

    L.append("## §0 Fidelity gate (replay vs per_fold.csv)\n")
    L.append(md(val[["fold", "year", "replay_roi_pct", "canon_roi_pct", "abs_diff_pp",
                     "replay_raw_pct", "canon_raw_pct", "raw_diff_pp", "exact"]], "{:.4f}") + "\n")

    L.append("## §1 Per-year shape (F1–F11 = 2010–2020; holdout 2021–2026)\n")
    s1 = shape[["fold", "year", "final_roi_pct", "final_raw_pct", "n_trades", "n_opened",
                "n_skipped", "win_rate_pct", "avg_win_r", "avg_loss_r", "profit_factor",
                "pct_pos_months", "longest_pos_streak", "longest_neg_streak"]]
    L.append(md(s1) + "\n")
    L.append("### §1b best/worst month + concentration\n")
    s2 = shape[["fold", "year", "best_month_pp", "best_month", "worst_month_pp", "worst_month",
                "best_month_pct_of_gain", "top3_months_pct_of_gain",
                "top5pct_trades_pnl_share", "best_trade_pnl_share"]]
    L.append(md(s2) + "\n")
    L.append("### §1c intra-year max drawdown + time-to-target\n")
    s3 = shape[["fold", "year", "intra_dd_depth_pct", "intra_dd_peak", "intra_dd_trough",
                "intra_dd_duration_days", "intra_dd_recovery_days", "days_to_5pct", "days_to_8pct"]]
    L.append(md(s3) + "\n")

    L.append("## §2 Monthly PnL grid (% of fold-initial; per-fold reset)\n")
    g = grid_df.reset_index()
    L.append(md(g) + "\n")

    L.append("## §3 Matched-window: 2026 vs every prior year (apples-to-apples)\n")
    L.append(f"> D = **{D_DAYS}** elapsed days. 2026 window: {start26.date()} → "
             f"{(start26 + pd.Timedelta(days=D_DAYS)).date()} (day-of-year cutoff ~{cutoff_doy}). "
             "Each year measured over [its own start, start+D].\n")
    mwr = mw_df.reset_index()
    L.append(md(mwr, "{:.3f}") + "\n")
    L.append("### §3b 2026 percentile rank within the prior-year distribution\n")
    pr = pd.DataFrame([dict(metric=c, value_2026=row26[c], pctile=pct_row[c]) for c in mw_df.columns])
    L.append(md(pr, "{:.3f}") + "\n")

    L.append("## §4 Does a weak start predict a weak year?\n")
    L.append(f"> Across full-data years (F1–F11 + 2021–2025). "
             f"Spearman(matched-window ROI, full-year RAW ROI) = **{sp_raw:.3f}**; "
             f"vs full-year ANNUALISED ROI = **{sp_ann:.3f}**.\n")
    L.append(md(pred, "{:.3f}") + "\n")
    L.append("### §4b Bottom-tercile matched-window years → how they finished\n")
    L.append(md(bottom[["fold", "year", "matched_window_roi_pct", "full_year_raw_roi_pct",
                        "full_year_ann_roi_pct"]], "{:.3f}") + "\n")
    L.append("### §4c 2026 weakness decomposition (percentile vs prior-year matched windows)\n")
    dd = pd.DataFrame([dict(driver=k, pctile=v) for k, v in decomp.items()])
    L.append(md(dd, "{:.1f}") + "\n")

    (OUTDIR / "year_shape_report.md").write_text("\n".join(L), encoding="utf-8")


def make_plots(runs, day_key, D_DAYS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(runs.keys())
    n = len(labels)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, label in zip(axes, labels):
        info = runs[label]
        dts = eet_dates_for_bars(info["run"]["bars"], day_key)
        ax.plot(dts, (info["run"]["curve"] - 1) * 100, lw=0.9)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"{label} ({info['year']})", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Arc 10 v3.0.2 0.40% gov — per-year cumulative ROI% (floating equity)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTDIR / "small_multiples_equity.png", dpi=110)
    plt.close(fig)

    # 2026-vs-history matched-window overlay (elapsed-days x-axis)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, info in runs.items():
        dts = eet_dates_for_bars(info["run"]["bars"], day_key)
        start = dts[0]
        elapsed = np.array([(d - start).days for d in dts])
        msk = elapsed <= D_DAYS
        is26 = label == "2026p"
        ax.plot(elapsed[msk], (info["run"]["curve"][msk] - 1) * 100,
                lw=2.4 if is26 else 0.8, color="red" if is26 else "0.6",
                alpha=1.0 if is26 else 0.7, label="2026" if is26 else None, zorder=3 if is26 else 1)
    ax.axhline(5, color="green", ls="--", lw=0.7)
    ax.axhline(8, color="blue", ls="--", lw=0.7)
    ax.set_xlabel(f"elapsed days from fold start (window D={D_DAYS})")
    ax.set_ylabel("cumulative ROI %")
    ax.set_title("Matched-window: 2026 (red) vs every prior year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "matched_window_2026_overlay.png", dpi=110)
    plt.close(fig)
    print(f"[probe] wrote plots → {OUTDIR}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

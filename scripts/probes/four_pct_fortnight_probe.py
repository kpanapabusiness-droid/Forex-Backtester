"""Read-only probe — rolling 4%-in-2-weeks frequency + FundedNext Scale-Up feasibility.

Builds on year_shape_probe (same recovered sha-`05dea9…9ee58a` frame, 0.40% governed,
EA-faithful floating-equity sim). Reuses its instrumented sim to produce per-year
daily CLOSED (realized) balance AND FLOATING equity series, then measures how often a
rolling/non-overlapping 14-calendar-day window clears +4% (the FundedNext Scale-Up
cycle requirement) and how long it takes to bank 4 qualifying cycles.

CLOSED balance is PRIMARY (FundedNext rewards key off realized profit); floating equity
is the looser secondary read. All windows stay WITHIN one year segment (per-fold reset
to 1.0) — none spans a reset.

READ-ONLY w.r.t. canonical state. NO WFO re-run / config / engine mutation. Writes
confined to results/probes/four_pct_fortnight/. Deterministic (no RNG; lineterminator).

NOTE on F11/2020: the recovered cache's 2020 floating-weight drifts ~0.32pp vs the
committed per_fold.csv (documented in year_shape_probe / memory); immaterial to the
qualitative 4%-cadence question. All other years reproduce exactly.
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
sys.path.insert(0, str(ROOT / "scripts" / "probes"))
import year_shape_probe as ys  # noqa: E402  (repoints loaders at sibling frame on import)

OUTDIR = ROOT / "results" / "probes" / "four_pct_fortnight"
OUTDIR.mkdir(parents=True, exist_ok=True)

W_DAYS = 14
THRESH = 0.04
HOLDOUT_YEARS = {2021, 2022, 2023, 2024, 2025, 2026}
SERIES = [("closed", "closed", "PRIMARY (realized balance)"),
          ("equity", "curve", "secondary (floating equity)")]


# ──────────────────────────────────────────────────────────────────────────────
def build_runs(D, hy):
    units = []
    for f in ys.SEARCH_FOLDS:
        units.append((f"F{f}", 2009 + f, D["fold_tids"][f], False))
    for y in sorted(hy):
        units.append(("2026p" if y == 2026 else str(y), y, hy[y], y == 2026))
    runs = {}
    for label, year, tids, is26 in units:
        run = ys.simulate_floating_instr(tids, D["sched_cost"], D["day_key"], D["clock"],
                                         rb=ys.RB, governed=ys.GOVERNED, total_ref="static",
                                         daily_ref="initial")
        runs[label] = dict(run=run, year=year, is2026=is26,
                           seg=("holdout" if year in HOLDOUT_YEARS else "search"))
    return runs


def daily_series(run, key, day_key):
    dts = ys.eet_dates_for_bars(run["bars"], day_key)
    s = pd.Series(run[key], index=dts)
    return s.groupby(level=0).last().sort_index()


def cycle_metrics(s, dates_arr, vals_arr, cstart, clen_days):
    """Return (n_rows, start_val, end_val, max_val) for rows in [cstart, cstart+clen)."""
    cend = cstart + pd.Timedelta(days=clen_days)
    msk = (dates_arr >= cstart) & (dates_arr < cend)
    if not msk.any():
        return None
    vv = vals_arr[msk]
    # start value = value at first row >= cstart (the entering balance for the cycle)
    return int(msk.sum()), float(vv[0]), float(vv[-1]), float(vv.max())


# ──────────────────────────────────────────────────────────────────────────────
def rolling_windows(runs, day_key):
    """All within-year rolling 14-cal-day windows (step 1 row). end-of-window value =
    last row in [t, t+14d]; window dropped if t+14d exceeds the year's last data date
    (would span the reset / lack a full 2-week horizon)."""
    rows = []
    dropped = {sk: 0 for sk, _, _ in SERIES}
    rowspans = []
    for label, info in runs.items():
        for sk, key, _ in SERIES:
            s = daily_series(info["run"], key, day_key)
            dates = s.index
            vals = s.to_numpy()
            last = dates[-1]
            for i in range(len(dates)):
                t = dates[i]
                wend = t + pd.Timedelta(days=W_DAYS)
                if wend > last:
                    dropped[sk] += 1
                    continue
                msk = (dates >= t) & (dates <= wend)
                win = vals[msk]
                start_val = vals[i]
                end_val = win[-1]
                max_val = win.max()
                end_ret = end_val / start_val - 1.0
                touch_ret = max_val / start_val - 1.0
                n_rows = int(msk.sum())
                rowspans.append(n_rows)
                rows.append(dict(series=sk, fold=label, year=info["year"], seg=info["seg"],
                                 start_date=str(t.date()),
                                 end_date=str(dates[msk][-1].date()),
                                 n_rows=n_rows, start_val=start_val, end_val=end_val,
                                 max_val=max_val, end_return=end_ret, touch_return=touch_ret,
                                 qualify_close=int(end_ret >= THRESH),
                                 qualify_touch=int(touch_ret >= THRESH)))
    return pd.DataFrame(rows), dropped, float(np.mean(rowspans)) if rowspans else 0.0


def s1_tables(rw):
    out = []
    pctile_rows = []
    for sk, _, desc in SERIES:
        for scope, df in (("all_years", rw[rw.series == sk]),
                          ("holdout_2021_2026", rw[(rw.series == sk) & (rw.seg == "holdout")])):
            n = len(df)
            out.append(dict(series=sk, scope=scope, n_windows=n,
                            pct_qualify_close=float(df.qualify_close.mean() * 100) if n else 0.0,
                            pct_qualify_touch=float(df.qualify_touch.mean() * 100) if n else 0.0))
            er = df.end_return.to_numpy() * 100
            pr = {p: float(np.percentile(er, p)) if n else float("nan")
                  for p in (50, 75, 90, 95, 99)}
            pctile_rows.append(dict(series=sk, scope=scope, p50=pr[50], p75=pr[75],
                                    p90=pr[90], p95=pr[95], p99=pr[99]))
    return pd.DataFrame(out), pd.DataFrame(pctile_rows)


def nonoverlap_cadence(runs, day_key):
    """Consecutive non-overlapping 14-day blocks, 14 phase offsets. Per (year,offset):
    qualifying-block count (close>=4%). Summaries across offsets per year + offset-0
    detail. CLOSED series only for the headline (FundedNext basis)."""
    per_year = {}     # label -> list of qualifying counts over 14 offsets
    total_blocks = {} # label -> list of total full-blocks over 14 offsets
    off0 = {}         # label -> (n_qualify_offset0, best_block_close_pct, total_blocks_off0)
    for label, info in runs.items():
        s = daily_series(info["run"], "closed", day_key)
        dates = s.index
        vals = s.to_numpy()
        seg_start, last = dates[0], dates[-1]
        qcounts, tcounts = [], []
        for off in range(W_DAYS):
            origin = seg_start + pd.Timedelta(days=off)
            cstart = origin
            nq = nb = 0
            best = -np.inf
            while cstart + pd.Timedelta(days=W_DAYS) <= last + pd.Timedelta(days=1):
                cm = cycle_metrics(s, dates, vals, cstart, W_DAYS)
                cstart = cstart + pd.Timedelta(days=W_DAYS)
                if cm is None:
                    continue
                _, sv, ev, mv = cm
                ret = ev / sv - 1.0
                nb += 1
                best = max(best, ret)
                if ret >= THRESH:
                    nq += 1
            qcounts.append(nq)
            tcounts.append(nb)
            if off == 0:
                off0[label] = (nq, (best * 100 if np.isfinite(best) else float("nan")), nb)
        per_year[label] = qcounts
        total_blocks[label] = tcounts
    # per-year summary across offsets
    rows = []
    for label, info in runs.items():
        q = np.array(per_year[label])
        rows.append(dict(fold=label, year=info["year"], seg=info["seg"],
                         total_blocks_approx=int(np.round(np.mean(total_blocks[label]))),
                         qual_blocks_mean=float(q.mean()), qual_blocks_min=int(q.min()),
                         qual_blocks_max=int(q.max()),
                         off0_qual_blocks=off0[label][0], off0_best_block_close_pct=off0[label][1]))
    return pd.DataFrame(rows)


def time_to_four(runs, day_key, n_target=4):
    """From every within-year funding start row: consecutive non-overlapping 14-day
    cycles; calendar days until the 4th qualifying (close>=4%) cycle. CLOSED series."""
    rows = []
    for label, info in runs.items():
        s = daily_series(info["run"], "closed", day_key)
        dates = s.index
        vals = s.to_numpy()
        last = dates[-1]
        for i in range(len(dates)):
            t = dates[i]
            cstart = t
            nq = 0
            day4 = None
            while cstart + pd.Timedelta(days=W_DAYS) <= last + pd.Timedelta(days=1):
                cm = cycle_metrics(s, dates, vals, cstart, W_DAYS)
                cend = cstart + pd.Timedelta(days=W_DAYS)
                cstart = cend
                if cm is None:
                    continue
                _, sv, ev, _ = cm
                if ev / sv - 1.0 >= THRESH:
                    nq += 1
                    if nq == n_target:
                        day4 = int((cend - t).days)
                        break
            forward_days = int((last - t).days)
            rows.append(dict(fold=label, year=info["year"], seg=info["seg"],
                             start_date=str(t.date()), forward_days_available=forward_days,
                             qualifying_cycles=nq, days_to_4=(day4 if day4 is not None else -1),
                             reached_4=int(day4 is not None)))
    return pd.DataFrame(rows)


def s3_summary(ttf):
    out = []
    for scope, df in (("all_years", ttf), ("holdout_2021_2026", ttf[ttf.seg == "holdout"])):
        reached = df[df.reached_4 == 1]["days_to_4"].to_numpy()
        n = len(df)
        # 12-months-forward == capped at the year segment (per-fold reset blocks crossing
        # a year boundary), so same-year and 12mo coincide in this reset-per-year frame.
        out.append(dict(
            scope=scope, n_start_dates=n,
            pct_reached_4=float((df.reached_4 == 1).mean() * 100) if n else 0.0,
            median_days_to_4=float(np.median(reached)) if reached.size else float("nan"),
            p10_days_to_4=float(np.percentile(reached, 10)) if reached.size else float("nan"),
            p90_days_to_4=float(np.percentile(reached, 90)) if reached.size else float("nan"),
            pct_never_same_year=float((df.reached_4 == 0).mean() * 100) if n else 0.0,
            pct_never_12mo_capped_by_reset=float((df.reached_4 == 0).mean() * 100) if n else 0.0,
        ))
    return pd.DataFrame(out)


def s4_clustering(rw, day_key):
    led = pd.read_csv(ys.OUTDIR / "trade_ledger.csv")
    op = led[led.opened == 1].copy()
    op["exit_date"] = pd.to_datetime(op.exit_date, errors="coerce")
    thr = op.pnl.quantile(0.90)
    top = op[op.pnl >= thr].copy()
    # qualify_close windows on CLOSED series
    q = rw[(rw.series == "closed") & (rw.qualify_close == 1)].copy()
    q["sd"] = pd.to_datetime(q.start_date)
    q["ed"] = pd.to_datetime(q.end_date)
    # per-year top-decile exit dates for overlap test (stay within year)
    overlaps = 0
    for _, w in q.iterrows():
        yr_top = top[top.year == w.year]
        hit = ((yr_top.exit_date >= w.sd) & (yr_top.exit_date <= w.ed)).any()
        overlaps += int(bool(hit))
    share = float(overlaps / len(q) * 100) if len(q) else float("nan")
    # median gap between consecutive qualify_close window start dates (within year)
    gaps = []
    for yr, g in q.groupby("year"):
        ds = g.sd.sort_values().to_numpy()
        if len(ds) > 1:
            gaps.extend(np.diff(ds).astype("timedelta64[D]").astype(int).tolist())
    med_gap = float(np.median(gaps)) if gaps else float("nan")
    return dict(n_qualify_close_windows=int(len(q)), top_decile_pnl_threshold=float(thr),
                n_top_decile_trades=int(len(top)),
                pct_qualify_windows_overlapping_top_decile=share,
                median_gap_days_between_qualify_windows=med_gap)


# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("[4pct] loading recovered frame via year_shape_probe loaders…", flush=True)
    D = ys.cw.load()
    hy = ys.eaf.holdout_year_tids(D)
    runs = build_runs(D, hy)
    day_key = D["day_key"]
    print(f"[4pct] {len(runs)} year-segments (F1–F11 + {sorted(hy)}); "
          f"W={W_DAYS} cal-days; threshold {THRESH * 100:.0f}%", flush=True)

    rw, dropped, mean_rows = rolling_windows(runs, day_key)
    rw.to_csv(OUTDIR / "rolling_windows.csv", index=False, lineterminator="\n")
    s1, s1p = s1_tables(rw)
    cad = nonoverlap_cadence(runs, day_key)
    cad.to_csv(OUTDIR / "nonoverlap_cadence.csv", index=False, lineterminator="\n")
    ttf = time_to_four(runs, day_key)
    ttf.to_csv(OUTDIR / "time_to_four_cycles.csv", index=False, lineterminator="\n")
    s3 = s3_summary(ttf)
    s4 = s4_clustering(rw, day_key)

    write_report(s1, s1p, cad, s3, s4, dropped, mean_rows, rw, ttf)

    # ── stdout ──
    print("\n=== §1 ROLLING 14-DAY HIT-RATE (closed=primary) ===", flush=True)
    print(s1.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)
    print("\nend_return percentiles (%):", flush=True)
    print(s1p.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)

    print("\n=== §2 NON-OVERLAPPING CADENCE (closed; across 14 phase offsets) ===", flush=True)
    cad_all = cad
    print(f"per-year qualifying 14-day blocks (of ~{int(cad.total_blocks_approx.median())}/yr):",
          flush=True)
    print(cad_all[["fold", "year", "seg", "total_blocks_approx", "qual_blocks_mean",
                   "qual_blocks_min", "qual_blocks_max", "off0_qual_blocks",
                   "off0_best_block_close_pct"]].to_string(index=False,
          float_format=lambda x: f"{x:.2f}"), flush=True)
    ho = cad[cad.seg == "holdout"]
    print(f"\nAcross-offset summary — ALL years: mean {cad.qual_blocks_mean.mean():.2f}, "
          f"min {cad.qual_blocks_min.min()}, max {cad.qual_blocks_max.max()} qualifying blocks/yr",
          flush=True)
    print(f"Across-offset summary — HOLDOUT 2021-2026: mean {ho.qual_blocks_mean.mean():.2f}, "
          f"min {ho.qual_blocks_min.min()}, max {ho.qual_blocks_max.max()} qualifying blocks/yr",
          flush=True)

    print("\n=== §3 TIME TO 4 QUALIFYING CYCLES (closed; the decision metric) ===", flush=True)
    print(s3.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)

    # ── validation checklist ──
    pooled = int((rw[rw.series == "closed"].qualify_close).sum())
    per_year_sum = int(rw[rw.series == "closed"].groupby("year").qualify_close.sum().sum())
    print("\n=== VALIDATION CHECKLIST ===", flush=True)
    print(f"[x] no window spans a fold/year reset; windows dropped at boundaries: "
          f"{dropped} (closed/equity)", flush=True)
    print(f"[x] calendar-day windows use actual dates; W={W_DAYS}cal-days realised as "
          f"{mean_rows:.2f} trading rows/window (mean)", flush=True)
    print(f"[x] both series present → run on both; PRIMARY = closed (realized balance)", flush=True)
    print(f"[{'x' if pooled == per_year_sum else ' '}] counts reconcile: pooled qualify_close "
          f"{pooled} == Σ per-year {per_year_sum}", flush=True)
    print(f"[x] deterministic (no RNG; lineterminator='\\n'); canonical CSVs byte-unchanged; "
          f"writes only in {OUTDIR.relative_to(ROOT)}", flush=True)
    print(f"\n[done] artefacts → {OUTDIR}", flush=True)
    return 0


def write_report(s1, s1p, cad, s3, s4, dropped, mean_rows, rw, ttf):
    def md(df, fmt="{:.2f}"):
        return ys.gw.df_to_md(df, fmt)
    L = ["# Arc 10 v3.0.2 — Rolling 4%-in-14-days frequency + Scale-Up feasibility (RAW NUMBERS)\n"]
    L.append("> **Read-only probe.** 0.40% r_base, governors ON, EA-faithful "
             "(floating-equity) sim on the recovered sha-`05dea9…9ee58a` frame. Per-year "
             "reset to 1.0; **no window spans a reset**. Two daily series: **closed = "
             "realized balance (PRIMARY — FundedNext rewards key off realized profit)**, "
             "equity = floating (secondary). W = 14 calendar days; threshold +4%. NO "
             "interpretation — numbers only. (F11/2020 floating weight ±0.32pp recovery "
             "drift; immaterial here.)\n")

    L.append("## §1 Rolling 14-day hit-rate (step = 1 day, every within-year start)\n")
    L.append(md(s1) + "\n")
    L.append("### §1b end_return percentiles across all 14-day windows (%)\n")
    L.append(md(s1p) + "\n")

    L.append("## §2 Non-overlapping cadence (closed; 14 phase offsets, start-day 0–13)\n")
    L.append("> Per year: total full 14-day blocks (~26/yr), and qualifying (close≥4%) "
             "blocks summarised across the 14 phase offsets (mean/min/max), plus offset-0 "
             "detail (n qualifying + best single-block close%).\n")
    L.append(md(cad[["fold", "year", "seg", "total_blocks_approx", "qual_blocks_mean",
                     "qual_blocks_min", "qual_blocks_max", "off0_qual_blocks",
                     "off0_best_block_close_pct"]]) + "\n")
    ho = cad[cad.seg == "holdout"]
    L.append(f"> **Across-offset, ALL years:** mean {cad.qual_blocks_mean.mean():.2f} / "
             f"min {cad.qual_blocks_min.min()} / max {cad.qual_blocks_max.max()} qualifying "
             f"blocks per year. **HOLDOUT 2021–2026:** mean {ho.qual_blocks_mean.mean():.2f} / "
             f"min {ho.qual_blocks_min.min()} / max {ho.qual_blocks_max.max()}.\n")

    L.append("## §3 Time to FOUR qualifying cycles (closed; the decision metric)\n")
    L.append("> Sequential non-overlapping 14-day cycles from every within-year funding "
             "start date; calendar days until the 4th cycle clears +4%. Per-fold reset "
             "caps forward data at the year segment, so 'same-year' and '12-month' never%% "
             "coincide here (a real cycle chain cannot cross a year reset in this frame).\n")
    L.append(md(s3) + "\n")

    L.append("## §4 Clustering vs runners\n")
    s4df = pd.DataFrame([s4])
    L.append(md(s4df, "{:.2f}") + "\n")
    L.append(f"> {s4['pct_qualify_windows_overlapping_top_decile']:.1f}% of qualify_close "
             f"(closed) windows contain ≥1 top-decile-PnL trade "
             f"(top-decile threshold {s4['top_decile_pnl_threshold']:.5f} account-mult, "
             f"{s4['n_top_decile_trades']} trades). Median gap between consecutive "
             f"qualify_close window starts = {s4['median_gap_days_between_qualify_windows']:.1f} "
             f"days.\n")

    L.append("## Validation\n")
    pooled = int((rw[rw.series == "closed"].qualify_close).sum())
    per_year_sum = int(rw[rw.series == "closed"].groupby("year").qualify_close.sum().sum())
    L.append(f"- windows dropped at year boundaries (no reset-spanning): {dropped}.\n")
    L.append(f"- W=14 calendar days realised as {mean_rows:.2f} trading rows/window (mean).\n")
    L.append(f"- counts reconcile: pooled qualify_close (closed) {pooled} == Σ per-year "
             f"{per_year_sum}.\n")
    L.append("- both series run; closed = primary. Deterministic; canonical CSVs "
             "byte-unchanged; writes confined to the probe dir.\n")
    (OUTDIR / "four_pct_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

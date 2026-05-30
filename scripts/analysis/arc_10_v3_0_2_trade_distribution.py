"""Arc 10 v3.0.2 — trade distribution analytics (EET, 3.5R, read-only).

Full distribution of where the 3,152 deployed trades fall and how they perform,
sliced by pair / year / calendar-month / pair*year / day-of-week, plus a
frequency-clustering check and a concentration cut.

Descriptive, read-only re-aggregation of the locked EET v3.0.2 deployed-policy
frame (path_analytics/B_exit_mfe.csv; trade_paths sha 05dea9...). realized R =
deployed `sl_partial_close_1r_runner_trail @ 3.5xATR`, both legs, per trade,
3.5R frame. win = realized R > 0. NO config / exit change, NO re-sim, NO WFO.
v3.0.2 locked.

Calendar: EET-local (Europe/Athens, EET/EEST w/ DST) — the project's 5ers EET
convention (core/time_utils/session_boundary). entry_time (stored UTC) is
tz-converted to EET; year/month/day-of-week are taken EET-local. EET-year == the
existing SUMMARY fold-tagged year for all 3152 trades (0 divergence), so the
fold mapping (fold k -> 2010+k-1, holdout = fold 12, 2021-04..2026) is
preserved; day-of-week DOES shift for 452 late-Friday-UTC bars that fall on the
next EET trading day — the EET reading is the faithful one. fold/segment columns
are reused from B_exit_mfe.csv unchanged.

NOT a pruning signal: per-pair realized R is descriptive only. Selecting pairs
on realized R is outcome-selection against the ex-ante population rule; any
pair-set change is a separate ex-ante-justified + WFO question, never a backtest
filter. (Stated in SUMMARY.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ARC = ROOT / "results" / "l_arc_10_v3.0.2"
OUTDIR = ARC / "path_analytics"
EET_TZ = "Europe/Athens"
THIN_N = 30
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    is_int = {c: pd.api.types.is_integer_dtype(df[c]) for c in cols}
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for i in range(len(df)):
        cells = []
        for c in cols:
            v = df[c].iloc[i]
            if is_int[c]:
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _pf(r: np.ndarray) -> float:
    win = r[r > 0].sum()
    loss = r[r <= 0].sum()
    return float(win / abs(loss)) if loss != 0 else float("inf")


def _agg(r: np.ndarray, grand_n: int, grand_total: float) -> dict:
    r = np.asarray(r, dtype=float)
    n = r.size
    win = r[r > 0]
    los = r[r <= 0]
    return dict(
        n=int(n),
        n_win=int(win.size),
        win_rate=float(win.size / n) if n else np.nan,
        mean_r=float(r.mean()) if n else np.nan,
        total_r=float(r.sum()),
        mean_win_r=float(win.mean()) if win.size else np.nan,
        mean_loss_r=float(los.mean()) if los.size else np.nan,
        profit_factor=_pf(r),
        share_of_trades=float(n / grand_n) if grand_n else np.nan,
        share_of_total_r=float(r.sum() / grand_total) if grand_total else np.nan,
    )


def main() -> int:
    der = pd.read_csv(OUTDIR / "B_exit_mfe.csv")[
        ["trade_id", "pair", "fold", "segment", "realized_r_3p5"]
    ].copy()
    pool = pd.read_parquet(ARC / "step_1" / "pool.parquet")[["trade_id", "entry_time"]]
    der = der.merge(pool, on="trade_id", how="left")
    et = pd.to_datetime(der["entry_time"], utc=True).dt.tz_convert(EET_TZ)
    der["year"] = et.dt.year
    der["month"] = et.dt.month
    der["dow"] = et.dt.dayofweek

    R = der["realized_r_3p5"].to_numpy()
    GN = len(der)
    GT = float(R.sum())

    # ── Cut 1: per pair ─────────────────────────────────────────────────────
    pr_rows = []
    for pair, g in der.groupby("pair"):
        d = {"pair": pair}
        d.update(_agg(g["realized_r_3p5"].to_numpy(), GN, GT))
        pr_rows.append(d)
    by_pair = pd.DataFrame(pr_rows).sort_values("total_r", ascending=False).reset_index(drop=True)
    by_pair["rank_total_r"] = by_pair["total_r"].rank(ascending=False, method="min").astype(int)
    by_pair["rank_mean_r"] = by_pair["mean_r"].rank(ascending=False, method="min").astype(int)
    by_pair["rank_gap"] = by_pair["rank_total_r"] - by_pair["rank_mean_r"]
    by_pair.to_csv(OUTDIR / "by_pair.csv", index=False, lineterminator="\n")

    # ── Cut 2: per year (+ fold/holdout) ────────────────────────────────────
    yr_rows = []
    for y, g in der.groupby("year"):
        r = g["realized_r_3p5"].to_numpy()
        yr_rows.append(dict(year=int(y), n=int(r.size),
                            win_rate=float((r > 0).mean()), mean_r=float(r.mean()),
                            total_r=float(r.sum())))
    by_year = pd.DataFrame(yr_rows).sort_values("year").reset_index(drop=True)
    by_year.to_csv(OUTDIR / "by_year.csv", index=False, lineterminator="\n")
    seg_rows = []
    for s, g in der.groupby("segment"):
        r = g["realized_r_3p5"].to_numpy()
        seg_rows.append(dict(segment=s, n=int(r.size), win_rate=float((r > 0).mean()),
                             mean_r=float(r.mean()), total_r=float(r.sum())))
    seg_tbl = pd.DataFrame(seg_rows).sort_values("segment").reset_index(drop=True)

    # ── Cut 3: per calendar month (pooled) ──────────────────────────────────
    mo_rows = []
    exp_uniform = GN / 12.0
    for mo in range(1, 13):
        r = der[der.month == mo]["realized_r_3p5"].to_numpy()
        mo_rows.append(dict(month=mo, n=int(r.size),
                            win_rate=float((r > 0).mean()) if r.size else np.nan,
                            mean_r=float(r.mean()) if r.size else np.nan,
                            total_r=float(r.sum()),
                            expected_n_if_uniform=exp_uniform,
                            n_over_expected=float(r.size / exp_uniform)))
    by_month = pd.DataFrame(mo_rows)
    by_month.to_csv(OUTDIR / "by_month.csv", index=False, lineterminator="\n")

    # ── Cut 4: frequency / clustering ───────────────────────────────────────
    ym = der.groupby(["year", "month"]).size().rename("n").reset_index()
    full_idx = pd.period_range(
        pd.Period(year=int(der.year.min()), month=int(der[der.year == der.year.min()].month.min()), freq="M"),
        pd.Period(year=int(der.year.max()), month=int(der[der.year == der.year.max()].month.max()), freq="M"),
        freq="M",
    )
    ym_series = (
        ym.assign(p=[pd.Period(year=int(y), month=int(m), freq="M") for y, m in zip(ym.year, ym.month)])
        .set_index("p")["n"]
        .reindex(full_idx, fill_value=0)
    )
    counts = ym_series.to_numpy()
    cv = float(counts.std() / counts.mean()) if counts.mean() else np.nan
    max_med = float(counts.max() / np.median(counts)) if np.median(counts) else np.nan
    # month-of-year uniformity chi-square (12 cells, expected = n/12)
    moy = by_month["n"].to_numpy().astype(float)
    chi2, pval = stats.chisquare(moy, f_exp=np.full(12, moy.sum() / 12.0))
    clustered = (cv > 0.5) or (pval < 0.05)
    freq_tbl = pd.DataFrame([
        dict(metric="active months (span, incl. zeros)", value=f"{len(counts)}"),
        dict(metric="monthly count mean", value=f"{counts.mean():.4f}"),
        dict(metric="monthly count median", value=f"{np.median(counts):.4f}"),
        dict(metric="monthly count CV (std/mean)", value=f"{cv:.4f}"),
        dict(metric="max-month / median-month", value=f"{max_med:.4f}"),
        dict(metric="month-of-year chi-square stat (dof=11)", value=f"{chi2:.4f}"),
        dict(metric="month-of-year chi-square p-value", value=f"{pval:.4g}"),
    ])

    # ── Cut 5: pair x year matrices ─────────────────────────────────────────
    cnt_mat = der.pivot_table(index="pair", columns="year", values="trade_id",
                              aggfunc="count", fill_value=0).astype(int)
    tr_mat = der.pivot_table(index="pair", columns="year", values="realized_r_3p5",
                             aggfunc="sum", fill_value=0.0)
    cnt_mat.to_csv(OUTDIR / "pair_year_count.csv", lineterminator="\n")
    tr_mat.to_csv(OUTDIR / "pair_year_total_r.csv", lineterminator="\n")

    # ── Cut 6: day-of-week ──────────────────────────────────────────────────
    dow_rows = []
    for d in range(7):
        r = der[der.dow == d]["realized_r_3p5"].to_numpy()
        dow_rows.append(dict(dow=d, weekday=DOW_NAMES[d], n=int(r.size),
                             win_rate=float((r > 0).mean()) if r.size else np.nan,
                             mean_r=float(r.mean()) if r.size else np.nan,
                             total_r=float(r.sum())))
    by_dow = pd.DataFrame(dow_rows)
    by_dow.to_csv(OUTDIR / "by_dow.csv", index=False, lineterminator="\n")

    # ── Cut 7: concentration ────────────────────────────────────────────────
    top5 = by_pair.head(5)
    bot5 = by_pair.tail(5).sort_values("total_r")
    top5_share = float(top5["total_r"].sum() / GT)
    neg = by_pair[by_pair["total_r"] < 0][["pair", "n", "total_r"]].sort_values("total_r")

    write_summary(by_pair, by_year, seg_tbl, by_month, freq_tbl, clustered, cv, pval,
                  cnt_mat, tr_mat, by_dow, top5, bot5, top5_share, neg, GN, GT)

    print("[by_pair top/bottom]\n", by_pair[["pair", "n", "win_rate", "mean_r", "total_r",
          "profit_factor", "rank_gap"]].head(5).to_string(index=False))
    print(" ...\n", by_pair[["pair", "n", "win_rate", "mean_r", "total_r",
          "profit_factor", "rank_gap"]].tail(5).to_string(index=False))
    print(f"[freq] CV={cv:.4f} chi2_p={pval:.4g} clustered={clustered}")
    print(f"[concentration] top5_share={top5_share:.4f} neg_pairs={len(neg)}")
    print(f"[done] wrote 6 CSVs + appended SUMMARY.md")
    return 0


def write_summary(by_pair, by_year, seg_tbl, by_month, freq_tbl, clustered, cv, pval,
                  cnt_mat, tr_mat, by_dow, top5, bot5, top5_share, neg, GN, GT):
    L = ["\n\n---\n\n## Trade distribution analytics (3.5R)\n"]
    L.append(
        "> Descriptive, read-only re-aggregation of the deployed-policy frame "
        "(realized R = `sl_partial_close_1r_runner_trail @ 3.5xATR`, both legs, "
        "3.5R frame; win = R>0). Calendar is EET-local (Europe/Athens, the 5ers "
        "EET convention); EET-year matches the fold-tagged year for all "
        f"{GN} trades, day-of-week shifts for 452 late-Friday-UTC bars onto the "
        "next EET trading day. **NOT a pruning signal:** per-pair realized R is "
        "descriptive only — selecting pairs on realized R is outcome-selection "
        "against the ex-ante population rule; any pair-set change is a separate "
        "ex-ante-justified + WFO question, never a backtest filter. No exit "
        "change, no WFO; v3.0.2 locked.\n"
    )

    # headline
    busiest = by_month.loc[by_month.n.idxmax()]
    sparsest = by_month.loc[by_month.n.idxmin()]
    L.append("### Headline\n")
    L.append(
        f"- **Best pairs (total R):** {', '.join(f'{r.pair} ({r.total_r:.1f}R, n={int(r.n)})' for r in top5.itertuples())}.\n"
        f"- **Worst pairs (total R):** {', '.join(f'{r.pair} ({r.total_r:.1f}R, n={int(r.n)})' for r in bot5.itertuples())}.\n"
        f"- **Top-5 pairs = {top5_share:.1%} of all realized R** (of {GT:.1f}R total); "
        f"{len(neg)} pair(s) net-negative.\n"
        f"- **Busiest month:** {int(busiest.month)} (n={int(busiest.n)}, "
        f"{busiest.n_over_expected:.2f}x uniform); **sparsest:** {int(sparsest.month)} "
        f"(n={int(sparsest.n)}, {sparsest.n_over_expected:.2f}x).\n"
        f"- **Frequency verdict:** monthly-count CV={cv:.2f}, month-of-year "
        f"chi-square p={pval:.3g} -> **{'CLUSTERED / bursty' if clustered else '~random-uniform'}**.\n"
    )

    L.append("### 1. Per pair (sorted by total R)\n")
    L.append("> Flag: pairs with n<30 are directional-only. `rank_gap = rank_total_r - "
             "rank_mean_r`: large negative = high-volume-low-edge; large positive = "
             "low-volume-high-edge.\n")
    cols1 = ["pair", "n", "n_win", "win_rate", "mean_r", "total_r", "mean_win_r",
             "mean_loss_r", "profit_factor", "share_of_trades", "share_of_total_r",
             "rank_total_r", "rank_mean_r", "rank_gap"]
    L.append(df_to_md(by_pair[cols1]) + "\n")
    thin_p = by_pair[by_pair.n < THIN_N]
    if len(thin_p):
        L.append("> **n<30 (directional-only):** "
                 + ", ".join(f"{r.pair}(n={int(r.n)})" for r in thin_p.itertuples()) + "\n")
    disagree = by_pair[by_pair.rank_gap.abs() >= 7].sort_values("rank_gap")
    if len(disagree):
        L.append("> **Rank disagreement (|gap|>=7):** "
                 + ", ".join(f"{r.pair}(total#{int(r.rank_total_r)}/mean#{int(r.rank_mean_r)})"
                             for r in disagree.itertuples()) + "\n")

    L.append("### 2. Per year\n")
    L.append(df_to_md(by_year) + "\n")
    L.append("Fold vs holdout:\n")
    L.append(df_to_md(seg_tbl) + "\n")
    thin_y = by_year[by_year.n < THIN_N]
    if len(thin_y):
        L.append("> **n<30 (directional-only):** "
                 + ", ".join(f"{int(r.year)}(n={int(r.n)})" for r in thin_y.itertuples()) + "\n")

    L.append("### 3. Per calendar month (pooled across years)\n")
    L.append(df_to_md(by_month) + "\n")

    L.append("### 4. Frequency / clustering\n")
    L.append(df_to_md(freq_tbl) + "\n")
    L.append(
        f"> Verdict: **{'CLUSTERED / bursty' if clustered else '~random-uniform'}** "
        f"(CV={cv:.2f}; month-of-year chi-square p={pval:.3g}). "
        + ("CV>0.5 and/or p<0.05 indicate non-uniform monthly arrival."
           if clustered else "Monthly arrival is close to uniform.") + "\n"
    )

    L.append("### 5. Pair x year — DIRECTIONAL ONLY (most cells n<30)\n")
    L.append("Trade count matrix:\n")
    cnt_out = cnt_mat.reset_index()
    cnt_out.columns = [str(c) for c in cnt_out.columns]
    L.append(df_to_md(cnt_out, "{:.0f}") + "\n")
    L.append("Total R matrix:\n")
    tr_out = tr_mat.reset_index()
    tr_out.columns = [str(c) for c in tr_out.columns]
    L.append(df_to_md(tr_out, "{:.1f}") + "\n")

    L.append("### 6. Day-of-week of entry (EET trading day)\n")
    L.append(df_to_md(by_dow) + "\n")
    thin_d = by_dow[by_dow.n < THIN_N]
    if len(thin_d):
        L.append("> **n<30 (directional-only):** "
                 + ", ".join(f"{r.weekday}(n={int(r.n)})" for r in thin_d.itertuples())
                 + " — these are late-Friday-UTC 4H bars rolled onto the next EET "
                 "trading day.\n")

    L.append("### 7. Concentration\n")
    L.append(f"Top-5 by total R (={top5_share:.1%} of all realized R):\n")
    L.append(df_to_md(top5[["pair", "n", "win_rate", "mean_r", "total_r", "profit_factor"]]) + "\n")
    L.append("Bottom-5 by total R:\n")
    L.append(df_to_md(bot5[["pair", "n", "win_rate", "mean_r", "total_r", "profit_factor"]]) + "\n")
    if len(neg):
        L.append("Net-negative pairs:\n")
        L.append(df_to_md(neg) + "\n")
    else:
        L.append("> No pair has net-negative total R.\n")

    with open(OUTDIR / "SUMMARY.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())

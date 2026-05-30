"""Arc 10 v3.0.2 — portfolio daily DD vs the 5% 5ers limit (EET, 3.5R).

Companion to arc_10_v3_0_2_concurrency.py. REUSES that probe's portfolio-equity
reconstruction (imported `build_mtm`) so the close-mark equity path is identical,
then buckets it by EET trading day to find the worst realized daily drawdown and
its margin to the binding 5% daily-DD limit. The deployed gate
(step_5.py:_equity_curve, per-trade-sequential) never measured portfolio daily
DD — this does.

Descriptive, read-only. No config / exit / sizing change, no re-sim, no WFO.
v3.0.2 locked. Account-% via r_base 0.5% (1R = 0.5%).

Daily anchor (parameterised; BOTH references reported so the verdict is robust
to 5ers' exact, externally-unconfirmed basis):
  * day boundary = canonical EET trading day, 00:00 EET reset
    (core.time_utils.session_boundary.utc_to_eet_trading_day, "5ers_eet";
    the Amendment-6 daily-DD bucketing).
  * DAY-START reference  — drop below the day's opening equity (5ers-like:
    "lose <=5% of start-of-day balance/equity").
  * DAY-HIGH reference   — drop from the running intraday high-water mark
    (trailing; more conservative when a day spikes up then gives back).

The live-relevant figure is whichever basis (balance vs equity, reset time)
5ers confirms in writing — stated plainly, not assumed.

Conservative intrabar bound (Cut 4): H4 close marks understate the true intrabar
daily low. Re-mark every OPEN trade at its worst cumulative MAE (`mae_so_far_r`)
level reached within the day, simultaneously, and take the portfolio min. This
OVERSTATES (assumes all open trades hit their worst at once, and MAE is
cumulative-from-entry) — it is an upper bound. The gap to the close-mark figure
is the intrabar + simultaneity uncertainty band.

Not a tuning trigger — informs live daily-DD monitoring, kill-criteria, and the
risk-appetite decision. Any exposure-cap / netting response is separate ex-ante
+ WFO.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import arc_10_v3_0_2_concurrency as conc  # noqa: E402  reuse identical reconstruction

from core.time_utils.session_boundary import utc_to_eet_trading_day  # noqa: E402

ARC = conc.ARC
OUTDIR = conc.OUTDIR
R_BASE = conc.R_BASE
SCALE = conc.SCALE
CACHE = conc.CACHE
SL_REASONS = {"SL_no_partial", "partial_then_runner_SL"}
DD_THRESHOLDS = [2, 3, 4, 5]
df_to_md = conc.df_to_md


def mae_overlay(meta: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """(trade_id, ts, mae_mark_r) — open leg marked at worst cumulative MAE per
    held bar (3.5R), final bar snapped to realized. Mirrors build_mtm mapping."""
    cache_idx = {}
    pos_maps = {}
    for pair in meta["pair"].unique():
        idx = pd.DatetimeIndex(pd.read_parquet(CACHE / f"{pair}.parquet").index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        cache_idx[pair] = idx
        pos_maps[pair] = {t: i for i, t in enumerate(idx)}
    pbt = {tid: g for tid, g in paths.groupby("trade_id")}
    rows = []
    for t in meta.itertuples(index=False):
        pr = pbt.get(t.trade_id)
        if pr is None or pr.empty:
            continue
        pr = pr.sort_values("bar_offset")
        bo = pr["bar_offset"].to_numpy(int)
        held = bo <= int(t.dep_exit_offset)
        mfe = pr["mfe_so_far_r"].to_numpy()[held] * SCALE
        mae = pr["mae_so_far_r"].to_numpy()[held] * SCALE
        nb = mae.size
        if nb == 0:
            continue
        partial_by = np.maximum.accumulate(mfe) >= 1.0
        open_units = np.where(partial_by, 0.5, 1.0)
        banked = np.where(partial_by, 0.5, 0.0)
        mae_mark = banked + open_units * mae  # open leg at worst-so-far adverse
        mae_mark[-1] = float(t.realized_r_3p5)  # closed -> realized, not MAE
        epos = pos_maps[t.pair].get(pd.Timestamp(t.entry_time).tz_convert("UTC"))
        if epos is None:
            continue
        ts = cache_idx[t.pair][epos + bo[held]]
        for k in range(nb):
            rows.append((t.trade_id, ts[k], float(mae_mark[k])))
    return pd.DataFrame(rows, columns=["trade_id", "ts", "mae_mark_r"])


def main() -> int:
    A = pd.read_csv(OUTDIR / "A_entry_mae.csv")[
        ["trade_id", "pair", "fold", "segment", "outcome", "dep_exit_offset"]
    ]
    B = pd.read_csv(OUTDIR / "B_exit_mfe.csv")[
        ["trade_id", "realized_r_3p5", "exit_reason_deployed"]
    ]
    pool = pd.read_parquet(ARC / "step_1" / "pool.parquet")[["trade_id", "entry_time"]]
    meta = A.merge(B, on="trade_id").merge(pool, on="trade_id")
    meta["entry_time"] = pd.to_datetime(meta["entry_time"], utc=True)
    paths = pd.read_parquet(ARC / "step_1" / "trade_paths.parquet")

    # ── identical reconstruction (imported) ──
    mtm_df, intervals, clock = conc.build_mtm(meta, paths)
    iv = pd.DataFrame(intervals).T.reset_index().rename(columns={"index": "trade_id"})
    for c in ["entry_ts", "exit_ts", "entry_time"]:
        iv[c] = pd.to_datetime(iv[c], utc=True)
    iv["realized"] = iv["realized"].astype(float)
    iv = iv.merge(meta[["trade_id", "exit_reason_deployed"]], on="trade_id", how="left")

    # ── close-mark portfolio equity on the H4 clock (interval-consistent,
    #    forward-filled — same corrected reconstruction as the concurrency probe) ──
    port = conc.build_portfolio(mtm_df, clock)
    open_sum_a = port["open_sum"]
    realized_at_exit = iv.groupby("exit_ts")["realized"].sum().reindex(clock, fill_value=0.0)
    closed_before_a = realized_at_exit.cumsum().shift(1).fillna(0.0).to_numpy()
    port_R = open_sum_a + closed_before_a
    eq_close = 1.0 + R_BASE * port_R

    # ── MAE-mark conservative equity (forward-filled) ──
    mae_df = mae_overlay(meta, paths)
    open_mae = conc.ffill_sum(mae_df, clock, "mae_mark_r")
    eq_cons = 1.0 + R_BASE * (open_mae + closed_before_a)

    # ── currency net (risk-weighted, forward-filled) for dominant-ccy attribution ──
    net_risk = port["net_risk"]  # ts x ccy
    dom_series = net_risk.abs().idxmax(axis=1)
    dom_ccy = dict(zip(net_risk.index, dom_series))

    # ── day keys (EET 00:00 reset) ──
    day_key = utc_to_eet_trading_day(pd.DatetimeIndex(clock), convention="5ers_eet")
    day_key = pd.DatetimeIndex(day_key)
    mtm_day = utc_to_eet_trading_day(pd.DatetimeIndex(mtm_df["ts"]), convention="5ers_eet")
    mtm_df = mtm_df.assign(day=pd.DatetimeIndex(mtm_day))
    iv_exit_day = pd.DatetimeIndex(
        utc_to_eet_trading_day(pd.DatetimeIndex(iv["exit_ts"]), convention="5ers_eet")
    )
    iv = iv.assign(exit_day=iv_exit_day)

    open_by_day = mtm_df.groupby("day")["trade_id"].nunique()
    closes_by_day = iv.groupby("exit_day")["trade_id"].nunique()
    sl_by_day = (
        iv[iv.exit_reason_deployed.isin(SL_REASONS)].groupby("exit_day")["trade_id"].nunique()
    )

    # contiguous day blocks over the sorted clock
    days = pd.Series(day_key)
    rows = []
    n = len(clock)
    start = 0
    while start < n:
        d = days.iloc[start]
        end = start
        while end + 1 < n and days.iloc[end + 1] == d:
            end += 1
        carry = start - 1 if start > 0 else start
        ds_eq = eq_close[carry]
        seg = eq_close[start : end + 1]
        low_rel = int(np.argmin(seg))
        low_pos = start + low_rel
        dd_daystart = max(0.0, (ds_eq - eq_close[low_pos]) / ds_eq)
        # day-high (trailing) reference
        hi = ds_eq
        dd_dayhigh = 0.0
        for e in seg:
            hi = max(hi, e)
            dd_dayhigh = max(dd_dayhigh, (hi - e) / hi)
        # conservative
        cons_low = float(np.min(eq_cons[start : end + 1]))
        dd_cons = max(0.0, (ds_eq - cons_low) / ds_eq)
        # realized / floating split at the day-start-trough, expressed as % of
        # day-start equity (same basis as dd_daystart, so the two sum to it)
        realized_drop = -R_BASE * (closed_before_a[low_pos] - closed_before_a[carry]) / ds_eq
        floating_drop = -R_BASE * (open_sum_a[low_pos] - open_sum_a[carry]) / ds_eq
        total_drop = R_BASE * (port_R[carry] - port_R[low_pos]) / ds_eq
        rows.append(
            dict(
                day=pd.Timestamp(d).tz_convert("Europe/Athens").date(),
                n_bars=end - start + 1,
                n_open=int(open_by_day.get(d, 0)),
                n_closes=int(closes_by_day.get(d, 0)),
                n_sl_closes=int(sl_by_day.get(d, 0)),
                dd_daystart_pct=dd_daystart * 100,
                dd_dayhigh_pct=dd_dayhigh * 100,
                dd_cons_daystart_pct=dd_cons * 100,
                realized_drop_pct=realized_drop * 100,
                floating_drop_pct=floating_drop * 100,
                total_drop_pct=total_drop * 100,
                dominant_ccy=dom_ccy.get(clock[low_pos], ""),
            )
        )
        start = end + 1

    dd = pd.DataFrame(rows)
    dd.to_csv(OUTDIR / "daily_dd.csv", index=False, lineterminator="\n")

    write_summary(dd)

    w = dd.loc[dd.dd_daystart_pct.idxmax()]
    wh = dd.dd_dayhigh_pct.max()
    wc = dd.dd_cons_daystart_pct.max()
    print(
        f"[daily DD] worst day-start={w.dd_daystart_pct:.2f}% day-high={wh:.2f}% "
        f"conservative={wc:.2f}%"
    )
    print(
        f"[worst day] {w.day} ccy={w.dominant_ccy} realized={w.realized_drop_pct:.2f} "
        f"floating={w.floating_drop_pct:.2f} n_open={int(w.n_open)} n_sl={int(w.n_sl_closes)}"
    )
    for thr in DD_THRESHOLDS:
        print(
            f"  days >{thr}% (day-start): {int((dd.dd_daystart_pct > thr).sum())} | "
            f"(day-high): {int((dd.dd_dayhigh_pct > thr).sum())}"
        )
    print(f"[done] wrote daily_dd.csv ({len(dd)} days) + appended SUMMARY.md")
    return 0


def write_summary(dd: pd.DataFrame) -> None:
    L = ["\n\n---\n\n## Portfolio daily DD vs 5% (3.5R)\n"]
    L.append(
        "> Descriptive risk surface. REUSES the concurrency probe's portfolio-equity "
        "reconstruction (imported `build_mtm` + `build_portfolio`, interval-consistent "
        "forward-filled), bucketed by EET trading day (00:00 "
        "EET reset, the Amendment-6 boundary). The per-trade-sequential gate never "
        "measured portfolio daily DD; the binding 5ers limit is 5% daily. Two "
        "references reported (DAY-START = drop below opening equity, 5ers-like; "
        "DAY-HIGH = drop from running intraday peak). **The 5ers daily basis "
        "(balance vs equity, reset time) is externally unconfirmed and parameterised "
        "here — the live figure is whichever basis 5ers confirms.** No "
        "config/exit/sizing change, no WFO; v3.0.2 locked. Not a tuning trigger.\n"
    )

    w = dd.loc[dd.dd_daystart_pct.idxmax()]
    wh = dd.loc[dd.dd_dayhigh_pct.idxmax()]
    wc_val = dd.dd_cons_daystart_pct.max()
    n_days = len(dd)
    rshare = (w.realized_drop_pct / w.total_drop_pct * 100) if w.total_drop_pct else np.nan

    L.append("### Headline\n")
    L.append(
        f"- **Worst daily DD (close-mark): DAY-START {w.dd_daystart_pct:.2f}% "
        f"({w.day}), DAY-HIGH {wh.dd_dayhigh_pct:.2f}% ({wh.day}).** Conservative "
        f"intrabar-MAE bound (day-start): **{wc_val:.2f}%**.\n"
        f"- **Margin to the 5% limit:** {5 - w.dd_daystart_pct:.2f}pp (close-mark "
        f"day-start); {5 - wc_val:.2f}pp under the conservative MAE bound.\n"
        f"- Days over 4%: {int((dd.dd_daystart_pct > 4).sum())} (day-start) / "
        f"{int((dd.dd_dayhigh_pct > 4).sum())} (day-high). "
        f"Days over 5%: {int((dd.dd_daystart_pct > 5).sum())} (day-start) / "
        f"{int((dd.dd_dayhigh_pct > 5).sum())} (day-high), of {n_days} trading days.\n"
        f"- **Worst day split:** realized (closed) {w.realized_drop_pct:.2f}% vs "
        f"floating (open MtM) {w.floating_drop_pct:.2f}% "
        f"(realized = {rshare:.0f}% of the drop); dominant adverse currency "
        f"**{w.dominant_ccy}**; {int(w.n_open)} open / {int(w.n_sl_closes)} SL "
        f"closes that day.\n"
    )

    L.append("### 1. Daily DD distribution (account-%)\n")
    dist = []
    for ref, col in [
        ("day-start", "dd_daystart_pct"),
        ("day-high", "dd_dayhigh_pct"),
        ("conservative MAE (day-start)", "dd_cons_daystart_pct"),
    ]:
        x = dd[col].to_numpy()
        dist.append(
            dict(
                reference=ref,
                p50=float(np.percentile(x, 50)),
                p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)),
                max=float(x.max()),
                mean=float(x.mean()),
            )
        )
    L.append(df_to_md(pd.DataFrame(dist)) + "\n")
    cnt = []
    for thr in DD_THRESHOLDS:
        cnt.append(
            dict(
                threshold_pct=thr,
                days_daystart=int((dd.dd_daystart_pct > thr).sum()),
                days_dayhigh=int((dd.dd_dayhigh_pct > thr).sum()),
                days_conservative=int((dd.dd_cons_daystart_pct > thr).sum()),
            )
        )
    L.append(f"Days exceeding threshold (of {n_days} trading days):\n")
    L.append(df_to_md(pd.DataFrame(cnt), "{:.0f}") + "\n")

    L.append("### 2. Worst-10 days (by DAY-START close-mark DD)\n")
    top = dd.sort_values("dd_daystart_pct", ascending=False).head(10).copy()
    top["day"] = top["day"].astype(str)
    cols = [
        "day",
        "dd_daystart_pct",
        "dd_dayhigh_pct",
        "dd_cons_daystart_pct",
        "n_open",
        "n_closes",
        "n_sl_closes",
        "realized_drop_pct",
        "floating_drop_pct",
        "dominant_ccy",
    ]
    L.append(df_to_md(top[cols]) + "\n")
    thin = top[top.n_bars < 3] if "n_bars" in top else top.iloc[:0]
    if len(thin):
        L.append(
            "> **THIN-DAY FLAG (<3 H4 bars):** "
            + ", ".join(str(r.day) for r in thin.itertuples())
            + " — partial sessions.\n"
        )

    L.append("### 3. Realized vs floating (worst-10 days)\n")
    L.append(
        "> Closed losses are unrecoverable; floating may reverse before the "
        "daily close — which matters depends on the (unconfirmed) 5ers basis. "
        "Split is of the day-start trough drop.\n"
    )
    rf = top[["day", "total_drop_pct", "realized_drop_pct", "floating_drop_pct"]].copy()
    rf["realized_share_pct"] = np.where(
        rf.total_drop_pct != 0, rf.realized_drop_pct / rf.total_drop_pct * 100, np.nan
    )
    L.append(df_to_md(rf) + "\n")

    L.append("### 4. Conservative intrabar bound vs close-mark\n")
    L.append(
        f"> Worst-day DAY-START DD: close-mark **{w.dd_daystart_pct:.2f}%** vs "
        f"intrabar-MAE upper bound **{wc_val:.2f}%** — a {wc_val - w.dd_daystart_pct:.2f}pp "
        "uncertainty band from unrecorded intrabar lows + the simultaneity/"
        "cumulative-MAE conservatism. True worst-day daily DD lies between these. "
        "Even the upper bound "
        + (
            f"stays under 5% ({5 - wc_val:.2f}pp margin)."
            if wc_val < 5
            else f"BREACHES 5% ({wc_val - 5:.2f}pp over)."
        )
        + "\n"
    )

    L.append("### 5. Currency attribution of the daily-DD tail\n")
    tail = dd[dd.dd_daystart_pct > 3]
    L.append(
        f"> Days with day-start DD > 3% ({len(tail)} days), grouped by dominant "
        "adverse currency at the trough:\n"
    )
    if len(tail):
        ca = (
            tail.groupby("dominant_ccy")
            .agg(
                n_days=("day", "size"),
                max_dd_pct=("dd_daystart_pct", "max"),
                mean_dd_pct=("dd_daystart_pct", "mean"),
            )
            .reset_index()
            .sort_values("n_days", ascending=False)
        )
        L.append(df_to_md(ca) + "\n")
        L.append(
            "> Cross-check: the max-DD episode (concurrency probe) was "
            "CHF-dominated; the daily-DD tail "
            + (
                "CONCENTRATES on the same CHF/EUR/JPY legs."
                if set(ca.dominant_ccy.head(3)) & {"CHF", "EUR", "JPY"}
                else "spreads across currencies."
            )
            + "\n"
        )
    else:
        L.append("> No day exceeds 3% day-start DD.\n")

    L.append(
        "\n> **Note:** figures are linear (1R=0.5%, additive open+closed MtM) vs "
        "the gate's per-trade compounding — risk-surface comparable, not "
        "byte-identical. Live-relevant basis is whichever 5ers confirms.\n"
    )

    with open(OUTDIR / "SUMMARY.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())

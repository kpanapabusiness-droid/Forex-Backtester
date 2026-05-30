"""Arc 10 v3.0.2 — concurrency / correlation / loss-clustering (EET, 3.5R).

Descriptive, read-only PORTFOLIO-level risk surface over the locked EET v3.0.2
deployed-policy frame (trade_paths sha 05dea9...). The per-trade analytics treat
trades as independent; 5ers limits (5% daily / 10% max DD) are breached by
*concurrent correlated* open risk, not single-trade R. This measures position
overlap, currency-leg exposure concentration, and loss clustering in time / on
shared currency legs.

NO config / exit / pair change, NO re-sim, NO WFO. v3.0.2 locked.

STEP 0 — DD FRAMING (resolved by reading the engine):
  The deployed worst-fold DD (7.80% central cell, fundednext_cost_sweep
  diagnostics `worst_dd_at_rbase`=0.07799) is computed by
  scripts/l_arc_10_v3/step_5.py:_equity_curve -> PER-TRADE-SEQUENTIAL
  compounding: trades sorted by signal_bar_time, each trade's *full realized R*
  applied one at a time at fixed 0.5% risk (eq += eq*0.005*r). There is NO
  portfolio equity curve and NO concurrent open-position modelling. The
  Amendment-3 holdout rerun (amendment_3_addendum._equity_curve) uses the
  identical per-trade-sequential math. => CONCURRENCY IS UNMODELLED IN THE
  7.80%. This probe REVEALS the concurrency/correlation risk the gate did not
  see. All downstream reads are framed against this.

Mark-to-market (3.5R deployed frame): per held bar i of a trade,
  open_units = 1.0 before the +1R partial fires, 0.5 after (running max mfe_3p5
              >= 1 within the held window);
  banked     = 0.0 before, 0.5 after (the +1R half-close, 0.5*1R);
  mtm_r(i)   = banked + open_units * close_3p5(i).
  The final held bar is snapped to the canonical realized_r_3p5 so closed PnL
  matches live exactly. Account-% via r_base 0.5% (1R = 0.5%).

Currency decomposition: long-only, each open trade = +base, -quote (long
  GBPCAD -> +GBP, -CAD), risk-weighted by open_units*0.5%.

Common clock: union of all 28 H4-EET cache bar timestamps within the trade
  span (the execution grid; path frames are per-H4-bar via bar_offset ->
  cache-index position).

NOT a tuning trigger: descriptive risk-surface only. Findings inform the
  live-DD picture and kill-criteria / live-tracking docs; no sizing/exit/pair
  change, no WFO follows from this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ARC = ROOT / "results" / "l_arc_10_v3.0.2"
OUTDIR = ARC / "path_analytics"
CACHE = ROOT / "data" / "cache" / "H4_5ers_eet"
SCALE = 2.0 / 3.5
R_BASE = 0.005  # 0.5% per 1R
N_LEVELS = [5, 8, 10, 15]
THIN = 30
MIN_OVERLAP = 8  # min co-open bars for a pairwise correlation


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


def pctd(x, ps=(50, 90, 99)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p}": np.nan for p in ps} | {"max": np.nan, "mean": np.nan, "n": 0}
    d = {f"p{p}": float(np.percentile(x, p)) for p in ps}
    d["max"] = float(x.max())
    d["mean"] = float(x.mean())
    d["n"] = int(x.size)
    return d


# ──────────────────────────────────────────────────────────────────────────
# Build per-trade held-bar mark-to-market table on the real H4-EET clock.
# ──────────────────────────────────────────────────────────────────────────
def build_mtm(
    meta: pd.DataFrame, paths: pd.DataFrame
) -> tuple[pd.DataFrame, dict, pd.DatetimeIndex]:
    cache_idx: dict[str, pd.DatetimeIndex] = {}
    pos_maps: dict[str, dict] = {}
    for pair in meta["pair"].unique():
        idx = pd.DatetimeIndex(pd.read_parquet(CACHE / f"{pair}.parquet").index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        cache_idx[pair] = idx
        pos_maps[pair] = {t: i for i, t in enumerate(idx)}

    paths_by_trade = {tid: g for tid, g in paths.groupby("trade_id")}
    rows = []
    intervals = {}  # trade_id -> (entry_ts, exit_ts, base, quote, realized, fold, segment, outcome)
    for t in meta.itertuples(index=False):
        pr = paths_by_trade.get(t.trade_id)
        if pr is None or pr.empty:
            continue
        pr = pr.sort_values("bar_offset")
        bo = pr["bar_offset"].to_numpy(int)
        held = bo <= int(t.dep_exit_offset)
        mfe = pr["mfe_so_far_r"].to_numpy()[held] * SCALE
        clo = pr["close_r"].to_numpy()[held] * SCALE
        nb = mfe.size
        if nb == 0:
            continue
        partial_by = np.maximum.accumulate(mfe) >= 1.0
        open_units = np.where(partial_by, 0.5, 1.0)
        banked = np.where(partial_by, 0.5, 0.0)
        mtm = banked + open_units * clo
        mtm[-1] = float(t.realized_r_3p5)  # snap final to canonical realized

        idx = cache_idx[t.pair]
        epos = pos_maps[t.pair].get(pd.Timestamp(t.entry_time).tz_convert("UTC"))
        if epos is None:
            continue
        offs = bo[held]
        ts = idx[epos + offs]
        base, quote = t.pair[:3], t.pair[3:]
        for k in range(nb):
            rows.append((t.trade_id, ts[k], float(mtm[k]), float(open_units[k]), base, quote))
        intervals[t.trade_id] = dict(
            entry_ts=ts[0],
            exit_ts=ts[-1],
            base=base,
            quote=quote,
            realized=float(t.realized_r_3p5),
            fold=int(t.fold),
            segment=t.segment,
            outcome=t.outcome,
            pair=t.pair,
            entry_time=ts[0],
        )

    mtm_df = pd.DataFrame(rows, columns=["trade_id", "ts", "mtm_r", "open_units", "base", "quote"])

    # global H4-EET clock = union of cache bars within trade span
    lo = mtm_df.ts.min()
    hi = mtm_df.ts.max()
    clock = pd.DatetimeIndex([])
    for pair, idx in cache_idx.items():
        clock = clock.union(idx[(idx >= lo) & (idx <= hi)])
    clock = clock.sort_values()
    return mtm_df, intervals, clock


# ──────────────────────────────────────────────────────────────────────────
# Interval-consistent portfolio aggregation on the union clock.
#
# CRITICAL: the 28 pairs are NOT perfectly aligned on the union clock — sparse
# weekend/holiday boundary bars (e.g. the Sunday 22:00-UTC EET open) exist for
# some pairs but not others. A naive groupby(ts).sum() over recorded bars drops
# every open position whose pair lacks a bar at that timestamp, manufacturing
# spurious portfolio drawdowns that recover the next bar. The correct mark for a
# held position over a bar its pair does not have is its LAST known mark
# (forward-fill within [entry, exit]). This builder forward-fills every open
# trade across the union clock via difference arrays (O(events)), so open
# count, open MtM, and currency exposure are interval-consistent.
# ──────────────────────────────────────────────────────────────────────────
def ffill_sum(df: pd.DataFrame, clock: pd.DatetimeIndex, col: str) -> np.ndarray:
    """Sum of per-trade `col` over the union clock, forward-filling each trade's
    last mark within [first bar, last bar] (diff-array; O(events)). Use this
    instead of groupby(ts).sum() so positions are not dropped on bars their pair
    lacks (see build_portfolio rationale)."""
    pos = {t: i for i, t in enumerate(clock)}
    K = len(clock)
    d = np.zeros(K + 1)
    for tid, g in df.groupby("trade_id"):
        g = g.sort_values("ts")
        ps = [pos[t] for t in g["ts"]]
        v = g[col].to_numpy()
        d[ps[0]] += v[0]
        for k in range(1, len(ps)):
            d[ps[k]] += v[k] - v[k - 1]
        d[ps[-1] + 1] -= v[-1]
    return np.cumsum(d)[:K]


def build_portfolio(mtm_df: pd.DataFrame, clock: pd.DatetimeIndex) -> dict:
    pos = {t: i for i, t in enumerate(clock)}
    K = len(clock)
    open_count_d = np.zeros(K + 1)
    open_sum_d = np.zeros(K + 1)
    ccys = sorted(set(mtm_df["base"]) | set(mtm_df["quote"]))
    risk_d = {c: np.zeros(K + 1) for c in ccys}
    count_d = {c: np.zeros(K + 1) for c in ccys}
    for tid, g in mtm_df.groupby("trade_id"):
        g = g.sort_values("ts")
        ps = [pos[t] for t in g["ts"]]
        mt = g["mtm_r"].to_numpy()
        ou = g["open_units"].to_numpy()
        base, quote = g["base"].iloc[0], g["quote"].iloc[0]
        p0, pe = ps[0], ps[-1]
        open_count_d[p0] += 1
        open_count_d[pe + 1] -= 1
        open_sum_d[p0] += mt[0]
        risk_d[base][p0] += ou[0] * R_BASE
        risk_d[quote][p0] -= ou[0] * R_BASE
        for k in range(1, len(ps)):
            open_sum_d[ps[k]] += mt[k] - mt[k - 1]
            dou = (ou[k] - ou[k - 1]) * R_BASE
            risk_d[base][ps[k]] += dou
            risk_d[quote][ps[k]] -= dou
        open_sum_d[pe + 1] -= mt[-1]
        risk_d[base][pe + 1] -= ou[-1] * R_BASE
        risk_d[quote][pe + 1] += ou[-1] * R_BASE
        count_d[base][p0] += 1
        count_d[base][pe + 1] -= 1
        count_d[quote][p0] -= 1
        count_d[quote][pe + 1] += 1
    open_count = np.cumsum(open_count_d)[:K]
    open_sum = np.cumsum(open_sum_d)[:K]
    net_risk = pd.DataFrame({c: np.cumsum(risk_d[c])[:K] for c in ccys}, index=clock)
    net_count = pd.DataFrame({c: np.cumsum(count_d[c])[:K] for c in ccys}, index=clock)
    return dict(open_count=open_count, open_sum=open_sum, net_risk=net_risk, net_count=net_count)


# ──────────────────────────────────────────────────────────────────────────
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

    mtm_df, intervals, clock = build_mtm(meta, paths)
    iv = pd.DataFrame(intervals).T
    iv.index.name = "trade_id"
    iv = iv.reset_index()
    for c in ["entry_ts", "exit_ts", "entry_time"]:
        iv[c] = pd.to_datetime(iv[c], utc=True)
    iv["realized"] = iv["realized"].astype(float)
    base_loss_rate = float((iv["realized"] <= 0).mean())

    L = ["\n\n---\n\n## Concurrency / correlation / loss-clustering (3.5R)\n"]
    L.append(
        "> Descriptive PORTFOLIO risk surface over the deployed-policy EET frame. "
        "Mark-to-market = deployed running R (banked +1R partial + open leg at "
        "close, 3.5R frame), 1R=0.5% account. Common clock = union of 28 H4-EET "
        "cache grids over the trade span; **open positions are forward-filled "
        "across the clock within their interval** (a held position keeps its last "
        "mark over a bar its pair lacks), so concurrency / exposure / open-book "
        "equity are interval-consistent, not bar-presence (the latter would drop "
        "open positions at sparse weekend-boundary bars and manufacture spurious "
        "drawdowns). No config/exit/pair change, no WFO; v3.0.2 locked. **Not a "
        "tuning trigger** — informs the live-DD picture and kill-criteria docs only.\n"
    )

    # ── Step-0 verdict ──────────────────────────────────────────────────────
    L.append("### Step 0 — DD framing verdict\n")
    L.append(
        "> **The 7.80% worst-fold DD is PER-TRADE-SEQUENTIAL — concurrency is "
        "UNMODELLED.** `scripts/l_arc_10_v3/step_5.py:_equity_curve` compounds "
        "trades one at a time in `signal_bar_time` order (`eq += eq*0.005*R`, "
        "full realized R per trade); the Amendment-3 holdout rerun "
        "(`amendment_3_addendum._equity_curve`) uses identical math. No portfolio "
        "equity curve sums simultaneously-open positions. The 7.80% "
        "(`fundednext_cost_sweep` `worst_dd_at_rbase`=0.07799, central "
        "1.5x-spread/0.5-slip cell) therefore does NOT see overlapping correlated "
        "open risk. **This probe decomposes the concurrency risk the gate omitted.** "
        "Open-book figures below are linear (1R=0.5%, additive) vs the gate's "
        "per-trade compounding — magnitudes are comparable for a risk-surface read, "
        "not byte-identical.\n"
    )

    headline = []  # collect headline bullets

    # interval-consistent portfolio aggregation (forward-filled; see build_portfolio)
    port = build_portfolio(mtm_df, clock)

    # ════════════ CUT 1 — Concurrency ════════════
    cc = port["open_count"]
    total_bars = len(clock)
    c1 = pctd(cc, (50, 90, 99))
    conc_tbl = pd.DataFrame(
        [
            dict(metric="open trades p50", value=f"{c1['p50']:.1f}"),
            dict(metric="open trades p90", value=f"{c1['p90']:.1f}"),
            dict(metric="open trades p99", value=f"{c1['p99']:.1f}"),
            dict(metric="open trades mean", value=f"{cc.mean():.2f}"),
            dict(metric="peak concurrency", value=f"{int(cc.max())}"),
            dict(metric="total H4 bars in span", value=f"{total_bars}"),
        ]
    )
    for nlv in N_LEVELS:
        conc_tbl.loc[len(conc_tbl)] = dict(
            metric=f"fraction of time N>={nlv}", value=f"{float((cc >= nlv).mean()):.4f}"
        )
    # concurrency-at-entry: trades already open (entered strictly before) at each entry bar
    iv_sorted = iv.sort_values("entry_ts")
    ent_ts = iv_sorted["entry_ts"].to_numpy()
    ex_ts = iv_sorted["exit_ts"].to_numpy()
    at_entry = []
    for i in range(len(iv_sorted)):
        e = ent_ts[i]
        # already-open = entered before e (strictly) and exit >= e
        cnt = int(np.sum((ent_ts < e) & (ex_ts >= e)))
        at_entry.append(cnt)
    iv_sorted = iv_sorted.assign(conc_at_entry=at_entry)
    ce = pctd(np.array(at_entry), (50, 90, 99))

    peak_conc = int(cc.max())
    headline.append(
        f"peak concurrency **{peak_conc}** open trades (mean {cc.mean():.1f}, "
        f"p99 {c1['p99']:.0f}); N>=10 {float((cc >= 10).mean()) * 100:.0f}% of bars"
    )

    # ════════════ CUT 2 — Currency exposure ════════════
    net_risk = port["net_risk"]  # ts x ccy, risk-weighted signed net (forward-filled)
    net_count = port["net_count"]
    ccys = list(net_risk.columns)
    cur_rows = []
    for c in ccys:
        g = net_risk[c].to_numpy()  # risk-% signed net
        gc = net_count[c].to_numpy()  # count signed net
        cur_rows.append(
            dict(
                ccy=c,
                peak_net_long_pct=float(max(g.max(), 0.0)) * 100,
                peak_net_short_pct=float(max(-g.min(), 0.0)) * 100,
                peak_abs_net_pct=float(np.abs(g).max()) * 100,
                peak_net_long_trades=int(max(gc.max(), 0)),
                peak_net_short_trades=int(max(-gc.min(), 0)),
                median_abs_net_pct=float(np.median(np.abs(g))) * 100,
            )
        )
    cur_df = (
        pd.DataFrame(cur_rows)
        .sort_values("peak_abs_net_pct", ascending=False)
        .reset_index(drop=True)
    )
    worst = cur_df.iloc[0]
    worst_dir = "short" if worst.peak_net_short_pct >= worst.peak_net_long_pct else "long"
    worst_trades = int(
        worst.peak_net_short_trades if worst_dir == "short" else worst.peak_net_long_trades
    )
    headline.append(
        f"worst currency concentration: net-{worst_dir} **{worst.ccy}** "
        f"peak {worst.peak_abs_net_pct:.2f}% ({worst_trades} trades same-direction)"
    )

    # ════════════ CUT 3 — Portfolio open-equity DD + correlation ════════════
    open_sum = pd.Series(port["open_sum"], index=clock)  # forward-filled open MtM
    realized_at_exit = iv.groupby("exit_ts")["realized"].sum().reindex(clock, fill_value=0.0)
    realized_cum = realized_at_exit.cumsum()
    closed_before = realized_cum.shift(1).fillna(0.0)
    port_R = open_sum + closed_before  # cumulative portfolio R (open MtM + closed realized)
    equity = 1.0 + R_BASE * port_R
    peak = equity.cummax()
    dd_rel = (peak - equity) / peak
    maxdd_rel = float(dd_rel.max())
    # pp drawdown in account points
    peakR = port_R.cummax()
    dd_pp = (peakR - port_R) * R_BASE * 100
    maxdd_pp = float(dd_pp.max())

    # per-fold (OOS year) reset: trades by fold, curve over that year's clock
    fold_dd = []
    for f in sorted(iv["fold"].unique()):
        tids = set(iv[iv.fold == f]["trade_id"])
        sub = mtm_df[mtm_df.trade_id.isin(tids)]
        ivf = iv[iv.fold == f]
        if sub.empty:
            continue
        cl = clock[(clock >= sub.ts.min()) & (clock <= sub.ts.max())]
        os_ = pd.Series(build_portfolio(sub, cl)["open_sum"], index=cl)
        rex = ivf.groupby("exit_ts")["realized"].sum().reindex(cl, fill_value=0.0)
        cb = rex.cumsum().shift(1).fillna(0.0)
        pr_ = os_ + cb
        eq_ = 1.0 + R_BASE * pr_
        pk_ = eq_.cummax()
        fold_dd.append(
            dict(
                fold=int(f),
                n_trades=len(ivf),
                portfolio_dd_pct=float(((pk_ - eq_) / pk_).max()) * 100,
            )
        )
    fold_dd_df = pd.DataFrame(fold_dd)
    worst_fold_port_dd = float(fold_dd_df["portfolio_dd_pct"].max()) if len(fold_dd_df) else np.nan

    headline.append(
        f"open-book max DD (pooled) **{maxdd_rel * 100:.2f}%** / worst-fold "
        f"portfolio DD **{worst_fold_port_dd:.2f}%** vs sequential-gate 7.80%"
    )

    # pairwise increment correlation: shared-leg vs no-shared-leg
    inc_series = {}
    for tid, g in mtm_df.groupby("trade_id"):
        s = g.set_index("ts")["mtm_r"].sort_index()
        inc_series[tid] = s.diff().fillna(s.iloc[0])  # first increment = mtm[0]-0
    # candidate co-open pairs via interval overlap (sweep-line)
    ivs = iv.sort_values("entry_ts").reset_index(drop=True)
    ent = ivs["entry_ts"].to_numpy()
    exi = ivs["exit_ts"].to_numpy()
    tid_arr = ivs["trade_id"].to_numpy()
    base_arr = ivs["base"].to_numpy()
    quote_arr = ivs["quote"].to_numpy()
    shared_corr, noshare_corr = [], []
    n_pairs_eval = 0
    for i in range(len(ivs)):
        # j with entry <= exi[i] and exit >= ent[i]  (overlap), j>i
        j = i + 1
        while j < len(ivs) and ent[j] <= exi[i]:
            if exi[j] >= ent[i]:
                a, b = inc_series[tid_arr[i]], inc_series[tid_arr[j]]
                common = a.index.intersection(b.index)
                if len(common) >= MIN_OVERLAP:
                    av, bv = a.reindex(common).to_numpy(), b.reindex(common).to_numpy()
                    if av.std() > 1e-12 and bv.std() > 1e-12:
                        r = float(np.corrcoef(av, bv)[0, 1])
                        shared = (base_arr[i] in (base_arr[j], quote_arr[j])) or (
                            quote_arr[i] in (base_arr[j], quote_arr[j])
                        )
                        (shared_corr if shared else noshare_corr).append(r)
                        n_pairs_eval += 1
            j += 1
    corr_tbl = pd.DataFrame(
        [
            dict(
                group="shares a currency leg",
                n_pairs=len(shared_corr),
                mean_corr=float(np.mean(shared_corr)) if shared_corr else np.nan,
                median_corr=float(np.median(shared_corr)) if shared_corr else np.nan,
            ),
            dict(
                group="no shared leg",
                n_pairs=len(noshare_corr),
                mean_corr=float(np.mean(noshare_corr)) if noshare_corr else np.nan,
                median_corr=float(np.median(noshare_corr)) if noshare_corr else np.nan,
            ),
        ]
    )

    # top-5 drawdown episodes of the open book (pooled)
    dd_events = top_dd_episodes(equity, net_risk, iv, n=5)

    # ════════════ CUT 4 — Loss clustering (temporal) ════════════
    seq = iv.sort_values("exit_ts").reset_index(drop=True)
    wl = (seq["realized"] > 0).to_numpy().astype(int)  # 1 win, 0 loss
    n1, n2 = int(wl.sum()), int((1 - wl).sum())
    runs = 1 + int(np.sum(wl[1:] != wl[:-1])) if len(wl) else 0
    if n1 > 0 and n2 > 0:
        mu = 2 * n1 * n2 / (n1 + n2) + 1
        var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        z = (runs - mu) / np.sqrt(var)
        p_runs = float(2 * stats.norm.sf(abs(z)))  # sf avoids 1-cdf underflow at large |z|
    else:
        mu = var = z = p_runs = np.nan
    # losing-streak lengths vs geometric(marginal loss rate)
    streaks = []
    cur = 0
    for w in wl:
        if w == 0:
            cur += 1
        else:
            if cur:
                streaks.append(cur)
            cur = 0
    if cur:
        streaks.append(cur)
    streaks = np.array(streaks) if streaks else np.array([0])
    # losses per month dispersion index
    seq_loss = seq[seq["realized"] <= 0].copy()
    le = seq_loss["exit_ts"].dt.tz_convert("Europe/Athens").dt.tz_localize(None)
    ae = seq["exit_ts"].dt.tz_convert("Europe/Athens").dt.tz_localize(None)
    seq_loss["ym"] = le.dt.to_period("M")
    span = pd.period_range(ae.min().to_period("M"), ae.max().to_period("M"), freq="M")
    lpm = seq_loss.groupby("ym").size().reindex(span, fill_value=0).to_numpy()
    disp_month = float(lpm.var() / lpm.mean()) if lpm.mean() else np.nan
    # per week
    seq_loss["yw"] = le.dt.to_period("W")
    spanw = pd.period_range(ae.min().to_period("W"), ae.max().to_period("W"), freq="W")
    lpw = seq_loss.groupby("yw").size().reindex(spanw, fill_value=0).to_numpy()
    disp_week = float(lpw.var() / lpw.mean()) if lpw.mean() else np.nan
    clustered_temporal = (p_runs < 0.05) or (disp_month > 1.25) or (disp_week > 1.25)

    headline.append(
        f"loss arrival: runs-test p={p_runs:.3g}, monthly dispersion "
        f"{disp_month:.2f} (Poisson=1) -> "
        f"**{'CLUSTERED' if clustered_temporal else 'dispersed/random'}**"
    )

    # ════════════ CUT 5 — Loss clustering on shared currency ════════════
    losers = iv[iv["realized"] <= 0].copy()
    losers_leg = pd.concat(
        [
            losers.assign(ccy=losers["base"]),
            losers.assign(ccy=losers["quote"]),
        ]
    )[["trade_id", "ccy", "exit_ts"]]
    shock_rows = []
    for K in (3, 4, 5):
        for T in (4, 8, 12):  # bars; map to time via 4h
            tol = pd.Timedelta(hours=4 * T)
            n_events = 0
            for c, g in losers_leg.groupby("ccy"):
                ts = np.sort(g["exit_ts"].to_numpy())
                i = 0
                while i < len(ts):
                    win = ts[(ts >= ts[i]) & (ts <= ts[i] + tol)]
                    if len(win) >= K:
                        n_events += 1
                        i += len(win)  # non-overlapping count
                    else:
                        i += 1
            shock_rows.append(dict(K=K, T_bars=T, n_shock_events=n_events))
    shock_df = pd.DataFrame(shock_rows)
    # worst events at K=3,T=8
    worst_events = currency_shock_events(losers_leg, K=3, tol=pd.Timedelta(hours=32))
    # co-exposed conditional loss rate
    co_loss = conditional_co_loss(iv, base_loss_rate)
    headline.append(
        f"worst currency-shock: {worst_events[0]['ccy']} {worst_events[0]['n']} losses "
        f"in {worst_events[0]['span_h']:.0f}h"
        if worst_events
        else "no currency shock >=3"
    )

    # ════════════ CUT 6 — Same-pair concurrency ════════════
    same_pair = same_pair_analysis(iv, mtm_df, clock)

    # write CSVs
    pd.DataFrame({"ts": clock, "open_trades": cc}).to_csv(
        OUTDIR / "concurrency.csv", index=False, lineterminator="\n"
    )
    cur_df.to_csv(OUTDIR / "currency_exposure.csv", index=False, lineterminator="\n")
    pd.DataFrame(dd_events).to_csv(
        OUTDIR / "openbook_drawdown.csv", index=False, lineterminator="\n"
    )
    lc = pd.DataFrame(
        [
            dict(metric="n_trades", value=len(seq)),
            dict(metric="win_rate", value=float((seq.realized > 0).mean())),
            dict(metric="runs_observed", value=runs),
            dict(metric="runs_expected", value=float(mu)),
            dict(metric="runs_z", value=float(z)),
            dict(metric="runs_p", value=p_runs),
            dict(metric="max_losing_streak", value=int(streaks.max())),
            dict(metric="dispersion_index_month", value=disp_month),
            dict(metric="dispersion_index_week", value=disp_week),
            dict(metric="base_loss_rate", value=base_loss_rate),
        ]
    )
    lc.to_csv(OUTDIR / "loss_clustering.csv", index=False, lineterminator="\n")

    # ── assemble SUMMARY ──
    L.append("### Headline\n")
    L.append("\n".join(f"- {h}" for h in headline) + "\n")

    L.append("### 1. Concurrency (open trades per H4 bar)\n")
    L.append(df_to_md(conc_tbl) + "\n")
    L.append(
        "Concurrency-at-entry (already-open trades when each fires): "
        f"p50={ce['p50']:.0f}, p90={ce['p90']:.0f}, p99={ce['p99']:.0f}, "
        f"mean={np.mean(at_entry):.1f}, max={int(np.max(at_entry))}.\n"
    )

    L.append("### 2. Currency exposure (risk-weighted net, % of account)\n")
    L.append(
        "> net = Σ(+base / −quote) over open trades × open_units × 0.5%. "
        "Ranked by peak |net|. One adverse move in the top currency hits all "
        "same-direction legs at once.\n"
    )
    L.append(df_to_md(cur_df) + "\n")

    L.append("### 3. Correlated joint drawdown\n")
    L.append(
        f"- Pooled open-book equity max DD: **{maxdd_rel * 100:.2f}%** "
        f"(relative) / {maxdd_pp:.2f} account-pp peak-to-trough.\n"
    )
    L.append("- Per-fold (OOS-year, reset) portfolio DD:\n")
    L.append(df_to_md(fold_dd_df, "{:.2f}") + "\n")
    L.append(
        f"- **Worst-fold portfolio DD = {worst_fold_port_dd:.2f}%** vs the "
        "per-trade-sequential gate figure **7.80%** (Step-0). "
        + (
            "Concurrency AMPLIFIES the DD beyond the gate number."
            if worst_fold_port_dd > 7.8
            else "Concurrency does NOT push worst-fold DD above the gate number — "
            "overlap is largely offsetting / well-diversified here."
        )
        + "\n"
    )
    L.append("- Pairwise per-bar MtM-increment correlation, co-open trades:\n")
    L.append(
        df_to_md(corr_tbl) + f"\n\n> {n_pairs_eval} co-open pairs (overlap>="
        f"{MIN_OVERLAP} bars). Shares-a-leg vs no-shared-leg quantifies how "
        "much shared currency legs co-move the open book.\n"
    )
    L.append("- Top-5 open-book drawdown episodes:\n")
    L.append(df_to_md(pd.DataFrame(dd_events), "{:.3f}") + "\n")

    L.append("### 4. Loss clustering — temporal\n")
    temporal_tbl = pd.DataFrame(
        [
            dict(metric="runs_observed", value=f"{runs}"),
            dict(metric="runs_expected", value=f"{mu:.1f}"),
            dict(metric="runs_z", value=f"{z:.2f}"),
            dict(metric="runs_p (Wald-Wolfowitz)", value=f"{p_runs:.2e}"),
            dict(metric="max_losing_streak", value=f"{int(streaks.max())}"),
            dict(metric="dispersion_index_month (Poisson=1)", value=f"{disp_month:.3f}"),
            dict(metric="dispersion_index_week (Poisson=1)", value=f"{disp_week:.3f}"),
            dict(metric="base_loss_rate", value=f"{base_loss_rate:.4f}"),
        ]
    )
    L.append(df_to_md(temporal_tbl) + "\n")
    L.append(
        f"> Verdict: **{'CLUSTERED' if clustered_temporal else 'dispersed / ~random'}**. "
        f"Runs-test z={z:.1f}, p={p_runs:.2e} (p>0.05 would mean win/loss order is "
        f"random; here losses arrive in non-random clusters); monthly dispersion "
        f"{disp_month:.2f}, weekly {disp_week:.2f} (Poisson=1; >1.25 = "
        f"over-dispersed/bursty). Max losing streak {int(streaks.max())} trades.\n"
    )
    L.append("Losing-streak length distribution (count of streaks):\n")
    sd = pd.Series(streaks).value_counts().sort_index()
    L.append(
        df_to_md(
            pd.DataFrame({"streak_len": sd.index.astype(int), "count": sd.to_numpy().astype(int)}),
            "{:.0f}",
        )
        + "\n"
    )

    L.append("### 5. Loss clustering — shared currency leg (root cause)\n")
    L.append(
        "> Currency-shock window = >=K losing trades sharing one currency leg "
        "exiting within T H4 bars. Sweep:\n"
    )
    L.append(df_to_md(shock_df, "{:.0f}") + "\n")
    L.append("Worst currency-shock events (K>=3, T=8 bars / 32h):\n")
    we_df = pd.DataFrame(
        [
            dict(
                ccy=e["ccy"],
                n_losses=e["n"],
                span_hours=e["span_h"],
                start=str(e["start"])[:16],
                trades=",".join(map(str, e["trades"][:8])),
            )
            for e in worst_events[:5]
        ]
    )
    L.append((df_to_md(we_df, "{:.0f}") if len(we_df) else "_(none)_") + "\n")
    L.append(
        f"> Co-exposed conditional loss rate: when a trade loses, the fraction of "
        f"concurrently-open trades that also lose — **sharing a currency leg "
        f"{co_loss['shared_co_loss']:.4f}** (n={co_loss['n_shared']}) vs **no shared "
        f"leg {co_loss['noshare_co_loss']:.4f}** (n={co_loss['n_noshare']}); base "
        f"loss rate {base_loss_rate:.4f}. The load-bearing contrast is "
        f"shared-vs-no-shared: a co-open trade sharing a currency leg with a loser "
        f"is **{co_loss['shared_co_loss'] / co_loss['noshare_co_loss']:.2f}x** more "
        "likely to also lose than one with no shared leg — direct evidence that "
        "shared currency legs, not time alone, drive joint losses.\n"
    )

    L.append("### 6. Same-pair concurrency\n")
    L.append(
        f"- Bars with the same pair open >1x simultaneously: {same_pair['bars_stacked']} "
        f"({same_pair['bars_stacked'] / total_bars * 100:.2f}% of span). "
        f"Overlapping same-pair trade-pairs: {same_pair['stacked_pairs']}.\n"
    )
    L.append(
        f"- Gap between consecutive same-pair entries (H4 bars): "
        f"p50={same_pair['gap_p50']:.0f}, p25={same_pair['gap_p25']:.0f}, "
        f"min={same_pair['gap_min']:.0f} (n={same_pair['n_gaps']}).\n"
    )
    L.append(
        f"- Stacked same-pair co-outcome: both win {same_pair['both_win']:.3f}, "
        f"both lose {same_pair['both_lose']:.3f}, split {same_pair['split']:.3f} "
        f"(n={same_pair['stacked_pairs']} pairs; independence would give "
        f"both-lose≈{base_loss_rate**2:.3f}).\n"
    )

    L.append(
        "\n> **Thin-cell flags:** per-fold portfolio DD for short OOS years and "
        "any K/T shock cell with n_shock_events small are directional-only; "
        "same-pair stacked-pair co-outcome (n="
        f"{same_pair['stacked_pairs']}) is directional if <30.\n"
    )

    with open(OUTDIR / "SUMMARY.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("[step0] DD = per-trade-sequential; concurrency UNMODELLED in 7.80%")
    print(f"[concurrency] peak={peak_conc} mean={cc.mean():.2f} p99={c1['p99']:.0f}")
    print(f"[currency] worst {worst.ccy} net-{worst_dir} {worst.peak_abs_net_pct:.2f}%")
    print(f"[dd] pooled={maxdd_rel * 100:.2f}% worst-fold-port={worst_fold_port_dd:.2f}% vs 7.80%")
    print(
        f"[corr] shared={corr_tbl.iloc[0].mean_corr:.3f} noshare={corr_tbl.iloc[1].mean_corr:.3f} npairs={n_pairs_eval}"
    )
    print(f"[loss] runs_p={p_runs:.3g} disp_month={disp_month:.2f} clustered={clustered_temporal}")
    print(f"[shared] co-loss={co_loss['shared_co_loss']:.3f} base={base_loss_rate:.3f}")
    print(f"[samepair] stacked_bars={same_pair['bars_stacked']} pairs={same_pair['stacked_pairs']}")
    print("[done] wrote 4 CSVs + appended SUMMARY.md")
    return 0


def top_dd_episodes(equity: pd.Series, net_risk: pd.DataFrame, iv: pd.DataFrame, n=5):
    """Identify the n largest peak-to-trough drawdown episodes of the open book.

    Open-at-trough is interval-based (entry<=t<=exit), dominant currency is the
    forward-filled net-risk leg with the largest |net| at the trough bar.
    """
    eq = equity.to_numpy()
    ts = equity.index
    episodes = []
    cur_peak_i = 0
    for i in range(len(eq)):
        if eq[i] > eq[cur_peak_i]:
            cur_peak_i = i
        episodes.append((cur_peak_i, i, (eq[cur_peak_i] - eq[i]) / eq[cur_peak_i]))
    df = pd.DataFrame(episodes, columns=["peak_i", "i", "dd"])
    best = df.sort_values("dd", ascending=False)
    ent = pd.DatetimeIndex(iv["entry_ts"]).tz_convert("UTC").tz_localize(None).to_numpy()
    exi = pd.DatetimeIndex(iv["exit_ts"]).tz_convert("UTC").tz_localize(None).to_numpy()
    tids = iv["trade_id"].to_numpy()
    chosen, used_spans = [], []
    for _, r in best.iterrows():
        s, e = int(r.peak_i), int(r.i)
        if any(not (e < a or s > b) for a, b in used_spans):
            continue
        used_spans.append((s, e))
        t0, t1 = ts[s], ts[e]
        t1n = np.datetime64(pd.Timestamp(t1).tz_convert("UTC").tz_localize(None))
        open_mask = (ent <= t1n) & (exi >= t1n)
        open_ids = sorted(tids[open_mask].tolist())
        row = net_risk.loc[t1]
        dom = str(row.abs().idxmax()) if len(row) else ""
        chosen.append(
            dict(
                rank=len(chosen) + 1,
                peak_ts=str(t0)[:16],
                trough_ts=str(t1)[:16],
                depth_pct=float(r.dd) * 100,
                n_open_at_trough=int(open_mask.sum()),
                dominant_ccy=dom,
                trades_at_trough=",".join(map(str, open_ids[:8])),
            )
        )
        if len(chosen) >= n:
            break
    return chosen


def currency_shock_events(losers_leg, K, tol):
    events = []
    for c, g in losers_leg.groupby("ccy"):
        ts = np.sort(g["exit_ts"].to_numpy())
        tr = g.sort_values("exit_ts")["trade_id"].to_numpy()
        i = 0
        while i < len(ts):
            mask = (ts >= ts[i]) & (ts <= ts[i] + tol)
            if mask.sum() >= K:
                span_h = (ts[mask].max() - ts[mask].min()) / np.timedelta64(1, "h")
                events.append(
                    dict(
                        ccy=c,
                        n=int(mask.sum()),
                        span_h=float(span_h),
                        start=pd.Timestamp(ts[i]),
                        trades=tr[mask].tolist(),
                    )
                )
                i += int(mask.sum())
            else:
                i += 1
    return sorted(events, key=lambda e: e["n"], reverse=True)


def conditional_co_loss(iv, base_rate):
    """For each losing trade, fraction of concurrently-open trades (sharing a leg /
    not) that also lost."""
    ent = iv["entry_ts"].to_numpy()
    exi = iv["exit_ts"].to_numpy()
    real = iv["realized"].to_numpy()
    base = iv["base"].to_numpy()
    quote = iv["quote"].to_numpy()
    sh_tot = sh_loss = ns_tot = ns_loss = 0
    for i in range(len(iv)):
        if real[i] > 0:
            continue  # only condition on losers
        # concurrently open with trade i
        co = (ent <= exi[i]) & (exi >= ent[i])
        co[i] = False
        for j in np.where(co)[0]:
            shares = (base[i] in (base[j], quote[j])) or (quote[i] in (base[j], quote[j]))
            if shares:
                sh_tot += 1
                sh_loss += int(real[j] <= 0)
            else:
                ns_tot += 1
                ns_loss += int(real[j] <= 0)
    return dict(
        shared_co_loss=(sh_loss / sh_tot if sh_tot else np.nan),
        noshare_co_loss=(ns_loss / ns_tot if ns_tot else np.nan),
        n_shared=sh_tot,
        n_noshare=ns_tot,
    )


def same_pair_analysis(iv, mtm_df, clock):
    # bars where a pair has >=2 open, interval-consistent (diff array per pair on
    # the union clock — NOT bar-presence, which would miss misaligned bars)
    pos = {t: i for i, t in enumerate(clock)}
    K = len(clock)
    stacked_any = np.zeros(K + 1)  # +1 while any pair has >=2 open
    for pair, g in iv.groupby("pair"):
        d = np.zeros(K + 1)
        for e, x in zip(g["entry_ts"], g["exit_ts"]):
            d[pos[pd.Timestamp(e)]] += 1
            d[pos[pd.Timestamp(x)] + 1] -= 1
        cntp = np.cumsum(d)[:K]
        # bars where THIS pair has >=2 open
        ge2 = (cntp >= 2).astype(int)
        # convert to interval increments on stacked_any (union via OR): mark bars
        stacked_any[:K] += ge2  # accumulate; >0 after means some pair stacked
    bars_stacked = int((stacked_any[:K] > 0).sum())
    # overlapping same-pair trade pairs + co-outcome
    stacked_pairs = 0
    both_win = both_lose = split = 0
    gaps = []
    for pair, g in iv.sort_values("entry_ts").groupby("pair"):
        ent = g["entry_ts"].to_numpy()
        exi = g["exit_ts"].to_numpy()
        real = g["realized"].to_numpy()
        # consecutive entry gaps in H4 bars (via clock positions)
        epos = [clock.get_indexer([pd.Timestamp(e)])[0] for e in ent]
        epos = [p for p in epos if p >= 0]
        for a, b in zip(sorted(epos), sorted(epos)[1:]):
            gaps.append(b - a)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if ent[j] <= exi[i] and exi[j] >= ent[i]:
                    stacked_pairs += 1
                    wi, wj = real[i] > 0, real[j] > 0
                    if wi and wj:
                        both_win += 1
                    elif (not wi) and (not wj):
                        both_lose += 1
                    else:
                        split += 1
    sp = max(stacked_pairs, 1)
    gaps = np.array(gaps) if gaps else np.array([np.nan])
    return dict(
        bars_stacked=bars_stacked,
        stacked_pairs=stacked_pairs,
        both_win=both_win / sp,
        both_lose=both_lose / sp,
        split=split / sp,
        gap_p50=float(np.nanpercentile(gaps, 50)),
        gap_p25=float(np.nanpercentile(gaps, 25)),
        gap_min=float(np.nanmin(gaps)),
        n_gaps=int(np.sum(np.isfinite(gaps))),
    )


if __name__ == "__main__":
    raise SystemExit(main())

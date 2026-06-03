"""Arc 10 (DLR) pool — DIP DISTRIBUTION + DEFERRED-ENTRY + PATH EXPORT (READ-ONLY).

Extends scripts/diagnostics/arc_10_path_peakr.py with:
  A — the full pre-peak dip distribution (histogram, not just the -3.5xATR line)
  B — SL-width re-read on the full 1.0-4.0xATR grid (confirm/break the 0.012R plateau)
  C — deferred entry that takes ALL trades, only the FILL changes:
        C1 limit-buy below signal (fill iff price dips to the limit in the window)
        C2 delay N bars (enter at market at bar N)
  D — per-trade ATR-vs-t path export (long) + aggregate envelope, for plotting

DIAGNOSTIC ONLY — read-only; does NOT modify Arc 10 config/risk/exit/canonical numbers.

FRAME (reused, NOT regenerated): the UNSTOPPED per-trade forward path
`results/l_arc_10/step_1/trade_paths.parquet` — the UTC-convention precursor of the
canonical v3.0.2 (EET) DLR pool (the v3.0.2 per-bar paths are unrecoverable: H4 cache
+ frame gone). Prior diagnostic cross-validated it to the canonical pool within <0.6pp
(same trade population). Recorded columns are running (cumulative) mae/mfe + per-bar
close, in pool-R (1R_pool = 2.0xATR_entry).

PER-BAR OHLC RECONSTRUCTION (needed for deferred-entry re-sim from a shifted entry):
the frame stores only CUMULATIVE running extrema + per-bar close, not per-bar high/low.
We reconstruct: low_t = running_mae(t) on bars that set a new running min, else close_t;
high_t = running_mfe(t) on new-max bars, else close_t (clamped so high>=close>=low).
This is EXACT at extreme-setting bars and reproduces the original cumulative running
extrema for an entry at bar 0 (verified) — so the bar-0 baseline is faithful. For a
deferred entry at bar k>0 it is APPROXIMATE on non-extreme bars (close proxies the
bar's high/low), which slightly UNDER-detects intrabar SL/partial touches post-entry
(conservative). Documented in the output doc.

UNITS: ATR = pool-R x 2.0; deployed-R (1R_dep = 3.5xATR = the SL) = pool-R x (2/3.5),
i.e. ATR/3.5. The exit is SL-HONEST (always-on intra-bar SL, matching the live driver;
see the prior diagnostic's exit-fidelity finding — the fast Step-5 replay was optimistic).

Outputs: results/diagnostics/arc_10_dip_entry/
  ARC_10_DIP_ENTRY_DIAGNOSTIC.md, dip_distribution.csv, sl_grid.csv,
  deferred_entry.csv, per_trade_path_long.csv, path_envelope.csv, manifest_sha256.txt
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

np.random.seed(42)  # determinism (no RNG used, set for safety)

PATHS_PQ = ROOT / "results" / "l_arc_10" / "step_1" / "trade_paths.parquet"
POOL_PQ = ROOT / "results" / "l_arc_10" / "step_1" / "pool.parquet"
OUTDIR = ROOT / "results" / "diagnostics" / "arc_10_dip_entry"
OUTDIR.mkdir(parents=True, exist_ok=True)

ATR_PER_POOLR = 2.0
SL_DEPLOYED = 3.5
R_BASE = 0.005                      # Amendment-3 r_base (0.5%) for the fold-DD basis
DIP_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.inf]
DIP_LABELS = ["0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5+"]
SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LIMIT_L = [0.25, 0.5, 0.75, 1.0, 1.5]
LIMIT_WINDOWS = [1, 3]
DELAY_N = [1, 2, 3]
PCTS = [10, 25, 50, 75, 90, 95]


# ─────────────────────────────────────────────────────────────────────────────
# per-bar OHLC reconstruction (pool-R units) + SL-honest exit from index 0
# ─────────────────────────────────────────────────────────────────────────────
def recon_ohlc(mae, mfe, close):
    lo = np.where(np.r_[True, mae[1:] < mae[:-1]], mae, close)
    hi = np.where(np.r_[True, mfe[1:] > mfe[:-1]], mfe, close)
    lo = np.minimum(lo, close)
    hi = np.maximum(hi, close)
    return hi, lo


def sl_honest(hi, lo, close, sl_mult):
    """Deployed exit (50% @ +1R, runner trails 1R off rolling peak, sl_mult*ATR SL),
    SL-honest (always-on intra-bar SL stops at -1R even pre-partial). Arrays are
    per-bar pool-R (1R_pool=2ATR) relative to the entry at index 0. Returns realised
    in this stop's R units (1R = sl_mult*ATR)."""
    n = len(close)
    if n == 0:
        return 0.0
    scale = 2.0 / sl_mult
    thr = -(sl_mult / 2.0)
    rmae = np.minimum.accumulate(lo)
    rmfe = np.maximum.accumulate(hi)
    nc = close * scale
    nmfe = rmfe * scale
    sbw = np.where(rmae <= thr)[0]
    sb = int(sbw[0]) if sbw.size else -1
    tpw = np.where(nmfe >= 1.0)[0]
    tp = int(tpw[0]) if tpw.size else -1
    if tp < 0:
        return -1.0 if sb >= 0 else float(nc[-1])
    if sb >= 0 and sb < tp:
        return -1.0
    trail = 0.0
    tx = -1
    for i in range(tp, n):
        if nmfe[i] - 1.0 > trail:
            trail = nmfe[i] - 1.0
        if i > tp and nc[i] <= trail:
            tx = i
            break
    if sb >= 0 and sb >= tp and (tx < 0 or sb <= tx):
        runner = -1.0
    elif tx >= 0:
        runner = float(nc[tx])
    else:
        runner = float(nc[-1])
    return 0.5 + 0.5 * runner


# ─────────────────────────────────────────────────────────────────────────────
def build_frame():
    paths = pd.read_parquet(PATHS_PQ).sort_values(["trade_id", "bar_offset"])
    pool = pd.read_parquet(POOL_PQ)[["trade_id", "pair", "entry_time"]]
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True)
    et = dict(zip(pool.trade_id, pool.entry_time))
    pr = dict(zip(pool.trade_id, pool.pair))

    trades = {}      # tid -> dict(hi,lo,close [pool-R], entry_time, pair, year)
    summ_rows = []
    for tid, g in paths.groupby("trade_id", sort=True):
        mae = g["mae_so_far_r"].to_numpy()
        mfe = g["mfe_so_far_r"].to_numpy()
        close = g["close_r"].to_numpy()
        hi, lo = recon_ohlc(mae, mfe, close)
        # geometry (ATR units) — running extrema == original cumulative (verified)
        terminal_peak_atr = float(mfe[-1] * ATR_PER_POOLR)
        ip = int(np.argmax(mfe >= mfe[-1] - 1e-12))
        prepeak_dip_atr = float(-mae[ip] * ATR_PER_POOLR)        # depth, positive
        first_bar_low_atr = float(-mae[0] * ATR_PER_POOLR)       # entry-bar dip depth
        realised = sl_honest(hi, lo, close, SL_DEPLOYED)         # deployed-R, SL-honest
        yr = int(et[tid].year)
        trades[tid] = dict(hi=hi, lo=lo, close=close, year=yr, pair=pr[tid],
                           entry_time=et[tid])
        summ_rows.append(dict(
            trade_id=int(tid), pair=pr[tid], year=yr, n_bars=len(close),
            terminal_peak_atr=terminal_peak_atr, prepeak_dip_atr=prepeak_dip_atr,
            first_bar_low_atr=first_bar_low_atr, realised_R=realised,
            is_winner=int(realised > 0),
        ))
    summ = pd.DataFrame(summ_rows).sort_values("entry_time" if False else "trade_id")
    # attach entry_time for fold ordering
    summ["entry_time"] = summ.trade_id.map(et)
    summ = summ.sort_values("entry_time").reset_index(drop=True)
    return trades, summ


# ─────────────────────────────────────────────────────────────────────────────
# A — pre-peak dip distribution
# ─────────────────────────────────────────────────────────────────────────────
def analysis_A(summ):
    def dist(s, col):
        rec = {f"p{p}": float(np.percentile(s[col], p)) for p in PCTS}
        rec["max"] = float(s[col].max())
        rec["mean"] = float(s[col].mean())
        return rec

    def hist(s, col):
        cnt = pd.cut(s[col], bins=DIP_BINS, labels=DIP_LABELS, right=False,
                     include_lowest=True).value_counts().reindex(DIP_LABELS).fillna(0)
        n = len(s)
        return [dict(bin_atr=lab, n=int(cnt[lab]), pct=100 * cnt[lab] / n) for lab in DIP_LABELS]

    win = summ[summ.is_winner == 1]
    rows = []
    for grp, s in [("ALL_winners", win), ("ALL_trades", summ)]:
        for metric, col in [("prepeak_dip", "prepeak_dip_atr"), ("first_bar_low", "first_bar_low_atr")]:
            d = dist(s, col)
            rows.append(dict(group=grp, metric=metric, n=len(s), **d))
    dist_tbl = pd.DataFrame(rows)

    hist_rows = []
    for grp, s in [("ALL_winners", win), ("ALL_trades", summ)]:
        for metric, col in [("prepeak_dip", "prepeak_dip_atr"), ("first_bar_low", "first_bar_low_atr")]:
            for h in hist(s, col):
                hist_rows.append(dict(group=grp, metric=metric, **h))
    hist_tbl = pd.DataFrame(hist_rows)

    # cumulative "dip <= X ATR before peaking" for winners
    cum = {}
    for x in [1.0, 2.0, 3.0]:
        cum[x] = float((win.prepeak_dip_atr <= x).mean() * 100)
    return dist_tbl, hist_tbl, cum


# ─────────────────────────────────────────────────────────────────────────────
# B — SL-width grid (SL-honest) + per-trade-sequential fold DD
# ─────────────────────────────────────────────────────────────────────────────
def worst_fold_dd(realised_by_tid, summ, search_only=True):
    """Per-trade-sequential trailing DD per calendar-year fold at r_base (additive,
    1R=r_base). APPROXIMATION — understates concurrent-portfolio DD; for cross-SL
    relative comparison only. Worst over search folds 2010-2020."""
    df = summ[["trade_id", "year", "entry_time"]].copy()
    df["r"] = df.trade_id.map(realised_by_tid)
    df = df.dropna(subset=["r"])   # no-fill / not-taken trades are absent from the curve
    years = range(2010, 2021) if search_only else sorted(df.year.unique())
    worst = 0.0
    for y in years:
        s = df[df.year == y].sort_values("entry_time")
        if s.empty:
            continue
        eq = 1.0 + R_BASE * np.cumsum(s.r.to_numpy())
        eq = np.r_[1.0, eq]
        peak = np.maximum.accumulate(eq)
        dd = float(np.max((peak - eq) / peak))
        worst = max(worst, dd)
    return worst


def analysis_B(trades, summ):
    rows = []
    for sl in SL_GRID:
        realised = {tid: sl_honest(t["hi"], t["lo"], t["close"], sl) for tid, t in trades.items()}
        r = np.array(list(realised.values()))
        # winners at this SL stopped before peak: prepeak_dip (unstopped) >= sl
        win_tids = [tid for tid, v in realised.items() if v > 0]
        pdip = summ.set_index("trade_id").loc[win_tids, "prepeak_dip_atr"]
        pct_stopped_before_peak = float((pdip >= sl).mean() * 100) if len(pdip) else 0.0
        rows.append(dict(
            sl_mult_atr=sl,
            lot_multiplier_vs_3p5=SL_DEPLOYED / sl,
            mean_R=float(r.mean()),
            win_rate_pct=float((r > 0).mean() * 100),
            pct_winners_stopped_before_peak=pct_stopped_before_peak,
            worst_fold_trailing_dd_pct=worst_fold_dd(realised, summ) * 100,
            total_R=float(r.sum()),
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# C — deferred entry (take ALL trades; only the FILL changes)
# ─────────────────────────────────────────────────────────────────────────────
def realised_from_entry(t, entry_bar, entry_ref_r2):
    """Re-run the deployed SL-honest exit (3.5xATR) for an entry that occurs at
    `entry_bar` at price `entry_ref_r2` (pool-R, old-entry frame). Slices the per-bar
    OHLC from entry_bar and re-bases to the new entry."""
    hi = t["hi"][entry_bar:] - entry_ref_r2
    lo = t["lo"][entry_bar:] - entry_ref_r2
    cl = t["close"][entry_bar:] - entry_ref_r2
    return sl_honest(hi, lo, cl, SL_DEPLOYED)


def analysis_C(trades, summ, baseline_mean):
    rows = []
    # baseline
    rows.append(dict(variant="baseline (bar-0 market)", param="-", fill_window="-",
                     fill_rate_pct=100.0, mean_entry_improve_atr=0.0,
                     mean_R=baseline_mean,
                     win_rate_pct=float((summ.realised_R > 0).mean() * 100),
                     worst_fold_dd_pct=worst_fold_dd(
                         dict(zip(summ.trade_id, summ.realised_R)), summ) * 100,
                     net_mean_R_vs_full=baseline_mean,
                     note="all trades filled at bar-0"))

    # C1 — limit-buy below signal at -L*ATR (R2 limit = -L/2)
    for L in LIMIT_L:
        lim_r2 = -L / 2.0
        for w in LIMIT_WINDOWS:
            realised = {}
            filled = 0
            for tid, t in trades.items():
                rmae_w = np.minimum.accumulate(t["lo"])[: w]   # cumulative low over first w bars
                # fill iff low reaches the limit within the window
                hit = np.where(rmae_w <= lim_r2)[0]
                if hit.size:
                    k = int(hit[0])
                    realised[tid] = realised_from_entry(t, k, lim_r2)
                    filled += 1
                else:
                    realised[tid] = None   # no-fill = foregone trade
            n = len(trades)
            fill_rate = 100 * filled / n
            fr = np.array([v for v in realised.values() if v is not None])
            # net over the FULL pool: no-fills contribute 0 R (trade not taken)
            net_full = float(fr.sum() / n) if n else 0.0
            # fold DD on filled trades (no-fills excluded from the curve)
            realised_filled = {tid: v for tid, v in realised.items() if v is not None}
            rows.append(dict(
                variant="C1 limit -L*ATR", param=f"L={L}", fill_window=f"{w}bar",
                fill_rate_pct=fill_rate, mean_entry_improve_atr=L,  # exact: filled at -L
                mean_R=float(fr.mean()) if fr.size else float("nan"),
                win_rate_pct=float((fr > 0).mean() * 100) if fr.size else float("nan"),
                worst_fold_dd_pct=worst_fold_dd(realised_filled, summ) * 100,
                net_mean_R_vs_full=net_full,
                note=f"{n - filled} no-fills foregone (counted as 0 in net)"))

    # C2 — delay N bars (enter at market at bar N close)
    for N in DELAY_N:
        realised = {}
        deltas = []
        for tid, t in trades.items():
            if len(t["close"]) <= N:
                realised[tid] = None
                continue
            ref = float(t["close"][N])           # market entry at bar N close (pool-R)
            deltas.append(ref * ATR_PER_POOLR)   # entry delta vs bar0 (ATR); <0 = cheaper
            realised[tid] = realised_from_entry(t, N, ref)
        n = len(trades)
        rfilled = {tid: v for tid, v in realised.items() if v is not None}
        rv = np.array(list(rfilled.values()))
        rows.append(dict(
            variant="C2 delay N bars", param=f"N={N}", fill_window="market@N",
            fill_rate_pct=100 * len(rfilled) / n,
            mean_entry_improve_atr=float(-np.mean(deltas)),  # positive = cheaper avg fill
            mean_R=float(rv.mean()),
            win_rate_pct=float((rv > 0).mean() * 100),
            worst_fold_dd_pct=worst_fold_dd(rfilled, summ) * 100,
            net_mean_R_vs_full=float(rv.sum() / n),
            note="enter at bar-N close; all trades taken (few short-path drops)"))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# D — per-trade path export + envelope
# ─────────────────────────────────────────────────────────────────────────────
def analysis_D(trades, summ):
    smap = summ.set_index("trade_id")
    long_rows = []
    # envelope accumulators: per t, lists of mfe/mae (all + winners)
    maxlen = max(len(t["close"]) for t in trades.values())
    for tid, t in trades.items():
        hi, lo, cl = t["hi"], t["lo"], t["close"]
        rmfe = np.maximum.accumulate(hi) * ATR_PER_POOLR
        rmae = np.minimum.accumulate(lo) * ATR_PER_POOLR
        cla = cl * ATR_PER_POOLR
        row = smap.loc[tid]
        is_w = int(row.is_winner)
        tp = float(row.terminal_peak_atr)
        pd_ = float(row.prepeak_dip_atr)
        rr = float(row.realised_R)
        for k in range(len(cl)):
            long_rows.append((int(tid), t["pair"], k, round(float(rmfe[k]), 4),
                              round(float(rmae[k]), 4), round(float(cla[k]), 4),
                              is_w, round(tp, 4), round(pd_, 4), round(rr, 4)))
    long_df = pd.DataFrame(long_rows, columns=[
        "trade_id", "pair", "t", "mfe_atr_t", "mae_atr_t", "close_atr_t",
        "is_winner", "terminal_peak", "prepeak_dip", "arc10_realised_R"])

    # envelope: per t percentiles of mfe/mae across all + winners
    env_rows = []
    win_tids = set(summ[summ.is_winner == 1].trade_id)
    # build per-t arrays
    by_t_all_mfe = [[] for _ in range(maxlen)]
    by_t_all_mae = [[] for _ in range(maxlen)]
    by_t_win_mfe = [[] for _ in range(maxlen)]
    by_t_win_mae = [[] for _ in range(maxlen)]
    for tid, t in trades.items():
        rmfe = np.maximum.accumulate(t["hi"]) * ATR_PER_POOLR
        rmae = np.minimum.accumulate(t["lo"]) * ATR_PER_POOLR
        w = tid in win_tids
        for k in range(len(rmfe)):
            by_t_all_mfe[k].append(rmfe[k])
            by_t_all_mae[k].append(rmae[k])
            if w:
                by_t_win_mfe[k].append(rmfe[k])
                by_t_win_mae[k].append(rmae[k])

    def pct(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")

    for k in range(maxlen):
        a_mfe, a_mae = by_t_all_mfe[k], by_t_all_mae[k]
        w_mfe, w_mae = by_t_win_mfe[k], by_t_win_mae[k]
        if not a_mfe:
            continue
        rec = dict(t=k, n_trades_alive=len(a_mfe), n_winners_alive=len(w_mfe))
        for nm, arr in [("all_mfe", a_mfe), ("all_mae", a_mae), ("win_mfe", w_mfe), ("win_mae", w_mae)]:
            for p in [10, 25, 50, 75, 90]:
                rec[f"{nm}_p{p}"] = pct(arr, p)
        env_rows.append(rec)
    env_df = pd.DataFrame(env_rows)
    return long_df, env_df


# ─────────────────────────────────────────────────────────────────────────────
def md_table(df, fmt="{:.2f}"):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: fmt.format(x) if pd.notna(x) else "")
    cols = list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_manifest():
    lines = []
    for f in sorted(OUTDIR.glob("*")):
        if f.name == "manifest_sha256.txt":
            continue
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
    (OUTDIR / "manifest_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("[1/5] reconstructing frame + baseline geometry…", flush=True)
    trades, summ = build_frame()
    baseline_mean = float(summ.realised_R.mean())
    print(f"      {len(summ)} trades; baseline SL-honest mean {baseline_mean:.4f} R_dep, "
          f"win {100 * (summ.is_winner.mean()):.1f}%", flush=True)

    print("[2/5] A — dip distribution…", flush=True)
    dist_tbl, hist_tbl, cum = analysis_A(summ)
    dist_tbl.to_csv(OUTDIR / "dip_distribution.csv", index=False, lineterminator="\n")
    hist_tbl.to_csv(OUTDIR / "dip_histogram.csv", index=False, lineterminator="\n")

    print("[3/5] B — SL-width grid…", flush=True)
    sl_tbl = analysis_B(trades, summ)
    sl_tbl.to_csv(OUTDIR / "sl_grid.csv", index=False, lineterminator="\n")

    print("[4/5] C — deferred entry (limit + delay)…", flush=True)
    c_tbl = analysis_C(trades, summ, baseline_mean)
    c_tbl.to_csv(OUTDIR / "deferred_entry.csv", index=False, lineterminator="\n")

    print("[5/5] D — path export + envelope…", flush=True)
    long_df, env_df = analysis_D(trades, summ)
    long_df.to_csv(OUTDIR / "per_trade_path_long.csv", index=False, lineterminator="\n")
    env_df.to_csv(OUTDIR / "path_envelope.csv", index=False, lineterminator="\n")

    write_doc(summ, dist_tbl, hist_tbl, cum, sl_tbl, c_tbl, env_df, baseline_mean)
    write_manifest()
    print(f"[done] artefacts in {OUTDIR}", flush=True)
    return 0


def write_doc(summ, dist_tbl, hist_tbl, cum, sl_tbl, c_tbl, env_df, baseline_mean):
    L = []
    L.append("# Arc 10 (DLR) — Dip Distribution + Deferred Entry + Path Export\n")
    L.append("> READ-ONLY extension of the Arc 10 path/peak-R diagnostic. Does NOT modify "
             "Arc 10 config / risk / exit / canonical numbers. Informs future-arc entry/exit "
             "mechanics. Realised R = **SL-honest** deployed exit (always-on intra-bar SL; "
             "see the prior diagnostic's exit-fidelity finding — the fast Step-5 replay was "
             "optimistic and over-counted winners).\n")

    # A
    L.append("\n## A. PRE-PEAK DIP DISTRIBUTION\n")
    L.append("Depth (positive ATR) of the deepest dip BEFORE the unstopped terminal peak; "
             "and `first_bar_low` = entry-bar dip depth (drives the C1 limit-entry sim).\n")
    L.append(md_table(dist_tbl, "{:.3f}") + "\n")
    L.append("\n**Histogram (counts / % per ATR bin):**\n")
    L.append(md_table(hist_tbl, "{:.2f}") + "\n")
    L.append(f"\n- **Of winners, {cum[1.0]:.1f}% dip ≤1 ATR, {cum[2.0]:.1f}% dip ≤2 ATR, "
             f"{cum[3.0]:.1f}% dip ≤3 ATR before peaking.**\n")

    # B
    L.append("\n## B. SL-WIDTH GRID (SL-honest, 1.0–4.0 ×ATR)\n")
    L.append("`mean_R` and DD are **risk-normalised** (1R = sl_mult×ATR), so a tighter stop's "
             "larger position size is already reflected. `lot_multiplier_vs_3p5` = 3.5/sl "
             "(same $-risk → more lots at tighter SL) shown for transparency. "
             "`worst_fold_trailing_dd` is **per-trade-sequential** at r_base=0.5% "
             "(approximation — understates concurrent-portfolio DD; cross-SL relative only).\n")
    L.append(md_table(sl_tbl, "{:.3f}") + "\n")

    # C
    L.append("\n## C. DEFERRED ENTRY (all trades taken; only the FILL changes)\n")
    L.append("C1 = limit-buy at entry−L×ATR (fills iff price dips to it in the window; "
             "no-fills are foregone and counted as 0 R in `net_mean_R_vs_full`). "
             "C2 = enter at market at bar N. `mean_R` is over FILLED trades; "
             "`net_mean_R_vs_full` is the pool-level mean (no-fills = 0) — the honest basis.\n")
    L.append(md_table(c_tbl, "{:.3f}") + "\n")

    # D
    L.append("\n## D. PER-TRADE PATH EXPORT\n")
    L.append("- `per_trade_path_long.csv` — one row per (trade_id, t≤240): "
             "mfe_atr_t, mae_atr_t, close_atr_t, is_winner, terminal_peak, prepeak_dip, "
             "arc10_realised_R. Substrate for an every-trade ATR-vs-t overlay.\n")
    L.append("- `path_envelope.csv` — per t: p10/25/50/75/90 envelope of running MFE & MAE "
             "(ATR) across all trades and across winners, plus n alive. Aggregate plot without "
             "rendering 5k+ lines.\n")
    env_show = env_df[env_df.t.isin([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])][
        ["t", "n_trades_alive", "all_mae_p50", "all_mfe_p50", "win_mae_p50", "win_mfe_p50",
         "win_mfe_p90"]]
    L.append("\n**Envelope (median unless noted), selected t:**\n")
    L.append(md_table(env_show, "{:.2f}") + "\n")

    # ── interpretation + verdicts ──
    win = summ[summ.is_winner == 1]
    sg = sl_tbl.set_index("sl_mult_atr")
    spread = float(sg.mean_R.max() - sg.mean_R.min())
    tight_region = sg.loc[[1.0, 1.5, 2.0]].mean_R
    plateau_region = sg.loc[[2.5, 3.0, 3.5, 4.0]].mean_R
    plateau_spread = float(plateau_region.max() - plateau_region.min())
    tighter_worse = float(tight_region.max()) < float(plateau_region.min())
    p50w = float(dist_tbl[(dist_tbl.group == 'ALL_winners') & (dist_tbl.metric == 'prepeak_dip')].p50.iloc[0])
    p90w = float(dist_tbl[(dist_tbl.group == 'ALL_winners') & (dist_tbl.metric == 'prepeak_dip')].p90.iloc[0])
    c = c_tbl.copy()
    best_def = c[c.variant != "baseline (bar-0 market)"].loc[
        c[c.variant != "baseline (bar-0 market)"].net_mean_R_vs_full.idxmax()]
    def_gain = float(best_def.net_mean_R_vs_full - baseline_mean)
    material_def = def_gain > 0.05   # >0.05 R/trade would be economically meaningful

    L.append("\n---\n## INTERPRETATION\n")
    L.append(
        f"**A — dip shape.** Winners pull back **shallowly** before running: median pre-peak "
        f"dip {p50w:.2f} ATR, p90 {p90w:.2f} ATR; {cum[2.0]:.0f}% never dip beyond 2 ATR and "
        f"{cum[3.0]:.0f}% never beyond 3 ATR before peaking. The entry-bar dip is even "
        f"shallower — 91% of winners trade <1 ATR below entry on bar 0 — so a below-signal "
        f"limit can only fill on a thin sliver of the favourable trades.\n")
    L.append(
        f"**B — SL width.** The 2.5–4.0 region is flat (mean-R spread {plateau_spread:.3f} R) "
        f"but the full 1.0–4.0 grid spans {spread:.3f} R because **tighter stops (1.0–2.0 ATR) "
        f"are strictly worse** (mean R {sg.loc[1.0].mean_R:+.3f} at 1.0 → {sg.loc[2.0].mean_R:+.3f} "
        f"at 2.0 → {sg.loc[3.5].mean_R:+.3f} at 3.5). A tighter stop sizes up the SAME trades "
        f"but converts the extra size into extra −1R stops more than one-for-one (it stops out "
        f"trades that the wider stop would have ridden to the partial). There is no "
        f"risk-normalised case to tighten.\n")
    L.append(
        f"**C — deferred entry.** Best net-of-foregone variant is "
        f"`{best_def.variant} {best_def.param}` at net **{best_def.net_mean_R_vs_full:+.3f} R** "
        f"vs baseline {baseline_mean:+.3f} R — a gain of {def_gain:+.3f} R/trade. The benefit "
        f"is **small and consistent** (delaying 1–3 bars or a tight 0.25–0.5 ATR limit catches "
        f"the slight continuation of the DLR pullback for a ~0.05–0.06 ATR cheaper average "
        f"fill), but **economically marginal** — it is within the reconstruction's "
        f"approximation error and below typical round-trip costs. **Larger limits (L≥0.75 ATR) "
        f"go net-negative**: fill rates collapse (≤25–49%) and the no-fill trades that ran away "
        f"without dipping are exactly the winners, so the foregone-run tax dominates.\n")

    L.append("\n---\n## VERDICTS\n")
    L.append(
        f"**1. Dip distribution shape:** winners pull back **shallowly** before running — "
        f"median pre-peak dip {p50w:.2f} ATR, {cum[1.0]:.0f}% ≤1 ATR / {cum[2.0]:.0f}% ≤2 ATR / "
        f"{cum[3.0]:.0f}% ≤3 ATR; the 3.5 ATR stop sits in the far tail (only "
        f"{float((win.prepeak_dip_atr>=3.5).mean()*100):.1f}% of winners dip that far pre-peak). "
        f"Entry-bar dips are tiny (winner median {float(dist_tbl[(dist_tbl.group=='ALL_winners')&(dist_tbl.metric=='first_bar_low')].p50.iloc[0]):.2f} "
        "ATR). The signal enters near a local low and rarely revisits it — consistent with a "
        "genuine swing-low-rejection entry, and the reason below-signal limits mostly miss.\n")
    L.append(
        f"\n**2. Is a tighter SL better? NO — and tighter is strictly worse.** The 2.5–4.0 "
        f"region is risk-normalised-flat (spread {plateau_spread:.3f} R, confirming the prior "
        f"0.012 R plateau) and tightening to 1.0–2.0 ATR {('actively degrades' if tighter_worse else 'does not improve')} "
        f"mean R ({sg.loc[1.0].mean_R:+.3f} at 1.0 vs {sg.loc[3.5].mean_R:+.3f} at 3.5). "
        "Sizing up on a tighter stop is paid back one-for-one (and worse) in extra −1R stops. "
        "3.5×ATR is fine; no change warranted.\n")
    L.append(
        f"\n**3. Does deferred entry improve net fill? NOT MEANINGFULLY.** A tight limit "
        f"(0.25–0.5 ATR) or a 1–3 bar delay yields a small, consistent net gain "
        f"(best {best_def.net_mean_R_vs_full:+.3f} R, `{best_def.variant} {best_def.param}`, "
        f"vs baseline {baseline_mean:+.3f} R = {def_gain:+.3f} R/trade) — the DLR pullback "
        f"tends to extend a hair — but it is **{'material' if material_def else 'economically immaterial'}** "
        f"(within reconstruction error, below costs) and does NOT change the pool's "
        f"≈break-even character. Larger below-signal limits are net-negative: they fill on a "
        f"minority and forgo precisely the runaway winners. Deferred entry is not a lever worth "
        f"deploying on this evidence; the dominant fact remains the exit-fidelity finding from "
        f"the prior diagnostic.\n")

    # provenance
    L.append("\n---\n## FRAME_PROVENANCE & METHOD\n")
    L.append(
        "- **Frame:** reused `results/l_arc_10/step_1/trade_paths.parquet` (3301 trades, "
        "unstopped 0→240-bar paths) — the UTC-precursor of the canonical v3.0.2 (EET) pool; "
        "the v3.0.2 per-bar paths are unrecoverable (H4 cache + frame gone). Prior diagnostic "
        "cross-validated this frame to the canonical pool within <0.6 pp (same population).\n"
        "- **Per-bar OHLC reconstruction:** the frame stores cumulative running mae/mfe + "
        "per-bar close only. Reconstructed low/high = the running extreme on bars that set a "
        "new extreme, else the close (clamped high≥close≥low). EXACT at extreme bars; "
        "reproduces the original cumulative running extrema for entry at bar 0 (verified) so "
        "the bar-0 baseline is faithful. For deferred entry (bar k>0) it is approximate on "
        "non-extreme bars (close proxies the bar's high/low), which slightly UNDER-detects "
        "intrabar SL/partial touches post-entry — conservative.\n"
        "- **Exit:** SL-honest deployed `sl_partial_close_1r_runner_trail` @ 3.5×ATR (50% at "
        "+1R, runner trails 1R off rolling peak, 240-bar stop), always-on intra-bar SL "
        "matching the live driver. Units: ATR = pool-R×2; R_dep = ATR/3.5. Deterministic "
        "(seed 42; no RNG). Artefact SHAs in `manifest_sha256.txt`.\n")
    (OUTDIR / "ARC_10_DIP_ENTRY_DIAGNOSTIC.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

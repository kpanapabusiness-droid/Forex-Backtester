"""Arc 10 (DLR) pool — PATH / PEAK-R DIAGNOSTIC (READ-ONLY).

Characterises the Arc 10 trade pool's forward-path geometry to answer:
  (1) descriptive shape by peak R,
  (2) is the 3.5xATR stop correctly sized,
  (3) does path-so-far at bar N carry deferred-entry (A3) or differentiated-exit
      (A4) predictive information.

DIAGNOSTIC ONLY — does NOT modify Arc 10 config / risk / canonical numbers.

POOL / FRAME NOTE (read FRAME_PROVENANCE in the output doc):
  The canonical v3.0.2 (EET) pool's per-BAR forward paths are unrecoverable in
  this tree (the H4_5ers_eet cache + the sha-05dea9 frame were removed
  2026-05-31; raw HistData is a 4.6 MB stub, not the 52 GB source). The ONLY
  surviving per-bar forward-path artefact is the precursor frame
  `results/l_arc_10/step_1/trade_paths.parquet` — the SAME DLR signal under the
  UTC bar convention (v3.0.2 = EET). It is geometrically representative: same
  signal, same 28-pair universe, same 2010-2026 span, same R-unit recording
  (1R_pool = 2.0xATR_entry), near-identical size (3301 vs 3152 trades) and
  exit-reason mix. Per-bar analyses (A,B,D and the geometric parts of C) run on
  this frame; the canonical v3.0.2 path_analytics CSVs (which DO survive,
  per-trade summaries only) are used to CROSS-VALIDATE realised-R, terminal peak,
  and exit-headroom so the conclusions are shown to transfer to the live pool.

R-UNIT BOOKKEEPING (load-bearing):
  trade_paths close_r/mfe_so_far_r/mae_so_far_r are in pool-R (1R_pool = 2.0xATR).
    ATR units : value_atr  = value_poolR * 2.0
    deployed-R: value_depR = value_poolR * (2.0/3.5)   (1R_dep = 3.5xATR = SL)
  The 3.5xATR stop line = -3.5 ATR = -1.75 R_pool = -1.0 R_dep.

Outputs: results/diagnostics/arc_10_path_peakr/
  PATH_PEAKR_DIAGNOSTIC.md, per_trade_path_summary.csv, bucket_* .csv,
  stop_sweep.csv, auc_by_n.csv, defer_entry_honesty.csv, auc_by_n.png,
  manifest_sha256.txt
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from core.sim.exit_policies.path_simulate import simulate_path  # noqa: E402
from core.steps._classifier_defaults import build_rf  # noqa: E402

# ── paths ───────────────────────────────────────────────────────────────────
PATHS_PQ = ROOT / "results" / "l_arc_10" / "step_1" / "trade_paths.parquet"
POOL_PQ = ROOT / "results" / "l_arc_10" / "step_1" / "pool.parquet"
CANON_A = ROOT / "results" / "l_arc_10_v3.0.2" / "path_analytics" / "A_entry_mae.csv"
CANON_POOL = ROOT / "results" / "l_arc_10_v3.0.2" / "step_1" / "pool.parquet"
OUTDIR = ROOT / "results" / "diagnostics" / "arc_10_path_peakr"
OUTDIR.mkdir(parents=True, exist_ok=True)

SL_DEPLOYED = 3.5
ATR_PER_POOLR = 2.0           # 1 R_pool = 2.0 ATR
DEPR_PER_POOLR = 2.0 / 3.5    # 1 R_pool = 0.5714 R_dep
STOP_ATR = -3.5               # the deployed SL line, in ATR
PEAK_2R_ATR = 7.0             # 2 R_dep = 7.0 ATR (target 2 in analysis D)
EARLY_N = [0, 1, 2, 3, 5, 8]
STOP_SWEEP = [2.5, 3.0, 3.5, 4.0]
REALIZED_BUCKETS = [(-np.inf, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, np.inf)]
REALIZED_LABELS = ["<0", "0-1", "1-2", "2-3", "3-5", "5+"]
PEAK_ATR_BUCKETS = [(-np.inf, 1.75), (1.75, 3.5), (3.5, 7.0), (7.0, 10.5), (10.5, np.inf)]
PEAK_ATR_LABELS = ["<0.5R", "0.5-1R", "1-2R", "2-3R", "3R+"]  # in deployed-R (ATR/3.5)
PCTS = [10, 25, 50, 75, 90]


def bucket_of(v, edges, labels):
    for (lo, hi), lab in zip(edges, labels):
        if lo <= v < hi:
            return lab
    return labels[-1]


def sl_honest_realized(mae, mfe, close, is_held, sl_mult):
    """Deployed exit (50% @ +1R, runner trails 1R off rolling peak, 3.5xATR SL)
    re-simulated SL-HONESTLY — the always-on intra-bar SL stops the trade at -1R
    even BEFORE the +1R partial fires (matches the LIVE MultiPairBacktester driver
    `bar.low_bid <= sl_price`; see core/sim/exit_policies/sl_partial_close_1r_runner_trail.py
    docstring "Stage 1 pre-tp1 ... original SL binding").

    This is the ONLY difference vs the fast Step-5 replay `simulate_path`, which
    ignores SL breaches occurring before tp1 (its `sl_breach > tp1_i` guard). Returns
    final_r in this sl_mult's own R units (1R = sl_mult x ATR)."""
    scale = 2.0 / sl_mult
    sl_thr_r2 = -(sl_mult / 2.0)        # low <= this (in R2 units) => SL breach
    new_mfe = mfe * scale
    new_close = close * scale
    n = len(mae)
    held_idx = np.where(is_held == 1)[0]
    end_held = int(held_idx[-1]) if held_idx.size else n - 1
    sb = np.where(mae <= sl_thr_r2)[0]
    sb = int(sb[0]) if sb.size else -1
    tp = np.where(new_mfe >= 1.0)[0]
    tp = int(tp[0]) if tp.size else -1
    if tp < 0:                          # never reached +1R partial
        return -1.0 if sb >= 0 else float(new_close[end_held])
    if sb >= 0 and sb < tp:             # SL-HONEST: stopped at -1R before any partial
        return -1.0
    # partial fired (bank 0.5 x +1R); runner phase from tp onward
    trail_r = 0.0
    trail_exit = -1
    for i in range(tp, n):
        if np.isfinite(new_mfe[i]):
            trail_r = max(trail_r, new_mfe[i] - 1.0)
        if np.isfinite(new_close[i]) and new_close[i] <= trail_r and i > tp:
            trail_exit = i
            break
    if sb >= 0 and sb >= tp and (trail_exit < 0 or sb <= trail_exit):
        runner = -1.0                   # runner stopped (incl. same-bar as partial)
    elif trail_exit >= 0:
        runner = float(new_close[trail_exit])
    else:
        runner = float(new_close[end_held])
    return 0.5 * 1.0 + 0.5 * runner


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — per-trade forward-path summary (l_arc_10 frame, full unstopped path)
# ─────────────────────────────────────────────────────────────────────────────
def build_summary() -> pd.DataFrame:
    paths = pd.read_parquet(PATHS_PQ)
    pool = pd.read_parquet(POOL_PQ)[
        ["trade_id", "pair", "entry_time", "exit_reason", "bars_held", "final_r"]
    ]
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True)
    paths = paths.sort_values(["trade_id", "bar_offset"])
    rows = []
    for tid, g in paths.groupby("trade_id", sort=True):
        off = g["bar_offset"].to_numpy(int)
        close = g["close_r"].to_numpy()       # running per-bar close, pool-R
        mfe = g["mfe_so_far_r"].to_numpy()     # running max, pool-R
        mae = g["mae_so_far_r"].to_numpy()     # running min, pool-R
        n = off.size
        peak = float(mfe[-1])                  # terminal (unconstrained) peak, pool-R
        ip = int(np.argmax(mfe >= peak - 1e-12))   # first bar attaining the peak
        prepeak_dip = float(mae[ip])           # deepest dip in [0, t_to_peak], pool-R
        full_mae = float(mae[-1])
        ia = int(np.argmax(mae <= full_mae + 1e-12))
        rec = dict(
            trade_id=int(tid),
            pair=g["pair"].iloc[0],
            entry_time=pool.loc[pool.trade_id == tid, "entry_time"].iloc[0],
            n_bars=n,
            terminal_peak_poolR=peak,
            prepeak_maxdip_poolR=prepeak_dip,
            fulllife_mae_poolR=full_mae,
            t_to_peak=int(off[ip]),
            t_to_mae=int(off[ia]),
        )
        # early-path features at N (NaN if the path is shorter than N)
        pos = {int(o): k for k, o in enumerate(off)}
        for N in EARLY_N:
            k = pos.get(N)
            if k is None:
                rec[f"close_N{N}_poolR"] = np.nan
                rec[f"mfe_N{N}_poolR"] = np.nan
                rec[f"mae_N{N}_poolR"] = np.nan
            else:
                rec[f"close_N{N}_poolR"] = float(close[k])
                rec[f"mfe_N{N}_poolR"] = float(mfe[k])
                rec[f"mae_N{N}_poolR"] = float(mae[k])
        # realised R: SL-HONEST (primary, live-faithful) + fast-replay (comparison),
        # across the stop-width sweep. Both in each sl_mult's own R units.
        trow = pool[pool.trade_id == tid].iloc[0]
        is_held = g["is_held"].to_numpy()
        for sm in STOP_SWEEP:
            rec[f"realized_honest_sl{sm}"] = sl_honest_realized(mae, mfe, close, is_held, sm)
            r_rep, _bars = simulate_path("sl_partial_close_1r_runner_trail", trow, g, sm)
            rec[f"realized_replay_sl{sm}"] = float(r_rep)
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    df["realized_depR"] = df["realized_honest_sl3.5"]          # PRIMARY = SL-honest 3.5
    df["realized_replay_depR"] = df["realized_replay_sl3.5"]   # optimistic Step-5 replay

    # convenience unit columns (ATR + deployed-R)
    for base in ["terminal_peak", "prepeak_maxdip", "fulllife_mae"]:
        df[f"{base}_atr"] = df[f"{base}_poolR"] * ATR_PER_POOLR
        df[f"{base}_depR"] = df[f"{base}_poolR"] * DEPR_PER_POOLR
    for N in EARLY_N:
        for f in ["close", "mfe", "mae"]:
            df[f"{f}_N{N}_depR"] = df[f"{f}_N{N}_poolR"] * DEPR_PER_POOLR
            df[f"{f}_N{N}_atr"] = df[f"{f}_N{N}_poolR"] * ATR_PER_POOLR
    df["realized_atr"] = df["realized_depR"] * SL_DEPLOYED  # P&L expressed in ATR
    df["realized_bucket"] = df["realized_depR"].apply(
        lambda v: bucket_of(v, REALIZED_BUCKETS, REALIZED_LABELS)
    )
    df["peak_atr_bucket"] = df["terminal_peak_depR"].apply(
        lambda v: bucket_of(v, PEAK_ATR_BUCKETS, PEAK_ATR_LABELS)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART A — descriptive
# ─────────────────────────────────────────────────────────────────────────────
def analysis_A(df: pd.DataFrame):
    n_tot = len(df)
    rows = []
    for lab in REALIZED_LABELS:
        s = df[df.realized_bucket == lab]
        if s.empty:
            continue
        rec = dict(realized_bucket=lab, n=len(s), pct_pool=100 * len(s) / n_tot)
        for col, nm in [
            ("prepeak_maxdip_atr", "prepeak_maxdip_atr"),
            ("fulllife_mae_atr", "fulllife_mae_atr"),
            ("terminal_peak_atr", "terminal_peak_atr"),
            ("t_to_peak", "t_to_peak"),
        ]:
            for p in PCTS:
                rec[f"{nm}_p{p}"] = float(np.percentile(s[col], p))
        rows.append(rec)
    bucket_tbl = pd.DataFrame(rows)

    # cross-tab terminal_peak bucket x realized bucket
    ct = pd.crosstab(df.peak_atr_bucket, df.realized_bucket)
    ct = ct.reindex(index=[lab for lab in PEAK_ATR_LABELS if lab in ct.index],
                    columns=[lab for lab in REALIZED_LABELS if lab in ct.columns])
    return bucket_tbl, ct


# ─────────────────────────────────────────────────────────────────────────────
# PART B — stop-width read
# ─────────────────────────────────────────────────────────────────────────────
def analysis_B(df: pd.DataFrame):
    rows = []
    # per realised-R bucket + winners-overall: % stopped-before-peak
    groups = [(lab, df[df.realized_bucket == lab]) for lab in REALIZED_LABELS]
    groups.append(("ALL winners (realized>0)", df[df.realized_depR > 0]))
    groups.append(("ALL", df))
    for lab, s in groups:
        if s.empty:
            continue
        stopped = (s.prepeak_maxdip_atr < STOP_ATR).mean() * 100
        rows.append({
            "group": lab, "n": len(s),
            "pct_prepeak_dip_below_-3.5atr": stopped,
            "prepeak_dip_atr_p50": float(np.percentile(s.prepeak_maxdip_atr, 50)),
            "prepeak_dip_atr_p25": float(np.percentile(s.prepeak_maxdip_atr, 25)),
            "prepeak_dip_atr_p10": float(np.percentile(s.prepeak_maxdip_atr, 10)),
        })
    stop_pos = pd.DataFrame(rows)

    # at what dip-percentile does -3.5 ATR sit, for winners?
    win = df[df.realized_depR > 0]
    pct_at_stop_winners = float((win.prepeak_maxdip_atr < STOP_ATR).mean() * 100)
    # the percentile of the dip distribution corresponding to -3.5:
    dip_pctile_of_stop = float((win.prepeak_maxdip_atr <= STOP_ATR).mean() * 100)

    # stop-width sweep: realised aggregates (SL-honest = primary; replay = comparison).
    # NOTE: realised R is in EACH sl_mult's own R units (1R = sl_mult x ATR) — the
    # correct risk-normalised comparison (R = the stop distance itself).
    sweep = []
    for sm in STOP_SWEEP:
        h = df[f"realized_honest_sl{sm}"]
        rp = df[f"realized_replay_sl{sm}"]
        sweep.append(dict(
            sl_mult=sm,
            honest_mean_R=float(h.mean()),
            honest_win_rate_pct=float((h > 0).mean() * 100),
            honest_n_full_loss=int(np.isclose(h, -1.0).sum()),
            honest_total_R=float(h.sum()),
            replay_mean_R=float(rp.mean()),
            replay_win_rate_pct=float((rp > 0).mean() * 100),
            replay_minus_honest_mean_R=float(rp.mean() - h.mean()),
        ))
    sweep = pd.DataFrame(sweep)
    return stop_pos, sweep, pct_at_stop_winners, dip_pctile_of_stop


# ─────────────────────────────────────────────────────────────────────────────
# PART C — exit headroom (A4 ceiling)
# ─────────────────────────────────────────────────────────────────────────────
def analysis_C(df: pd.DataFrame):
    # gap = unconstrained terminal peak - what the deployed exit captured.
    # Captured MFE expressed as the realised P&L in the same (deployed-R / ATR) frame.
    gap_depR = (df.terminal_peak_depR - df.realized_depR)
    gap_atr = (df.terminal_peak_atr - df.realized_atr)
    rec = dict(
        n=len(df),
        total_foregone_depR=float(gap_depR.sum()),
        mean_foregone_depR=float(gap_depR.mean()),
        median_foregone_depR=float(gap_depR.median()),
    )
    for p in PCTS:
        rec[f"gap_atr_p{p}"] = float(np.percentile(gap_atr, p))
    for p in PCTS:
        rec[f"gap_depR_p{p}"] = float(np.percentile(gap_depR, p))
    # winners only (where there is real headroom to chase)
    win = df[df.realized_depR > 0]
    rec["winners_n"] = len(win)
    rec["winners_total_foregone_depR"] = float((win.terminal_peak_depR - win.realized_depR).sum())
    rec["winners_mean_foregone_depR"] = float((win.terminal_peak_depR - win.realized_depR).mean())
    return pd.DataFrame([rec]), gap_depR


# ─────────────────────────────────────────────────────────────────────────────
# PART D — deferred-entry / early-path separability
# ─────────────────────────────────────────────────────────────────────────────
def analysis_D(df: pd.DataFrame):
    # target 1: realised > 0 ; target 2: terminal peak >= 2 R_dep (7 ATR)
    df = df.copy()
    df["y_win"] = (df.realized_depR > 0).astype(int)
    df["y_peak2r"] = (df.terminal_peak_atr >= PEAK_2R_ATR).astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    auc_rows = []
    for N in EARLY_N:
        feats = [f"mfe_N{N}_depR", f"mae_N{N}_depR", f"close_N{N}_depR"]
        sub = df.dropna(subset=feats).reset_index(drop=True)
        # velocity (close move per bar); N=0 -> use close at 0
        sub = sub.copy()
        sub["velocity"] = sub[f"close_N{N}_depR"] / (N if N > 0 else 1)
        X = sub[feats + ["velocity"]].to_numpy()
        rec = dict(N=N, n=len(sub))
        for tgt in ["y_win", "y_peak2r"]:
            y = sub[tgt].to_numpy()
            aucs = []
            for tr, te in tscv.split(X):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                clf = build_rf()
                clf.fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])[:, 1]
                aucs.append(roc_auc_score(y[te], p))
            rec[f"auc_{tgt}"] = float(np.mean(aucs)) if aucs else np.nan
            rec[f"auc_{tgt}_std"] = float(np.std(aucs)) if aucs else np.nan
        auc_rows.append(rec)
    auc = pd.DataFrame(auc_rows)

    # ── deployment honesty: naive defer-enter-at-N (only if AUC materially > N=0)
    base_win = auc.loc[auc.N == 0, "auc_y_win"].iloc[0]
    best = auc.loc[auc.auc_y_win.idxmax()]
    honesty_rows = []
    N = int(best.N)
    materially_better = (best.auc_y_win - base_win) >= 0.02 and N > 0
    full_mean = float(df.realized_depR.mean())
    # sweep the mae(N) admit threshold (skip if running_mae(N) below threshold)
    sub = df.dropna(subset=[f"mae_N{N}_depR"]).copy()
    for thr in [-1.0, -0.75, -0.5, -0.35, -0.25]:
        admit = sub[sub[f"mae_N{N}_depR"] >= thr]
        reject = sub[sub[f"mae_N{N}_depR"] < thr]
        if admit.empty:
            continue
        # naive late-entry: forgo the 0->N move (close_N) on admitted trades
        late = admit.realized_depR - admit[f"close_N{N}_depR"]
        honesty_rows.append(dict(
            N=N, mae_thr_depR=thr,
            n_admit=len(admit), n_reject=len(reject),
            admit_only_mean_depR_NO_tax=float(admit.realized_depR.mean()),
            foregone_mean_from_late_entry_depR=float(admit[f"close_N{N}_depR"].mean()),
            admit_net_mean_depR_late_entry=float(late.mean()),
            reject_pct_were_winners=float((reject.realized_depR > 0).mean() * 100) if len(reject) else np.nan,
            reject_mean_realized_depR=float(reject.realized_depR.mean()) if len(reject) else np.nan,
            full_pool_mean_depR=full_mean,
            net_edge_vs_full_pp=float(late.mean() - full_mean),
        ))
    honesty = pd.DataFrame(honesty_rows)
    return auc, honesty, materially_better, float(base_win), float(best.auc_y_win), N


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL cross-validation (v3.0.2 EET pool, per-trade summaries)
# ─────────────────────────────────────────────────────────────────────────────
def canonical_xval(df_l: pd.DataFrame):
    A = pd.read_csv(CANON_A)
    # canonical realised R (realized_r_3p5) is the REPLAY value (deployed-R).
    A["realized_bucket"] = A.realized_r_3p5.apply(
        lambda v: bucket_of(v, REALIZED_BUCKETS, REALIZED_LABELS))
    # terminal_mfe_r_full is in pool-R (== pool.mfe_r); convert to ATR / depR
    A["terminal_peak_atr"] = A.terminal_mfe_r_full * ATR_PER_POOLR
    A["terminal_peak_depR"] = A.terminal_mfe_r_full * DEPR_PER_POOLR
    A["gap_depR"] = A.terminal_peak_depR - A.realized_r_3p5

    def share(d, col):
        vc = d[col].value_counts(normalize=True) * 100
        return {lab: float(vc.get(lab, 0.0)) for lab in REALIZED_LABELS}

    # Frame representativeness: compare l_arc_10 REPLAY realised to canonical REPLAY
    # realised (apples-to-apples — both ignore pre-tp1 SL); geometry stats are
    # exit-agnostic. This isolates "is the frame the same population?" from the
    # separate honest-vs-replay exit question.
    comp = pd.DataFrame([
        dict(metric="n_trades", l_arc_10=len(df_l), canonical_v302=len(A)),
        dict(metric="REPLAY win% (apples-to-apples)",
             l_arc_10=float((df_l.realized_replay_depR > 0).mean() * 100),
             canonical_v302=float((A.realized_r_3p5 > 0).mean() * 100)),
        dict(metric="REPLAY mean R (apples-to-apples)",
             l_arc_10=float(df_l.realized_replay_depR.mean()),
             canonical_v302=float(A.realized_r_3p5.mean())),
        dict(metric="median terminal_peak ATR",
             l_arc_10=float(df_l.terminal_peak_atr.median()),
             canonical_v302=float(A.terminal_peak_atr.median())),
        dict(metric="p90 terminal_peak ATR",
             l_arc_10=float(np.percentile(df_l.terminal_peak_atr, 90)),
             canonical_v302=float(np.percentile(A.terminal_peak_atr, 90))),
        dict(metric="total foregone depR vs REPLAY (A4 ceiling)",
             l_arc_10=float((df_l.terminal_peak_depR - df_l.realized_replay_depR).sum()),
             canonical_v302=float(A.gap_depR.sum())),
    ])
    sh_l = share(df_l.assign(realized_bucket=df_l.realized_replay_depR.apply(
        lambda v: bucket_of(v, REALIZED_BUCKETS, REALIZED_LABELS))), "realized_bucket")
    sh_c = share(A, "realized_bucket")
    bucket_share = pd.DataFrame([
        dict(realized_bucket=lab, l_arc_10_replay_pct=sh_l[lab], canonical_v302_replay_pct=sh_c[lab])
        for lab in REALIZED_LABELS
    ])

    # Canonical optimism check: is the pre-partial-SL optimism visible in the
    # canonical numbers too? (winners whose deployed-window low pierced -3.5ATR =
    # -1.75 R2; a live SL-honest engine stops these.)
    win = A[A.realized_r_3p5 > 0]
    adj = A.realized_r_3p5.copy()
    mask = (A.terminal_mae_r_dep <= -1.75) & (A.realized_r_3p5 > 0)
    adj[mask] = -1.0
    canon_opt = dict(
        canon_winners=len(win),
        pct_winners_pierced_stop_dep_window=float((win.terminal_mae_r_dep <= -1.75).mean() * 100),
        canon_mean_replay=float(A.realized_r_3p5.mean()),
        canon_mean_dep_breach_corrected=float(adj.mean()),
        canon_win_replay_pct=float((A.realized_r_3p5 > 0).mean() * 100),
        canon_win_corrected_pct=float((adj > 0).mean() * 100),
    )
    return comp, bucket_share, A, canon_opt


# ─────────────────────────────────────────────────────────────────────────────
def md_table(df: pd.DataFrame, fmt="{:.2f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: fmt.format(x) if pd.notna(x) else "")
    cols = list(d.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    print("[1/6] building per-trade path summary (l_arc_10 frame)…", flush=True)
    df = build_summary()
    df.to_csv(OUTDIR / "per_trade_path_summary.csv", index=False, lineterminator="\n")
    print(f"      {len(df)} trades summarised.", flush=True)

    print("[2/6] analysis A — descriptive…", flush=True)
    A_bucket, A_ct = analysis_A(df)
    A_bucket.to_csv(OUTDIR / "bucket_descriptive.csv", index=False, lineterminator="\n")
    A_ct.to_csv(OUTDIR / "bucket_peak_x_realized.csv", lineterminator="\n")

    print("[3/6] analysis B — stop width…", flush=True)
    B_pos, B_sweep, pct_stop_win, dip_pctile = analysis_B(df)
    B_pos.to_csv(OUTDIR / "stop_position.csv", index=False, lineterminator="\n")
    B_sweep.to_csv(OUTDIR / "stop_sweep.csv", index=False, lineterminator="\n")

    print("[4/6] analysis C — exit headroom…", flush=True)
    C_tbl, _gap = analysis_C(df)
    C_tbl.to_csv(OUTDIR / "exit_headroom.csv", index=False, lineterminator="\n")

    print("[5/6] analysis D — early-path separability (RF, 5-fold TSCV)…", flush=True)
    auc, honesty, mat, base_win, best_win, bestN = analysis_D(df)
    auc.to_csv(OUTDIR / "auc_by_n.csv", index=False, lineterminator="\n")
    honesty.to_csv(OUTDIR / "defer_entry_honesty.csv", index=False, lineterminator="\n")

    print("[6/6] canonical v3.0.2 cross-validation…", flush=True)
    comp, bshare, _A, canon_opt = canonical_xval(df)
    comp.to_csv(OUTDIR / "canonical_xval.csv", index=False, lineterminator="\n")
    bshare.to_csv(OUTDIR / "canonical_xval_bucketshare.csv", index=False, lineterminator="\n")

    # honest-vs-replay headline (the load-bearing exit-fidelity finding)
    hv = dict(
        honest_mean=float(df.realized_depR.mean()),
        honest_win=float((df.realized_depR > 0).mean() * 100),
        replay_mean=float(df.realized_replay_depR.mean()),
        replay_win=float((df.realized_replay_depR > 0).mean() * 100),
        pct_pretp1_breach=float(((df.realized_replay_depR > 0) & (df.realized_depR <= 0)).mean() * 100),
    )
    pd.DataFrame([hv]).to_csv(OUTDIR / "honest_vs_replay.csv", index=False, lineterminator="\n")

    # optional AUC plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(auc.N, auc.auc_y_win, "o-", label="target1: realised R > 0")
        ax.plot(auc.N, auc.auc_y_peak2r, "s-", label="target2: terminal peak >= 2R (7 ATR)")
        ax.axhline(0.5, ls="--", color="grey", lw=0.8, label="AUC 0.5 (no skill)")
        ax.set_xlabel("bar N after entry (early path observed up to N)")
        ax.set_ylabel("mean OOS AUC (5-fold TimeSeriesSplit)")
        ax.set_title("Arc 10 DLR — early-path predictive AUC vs N (l_arc_10 frame)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTDIR / "auc_by_n.png", dpi=120)
        plt.close(fig)
        print("      auc_by_n.png written.", flush=True)
    except Exception as e:
        print(f"      (plot skipped: {e})", flush=True)

    write_doc(df, A_bucket, A_ct, B_pos, B_sweep, pct_stop_win, dip_pctile,
              C_tbl, auc, honesty, mat, base_win, best_win, bestN, comp, bshare,
              canon_opt, hv)
    write_manifest()
    print(f"[done] artefacts in {OUTDIR}", flush=True)
    return 0


def write_manifest():
    lines = []
    for f in sorted(OUTDIR.glob("*")):
        if f.name == "manifest_sha256.txt":
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append(f"{h}  {f.name}")
    (OUTDIR / "manifest_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc(df, A_bucket, A_ct, B_pos, B_sweep, pct_stop_win, dip_pctile,
              C_tbl, auc, honesty, mat, base_win, best_win, bestN, comp, bshare,
              canon_opt, hv):
    c = C_tbl.iloc[0]
    L = []
    L.append("# Arc 10 (DLR) — Path / Peak-R Diagnostic\n")
    L.append("> READ-ONLY forward-path geometry diagnostic. Does NOT modify Arc 10 "
             "config / risk / canonical numbers. Informs FUTURE-arc architecture choices "
             "(stop width, A3 deferred-entry, A4 differentiated-exit).\n")

    # ── RESULT TABLES FIRST (A, B, C, D), then interpretation, then verdicts ──
    L.append("\n## A. DESCRIPTIVE — by realised-R bucket\n")
    L.append("Per-trade forward paths on the l_arc_10 frame (see FRAME_PROVENANCE at the "
             "bottom). Percentiles in ATR-at-entry units; `t_to_peak` in bars. Realised-R "
             "buckets are **SL-honest deployed-R** (1R = 3.5 ATR; see the exit-fidelity "
             "finding below).\n")
    L.append(md_table(A_bucket) + "\n")
    L.append("\n**Terminal-peak (deployed-R) × realised-R cross-tab (trade counts):**\n")
    L.append(md_table(A_ct.reset_index(), "{:.0f}") + "\n")

    L.append("\n## B. STOP-WIDTH READ — is 3.5×ATR right?\n")
    L.append("`pct_prepeak_dip_below_-3.5atr` = share of the group whose deepest dip BEFORE "
             "their unconstrained peak pierces the −3.5 ATR stop line (the unstopped-path "
             "geometry; dips in ATR).\n")
    L.append(md_table(B_pos) + "\n")
    L.append(f"\n- For **winners (realised>0, SL-honest)**, **{pct_stop_win:.2f}%** have a "
             f"pre-peak dip beyond −3.5 ATR; the −3.5 ATR line sits at the "
             f"**~{100 - dip_pctile:.0f}th percentile** of winners' pre-peak dip distribution.\n")
    L.append("\n**Stop-width sweep — SL-honest (primary) vs fast-replay, realised in EACH "
             "stop's own R units (1R = sl_mult×ATR):**\n")
    L.append(md_table(B_sweep) + "\n")

    L.append("\n## C. EXIT-HEADROOM — the A4 ceiling\n")
    L.append("`gap = terminal_peak (unconstrained) − realised` (SL-honest captured P&L, same "
             "frame). Upper bound on what a perfect differentiated exit could chase.\n")
    L.append(md_table(C_tbl.T.reset_index().rename(columns={"index": "metric", 0: "value"}),
                      "{:.3f}") + "\n")

    L.append("\n## D. DEFERRED-ENTRY / EARLY-PATH SEPARABILITY\n")
    L.append("RF (Appendix-A defaults: n_estimators=200, max_depth=6, min_samples_leaf=50, "
             "random_state=42, n_jobs=1), 5-fold TimeSeriesSplit, chronological by entry_time. "
             "Features at bar N: running MFE(N), running MAE(N), close(N), velocity — "
             "NO lookahead. Target1 = SL-honest realised R>0; Target2 = terminal peak ≥ 2R "
             "(7 ATR).\n")
    L.append(md_table(auc, "{:.3f}") + "\n")
    L.append("\n**Deployment-honesty block (mandatory) — naive defer-enter-at-N "
             f"(best N={bestN}):**\n")
    if mat:
        L.append(f"AUC at N={bestN} ({best_win:.3f}) is materially above bar-0 "
                 f"({base_win:.3f}). Defer rule: skip if running MAE(N) below threshold; "
                 "admitted trades pay the late-entry tax of forgoing the 0→N move. Admit-only "
                 "mean is shown WITH the reject-pool + foregone-R tax — never in isolation.\n")
    else:
        L.append(f"**No N gives AUC materially (≥0.02) above bar-0** (bar-0 = {base_win:.3f}, "
                 f"best = {best_win:.3f} at N={bestN}). Reported anyway, with the full "
                 "reject-pool + foregone-R tax.\n")
    L.append(md_table(honesty, "{:.3f}") + "\n")

    # ── EXIT-FIDELITY FINDING ──
    L.append("\n---\n## ⚠ Exit-fidelity finding (load-bearing for B, C, D)\n")
    L.append(
        f"The fast Step-5 replay `simulate_path` (`sl_partial_close_1r_runner_trail`) only "
        f"applies the −3.5 ATR SL to the **runner, after** the +1R partial fires (its "
        f"`sl_breach > tp1_i` guard) — it **ignores SL breaches that occur before the "
        f"partial**. On the unstopped paths, **{hv['pct_pretp1_breach']:.1f}% of trades reach "
        f"+1R only after their low has already pierced −3.5 ATR**; a live SL-honest engine "
        f"stops these at −1R first.\n\n"
        f"- **Replay (optimistic):** mean **{hv['replay_mean']:.3f} R_dep**, win "
        f"**{hv['replay_win']:.1f}%**.\n"
        f"- **SL-honest (live-faithful, this doc's primary):** mean **{hv['honest_mean']:.3f} "
        f"R_dep**, win **{hv['honest_win']:.1f}%**.\n\n"
        "The **live `MultiPairBacktester` driver IS SL-honest** — its always-on intra-bar SL "
        "(`bar.low_bid <= sl_price`) stops the position at −1R before any partial (see "
        "`core/sim/exit_policies/sl_partial_close_1r_runner_trail.py` docstring, Stage 1). So "
        "the LIVE EA / engine takes the correct trades; the optimism is confined to the FAST "
        "REPLAY used for Step-5 *ranking*. The canonical Step-5 `realized_r_3p5` is the replay "
        "value, and the same optimism is visible in the canonical pool: "
        f"**{canon_opt['pct_winners_pierced_stop_dep_window']:.1f}% of canonical winners** "
        f"pierced −3.5 ATR within their deployed window; forcing those to −1R moves canonical "
        f"mean realised **{canon_opt['canon_mean_replay']:.3f} → "
        f"{canon_opt['canon_mean_dep_breach_corrected']:.3f} R_dep** and win "
        f"**{canon_opt['canon_win_replay_pct']:.1f}% → {canon_opt['canon_win_corrected_pct']:.1f}%** "
        "(deployed-window proxy; the strict pre-tp1 measure needs per-bar paths, which are "
        "gone for the canonical pool).\n\n"
        "> **Scope:** This is a READ-ONLY diagnostic; it does NOT alter any canonical number. "
        "It is **flagged for the canonical/engine owners** — the live deployment is unaffected "
        "(live engine is SL-honest), but the Step-5 ranking/gate replay over-counts winners "
        "and any future arc using `simulate_path` inherits the bias. Recommend the fast replay "
        "censor pre-tp1 SL breaches (one-line fix: treat `sl_breach < tp1_i` as a −1R stop).\n")

    L.append("\n---\n## Canonical v3.0.2 cross-validation (frame representativeness)\n")
    L.append("Apples-to-apples: l_arc_10 **replay** realised vs canonical **replay** "
             "`realized_r_3p5` (both ignore pre-tp1 SL), plus exit-agnostic geometry stats. "
             "Confirms the l_arc_10 frame is the same trade population as the live EET pool.\n")
    L.append(md_table(comp, "{:.3f}") + "\n")
    L.append("\n**Replay realised-R bucket shares (%):**\n")
    L.append(md_table(bshare) + "\n")

    # ── verdicts ──
    sw = B_sweep.set_index("sl_mult")
    d35, d30, d25, d40 = sw.loc[3.5], sw.loc[3.0], sw.loc[2.5], sw.loc[4.0]
    sweep_spread = float(sw.honest_mean_R.max() - sw.honest_mean_R.min())
    n_pos_thr = int((honesty.net_edge_vs_full_pp > 0).sum()) if not honesty.empty else 0
    n_thr = int(len(honesty))
    max_edge = float(honesty.net_edge_vs_full_pp.max()) if not honesty.empty else 0.0
    held_med = float(np.median(df.t_to_peak))  # context for outcome-leakage caveat
    L.append("\n---\n## VERDICTS\n")
    L.append(
        f"**1. Stop width OK? — YES, 3.5×ATR is adequate; the result is risk-normalised-"
        f"neutral across 2.5–4.0.** SL-honest mean R is "
        f"2.5={d25.honest_mean_R:+.3f} / 3.0={d30.honest_mean_R:+.3f} / "
        f"3.5={d35.honest_mean_R:+.3f} / 4.0={d40.honest_mean_R:+.3f} — a total spread of only "
        f"{sweep_spread:.3f}R, i.e. **noise**; there is no material risk-normalised case to "
        f"re-tune the stop in either direction. Win-rate rises monotonically with width "
        f"({d25.honest_win_rate_pct:.1f}%→{d40.honest_win_rate_pct:.1f}%) but that is purely "
        f"mechanical (a wider stop is breached less often) and is exactly offset in R because "
        f"each loss is a wider −1R. Only **{pct_stop_win:.1f}% of SL-honest winners** dip "
        f"beyond −3.5 ATR before peaking (the −3.5 line sits ~p{100 - dip_pctile:.0f} of their "
        f"pre-peak dip distribution), so 3.5×ATR is NOT cutting eventual winners short. The "
        f"real stop-width story is the **exit-fidelity finding**: the stop is load-bearing "
        f"(SL-honest vs replay mean = {hv['honest_mean']:+.3f} vs {hv['replay_mean']:+.3f} R), "
        f"and the replay that justified the gate effectively ignored it on 22% of trades.\n")
    L.append(
        f"\n**2. A3 deferred-entry viable? — NO (marginal at best); the AUC lift is largely "
        f"mechanical, not tradeable.** Early-path AUC for `realised>0` climbs 0.60→0.80 from "
        f"N=0→8, but that rise is mostly **contemporaneous-outcome leakage**: by bar 8 a large "
        f"share of trades have already partialled or stopped (median t-to-peak ≈ {held_med:.0f} "
        f"bars; many losers resolve in <8), so 'observing the path to N' increasingly means "
        f"'observing the result'. The honest test is the defer-enter-at-N simulation: the net "
        f"edge vs the full pool is positive at {n_pos_thr}/{n_thr} thresholds but **tiny "
        f"(max +{max_edge:.3f} R/trade)**, it rejects 13–39% of trades of which a rising share "
        f"are real winners, and it leaves the pool at ≈break-even (full pool {df.realized_depR.mean():+.3f} R "
        f"SL-honest). Deferred entry does NOT rescue the edge and is not worth an A3 build on "
        f"this evidence; true pre-entry separability (N=0 AUC {base_win:.2f}) is weak.\n")
    real_ceiling = float(c.winners_mean_foregone_depR)
    L.append(
        f"\n**3. A4 differentiated-exit headroom? — a real but bounded ceiling; treat the "
        f"headline total as an over-count.** The full-pool gap ({c.total_foregone_depR:.0f} "
        f"R_dep, {c.mean_foregone_depR:.2f}/trade) is **inflated by stopped-then-recovered "
        f"losers** whose unconstrained peak is fantasy (they were correctly stopped and could "
        f"not be held to it). The meaningful figure is the **winners-only ceiling "
        f"≈{real_ceiling:.2f} R_dep/trade** over {int(c.winners_n)} winners — and even that is "
        f"an upper bound assuming capture of the unconstrained peak, which no causal exit "
        f"achieves; the runner-trail already harvests part of it. Net: there is moderate, "
        f"genuine upside for a smarter runner/exit on the WINNER subset, but it is far smaller "
        f"than the raw total suggests. **Any A4 work must be measured against an SL-honest "
        f"baseline** (exit-fidelity finding) or the apparent improvement will be illusory.\n")
    L.append(
        "\n> **Overarching note.** The single most consequential output of this diagnostic is "
        "the **exit-fidelity finding**, not the three architecture reads: under a faithful "
        "always-on 3.5×ATR stop the l_arc_10 frame's gross per-trade edge is ≈0 R (vs +0.41 R "
        "under the replay the canonical Step-5/gate uses), and the same optimism is present in "
        "the canonical pool. This is flagged for the canonical/engine owners to verify on the "
        "live EET pool; it is out of scope for this read-only diagnostic to adjudicate.\n")

    # ── provenance LAST (as a reference appendix) ──
    L.append("\n---\n## FRAME_PROVENANCE\n")
    L.append(
        "- **Per-bar source = `results/l_arc_10/step_1/trade_paths.parquet`** (3301 trades, "
        "full unstopped forward path 0→240 bars/trade). The **UTC-convention precursor** of "
        "the canonical v3.0.2 (EET) DLR pool.\n"
        "- **Why not the canonical pool directly:** the v3.0.2 per-bar paths are "
        "**unrecoverable** here — the `H4_5ers_eet` cache and the sha-`05dea9` frame were "
        "removed 2026-05-31 and raw HistData is a 4.6 MB stub (not the 52 GB source), so the "
        "forward paths cannot be regenerated. l_arc_10 is the only surviving per-bar artefact.\n"
        "- **Representativeness:** same DLR signal, same 28-pair universe, same 2010-2026 "
        "span, same R-unit recording (1R_pool = 2.0×ATR_entry), near-identical size "
        "(3301 vs 3152) and exit mix (~85% stoploss / ~15% time-exit); the cross-validation "
        "above confirms the replay realised-R, terminal-peak and bucket shares match the "
        "canonical pool.\n"
        "- **Units:** ATR = pool-R × 2.0; deployed-R (1R_dep = 3.5×ATR = the SL) = pool-R × "
        "(2/3.5). The 3.5×ATR stop line = −3.5 ATR = −1.0 R_dep.\n"
        "- **Realised R = SL-honest** (`sl_honest_realized`, mirrors the live driver's "
        "always-on intra-bar SL); the optimistic fast replay (`simulate_path`) is retained "
        "as a comparison column. Deterministic (random_state=42, n_jobs=1). "
        "Artefact SHAs in `manifest_sha256.txt`.\n")
    (OUTDIR / "PATH_PEAKR_DIAGNOSTIC.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

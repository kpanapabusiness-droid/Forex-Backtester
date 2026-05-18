"""Arc 9 - Step 5 validation experiment (no-filter WFO on cluster 0).

L_ARC_PROTOCOL v2.3 §10 (Step 5 = WFO truth, was Step 6 under v2.2). This is an
EXPERIMENT under a held-open Arc 9, NOT a deployment candidate run. The
question being answered:

  Does the §8 extractability gate correctly identify non-deployable cohorts,
  or did it over-reject a cohort with real economic edge?

Inputs:
  - results/l_arc_9/step2_clustering/clusters_K3.csv (cid=0)
  - results/l_arc_9/step1_verbatim/trades_all.csv (entry info, ATR at signal)
  - 4H pair data (raw OHLC + spread) per configs/wfo_l_arc_9.yaml
  - configs/wfo_kh24.yaml (7-fold OOS windows)

Trade construction:
  - Every cluster 0 trade is taken at its Step 1 entry timestamp (entry at
    next-bar open with bid/ask fill per SPREAD_SEMANTICS_LOCK).
  - Initial SL: entry - 2.0 × ATR (R = selected SL per Step 3, here 2.0×ATR).
  - §11 Stepwise climber Pipeline E exit policy:
      * MFE-lock at 1R: when mfe_so_far >= 1R favorable, move SL to entry
        (break-even MFE-lock).
      * Trail 0.75R from new high: after MFE-lock, SL trails 0.75R behind
        peak_close (close-based trail, bar-close updates only — matches KH-24
        anchor convention §14).
  - Intrabar SL trigger on mid (low <= SL); fill on bid (long).
  - Time exit at bar 240 (40 days at 4H) at next-bar open if no SL hit.

Per-fold accounting:
  - 7-fold WFO OOS windows from KH-24 anchor (configs/wfo_kh24.yaml).
  - For each fold: trades whose entry_time falls in [oos_start, oos_end).
  - Compounded equity from starting balance $10k at risk_pct_per_trade = 0.5%.
    risk_dollars_n = equity_{n-1} * 0.005; pnl_n = risk_dollars_n * final_r_n.
    equity_n = equity_{n-1} + pnl_n.
  - Max DD per fold = max over bars of (peak_equity - trough_equity) / peak * 100,
    computed on the per-trade equity curve within the fold.
  - Per-fold ROI = (equity_end / equity_start - 1) * 100.
  - Annualised ROI per fold = (1 + fold_ROI/100)^(1/years_in_fold) - 1
    (years_in_fold = (oos_end - oos_start).days / 365.25).
  - Full-data ROI = compounded across all folds in order.
  - Full-data DD = max DD over the full equity curve.

Output:
  results/l_arc_9/experiments/step5_validation/
    cluster0_trades_resimulated.csv  - per-trade final_r, exit_reason, exit_time
    per_fold_metrics.csv             - per-fold ROI, DD, trade count, sign
    full_data_metrics.json           - compounded ROI, DD, trade count
    determinism_check.json           - 2-run sha256 comparison
    STEP5_VALIDATION_RESULT.md       - headline + tables + interpretation

Hard rules per dispatch:
  - No admission filter.
  - No engine modification.
  - Real per-bar spread (floor only when raw spread = 0).
  - Risk 0.5%.
  - Determinism: 2-run byte-identical.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.spread_floor import (  # noqa: E402
    STATE_CFG_KEY,
    SpreadFloorState,
    apply_spread_floor_to_pips,
    load_spread_floor,
)

# Locked exit-policy parameters per §11 Stepwise climber Pipeline E.
# 1R = selected SL distance = 2.0 × ATR (cluster 0 Step 3 result).
MFE_LOCK_TRIGGER_R: float = 1.0
TRAIL_DISTANCE_R: float = 0.75
SELECTED_SL_ATR_MULT: float = 2.0  # = 1R for cluster 0
HOLD_BARS: int = 240  # 4H × 240 = 40 days
ENTRY_BAR_OFFSET: int = 1
DIRECTION_INT: int = 1  # long
RISK_PCT: float = 0.005
STARTING_BALANCE: float = 10_000.0

# §10 gate thresholds (v2.3 Step 5 = WFO).
PASS_DEPLOYABLE = {
    "worst_fold_roi_min_pct_annualised": 5.0,
    "mean_fold_roi_min_pct_annualised": 8.0,
    "worst_fold_dd_max_pct": 8.0,
    "all_folds_positive": True,
    "trade_count_per_fold_min": 15,
    "full_data_roi_min_pct_annualised": 5.0,
    "full_data_dd_max_pct": 10.0,
}
PASS_VIABLE = {
    "worst_fold_roi_min_pct_annualised": 0.0,  # > 0%
    "mean_fold_roi_min_pct_annualised": 3.0,
    "worst_fold_dd_max_pct": 8.0,
    "all_folds_positive": True,
    "trade_count_per_fold_min": 5,
    "full_data_roi_min_pct_annualised": 3.0,
    "full_data_dd_max_pct": 10.0,
}


def _load_pair_csv(pair: str, data_dir: Path) -> pd.DataFrame:
    fpath = data_dir / f"{pair}.csv"
    df = pd.read_csv(fpath)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _spread_pips_at_row(pair: str, row: pd.Series, cfg: dict, state: SpreadFloorState) -> float:
    raw_pips: float
    if "spread" in row.index and pd.notna(row["spread"]):
        try:
            points = float(row["spread"])
            divisor = float(state.points_per_pip)
            raw_pips = points / divisor if divisor > 0 and math.isfinite(points) else 0.0
        except Exception:
            raw_pips = 0.0
    else:
        raw_pips = 0.0
    return float(apply_spread_floor_to_pips(cfg, pair, raw_pips))


def _wilder_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    prev = np.empty(n, dtype=float)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


@dataclass
class _ResimTrade:
    trade_id: int
    pair: str
    signal_bar_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    sl_at_entry_price: float
    sl_distance_price: float  # 2.0 * ATR_at_signal
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str  # stoploss | trail_stop | mfe_lock_breakeven | time_exit | end_of_data
    bars_held: int
    mfe_lock_active_at_exit: int  # 0/1
    mfe_lock_activated_at_bar: int  # bar offset, -1 if never
    peak_close_in_R: float  # peak close in baseline-R from entry_fill
    mfe_r: float
    mae_r: float
    final_r: float
    spread_pips_used: float
    spread_pips_exit: float


def _resimulate_trade(
    trade_row: pd.Series,
    df_pair: pd.DataFrame,
    cfg: dict,
    spread_state: SpreadFloorState,
) -> Optional[_ResimTrade]:
    pair = str(trade_row["pair"])
    sig_t = pd.Timestamp(trade_row["signal_bar_time"])
    pip_size = _pip_size(pair)

    dates = df_pair["date"].to_numpy()
    opens = df_pair["open"].astype(float).to_numpy()
    highs = df_pair["high"].astype(float).to_numpy()
    lows = df_pair["low"].astype(float).to_numpy()
    closes = df_pair["close"].astype(float).to_numpy()
    n = len(df_pair)

    idx_arr = np.where(dates == np.datetime64(sig_t))[0]
    if idx_arr.size == 0:
        return None
    sig_idx = int(idx_arr[0])
    entry_idx = sig_idx + ENTRY_BAR_OFFSET
    if entry_idx >= n:
        return None

    atr_at_sig = float(trade_row["atr14_at_signal"])
    if not math.isfinite(atr_at_sig) or atr_at_sig <= 0:
        return None
    sl_distance = SELECTED_SL_ATR_MULT * atr_at_sig

    # Entry fill (long: ask = mid + spread/2 of execution bar).
    entry_row = df_pair.iloc[entry_idx]
    entry_mid = float(opens[entry_idx])
    sp_entry_pips = _spread_pips_at_row(pair, entry_row, cfg, spread_state)
    entry_fill = entry_mid + DIRECTION_INT * (sp_entry_pips * pip_size) / 2.0

    sl_initial_price = entry_fill - DIRECTION_INT * sl_distance
    sl = sl_initial_price
    mfe_lock_active = False
    mfe_lock_activated_at_bar = -1
    peak_close_after_lock = entry_fill  # not used until lock active
    mfe_so_far_price = 0.0
    mae_so_far_price = 0.0
    peak_close_overall = entry_fill

    time_exit_idx = entry_idx + HOLD_BARS
    end_of_data_idx = n - 1
    last_bar_for_path = min(entry_idx + HOLD_BARS, n - 1)

    sl_hit_idx = -1
    sl_hit_price = float("nan")
    exit_reason = ""
    bars_held = 0

    for k in range(entry_idx, last_bar_for_path + 1):
        bar_offset = k - entry_idx
        hk = highs[k]
        lk = lows[k]
        ck = closes[k]

        # Update MFE / MAE in price space (used for telemetry only — SL logic
        # uses high vs entry+1R threshold below).
        cand_mfe_price = hk - entry_fill
        cand_mae_price = entry_fill - lk
        if cand_mfe_price > mfe_so_far_price:
            mfe_so_far_price = cand_mfe_price
        if cand_mae_price > mae_so_far_price:
            mae_so_far_price = cand_mae_price

        # ----- Intrabar SL check first (before any same-bar SL update) -----
        # SL trigger on mid (low <= sl_price). Fill on bid (long): sl - spread/2.
        # We check against the *current* sl (which reflects updates from bar
        # CLOSES — trail updates on bar close per anchor convention).
        # On the entry bar (k == entry_idx) the SL is the initial 2R level;
        # intrabar SL can fire (price moves against entry immediately).
        if k < time_exit_idx and lk <= sl:
            sl_hit_idx = k
            sl_hit_price = sl
            exit_reason = "trail_stop" if mfe_lock_active else "stoploss"
            bars_held = k - entry_idx + 1
            break

        # ----- Bar-close updates for MFE-lock + trail -----
        # MFE-lock trigger: mfe_so_far_r >= 1R favorable (intrabar high reached
        # +1R). We use high (not close) to be consistent with §11 "MFE-lock at
        # 1R" — the lock fires the instant MFE hits 1R, applied at bar close
        # for the next-bar SL.
        if not mfe_lock_active and mfe_so_far_price >= MFE_LOCK_TRIGGER_R * sl_distance:
            mfe_lock_active = True
            mfe_lock_activated_at_bar = bar_offset
            # MFE-lock = move SL to entry (breakeven, no further loss).
            new_sl = entry_fill
            if new_sl > sl:
                sl = new_sl
            peak_close_after_lock = ck

        # Trail update (post-MFE-lock only): trail 0.75R behind peak_close.
        if mfe_lock_active:
            if ck > peak_close_after_lock:
                peak_close_after_lock = ck
            trail_sl = peak_close_after_lock - DIRECTION_INT * TRAIL_DISTANCE_R * sl_distance
            if trail_sl > sl:
                sl = trail_sl

        if ck > peak_close_overall:
            peak_close_overall = ck

        # Time-exit at bar entry+HOLD_BARS (open of that bar; we exit on it).
        if k >= time_exit_idx:
            bars_held = HOLD_BARS
            break

    # Resolve exit fill.
    if sl_hit_idx >= 0:
        hit_row = df_pair.iloc[sl_hit_idx]
        sp_exit_pips = _spread_pips_at_row(pair, hit_row, cfg, spread_state)
        exit_fill = sl_hit_price - DIRECTION_INT * (sp_exit_pips * pip_size) / 2.0
        exit_time = pd.Timestamp(dates[sl_hit_idx])
    elif time_exit_idx <= end_of_data_idx:
        te_row = df_pair.iloc[time_exit_idx]
        sp_exit_pips = _spread_pips_at_row(pair, te_row, cfg, spread_state)
        exit_mid = float(opens[time_exit_idx])
        exit_fill = exit_mid - DIRECTION_INT * (sp_exit_pips * pip_size) / 2.0
        exit_reason = "time_exit"
        exit_time = pd.Timestamp(dates[time_exit_idx])
        bars_held = HOLD_BARS
    else:
        # Ran out of data before hitting SL or time-exit.
        eod_row = df_pair.iloc[end_of_data_idx]
        sp_exit_pips = _spread_pips_at_row(pair, eod_row, cfg, spread_state)
        eod_mid = float(closes[end_of_data_idx])
        exit_fill = eod_mid - DIRECTION_INT * (sp_exit_pips * pip_size) / 2.0
        exit_reason = "end_of_data"
        exit_time = pd.Timestamp(dates[end_of_data_idx])
        bars_held = end_of_data_idx - entry_idx + 1

    final_r = DIRECTION_INT * (exit_fill - entry_fill) / sl_distance
    mfe_r = mfe_so_far_price / sl_distance
    mae_r = -mae_so_far_price / sl_distance
    peak_close_in_R = (peak_close_overall - entry_fill) / sl_distance

    return _ResimTrade(
        trade_id=int(trade_row["trade_id"]),
        pair=pair,
        signal_bar_time=sig_t,
        entry_time=pd.Timestamp(dates[entry_idx]),
        entry_price=entry_fill,
        sl_at_entry_price=sl_initial_price,
        sl_distance_price=sl_distance,
        exit_time=exit_time,
        exit_price=exit_fill,
        exit_reason=exit_reason,
        bars_held=bars_held,
        mfe_lock_active_at_exit=int(mfe_lock_active),
        mfe_lock_activated_at_bar=mfe_lock_activated_at_bar,
        peak_close_in_R=peak_close_in_R,
        mfe_r=mfe_r,
        mae_r=mae_r,
        final_r=final_r,
        spread_pips_used=sp_entry_pips,
        spread_pips_exit=sp_exit_pips,
    )


def _compute_fold_metrics(
    fold_trades: pd.DataFrame, fold_oos_start: pd.Timestamp, fold_oos_end: pd.Timestamp,
) -> Dict[str, Any]:
    n = len(fold_trades)
    if n == 0:
        return {
            "n_trades": 0, "final_r_mean": float("nan"), "final_r_sign_positive": 0,
            "fold_roi_pct": float("nan"), "annualised_roi_pct": float("nan"),
            "max_dd_pct": float("nan"), "ending_equity": STARTING_BALANCE,
        }
    fold_trades = fold_trades.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    equity = [STARTING_BALANCE]
    for _, row in fold_trades.iterrows():
        cur = equity[-1]
        risk_dollars = cur * RISK_PCT
        pnl = risk_dollars * float(row["final_r"])
        equity.append(cur + pnl)
    eq_arr = np.array(equity, dtype=float)
    # Peak-to-trough drawdown on equity curve.
    cummax = np.maximum.accumulate(eq_arr)
    dd = (cummax - eq_arr) / cummax
    max_dd_pct = float(dd.max() * 100.0)
    fold_roi_pct = float((eq_arr[-1] / eq_arr[0] - 1.0) * 100.0)
    days = (fold_oos_end - fold_oos_start).days
    years = max(days / 365.25, 1e-9)
    if eq_arr[-1] > 0:
        ann_factor = (eq_arr[-1] / eq_arr[0]) ** (1.0 / years) - 1.0
        annualised_roi_pct = float(ann_factor * 100.0)
    else:
        annualised_roi_pct = float("-inf")
    fr_mean = float(fold_trades["final_r"].mean())
    sign_positive = int(fr_mean > 0 and fold_roi_pct > 0)
    return {
        "n_trades": int(n),
        "final_r_mean": fr_mean,
        "final_r_sign_positive": sign_positive,
        "fold_roi_pct": fold_roi_pct,
        "annualised_roi_pct": annualised_roi_pct,
        "max_dd_pct": max_dd_pct,
        "ending_equity": float(eq_arr[-1]),
    }


def _full_data_equity(all_trades: pd.DataFrame, fold_starts_ends: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> Dict[str, Any]:
    df = all_trades.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    equity = [STARTING_BALANCE]
    for _, row in df.iterrows():
        cur = equity[-1]
        risk_dollars = cur * RISK_PCT
        pnl = risk_dollars * float(row["final_r"])
        equity.append(cur + pnl)
    eq_arr = np.array(equity, dtype=float)
    cummax = np.maximum.accumulate(eq_arr)
    dd = (cummax - eq_arr) / cummax
    max_dd_pct = float(dd.max() * 100.0)
    roi_pct = float((eq_arr[-1] / eq_arr[0] - 1.0) * 100.0)
    if fold_starts_ends:
        total_days = (fold_starts_ends[-1][1] - fold_starts_ends[0][0]).days
        years = max(total_days / 365.25, 1e-9)
        if eq_arr[-1] > 0:
            ann_pct = ((eq_arr[-1] / eq_arr[0]) ** (1.0 / years) - 1.0) * 100.0
        else:
            ann_pct = float("-inf")
    else:
        ann_pct = float("nan")
    return {
        "n_trades": int(len(df)),
        "full_data_roi_pct": roi_pct,
        "full_data_annualised_roi_pct": ann_pct,
        "full_data_max_dd_pct": max_dd_pct,
        "ending_equity": float(eq_arr[-1]),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    cfg_arc: dict, cfg_kh24: dict, out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    pairs_set = set(cfg_arc["pairs"])
    data_4h = cfg_arc["data"]["data_dirs"]["4H"]
    data_dir = Path(data_4h) if Path(data_4h).is_absolute() else _REPO_ROOT / data_4h
    date_start = str(cfg_arc["data"]["date_start"])
    date_end = str(cfg_arc["data"]["date_end"])

    # Load fold defs from KH-24 anchor.
    folds_yaml = cfg_kh24["wfo"]["folds"]
    fold_oos: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = [
        (int(f["fold"]), pd.Timestamp(f["oos_start"]), pd.Timestamp(f["oos_end"]))
        for f in folds_yaml
    ]
    fold_start_end_only: List[Tuple[pd.Timestamp, pd.Timestamp]] = [
        (s, e) for _, s, e in fold_oos
    ]

    # Load cluster 0 trade IDs + trade details.
    clusters = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step2_clustering" / "clusters_K3.csv")
    cid0_ids = set(clusters[clusters["cluster_id"] == 0]["trade_id"].astype(int))
    trades_all = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    trades_all["entry_time"] = pd.to_datetime(trades_all["entry_time"])
    trades_all["signal_bar_time"] = pd.to_datetime(trades_all["signal_bar_time"])
    trades_c0 = trades_all[trades_all["trade_id"].astype(int).isin(cid0_ids)].reset_index(drop=True)
    print(f"[step5 val] cluster 0 trade pool: {len(trades_c0)} trades")

    # Spread state.
    spread_state = load_spread_floor(cfg_arc)
    cfg_arc[STATE_CFG_KEY] = spread_state
    cfg_arc.setdefault("spreads", {})
    cfg_arc["spreads"].setdefault("points_per_pip", float(spread_state.points_per_pip))

    # Per-pair data cache (slice to arc window).
    pair_cache: Dict[str, pd.DataFrame] = {}
    pairs_used = sorted(trades_c0["pair"].unique().tolist())
    for pair in pairs_used:
        df_raw = _load_pair_csv(pair, data_dir)
        s_ts = pd.Timestamp(date_start)
        e_ts = pd.Timestamp(date_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df_raw[(df_raw["date"] >= s_ts) & (df_raw["date"] <= e_ts)].reset_index(drop=True)
        pair_cache[pair] = df

    # Re-simulate each cluster 0 trade with §11 Stepwise exit.
    resim_rows: List[_ResimTrade] = []
    for _, row in trades_c0.iterrows():
        df_pair = pair_cache[row["pair"]]
        r = _resimulate_trade(row, df_pair, cfg_arc, spread_state)
        if r is not None:
            resim_rows.append(r)
    print(f"[step5 val] re-simulated: {len(resim_rows)} trades")

    rs = pd.DataFrame([{
        "trade_id": r.trade_id,
        "pair": r.pair,
        "signal_bar_time": r.signal_bar_time.strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time": r.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        "entry_price": r.entry_price,
        "sl_at_entry_price": r.sl_at_entry_price,
        "sl_distance_price": r.sl_distance_price,
        "exit_time": r.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_price": r.exit_price,
        "exit_reason": r.exit_reason,
        "bars_held": r.bars_held,
        "mfe_lock_active_at_exit": int(r.mfe_lock_active_at_exit),
        "mfe_lock_activated_at_bar": int(r.mfe_lock_activated_at_bar),
        "peak_close_in_R": r.peak_close_in_R,
        "mfe_r": r.mfe_r,
        "mae_r": r.mae_r,
        "final_r": r.final_r,
        "spread_pips_used": r.spread_pips_used,
        "spread_pips_exit": r.spread_pips_exit,
    } for r in resim_rows])
    rs = rs.sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)
    rs.to_csv(out_dir / "cluster0_trades_resimulated.csv", index=False,
              float_format="%.10g", lineterminator="\n")

    # Per-fold metrics. Filter trades by entry_time within OOS window.
    rs_dt = rs.copy()
    rs_dt["entry_time_dt"] = pd.to_datetime(rs_dt["entry_time"])
    fold_rows: List[Dict[str, Any]] = []
    for (fold_id, s, e) in fold_oos:
        mask = (rs_dt["entry_time_dt"] >= s) & (rs_dt["entry_time_dt"] < e)
        fmetrics = _compute_fold_metrics(rs_dt[mask], s, e)
        fmetrics["fold"] = fold_id
        fmetrics["oos_start"] = s.strftime("%Y-%m-%d")
        fmetrics["oos_end"] = e.strftime("%Y-%m-%d")
        fold_rows.append(fmetrics)
    fold_df = pd.DataFrame(fold_rows)[[
        "fold", "oos_start", "oos_end", "n_trades", "final_r_mean",
        "final_r_sign_positive", "fold_roi_pct", "annualised_roi_pct",
        "max_dd_pct", "ending_equity",
    ]]
    fold_df.to_csv(out_dir / "per_fold_metrics.csv", index=False,
                   float_format="%.10g", lineterminator="\n")

    # Full-data (across all folds in time order).
    rs_in_window = rs_dt[
        (rs_dt["entry_time_dt"] >= fold_start_end_only[0][0]) &
        (rs_dt["entry_time_dt"] < fold_start_end_only[-1][1])
    ]
    full_metrics = _full_data_equity(rs_in_window, fold_start_end_only)
    (out_dir / "full_data_metrics.json").write_text(
        json.dumps(full_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return rs, fold_df, full_metrics, fold_start_end_only


def evaluate_gates(
    fold_df: pd.DataFrame, full_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if fold_df["n_trades"].sum() == 0:
        return {"pass_deployable": False, "pass_viable": False, "details": {}}
    worst_ann = float(fold_df["annualised_roi_pct"].min())
    mean_ann = float(fold_df["annualised_roi_pct"].mean())
    worst_dd = float(fold_df["max_dd_pct"].max())
    min_trades = int(fold_df["n_trades"].min())
    all_pos = bool((fold_df["fold_roi_pct"] > 0).all())
    full_ann = float(full_metrics["full_data_annualised_roi_pct"])
    full_dd = float(full_metrics["full_data_max_dd_pct"])

    def check(thr_dict):
        c = {
            "worst_fold_ann_roi": (worst_ann, thr_dict["worst_fold_roi_min_pct_annualised"], worst_ann >= thr_dict["worst_fold_roi_min_pct_annualised"]),
            "mean_fold_ann_roi": (mean_ann, thr_dict["mean_fold_roi_min_pct_annualised"], mean_ann >= thr_dict["mean_fold_roi_min_pct_annualised"]),
            "worst_fold_dd_pct": (worst_dd, thr_dict["worst_fold_dd_max_pct"], worst_dd <= thr_dict["worst_fold_dd_max_pct"]),
            "all_folds_positive": (all_pos, thr_dict["all_folds_positive"], all_pos == thr_dict["all_folds_positive"]),
            "trade_count_per_fold_min": (min_trades, thr_dict["trade_count_per_fold_min"], min_trades >= thr_dict["trade_count_per_fold_min"]),
            "full_data_ann_roi": (full_ann, thr_dict["full_data_roi_min_pct_annualised"], full_ann >= thr_dict["full_data_roi_min_pct_annualised"]),
            "full_data_dd_pct": (full_dd, thr_dict["full_data_dd_max_pct"], full_dd <= thr_dict["full_data_dd_max_pct"]),
        }
        all_pass = all(v[2] for v in c.values())
        return c, all_pass

    pd_checks, pd_pass = check(PASS_DEPLOYABLE)
    pv_checks, pv_pass = check(PASS_VIABLE)
    return {
        "pass_deployable": pd_pass,
        "pass_viable": pv_pass,
        "pd_checks": pd_checks,
        "pv_checks": pv_checks,
        "summary": {
            "worst_fold_ann_roi_pct": worst_ann,
            "mean_fold_ann_roi_pct": mean_ann,
            "worst_fold_max_dd_pct": worst_dd,
            "min_trades_per_fold": min_trades,
            "all_folds_positive": all_pos,
            "full_data_ann_roi_pct": full_ann,
            "full_data_max_dd_pct": full_dd,
        },
    }


def write_report(
    fold_df: pd.DataFrame, full_metrics: Dict[str, Any], gates: Dict[str, Any],
    out_path: Path,
) -> None:
    if gates["pass_deployable"]:
        headline = "PASS-DEPLOYABLE"
    elif gates["pass_viable"]:
        headline = "PASS-VIABLE"
    else:
        headline = "FAIL"
    s = gates["summary"]
    lines: List[str] = []
    lines.append("# Arc 9 Step 5 Validation - no-filter WFO on cluster 0")
    lines.append("")
    lines.append("> Held-open experiment under v2.3 §10 (Step 5 = WFO truth). NOT a")
    lines.append("> deployment candidate run. Question: does the §8 extractability gate")
    lines.append("> correctly identify non-deployable cohorts, or did it over-reject a")
    lines.append("> cohort with real economic edge?")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"**{headline}** — worst-fold annualised ROI {s['worst_fold_ann_roi_pct']:.2f}%, "
        f"mean fold annualised ROI {s['mean_fold_ann_roi_pct']:.2f}%, "
        f"worst-fold max DD {s['worst_fold_max_dd_pct']:.2f}%."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Trade pool: 365 cluster 0 trades from Step 1 / Step 2 (cid=0 at K=3).")
    lines.append("- Entry: every cluster 0 signal taken at signal-bar-close → next-bar open (long).")
    lines.append("- SL: entry - 2.0×ATR (R = selected SL per Step 3, here 2.0×ATR).")
    lines.append("- Exit policy: §11 Stepwise climber Pipeline E.")
    lines.append("    * MFE-lock at 1R: when intrabar mfe ≥ 1R favorable, move SL to entry (break-even).")
    lines.append("    * Trail 0.75R from peak_close after MFE-lock activates (bar-close updates).")
    lines.append("- Intrabar SL trigger on mid (low ≤ SL), fill on bid (long): SL - spread/2 of trigger bar.")
    lines.append("- Time exit at bar entry+240 (40 calendar days at 4H), fill at open of bar 240 minus spread/2.")
    lines.append("- Risk per trade: 0.5% of compounded equity (starting $10k).")
    lines.append("- Folds: 7-fold WFO OOS windows from `configs/wfo_kh24.yaml` (anchor).")
    lines.append("- Per-bar spread: real MT5 spread, floor only when raw spread = 0 (SPREAD_SEMANTICS_LOCK).")
    lines.append("")
    lines.append("## Per-fold table")
    lines.append("")
    lines.append("| Fold | OOS window | Trades | Final R mean | Sign | Compounded ROI (%) | Annualised ROI (%) | Max DD (%) | Ending equity |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in fold_df.iterrows():
        sign_str = "+" if r["final_r_sign_positive"] else "-"
        lines.append(
            f"| F{int(r['fold'])} | {r['oos_start']} → {r['oos_end']} | "
            f"{int(r['n_trades'])} | {r['final_r_mean']:+.3f} | {sign_str} | "
            f"{r['fold_roi_pct']:+.2f} | {r['annualised_roi_pct']:+.2f} | "
            f"{r['max_dd_pct']:.2f} | ${r['ending_equity']:,.2f} |"
        )
    lines.append("")
    lines.append("## Full-data (compounded across folds in time order)")
    lines.append("")
    lines.append(f"- n trades: {full_metrics['n_trades']}")
    lines.append(f"- Compounded ROI: {full_metrics['full_data_roi_pct']:+.2f}%")
    lines.append(f"- Annualised ROI: {full_metrics['full_data_annualised_roi_pct']:+.2f}%")
    lines.append(f"- Max DD: {full_metrics['full_data_max_dd_pct']:.2f}%")
    lines.append(f"- Ending equity: ${full_metrics['ending_equity']:,.2f}")
    lines.append("")
    lines.append("## Gates evaluated (§10 pass-deployable + pass-viable)")
    lines.append("")
    lines.append("### Pass-deployable")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Pass? |")
    lines.append("|---|---|---|---|")
    pdc = gates["pd_checks"]
    lines.append(f"| Worst-fold annualised ROI | ≥ {PASS_DEPLOYABLE['worst_fold_roi_min_pct_annualised']}% | {pdc['worst_fold_ann_roi'][0]:+.2f}% | {'PASS' if pdc['worst_fold_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Mean fold annualised ROI | ≥ {PASS_DEPLOYABLE['mean_fold_roi_min_pct_annualised']}% | {pdc['mean_fold_ann_roi'][0]:+.2f}% | {'PASS' if pdc['mean_fold_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Worst-fold max DD | ≤ {PASS_DEPLOYABLE['worst_fold_dd_max_pct']}% | {pdc['worst_fold_dd_pct'][0]:.2f}% | {'PASS' if pdc['worst_fold_dd_pct'][2] else 'FAIL'} |")
    lines.append(f"| All folds positive | required | {pdc['all_folds_positive'][0]} | {'PASS' if pdc['all_folds_positive'][2] else 'FAIL'} |")
    lines.append(f"| Trade count per fold | ≥ {PASS_DEPLOYABLE['trade_count_per_fold_min']} | min {pdc['trade_count_per_fold_min'][0]} | {'PASS' if pdc['trade_count_per_fold_min'][2] else 'FAIL'} |")
    lines.append(f"| Full-data annualised ROI | ≥ {PASS_DEPLOYABLE['full_data_roi_min_pct_annualised']}% | {pdc['full_data_ann_roi'][0]:+.2f}% | {'PASS' if pdc['full_data_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Full-data max DD | ≤ {PASS_DEPLOYABLE['full_data_dd_max_pct']}% | {pdc['full_data_dd_pct'][0]:.2f}% | {'PASS' if pdc['full_data_dd_pct'][2] else 'FAIL'} |")
    lines.append(f"| **Overall pass-deployable** | all 7 | - | **{'PASS' if gates['pass_deployable'] else 'FAIL'}** |")
    lines.append("")
    lines.append("### Pass-viable")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Pass? |")
    lines.append("|---|---|---|---|")
    pvc = gates["pv_checks"]
    lines.append(f"| Worst-fold annualised ROI | > {PASS_VIABLE['worst_fold_roi_min_pct_annualised']}% (positive) | {pvc['worst_fold_ann_roi'][0]:+.2f}% | {'PASS' if pvc['worst_fold_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Mean fold annualised ROI | ≥ {PASS_VIABLE['mean_fold_roi_min_pct_annualised']}% | {pvc['mean_fold_ann_roi'][0]:+.2f}% | {'PASS' if pvc['mean_fold_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Worst-fold max DD | ≤ {PASS_VIABLE['worst_fold_dd_max_pct']}% | {pvc['worst_fold_dd_pct'][0]:.2f}% | {'PASS' if pvc['worst_fold_dd_pct'][2] else 'FAIL'} |")
    lines.append(f"| All folds positive | required | {pvc['all_folds_positive'][0]} | {'PASS' if pvc['all_folds_positive'][2] else 'FAIL'} |")
    lines.append(f"| Trade count per fold | ≥ {PASS_VIABLE['trade_count_per_fold_min']} | min {pvc['trade_count_per_fold_min'][0]} | {'PASS' if pvc['trade_count_per_fold_min'][2] else 'FAIL'} |")
    lines.append(f"| Full-data annualised ROI | ≥ {PASS_VIABLE['full_data_roi_min_pct_annualised']}% | {pvc['full_data_ann_roi'][0]:+.2f}% | {'PASS' if pvc['full_data_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Full-data max DD | ≤ {PASS_VIABLE['full_data_dd_max_pct']}% | {pvc['full_data_dd_pct'][0]:.2f}% | {'PASS' if pvc['full_data_dd_pct'][2] else 'FAIL'} |")
    lines.append(f"| **Overall pass-viable** | all 7 | - | **{'PASS' if gates['pass_viable'] else 'FAIL'}** |")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc 9 Step 5 validation - no-filter WFO on cluster 0.")
    p.add_argument("--arc-cfg", type=Path, default=_REPO_ROOT / "configs" / "wfo_l_arc_9.yaml")
    p.add_argument("--kh24-cfg", type=Path, default=_REPO_ROOT / "configs" / "wfo_kh24.yaml")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_validation")
    p.add_argument("--verify-determinism", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_arc = yaml.safe_load(args.arc_cfg.read_text(encoding="utf-8"))
    cfg_kh24 = yaml.safe_load(args.kh24_cfg.read_text(encoding="utf-8"))

    rs1, fold_df1, full_m1, fold_se = run(cfg_arc, cfg_kh24, args.out_dir)
    gates = evaluate_gates(fold_df1, full_m1)

    if args.verify_determinism:
        # Re-run from fresh configs.
        cfg_arc2 = yaml.safe_load(args.arc_cfg.read_text(encoding="utf-8"))
        cfg_kh24_2 = yaml.safe_load(args.kh24_cfg.read_text(encoding="utf-8"))
        # Run into a temp scratch dir to compare sha256 without overwriting.
        scratch = args.out_dir / "_determinism_scratch"
        scratch.mkdir(exist_ok=True)
        rs2, fold_df2, full_m2, _ = run(cfg_arc2, cfg_kh24_2, scratch)
        sha1_trades = _sha256_file(args.out_dir / "cluster0_trades_resimulated.csv")
        sha2_trades = _sha256_file(scratch / "cluster0_trades_resimulated.csv")
        sha1_folds = _sha256_file(args.out_dir / "per_fold_metrics.csv")
        sha2_folds = _sha256_file(scratch / "per_fold_metrics.csv")
        det = {
            "trades_run1_sha256": sha1_trades,
            "trades_run2_sha256": sha2_trades,
            "folds_run1_sha256": sha1_folds,
            "folds_run2_sha256": sha2_folds,
            "byte_identical": bool(sha1_trades == sha2_trades and sha1_folds == sha2_folds),
        }
        (args.out_dir / "determinism_check.json").write_text(
            json.dumps(det, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Cleanup scratch.
        for f in scratch.iterdir():
            f.unlink()
        scratch.rmdir()
        print(f"[step5 val] determinism: {'PASS' if det['byte_identical'] else 'FAIL'}")

    write_report(fold_df1, full_m1, gates, args.out_dir / "STEP5_VALIDATION_RESULT.md")

    # Console headline.
    if gates["pass_deployable"]:
        head = "PASS-DEPLOYABLE"
    elif gates["pass_viable"]:
        head = "PASS-VIABLE"
    else:
        head = "FAIL"
    s = gates["summary"]
    print(f"[step5 val] HEADLINE: {head}")
    print(f"  worst-fold ann ROI: {s['worst_fold_ann_roi_pct']:+.2f}%  "
          f"mean-fold ann ROI: {s['mean_fold_ann_roi_pct']:+.2f}%  "
          f"worst-fold DD: {s['worst_fold_max_dd_pct']:.2f}%")
    print(f"  full-data ann ROI: {s['full_data_ann_roi_pct']:+.2f}%  "
          f"full-data DD: {s['full_data_max_dd_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

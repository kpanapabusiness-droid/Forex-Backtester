"""Phase E — shadow trade-sets (op spec §5.11).

For each shadow config, re-simulate per-trade outcomes on the same step 1 taken
signal-set, varying one axis of execution at a time:
  - Entry delays: bar_offset ∈ {1, 2, 3, 5, 10}      (1 verbatim baseline)
  - SL distances: sl_atr_mult ∈ {1.0, 1.5, 2.0, 2.5, 3.0}  (2.0 verbatim)
  - Time exits:   h ∈ {1, 3, 6, 12, 24, 48, 120, 240}  (120 verbatim)

Engine-parity simulator: long-direction only; intrabar SL scan; entry/exit fills
include half-spread; spread floored per `configs/spread_floors_5ers.yaml`.
For entry-delay shadows: drop trades whose delayed entry would push past the
verbatim time-exit bar. Report dropped fraction.
"""
# ruff: noqa: E402, E701, E702, F841, I001
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_3.step2._io import (
    ENTRY_DELAYS,
    PAIRS,
    POINTS_PER_PIP,
    SL_DISTANCES,
    SPREAD_FLOOR_PATH,
    STEP2_DIR,
    TIME_EXIT_H,
    VERBATIM_SL_ATR_MULT,
    VERBATIM_TIME_EXIT_H,
    load_all_floors,
    load_pair_1h,
    load_trades_verbatim,
    pip_size,
)

SHADOW_DIR = STEP2_DIR / "shadow_tradesets"


def _floor_spread_pips(raw_pips: float, floor_pips: float) -> float:
    if not np.isfinite(raw_pips):
        raw_pips = 0.0
    if floor_pips is None or not np.isfinite(floor_pips):
        return raw_pips
    return max(raw_pips, floor_pips)


def _simulate_trade(open_arr: np.ndarray, high_arr: np.ndarray, low_arr: np.ndarray,
                    spread_arr: np.ndarray,
                    signal_idx: int, atr_at_sig: float,
                    bar_offset: int, sl_atr_mult: float, h_exit: int,
                    pair_pip: float, floor_pips: float) -> dict:
    """Long-direction shadow trade. Returns dict of outcomes.

    Engine-parity convention (matches core engine):
      - entry_fill = open[signal_idx + bar_offset] + spread_pips_entry * pip / 2
      - sl_price   = entry_fill - sl_atr_mult * atr_at_sig
      - Time-exit at bar (signal_idx + bar_offset + h_exit) open with that bar's spread.
      - Intrabar SL check at each held bar (low <= sl_price).
    """
    n = len(open_arr)
    e = signal_idx + bar_offset
    x_time = e + h_exit
    if x_time >= n:
        return {"valid": False, "drop_reason": "no_exit_bar"}
    raw_sp_e = float(spread_arr[e]) / POINTS_PER_PIP if np.isfinite(spread_arr[e]) else 0.0
    sp_e = _floor_spread_pips(raw_sp_e, floor_pips)
    entry_mid = float(open_arr[e])
    entry_fill = entry_mid + (sp_e * pair_pip) / 2.0
    sl_distance_price = sl_atr_mult * atr_at_sig
    sl_price = entry_fill - sl_distance_price

    sl_hit_idx = -1
    for k in range(e, x_time):
        if low_arr[k] <= sl_price:
            sl_hit_idx = k
            break

    if sl_hit_idx >= 0:
        raw_sp_x = float(spread_arr[sl_hit_idx]) / POINTS_PER_PIP if np.isfinite(spread_arr[sl_hit_idx]) else 0.0
        sp_x = _floor_spread_pips(raw_sp_x, floor_pips)
        exit_fill = sl_price - (sp_x * pair_pip) / 2.0
        exit_reason = "sl_hit"
        exit_bar = sl_hit_idx
    else:
        raw_sp_x = float(spread_arr[x_time]) / POINTS_PER_PIP if np.isfinite(spread_arr[x_time]) else 0.0
        sp_x = _floor_spread_pips(raw_sp_x, floor_pips)
        exit_mid = float(open_arr[x_time])
        exit_fill = exit_mid - (sp_x * pair_pip) / 2.0
        exit_reason = "time_exit"
        exit_bar = x_time

    price_pnl = exit_fill - entry_fill
    R = price_pnl / sl_distance_price
    spread_cost_price = (sp_e + sp_x) * pair_pip / 2.0
    spread_cost_R = spread_cost_price / sl_distance_price
    gross_r = R + spread_cost_R
    spread_floored = (sp_e > raw_sp_e) or (sp_x > raw_sp_x)
    return {
        "valid": True, "entry_bar_idx": e, "exit_bar_idx": exit_bar,
        "entry_fill": entry_fill, "exit_fill": exit_fill,
        "sl_price": sl_price, "sl_distance_price": sl_distance_price,
        "exit_reason": exit_reason, "R": R, "net_r": R, "gross_r": gross_r,
        "spread_cost_R": spread_cost_R,
        "spread_pips_entry": sp_e, "spread_pips_exit": sp_x,
        "spread_floored": bool(spread_floored), "bars_held": exit_bar - e,
    }


def _run_shadow_config(pair_arrs: Dict[str, dict], pair_ts_idx: Dict[str, Dict[int, int]],
                       floors: Dict[str, float],
                       trades: pd.DataFrame, bar_offset: int, sl_atr_mult: float,
                       h_exit: int) -> tuple[pd.DataFrame, int]:
    """Returns (trade-level DataFrame, n_dropped due to no_exit_bar/entry_past_verbatim_exit)."""
    rows = []
    n_dropped = 0
    sig_ts = pd.to_datetime(trades["signal_bar_ts"]).astype("int64").to_numpy()
    pairs = trades["pair"].to_numpy()
    folds = trades["fold_id"].to_numpy()
    tids = trades["trade_id"].to_numpy()
    atrs = trades["atr_1h_wilder_at_signal"].astype(float).to_numpy()
    for k in range(len(trades)):
        pair = str(pairs[k])
        d = pair_arrs[pair]
        sig_idx = pair_ts_idx[pair].get(int(sig_ts[k]))
        if sig_idx is None:
            n_dropped += 1; continue
        atr_at_sig = float(atrs[k])
        if not np.isfinite(atr_at_sig) or atr_at_sig <= 0:
            n_dropped += 1; continue
        # For entry-delay shadows, drop if delay pushes entry past verbatim time-exit bar
        if bar_offset > 1:
            verbatim_exit = sig_idx + 1 + VERBATIM_TIME_EXIT_H
            if sig_idx + bar_offset >= verbatim_exit:
                n_dropped += 1; continue
        pair_pip = pip_size(pair)
        floor_pips = floors.get(pair, None)
        out = _simulate_trade(d["open"], d["high"], d["low"], d["spread"],
                              sig_idx, atr_at_sig, bar_offset, sl_atr_mult, h_exit,
                              pair_pip, floor_pips)
        if not out["valid"]:
            n_dropped += 1; continue
        rows.append({"trade_id": int(tids[k]), "fold_id": int(folds[k]), "pair": pair, **out})
    return pd.DataFrame(rows), n_dropped


def _summarise(df: pd.DataFrame, shadow_label: str, n_dropped: int = 0) -> dict:
    n = len(df)
    if n == 0:
        return {"shadow": shadow_label, "n": 0, "n_dropped": int(n_dropped)}
    exit_mix = df["exit_reason"].value_counts(normalize=True).to_dict()
    out = {
        "shadow": shadow_label, "n": int(n), "n_dropped": int(n_dropped),
        "mean_net_r": float(df["net_r"].mean()),
        "median_net_r": float(df["net_r"].median()),
        "p5_net_r": float(df["net_r"].quantile(0.05)),
        "p95_net_r": float(df["net_r"].quantile(0.95)),
        "win_pct": float((df["net_r"] > 0).mean() * 100.0),
        "frac_sl_hit": float((df["exit_reason"] == "sl_hit").mean()),
        "frac_time_exit": float((df["exit_reason"] == "time_exit").mean()),
        "frac_floored": float((df["spread_floored"]).mean()),
        "mean_bars_held": float(df["bars_held"].mean()),
    }
    # Per-fold mean R
    for fid in sorted(df["fold_id"].unique()):
        out[f"fold{int(fid)}_mean_r"] = float(df[df["fold_id"] == fid]["net_r"].mean())
        out[f"fold{int(fid)}_n"] = int((df["fold_id"] == fid).sum())
    return out


def run_phase_e() -> None:
    t0 = time.time()
    print("[Phase E] loading data...")
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    (SHADOW_DIR / "entry_delay").mkdir(parents=True, exist_ok=True)
    (SHADOW_DIR / "sl_distance").mkdir(parents=True, exist_ok=True)
    (SHADOW_DIR / "time_exit").mkdir(parents=True, exist_ok=True)

    trades = load_trades_verbatim()
    floors = load_all_floors(SPREAD_FLOOR_PATH)
    pair_arrs: Dict[str, dict] = {}
    pair_ts_idx: Dict[str, Dict[int, int]] = {}
    for pair in PAIRS:
        df = load_pair_1h(pair)
        pair_arrs[pair] = {
            "open": df["open"].astype(float).values,
            "high": df["high"].astype(float).values,
            "low": df["low"].astype(float).values,
            "close": df["close"].astype(float).values,
            "spread": df["spread"].astype(float).values if "spread" in df.columns else np.zeros(len(df)),
        }
        ts_int = df["time"].astype("int64").to_numpy()
        pair_ts_idx[pair] = {int(t): i for i, t in enumerate(ts_int)}

    summaries: List[dict] = []

    # Entry-delay sweep (bar_offset varies; SL=2.0, h=120 verbatim)
    print("[Phase E] entry-delay sweep (SL=2.0 ATR, h=120 verbatim)...")
    entry_curve_rows = []
    for bo in ENTRY_DELAYS:
        t_s = time.time()
        df, n_dropped = _run_shadow_config(pair_arrs, pair_ts_idx, floors, trades,
                                bar_offset=bo, sl_atr_mult=VERBATIM_SL_ATR_MULT,
                                h_exit=VERBATIM_TIME_EXIT_H)
        label = f"entry_delay__n{bo}_sl{VERBATIM_SL_ATR_MULT}_h{VERBATIM_TIME_EXIT_H}"
        df.to_csv(SHADOW_DIR / "entry_delay" / f"n{bo}.csv",
                  index=False, lineterminator="\n", float_format="%.6g")
        s = _summarise(df, label, n_dropped)
        s["dropped_frac"] = (n_dropped / len(trades)) if len(trades) else 0.0
        summaries.append(s)
        row = {"bar_offset": bo, **{k: v for k, v in s.items() if k != "shadow"}}
        entry_curve_rows.append(row)
        print(f"  bo={bo}: n={s.get('n', 0)} dropped={n_dropped} mean_R={s.get('mean_net_r', 0):.5f} "
              f"sl%={s.get('frac_sl_hit', 0)*100:.2f} ({time.time()-t_s:.1f}s)")
    pd.DataFrame(entry_curve_rows).to_csv(SHADOW_DIR / "entry_delay_curve.csv",
                                          index=False, lineterminator="\n")

    # SL-distance sweep (bar_offset=1, sl varies, h=120 verbatim)
    print("[Phase E] SL-distance sweep (bo=1, h=120 verbatim)...")
    sl_rows = []
    for mult in SL_DISTANCES:
        t_s = time.time()
        df, n_dropped = _run_shadow_config(pair_arrs, pair_ts_idx, floors, trades,
                                bar_offset=1, sl_atr_mult=mult, h_exit=VERBATIM_TIME_EXIT_H)
        label = f"sl_distance__n1_sl{mult}_h{VERBATIM_TIME_EXIT_H}"
        df.to_csv(SHADOW_DIR / "sl_distance" / f"sl{mult}.csv",
                  index=False, lineterminator="\n", float_format="%.6g")
        s = _summarise(df, label, n_dropped)
        summaries.append(s)
        sl_rows.append({"sl_atr_mult": mult, **{k: v for k, v in s.items() if k != "shadow"}})
        print(f"  sl={mult}: n={s.get('n', 0)} mean_R={s.get('mean_net_r', 0):.5f} "
              f"sl%={s.get('frac_sl_hit', 0)*100:.2f} ({time.time()-t_s:.1f}s)")
    pd.DataFrame(sl_rows).to_csv(SHADOW_DIR / "sl_distance_sweep.csv",
                                 index=False, lineterminator="\n")

    # Time-exit sweep (bo=1, sl=2.0, h varies)
    print("[Phase E] time-exit sweep (bo=1, SL=2.0 ATR)...")
    te_rows = []
    features_path = STEP2_DIR / "signals_features.csv"
    for h in TIME_EXIT_H:
        t_s = time.time()
        df, n_dropped = _run_shadow_config(pair_arrs, pair_ts_idx, floors, trades,
                                bar_offset=1, sl_atr_mult=VERBATIM_SL_ATR_MULT, h_exit=h)
        label = f"time_exit__n1_sl{VERBATIM_SL_ATR_MULT}_h{h}"
        df.to_csv(SHADOW_DIR / "time_exit" / f"h{h}.csv",
                  index=False, lineterminator="\n", float_format="%.6g")
        s = _summarise(df, label, n_dropped)
        summaries.append(s)
        # capture ratio vs fwd_mfe at the same h
        median_cap = np.nan
        if len(df):
            col = f"fwd_mfe_h{h}_atr"
            try:
                f_h = pd.read_csv(features_path, usecols=["trade_id", col])
                m = df.merge(f_h, on="trade_id", how="left")
                denom = m[col] / VERBATIM_SL_ATR_MULT
                cap = m["R"] / denom.replace(0, np.nan)
                median_cap = float(cap.median(skipna=True))
            except Exception:
                pass
        te_rows.append({"h": h, "median_capture_ratio_vs_fwd_mfe": median_cap,
                        **{k: v for k, v in s.items() if k != "shadow"}})
        print(f"  h={h}: n={s.get('n', 0)} mean_R={s.get('mean_net_r', 0):.5f} "
              f"sl%={s.get('frac_sl_hit', 0)*100:.2f} cap_med={median_cap:.3f} ({time.time()-t_s:.1f}s)")
    pd.DataFrame(te_rows).to_csv(SHADOW_DIR / "time_exit_curve.csv",
                                 index=False, lineterminator="\n")

    pd.DataFrame(summaries).to_csv(SHADOW_DIR / "shadow_summary.csv",
                                   index=False, lineterminator="\n")
    print(f"[Phase E] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run_phase_e()

# ruff: noqa: E402  (sys.path.insert needed before project imports)
"""Phase E — shadow trade-sets (op spec §5.11).

For each shadow config, re-simulate per-trade outcomes on the same signal-set,
varying one axis of execution at a time:
  - Entry delays: bar_offset ∈ {1, 2, 3, 5, 10}      (1 is verbatim baseline)
  - SL distances: sl_atr_mult ∈ {1.0, 1.5, 2.0, 2.5, 3.0}  (2.0 is baseline)
  - Time exits:   h ∈ {1, 3, 6, 12, 24, 48, 120, 240}  (1 is baseline)

The signal evaluation is unchanged across shadows; only execution differs.
We compute R directly (= price_pnl / sl_distance_price - spread_cost_R) since
R is scale-invariant under risk percent.

Spread handling: per-bar `spread` column from the pair's 1H data, divided by
points_per_pip=10 to get pips, floored at the locked per-pair floor from
configs/spread_floors_5ers.yaml. Engine-parity.

Outputs (under step2_descriptive/shadow_tradesets/):
  - shadow_summary.csv               (one row per shadow config; all 17 nominal configs)
  - entry_delay/<bar_offset>.csv     (trade-level)
  - sl_distance/<mult>.csv           (trade-level)
  - time_exit/<h>.csv                (trade-level)
  - entry_delay_curve.csv
  - sl_distance_sweep.csv
  - time_exit_curve.csv
"""

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

from scripts.l_arc_1.step2._io import (
    ENTRY_DELAYS,
    PAIRS,
    SL_DISTANCES,
    STEP2_DIR,
    TIME_EXIT_H,
    load_all_floors,
    load_pair_1h,
    load_trades_verbatim,
    pip_size,
)

SHADOW_DIR = STEP2_DIR / "shadow_tradesets"

POINTS_PER_PIP = 10.0
SPREAD_FLOOR_PATH = REPO_ROOT / "configs" / "spread_floors_5ers.yaml"


def _floor_spread_pips(raw_pips: float, floor_pips: float) -> float:
    if not np.isfinite(raw_pips):
        raw_pips = 0.0
    if floor_pips is None or not np.isfinite(floor_pips):
        return raw_pips
    return max(raw_pips, floor_pips)


def _simulate_trade(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    spread_arr: np.ndarray,
    signal_idx: int,
    atr_at_sig: float,
    bar_offset: int,
    sl_atr_mult: float,
    h_exit: int,
    pair_pip: float,
    floor_pips: float,
) -> dict:
    """Long-direction shadow trade. Returns dict of outcomes.

    Conventions match `core/signals/l4_univariate_extreme._execute_signals`:
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

    # Intrabar SL scan over [e .. x_time - 1] inclusive (held bars).
    sl_hit_idx = -1
    for k in range(e, x_time):
        if low_arr[k] <= sl_price:
            sl_hit_idx = k
            break

    if sl_hit_idx >= 0:
        raw_sp_x = (
            float(spread_arr[sl_hit_idx]) / POINTS_PER_PIP
            if np.isfinite(spread_arr[sl_hit_idx])
            else 0.0
        )
        sp_x = _floor_spread_pips(raw_sp_x, floor_pips)
        exit_fill = sl_price - (sp_x * pair_pip) / 2.0
        exit_reason = "stop_loss"
        exit_bar = sl_hit_idx
    else:
        raw_sp_x = (
            float(spread_arr[x_time]) / POINTS_PER_PIP if np.isfinite(spread_arr[x_time]) else 0.0
        )
        sp_x = _floor_spread_pips(raw_sp_x, floor_pips)
        exit_mid = float(open_arr[x_time])
        exit_fill = exit_mid - (sp_x * pair_pip) / 2.0
        exit_reason = "time_exit"
        exit_bar = x_time

    price_pnl = exit_fill - entry_fill  # long
    R = price_pnl / sl_distance_price
    spread_cost_price = (sp_e + sp_x) * pair_pip / 2.0
    spread_cost_R = spread_cost_price / sl_distance_price
    gross_r = R + spread_cost_R
    spread_floored = (sp_e > raw_sp_e) or (sp_x > raw_sp_x)
    return {
        "valid": True,
        "entry_bar_idx": e,
        "exit_bar_idx": exit_bar,
        "entry_fill": entry_fill,
        "exit_fill": exit_fill,
        "sl_price": sl_price,
        "sl_distance_price": sl_distance_price,
        "exit_reason": exit_reason,
        "R": R,
        "net_r": R,
        "gross_r": gross_r,
        "spread_cost_R": spread_cost_R,
        "spread_pips_entry": sp_e,
        "spread_pips_exit": sp_x,
        "spread_floored": bool(spread_floored),
    }


def _run_shadow_config(
    pair_arrs: Dict[str, dict],
    pair_ts_idx: Dict[str, Dict[int, int]],
    floors: Dict[str, float],
    trades: pd.DataFrame,
    bar_offset: int,
    sl_atr_mult: float,
    h_exit: int,
) -> pd.DataFrame:
    """Run shadow simulation over all trades. Returns trade-level DataFrame."""
    rows = []
    sig_ts = pd.to_datetime(trades["signal_bar_ts"]).astype("int64").to_numpy()
    pairs = trades["pair"].to_numpy()
    folds = trades["fold_id"].to_numpy()
    tids = trades["trade_id"].to_numpy()
    atrs = trades["atr_at_signal"].astype(float).to_numpy()

    for k in range(len(trades)):
        pair = str(pairs[k])
        d = pair_arrs[pair]
        sig_idx = pair_ts_idx[pair].get(int(sig_ts[k]))
        if sig_idx is None:
            continue
        atr_at_sig = float(atrs[k])
        if not np.isfinite(atr_at_sig) or atr_at_sig <= 0:
            continue
        pair_pip = pip_size(pair)
        floor_pips = floors.get(pair, None)
        out = _simulate_trade(
            d["open"],
            d["high"],
            d["low"],
            d["spread"],
            sig_idx,
            atr_at_sig,
            bar_offset,
            sl_atr_mult,
            h_exit,
            pair_pip,
            floor_pips,
        )
        if not out["valid"]:
            continue
        rows.append(
            {
                "trade_id": int(tids[k]),
                "fold_id": int(folds[k]),
                "pair": pair,
                **out,
            }
        )
    return pd.DataFrame(rows)


def _summarise(df: pd.DataFrame, shadow_label: str) -> dict:
    n = len(df)
    if n == 0:
        return {"shadow": shadow_label, "n": 0}
    df["exit_reason"].value_counts(normalize=True).to_dict()
    return {
        "shadow": shadow_label,
        "n": int(n),
        "mean_net_r": float(df["net_r"].mean()),
        "median_net_r": float(df["net_r"].median()),
        "p5_net_r": float(df["net_r"].quantile(0.05)),
        "p95_net_r": float(df["net_r"].quantile(0.95)),
        "win_pct": float((df["net_r"] > 0).mean() * 100.0),
        "frac_sl_hit": float((df["exit_reason"] == "stop_loss").mean()),
        "frac_time_exit": float((df["exit_reason"] == "time_exit").mean()),
        "frac_floored": float((df["spread_floored"]).mean()),
    }


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
            "spread": df["spread"].astype(float).values
            if "spread" in df.columns
            else np.zeros(len(df)),
        }
        ts_int = df["time"].astype("int64").to_numpy()
        pair_ts_idx[pair] = {int(t): i for i, t in enumerate(ts_int)}

    summaries: List[dict] = []

    # Sweep entry delays (bar_offset varies; SL=2.0, h=1)
    print("[Phase E] entry-delay sweep...")
    entry_curve_rows = []
    for bo in ENTRY_DELAYS:
        t_s = time.time()
        df = _run_shadow_config(
            pair_arrs, pair_ts_idx, floors, trades, bar_offset=bo, sl_atr_mult=2.0, h_exit=1
        )
        label = f"entry_delay__n{bo}_sl2.0_h1"
        df.to_csv(
            SHADOW_DIR / "entry_delay" / f"n{bo}.csv",
            index=False,
            lineterminator="\n",
            float_format="%.6g",
        )
        s = _summarise(df, label)
        summaries.append(s)
        entry_curve_rows.append({"bar_offset": bo, **{k: v for k, v in s.items() if k != "shadow"}})
        print(
            f"  bo={bo}: n={s.get('n', 0)} mean_R={s.get('mean_net_r', 0):.5f} "
            f"sl%={s.get('frac_sl_hit', 0) * 100:.2f} ({time.time() - t_s:.1f}s)"
        )

    pd.DataFrame(entry_curve_rows).to_csv(
        SHADOW_DIR / "entry_delay_curve.csv", index=False, lineterminator="\n"
    )

    # Sweep SL distance (sl_atr_mult varies; bo=1, h=1)
    print("[Phase E] SL-distance sweep...")
    sl_rows = []
    for mult in SL_DISTANCES:
        t_s = time.time()
        df = _run_shadow_config(
            pair_arrs, pair_ts_idx, floors, trades, bar_offset=1, sl_atr_mult=mult, h_exit=1
        )
        label = f"sl_distance__n1_sl{mult}_h1"
        df.to_csv(
            SHADOW_DIR / "sl_distance" / f"sl{mult}.csv",
            index=False,
            lineterminator="\n",
            float_format="%.6g",
        )
        s = _summarise(df, label)
        summaries.append(s)
        sl_rows.append({"sl_atr_mult": mult, **{k: v for k, v in s.items() if k != "shadow"}})
        print(
            f"  sl={mult}: n={s.get('n', 0)} mean_R={s.get('mean_net_r', 0):.5f} "
            f"sl%={s.get('frac_sl_hit', 0) * 100:.2f} ({time.time() - t_s:.1f}s)"
        )

    pd.DataFrame(sl_rows).to_csv(
        SHADOW_DIR / "sl_distance_sweep.csv", index=False, lineterminator="\n"
    )

    # Sweep time exit (h varies; bo=1, sl=2.0)
    print("[Phase E] time-exit sweep...")
    te_rows = []
    for h in TIME_EXIT_H:
        t_s = time.time()
        df = _run_shadow_config(
            pair_arrs, pair_ts_idx, floors, trades, bar_offset=1, sl_atr_mult=2.0, h_exit=h
        )
        label = f"time_exit__n1_sl2.0_h{h}"
        df.to_csv(
            SHADOW_DIR / "time_exit" / f"h{h}.csv",
            index=False,
            lineterminator="\n",
            float_format="%.6g",
        )
        s = _summarise(df, label)
        summaries.append(s)
        # capture ratio: realised R / fwd_mfe_atr_at_h (per trade), pool median
        if len(df):
            features_path = STEP2_DIR / "signals_features.csv"
            # Only compute capture ratio if column exists
            try:
                f_h = pd.read_csv(features_path, usecols=["trade_id", f"fwd_mfe_h{h}_atr"])
                m = df.merge(f_h, on="trade_id", how="left")
                # capture = R / (fwd_mfe_h_atr / sl_distance_atr) — convert atr→sl units
                # Note: realised R is in sl-units, fwd_mfe is in atr units; divide by sl_atr_mult=2
                denom = m[f"fwd_mfe_h{h}_atr"] / 2.0
                cap = m["R"] / denom.replace(0, np.nan)
                median_cap = float(cap.median(skipna=True))
            except Exception:
                median_cap = np.nan
        else:
            median_cap = np.nan
        te_rows.append(
            {
                "h": h,
                "median_capture_ratio_vs_fwd_mfe": median_cap,
                **{k: v for k, v in s.items() if k != "shadow"},
            }
        )
        print(
            f"  h={h}: n={s.get('n', 0)} mean_R={s.get('mean_net_r', 0):.5f} "
            f"sl%={s.get('frac_sl_hit', 0) * 100:.2f} cap_med={median_cap:.3f} ({time.time() - t_s:.1f}s)"
        )

    pd.DataFrame(te_rows).to_csv(
        SHADOW_DIR / "time_exit_curve.csv", index=False, lineterminator="\n"
    )

    pd.DataFrame(summaries).to_csv(
        SHADOW_DIR / "shadow_summary.csv", index=False, lineterminator="\n"
    )
    print(f"[Phase E] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_phase_e()

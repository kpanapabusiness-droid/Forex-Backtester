"""Phase G — random-entry baseline (op spec §5.13).

Random entries on the same 28 pairs, same OOS windows, matched per-(pair, fold)
counts. Same SL (2.0 × ATR), time exit (h=1), spread treatment as the
verbatim signal. Seed = 1234.

Outputs:
  random_baseline/random_entry_distribution.csv
  random_baseline/comparison.csv
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
    PAIRS, RANDOM_SEED, STEP2_DIR, compute_signal_mask, load_all_floors,
    load_pair_1h, load_trades_verbatim, pip_size, wilder_atr,
)
from scripts.l_arc_1.step2.phase_e_shadows import _simulate_trade, SPREAD_FLOOR_PATH

RAND_DIR = STEP2_DIR / "random_baseline"

FOLD_BOUNDARIES = [
    (1, "2020-10-01", "2021-07-01"),
    (2, "2021-07-01", "2022-04-01"),
    (3, "2022-04-01", "2023-01-01"),
    (4, "2023-01-01", "2023-10-01"),
    (5, "2023-10-01", "2024-07-01"),
    (6, "2024-07-01", "2025-04-01"),
    (7, "2025-04-01", "2026-01-01"),
]


def run_phase_g() -> None:
    t0 = time.time()
    print("[Phase G] loading...")
    RAND_DIR.mkdir(parents=True, exist_ok=True)
    trades = load_trades_verbatim()
    floors = load_all_floors(SPREAD_FLOOR_PATH)

    pair_arrs: Dict[str, dict] = {}
    pair_atr: Dict[str, np.ndarray] = {}
    pair_time_int: Dict[str, np.ndarray] = {}
    for pair in PAIRS:
        df = load_pair_1h(pair)
        pair_arrs[pair] = {
            "open": df["open"].astype(float).values,
            "high": df["high"].astype(float).values,
            "low": df["low"].astype(float).values,
            "close": df["close"].astype(float).values,
            "spread": df["spread"].astype(float).values if "spread" in df.columns else np.zeros(len(df)),
        }
        pair_time_int[pair] = df["time"].astype("datetime64[ns]").astype("int64").to_numpy()
        # Compute ATR using engine convention
        pair_atr[pair] = wilder_atr(
            df["high"].astype(float).values,
            df["low"].astype(float).values,
            df["close"].astype(float).values,
            period=14,
        )

    # Per-(pair, fold) counts from verbatim
    counts = trades.groupby(["pair", "fold_id"]).size().to_dict()

    rng = np.random.default_rng(RANDOM_SEED)
    rand_rows: List[dict] = []

    for pair in PAIRS:
        df_time = pair_time_int[pair]
        atr = pair_atr[pair]
        n_bars = len(df_time)
        for fold_id, oos_start, oos_end in FOLD_BOUNDARIES:
            k = counts.get((pair, fold_id), 0)
            if k == 0:
                continue
            start_ts = int(pd.Timestamp(oos_start).value)
            end_ts = int(pd.Timestamp(oos_end).value)
            # Valid signal indices: time within [start, end), index < n_bars - 2 (need entry + time exit + 1 buffer), ATR finite
            valid_mask = (df_time >= start_ts) & (df_time < end_ts) & np.isfinite(atr) & (atr > 0)
            valid_mask[-2:] = False  # buffer
            valid_idx = np.flatnonzero(valid_mask)
            if valid_idx.size == 0:
                continue
            # Sample with replacement if k > valid_idx.size; otherwise without
            replace = k > valid_idx.size
            sample = rng.choice(valid_idx, size=k, replace=replace)
            for sig_idx in sample:
                atr_at_sig = float(atr[sig_idx])
                d = pair_arrs[pair]
                out = _simulate_trade(
                    d["open"], d["high"], d["low"], d["spread"],
                    int(sig_idx), atr_at_sig, bar_offset=1, sl_atr_mult=2.0, h_exit=1,
                    pair_pip=pip_size(pair), floor_pips=floors.get(pair, None),
                )
                if not out["valid"]:
                    continue
                rand_rows.append({
                    "pair": pair, "fold_id": fold_id,
                    "signal_bar_ts": pd.Timestamp(df_time[sig_idx]).isoformat(),
                    "atr_at_signal": atr_at_sig,
                    "net_r": out["net_r"],
                    "exit_reason": out["exit_reason"],
                    "spread_floored": out["spread_floored"],
                })

    df_r = pd.DataFrame(rand_rows)
    df_r.to_csv(RAND_DIR / "random_entry_distribution.csv",
                index=False, lineterminator="\n", float_format="%.6g")
    print(f"  random trades: {len(df_r):,}")

    # Comparison vs verbatim
    print("[Phase G] writing comparison...")
    PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    def summarise(s: pd.Series, label: str) -> dict:
        n = len(s)
        out = {"set": label, "n": int(n),
               "mean": float(s.mean()), "std": float(s.std(ddof=1)),
               "skew": float(s.skew()), "kurt": float(s.kurt()),
               "min": float(s.min()), "max": float(s.max())}
        for p in PCTS:
            out[f"p{p}"] = float(s.quantile(p / 100.0))
        return out

    verbatim_r = trades["net_r"].astype(float) if "net_r" in trades.columns else trades["R"].astype(float)
    rand_r = df_r["net_r"].astype(float)
    summary = pd.DataFrame([
        summarise(verbatim_r, "verbatim"),
        summarise(rand_r, "random_baseline"),
    ])
    summary.to_csv(RAND_DIR / "comparison.csv", index=False, lineterminator="\n")

    # Descriptive yes/no on whether the distribution differs visibly
    diff_mean = abs(rand_r.mean() - verbatim_r.mean())
    diff_p50 = abs(rand_r.median() - verbatim_r.median())
    diff_p95 = abs(rand_r.quantile(0.95) - verbatim_r.quantile(0.95))
    # threshold: median differs by more than 1% of either's std OR 5% of either's p95
    pooled_std = max(verbatim_r.std(), rand_r.std())
    differs = (diff_mean > 0.005) or (diff_p50 > 0.005) or (diff_p95 > 0.05 * abs(verbatim_r.quantile(0.95)))
    differs_path = RAND_DIR / "differs_from_verbatim.txt"
    differs_path.write_text(
        f"differs (descriptive heuristic): {differs}\n"
        f"|mean diff|:   {diff_mean:.6f}\n"
        f"|p50 diff|:    {diff_p50:.6f}\n"
        f"|p95 diff|:    {diff_p95:.6f}\n"
        f"verbatim std: {verbatim_r.std():.6f}\n"
        f"random   std: {rand_r.std():.6f}\n",
        encoding="utf-8",
    )
    print(f"[Phase G] differs: {differs}; done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run_phase_g()

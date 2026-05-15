"""Phase G — random-entry baseline (op spec §5.13).

Random entries on the 28 pairs, same OOS windows, matched per-(pair, fold)
counts. Same SL (2.0 ATR), time exit (h=120), spread treatment as the verbatim
signal. Hash-based seed (Amendment 11).
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
    PAIRS,
    SPREAD_FLOOR_PATH,
    STEP2_DIR,
    VERBATIM_SL_ATR_MULT,
    VERBATIM_TIME_EXIT_H,
    hash_seed,
    load_all_floors,
    load_pair_1h,
    load_trades_verbatim,
    pip_size,
    wilder_atr,
)
from scripts.l_arc_3.step2.phase_e_shadows import _simulate_trade

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
            "spread": df["spread"].astype(float).values
            if "spread" in df.columns
            else np.zeros(len(df)),
        }
        pair_time_int[pair] = df["time"].astype("datetime64[ns]").astype("int64").to_numpy()
        pair_atr[pair] = wilder_atr(
            df["high"].astype(float).values,
            df["low"].astype(float).values,
            df["close"].astype(float).values,
            period=14,
        )

    counts = trades.groupby(["pair", "fold_id"]).size().to_dict()

    seed_root = hash_seed("l_arc_3_step2_random_baseline")
    rand_rows: List[dict] = []
    for pair in PAIRS:
        df_time = pair_time_int[pair]
        atr = pair_atr[pair]
        for fold_id, oos_start, oos_end in FOLD_BOUNDARIES:
            k = counts.get((pair, fold_id), 0)
            if k == 0:
                continue
            seed_pf = hash_seed(
                f"l_arc_3_step2_random__{pair}__fold{fold_id}__h{VERBATIM_TIME_EXIT_H}"
            )
            rng = np.random.default_rng(seed_root ^ seed_pf)
            start_ts = int(pd.Timestamp(oos_start).value)
            end_ts = int(pd.Timestamp(oos_end).value)
            valid_mask = (df_time >= start_ts) & (df_time < end_ts) & np.isfinite(atr) & (atr > 0)
            # Need enough lookahead bars for time exit
            for i in range(min(VERBATIM_TIME_EXIT_H + 2, len(valid_mask))):
                valid_mask[-(i + 1)] = False
            valid_idx = np.flatnonzero(valid_mask)
            if valid_idx.size == 0:
                continue
            replace = k > valid_idx.size
            sample = rng.choice(valid_idx, size=k, replace=replace)
            for sig_idx in sample:
                atr_at_sig = float(atr[sig_idx])
                d = pair_arrs[pair]
                out = _simulate_trade(
                    d["open"],
                    d["high"],
                    d["low"],
                    d["spread"],
                    int(sig_idx),
                    atr_at_sig,
                    bar_offset=1,
                    sl_atr_mult=VERBATIM_SL_ATR_MULT,
                    h_exit=VERBATIM_TIME_EXIT_H,
                    pair_pip=pip_size(pair),
                    floor_pips=floors.get(pair, None),
                )
                if not out["valid"]:
                    continue
                rand_rows.append(
                    {
                        "pair": pair,
                        "fold_id": fold_id,
                        "signal_bar_ts": pd.Timestamp(df_time[sig_idx]).isoformat(),
                        "atr_at_signal": atr_at_sig,
                        "net_r": out["net_r"],
                        "exit_reason": out["exit_reason"],
                        "spread_floored": out["spread_floored"],
                        "bars_held": out["bars_held"],
                    }
                )

    df_r = pd.DataFrame(rand_rows)
    df_r.to_csv(
        RAND_DIR / "random_entry_distribution.csv",
        index=False,
        lineterminator="\n",
        float_format="%.6g",
    )
    print(f"  random trades: {len(df_r):,}")

    PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    def summarise(s: pd.Series, label: str) -> dict:
        n = len(s)
        out = {
            "set": label,
            "n": int(n),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "skew": float(s.skew()),
            "kurt": float(s.kurt()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
        for p in PCTS:
            out[f"p{p}"] = float(s.quantile(p / 100.0))
        return out

    verbatim_r = trades["net_r"].astype(float)
    rand_r = df_r["net_r"].astype(float)
    summary = pd.DataFrame(
        [
            summarise(verbatim_r, "verbatim"),
            summarise(rand_r, "random_baseline"),
        ]
    )
    summary.to_csv(RAND_DIR / "comparison.csv", index=False, lineterminator="\n")

    diff_mean = abs(rand_r.mean() - verbatim_r.mean())
    diff_p50 = abs(rand_r.median() - verbatim_r.median())
    diff_p95 = abs(rand_r.quantile(0.95) - verbatim_r.quantile(0.95))
    pooled_std = max(verbatim_r.std(), rand_r.std())
    # For h=120 the SL is a 2 ATR distance, R magnitude could be larger; tolerate looser absolute thresholds
    differs = (
        diff_mean > 0.01 * max(1.0, pooled_std)
        or diff_p50 > 0.01 * max(1.0, pooled_std)
        or diff_p95 > 0.05 * max(1.0, abs(verbatim_r.quantile(0.95)))
    )
    differs_path = RAND_DIR / "differs_from_verbatim.txt"
    differs_path.write_text(
        f"differs (descriptive heuristic): {differs}\n"
        f"|mean diff|:   {diff_mean:.6f}\n"
        f"|p50 diff|:    {diff_p50:.6f}\n"
        f"|p95 diff|:    {diff_p95:.6f}\n"
        f"verbatim mean / std / p50 / p95: {verbatim_r.mean():.4f} / {verbatim_r.std():.4f} / "
        f"{verbatim_r.median():.4f} / {verbatim_r.quantile(0.95):.4f}\n"
        f"random   mean / std / p50 / p95: {rand_r.mean():.4f} / {rand_r.std():.4f} / "
        f"{rand_r.median():.4f} / {rand_r.quantile(0.95):.4f}\n"
        f"verbatim exit mix: {trades['exit_reason_canonical'].value_counts(normalize=True).to_dict()}\n"
        f"random   exit mix: {df_r['exit_reason'].value_counts(normalize=True).to_dict()}\n",
        encoding="utf-8",
    )
    print(f"[Phase G] differs: {differs}; done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_phase_g()

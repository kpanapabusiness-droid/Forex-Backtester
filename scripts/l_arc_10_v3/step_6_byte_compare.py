"""Arc 10 v3.0 — Step 6 byte-compare verification.

Samples N random trades from step_1/pool.parquet, re-runs the DLR signal
producer on the relevant pair's H4+D1 data, and compares the recomputed
signal-bar features to the pool values column-by-column.

Verifies:
  - L1_value, L0_value, L1_age_d1_bars, L0_age_d1_bars
  - L1_to_atr_proximity, reject_buffer_atr, upper_fraction
  - atr14_at_signal
  - signal flag is True at the recomputed signal_bar_time

Independently verifies Arc 9 lesson: D1 swing-low detection at signal-bar
must obey right-edge offset = k + 1 (=4), so its confirmation window
spans only D1 bars strictly before D1[d_t].

Output:
  results/l_arc_10/step_6/byte_compare_log.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import signals.lchar_dlr_long as dlr  # noqa: E402
from scripts.l_arc_10_v3._common import (  # noqa: E402
    bid_view_for_signal,
    load_config,
    load_pair_tf,
    window_slice,
)

POOL_PATH = REPO_ROOT / "results" / "l_arc_10" / "step_1" / "pool.parquet"
CONFIG_PATH = REPO_ROOT / "configs" / "l_arc_10_v3" / "arc_open.yaml"
OUT_PATH = REPO_ROOT / "results" / "l_arc_10" / "step_6" / "byte_compare_log.json"

NUM_SAMPLES = 5
RTOL = 1e-9
ATOL = 1e-9


def _samples(pool: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_avail = len(pool)
    if n_avail == 0:
        return pool.iloc[0:0]
    idx = rng.choice(n_avail, size=min(n, n_avail), replace=False)
    return pool.iloc[idx].sort_values("signal_bar_time").reset_index(drop=True)


def _compare(name: str, pool_val: float, recomputed: float) -> dict:
    pool_v = float(pool_val) if not pd.isna(pool_val) else float("nan")
    rec_v = float(recomputed) if not pd.isna(recomputed) else float("nan")
    both_nan = math.isnan(pool_v) and math.isnan(rec_v)
    if both_nan:
        return dict(name=name, pool=pool_v, recomputed=rec_v, match=True, abs_diff=0.0)
    if math.isnan(pool_v) or math.isnan(rec_v):
        return dict(name=name, pool=pool_v, recomputed=rec_v, match=False, abs_diff=float("nan"))
    diff = abs(pool_v - rec_v)
    match = bool(diff <= ATOL + RTOL * max(abs(pool_v), abs(rec_v)))
    return dict(name=name, pool=pool_v, recomputed=rec_v, match=match, abs_diff=diff)


def _verify_arc9_swing_invariance(
    df_h4_bid: pd.DataFrame,
    df_d1_bid: pd.DataFrame,
    sig_idx_4h: int,
) -> dict:
    """Independent Arc 9 lesson check.

    Per producer at signals/lchar_dlr_long.py:229, d_search_max = d_t - 4.
    Re-derive d_t for this signal-bar timestamp, then verify:
      - The L1 D1 index used by the producer satisfies l1_idx <= d_t - 4.
      - The L1 swing-low's confirmation window spans D1 bars
        [l1_idx - 3 .. l1_idx + 3], all of which are <= d_t - 1
        (strictly prior to D1[d_t]).
      - Perturbing D1[d_t] to NaN leaves the signal output unchanged at
        sig_idx_4h.
    """
    sig_row = dlr.compute_signal(df_h4_bid, df_d1_bid, signal_col="signal").reset_index(drop=True)
    d_t = int(sig_row["d_t_idx"].iloc[sig_idx_4h])
    d_search_max = int(sig_row["d_for_l1_search_max"].iloc[sig_idx_4h])
    l1_age = float(sig_row["L1_age_d1_bars"].iloc[sig_idx_4h])
    l1_idx = int(d_t - l1_age) if not math.isnan(l1_age) else -1

    # Check 1: right-edge constraint
    right_edge_ok = (d_search_max == d_t - 4) and (l1_idx <= d_search_max)

    # Check 2: confirmation window stays before d_t
    confirmation_window_max = l1_idx + 3
    confirmation_window_ok = confirmation_window_max <= d_t - 1

    # Check 3: NaN-perturbation invariance at D1[d_t]
    bar_date = pd.Timestamp(df_h4_bid["date"].iloc[sig_idx_4h]).normalize()
    d1_dates = pd.to_datetime(df_d1_bid["date"]).dt.normalize()
    match = d1_dates == bar_date
    d1_pert = df_d1_bid.copy()
    if match.any():
        d1_pert.loc[match, ["open", "high", "low", "close"]] = np.nan
    pert_sig = dlr.compute_signal(df_h4_bid, d1_pert, signal_col="signal").reset_index(drop=True)
    nan_invariance_ok = bool(pert_sig["signal"].iloc[sig_idx_4h]) == bool(sig_row["signal"].iloc[sig_idx_4h])

    return dict(
        d_t=d_t,
        d_search_max=d_search_max,
        l1_idx=l1_idx,
        l1_age=l1_age,
        right_edge_constraint_ok=bool(right_edge_ok),
        confirmation_window_in_past_ok=bool(confirmation_window_ok),
        confirmation_window_max_d1_idx=int(confirmation_window_max),
        nan_perturbation_invariance_ok=bool(nan_invariance_ok),
        all_ok=bool(right_edge_ok and confirmation_window_ok and nan_invariance_ok),
    )


def main() -> int:
    cfg = load_config(CONFIG_PATH)
    pool = pd.read_parquet(POOL_PATH)
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)

    samples = _samples(pool, NUM_SAMPLES, seed=42)
    if len(samples) == 0:
        print("[step_6_byte_compare] empty pool — nothing to verify", flush=True)
        return 1

    # Cache per-pair signal frames lazily
    per_pair_cache: dict[str, dict] = {}

    log_entries = []
    overall_ok = True

    for _, row in samples.iterrows():
        pair = row["pair"]
        if pair not in per_pair_cache:
            h4_full = load_pair_tf(pair, "H4", cfg)
            d1_full = load_pair_tf(pair, "D1", cfg)
            h4_w = window_slice(h4_full, cfg["window"]["start"], cfg["window"]["end"])
            pad_start = (pd.Timestamp(cfg["window"]["start"], tz="UTC") - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
            d1_w = window_slice(d1_full, pad_start, cfg["window"]["end"])
            df_h4_bid = bid_view_for_signal(h4_w)
            df_d1_bid = bid_view_for_signal(d1_w)
            sig_df = dlr.compute_signal(df_h4_bid, df_d1_bid, signal_col="signal").reset_index(drop=True)
            per_pair_cache[pair] = dict(
                df_h4_bid=df_h4_bid,
                df_d1_bid=df_d1_bid,
                sig_df=sig_df,
            )

        cache = per_pair_cache[pair]
        df_h4_bid = cache["df_h4_bid"]
        sig_df = cache["sig_df"]

        # Find sig_idx_4h matching this trade's signal_bar_time
        target_ts = pd.Timestamp(row["signal_bar_time"]).tz_convert(None)
        bar_dates = pd.to_datetime(df_h4_bid["date"])
        if bar_dates.dt.tz is not None:
            bar_dates = bar_dates.dt.tz_convert(None)
        match_idx = np.where(bar_dates.to_numpy() == target_ts.to_datetime64())[0]
        if match_idx.size == 0:
            log_entries.append(dict(
                trade_id=int(row["trade_id"]), pair=pair, status="NOT_FOUND",
                signal_bar_time=str(row["signal_bar_time"]),
            ))
            overall_ok = False
            continue
        sig_idx_4h = int(match_idx[0])

        # Verify signal flag is True
        signal_flag = bool(sig_df["signal"].iloc[sig_idx_4h])

        # Byte-compare signal-bar features
        comparisons = []
        for col in [
            "L1_value",
            "L0_value",
            "L1_age_d1_bars",
            "L0_age_d1_bars",
            "L1_to_atr_proximity",
            "reject_buffer_atr",
            "upper_fraction",
        ]:
            comparisons.append(_compare(col, row[col], sig_df[col].iloc[sig_idx_4h]))
        comparisons.append(_compare("atr14_at_signal", row["atr14_at_signal"], sig_df["atr14"].iloc[sig_idx_4h]))

        all_match = all(c["match"] for c in comparisons)

        # Arc 9 lesson verification
        arc9 = _verify_arc9_swing_invariance(
            df_h4_bid=df_h4_bid,
            df_d1_bid=cache["df_d1_bid"],
            sig_idx_4h=sig_idx_4h,
        )

        trade_ok = signal_flag and all_match and arc9["all_ok"]
        overall_ok = overall_ok and trade_ok

        log_entries.append(dict(
            trade_id=int(row["trade_id"]),
            pair=pair,
            sig_idx_4h=sig_idx_4h,
            signal_bar_time=str(row["signal_bar_time"]),
            signal_flag_true_at_recompute=signal_flag,
            feature_comparisons=comparisons,
            all_features_match=all_match,
            arc9_swing_invariance=arc9,
            trade_ok=trade_ok,
        ))

        status = "OK" if trade_ok else "FAIL"
        print(
            f"  [{status}] trade {row['trade_id']} ({pair} @ {row['signal_bar_time']}) "
            f"sig={signal_flag} feat={all_match} arc9={arc9['all_ok']}",
            flush=True,
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            dict(
                n_samples=int(len(samples)),
                seed=42,
                rtol=RTOL,
                atol=ATOL,
                overall_ok=overall_ok,
                entries=log_entries,
            ),
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[step_6_byte_compare] overall_ok={overall_ok} wrote {OUT_PATH}", flush=True)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

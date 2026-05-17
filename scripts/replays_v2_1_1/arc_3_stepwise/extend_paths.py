"""Forward-extend Arc 3 trades_paths.csv to bar offset 240 with is_held flag.

Arc 3's step1_backtest.py truncates trades_paths.csv at min(SL_hit, time_exit_idx=120)
and emits no `is_held` column. v2.1.1 §7 SL sweep requires forward observation bars
out to entry+240 regardless of exit reason (per §17 SL-free path definition).

This script does NOT modify Arc 3's frozen step1 output. It reads:
  - results/l_arc_3/step1_plumbing/trades_all.csv  (entry_time, entry_price, initial_sl_price, pair)
  - results/l_arc_3/step1_plumbing/trades_paths.csv  (existing held bars, bar 0..exit_bar)
  - data/1hr/<pair>.csv  (1H OHLC for forward window)

And writes:
  - results/replays_v2_1_1/arc_3_stepwise/trades_paths_extended.csv

Schema: trade_id, pair, bar_offset, bar_time, close_r, mfe_so_far_r, mae_so_far_r, is_held.
Held bars (bar 0..exit_bar) are copied byte-equivalent from the source. Forward bars
(exit_bar+1..240) are computed against the same entry_price + SL distance as the
original simulation, using identical per-bar excursion math.

Per-bar excursion math (long-only) mirrors scripts/arc_3/step1_backtest.py:
  cand_mfe_price = high - entry_fill
  cand_mae_price = entry_fill - low
  mfe_so_far_price = max(mfe_so_far_price, cand_mfe_price)
  mae_so_far_price = max(mae_so_far_price, cand_mae_price)
  mfe_so_far_r = mfe_so_far_price / sl_distance_price   where sl_distance_price = entry_price - initial_sl_price
  mae_so_far_r = -mae_so_far_price / sl_distance_price
  close_r      = (close - entry_fill) / sl_distance_price

Deterministic; two-run byte-identical (no RNG, no parallelism, stable iteration order).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FORWARD_BARS = 240


def _load_pair_csv(pair: str, data_dir: Path) -> pd.DataFrame:
    fpath = data_dir / f"{pair}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"missing data file: {fpath}")
    df = pd.read_csv(fpath)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def extend_paths(
    trades_all_path: Path,
    trades_paths_path: Path,
    data_1h_dir: Path,
    output_path: Path,
    forward_bars: int = FORWARD_BARS,
) -> dict:
    trades_all = pd.read_csv(trades_all_path)
    trades_paths = pd.read_csv(trades_paths_path)
    trades_all["entry_time"] = pd.to_datetime(trades_all["entry_time"])

    # Pre-compute per-trade lookups.
    trade_meta = trades_all.set_index("trade_id")[
        ["pair", "entry_time", "entry_price", "initial_sl_price"]
    ].to_dict("index")

    # Group existing paths by trade for fast access.
    held_by_trade: dict[int, pd.DataFrame] = {
        tid: g.sort_values("bar_offset").reset_index(drop=True)
        for tid, g in trades_paths.groupby("trade_id", sort=True)
    }

    # Group trades by pair for data loading.
    by_pair: dict[str, list[int]] = {}
    for tid, meta in trade_meta.items():
        by_pair.setdefault(meta["pair"], []).append(tid)
    for pair in by_pair:
        by_pair[pair].sort()

    out_rows: list[dict] = []
    extension_stats = {
        "trades_with_extension": 0,
        "trades_no_extension_needed": 0,
        "trades_data_runout": 0,
        "total_held_rows": 0,
        "total_forward_rows": 0,
    }

    for pair in sorted(by_pair.keys()):
        df_1h = _load_pair_csv(pair, data_1h_dir)
        dates = df_1h["date"].to_numpy()
        highs = df_1h["high"].astype(float).to_numpy()
        lows = df_1h["low"].astype(float).to_numpy()
        closes = df_1h["close"].astype(float).to_numpy()
        n = len(df_1h)

        for tid in by_pair[pair]:
            meta = trade_meta[tid]
            entry_time = meta["entry_time"]
            entry_price = float(meta["entry_price"])
            initial_sl_price = float(meta["initial_sl_price"])
            sl_distance_price = entry_price - initial_sl_price  # long-only positive
            if sl_distance_price <= 0:
                raise ValueError(f"trade {tid}: non-positive sl_distance_price {sl_distance_price}")

            held = held_by_trade[tid]
            last_held_offset = int(held["bar_offset"].max())
            # Re-emit held bars (read-through; preserves byte values).
            for _, r in held.iterrows():
                out_rows.append(
                    {
                        "trade_id": int(r["trade_id"]),
                        "pair": pair,
                        "bar_offset": int(r["bar_offset"]),
                        "bar_time": r["bar_time"],
                        "close_r": float(r["close_r"]),
                        "mfe_so_far_r": float(r["mfe_so_far_r"]),
                        "mae_so_far_r": float(r["mae_so_far_r"]),
                        "is_held": 1,
                    }
                )
            extension_stats["total_held_rows"] += len(held)

            if last_held_offset >= forward_bars:
                extension_stats["trades_no_extension_needed"] += 1
                continue

            # Locate entry bar index by date match.
            entry_idx_arr = np.where(dates == np.datetime64(entry_time))[0]
            if entry_idx_arr.size == 0:
                raise ValueError(
                    f"trade {tid}: entry_time {entry_time} not found in {pair} 1H data"
                )
            entry_idx = int(entry_idx_arr[0])

            # Recover running mfe/mae prices from last held bar.
            mfe_so_far_r = float(held.iloc[-1]["mfe_so_far_r"])
            mae_so_far_r = float(held.iloc[-1]["mae_so_far_r"])
            mfe_so_far_price = mfe_so_far_r * sl_distance_price
            mae_so_far_price = -mae_so_far_r * sl_distance_price  # stored mae_r is negative

            target_end = min(entry_idx + forward_bars, n - 1)
            first_forward = last_held_offset + 1
            forward_emitted_this_trade = 0
            for bar_off in range(first_forward, forward_bars + 1):
                bar_idx = entry_idx + bar_off
                if bar_idx > target_end:
                    break
                hk = float(highs[bar_idx])
                lk = float(lows[bar_idx])
                ck = float(closes[bar_idx])
                cand_mfe_price = hk - entry_price
                cand_mae_price = entry_price - lk
                if cand_mfe_price > mfe_so_far_price:
                    mfe_so_far_price = cand_mfe_price
                if cand_mae_price > mae_so_far_price:
                    mae_so_far_price = cand_mae_price
                mfe_r_val = mfe_so_far_price / sl_distance_price
                mae_r_val = -mae_so_far_price / sl_distance_price
                close_r_val = (ck - entry_price) / sl_distance_price
                out_rows.append(
                    {
                        "trade_id": int(tid),
                        "pair": pair,
                        "bar_offset": int(bar_off),
                        "bar_time": pd.Timestamp(dates[bar_idx]).isoformat(sep=" "),
                        "close_r": float(close_r_val),
                        "mfe_so_far_r": float(mfe_r_val),
                        "mae_so_far_r": float(mae_r_val),
                        "is_held": 0,
                    }
                )
                forward_emitted_this_trade += 1
            extension_stats["total_forward_rows"] += forward_emitted_this_trade
            if forward_emitted_this_trade > 0:
                extension_stats["trades_with_extension"] += 1
            if entry_idx + forward_bars > n - 1:
                extension_stats["trades_data_runout"] += 1

    out_df = (
        pd.DataFrame(
            out_rows,
            columns=[
                "trade_id",
                "pair",
                "bar_offset",
                "bar_time",
                "close_r",
                "mfe_so_far_r",
                "mae_so_far_r",
                "is_held",
            ],
        )
        .sort_values(["trade_id", "bar_offset"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, float_format="%.10g")
    extension_stats["output_rows"] = len(out_df)
    extension_stats["output_path"] = str(output_path)
    return extension_stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trades_all",
        default=str(_REPO_ROOT / "results/l_arc_3/step1_plumbing/trades_all.csv"),
    )
    parser.add_argument(
        "--trades_paths",
        default=str(_REPO_ROOT / "results/l_arc_3/step1_plumbing/trades_paths.csv"),
    )
    parser.add_argument(
        "--data_1h_dir",
        default=str(_REPO_ROOT / "data/1hr"),
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "results/replays_v2_1_1/arc_3_stepwise/trades_paths_extended.csv"),
    )
    parser.add_argument("--forward_bars", type=int, default=FORWARD_BARS)
    args = parser.parse_args()

    stats = extend_paths(
        trades_all_path=Path(args.trades_all),
        trades_paths_path=Path(args.trades_paths),
        data_1h_dir=Path(args.data_1h_dir),
        output_path=Path(args.output),
        forward_bars=args.forward_bars,
    )
    print("extend_paths complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

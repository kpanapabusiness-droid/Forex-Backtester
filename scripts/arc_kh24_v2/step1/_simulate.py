"""Trade execution + per-bar path emission for the KH-24 v2.0 step-1 self-test.

Bare signal pool only:
- Entry: bar offset 0 = signal_idx + 1 open + spread/2 (long fills ask).
- Hard SL: entry - 2 * ATR(14) at signal bar N. Anchored to entry, immutable.
- SL search: bar offsets 0..HOLD_BARS-1 inclusive. Intrabar low <= SL → exit
  at SL with spread from current bar (long fills bid = SL - spread/2).
- Time exit: bar offset HOLD_BARS open with spread from that execution bar
  (long fills bid = open - spread/2). exit_reason = "bar_240_system_exit".
- Trades whose 240-bar forward window runs past data end are SKIPPED (no
  fallback per SPREAD_SEMANTICS_LOCK.md "no fallback for missing next bar").

Path rows: bar offsets 0..bars_held inclusive, computed from MID OHLC of each
bar (close_r / mfe_so_far_r / mae_so_far_r normalised to R = 2 * ATR_at_N).

No filters, no trail, no kijun_d1. Pure population.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

POINTS_PER_PIP = 10.0  # MT5 5-digit broker convention


def _pip_size(pair: str) -> float:
    return 0.01 if pair.upper().endswith("_JPY") else 0.0001


class ExecParams(NamedTuple):
    hard_sl_atr_mult: float = 2.0
    hold_bars: int = 240


def simulate_pair(
    pair: str,
    df_4h: pd.DataFrame,
    sig_mask: np.ndarray,
    atr_4h: np.ndarray,
    params: ExecParams,
) -> tuple[list[dict], list[dict]]:
    """Execute every signal in `sig_mask` and emit trade rows + per-bar paths.

    Parameters
    ----------
    pair        : pair label (e.g., "EUR_USD"), used in trade_id and the row.
    df_4h       : 4H OHLCV+spread frame, sorted by date, integer index 0..n-1.
                  Must contain columns: date, open, high, low, close, spread.
    sig_mask    : boolean array of length n; True at signal bars.
    atr_4h      : float array of length n; Wilder ATR(14) at each bar.
    params      : execution parameters (hard SL multiple, hold bars).

    Returns
    -------
    trades : list of per-trade dicts (one row per trade in trades_all.csv).
    paths  : list of per-(trade, bar) dicts (one row per bar in trades_paths.csv).
    """
    n = len(df_4h)
    dates = pd.to_datetime(df_4h["date"]).values  # datetime64[ns]
    op = df_4h["open"].values.astype(float)
    hi = df_4h["high"].values.astype(float)
    lo = df_4h["low"].values.astype(float)
    cl = df_4h["close"].values.astype(float)
    sp_points = (
        df_4h["spread"].fillna(0.0).values.astype(float)
        if "spread" in df_4h.columns
        else np.zeros(n, dtype=float)
    )
    sp_pips = sp_points / POINTS_PER_PIP
    pip = _pip_size(pair)
    sp_price = sp_pips * pip

    H = int(params.hold_bars)
    sl_mult = float(params.hard_sl_atr_mult)

    trades: list[dict] = []
    paths: list[dict] = []

    sig_indices = np.flatnonzero(sig_mask)
    for sig_idx in sig_indices:
        entry_idx = int(sig_idx) + 1
        # Entry bar must exist
        if entry_idx >= n:
            continue
        atr_at_n = float(atr_4h[sig_idx])
        if not np.isfinite(atr_at_n) or atr_at_n <= 0:
            continue

        # Long entry: mid open + spread/2 (ask).
        entry_open_mid = float(op[entry_idx])
        entry_spread_price = float(sp_price[entry_idx])
        entry_price = entry_open_mid + entry_spread_price / 2.0
        sl_distance_price = sl_mult * atr_at_n  # = R (1R = 2 ATR per protocol §17)
        sl_at_entry_price = entry_price - sl_distance_price
        R = sl_distance_price  # entry - sl

        # Determine forward window data availability. For a deterministic 240-bar
        # forward window we require bars entry_idx .. entry_idx + H inclusive.
        # If not all bars present, skip trade.
        if entry_idx + H >= n:
            continue

        # SL search: intrabar low <= SL on bar offsets 0..H-1.
        sl_hit_offset = -1
        for k in range(H):
            bar = entry_idx + k
            if lo[bar] <= sl_at_entry_price:
                sl_hit_offset = k
                break

        if sl_hit_offset >= 0:
            exit_bar = entry_idx + sl_hit_offset
            exit_time = pd.Timestamp(dates[exit_bar])
            exit_spread_price = float(sp_price[exit_bar])
            exit_price = sl_at_entry_price - exit_spread_price / 2.0
            spread_pips_exit = float(sp_pips[exit_bar])
            exit_reason = "hard_sl"
            bars_held = int(sl_hit_offset)
        else:
            exit_bar = entry_idx + H  # bar offset H = 240 = execution bar
            exit_time = pd.Timestamp(dates[exit_bar])
            exit_spread_price = float(sp_price[exit_bar])
            exit_price = float(op[exit_bar]) - exit_spread_price / 2.0
            spread_pips_exit = float(sp_pips[exit_bar])
            exit_reason = "bar_240_system_exit"
            bars_held = H  # 240

        # Path rows: bar offsets 0..bars_held inclusive, MID OHLC based.
        # close_r, mfe_so_far_r (cumulative max), mae_so_far_r (cumulative min).
        signal_ts = pd.Timestamp(dates[int(sig_idx)])
        trade_id = f"{pair}_{signal_ts.strftime('%Y-%m-%dT%H:%M:%S')}"

        mfe_so_far = float("-inf")
        mae_so_far = float("+inf")
        for k in range(bars_held + 1):
            bar = entry_idx + k
            close_mid = float(cl[bar])
            high_excursion = (float(hi[bar]) - entry_price) / R
            low_excursion = (float(lo[bar]) - entry_price) / R
            close_r = (close_mid - entry_price) / R
            if high_excursion > mfe_so_far:
                mfe_so_far = high_excursion
            if low_excursion < mae_so_far:
                mae_so_far = low_excursion
            paths.append(
                {
                    "trade_id": trade_id,
                    "bar_offset": k,
                    "timestamp": pd.Timestamp(dates[bar]).isoformat(),
                    "close_mid": close_mid,
                    "close_r": close_r,
                    "mfe_so_far_r": mfe_so_far,
                    "mae_so_far_r": mae_so_far,
                }
            )

        final_r = (exit_price - entry_price) / R
        trades.append(
            {
                "trade_id": trade_id,
                "pair": pair,
                "entry_time": pd.Timestamp(dates[entry_idx]).isoformat(),
                "entry_price": entry_price,
                "signal_bar_atr_14": atr_at_n,
                "sl_at_entry_price": sl_at_entry_price,
                "exit_time": exit_time.isoformat(),
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "final_r": final_r,
                "mfe_r": mfe_so_far,
                "mae_r": mae_so_far,
                "spread_pips_used": float(sp_pips[entry_idx]),
                "spread_pips_exit": spread_pips_exit,
            }
        )

    return trades, paths

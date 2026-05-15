"""R-unit trade simulator for L Arc 2 Step 4.

Resolutions per open_questions.md §5, §6:
- entry_price = trade_paths[trade_id, offset=0].open
- SL_distance_price = 2.0 * atr_at_signal_1h (long-only, all arc-2 trades)
- SL_distance_logret = log(entry_price / (entry_price - SL_distance_price))
- R_gross at bar k (time exit) = cum_logret_from_entry[k] / SL_distance_logret
- SL hit: first bar k where mae_to_date_atr[k] >= 2.0  → R_gross = -1
- R_net = R_gross - spread_cost_R  (structural per-trade, ~constant)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeStaticContext:
    """Per-trade static fields needed by the simulator."""

    trade_id: int
    pair: str
    fold_id: int
    fire_bar_ts: str
    atr_at_signal_1h: float
    signal_bar_close: float
    spread_cost_R: float
    direction: int  # +1 long, -1 short (arc 2 is all long)


def load_static_context(signals: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trade_id",
        "pair",
        "fold_id",
        "signal_bar_ts",
        "atr_at_signal_1h",
        "signal_bar_close",
        "spread_cost_R",
        "direction",
    ]
    out = signals[cols].copy()
    # Normalise direction: arc 2 trades are all long; map any string to +1
    if out["direction"].dtype == object:
        out["direction"] = out["direction"].map(
            lambda x: 1 if str(x).lower() in ("long", "+1", "1") else -1
        )
    return out


def compute_sl_distance_price(atr_at_signal: np.ndarray) -> np.ndarray:
    """SL distance in price units (long-only). = 2.0 * ATR(14)_1H."""
    return 2.0 * atr_at_signal


def logret_to_r_units(
    cum_logret: np.ndarray, entry_price: np.ndarray, sl_dist_price: np.ndarray
) -> np.ndarray:
    """Convert cum_logret_from_entry to R-units using linear P&L convention.

    R = (exit_price - entry_price) / SL_distance_price
      = entry_price * (exp(cum_logret) - 1) / SL_distance_price
    Matches verbatim engine convention (linear P&L), not log ratio.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        return entry_price * (np.exp(cum_logret) - 1.0) / sl_dist_price


def simulate_time_exit_h(
    *,
    horizon: int,
    trade_ids: np.ndarray,
    signals: pd.DataFrame,
    paths_long: pd.DataFrame,
) -> pd.DataFrame:
    """Simulate net R for each trade under time exit at bar `horizon`.

    SL hit at first bar k in [0,horizon] where mae_to_date_atr >= 2.0.
    Else exit at bar offset `horizon`'s OPEN (matches verbatim engine convention).
    R = (exit_price - entry_price) / sl_dist_price.

    Returns DataFrame: trade_id, action_bar, exit_reason, gross_r, net_r,
    mfe_at_exit, mae_at_exit.
    """
    p = paths_long[
        paths_long["trade_id"].isin(trade_ids) & (paths_long["bar_offset"] <= horizon)
    ].copy()
    p = p.sort_values(["trade_id", "bar_offset"])

    sl_hit_mask = p["mae_to_date_atr"] >= 2.0
    sl_first = p[sl_hit_mask].groupby("trade_id")["bar_offset"].min()

    entry = p[p["bar_offset"] == 0].set_index("trade_id")["open"]
    horizon_row = p[p["bar_offset"] == horizon].set_index("trade_id")

    # Fall back to last available row if horizon row missing (data_end)
    last_row = p.loc[p.groupby("trade_id")["bar_offset"].idxmax()].set_index("trade_id")

    static = signals.set_index("trade_id").loc[trade_ids, ["atr_at_signal_1h", "spread_cost_R"]]

    entry_price = entry.reindex(trade_ids).values
    atr_sig = static["atr_at_signal_1h"].values
    spread_R = static["spread_cost_R"].values
    sl_dist_price = compute_sl_distance_price(atr_sig)
    sl_first_arr = sl_first.reindex(trade_ids).values

    horizon_open = horizon_row["open"].reindex(trade_ids).values
    last_close = last_row["close"].reindex(trade_ids).values
    last_off = last_row["bar_offset"].reindex(trade_ids).values
    mfe_at_horizon = horizon_row["mfe_to_date_atr"].reindex(trade_ids).values
    mae_at_horizon = horizon_row["mae_to_date_atr"].reindex(trade_ids).values
    mfe_at_last = last_row["mfe_to_date_atr"].reindex(trade_ids).values
    mae_at_last = last_row["mae_to_date_atr"].reindex(trade_ids).values

    action_bar = np.empty(len(trade_ids), dtype=np.float64)
    gross_r = np.empty(len(trade_ids), dtype=np.float64)
    exit_reason = np.empty(len(trade_ids), dtype=object)
    mfe_at_exit = np.empty(len(trade_ids), dtype=np.float64)
    mae_at_exit = np.empty(len(trade_ids), dtype=np.float64)

    for i in range(len(trade_ids)):
        sl_b = sl_first_arr[i]
        if pd.notna(sl_b):
            action_bar[i] = float(sl_b)
            gross_r[i] = -1.0
            exit_reason[i] = "sl_hit"
            mfe_at_exit[i] = mfe_at_last[i] if not np.isnan(mfe_at_last[i]) else np.nan
            mae_at_exit[i] = 2.0
        else:
            if (
                not np.isnan(horizon_open[i])
                and not np.isnan(entry_price[i])
                and sl_dist_price[i] > 0
            ):
                action_bar[i] = float(horizon)
                gross_r[i] = float((horizon_open[i] - entry_price[i]) / sl_dist_price[i])
                exit_reason[i] = "time_exit"
                mfe_at_exit[i] = mfe_at_horizon[i] if not np.isnan(mfe_at_horizon[i]) else np.nan
                mae_at_exit[i] = mae_at_horizon[i] if not np.isnan(mae_at_horizon[i]) else np.nan
            elif (
                not np.isnan(last_close[i])
                and not np.isnan(entry_price[i])
                and sl_dist_price[i] > 0
            ):
                # data_end fallback: exit at last available close (best we can do)
                action_bar[i] = float(last_off[i])
                gross_r[i] = float((last_close[i] - entry_price[i]) / sl_dist_price[i])
                exit_reason[i] = "data_end"
                mfe_at_exit[i] = mfe_at_last[i] if not np.isnan(mfe_at_last[i]) else np.nan
                mae_at_exit[i] = mae_at_last[i] if not np.isnan(mae_at_last[i]) else np.nan
            else:
                action_bar[i] = np.nan
                gross_r[i] = np.nan
                exit_reason[i] = "missing"
                mfe_at_exit[i] = np.nan
                mae_at_exit[i] = np.nan

    net_r = gross_r - spread_R

    return pd.DataFrame(
        {
            "trade_id": trade_ids,
            "action_bar": action_bar,
            "exit_reason": exit_reason,
            "gross_r": gross_r,
            "net_r": net_r,
            "spread_cost_r": spread_R,
            "mfe_at_exit": mfe_at_exit,
            "mae_at_exit": mae_at_exit,
        }
    )


def simulate_close_at_bar_t(
    *,
    t: int,
    trade_ids: np.ndarray,
    signals: pd.DataFrame,
    paths_long: pd.DataFrame,
) -> pd.DataFrame:
    """Simulate close-at-market at bar t (cluster-conditional early-exit).

    If trade SL'd before bar t in [0, t], use SL hit (R=-1). Otherwise exit at bar t close.
    """
    p = paths_long[paths_long["trade_id"].isin(trade_ids) & (paths_long["bar_offset"] <= t)].copy()
    p = p.sort_values(["trade_id", "bar_offset"])

    sl_hit_mask = p["mae_to_date_atr"] >= 2.0
    sl_first = p[sl_hit_mask].groupby("trade_id")["bar_offset"].min()
    sl_first_arr = sl_first.reindex(trade_ids).values

    entry = p[p["bar_offset"] == 0].set_index("trade_id")["open"]
    bar_t_row = p[p["bar_offset"] == t].set_index("trade_id")
    static = signals.set_index("trade_id").loc[trade_ids, ["atr_at_signal_1h", "spread_cost_R"]]

    entry_price = entry.reindex(trade_ids).values
    atr_sig = static["atr_at_signal_1h"].values
    spread_R = static["spread_cost_R"].values
    sl_dist_price = compute_sl_distance_price(atr_sig)

    bar_t_open = bar_t_row["open"].reindex(trade_ids).values
    mfe_at_t = bar_t_row["mfe_to_date_atr"].reindex(trade_ids).values
    mae_at_t = bar_t_row["mae_to_date_atr"].reindex(trade_ids).values

    action_bar = np.empty(len(trade_ids), dtype=np.float64)
    gross_r = np.empty(len(trade_ids), dtype=np.float64)
    exit_reason = np.empty(len(trade_ids), dtype=object)

    for i, tid in enumerate(trade_ids):
        sl_b = sl_first_arr[i]
        if pd.notna(sl_b) and sl_b <= t:
            action_bar[i] = float(sl_b)
            gross_r[i] = -1.0
            exit_reason[i] = "sl_hit_before_t"
        else:
            if np.isnan(bar_t_open[i]) or sl_dist_price[i] <= 0 or np.isnan(entry_price[i]):
                action_bar[i] = np.nan
                gross_r[i] = np.nan
                exit_reason[i] = "missing_bar_t"
            else:
                action_bar[i] = float(t)
                # Close-at-market at bar t's open (consistent with time-exit convention)
                gross_r[i] = float((bar_t_open[i] - entry_price[i]) / sl_dist_price[i])
                exit_reason[i] = "close_at_t"

    net_r = gross_r - spread_R
    return pd.DataFrame(
        {
            "trade_id": trade_ids,
            "action_bar": action_bar,
            "exit_reason": exit_reason,
            "gross_r": gross_r,
            "net_r": net_r,
            "spread_cost_r": spread_R,
            "mfe_at_exit": mfe_at_t,
            "mae_at_exit": mae_at_t,
        }
    )


def simulate_delayed_entry(
    *,
    t: int,
    trade_ids: np.ndarray,
    signals: pd.DataFrame,
    paths_long: pd.DataFrame,
    held_ctx_t: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Simulate delayed entry at bar_offset=t close, time exit at bar_offset=120.

    Per spec: delayed entry uses fresh ATR(14)_1H × atr_regime_ratio at t for SL
    distance, and time exit at original fire+120 (not 120 from new entry).
    """
    horizon_from_entry = 120  # h=120 from fire bar (= bar N+1)
    # New entry occurs at bar_offset=t. Hold from t to horizon_from_entry.
    p = paths_long[
        paths_long["trade_id"].isin(trade_ids) & (paths_long["bar_offset"] <= horizon_from_entry)
    ].copy()
    p = p.sort_values(["trade_id", "bar_offset"])

    # Delayed entry at bar_offset=t's OPEN (consistent with verbatim entry-at-OPEN convention)
    new_entry_row = p[p["bar_offset"] == t].set_index("trade_id")
    new_entry_price = new_entry_row["open"].reindex(trade_ids).values
    new_entry_cum_log = new_entry_row["cum_logret_from_entry"].reindex(trade_ids).values

    static = signals.set_index("trade_id").loc[
        trade_ids, ["atr_at_signal_1h", "spread_cost_R", "atr_ratio_to_baseline"]
    ]
    atr_sig = static["atr_at_signal_1h"].values
    spread_R = static["spread_cost_R"].values
    atr_ratio_signal = static["atr_ratio_to_baseline"].values

    # ATR at bar t = atr_at_signal × (atr_regime_ratio_at_t / atr_ratio_to_baseline_at_signal)
    if held_ctx_t is not None:
        held_t = held_ctx_t.set_index("trade_id").reindex(trade_ids)
        regime_col = f"atr_regime_ratio_t{t}"
        regime_ratio_at_t = held_t[regime_col].values if regime_col in held_t.columns else None
    else:
        regime_ratio_at_t = None

    if regime_ratio_at_t is not None:
        # atr_regime_ratio = ATR_t / ATR_baseline (consistent with signal-time atr_ratio_to_baseline)
        scale = np.where(
            (atr_ratio_signal > 0) & ~np.isnan(regime_ratio_at_t),
            regime_ratio_at_t / atr_ratio_signal,
            1.0,
        )
    else:
        scale = np.ones_like(atr_sig)
    atr_at_t = atr_sig * scale

    new_sl_dist_price = 2.0 * atr_at_t

    # For each trade, iterate bars t+1 .. 120 to detect SL relative to new entry.
    # SL price = new_entry_price - 2 ATR_at_t.
    # SL hit at bar k > t if low at bar k <= SL price.
    after_t = p[p["bar_offset"] > t].copy()
    after_t = after_t.merge(
        pd.DataFrame(
            {
                "trade_id": trade_ids,
                "new_entry_price": new_entry_price,
                "new_sl_price": new_entry_price - new_sl_dist_price,
                "new_entry_cum_log": new_entry_cum_log,
                "new_sl_dist_price": new_sl_dist_price,
            }
        ),
        on="trade_id",
        how="left",
    )
    after_t["sl_hit"] = after_t["low"] <= after_t["new_sl_price"]
    sl_first_after_t = after_t[after_t["sl_hit"]].groupby("trade_id")["bar_offset"].min()
    sl_first_arr = sl_first_after_t.reindex(trade_ids).values

    # Exit at bar_offset = horizon_from_entry's OPEN
    last_row_120 = p[p["bar_offset"] == horizon_from_entry].set_index("trade_id").reindex(trade_ids)
    exit_price_120 = last_row_120["open"].values
    mfe_120 = last_row_120["mfe_to_date_atr"].values
    mae_120 = last_row_120["mae_to_date_atr"].values

    action_bar = np.empty(len(trade_ids), dtype=np.float64)
    gross_r = np.empty(len(trade_ids), dtype=np.float64)
    exit_reason = np.empty(len(trade_ids), dtype=object)
    mfe_out = np.empty(len(trade_ids), dtype=np.float64)
    mae_out = np.empty(len(trade_ids), dtype=np.float64)

    for i, tid in enumerate(trade_ids):
        if np.isnan(new_entry_price[i]):
            action_bar[i] = np.nan
            gross_r[i] = np.nan
            exit_reason[i] = "missing_bar_t"
            mfe_out[i] = np.nan
            mae_out[i] = np.nan
            continue

        sl_b = sl_first_arr[i]
        if pd.notna(sl_b):
            action_bar[i] = float(sl_b)
            gross_r[i] = -1.0  # SL hit relative to new entry
            exit_reason[i] = "sl_hit_post_delayed_entry"
        else:
            if (
                np.isnan(exit_price_120[i])
                or new_sl_dist_price[i] <= 0
                or np.isnan(new_entry_price[i])
            ):
                action_bar[i] = float("nan")
                gross_r[i] = np.nan
                exit_reason[i] = "missing_h120"
            else:
                action_bar[i] = float(horizon_from_entry)
                # Linear P&L: (exit_open_at_120 - new_entry_open_at_t) / new_sl_dist_price
                gross_r[i] = float((exit_price_120[i] - new_entry_price[i]) / new_sl_dist_price[i])
                exit_reason[i] = "time_exit_delayed"

        mfe_out[i] = float(mfe_120[i]) if not np.isnan(mfe_120[i]) else np.nan
        mae_out[i] = float(mae_120[i]) if not np.isnan(mae_120[i]) else np.nan

    net_r = gross_r - spread_R
    return pd.DataFrame(
        {
            "trade_id": trade_ids,
            "action_bar": action_bar,
            "exit_reason": exit_reason,
            "gross_r": gross_r,
            "net_r": net_r,
            "spread_cost_r": spread_R,
            "mfe_at_exit": mfe_out,
            "mae_at_exit": mae_out,
        }
    )

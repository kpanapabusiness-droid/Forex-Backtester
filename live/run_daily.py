"""
Entry point: python -m live.run_daily
Load MT5 market/positions/history, run frozen v1_system on imported data,
apply exposure gate, emit reports to live_out/.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from core.backtester import apply_indicators_with_cache, load_config
from core.signal_logic import apply_signal_logic
from core.utils import normalize_ohlcv_schema
from live.exposure_gate import apply_exposure_gate
from live.mt5_import import (
    canonical_df_for_engine,
    get_market_export_dates,
    load_history_csv,
    load_market_tsvs,
    load_positions_csv,
    symbol_to_config_pair,
)
from live.reporting import (
    write_actions_csv,
    write_daily_positions_csv,
    write_daily_summary,
    write_orders_csv,
    write_trade_history_csv,
)

LIVE_FEED_MARKET = Path("live_feed/market")
LIVE_FEED_POSITIONS = Path("live_feed/positions/mt5_positions.csv")
LIVE_FEED_HISTORY = Path("live_feed/history/mt5_trade_history.csv")
LIVE_OUT = Path("live_out")
CONFIG_PATH = Path("configs/v1_system.yaml")

TP1_ATR_MULT = 1.0
SL_ATR_MULT = 1.5
RISK_PCT_PER_ORDER = 0.25


def _select_closed_bar_index(
    signals_df: pd.DataFrame,
    *,
    today: pd.Timestamp | None = None,
    export_day: str | None = None,
) -> int:
    """
    Return the index of the last CLOSED daily bar.

    If the last row's date equals the reference day (export_day if provided,
    else today at run time) and there is at least one prior row, treat the
    last bar as forming and use the second-to-last row. Otherwise use the
    last row. export_day (YYYY-MM-DD from file mtime in Australia/Melbourne)
    makes stale mid-candle exports use the previous closed bar.
    """
    n = len(signals_df)
    if n == 0:
        raise ValueError("signals_df must have at least one row")

    if "date" not in signals_df.columns:
        return n - 1

    dates = pd.to_datetime(signals_df["date"], errors="coerce")
    last_idx = n - 1
    last_date = dates.iloc[last_idx]

    if export_day is not None:
        reference = pd.Timestamp(export_day).normalize()
    elif today is not None:
        reference = today.normalize()
    else:
        reference = pd.Timestamp.now("UTC").normalize()

    if pd.notna(last_date) and last_date.normalize() == reference and last_idx >= 1:
        return last_idx - 1
    return last_idx


def _last_bar_signal(
    df: pd.DataFrame, pair: str, cfg: dict, *, export_day: str | None = None
) -> dict[str, Any]:
    """Run indicators + signal logic on df, return closed-bar entry_signal, exit_signal, atr, date, close."""
    engine_df = canonical_df_for_engine(df)
    if engine_df.empty or len(engine_df) < 2:
        return {}
    engine_df = normalize_ohlcv_schema(engine_df.copy())
    engine_df = engine_df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    engine_df["pair"] = pair
    base = apply_indicators_with_cache(engine_df.copy(), pair, cfg)
    signals_df = apply_signal_logic(base, cfg)
    for col in ["entry_signal", "exit_signal", "exit_signal_final"]:
        if col in signals_df.columns:
            signals_df[col] = (
                pd.to_numeric(signals_df[col], errors="coerce").fillna(0).clip(-1, 1).astype(int)
            )
    if len(signals_df) == 0:
        return {}

    closed_idx = _select_closed_bar_index(signals_df, export_day=export_day)
    closed_row = signals_df.iloc[closed_idx]
    n = len(signals_df)
    forming_ignored = closed_idx < n - 1
    atr = float(closed_row.get("atr", 0.0)) if "atr" in closed_row else 0.0
    entry_sig = int(closed_row.get("entry_signal", 0))
    exit_sig = int(closed_row.get("exit_signal_final", closed_row.get("exit_signal", 0)))
    date_val = closed_row.get("date")
    close_val = float(closed_row.get("close", 0.0))
    return {
        "entry_signal": entry_sig,
        "exit_signal": exit_sig,
        "atr": atr,
        "date": date_val,
        "close": close_val,
        "forming_ignored": forming_ignored,
    }


def _open_positions_symbols(positions_df: pd.DataFrame) -> set[str]:
    """Return set of symbols (6-letter) from positions dataframe."""
    if positions_df is None or (hasattr(positions_df, "empty") and positions_df.empty):
        return set()
    out = set()
    for _, row in positions_df.iterrows():
        sym = str(row.get("symbol", "")).upper().replace("_", "")[:6]
        if len(sym) >= 6:
            out.add(sym)
    return out


def run_daily(
    market_dir: Path | None = None,
    positions_path: Path | None = None,
    history_path: Path | None = None,
    out_dir: Path | None = None,
    config_path: Path | None = None,
) -> None:
    market_dir = market_dir or LIVE_FEED_MARKET
    positions_path = positions_path or LIVE_FEED_POSITIONS
    history_path = history_path or LIVE_FEED_HISTORY
    out_dir = out_dir or LIVE_OUT
    config_path = config_path or CONFIG_PATH

    cfg = load_config(config_path)
    pairs_cfg = cfg.get("pairs") or []
    run_id = (cfg.get("output") or {}).get("slug", "live-daily")
    ts_utc = pd.Timestamp.now("UTC")
    timestamp_utc = f"{ts_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    timestamp_melbourne = (
        f"{ts_utc.tz_convert('Australia/Melbourne').strftime('%Y-%m-%d %H:%M:%S')} Australia/Melbourne"
    )
    timestamp = ts_utc.strftime("%Y-%m-%d %H:%M:%S")

    market_dfs = load_market_tsvs(market_dir)
    export_dates = get_market_export_dates(market_dir)
    positions_df = load_positions_csv(positions_path)
    history_df = load_history_csv(history_path)

    open_symbols = _open_positions_symbols(positions_df)

    if not market_dfs:
        per_symbol = {}
        for pair in pairs_cfg:
            sym = pair.replace("_", "")[:6]
            per_symbol[sym] = {"action": "HOLD", "reason": "no_market_data"}
        write_daily_summary(
            out_dir, timestamp_utc, timestamp_melbourne, run_id, per_symbol,
            closed_d1_date=None, forming_ignored=False,
        )
        write_orders_csv(out_dir, [], run_id)
        write_actions_csv(out_dir, [], run_id)
        write_daily_positions_csv(out_dir, [], run_id)
        write_trade_history_csv(out_dir, [], run_id, history_df)
        return

    last_signals: dict[str, dict] = {}
    for symbol, df in market_dfs.items():
        pair = symbol_to_config_pair(symbol)
        if pair not in pairs_cfg:
            continue
        last_signals[symbol] = _last_bar_signal(
            df, pair, cfg, export_day=export_dates.get(symbol)
        )

    closed_d1_dates = [
        pd.Timestamp(s["date"]).strftime("%Y-%m-%d")
        for s in last_signals.values()
        if s and s.get("date") is not None and pd.notna(s.get("date"))
    ]
    closed_d1_date = max(closed_d1_dates) if closed_d1_dates else None
    forming_ignored = any(s.get("forming_ignored", False) for s in last_signals.values())

    candidate_signals = {}
    for symbol, sig in last_signals.items():
        if not sig or sig.get("entry_signal", 0) == 0:
            continue
        if symbol in open_symbols:
            continue
        candidate_signals[symbol] = sig["entry_signal"]

    approved, skipped = apply_exposure_gate(positions_df, candidate_signals)

    entry_cfg = cfg.get("entry") or {}
    sl_atr = float(entry_cfg.get("sl_atr", SL_ATR_MULT))
    tp1_atr = float(entry_cfg.get("tp1_atr", TP1_ATR_MULT))

    per_symbol: dict[str, dict[str, Any]] = {}
    orders: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    daily_positions: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []

    for symbol in sorted(set(market_dfs.keys()) | open_symbols):
        sig = last_signals.get(symbol, {})
        exit_sig = sig.get("exit_signal", 0)
        atr = sig.get("atr", 0.0)
        close = sig.get("close", 0.0)
        date_val = sig.get("date", timestamp)

        if symbol in open_symbols:
            # Existing MT5 position: we never change engine decisions here, only report.
            if exit_sig != 0:
                per_symbol[symbol] = {
                    "action": "EXIT",
                    "reason": "signal",
                    # Phase 8 SL reporting scaffold: human should close at market.
                    "sl_action": "CLOSE_EXIT",
                    "new_sl_price": "",
                    "sl_reason": "exit_signal",
                }
                actions.append(
                    {
                        "date_time": timestamp,
                        "action": "EXIT",
                        "symbol": symbol,
                        "reason": "signal",
                        "run_id": run_id,
                    }
                )
            else:
                # We do not have full trade state here; default to HOLD for SL.
                per_symbol[symbol] = {
                    "action": "HOLD",
                    "reason": "open_position",
                    "sl_action": "HOLD",
                    "new_sl_price": "",
                    "sl_reason": "open_position",
                }
                actions.append(
                    {
                        "date_time": timestamp,
                        "action": "HOLD",
                        "symbol": symbol,
                        "reason": "open_position",
                        "run_id": run_id,
                    }
                )

            daily_positions.append(
                {
                    "symbol": symbol,
                    "position_type": "open",
                    "decision": per_symbol[symbol]["action"],
                    "reason": per_symbol[symbol].get("reason", ""),
                    "sl_action": per_symbol[symbol].get("sl_action", ""),
                    "new_sl_price": per_symbol[symbol].get("new_sl_price", ""),
                    "run_id": run_id,
                }
            )
            continue

        if symbol in approved:
            direction_int = approved[symbol]
            direction = "long" if direction_int > 0 else "short"
            if not math.isfinite(atr) or atr <= 0:
                per_symbol[symbol] = {"action": "HOLD", "reason": "invalid_atr"}
                actions.append({
                    "date_time": timestamp, "action": "HOLD", "symbol": symbol,
                    "reason": "invalid_atr", "run_id": run_id,
                })
                continue
            tp1_price = close + direction_int * (tp1_atr * atr)
            sl_price = close - direction_int * (sl_atr * atr)
            per_symbol[symbol] = {
                "action": "OPEN",
                "reason": "approved",
                "direction": direction,
                "sl_price": sl_price,
                "tp1_price": tp1_price,
            }
            date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d %H:%M:%S") if date_val else timestamp
            orders.append({
                "date_time": date_str, "symbol": symbol, "direction": direction,
                "risk_pct": RISK_PCT_PER_ORDER, "sl_price": sl_price, "tp_price": tp1_price,
                "tag": "TP1", "run_id": run_id,
            })
            orders.append({
                "date_time": date_str, "symbol": symbol, "direction": direction,
                "risk_pct": RISK_PCT_PER_ORDER, "sl_price": sl_price, "tp_price": "",
                "tag": "RUNNER", "run_id": run_id,
            })
            signal_id = f"{symbol}_{direction}_{pd.to_datetime(date_val).strftime('%Y%m%d')}"
            ledger_rows.append({
                "signal_id": signal_id, "run_id": run_id, "date_time_decision": date_str,
                "symbol": symbol, "direction": direction,
                "sl_price": sl_price, "tp1_price": tp1_price,
                "tp1_ticket": "", "runner_ticket": "",
                "entry_price_tp1": "", "entry_price_runner": "",
                "entry_time": "", "exit_time_tp1": "", "exit_time_runner": "",
                "exit_price_tp1": "", "exit_price_runner": "",
                "status": "open", "reason": "approved",
            })
            daily_positions.append(
                {
                    "symbol": symbol,
                    "position_type": "new",
                    "decision": "OPEN",
                    "reason": "approved",
                    # Phase 8: we only provide static SL plan for new entries, not dynamic SL updates.
                    "sl_action": "",
                    "new_sl_price": "",
                    "run_id": run_id,
                }
            )
        elif symbol in skipped:
            per_symbol[symbol] = {"action": "SKIP", "reason": skipped[symbol]}
            actions.append({
                "date_time": timestamp, "action": "SKIP", "symbol": symbol,
                "reason": skipped[symbol], "run_id": run_id,
            })
            daily_positions.append(
                {
                    "symbol": symbol,
                    "position_type": "none",
                    "decision": "SKIP",
                    "reason": skipped[symbol],
                    "sl_action": "",
                    "new_sl_price": "",
                    "run_id": run_id,
                }
            )
        else:
            per_symbol[symbol] = {"action": "HOLD", "reason": "no_signal"}
            actions.append({
                "date_time": timestamp, "action": "HOLD", "symbol": symbol,
                "reason": "no_signal", "run_id": run_id,
            })
            daily_positions.append(
                {
                    "symbol": symbol,
                    "position_type": "none",
                    "decision": "HOLD",
                    "reason": "no_signal",
                    "sl_action": "",
                    "new_sl_price": "",
                    "run_id": run_id,
                }
            )

    for _, row in positions_df.iterrows():
        sym = str(row.get("symbol", "")).upper().replace("_", "")[:6]
        if sym and sym not in [r.get("symbol") for r in daily_positions]:
            daily_positions.append(
                {
                    "symbol": sym,
                    "position_type": "open",
                    "decision": "HOLD",
                    "reason": "mt5_position",
                    "sl_action": "HOLD",
                    "new_sl_price": "",
                    "run_id": run_id,
                }
            )

    write_daily_summary(
        out_dir, timestamp_utc, timestamp_melbourne, run_id, per_symbol,
        closed_d1_date=closed_d1_date, forming_ignored=forming_ignored,
    )
    write_orders_csv(out_dir, orders, run_id)
    write_actions_csv(out_dir, actions, run_id)
    write_daily_positions_csv(out_dir, daily_positions, run_id)
    write_trade_history_csv(out_dir, ledger_rows, run_id, history_df)


if __name__ == "__main__":
    run_daily()

"""
Write live bridge outputs: daily_summary.txt, orders.csv, actions.csv,
daily_positions.csv, trade_history.csv (append/update ledger).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


def write_daily_summary(
    out_dir: str | Path,
    timestamp_utc: str,
    timestamp_melbourne: str,
    run_id: str,
    per_symbol: dict[str, dict[str, Any]],
    *,
    closed_d1_date: str | None = None,
    forming_ignored: bool = False,
) -> None:
    """
    Write live_out/daily_summary.txt (human readable).

    per_symbol[symbol] schema (minimal):
      - action: "OPEN" | "EXIT" | "HOLD" | "SKIP"
      - reason: str

    Optional Phase 8 fields:
      - direction: "long" | "short"
      - sl_price: float
      - tp1_price: float
      - sl_action: "HOLD" | "MOVE_TO_BE" | "TRAIL_TO" | "CLOSE_SCRATCH" | "CLOSE_EXIT"
      - new_sl_price: float | ""
      - sl_reason: str

    These extras are reporting-only; they do not affect engine trading decisions.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "daily_summary.txt"
    closed_line = (
        f"Signals computed for CLOSED D1 candle date: {closed_d1_date}"
        if closed_d1_date
        else "Signals computed for CLOSED D1 candle date: N/A"
    )
    forming_line = f"Forming candle ignored: {'yes' if forming_ignored else 'no'}"
    lines = [
        f"timestamp_utc: {timestamp_utc}",
        f"timestamp_melbourne: {timestamp_melbourne}",
        closed_line,
        forming_line,
        f"run_id: {run_id}",
        "",
    ]
    for sym in sorted(per_symbol.keys()):
        rec = per_symbol[sym]
        action = rec.get("action", "HOLD")
        reason = rec.get("reason", "")

        # Primary action line
        line = f"  {sym}: {action}"
        if reason:
            line += f" — {reason}"
        if action == "OPEN":
            direction = rec.get("direction", "")
            sl = rec.get("sl_price")
            tp1 = rec.get("tp1_price")
            line += f" direction={direction} SL={sl} TP1={tp1} (2 orders: TP1 + RUNNER)"
        lines.append(line)

        # Phase 8: per-symbol SL management / reconciliation hints
        sl_action = rec.get("sl_action")
        new_sl = rec.get("new_sl_price")
        sl_reason = rec.get("sl_reason") or ""

        if action == "OPEN":
            # Static SL management plan for new trades (TP1→BE + trailing)
            if sl is not None and tp1 is not None:
                lines.append(
                    "    SL management: "
                    "After TP1 hits at TP1, move runner SL to entry; "
                    "once price has moved +2×ATR from entry, trail SL 1.5×ATR behind close."
                )
        # For existing positions or anything providing explicit SL action, surface it
        if sl_action:
            detail = f"    sl_action={sl_action}"
            if new_sl not in (None, ""):
                detail += f" new_sl_price={new_sl}"
            if sl_reason:
                detail += f" reason={sl_reason}"
            lines.append(detail)

    path.write_text("\n".join(lines), encoding="utf-8")


def write_orders_csv(
    out_dir: str | Path,
    orders: list[dict[str, Any]],
    run_id: str,
) -> None:
    """
    Write live_out/orders.csv. Exactly 2 rows per approved OPEN: TP1 + RUNNER.
    Columns: date_time,symbol,direction,risk_pct,sl_price,tp_price,tag,run_id
    TP1 order has tp_price filled; RUNNER tp_price blank. risk_pct 0.25 each.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "orders.csv"
    cols = ["date_time", "symbol", "direction", "risk_pct", "sl_price", "tp_price", "tag", "run_id"]
    if not orders:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in orders:
            out_row = {c: row.get(c, "") for c in cols}
            w.writerow(out_row)


def write_actions_csv(
    out_dir: str | Path,
    actions: list[dict[str, Any]],
    run_id: str,
) -> None:
    """
    Write live_out/actions.csv. Columns: date_time,action,symbol,reason,run_id
    action in EXIT, HOLD, SKIP (OPEN decisions are in orders.csv).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "actions.csv"
    cols = ["date_time", "action", "symbol", "reason", "run_id"]
    if not actions:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in actions:
            out_row = {c: row.get(c, "") for c in cols}
            w.writerow(out_row)


def write_daily_positions_csv(
    out_dir: str | Path,
    snapshot: list[dict[str, Any]],
    run_id: str,
) -> None:
    """
    Write live_out/daily_positions.csv: current MT5 positions plus today's
    decisions (OPEN/EXIT/HOLD/SKIP).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "daily_positions.csv"
    if not snapshot:
        path.write_text("symbol,position_type,decision,reason,run_id\n", encoding="utf-8")
        return
    df = pd.DataFrame(snapshot)
    df.to_csv(path, index=False)


def trade_history_columns() -> list[str]:
    """Column list for trade_history.csv ledger."""
    return [
        "signal_id", "run_id", "date_time_decision", "symbol", "direction",
        "sl_price", "tp1_price",
        "tp1_ticket", "runner_ticket",
        "entry_price_tp1", "entry_price_runner",
        "entry_time", "exit_time_tp1", "exit_time_runner",
        "exit_price_tp1", "exit_price_runner",
        "status", "reason",
    ]


def write_trade_history_csv(
    out_dir: str | Path,
    ledger_rows: list[dict[str, Any]],
    run_id: str,
    mt5_history_df: pd.DataFrame | None,
) -> None:
    """
    Write/update live_out/trade_history.csv. One row per signal_id.
    Update exit fields from mt5_trade_history when trades close.
    If no MT5 tickets exist, still append signal row with blank ticket fields.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "trade_history.csv"
    cols = trade_history_columns()

    existing = pd.DataFrame()
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            pass
    if not existing.empty and not all(c in existing.columns for c in cols):
        existing = pd.DataFrame()

    new_df = pd.DataFrame(ledger_rows)
    if new_df.empty:
        if existing.empty:
            pd.DataFrame(columns=cols).to_csv(path, index=False)
        return
    for c in cols:
        if c not in new_df.columns:
            new_df[c] = ""
    new_df = new_df[cols]

    if mt5_history_df is not None and not mt5_history_df.empty:
        ticket_to_close = {}
        for _, close_row in mt5_history_df.iterrows():
            ticket = str(close_row.get("ticket", ""))
            if ticket:
                ticket_to_close[ticket] = {
                    "close_time": close_row.get("close_time", ""),
                    "close_price": close_row.get("close_price", ""),
                }
        for idx in new_df.index:
            tp1_ticket = str(new_df.at[idx, "tp1_ticket"] or "")
            runner_ticket = str(new_df.at[idx, "runner_ticket"] or "")
            if tp1_ticket and tp1_ticket in ticket_to_close:
                new_df.at[idx, "exit_time_tp1"] = ticket_to_close[tp1_ticket]["close_time"]
                new_df.at[idx, "exit_price_tp1"] = ticket_to_close[tp1_ticket]["close_price"]
            if runner_ticket and runner_ticket in ticket_to_close:
                new_df.at[idx, "exit_time_runner"] = ticket_to_close[runner_ticket]["close_time"]
                new_df.at[idx, "exit_price_runner"] = ticket_to_close[runner_ticket]["close_price"]

    if not existing.empty:
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["signal_id"], keep="last")
    else:
        merged = new_df
    merged[cols].to_csv(path, index=False)

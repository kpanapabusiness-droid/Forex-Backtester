"""
Phase D-3/D-4 — Minimal ROI backtest of sequence proto-signals.

Runs full-universe backtest with external entry signals from D-2.5.
Entry at next bar open; SL/TP/trailing per locked execution.
Phase D-4: time stop removed; position exits only via SL, TP1, or trailing stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtester import (  # noqa: E402
    TRADES_COLS,
    simulate_pair_trades,
    write_trades_csv_with_diagnostics,
)
from core.utils import (  # noqa: E402
    calculate_atr,
    normalize_ohlcv_schema,
    slice_df_by_dates,
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _require_config(raw: dict, signal_name: str | None) -> dict:
    cfg = dict(raw or {})
    ext = cfg.get("external_signals") or {}
    if not ext.get("path"):
        raise ValueError("Phase D3 config must set external_signals.path.")
    sig = signal_name or ext.get("signal_name")
    if not sig:
        raise ValueError("Phase D3 requires --signal CLI argument or external_signals.signal_name in config.")
    cfg["external_signals_path"] = str(ext["path"])
    cfg["external_signal_name"] = str(sig)
    return cfg


def _apply_breakout_filter(
    signals: pd.DataFrame,
    features: pd.DataFrame,
    *,
    breakout_up_col: str,
    breakout_dn_col: str,
) -> pd.DataFrame:
    """
    Gate signals by breakout confirmation: long only if breakout_up==1, short only if breakout_dn==1.
    NaN or missing confirmation -> fail closed (signal becomes 0).
    """
    out = signals.merge(
        features[["pair", "date", breakout_up_col, breakout_dn_col]],
        on=["pair", "date"],
        how="left",
        suffixes=("", "_feat"),
    )
    up_val = out[breakout_up_col]
    dn_val = out[breakout_dn_col]
    keep_long = (out["direction"] == "long") & (up_val == 1) & up_val.notna()
    keep_short = (out["direction"] == "short") & (dn_val == 1) & dn_val.notna()
    out["signal"] = out["signal"].where(keep_long | keep_short, 0)
    out["signal"] = out["signal"].fillna(0).astype(int)
    return out.drop(columns=[breakout_up_col, breakout_dn_col], errors="ignore")


def _load_and_validate_entry_filter(
    cfg: dict, signals: pd.DataFrame
) -> tuple[pd.DataFrame | None, str, str]:
    """
    If entry_filter.require_breakout_confirm: load features, validate, return (feats, up_col, dn_col).
    Else return (None, "", "").
    """
    entry_cfg = cfg.get("entry_filter") or {}
    if not entry_cfg.get("require_breakout_confirm"):
        return None, "", ""
    features_path_val = entry_cfg.get("features_path")
    if not features_path_val or not str(features_path_val).strip():
        raise ValueError(
            "entry_filter.require_breakout_confirm enabled but entry_filter.features_path is missing."
        )
    path = Path(features_path_val)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"entry_filter features_path not found: {path}")
    up_col = entry_cfg.get("breakout_up_col", "breakout_up_20")
    dn_col = entry_cfg.get("breakout_dn_col", "breakout_dn_20")
    feats = pd.read_parquet(path)
    feats["date"] = pd.to_datetime(feats["date"])
    for col in [up_col, dn_col]:
        if col not in feats.columns:
            raise ValueError(
                f"entry_filter: breakout column '{col}' not found in features. "
                f"Available: {list(feats.columns)}"
            )
    sig_dates = set(signals["date"].dt.normalize())
    feat_dates = set(feats["date"].dt.normalize())
    if not sig_dates & feat_dates:
        raise ValueError(
            "entry_filter: no overlapping dates between signals and features. "
            "Ensure features cover the signal date range."
        )
    return feats, up_col, dn_col


def _load_external_signals(path: Path, signal_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"External signals not found: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if "signal_name" not in df.columns:
        raise ValueError("External signals must have 'signal_name' column.")
    sub = df[df["signal_name"] == signal_name].copy()
    if sub.empty:
        raise ValueError(
            f"Signal '{signal_name}' not found in {path}. "
            f"Available: {df['signal_name'].unique().tolist()}"
        )
    return sub


def _signals_to_entry_intent(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Convert (pair, date, direction, signal) to (pair, date, entry_signal).
    entry_signal: +1 long, -1 short, 0 none.
    If both long and short fire for same (pair, date), prefer long (deterministic).
    """
    rows = []
    for (pair, date), grp in signals.groupby(["pair", "date"]):
        entry = 0
        long_fire = (grp["direction"] == "long") & (grp["signal"] == 1)
        short_fire = (grp["direction"] == "short") & (grp["signal"] == 1)
        if long_fire.any():
            entry = 1
        elif short_fire.any():
            entry = -1
        rows.append({"pair": pair, "date": date, "entry_signal": entry})
    return pd.DataFrame(rows)


def _validate_d4_no_time_stop(cfg: dict) -> None:
    """Phase D-4: time stop must be disabled (null or 0)."""
    exit_cfg = cfg.get("exit") or {}
    ts = exit_cfg.get("time_stop_bars")
    if ts is not None and int(ts) > 0:
        raise ValueError(
            f"Phase D-4 requires time_stop_bars disabled. Got time_stop_bars={ts}. "
            "Set exit.time_stop_bars: null to run without time stop."
        )


def _validate_d41_tp1_move_sl_required(cfg: dict) -> None:
    """Phase D-4.1: tp1_move_sl_to_be must be explicitly set (no accidental default)."""
    exit_cfg = cfg.get("exit") or {}
    if "tp1_move_sl_to_be" not in exit_cfg:
        raise ValueError(
            "Phase D-4.1 requires exit.tp1_move_sl_to_be to be explicitly set. "
            "Add exit.tp1_move_sl_to_be: true or false to config."
        )


def _run_from_config(config_path: Path, signal_name: str | None) -> None:
    raw = _load_yaml(config_path)
    cfg = _require_config(raw, signal_name)
    _validate_d4_no_time_stop(cfg)
    _validate_d41_tp1_move_sl_required(cfg)

    signal_name = cfg["external_signal_name"]
    pairs = cfg.get("pairs") or []
    if not pairs:
        raise ValueError("Config must set pairs list.")
    data_dir = Path(cfg.get("data_dir", "data/daily"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    date_start = (cfg.get("date_range") or {}).get("start")
    date_end = (cfg.get("date_range") or {}).get("end")
    if not date_start or not date_end:
        raise ValueError("Config must set date_range.start and date_range.end.")

    out_base = Path(cfg.get("outputs", {}).get("dir", "results/phaseD3"))
    if not out_base.is_absolute():
        out_base = ROOT / out_base
    run_slug = (cfg.get("outputs") or {}).get("run_slug")
    out_dir = out_base / (run_slug or signal_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_path = Path(cfg["external_signals_path"])
    if not signals_path.is_absolute():
        signals_path = ROOT / signals_path
    signals_df = _load_external_signals(signals_path, signal_name)
    feats, up_col, dn_col = _load_and_validate_entry_filter(cfg, signals_df)
    if feats is not None and up_col and dn_col:
        signals_df = _apply_breakout_filter(
            signals_df, feats, breakout_up_col=up_col, breakout_dn_col=dn_col
        )
    entry_intent = _signals_to_entry_intent(signals_df)

    starting_balance = float(
        (cfg.get("risk") or {}).get("starting_balance", 10_000.0)
    )
    equity_state = {"balance": starting_balance}
    track_equity = True
    all_trades: list = []
    equity_frames: list = []

    minimal_cfg = {
        "indicators": {"c1": None, "use_c2": False, "use_baseline": False, "use_volume": False, "use_exit": False},
        "entry": cfg.get("entry", {}),
        "exit": cfg.get("exit", {}),
        "risk": cfg.get("risk", {}),
        "spreads": cfg.get("spreads", {}),
        "filters": cfg.get("filters", {}),
    }

    for pair in pairs:
        cands = [
            f"{pair}.csv",
            f"{pair.upper()}.csv",
            f"{pair.replace('/', '_')}.csv",
        ]
        path = None
        for pat in cands:
            hit = list(data_dir.glob(pat))
            if hit:
                path = hit[0]
                break
        if path is None:
            continue
        df = pd.read_csv(path)
        df = normalize_ohlcv_schema(df)
        df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
        df["pair"] = pair

        df, _ = slice_df_by_dates(df, date_start, date_end)
        if df.empty:
            continue

        df = calculate_atr(df)
        pair_intent = entry_intent[entry_intent["pair"] == pair]
        if pair_intent.empty:
            df["entry_signal"] = 0
        else:
            merged = df[["pair", "date"]].merge(
                pair_intent[["date", "entry_signal"]],
                on="date",
                how="left",
            )
            df["entry_signal"] = merged["entry_signal"].fillna(0).astype(int)
        df["exit_signal"] = 0

        if track_equity:
            trades, eq = simulate_pair_trades(
                df, pair, minimal_cfg, equity_state, return_equity=True
            )
            all_trades.extend(trades)
            if eq is not None and not eq.empty:
                equity_frames.append(eq)
        else:
            trades = simulate_pair_trades(
                df, pair, minimal_cfg, equity_state, return_equity=False
            )
            all_trades.extend(trades)

    trades_df = (
        pd.DataFrame(all_trades, columns=TRADES_COLS)
        if all_trades
        else pd.DataFrame(columns=TRADES_COLS)
    )
    for c in TRADES_COLS:
        if c not in trades_df.columns:
            trades_df[c] = pd.NA
    trades_df = trades_df.reindex(columns=TRADES_COLS)

    write_trades_csv_with_diagnostics(
        trades_df, out_dir, minimal_cfg, run_slug=signal_name
    )

    if track_equity and equity_frames:
        eq = pd.concat(equity_frames, ignore_index=True)
        eq_wide = (
            eq.pivot_table(
                index="date", columns="pair", values="pnl_realized_cum", aggfunc="last"
            )
            .sort_index()
            .ffill()
            .fillna(0.0)
        )
        eq_wide["pnl_realized_cum_total"] = eq_wide.sum(axis=1, numeric_only=True)
        equity_curve = eq_wide[["pnl_realized_cum_total"]].rename(
            columns={"pnl_realized_cum_total": "equity"}
        )
        equity_curve["equity"] = starting_balance + equity_curve["equity"]
        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = equity_curve["equity"] - equity_curve["peak"]
        equity_curve = equity_curve.reset_index()
        equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)

    try:
        from core.utils import summarize_results

        txt, _ = summarize_results(
            out_dir, starting_balance=starting_balance
        )
    except Exception:
        total = len(trades_df)
        wins = int(trades_df.get("win", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if total else 0
        losses = int(trades_df.get("loss", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if total else 0
        roi = float(trades_df.get("pnl", 0).fillna(0).sum())
        txt = (
            f"Total Trades : {total}\n"
            f"Wins         : {wins}\n"
            f"Losses       : {losses}\n"
            f"ROI ($)      : {roi:.2f}\n"
            f"ROI (%)      : {(roi / starting_balance * 100):.2f}\n"
        )
    (out_dir / "summary.txt").write_text(txt or "", encoding="utf-8")

    print(f"Phase D3 backtest complete. Signal={signal_name} Outputs: {out_dir}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-3 — Sequence signal ROI backtest.",
    )
    parser.add_argument("-c", "--config", required=True, help="Path to D3 config YAML.")
    parser.add_argument(
        "--signal",
        default=None,
        help="Signal name (optional if config sets external_signals.signal_name).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    _run_from_config(config_path, args.signal if args.signal else None)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase F: CEB v3 ROI sanity run (SL=2R, trailing, two-leg). Prints compact summary from outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtester import load_config, run_backtest
from core.utils import summarize_results


def _compute_extra_metrics(results_dir: Path, cfg: dict) -> dict:
    """Compute profit_factor, expectancy_R, years, trades_per_year from trades/equity."""
    out: dict = {}
    trades_path = results_dir / "trades.csv"
    equity_path = results_dir / "equity_curve.csv"

    risk_cfg = cfg.get("risk") or {}
    starting_balance = float(risk_cfg.get("starting_balance", 10_000.0))
    risk_pct = risk_cfg.get("risk_per_trade")
    if risk_pct is None and "risk_per_trade_pct" in risk_cfg:
        risk_pct = float(risk_cfg["risk_per_trade_pct"]) / 100.0
    if risk_pct is None:
        risk_pct = 0.02
    risk_per_trade = float(starting_balance * risk_pct)

    if trades_path.exists():
        import pandas as pd

        df = pd.read_csv(trades_path)
        pnl = pd.to_numeric(df.get("pnl", 0), errors="coerce").fillna(0.0)
        win = df.get("win", pd.Series(dtype=bool))
        if "win" in df.columns:
            win = win.fillna(False).astype(bool)
        else:
            win = pnl > 0
        loss = df.get("loss", pd.Series(dtype=bool))
        if "loss" in df.columns:
            loss = loss.fillna(False).astype(bool)
        else:
            loss = pnl < 0

        gross_profit = float(pnl[win].sum())
        gross_loss = float(pnl[loss].sum())
        if gross_loss < 0:
            out["profit_factor"] = gross_profit / abs(gross_loss)
        else:
            out["profit_factor"] = float("inf") if gross_profit > 0 else 0.0
        if risk_per_trade > 0 and len(df) > 0:
            out["expectancy_R"] = float(pnl.sum()) / (len(df) * risk_per_trade)
            ns = (win | loss).sum()
            if ns > 0:
                out["avg_trade_R"] = float(pnl[win | loss].sum()) / (ns * risk_per_trade)
            else:
                out["avg_trade_R"] = 0.0
        else:
            out["expectancy_R"] = 0.0
            out["avg_trade_R"] = 0.0

    years = 0.0
    if equity_path.exists():
        import pandas as pd

        eq = pd.read_csv(equity_path)
        if "date" in eq.columns and len(eq) >= 2:
            eq["date"] = pd.to_datetime(eq["date"])
            days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
            years = max(0.0, days / 365.25)
    if years <= 0:
        dr = cfg.get("date_range") or {}
        start_s = dr.get("start") or "2019-01-01"
        end_s = dr.get("end") or "2026-01-01"
        try:
            from pandas import to_datetime

            days = (to_datetime(end_s) - to_datetime(start_s)).days
            years = max(0.0, days / 365.25)
        except Exception:
            years = 7.0

    out["years"] = years
    if years > 0 and trades_path.exists():
        import pandas as pd

        n = len(pd.read_csv(trades_path))
        out["trades_per_year"] = n / years
    else:
        out["trades_per_year"] = 0.0

    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    default_config = root / "configs" / "phaseF_roi_sanity" / "ceb_v3_sl2r_trailing.yaml"

    parser = argparse.ArgumentParser(description="Phase F: CEB v3 ROI sanity run (SL=2R, trailing, two-leg)")
    parser.add_argument("-c", "--config", type=str, default=str(default_config), help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()

    cfg = load_config(str(config_path))
    out_dir = (cfg.get("outputs") or {}).get("dir") or (cfg.get("output") or {}).get("results_dir") or "results"
    if not Path(out_dir).is_absolute():
        out_dir = str(root / out_dir)
    results_dir = Path(out_dir).resolve()

    run_backtest(config_path=str(config_path), results_dir=str(results_dir))

    _, metrics = summarize_results(str(results_dir), starting_balance=cfg.get("risk", {}).get("starting_balance", 10_000.0))
    extra = _compute_extra_metrics(results_dir, cfg)

    total = metrics.get("total_trades", 0)
    win_rate = metrics.get("win_rate_ns", 0.0)
    expectancy = metrics.get("expectancy", 0.0)
    roi_pct = metrics.get("roi_pct", 0.0)
    max_dd = metrics.get("max_dd_pct", 0.0)

    print("\n" + "=" * 60)
    print("CEB v3 ROI SANITY — Compact Summary")
    print("=" * 60)
    print(f"  total_trades      : {total}")
    print(f"  win_rate (ns) %   : {win_rate:.2f}")
    print(f"  expectancy ($)    : {expectancy:.2f}")
    print(f"  expectancy (R)    : {extra.get('expectancy_R', 0.0):.4f}")
    print(f"  net ROI %         : {roi_pct:.2f}")
    print(f"  max drawdown %    : {max_dd:.2f}")
    print(f"  profit_factor     : {extra.get('profit_factor', 0.0):.2f}")
    print(f"  avg trade R       : {extra.get('avg_trade_R', 0.0):.4f}")
    print(f"  years             : {extra.get('years', 0):.2f}")
    print(f"  trades/year       : {extra.get('trades_per_year', 0):.1f}")
    print("=" * 60)
    print(f"  output_dir        : {results_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

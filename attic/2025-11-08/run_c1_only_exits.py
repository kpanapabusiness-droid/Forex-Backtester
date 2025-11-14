#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtester import run_backtest  # noqa: E402


def build_run_config(pair: str, timeframe: str, start: str, end: str, c1_name: str) -> dict:
    data_dir = {
        "D": "data/daily",
        "daily": "data/daily",
        "H1": "data/hourly",
        "H4": "data/4h",
    }.get(timeframe, "data/daily")

    cfg = {
        "pairs": [pair],
        "timeframe": timeframe,
        "data_dir": data_dir,
        "indicators": {
            "c1": c1_name,
            "use_c2": False,
            "c2": None,
            "use_baseline": False,
            "baseline": None,
            "use_volume": False,
            "volume": None,
            "use_exit": False,
            "exit": None,
        },
        "rules": {
            "one_candle_rule": False,
            "pullback_rule": False,
            "bridge_too_far_days": 7,
            "allow_baseline_as_catalyst": False,
        },
        "exit": {
            "use_trailing_stop": True,
            "move_to_breakeven_after_atr": True,
            "exit_on_c1_reversal": True,
            "exit_on_baseline_cross": False,
            "exit_on_exit_signal": False,
        },
        "spreads": {"enabled": False, "default_pips": 0.0, "mode": "fixed", "atr_mult": 0.0},
        "tracking": {
            "in_sim_equity": True,
            "track_win_loss_scratch": True,
            "track_roi": True,
            "track_drawdown": True,
            "verbose_logs": False,
        },
        "cache": {"enabled": True, "dir": "cache", "format": "parquet", "scope_key": None, "roles": None},
        "validation": {"enabled": True, "fail_fast": True, "strict_contract": False},
        "date_range": {"start": start, "end": end},
        "output": {"results_dir": "results"},
        # Optional metadata for sweep semantics (engine ignores unknown keys safely):
        "classification": {"use_tp1_leg_only": True},
        "trades": {
            "two_legs_per_signal": True,
            "exit_on_c1_flip": True,
            "leg_a": {"tp1_enabled": True, "breakeven_after_tp1": False},
            "leg_b": {"tp1_enabled": False, "breakeven_after_tp1": True},
        },
        "execution": {"spread_pips": 0},
        # trailing_stop keys are advisory for sweep docs; engine uses entry.trail_after_atr defaults
        "trailing_stop": {"enabled": True, "leg_a": {"enabled": False}, "leg_b": {"enabled": True}},
    }

    return cfg


def main():
    ap = argparse.ArgumentParser(description="Run C1-only exits sweep")
    ap.add_argument("--config", required=True, help="Path to sweeps/c1_only_exits.yaml")
    args = ap.parse_args()

    sweeps_path = Path(args.config)
    if not sweeps_path.exists():
        print(f"Config not found: {sweeps_path}")
        sys.exit(2)

    with sweeps_path.open("r", encoding="utf-8") as f:
        sweeps = yaml.safe_load(f) or {}

    dfl = sweeps.get("defaults", {})
    pairs = dfl.get("pair") or []
    timeframe = str(dfl.get("timeframe", "D"))
    start = str(dfl.get("from"))
    end = str(dfl.get("to"))
    base_out = Path(dfl.get("output_dir", "results/c1_only_exits"))

    axes = sweeps.get("sweep", {}).get("axes", [])
    c1_axis = next((ax for ax in axes if ax.get("name") == "c1"), None)
    if not c1_axis:
        print("No c1 axis defined in sweep")
        sys.exit(2)
    c1_values = c1_axis.get("values") or []

    total = 0
    for pair in pairs:
        for c1 in c1_values:
            cfg = build_run_config(pair=pair, timeframe=timeframe, start=start, end=end, c1_name=c1)
            out_dir = base_out / pair / c1
            out_dir.mkdir(parents=True, exist_ok=True)
            # Print core flags summary per run
            print(
                f"Running {pair} | {c1} → {out_dir} | "
                f"spread=0, one_candle=false, baseline/volume/exit=disabled, "
                f"two_legs=true, legA(tp1=True, BE=False), legB(tp1=False, BE=True), "
                f"TS=enabled (engine activation defaults)"
            )
            run_backtest(cfg, results_dir=str(out_dir))
            total += 1

    print(f"\n✅ Completed {total} runs. Output root: {base_out}")


if __name__ == "__main__":
    main()



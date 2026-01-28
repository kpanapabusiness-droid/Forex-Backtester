#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_c1_volume_results.py
--------------------------------

Aggregate all C1 + exit_twiggs_money_flow + volume sweep runs into a single CSV.

Assumes per-run folders are named:

    c1_<c1_name>__vol_<volume_name>__exit_<exit_name>__pair_<pair>

and live under:

    results/c1_w_exit_plus_vol_results/

Usage:
    python scripts/aggregate_c1_volume_results.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import summarize_results  # noqa: E402

# Keep in sync with run_c1_volume_sweep.py
RESULTS_ROOT = PROJECT_ROOT / "results" / "c1_w_exit_plus_vol_results"


def parse_slug(name: str) -> Dict[str, str]:
    """
    Parse a run directory name of the form:
      c1_<c1>__vol_<vol>__exit_<exit>__pair_<pair>
    """
    meta: Dict[str, str] = {"c1": "", "volume": "", "exit": "", "pair": ""}

    # Quick pattern match
    pattern = r"^c1_(.+?)__vol_(.+?)__exit_(.+?)__pair_(.+)$"
    m = re.match(pattern, name)
    if m:
        meta["c1"] = m.group(1)
        meta["volume"] = m.group(2)
        meta["exit"] = m.group(3)
        meta["pair"] = m.group(4)
        return meta

    # Fallback: split manually
    parts = name.split("__")
    for part in parts:
        if part.startswith("c1_"):
            meta["c1"] = part[len("c1_") :]
        elif part.startswith("vol_"):
            meta["volume"] = part[len("vol_") :]
        elif part.startswith("exit_"):
            meta["exit"] = part[len("exit_") :]
        elif part.startswith("pair_"):
            meta["pair"] = part[len("pair_") :]
    return meta


def scan_runs(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not root.exists():
        print(f"Results root does not exist: {root}", file=sys.stderr)
        return rows

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        trades_path = run_dir / "trades.csv"
        summary_path = run_dir / "summary.txt"

        if not trades_path.exists() and not summary_path.exists():
            continue

        slug_meta = parse_slug(run_dir.name)
        c1_name = slug_meta["c1"]
        volume_name = slug_meta["volume"]
        exit_name = slug_meta["exit"]
        pair = slug_meta["pair"]

        # Use summarize_results as source of truth for metrics
        try:
            starting_balance = 10000.0
            cfg_path = run_dir / "config.yaml"
            if cfg_path.exists():
                import yaml

                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                risk_cfg = (cfg.get("risk") or {}) if isinstance(cfg, dict) else {}
                if "starting_balance" in risk_cfg:
                    starting_balance = float(risk_cfg["starting_balance"])

            _, metrics = summarize_results(run_dir, starting_balance=starting_balance)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: summarize_results failed for {run_dir}: {exc}", file=sys.stderr)
            metrics = {}

        row: Dict[str, Any] = {
            "pair": pair,
            "c1": c1_name,
            "volume": volume_name,
            "exit": exit_name,
            "run_dir": str(run_dir),
        }

        # Map common keys from summarize_results
        row["total_trades"] = metrics.get("total_trades", 0)
        row["wins"] = metrics.get("wins", 0)
        row["losses"] = metrics.get("losses", 0)
        row["scratches"] = metrics.get("scratches", 0)
        row["win_rate_ns"] = metrics.get("win_rate_ns", 0.0)
        row["roi_pct"] = metrics.get("roi_pct", 0.0)
        row["max_dd_pct"] = metrics.get("max_dd_pct", 0.0)
        row["mar"] = metrics.get("mar", 0.0)
        row["expectancy"] = metrics.get("expectancy", 0.0)
        row["sharpe"] = metrics.get("sharpe", 0.0)
        row["sortino"] = metrics.get("sortino", 0.0)

        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate C1 + exit_twiggs_money_flow + volume sweep results",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(RESULTS_ROOT),
        help="Root directory where c1_w_exit_plus_vol_results are stored",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path (default: results-root/c1_volume_aggregated.csv or results/c1_volume_aggregated_full.csv if results-root contains 'full')",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = scan_runs(root)

    if not rows:
        print(f"No runs found under {root}", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    
    # Auto-determine output path if not specified
    if args.output:
        out_path = Path(args.output)
    else:
        # If results root contains 'full', write to results/ with _full suffix
        if "full" in str(root).lower():
            out_path = PROJECT_ROOT / "results" / "c1_volume_aggregated_full.csv"
        else:
            out_path = root / "c1_volume_aggregated.csv"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote aggregated results to {out_path}")
    print(f"   Rows: {len(df)}, Unique C1s: {df['c1'].nunique() if 'c1' in df.columns else 'N/A'}")


if __name__ == "__main__":
    main()



# Phase 5 — C1 parameter WFO leaderboard. One row per (C1, params). Deterministic, rejection rules.
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Rejection-rule constants (explicit, documented). Any triggered => status=REJECT.
TRADES_MAX_CEILING = 50000
"""Absolute ceiling for trades_max; above this => REJECT (absurdly high vs baseline identity)."""

ROI_COLLAPSE_THRESHOLD_PCT = 25.0
"""If (median_fold_roi - worst_fold_roi) > this %, REJECT (ROI collapses in one fold)."""

SCRATCH_RATE_MEAN_CEILING = 0.5
"""Scratch rate (fraction of trades that are scratches) mean across folds; above => REJECT."""

LEADERBOARD_COLUMNS = [
    "c1_name",
    "params_hash",
    "param_description",
    "worst_fold_roi",
    "worst_fold_dd",
    "median_fold_roi",
    "roi_std",
    "trades_min",
    "trades_max",
    "scratch_rate_mean",
    "scratch_rate_std",
    "status",
    "reject_reason",
]


def _params_fingerprint(c1_name: str, params: dict) -> str:
    blob = json.dumps({"c1": c1_name, "params": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _param_description(params: dict) -> str:
    if not params:
        return ""
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    return "; ".join(parts)


def _parse_fold_metrics(oos_dir: Path) -> dict[str, Any]:
    from scripts.batch_sweeper import parse_summary_or_trades  # noqa: E402

    metrics = parse_summary_or_trades(oos_dir)
    roi_pct = float(metrics.get("roi_pct") or 0.0)
    max_dd_pct = float(metrics.get("max_dd_pct") or 0.0)
    total_trades = int(metrics.get("total_trades") or 0)
    scratches = int(metrics.get("scratches") or 0)
    scratch_rate = (scratches / total_trades) if total_trades > 0 else 0.0
    return {
        "roi_pct": roi_pct,
        "max_dd_pct": max_dd_pct,
        "trades": total_trades,
        "scratches": scratches,
        "scratch_rate": scratch_rate,
    }


def _discover_runs(phase5_root: Path) -> list[tuple[str, Path]]:
    """Return (c1_name, run_dir) for each run under results/phase5/<c1_name>/."""
    phase5_root = phase5_root.resolve()
    if not phase5_root.exists():
        return []
    runs: list[tuple[str, Path]] = []
    for c1_dir in sorted(phase5_root.iterdir()):
        if not c1_dir.is_dir():
            continue
        c1_name = c1_dir.name
        for run_dir in sorted(c1_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            fold_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")]
            oos_exists = any((f / "out_of_sample").exists() for f in fold_dirs)
            if oos_exists:
                runs.append((c1_name, run_dir))
    return runs


def _load_base_config_used(run_dir: Path) -> tuple[str, dict]:
    """Load base_config_used.yaml; return (c1_name, indicator_params for c1)."""
    path = run_dir / "base_config_used.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing base_config_used.yaml in {run_dir}")
    import yaml  # noqa: E402

    with path.open("r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    ind = cfg.get("indicators") or {}
    c1_name = ind.get("c1")
    if not c1_name:
        raise ValueError(f"base_config_used.yaml in {run_dir} has no indicators.c1")
    params = (cfg.get("indicator_params") or {}).get(c1_name) or {}
    return str(c1_name), dict(params)


def _fold_metrics_for_run(run_dir: Path) -> list[dict]:
    """Parse each fold_XX/out_of_sample in run_dir; return list of metric dicts (sorted by fold)."""
    fold_dirs = sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and p.name.startswith("fold_")
    )
    rows: list[dict] = []
    for fold_dir in fold_dirs:
        oos_dir = fold_dir / "out_of_sample"
        if not oos_dir.exists():
            continue
        try:
            row = _parse_fold_metrics(oos_dir)
            idx_str = fold_dir.name.split("_", 1)[1]
            row["fold_idx"] = int(idx_str)
            rows.append(row)
        except Exception:
            continue
    rows.sort(key=lambda r: r["fold_idx"])
    return rows


def _compute_row(c1_name: str, params_hash: str, param_desc: str, fold_rows: list[dict]) -> dict:
    """Build one leaderboard row; apply rejection rules."""
    if not fold_rows:
        return {
            "c1_name": c1_name,
            "params_hash": params_hash,
            "param_description": param_desc,
            "worst_fold_roi": 0.0,
            "worst_fold_dd": 0.0,
            "median_fold_roi": 0.0,
            "roi_std": 0.0,
            "trades_min": 0,
            "trades_max": 0,
            "scratch_rate_mean": 0.0,
            "scratch_rate_std": 0.0,
            "status": "REJECT",
            "reject_reason": "no_folds",
        }
    roi_list = [r["roi_pct"] for r in fold_rows]
    dd_list = [r["max_dd_pct"] for r in fold_rows]
    trades_list = [r["trades"] for r in fold_rows]
    scratch_rates = [r["scratch_rate"] for r in fold_rows]

    worst_fold_roi = float(min(roi_list))
    worst_fold_dd = float(min(dd_list))
    median_fold_roi = float(pd.Series(roi_list).median())
    roi_std = float(pd.Series(roi_list).std()) if len(roi_list) > 1 else 0.0
    trades_min = int(min(trades_list))
    trades_max = int(max(trades_list))
    scratch_rate_mean = float(pd.Series(scratch_rates).mean())
    scratch_rate_std = float(pd.Series(scratch_rates).std()) if len(scratch_rates) > 1 else 0.0

    status = "PASS"
    reject_reason = ""

    if any(t == 0 for t in trades_list):
        status = "REJECT"
        reject_reason = "zero_trades_in_fold"
    elif trades_max > TRADES_MAX_CEILING:
        status = "REJECT"
        reject_reason = "trades_max_above_ceiling"
    elif (median_fold_roi - worst_fold_roi) > ROI_COLLAPSE_THRESHOLD_PCT:
        status = "REJECT"
        reject_reason = "roi_collapse_vs_median"
    elif scratch_rate_mean > SCRATCH_RATE_MEAN_CEILING:
        status = "REJECT"
        reject_reason = "scratch_rate_above_ceiling"

    return {
        "c1_name": c1_name,
        "params_hash": params_hash,
        "param_description": param_desc,
        "worst_fold_roi": round(worst_fold_roi, 4),
        "worst_fold_dd": round(worst_fold_dd, 4),
        "median_fold_roi": round(median_fold_roi, 4),
        "roi_std": round(roi_std, 4),
        "trades_min": trades_min,
        "trades_max": trades_max,
        "scratch_rate_mean": round(scratch_rate_mean, 6),
        "scratch_rate_std": round(scratch_rate_std, 6),
        "status": status,
        "reject_reason": reject_reason,
    }


def build_leaderboard(phase5_root: Path) -> pd.DataFrame:
    """Discover all runs under phase5_root, aggregate per (c1_name, params_hash), apply rejection."""
    runs = _discover_runs(phase5_root)
    rows: list[dict] = []
    for c1_name, run_dir in runs:
        try:
            base_c1, params = _load_base_config_used(run_dir)
            params_hash = _params_fingerprint(base_c1, params)
            param_desc = _param_description(params)
            fold_rows = _fold_metrics_for_run(run_dir)
            row = _compute_row(c1_name, params_hash, param_desc, fold_rows)
            rows.append(row)
        except (FileNotFoundError, ValueError):
            continue
    if not rows:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["c1_name", "params_hash"], keep="first")
    df = df.sort_values(by=["c1_name", "params_hash"]).reset_index(drop=True)
    return df[LEADERBOARD_COLUMNS]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5 — C1 parameter WFO leaderboard (one row per C1-param combo)."
    )
    parser.add_argument(
        "--results-dir",
        default="results/phase5",
        help="Root directory (e.g. results/phase5).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <results-dir>/leaderboard_c1_params.csv).",
    )
    args = parser.parse_args(argv)
    root = Path(args.results_dir)
    out_path = Path(args.output) if args.output else root / "leaderboard_c1_params.csv"
    df = build_leaderboard(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

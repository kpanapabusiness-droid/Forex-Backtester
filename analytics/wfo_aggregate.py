from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def aggregate_wfo_run(run_dir: str | Path) -> None:
    """
    Aggregate a completed WFO v2 run directory into per‑fold CSV + robustness summary.

    Expects structure:
      <run_dir>/fold_XX/fold_dates.json
      <run_dir>/fold_XX/out_of_sample/summary.txt
      (optional) <run_dir>/fold_XX/params_hash.txt
    """
    from scripts.batch_sweeper import parse_summary_or_trades  # noqa: E402

    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        msg = f"WFO run directory not found: {run_dir}"
        if "<run_id>" in str(run_dir):
            msg += " (replace <run_id> with the actual run folder, e.g. results/wfo/20260130_204704)"
        else:
            parent = run_path.parent
            if parent.exists():
                subdirs = sorted([d.name for d in parent.iterdir() if d.is_dir()], reverse=True)
                if subdirs:
                    msg += f". Available under {parent}: {subdirs[:5]}"
        raise FileNotFoundError(msg)

    fold_dirs = sorted(
        [p for p in run_path.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: p.name,
    )
    if not fold_dirs:
        raise ValueError(f"No fold_* directories found under {run_dir}")

    rows: List[Dict[str, Any]] = []
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        try:
            fold_id = int(fold_name.split("_")[1])
        except Exception:
            continue

        dates_path = fold_dir / "fold_dates.json"
        if dates_path.exists():
            fold_dates = json.loads(dates_path.read_text(encoding="utf-8"))
        else:
            fold_dates = {}

        oos_dir = fold_dir / "out_of_sample"
        if not oos_dir.exists():
            continue

        metrics = parse_summary_or_trades(oos_dir)
        params_hash_path = fold_dir / "params_hash.txt"
        params_hash = (
            params_hash_path.read_text(encoding="utf-8").strip()
            if params_hash_path.exists()
            else None
        )

        row: Dict[str, Any] = {
            "fold_id": fold_id,
            "train_start": fold_dates.get("train_start"),
            "train_end": fold_dates.get("train_end"),
            "test_start": fold_dates.get("test_start"),
            "test_end": fold_dates.get("test_end"),
            "roi_pct": metrics.get("roi_pct", 0.0),
            "max_dd_pct": metrics.get("max_dd_pct", 0.0),
            "total_trades": metrics.get("total_trades", 0),
            "params_hash": params_hash,
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"No usable folds found under {run_dir}")

    df = pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)
    summary_csv = run_path / "wfo_summary.csv"
    df.to_csv(summary_csv, index=False)

    agg = _compute_robustness(df)
    agg_path = run_path / "wfo_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")


def _compute_robustness(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute worst fold, median metrics, aggregate max DD (min of fold DDs), consistency score."""
    df_num = df.copy()
    df_num["roi_pct"] = pd.to_numeric(df_num["roi_pct"], errors="coerce").fillna(0.0)
    df_num["max_dd_pct"] = pd.to_numeric(df_num["max_dd_pct"], errors="coerce").fillna(0.0)
    df_num["total_trades"] = pd.to_numeric(df_num["total_trades"], errors="coerce").fillna(0).astype(
        int
    )

    worst_idx = df_num["roi_pct"].idxmin()
    worst_row = df_num.loc[worst_idx]
    worst_case_fold = {
        "fold_id": int(worst_row["fold_id"]),
        "roi_pct": float(worst_row["roi_pct"]),
        "max_dd_pct": float(worst_row["max_dd_pct"]),
        "total_trades": int(worst_row["total_trades"]),
    }

    median_fold = {
        "roi_pct": float(df_num["roi_pct"].median()),
        "max_dd_pct": float(df_num["max_dd_pct"].median()),
        "total_trades": float(df_num["total_trades"].median()),
    }

    aggregate_max_dd_pct = float(df_num["max_dd_pct"].min()) if "max_dd_pct" in df_num else 0.0

    total_folds = int(len(df_num))
    positive = df_num["roi_pct"] > 0
    positive_count = int(positive.sum())
    consistency_score = {
        "total_folds": total_folds,
        "positive_roi_folds": positive_count,
        "positive_roi_fraction": float(positive_count / total_folds if total_folds else 0.0),
    }

    return {
        "worst_case_fold": worst_case_fold,
        "median_fold": median_fold,
        "aggregate_max_dd_pct": aggregate_max_dd_pct,
        "consistency_score": consistency_score,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate WFO v2 folds into summary CSV and robustness JSON."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed WFO v2 run directory (e.g., results/wfo/<run_id>).",
    )
    args = parser.parse_args()
    aggregate_wfo_run(args.run_dir)


if __name__ == "__main__":
    main()


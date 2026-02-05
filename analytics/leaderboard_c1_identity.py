from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class C1FoldStats:
    c1_name: str
    folds_df: pd.DataFrame


REQUIRED_FOLD_COLUMNS = {
    "fold_idx",
    "test_start",
    "test_end",
    "roi_pct",
    "max_dd_pct",
    "trades",
    "scratches",
}


def _find_wfo_folds(results_dir: Path) -> Dict[str, C1FoldStats]:
    """
    Discover per‑C1 WFO fold CSVs under results_dir.

    Contract (Phase 4 C1 identity):
      - results_dir contains one subdirectory per C1, e.g. results/phase4/wfo_c1_fisher_transform/
      - each C1 dir contains a wfo_folds.csv compatible with analytics.wfo.run_wfo output
        (at minimum the REQUIRED_FOLD_COLUMNS).
    """
    results_dir = results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"results-dir not found: {results_dir}")

    c1_to_folds: Dict[str, C1FoldStats] = {}

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        folds_path = child / "wfo_folds.csv"
        if not folds_path.exists():
            continue

        # Derive C1 name from directory name; allow prefixes like wfo_c1_<name>
        dir_name = child.name
        c1_name = dir_name
        if dir_name.lower().startswith("wfo_c1_"):
            c1_name = dir_name[len("wfo_c1_") :]

        df = pd.read_csv(folds_path)
        if df.empty:
            continue

        missing = REQUIRED_FOLD_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{folds_path} missing required columns: {sorted(missing)}")

        # Normalize types
        df = df.copy()
        for col in ["fold_idx", "trades", "scratches"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in ["roi_pct", "max_dd_pct"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

        c1_to_folds[c1_name] = C1FoldStats(c1_name=c1_name, folds_df=df)

    return c1_to_folds


def _compute_row(stats: C1FoldStats) -> Dict[str, object]:
    df = stats.folds_df
    c1_name = stats.c1_name

    # Worst fold = minimum ROI%
    worst_idx = df["roi_pct"].idxmin()
    worst_row = df.loc[worst_idx]

    folds = int(df["fold_idx"].nunique())
    total_trades = int(df["trades"].sum())
    min_fold_trades = int(df["trades"].min())
    max_fold_trades = int(df["trades"].max())

    # Scratch% of total for worst fold
    worst_trades = int(worst_row["trades"])
    worst_scratches = int(worst_row["scratches"])
    worst_scratch_pct = float((worst_scratches / worst_trades) * 100.0) if worst_trades > 0 else 0.0

    median_fold_roi_pct = float(df["roi_pct"].median())

    # Simple, explicit reject logic for Phase 4:
    # - Reject if too few folds or trades
    # - Reject if worst fold ROI% is negative
    status = "SURVIVOR"
    reject_reason = ""
    if folds < 2 or total_trades < 10:
        status = "REJECT"
        reject_reason = "insufficient_data"
    elif worst_row["roi_pct"] < 0:
        status = "REJECT"
        reject_reason = "worst_fold_negative_roi"

    return {
        "c1_name": c1_name,
        "folds": folds,
        "worst_fold_id": int(worst_row["fold_idx"]),
        "worst_fold_start": str(worst_row["test_start"]),
        "worst_fold_end": str(worst_row["test_end"]),
        "worst_fold_roi_pct": float(worst_row["roi_pct"]),
        "worst_fold_max_dd_pct": float(worst_row["max_dd_pct"]),
        "worst_fold_scratch_pct": worst_scratch_pct,
        "worst_fold_trades": worst_trades,
        "median_fold_roi_pct": median_fold_roi_pct,
        "total_trades": total_trades,
        "min_fold_trades": min_fold_trades,
        "max_fold_trades": max_fold_trades,
        "status": status,
        "reject_reason": reject_reason,
    }


def build_leaderboard(results_dir: Path) -> pd.DataFrame:
    c1_to_folds = _find_wfo_folds(results_dir)
    if not c1_to_folds:
        return pd.DataFrame(
            columns=[
                "c1_name",
                "folds",
                "worst_fold_id",
                "worst_fold_start",
                "worst_fold_end",
                "worst_fold_roi_pct",
                "worst_fold_max_dd_pct",
                "worst_fold_scratch_pct",
                "worst_fold_trades",
                "median_fold_roi_pct",
                "total_trades",
                "min_fold_trades",
                "max_fold_trades",
                "status",
                "reject_reason",
                "rank",
            ]
        )

    rows: List[Dict[str, object]] = []
    for stats in c1_to_folds.values():
        rows.append(_compute_row(stats))

    df = pd.DataFrame(rows)

    # Deterministic ordering:
    #   1) Survivors first, then rejects
    #   2) Within each group, sort by worst_fold_roi_pct desc, then c1_name asc
    df["status_rank"] = df["status"].map({"SURVIVOR": 0, "REJECT": 1}).fillna(2).astype(int)
    df = df.sort_values(
        by=["status_rank", "worst_fold_roi_pct", "c1_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    df["rank"] = df.index + 1
    df = df.drop(columns=["status_rank"])
    return df


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 — C1 identity WFO leaderboard (one row per C1)."
    )
    parser.add_argument(
        "--results-dir",
        default="results/phase4",
        help="Root directory containing per‑C1 WFO runs (default: results/phase4).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.results_dir)
    leaderboard_path = root / "leaderboard_c1_identity.csv"

    df = build_leaderboard(root)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(leaderboard_path, index=False)


if __name__ == "__main__":
    main()


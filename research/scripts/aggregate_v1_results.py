#!/usr/bin/env python3
"""Aggregate WFO V1 (C1 + Exit + Volume) fold summaries into combined CSVs and leaderboard."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

DEFAULT_ROOT = Path("results")
DEFAULT_OUT_DIR = Path("results") / "c1_with_exit_plus_vol_results"
DEFAULT_SUMMARY_PATH = DEFAULT_OUT_DIR / "v1_folds_summary.csv"
DEFAULT_LEADERBOARD_PATH = DEFAULT_OUT_DIR / "v1_leaderboard.csv"

PREFIX = "wfo_v1_"
# Pattern: wfo_v1_<c1_name>__<volume_name>
NAME_PATTERN = re.compile(r"^wfo_v1_(.+?)__(.+?)$", re.IGNORECASE)

REQUIRED_COLUMNS = [
    "fold_idx",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "trades",
    "wins",
    "losses",
    "scratches",
    "win_pct_ns",
    "loss_pct_ns",
    "expectancy",
    "roi_pct",
    "max_dd_pct",
]


def parse_run_name(run_name: str) -> Optional[Tuple[str, str]]:
    """Extract (c1_name, volume_name) from run name like 'wfo_v1_kalman_filter__adx'."""
    match = NAME_PATTERN.match(run_name)
    if match:
        return (match.group(1), match.group(2))
    return None


def _ensure_columns(df: pd.DataFrame, c1_name: str, volume_name: str) -> pd.DataFrame:
    """Add missing columns with defaults so downstream logic is robust."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col == "fold_idx":
                df[col] = range(1, len(df) + 1)
            elif col in {"trades", "wins", "losses", "scratches"}:
                df[col] = 0
            else:
                df[col] = 0.0

    # Ensure c1_name and volume_name columns
    if "c1_name" in df.columns:
        df = df.drop(columns=["c1_name"])
    if "volume_name" in df.columns:
        df = df.drop(columns=["volume_name"])

    df.insert(0, "volume_name", volume_name)
    df.insert(0, "c1_name", c1_name)

    # Compute scratch_pct if not present
    if "scratch_pct" not in df.columns:
        if "scratch_pct_ns" in df.columns:
            df["scratch_pct"] = df["scratch_pct_ns"]
        else:
            with pd.option_context("mode.chained_assignment", None):
                trades_float = pd.to_numeric(df["trades"], errors="coerce")
                scratches_float = pd.to_numeric(df["scratches"], errors="coerce").clip(lower=0.0)
                df["scratch_pct"] = 0.0
                mask = (trades_float > 0) & trades_float.notna()
                df.loc[mask, "scratch_pct"] = (scratches_float[mask] / trades_float[mask]) * 100.0

    return df[
        [
            "c1_name",
            "volume_name",
            "fold_idx",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "trades",
            "wins",
            "losses",
            "scratches",
            "win_pct_ns",
            "loss_pct_ns",
            "expectancy",
            "roi_pct",
            "max_dd_pct",
            "scratch_pct",
        ]
    ]


def load_fold_summaries(root: Path) -> pd.DataFrame:
    """Load every wfo_folds.csv from wfo_v1_*__* directories."""
    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")

    fold_frames: List[pd.DataFrame] = []

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        name = child.name
        if not name.lower().startswith(PREFIX):
            continue

        # Parse c1_name and volume_name from directory name
        parsed = parse_run_name(name)
        if parsed is None:
            print(f"⚠️  Skipping {child}: name doesn't match pattern wfo_v1_<c1>__<volume>")
            continue

        c1_name, volume_name = parsed

        folds_path = child / "wfo_folds.csv"
        if not folds_path.exists():
            print(f"⚠️  Skipping {child}: missing wfo_folds.csv")
            continue

        try:
            df = pd.read_csv(folds_path)
            if df.empty:
                print(f"⚠️  Skipping {child}: wfo_folds.csv is empty")
                continue

            normalized = _ensure_columns(df.copy(), c1_name, volume_name)
            fold_frames.append(normalized)
        except Exception as e:
            print(f"⚠️  Error loading {folds_path}: {e}")
            continue

    if not fold_frames:
        return pd.DataFrame(columns=["c1_name", "volume_name"])

    return pd.concat(fold_frames, ignore_index=True)


def build_leaderboard(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Produce leaderboard metrics grouped by (c1_name, volume_name)."""
    if summary_df.empty:
        return summary_df

    grouped = summary_df.groupby(["c1_name", "volume_name"], dropna=False)
    leaderboard = grouped.agg(
        folds=("fold_idx", "nunique"),
        median_roi_pct=("roi_pct", "median"),
        worst_fold_roi_pct=("roi_pct", "min"),
        median_max_dd_pct=("max_dd_pct", "median"),
        worst_fold_max_dd_pct=("max_dd_pct", "max"),
        median_expectancy=("expectancy", "median"),
        mean_expectancy=("expectancy", "mean"),
        total_trades=("trades", "sum"),
        median_trades=("trades", "median"),
        mean_win_pct=("win_pct_ns", "mean"),
        median_win_pct=("win_pct_ns", "median"),
        mean_scratch_pct=("scratch_pct", "mean"),
        median_scratch_pct=("scratch_pct", "median"),
    ).reset_index()

    # Sort by median ROI descending, then median max DD ascending (lower is better)
    leaderboard = leaderboard.sort_values(
        ["median_roi_pct", "median_max_dd_pct"], ascending=[False, True]
    ).reset_index(drop=True)

    return leaderboard


def print_compact_leaderboard(leaderboard_df: pd.DataFrame) -> None:
    """Print a compact leaderboard to stdout."""
    if leaderboard_df.empty:
        print("(No results to display)")
        return

    # Select key columns for display
    display_cols = [
        "c1_name",
        "volume_name",
        "median_roi_pct",
        "worst_fold_roi_pct",
        "median_max_dd_pct",
        "median_expectancy",
        "total_trades",
        "median_win_pct",
    ]

    available_cols = [col for col in display_cols if col in leaderboard_df.columns]
    display_df = leaderboard_df[available_cols].copy()

    # Format numeric columns for readability
    for col in display_df.select_dtypes(include=["float64"]).columns:
        if "pct" in col or "roi" in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        elif "expectancy" in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        elif "trades" in col:
            display_df[col] = display_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "0")

    print("\n" + "=" * 100)
    print("V1 Leaderboard (C1 + Exit + Volume)")
    print("=" * 100)
    print(display_df.to_string(index=False))
    print("=" * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate WFO V1 (C1 + Exit + Volume) fold summaries into CSVs and leaderboard."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"WFO results root directory (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for aggregated results (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        summary_df = load_fold_summaries(args.root)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    if summary_df.empty:
        print(
            f"⚠️  No wfo_v1_*__* runs with wfo_folds.csv found under {args.root}. Nothing to aggregate."
        )
        sys.exit(0)

    # Ensure output directory exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Write summary CSV
    summary_path = args.out_dir / "v1_folds_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Wrote folds summary: {summary_path} ({len(summary_df)} rows)")

    # Build and write leaderboard
    leaderboard_df = build_leaderboard(summary_df)
    leaderboard_path = args.out_dir / "v1_leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)
    print(f"✅ Wrote leaderboard: {leaderboard_path} ({len(leaderboard_df)} combinations)")

    # Print compact leaderboard
    print_compact_leaderboard(leaderboard_df)

    print(f"\n✅ Aggregated {len(summary_df)} fold rows across {len(leaderboard_df)} (C1, Volume) combinations.")


if __name__ == "__main__":
    main()


















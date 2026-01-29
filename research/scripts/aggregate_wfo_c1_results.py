#!/usr/bin/env python3
"""Aggregate WFO C1 fold summaries into combined CSVs and leaderboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_ROOT = Path("results")
DEFAULT_SUMMARY_PATH = Path("results") / "wfo_c1_summary.csv"
DEFAULT_LEADERBOARD_PATH = Path("results") / "wfo_c1_leaderboard.csv"


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


ALLOWED_EXACT_RUNS = {"wfo_default", "wfo_tier1_c1"}
PREFIX = "wfo_c1_"


def _ensure_columns(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """Add missing columns with defaults so downstream logic is robust."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col == "fold_idx":
                df[col] = range(len(df))
            elif col in {"trades", "wins", "losses", "scratches"}:
                df[col] = 0
            else:
                df[col] = 0.0
    if "indicator" in df.columns:
        df = df.drop(columns=["indicator"])
    df.insert(0, "indicator", indicator)
    if "scratch_pct_ns" in df.columns:
        df["scratch_pct"] = df["scratch_pct_ns"]
    else:
        with pd.option_context("mode.chained_assignment", None):
            trades_float = pd.to_numeric(df["trades"], errors="coerce")
            scratches_float = pd.to_numeric(df["scratches"], errors="coerce").clip(lower=0.0)
            # Avoid division by zero: set scratch_pct to 0.0 when trades is 0 or NaN
            df["scratch_pct"] = 0.0
            mask = (trades_float > 0) & trades_float.notna()
            df.loc[mask, "scratch_pct"] = (scratches_float[mask] / trades_float[mask]) * 100.0
    return df[
        [
            "indicator",
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
    """Load every wfo_folds.csv under the root directory."""
    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")

    indicator_dirs: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        lower = name.lower()
        if lower.startswith(PREFIX):
            indicator_dirs.append(child)
        elif lower in ALLOWED_EXACT_RUNS:
            indicator_dirs.append(child)

    fold_frames: List[pd.DataFrame] = []

    for indicator_dir in indicator_dirs:
        folds_path = indicator_dir / "wfo_folds.csv"
        if not folds_path.exists():
            print(f"⚠️  Skipping {indicator_dir}: missing wfo_folds.csv")
            continue

        indicator_name = (
            indicator_dir.name[len(PREFIX) :]
            if indicator_dir.name.lower().startswith(PREFIX)
            else indicator_dir.name.split("wfo_", 1)[-1]
        )

        df = pd.read_csv(folds_path)
        if df.empty:
            continue
        normalized = _ensure_columns(df.copy(), indicator_name)
        fold_frames.append(normalized)

    if not fold_frames:
        return pd.DataFrame(columns=["indicator"])

    return pd.concat(fold_frames, ignore_index=True)


def build_leaderboard(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Produce leaderboard metrics grouped by indicator."""
    if summary_df.empty:
        return summary_df
    grouped = summary_df.groupby("indicator", dropna=False)
    leaderboard = grouped.agg(
        folds=("fold_idx", "nunique"),
        median_roi_pct=("roi_pct", "median"),
        worst_fold_roi_pct=("roi_pct", "min"),
        median_expectancy=("expectancy", "median"),
        median_max_dd_pct=("max_dd_pct", "median"),
        median_trades=("trades", "median"),
        median_win_pct_ns=("win_pct_ns", "median"),
        median_scratch_pct=("scratch_pct", "median"),
    ).reset_index()
    leaderboard = leaderboard.sort_values("median_roi_pct", ascending=False).reset_index(drop=True)
    return leaderboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate WFO C1 fold summaries into CSVs and leaderboard."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="WFO results root.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output path for per-fold summary CSV.",
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=DEFAULT_LEADERBOARD_PATH,
        help="Output path for per-indicator leaderboard CSV.",
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
            f"⚠️  No wfo_c1_* runs with wfo_folds.csv found under {args.root}. Nothing to aggregate."
        )
        sys.exit(0)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary, index=False)

    leaderboard_df = build_leaderboard(summary_df)
    args.leaderboard.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_df.to_csv(args.leaderboard, index=False)

    print(f"✅ Aggregated {len(summary_df)} fold rows across {leaderboard_df.shape[0]} indicators.")
    print("📊 Leaderboard:")
    print(leaderboard_df.to_string(index=False))


if __name__ == "__main__":
    main()



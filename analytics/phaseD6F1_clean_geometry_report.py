"""
Phase D-6F.1: CLI for opportunity geometry report from clean labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from analytics.phaseD6F1_clean_geometry import (
    PHASE_D_DISCOVERY_END,
    run_geometry_analysis,
)


def _load_clean(path: Path) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Clean labels not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-6F.1 — Opportunity geometry report from clean labels.",
    )
    parser.add_argument("--clean", required=True, help="Path to opportunity_labels_clean.csv")
    parser.add_argument("--outdir", required=True, help="Output directory (e.g. results/phaseD/labels/clean_geometry)")
    parser.add_argument(
        "--split-mode",
        choices=("auto", "explicit"),
        default="auto",
        help="auto: use Phase D convention (2022-12-31); explicit: use --split-date",
    )
    parser.add_argument(
        "--split-date",
        default=None,
        help="Discovery end date when --split-mode explicit (YYYY-MM-DD)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    df = _load_clean(Path(args.clean))

    if args.split_mode == "explicit" and args.split_date:
        discovery_end = args.split_date
    else:
        discovery_end = PHASE_D_DISCOVERY_END

    paths = run_geometry_analysis(
        df,
        out_dir=Path(args.outdir),
        discovery_end=discovery_end,
    )
    print("Phase D-6F.1 geometry report written to:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()

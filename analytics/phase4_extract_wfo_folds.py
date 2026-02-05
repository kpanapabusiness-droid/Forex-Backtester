# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


REQUIRED_METRICS = ("roi_pct", "max_dd_pct", "total_trades", "scratches")


@dataclass(frozen=True)
class FoldRecord:
    fold_idx: int
    test_start: str
    test_end: str
    roi_pct: float
    max_dd_pct: float
    trades: int
    scratches: int


def _parse_fold_dir(fold_dir: Path) -> FoldRecord:
    """Parse a single fold_* directory into a FoldRecord."""
    from scripts.batch_sweeper import parse_summary_or_trades  # noqa: E402

    name = fold_dir.name
    if not name.startswith("fold_"):
        raise ValueError(f"Unexpected fold directory name (expected 'fold_<idx>'): {name}")

    try:
        idx_str = name.split("_", 1)[1]
        fold_idx = int(idx_str)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unable to parse fold index from {name}") from exc

    dates_path = fold_dir / "fold_dates.json"
    if not dates_path.exists():
        raise FileNotFoundError(f"Missing fold_dates.json for {name} under {fold_dir.parent}")

    dates = json.loads(dates_path.read_text(encoding="utf-8"))
    try:
        test_start = str(dates["test_start"])
        test_end = str(dates["test_end"])
    except KeyError as exc:
        raise KeyError(f"fold_dates.json for {name} missing key: {exc}") from exc

    oos_dir = fold_dir / "out_of_sample"
    if not oos_dir.exists():
        raise FileNotFoundError(f"Missing out_of_sample directory for {name} under {fold_dir}")

    # Read metrics from summary.txt or trades.csv (fallback) using shared helper.
    metrics = parse_summary_or_trades(oos_dir)

    missing = [key for key in REQUIRED_METRICS if metrics.get(key) is None]
    if missing:
        raise ValueError(
            f"summary.txt under {oos_dir} missing required metrics: {', '.join(missing)}"
        )

    roi_pct = float(metrics["roi_pct"])
    max_dd_pct = float(metrics["max_dd_pct"])
    trades = int(metrics["total_trades"])
    scratches = int(metrics["scratches"])

    return FoldRecord(
        fold_idx=fold_idx,
        test_start=test_start,
        test_end=test_end,
        roi_pct=roi_pct,
        max_dd_pct=max_dd_pct,
        trades=trades,
        scratches=scratches,
    )


def extract_wfo_v2_folds(wfo_root: Path) -> None:
    """
    Convert WFO v2 run outputs under wfo_root into one wfo_folds.csv per C1.

    Expected layout:
      wfo_root/
        wfo_c1_<c1_name>/
          <run_id>/
            fold_XX/
              fold_dates.json
              out_of_sample/summary.txt

    Output:
      wfo_root.parent/
        wfo_c1_<c1_name>/wfo_folds.csv
    """
    wfo_root = wfo_root.resolve()
    if not wfo_root.exists():
        raise FileNotFoundError(f"WFO root not found: {wfo_root}")

    phase4_root = wfo_root.parent

    processed_c1 = 0
    wrote_csv = 0

    # Discover per-C1 directories under wfo_root.
    c1_dirs = sorted(
        p for p in wfo_root.iterdir() if p.is_dir() and p.name.startswith("wfo_c1_")
    )
    for c1_dir in c1_dirs:
        processed_c1 += 1

        # Choose the most recent run directory (lexicographically max).
        run_dirs = [d for d in c1_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            raise ValueError(f"No run directories found under {c1_dir}")
        latest_run = max(run_dirs, key=lambda p: p.name)

        fold_dirs = sorted(
            p for p in latest_run.iterdir() if p.is_dir() and p.name.startswith("fold_")
        )
        if not fold_dirs:
            raise ValueError(f"No fold_* directories found under {latest_run}")

        fold_records: List[FoldRecord] = [_parse_fold_dir(fold_dir) for fold_dir in fold_dirs]
        if not fold_records:
            raise ValueError(f"No usable fold_* directories under {latest_run}")

        fold_records.sort(key=lambda r: r.fold_idx)

        rows = [
            {
                "fold_idx": rec.fold_idx,
                "test_start": rec.test_start,
                "test_end": rec.test_end,
                "roi_pct": rec.roi_pct,
                "max_dd_pct": rec.max_dd_pct,
                "trades": rec.trades,
                "scratches": rec.scratches,
            }
            for rec in fold_records
        ]

        df = pd.DataFrame(
            rows,
            columns=[
                "fold_idx",
                "test_start",
                "test_end",
                "roi_pct",
                "max_dd_pct",
                "trades",
                "scratches",
            ],
        )

        out_dir = phase4_root / c1_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "wfo_folds.csv"
        df.to_csv(out_path, index=False)

        wrote_csv += 1

    if processed_c1 == 0 or wrote_csv == 0:
        raise ValueError(f"No C1 directories with folds found under {wfo_root}")

    print(f"processed_c1={processed_c1} wrote_csv={wrote_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 — Extract WFO v2 raw outputs into per‑C1 wfo_folds.csv files."
    )
    parser.add_argument(
        "--wfo-root",
        required=True,
        help="Path to Phase 4 WFO output root (e.g., results/phase4/wfo).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_wfo_v2_folds(Path(args.wfo_root))


if __name__ == "__main__":
    main()


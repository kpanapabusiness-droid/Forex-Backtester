from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Phase D1 labels parquet not found at {path}")
    return pd.read_parquet(path)


def _summarize_by_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-pair summary with basic counts and rates (no ROI, no indicators)."""
    g = df.groupby("pair", dropna=False)

    def _sum_bool(series: pd.Series) -> int:
        return int(pd.to_numeric(series, errors="coerce").fillna(0).astype(bool).sum())

    out = g.agg(
        rows=("pair", "size"),
        discovery_rows=("dataset_split", lambda s: int((s == "discovery").sum())),
        validation_rows=("dataset_split", lambda s: int((s == "validation").sum())),
        zone_a_count=("zone_a_1r_10", _sum_bool),
        zone_b_count=("zone_b_3r_20", _sum_bool),
        zone_c_count=("zone_c_6r_40", _sum_bool),
        mfe_10_r_mean=("mfe_10_r", "mean"),
        mfe_20_r_mean=("mfe_20_r", "mean"),
        mfe_40_r_mean=("mfe_40_r", "mean"),
    ).reset_index()

    # Rates per direction-date cell
    for col in ("zone_a_count", "zone_b_count", "zone_c_count"):
        rate_col = col.replace("_count", "_rate")
        out[rate_col] = out[col] / out["rows"].replace(0, np.nan)

    return out


def _summarize_global(df: pd.DataFrame) -> dict:
    """Lightweight global summary for sanity checks (no strategy metrics)."""
    total_rows = int(len(df))
    total_pairs = int(df["pair"].nunique())

    a = int(pd.to_numeric(df["zone_a_1r_10"], errors="coerce").fillna(0).astype(bool).sum())
    b = int(pd.to_numeric(df["zone_b_3r_20"], errors="coerce").fillna(0).astype(bool).sum())
    c = int(pd.to_numeric(df["zone_c_6r_40"], errors="coerce").fillna(0).astype(bool).sum())

    summary = {
        "total_rows": total_rows,
        "total_pairs": total_pairs,
        "zone_a_1r_10_count": a,
        "zone_b_3r_20_count": b,
        "zone_c_6r_40_count": c,
        "zone_a_1r_10_rate": a / total_rows if total_rows else 0.0,
        "zone_b_3r_20_rate": b / total_rows if total_rows else 0.0,
        "zone_c_6r_40_rate": c / total_rows if total_rows else 0.0,
        # Sanity check flags
        "rarity_ordering_ok": (a >= b >= c),
    }

    return summary


def run_phaseD1_label_summaries(
    labels_path: Path | None = None,
    out_dir: Path | None = None,
) -> Tuple[Path, Path]:
    """
    Entry point for Phase D1 label summaries.

    Reads opportunity_labels.parquet and writes:
      - summary_by_pair.csv
      - summary_global.json
    under results/phaseD/labels/ by default.
    """
    if labels_path is None:
        labels_path = ROOT / "results" / "phaseD" / "labels" / "opportunity_labels.parquet"
    if out_dir is None:
        out_dir = labels_path.parent

    df = _load_labels(labels_path)
    by_pair = _summarize_by_pair(df)
    global_summary = _summarize_global(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_by_pair_path = out_dir / "summary_by_pair.csv"
    summary_global_path = out_dir / "summary_global.json"

    by_pair.to_csv(summary_by_pair_path, index=False, float_format="%.8f")
    summary_global_path.write_text(json.dumps(global_summary, indent=2, sort_keys=True), encoding="utf-8")

    return summary_by_pair_path, summary_global_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-1 — Build lightweight summaries from opportunity labels.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to opportunity_labels.parquet (default: results/phaseD/labels/opportunity_labels.parquet).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for summaries (default: same dir as labels parquet).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    labels_path = Path(args.labels) if args.labels is not None else None
    out_dir = Path(args.outdir) if args.outdir is not None else None
    s_csv, s_json = run_phaseD1_label_summaries(labels_path=labels_path, out_dir=out_dir)
    print(f"Phase D1 summaries written to:\n  {s_csv}\n  {s_json}")


if __name__ == "__main__":
    main()


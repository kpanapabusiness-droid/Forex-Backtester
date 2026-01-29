#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase1_volume_referee.py — Phase 1 Volume Study Validator & Comparison Report
-----------------------------------------------------------------------------

Validates that trades_with_volume <= trades_without_volume and generates
comparison reports with deltas per volume indicator.

Usage:
    python scripts/phase1_volume_referee.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive)."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None


def find_join_keys(baseline_df: pd.DataFrame, volume_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Auto-detect join keys from common ID columns. Returns list of (baseline_col, volume_col) tuples."""
    baseline_cols_lower = {c.lower(): c for c in baseline_df.columns}
    volume_cols_lower = {c.lower(): c for c in volume_df.columns}

    # Explicit check: if both have 'pair', 'c1', 'exit', use them directly
    required_keys = ["pair", "c1", "exit"]
    all_present = all(
        key.lower() in baseline_cols_lower and key.lower() in volume_cols_lower
        for key in required_keys
    )

    if all_present:
        # Force use of ['pair', 'c1', 'exit'] when all three are present
        join_keys = [
            (baseline_cols_lower[key.lower()], volume_cols_lower[key.lower()])
            for key in required_keys
        ]
        return join_keys

    # Fallback: heuristic detection
    candidate_keys = [
        "pair",
        "symbol",
        "tf",
        "timeframe",
        "c1_name",
        "c1",
        "exit_name",
        "exit",
        "config_id",
        "run_id",
    ]

    join_keys: List[Tuple[str, str]] = []
    for key in candidate_keys:
        key_lower = key.lower()
        if key_lower in baseline_cols_lower and key_lower in volume_cols_lower:
            baseline_col = baseline_cols_lower[key_lower]
            volume_col = volume_cols_lower[key_lower]
            # Check that both columns have non-null overlap
            baseline_vals = set(baseline_df[baseline_col].dropna().astype(str))
            volume_vals = set(volume_df[volume_col].dropna().astype(str))
            if baseline_vals & volume_vals:  # intersection
                join_keys.append((baseline_col, volume_col))

    if not join_keys:
        raise ValueError(
            f"No suitable join keys found. Baseline columns: {list(baseline_df.columns)}, "
            f"Volume columns: {list(volume_df.columns)}"
        )

    # Require at least 3 join keys (pair + c1 + exit minimum)
    if len(join_keys) < 3:
        found_keys = [k[0] for k in join_keys]
        raise ValueError(
            f"Insufficient join keys found: {found_keys}. "
            f"Require at least 3 keys including pair, c1, and exit. "
            f"Baseline columns: {list(baseline_df.columns)}, "
            f"Volume columns: {list(volume_df.columns)}. "
            f"CSVs are not comparable without proper identity columns."
        )

    # Verify required keys are present
    found_key_names = [k[0].lower() for k in join_keys]
    required = ["pair", "c1", "exit"]
    missing = [r for r in required if r not in found_key_names and f"{r}_name" not in found_key_names]
    if missing:
        raise ValueError(
            f"Missing required join keys: {missing}. "
            f"Found keys: {[k[0] for k in join_keys]}. "
            f"Must include pair, c1 (or c1_name), and exit (or exit_name)."
        )

    return join_keys


def find_csv_files(results_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find baseline (no-volume) and volume CSV files."""
    baseline_path: Optional[Path] = None
    volume_path: Optional[Path] = None

    # Prefer specific known files
    volume_candidates = [
        results_dir / "c1_volume_aggregated.csv",
        results_dir / "results" / "c1_volume_aggregated.csv",
    ]

    baseline_candidates = [
        results_dir / "c1 results" / "c1_comparison_2023_2025.csv",
        results_dir / "c1_comparison_2023_2025.csv",
    ]

    # Also search for aggregated CSVs
    for pattern in ["**/c1_aggregated*.csv", "**/c1_comparison*.csv"]:
        for path in results_dir.glob(pattern):
            if baseline_path is None:
                baseline_path = path
                break

    for pattern in ["**/c1_volume*.csv", "**/*volume*.csv"]:
        for path in results_dir.glob(pattern):
            if "volume" in path.name.lower() and volume_path is None:
                volume_path = path
                break

    # Check explicit candidates
    for candidate in volume_candidates:
        if candidate.exists():
            volume_path = candidate
            break

    for candidate in baseline_candidates:
        if candidate.exists():
            baseline_path = candidate
            break

    return baseline_path, volume_path


def validate_invariant(
    joined_df: pd.DataFrame,
    trades_col_baseline: str,
    trades_col_volume: str,
    max_violations: int = 20,
) -> None:
    """Enforce trades_with_volume <= trades_without_volume."""
    violations = joined_df[
        joined_df[trades_col_volume] > joined_df[trades_col_baseline]
    ].copy()

    if len(violations) > 0:
        # Build error message with key fields
        key_cols = [c for c in joined_df.columns if c not in [trades_col_baseline, trades_col_volume]]
        display_cols = key_cols[:5] + [trades_col_baseline, trades_col_volume]  # Show first 5 key cols

        msg_parts = [
            f"❌ INVARIANT VIOLATION: Found {len(violations)} rows where trades_with_volume > trades_without_volume",
            "",
            "First violations:",
        ]

        for idx, row in violations.head(max_violations).iterrows():
            row_str = " | ".join([f"{col}={row[col]}" for col in display_cols if col in row])
            msg_parts.append(f"  {row_str}")

        if len(violations) > max_violations:
            msg_parts.append(f"  ... and {len(violations) - max_violations} more")

        raise SystemExit("\n".join(msg_parts))


def compute_deltas(
    joined_df: pd.DataFrame,
    baseline_cols: Dict[str, str],
    volume_cols: Dict[str, str],
    volume_name_col: Optional[str],
) -> pd.DataFrame:
    """Compute deltas per volume indicator."""
    delta_rows: List[Dict[str, Any]] = []

    # Group by volume indicator if present
    if volume_name_col and volume_name_col in joined_df.columns:
        groups = joined_df.groupby(volume_name_col, dropna=False)
    else:
        groups = [(None, joined_df)]

    for vol_name, group_df in groups:
        row: Dict[str, Any] = {}

        if vol_name is not None:
            row["volume_indicator"] = vol_name

        # Compute deltas for each metric
        for metric, (base_col, vol_col) in {
            "trades": (baseline_cols["trades"], volume_cols["trades"]),
            "scratches": (baseline_cols.get("scratches"), volume_cols.get("scratches")),
            "roi": (baseline_cols.get("roi"), volume_cols.get("roi")),
            "mar": (baseline_cols.get("mar"), volume_cols.get("mar")),
            "expectancy": (baseline_cols.get("expectancy"), volume_cols.get("expectancy")),
        }.items():
            if base_col and vol_col and base_col in group_df.columns and vol_col in group_df.columns:
                base_vals = pd.to_numeric(group_df[base_col], errors="coerce").fillna(0)
                vol_vals = pd.to_numeric(group_df[vol_col], errors="coerce").fillna(0)
                delta = vol_vals - base_vals
                row[f"delta_{metric}"] = float(delta.mean()) if len(delta) > 0 else 0.0
                row[f"baseline_{metric}_mean"] = float(base_vals.mean()) if len(base_vals) > 0 else 0.0
                row[f"volume_{metric}_mean"] = float(vol_vals.mean()) if len(vol_vals) > 0 else 0.0

        # Count matched rows
        row["matched_rows"] = len(group_df)

        delta_rows.append(row)

    return pd.DataFrame(delta_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 Volume Study Validator & Comparison Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root directory to search for result CSVs",
    )
    parser.add_argument(
        "--baseline-csv",
        type=str,
        default=None,
        help="Explicit path to baseline (no-volume) CSV",
    )
    parser.add_argument(
        "--volume-csv",
        type=str,
        default=None,
        help="Explicit path to volume CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase1_volume",
        help="Output directory for comparison reports",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Find CSV files
    if args.baseline_csv:
        baseline_path = Path(args.baseline_csv)
    else:
        baseline_path, _ = find_csv_files(results_dir)
        if baseline_path is None:
            # Try to find any aggregated CSV as baseline
            for pattern in ["**/*aggregated*.csv", "**/*comparison*.csv"]:
                candidates = list(results_dir.glob(pattern))
                if candidates:
                    baseline_path = candidates[0]
                    break

    if args.volume_csv:
        volume_path = Path(args.volume_csv)
    else:
        _, volume_path = find_csv_files(results_dir)

    if baseline_path is None or not baseline_path.exists():
        raise SystemExit(f"❌ Baseline CSV not found. Searched in {results_dir}")

    if volume_path is None or not volume_path.exists():
        raise SystemExit(f"❌ Volume CSV not found. Searched in {results_dir}")

    print(f"📊 Baseline CSV: {baseline_path}")
    print(f"📊 Volume CSV: {volume_path}")

    # Load DataFrames
    baseline_df = pd.read_csv(baseline_path)
    volume_df = pd.read_csv(volume_path)

    print(f"   Baseline rows: {len(baseline_df)}, columns: {list(baseline_df.columns)}")
    print(f"   Volume rows: {len(volume_df)}, columns: {list(volume_df.columns)}")

    # Auto-detect columns
    trades_col_baseline = find_column(baseline_df, ["trades", "total_trades", "n_trades", "trade_count"])
    trades_col_volume = find_column(volume_df, ["trades", "total_trades", "n_trades", "trade_count"])

    if not trades_col_baseline:
        raise SystemExit(f"❌ Could not find trades column in baseline CSV. Available: {list(baseline_df.columns)}")
    if not trades_col_volume:
        raise SystemExit(f"❌ Could not find trades column in volume CSV. Available: {list(volume_df.columns)}")

    baseline_cols = {
        "trades": trades_col_baseline,
        "scratches": find_column(baseline_df, ["scratches", "n_scratches", "scratch_count"]),
        "roi": find_column(baseline_df, ["roi", "ROI", "roi_pct", "return", "net_roi"]),
        "mar": find_column(baseline_df, ["mar", "MAR"]),
        "expectancy": find_column(baseline_df, ["expectancy", "exp"]),
    }

    volume_cols = {
        "trades": trades_col_volume,
        "scratches": find_column(volume_df, ["scratches", "n_scratches", "scratch_count"]),
        "roi": find_column(volume_df, ["roi", "ROI", "roi_pct", "return", "net_roi"]),
        "mar": find_column(volume_df, ["mar", "MAR"]),
        "expectancy": find_column(volume_df, ["expectancy", "exp"]),
    }

    volume_name_col = find_column(volume_df, ["volume_name", "volume_indicator", "volume"])

    print(f"   Using baseline trades column: {trades_col_baseline}")
    print(f"   Using volume trades column: {trades_col_volume}")

    # Find join keys
    try:
        join_key_pairs = find_join_keys(baseline_df, volume_df)
        baseline_join_cols = [k[0] for k in join_key_pairs]
        volume_join_cols = [k[1] for k in join_key_pairs]
        print(f"   Using join keys: {baseline_join_cols}")
    except ValueError as e:
        raise SystemExit(f"❌ {e}")

    # Perform join
    joined_df = baseline_df.merge(
        volume_df,
        left_on=baseline_join_cols,
        right_on=volume_join_cols,
        how="inner",
        suffixes=("_baseline", "_volume"),
    )

    if len(joined_df) == 0:
        raise SystemExit(
            f"❌ No matching rows after join on {baseline_join_cols}. "
            f"Baseline has {len(baseline_df)} rows, volume has {len(volume_df)} rows."
        )

    # Reject suspicious join explosions
    max_input_size = max(len(baseline_df), len(volume_df))
    if len(joined_df) > 2 * max_input_size:
        raise SystemExit(
            f"❌ Join explosion detected: {len(joined_df)} joined rows from "
            f"baseline={len(baseline_df)}, volume={len(volume_df)}. "
            f"Joined rows exceed 2x max input size ({2 * max_input_size}). "
            f"This suggests duplicate keys or incorrect join logic."
        )

    print(f"✅ Joined {len(joined_df)} matching rows")

    # Determine final column names after merge
    final_trades_baseline = trades_col_baseline if trades_col_baseline in joined_df.columns else f"{trades_col_baseline}_baseline"
    final_trades_volume = trades_col_volume if trades_col_volume in joined_df.columns else f"{trades_col_volume}_volume"

    # Validate invariant
    print("🔍 Validating invariant: trades_with_volume <= trades_without_volume")
    validate_invariant(joined_df, final_trades_baseline, final_trades_volume)
    print("✅ Invariant check passed")

    # Compute deltas
    print("📈 Computing deltas per volume indicator")
    delta_df = compute_deltas(joined_df, baseline_cols, volume_cols, volume_name_col)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_csv = output_dir / "phase1_volume_comparison.csv"
    joined_df.to_csv(comparison_csv, index=False)
    print(f"✅ Wrote comparison CSV: {comparison_csv}")

    delta_csv = output_dir / "phase1_volume_deltas.csv"
    delta_df.to_csv(delta_csv, index=False)
    print(f"✅ Wrote deltas CSV: {delta_csv}")

    # Generate markdown report with decision logic
    decision_md = output_dir / "phase1_volume_decision.md"
    
    # Compute decision
    decision = "KILL"  # Default
    decision_reason = []
    
    if not delta_df.empty:
        # Check if any indicator shows improvement
        improved_indicators = []
        for _, row in delta_df.iterrows():
            vol_name = row.get("volume_indicator", "unknown")
            delta_exp = row.get("delta_expectancy", 0.0)
            delta_mar = row.get("delta_mar", 0.0)
            delta_trades = row.get("delta_trades", 0.0)
            
            # KEEP criteria: expectancy + MAR up, trades down or equal
            if delta_exp > 0 and delta_mar > 0 and delta_trades <= 0:
                improved_indicators.append(vol_name)
        
        if improved_indicators:
            decision = "KEEP"
            decision_reason.append(f"Indicators showing improvement: {', '.join(improved_indicators)}")
        else:
            # Check for marginal improvements (PARK)
            marginal = []
            for _, row in delta_df.iterrows():
                vol_name = row.get("volume_indicator", "unknown")
                delta_exp = row.get("delta_expectancy", 0.0)
                delta_mar = row.get("delta_mar", 0.0)
                if (delta_exp > 0 and delta_mar <= 0) or (delta_exp <= 0 and delta_mar > 0):
                    marginal.append(vol_name)
            
            if marginal:
                decision = "PARK"
                decision_reason.append(f"Marginal/inconsistent improvements: {', '.join(marginal)}")
            else:
                decision = "KILL"
                decision_reason.append("No indicators show improvement in both expectancy and MAR")
    else:
        decision_reason.append("No delta data available")
    
    with decision_md.open("w", encoding="utf-8") as f:
        f.write("# Phase 1 Volume Study Decision Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        f.write("## Decision\n\n")
        f.write(f"**{decision}**\n\n")
        for reason in decision_reason:
            f.write(f"- {reason}\n")
        f.write("\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Baseline CSV: `{baseline_path}`\n")
        f.write(f"- Volume CSV: `{volume_path}`\n")
        f.write(f"- Matched rows: {len(joined_df)}\n")
        f.write(
            "- ✅ Invariant validated: trades_with_volume <= trades_without_volume\n\n"
        )

        f.write("## Deltas by Volume Indicator\n\n")
        if delta_df.empty:
            f.write("No deltas computed (missing required columns).\n\n")
        else:
            f.write(delta_df.to_markdown(index=False))
            f.write("\n\n")
            
            # Add interpretation
            f.write("### Delta Interpretation\n\n")
            f.write("- **Δ trades**: Positive = more trades with volume (bad if > 0)\n")
            f.write("- **Δ scratches**: Change in scratch count\n")
            f.write("- **Δ ROI**: Change in ROI percentage\n")
            f.write("- **Δ MAR**: Change in MAR (higher is better)\n")
            f.write("- **Δ expectancy**: Change in expectancy (higher is better)\n\n")
            f.write("**Decision Criteria:**\n")
            f.write("- **KEEP**: expectancy ↑ AND MAR ↑ AND trades ≤ 0\n")
            f.write("- **PARK**: Mixed improvements (one metric up, one down)\n")
            f.write("- **KILL**: No improvement OR invariant violation\n\n")

        f.write("## Detailed Comparison\n\n")
        f.write(f"See `{comparison_csv.name}` for row-level comparison.\n\n")

    print(f"✅ Wrote decision report: {decision_md}")


if __name__ == "__main__":
    main()


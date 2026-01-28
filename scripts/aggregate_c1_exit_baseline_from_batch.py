#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_c1_exit_baseline_from_batch.py
-----------------------------------------

Aggregate baseline (no-volume) results from batch_sweeper output.
Reads results/c1_batch_results.csv and groups by pair + c1 + exit.

Output: results/c1_exit_baseline_aggregated.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_roles_json(roles_str: str) -> Dict[str, Any]:
    """Parse roles JSON string and extract c1 and exit."""
    try:
        roles = json.loads(roles_str) if isinstance(roles_str, str) else roles_str
        return {
            "c1": roles.get("c1") or "",
            "exit": roles.get("exit") or "",
        }
    except Exception:
        return {"c1": "", "exit": ""}


def normalize_exit_name(exit_val: Any) -> str:
    """Normalize exit indicator name."""
    if not exit_val or exit_val in [None, False, "null", "None", ""]:
        return "exit_twiggs_money_flow"  # Default exit
    s = str(exit_val).strip()
    if s.startswith("exit_"):
        return s
    return f"exit_{s}"


def extract_pairs_from_config(run_slug: str, history_dir: Path) -> List[str]:
    """Extract pair(s) from archived config file in results_history. Returns list of pairs."""
    run_dir = history_dir / run_slug
    config_path = run_dir / "config.yaml"
    
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            # Check for pairs in various locations
            pairs = cfg.get("pairs") or []
            if isinstance(pairs, list) and len(pairs) > 0:
                return pairs
            # Fallback: check data config
            data_cfg = cfg.get("data") or {}
            if "pair" in data_cfg:
                pair_val = data_cfg["pair"]
                if isinstance(pair_val, list):
                    return pair_val
                return [pair_val] if pair_val else []
        except Exception:
            pass
    
    return []


def get_pairs_from_sweeps_config(sweeps_yaml: Path) -> List[str]:
    """Extract pairs list from sweeps.yaml as fallback."""
    try:
        with sweeps_yaml.open("r", encoding="utf-8") as f:
            sweeps = yaml.safe_load(f)
        static_overrides = sweeps.get("static_overrides") or {}
        pairs = static_overrides.get("pairs") or []
        if isinstance(pairs, list):
            return pairs
    except Exception:
        pass
    return []


def aggregate_baseline(batch_csv: Path, history_dir: Optional[Path] = None, sweeps_yaml: Optional[Path] = None) -> pd.DataFrame:
    """Read batch CSV and aggregate by canonical keys (pair + c1 + exit + timeframe + from_date + to_date)."""
    if not batch_csv.exists():
        raise FileNotFoundError(f"Batch CSV not found: {batch_csv}")

    # Count total lines (excluding header) for quality gate
    with batch_csv.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f) - 1  # Subtract header
        if total_lines < 0:
            total_lines = 0

    # Read with tolerant parsing (skip bad lines)
    try:
        df = pd.read_csv(batch_csv, engine="python", on_bad_lines="skip")
    except TypeError:
        # Fallback for older pandas versions that don't support on_bad_lines
        # Use error_bad_lines=False, warn_bad_lines=False instead
        import warnings
        warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
        df = pd.read_csv(batch_csv, engine="python", error_bad_lines=False, warn_bad_lines=False)

    # Compute skipped lines
    loaded_rows = len(df)
    skipped_rows = total_lines - loaded_rows

    # Enforce strict quality gate: max 0.5% skipped lines (minimum 1 line allowed)
    max_allowed_skipped = max(1, int(total_lines * 0.005))
    if skipped_rows > max_allowed_skipped:
        raise RuntimeError(
            f"CSV quality check failed: {skipped_rows} lines skipped (threshold: {max_allowed_skipped}). "
            f"Total lines: {total_lines}, Loaded: {loaded_rows}. "
            f"File may be corrupted or malformed. Please check {batch_csv}"
        )

    # Print statistics
    print(f"📊 CSV read statistics:")
    print(f"   Total lines (excl. header): {total_lines}")
    print(f"   Loaded rows: {loaded_rows}")
    print(f"   Skipped rows: {skipped_rows}")

    # PRIMARY PATH: Use canonical keys from batch CSV (required)
    required_canonical = ["pair", "c1", "exit"]
    missing_required = [c for c in required_canonical if c not in df.columns]
    
    if missing_required:
        # LEGACY FALLBACK: Try to extract from roles JSON and config archaeology
        print(f"⚠️  Warning: Missing canonical columns {missing_required}. Using legacy extraction.")
        print(f"   This should not happen with updated batch_sweeper.py. Please rerun sweeps.")
        
        # Extract pair from config archaeology (legacy)
        if "pair" not in df.columns:
            if history_dir is None:
                history_dir = PROJECT_ROOT / "results" / "results_history"
            
            if history_dir.exists():
                print(f"   Extracting pair from archived configs in {history_dir}")
                pair_lists = df["run_slug"].apply(
                    lambda slug: extract_pairs_from_config(slug, history_dir)
                )
                
                expanded_rows = []
                for idx, row in df.iterrows():
                    pairs = pair_lists.iloc[idx]
                    if not pairs:
                        continue
                    for pair in pairs:
                        new_row = row.copy()
                        new_row["pair"] = pair
                        expanded_rows.append(new_row)
                
                if expanded_rows:
                    df = pd.DataFrame(expanded_rows)
                else:
                    # Final fallback: sweeps.yaml
                    if sweeps_yaml is None:
                        sweeps_yaml = PROJECT_ROOT / "configs" / "sweeps.yaml"
                    if sweeps_yaml.exists():
                        pairs_list = get_pairs_from_sweeps_config(sweeps_yaml)
                        if pairs_list:
                            expanded_rows = []
                            for _, row in df.iterrows():
                                for pair in pairs_list:
                                    new_row = row.copy()
                                    new_row["pair"] = pair
                                    expanded_rows.append(new_row)
                            df = pd.DataFrame(expanded_rows)
                        else:
                            raise ValueError(
                                f"Could not determine 'pair'. Available columns: {list(df.columns)}. "
                                f"Please rerun batch_sweeper.py to generate canonical schema."
                            )
                    else:
                        raise ValueError(
                            f"Could not determine 'pair'. Available columns: {list(df.columns)}. "
                            f"Please rerun batch_sweeper.py to generate canonical schema."
                        )
        
        # Extract c1 and exit from roles JSON (legacy)
        if "c1" not in df.columns or "exit" not in df.columns:
            if "roles" in df.columns:
                roles_df = df["roles"].apply(parse_roles_json)
                if "c1" not in df.columns:
                    df["c1"] = roles_df.apply(lambda x: x.get("c1", "") or "")
                if "exit" not in df.columns:
                    df["exit"] = roles_df.apply(lambda x: normalize_exit_name(x.get("exit")))
            else:
                if "c1" not in df.columns:
                    df["c1"] = ""
                if "exit" not in df.columns:
                    df["exit"] = "exit_twiggs_money_flow"
    else:
        # Canonical path: use direct columns
        print(f"✅ Using canonical keys from batch CSV: {required_canonical}")
        # Ensure exit is normalized
        if "exit" in df.columns:
            df["exit"] = df["exit"].apply(normalize_exit_name)

    # Filter out rows with volume indicators (baseline only)
    # Check both canonical 'volume' column and legacy 'roles' JSON
    if "volume" in df.columns:
        # Canonical: filter where volume is not "none"
        df = df[df["volume"].isin([None, "", "none", False])]
    elif "roles" in df.columns:
        # Legacy: filter from roles JSON
        df = df[df["roles"].apply(
            lambda x: json.loads(x).get("volume") in [None, False, "null", "None", ""] 
            if isinstance(x, str) else True
        )]

    # Group by canonical keys - MUST include pair for pair-level granularity
    # This ensures baseline aggregation matches volume aggregation granularity (pair × c1 × exit)
    group_cols = ["pair", "c1", "exit"]
    optional_group_cols = ["timeframe", "from_date", "to_date"]
    
    # Add optional keys if present
    for col in optional_group_cols:
        if col in df.columns:
            group_cols.append(col)
    
    available_group_cols = [c for c in group_cols if c in df.columns]

    if not available_group_cols:
        raise ValueError(f"No grouping columns found. Available columns: {list(df.columns)}")
    
    # CRITICAL: pair, c1, and exit are REQUIRED for pair-level aggregation
    # Do NOT pool or average across pairs - each pair must be aggregated separately
    if "pair" not in available_group_cols or "c1" not in available_group_cols or "exit" not in available_group_cols:
        raise ValueError(
            f"Missing required grouping columns. Required: pair, c1, exit. "
            f"Available: {available_group_cols}. "
            f"Please rerun batch_sweeper.py to generate canonical schema."
        )

    # Aggregate metrics
    agg_dict: Dict[str, Any] = {
        "total_trades": "sum",
        "wins": "sum",
        "losses": "sum",
        "scratches": "sum",
    }

    # Add ratio metrics as mean
    for col in ["roi_pct", "max_dd_pct", "expectancy"]:
        if col in df.columns:
            agg_dict[col] = "mean"

    # Check for mar column
    if "mar" in df.columns:
        agg_dict["mar"] = "mean"
    elif "score" in df.columns:
        # Approximate MAR from score if available (or compute later)
        pass

    # Group and aggregate
    grouped = df.groupby(available_group_cols, dropna=False).agg(agg_dict).reset_index()

    # Compute derived metrics
    if "total_trades" in grouped.columns:
        non_scratch = grouped["wins"] + grouped["losses"]
        mask = non_scratch > 0
        grouped["win_rate_ns"] = 0.0
        if mask.any():
            grouped.loc[mask, "win_rate_ns"] = (grouped.loc[mask, "wins"] / non_scratch[mask]) * 100.0

    # Ensure exit column is normalized
    if "exit" in grouped.columns:
        grouped["exit"] = grouped["exit"].apply(normalize_exit_name)

    # Ensure pair column is present in output (required for Phase 1 joins)
    if "pair" not in grouped.columns:
        raise ValueError("Output missing 'pair' column - cannot join with volume results")
    
    # Sort by pair, c1, exit
    grouped = grouped.sort_values(available_group_cols)

    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate baseline (no-volume) results from batch CSV",
    )
    parser.add_argument(
        "--batch-csv",
        type=str,
        default="results/c1_batch_results.csv",
        help="Path to batch results CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/c1_exit_baseline_aggregated.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        default="results/results_history",
        help="Directory containing archived run configs",
    )
    parser.add_argument(
        "--sweeps-yaml",
        type=str,
        default="configs/sweeps.yaml",
        help="Path to sweeps.yaml (used as fallback for pairs)",
    )
    args = parser.parse_args()

    batch_csv = Path(args.batch_csv)
    output_path = Path(args.output)
    history_dir = PROJECT_ROOT / args.history_dir
    sweeps_yaml = PROJECT_ROOT / args.sweeps_yaml

    print(f"Reading batch CSV: {batch_csv}")
    df = aggregate_baseline(batch_csv, history_dir=history_dir, sweeps_yaml=sweeps_yaml)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Wrote aggregated baseline: {output_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()


"""
Phase E-1.4: Corridor sweeps for Phase E archetypes.
Expands c1_params_grid into combos, runs phaseE1 signal geometry per combo,
aggregates leaderboard_geometry_lock.csv into results_summary.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _params_hash(params: dict) -> str:
    """Stable SHA256 hash of sorted JSON representation."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def expand_grid(grid: dict) -> list[dict]:
    """Expand c1_params_grid into combos in deterministic order (sorted keys)."""
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append(dict(zip(keys, prod)))
    return combos


def _resolve_base_config(sweep_cfg: dict, sweep_path: Path) -> dict:
    """Load base_config and deep-merge sweep overrides on top."""
    base_ref = sweep_cfg.get("base_config")
    if not base_ref:
        return dict(sweep_cfg)
    base_path = (sweep_path.parent / base_ref).resolve()
    if not base_path.exists():
        return dict(sweep_cfg)
    base = _load_yaml(base_path)
    out = dict(base)
    for k, v in sweep_cfg.items():
        if k == "base_config":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out.get(k, {}), **v}
        else:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase E-1.4 — Corridor sweeps for Phase E archetypes.",
    )
    parser.add_argument(
        "--sweep-config",
        required=True,
        help="Path to configs/phaseE1_sweeps/*_sweep.yaml",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory (e.g. results/phaseE1_sweeps/ceb)",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=200,
        help="Cap number of parameter combinations (default: 200)",
    )
    args = parser.parse_args(argv)

    sweep_path = Path(args.sweep_config).resolve()
    if not sweep_path.exists():
        print(f"Error: Sweep config not found: {sweep_path}", file=sys.stderr)
        return 1

    sweep_cfg = _load_yaml(sweep_path)
    cfg = _resolve_base_config(sweep_cfg, sweep_path)
    grid = (cfg.get("indicators") or {}).get("c1_params_grid")
    if not grid:
        print("Error: indicators.c1_params_grid is required.", file=sys.stderr)
        return 1

    combos = expand_grid(grid)
    if args.max_combos < len(combos):
        combos = combos[: args.max_combos]
        print(f"Info: Capped to {args.max_combos} combos (grid had more).")

    from scripts.phaseE1_run_signal_geometry import _find_clean_labels, run_from_config

    clean_path = _find_clean_labels(ROOT)
    if clean_path is None:
        print(
            "Error: No opportunity_labels_clean.csv found. Run phaseD6F first.",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    c1 = (cfg.get("indicators") or {}).get("c1") or (cfg.get("system") or {}).get("c1")
    if not c1:
        c1 = sweep_cfg.get("indicators", {}).get("c1")
    if not c1 or c1 == "PLACEHOLDER":
        print("Error: Config must set indicators.c1.", file=sys.stderr)
        return 1

    aggregated: list[dict] = []
    for i, params in enumerate(combos):
        phash = _params_hash(params)
        run_out = out_dir / phash
        run_out.mkdir(parents=True, exist_ok=True)
        run_cfg = dict(cfg)
        run_cfg.setdefault("indicators", {})["c1_params"] = params
        run_cfg["indicators"]["c1"] = c1
        ret = run_from_config(run_cfg, clean_path, run_out)
        if ret != 0:
            print(f"Warning: Run failed for combo {i + 1}/{len(combos)} params={params}")
            aggregated.append({
                "params_hash": phash,
                "params_json": json.dumps(params, sort_keys=True),
                "PASS": False,
                "reject_reason": "run_failed",
                "P_3R_before_2R_disc": None,
                "P_3R_before_2R_val": None,
                "discovery_lift": None,
                "validation_lift": None,
                "annual_signals_per_pair": None,
                "clustering_ratio": None,
            })
            continue
        lb_path = run_out / "leaderboard_geometry_lock.csv"
        if not lb_path.exists():
            aggregated.append({
                "params_hash": phash,
                "params_json": json.dumps(params, sort_keys=True),
                "PASS": False,
                "reject_reason": "no_leaderboard",
                "P_3R_before_2R_disc": None,
                "P_3R_before_2R_val": None,
                "discovery_lift": None,
                "validation_lift": None,
                "annual_signals_per_pair": None,
                "clustering_ratio": None,
            })
            continue
        lb = pd.read_csv(lb_path)
        row = lb.iloc[0].to_dict()
        row["params_hash"] = phash
        row["params_json"] = json.dumps(params, sort_keys=True)
        aggregated.append(row)

    if not aggregated:
        print("No runs completed.")
        return 0

    df = pd.DataFrame(aggregated)
    pass_first = df["PASS"].fillna(False).astype(bool)
    pass_df = df[pass_first].copy()
    fail_df = df[~pass_first].copy()
    pass_df = pass_df.sort_values(
        ["P_3R_before_2R_val", "clustering_ratio"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    fail_df = fail_df.sort_values(
        ["P_3R_before_2R_val", "clustering_ratio"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    result = pd.concat([pass_df, fail_df], ignore_index=True)

    cols = [
        "params_hash",
        "params_json",
        "PASS",
        "reject_reason",
        "P_3R_before_2R_disc",
        "P_3R_before_2R_val",
        "discovery_lift",
        "validation_lift",
        "annual_signals_per_pair",
        "clustering_ratio",
    ]
    result = result[[c for c in cols if c in result.columns]]
    result.to_csv(out_dir / "results_summary.csv", index=False, float_format="%.6f")
    print(f"Corridor sweep complete: {out_dir / 'results_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

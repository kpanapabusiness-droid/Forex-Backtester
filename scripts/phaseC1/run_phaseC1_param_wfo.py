from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASEC1_CONFIG_DIR = ROOT / "configs" / "phaseC1"
WFO_DONE_MARKER = "wfo_done.json"


def _git_hash_if_available() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=ROOT,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()[:12]
    except Exception:
        pass
    return None


def _load_eligible_variants(
    diagnostics_csv: Path,
    all_variants: List[object],
) -> List[object]:
    import pandas as pd

    if not diagnostics_csv.exists():
        raise FileNotFoundError(f"Diagnostics CSV not found: {diagnostics_csv}")
    df = pd.read_csv(diagnostics_csv)
    if df.empty:
        return []
    eligible_ids = set(df.loc[df["eligibility"] == "ELIGIBLE", "variant_id"].astype(str))
    by_id = {v.variant_id: v for v in all_variants}
    missing = sorted(id_ for id_ in eligible_ids if id_ not in by_id)
    if missing:
        raise ValueError(f"Diagnostics CSV references unknown variant ids: {missing}")
    return [by_id[id_] for id_ in sorted(eligible_ids)]


def run_phaseC1_param_wfo(
    wfo_shell_path: Path,
    base_config_path: Path,
    grids_path: Path,
    diagnostics_csv: Path,
    results_root: Path,
    skip_if_done: bool = True,
) -> None:
    """
    Run WFO v2 once per ELIGIBLE C1 parameter variant (from diagnostics CSV).

    Each variant gets its own directory:
      results/phaseC1/wfo_runs/<variant_id>/
    """
    from analytics.phaseC1.phaseC1_participation_diagnostics import load_c1_param_variants
    from core.utils import read_yaml

    wfo_shell_path = wfo_shell_path.resolve()
    base_config_path = base_config_path.resolve()
    grids_path = grids_path.resolve()
    diagnostics_csv = diagnostics_csv.resolve()
    results_root = results_root.resolve()

    if not wfo_shell_path.exists():
        raise FileNotFoundError(f"WFO shell config not found: {wfo_shell_path}")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Phase C.1 base config not found: {base_config_path}")

    base_template = read_yaml(base_config_path)
    wfo_shell = yaml.safe_load(wfo_shell_path.read_text(encoding="utf-8")) or {}

    all_variants = load_c1_param_variants(grids_path)
    eligible_variants = _load_eligible_variants(diagnostics_csv, all_variants)

    wfo_runs_root = results_root / "wfo_runs"
    wfo_runs_root.mkdir(parents=True, exist_ok=True)

    for idx, variant in enumerate(eligible_variants, start=1):
        var_root = wfo_runs_root / variant.variant_id
        var_root.mkdir(parents=True, exist_ok=True)

        if skip_if_done and (var_root / WFO_DONE_MARKER).exists():
            print(f"[{idx}/{len(eligible_variants)}] Skip {variant.variant_id} (already done).")
            continue

        base_cfg = dict(base_template)
        base_cfg.setdefault("indicators", {})
        base_cfg["indicators"]["c1"] = variant.base_name
        base_cfg.setdefault("indicator_params", {})
        base_cfg["indicator_params"][variant.base_name] = dict(variant.params)

        base_cfg_path = var_root / "base_config.yaml"
        base_cfg_path.write_text(
            yaml.safe_dump(base_cfg, sort_keys=False),
            encoding="utf-8",
        )

        wfo_cfg = dict(wfo_shell)
        wfo_cfg["base_config"] = base_cfg_path.name
        wfo_cfg["output_root"] = str(var_root)
        wfo_cfg_path = var_root / "phaseC1_wfo.yaml"
        wfo_cfg_path.write_text(
            yaml.safe_dump(wfo_cfg, sort_keys=False),
            encoding="utf-8",
        )

        print(f"[{idx}/{len(eligible_variants)}] Running WFO for {variant.variant_id} ...")
        cmd = [sys.executable, "scripts/walk_forward.py", "--config", str(wfo_cfg_path)]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            raise RuntimeError(
                f"WFO failed for {variant.variant_id} (exit code {result.returncode})"
            )

        fold_dates = [
            {k: f.get(k) for k in ("train_start", "train_end", "test_start", "test_end")}
            for f in (wfo_cfg.get("folds") or [])
        ]
        done_meta = {
            "variant": variant.variant_id,
            "base_c1_name": variant.base_name,
            "params": dict(variant.params),
            "fold_dates": fold_dates,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_hash": _git_hash_if_available(),
        }
        (var_root / WFO_DONE_MARKER).write_text(
            json.dumps(done_meta, indent=2),
            encoding="utf-8",
        )
        print(f"  Done: {var_root}")

    if not eligible_variants:
        print("No ELIGIBLE variants found in diagnostics CSV; nothing to run.")
    else:
        print("\nPhase C.1 parameter WFO runs complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase C.1 — Run WFO v2 for ELIGIBLE C1 parameter variants.",
    )
    parser.add_argument(
        "--wfo-shell",
        default=str(PHASEC1_CONFIG_DIR / "phaseC1_wfo_shell.yaml"),
        help="Path to phaseC1_wfo_shell.yaml.",
    )
    parser.add_argument(
        "--base-config",
        default=str(PHASEC1_CONFIG_DIR / "phaseC1_base.yaml"),
        help="Path to phaseC1_base.yaml.",
    )
    parser.add_argument(
        "--param-grids",
        default=str(PHASEC1_CONFIG_DIR / "phaseC1_param_grids.yaml"),
        help="Path to phaseC1_param_grids.yaml.",
    )
    parser.add_argument(
        "--diagnostics-csv",
        default=str(ROOT / "results" / "phaseC1" / "diagnostics" / "participation_stats.csv"),
        help="Path to participation_stats.csv produced by Phase C.1 diagnostics.",
    )
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results" / "phaseC1"),
        help="Results root for Phase C.1 WFO runs (default: results/phaseC1).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Run all ELIGIBLE variants even if wfo_done.json exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phaseC1_param_wfo(
        wfo_shell_path=Path(args.wfo_shell),
        base_config_path=Path(args.base_config),
        grids_path=Path(args.param_grids),
        diagnostics_csv=Path(args.diagnostics_csv),
        results_root=Path(args.results_root),
        skip_if_done=not args.no_skip,
    )


if __name__ == "__main__":
    main()


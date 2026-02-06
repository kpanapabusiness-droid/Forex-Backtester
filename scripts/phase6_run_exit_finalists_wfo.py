# Phase 6.3 — Run WFO v2 for each exit finalist: Mode Y then Mode X (Mode X vs Y showdown).
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DIR = ROOT / "configs" / "phase6_exit" / "c1_as_exit_final"
RESULTS_ROOT = ROOT / "results" / "phase6_exit" / "c1_as_exit_final"
SHELL_PATH = ROOT / "configs" / "phase6_exit" / "c1_as_exit" / "phase6_c1_as_exit_shell.yaml"
FINALISTS_CSV = ROOT / "results" / "phase6_exit" / "c1_as_exit" / "finalists.csv"
MODES = [("mode_Y", "flip_only"), ("mode_X", "disagree")]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _ensure_configs(
    shell_path: Path,
    config_dir: Path,
    results_root: Path,
    exit_c1_name: str,
) -> None:
    """Generate phase6_exit_<name>_mode_Y.yaml and _mode_X.yaml (and WFO configs) in config_dir."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(results_root)
    shell = _load_yaml(Path(shell_path))

    for mode_slug, c1_exit_mode in MODES:
        base_name = f"phase6_exit_{exit_c1_name}_{mode_slug}.yaml"
        base_path = config_dir / base_name
        cfg = dict(shell)
        exit_cfg = dict(cfg.get("exit") or {})
        exit_cfg["exit_c1_name"] = exit_c1_name
        exit_cfg["c1_exit_mode"] = c1_exit_mode
        cfg["exit"] = exit_cfg
        base_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        wfo_name = f"wfo_{exit_c1_name}_{mode_slug}.yaml"
        wfo_path = config_dir / wfo_name
        output_root = str(results_root / exit_c1_name / mode_slug)
        wfo_cfg = {
            "base_config": base_name,
            "data_scope": {"from_date": "2019-01-01", "to_date": "2026-01-01"},
            "fold_scheme": {"train_months": 36, "test_months": 12, "step_months": 12},
            "engine": {"cache_on": False, "spreads_on": True},
            "output_root": output_root,
        }
        wfo_path.write_text(yaml.safe_dump(wfo_cfg, sort_keys=False), encoding="utf-8")


def run_finalists_wfo(
    finalists_csv: Path,
    shell_path: Path,
    config_dir: Path,
    results_root: Path,
) -> None:
    """Load finalists, for each run Mode Y WFO then Mode X WFO. Fail-fast on engine errors only."""
    finalists_csv = Path(finalists_csv)
    if not finalists_csv.exists():
        raise FileNotFoundError(f"Finalists CSV not found: {finalists_csv}")
    df = pd.read_csv(finalists_csv)
    if "exit_c1_name" not in df.columns:
        raise ValueError("finalists.csv must have column exit_c1_name")
    names = df["exit_c1_name"].astype(str).str.strip().tolist()
    names = [n for n in names if n and not n.startswith("nan")]

    shell_path = Path(shell_path)
    config_dir = Path(config_dir)
    results_root = Path(results_root)
    if not shell_path.exists():
        raise FileNotFoundError(f"Shell config not found: {shell_path}")

    for exit_c1_name in names:
        _ensure_configs(shell_path, config_dir, results_root, exit_c1_name)
        for mode_slug, _ in MODES:
            wfo_path = config_dir / f"wfo_{exit_c1_name}_{mode_slug}.yaml"
            if not wfo_path.exists():
                raise FileNotFoundError(f"WFO config not found: {wfo_path}")
            print(f"\nPhase 6.3 — exit_c1_name={exit_c1_name} {mode_slug}")
            cmd = [sys.executable, "scripts/walk_forward.py", "--config", str(wfo_path)]
            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                raise RuntimeError(
                    f"WFO failed for {exit_c1_name} {mode_slug} (exit code {result.returncode})"
                )
    print("\nPhase 6.3 finalist WFO runs complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6.3: Run Mode Y and Mode X WFO for each exit finalist."
    )
    parser.add_argument("--finalists", default=str(FINALISTS_CSV), help="Path to finalists.csv.")
    parser.add_argument("--shell", default=str(SHELL_PATH), help="Shell config (exit block overridden).")
    parser.add_argument("--config-dir", default=str(CONFIG_DIR), help="Generated config directory.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT), help="Output root for runs.")
    args = parser.parse_args()
    run_finalists_wfo(
        finalists_csv=Path(args.finalists),
        shell_path=Path(args.shell),
        config_dir=Path(args.config_dir),
        results_root=Path(args.results_root),
    )


if __name__ == "__main__":
    main()

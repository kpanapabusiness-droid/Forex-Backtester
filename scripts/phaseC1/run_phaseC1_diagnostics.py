from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from analytics.phaseC1.phaseC1_participation_diagnostics import (  # noqa: E402
    run_phaseC1_participation_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase C.1 — Run C1 participation diagnostics for all parameter variants.",
    )
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs" / "phaseC1" / "phaseC1_base.yaml"),
        help="Path to Phase C.1 base config YAML.",
    )
    parser.add_argument(
        "--param-grids",
        default=str(ROOT / "configs" / "phaseC1" / "phaseC1_param_grids.yaml"),
        help="Path to Phase C.1 param grids YAML.",
    )
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results" / "phaseC1"),
        help="Results root for Phase C.1 diagnostics (default: results/phaseC1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phaseC1_participation_diagnostics(
        base_config_path=Path(args.base_config),
        grids_path=Path(args.param_grids),
        results_root=Path(args.results_root),
    )


if __name__ == "__main__":
    main()


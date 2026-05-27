"""Sidecar CLI entry point: ``python -m deployment.sidecar [--args]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from deployment.sidecar.config import load_sidecar_config
from deployment.sidecar.sidecar import initialize_and_run


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m deployment.sidecar",
        description="Arc 10 DLR Phase 1 sidecar — UTC-native bar fetch + signal emission",
    )
    p.add_argument(
        "--winning-config",
        type=Path,
        required=True,
        help="Path to winning_config.yaml (e.g. configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml)",
    )
    p.add_argument(
        "--sidecar-config",
        type=Path,
        default=None,
        help="Optional sidecar.yaml override (see deployment/README.md §sidecar-config-schema)",
    )
    p.add_argument(
        "--sidecar-root",
        type=Path,
        required=True,
        help="Parent directory of signals_out/, signals_processed/, etc.",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Run N cycles then exit (default: run forever)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    cfg = load_sidecar_config(
        winning_config_path=args.winning_config,
        sidecar_yaml_path=args.sidecar_config,
        sidecar_root=args.sidecar_root,
    )
    initialize_and_run(cfg, iterations=args.iterations)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised in deployment
    sys.exit(main())

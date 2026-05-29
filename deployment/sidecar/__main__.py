"""Sidecar CLI entry point: ``python -m deployment.sidecar [--args]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from deployment.sidecar.config import load_sidecar_config
from deployment.sidecar.mt5_data_fetcher import Mt5ConnectParams
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
    p.add_argument(
        "--mt5-path",
        default=None,
        help=(
            "Absolute path to the target broker's MT5 terminal64.exe. Makes the "
            "sidecar attach deterministically to that terminal — required when "
            "multiple MT5 terminals run on one host (multi-broker VPS). Omitted: "
            "legacy default-attach (first terminal to answer)."
        ),
    )
    p.add_argument(
        "--mt5-login",
        type=int,
        default=None,
        help="MT5 account login for sidecar-side re-auth (optional; rarely needed).",
    )
    p.add_argument(
        "--mt5-password",
        default=None,
        help="MT5 account password for sidecar-side re-auth (optional; rarely needed).",
    )
    p.add_argument(
        "--mt5-server",
        default=None,
        help="MT5 broker server name for sidecar-side re-auth (optional; rarely needed).",
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
    connect = Mt5ConnectParams(
        path=args.mt5_path,
        login=args.mt5_login,
        password=args.mt5_password,
        server=args.mt5_server,
    )
    initialize_and_run(cfg, iterations=args.iterations, connect=connect)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised in deployment
    sys.exit(main())

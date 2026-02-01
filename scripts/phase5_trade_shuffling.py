# scripts/phase5_trade_shuffling.py
# Phase 5.1 — CLI: load WFO OOS trades, run trade shuffling, write results/phase5/.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def _resolve_run_dir_and_trades_paths(root: Path) -> tuple[Path, list[Path]]:
    """
    Resolve the WFO run directory and list of fold OOS trades.csv paths.
    - If root contains fold_*/out_of_sample/trades.csv directly → run_dir = root.
    - Else if root contains subdirectories (run folders) → pick latest by mtime, recurse.
    Returns (run_dir, list of Path to trades.csv). May return ([], []) if nothing found.
    """
    root = Path(root).resolve()
    if not root.exists() or not root.is_dir():
        return root, []

    fold_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: p.name,
    )
    trades_paths = []
    for fold_dir in fold_dirs:
        trades_path = fold_dir / "out_of_sample" / "trades.csv"
        if trades_path.exists():
            trades_paths.append(trades_path)

    if trades_paths:
        return root, trades_paths

    run_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not run_dirs:
        return root, []

    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return _resolve_run_dir_and_trades_paths(latest)


def load_oos_trades_from_wfo(
    wfo_results_dir: str | Path, *, allow_empty: bool = False
) -> tuple[pd.DataFrame, Path, int]:
    """
    Discover WFO run directory and concatenate all fold OOS trades.csv.
    - If wfo_results_dir contains fold_*/out_of_sample/trades.csv directly → use it.
    - Else if it contains run subdirectories → select the most recent by mtime.
    - Requires at least one fold OOS trades.csv; otherwise raises ValueError.
    - If all fold CSVs are empty (no trade rows), raises ValueError unless allow_empty=True.
    Returns (trades_df, run_dir_resolved, n_folds).
    """
    root = Path(wfo_results_dir).resolve()
    if not root.exists():
        raise ValueError(
            f"WFO results directory does not exist: {root}\n"
            "Expected: a directory containing either (a) fold_XX/out_of_sample/trades.csv "
            "or (b) run subdirectories (e.g. 20260201_121546) each with fold_XX/out_of_sample/.\n"
            "Override with: --wfo-results-dir <path>"
        )

    run_dir, trades_paths = _resolve_run_dir_and_trades_paths(root)

    if not trades_paths:
        raise ValueError(
            f"No OOS trades found under: {run_dir}\n"
            "Expected structure: fold_01/out_of_sample/trades.csv, fold_02/out_of_sample/trades.csv, ...\n"
            "Ensure a WFO v2 run has completed and produced out-of-sample trades.\n"
            "Override with: --wfo-results-dir <path-to-run-directory>"
        )

    frames = []
    for trades_path in sorted(trades_paths):
        try:
            df = pd.read_csv(trades_path)
            fold_dir = trades_path.parent.parent
            if "fold_id" not in df.columns:
                try:
                    fid = int(fold_dir.name.split("_")[1])
                    df["fold_id"] = fid
                except Exception:
                    pass
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise ValueError(
            f"No readable OOS trades under: {run_dir}\n"
            f"Found {len(trades_paths)} trades.csv file(s) but none could be read.\n"
            "Override with: --wfo-results-dir <path>"
        )

    out_df = pd.concat(frames, ignore_index=True)
    if out_df.empty and not allow_empty:
        raise ValueError(
            f"No OOS trade rows under: {run_dir}\n"
            f"Found {len(trades_paths)} fold trades.csv file(s) but all are empty (no trade rows).\n"
            "Run WFO v2 to produce out-of-sample trades, or use --wfo-results-dir <path>, "
            "or pass --allow-empty-trades to run with 0 trades."
        )
    return out_df, run_dir, len(trades_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5.1: Trade shuffling / path risk. Load WFO OOS trades, run shuffling, write results/phase5/."
    )
    parser.add_argument(
        "--wfo-results-dir",
        type=str,
        default="results/wfo",
        help="WFO run directory (e.g. results/wfo or results/wfo/<run_id>). Default: results/wfo",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/phase5",
        help="Output directory for runs CSV and summary. Default: results/phase5",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=10_000,
        help="Number of shuffle simulations. Default: 10000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility. Default: 123",
    )
    parser.add_argument(
        "--starting-balance",
        type=float,
        default=None,
        help="Starting balance (default: from config or 10000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config YAML to read starting_balance from risk.starting_balance",
    )
    parser.add_argument(
        "--allow-empty-trades",
        action="store_true",
        help="If WFO run has no OOS trade rows, run anyway and write n_trades=0 summary (default: error)",
    )
    args = parser.parse_args()

    trades_df, wfo_run_dir, n_folds = load_oos_trades_from_wfo(
        args.wfo_results_dir, allow_empty=args.allow_empty_trades
    )
    n_trades_loaded = len(trades_df)
    print(f"WFO run directory: {wfo_run_dir.resolve()}")
    print(f"Folds: {n_folds}, OOS trades loaded: {n_trades_loaded}")
    if n_trades_loaded == 0:
        print("Note: 0 OOS trades — writing empty shuffling summary.")

    starting_balance = args.starting_balance
    if starting_balance is None and args.config:
        try:
            import yaml
            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            risk = (cfg or {}).get("risk") or {}
            starting_balance = float(risk.get("starting_balance", 10_000.0))
        except Exception:
            starting_balance = 10_000.0
    if starting_balance is None:
        starting_balance = 10_000.0

    from analytics.trade_shuffling import run_trade_shuffling

    summary = run_trade_shuffling(
        trades_df=trades_df,
        starting_balance=starting_balance,
        n_sims=args.n_sims,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(f"Trade shuffling complete. n_trades={summary.get('n_trades', 0)} n_sims={summary.get('n_sims', 0)}")
    print(f"Output: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()

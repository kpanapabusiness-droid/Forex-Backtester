# Phase 6.3 — Select top exit-C1 finalists from Phase 6.2 leaderboard for Mode X vs Y showdown.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

C1_AS_EXIT_ROOT = ROOT / "results" / "phase6_exit" / "c1_as_exit"
LEADERBOARD_CSV = C1_AS_EXIT_ROOT / "leaderboard_exit_c1.csv"
FINALISTS_CSV = C1_AS_EXIT_ROOT / "finalists.csv"
TOP_N = 5
REASON_SELECTED = "top5_dd_expectancy"


def select_finalists(
    leaderboard_path: Path,
    output_path: Path,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Load leaderboard, filter PASS + no flags, rank by DD asc then expectancy desc, take top_n."""
    path = Path(leaderboard_path)
    if not path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {path}")
    df = pd.read_csv(path)

    for col in ("decision", "insufficient_trades_flag", "churn_trade_ratio_flag", "churn_hold_ratio_flag"):
        if col not in df.columns:
            raise ValueError(f"Leaderboard missing column: {col}")

    passed = df[
        (df["decision"].astype(str) == "PASS")
        & (df["insufficient_trades_flag"].astype(str).str.lower().isin(("false", "0", "")))
        & (df["churn_trade_ratio_flag"].astype(str).str.lower().isin(("false", "0", "")))
        & (df["churn_hold_ratio_flag"].astype(str).str.lower().isin(("false", "0", "")))
        & (df["exit_c1_name"].astype(str) != "c1_coral")
    ].copy()

    if passed.empty:
        out = pd.DataFrame(columns=["exit_c1_name", "reason_selected"])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        return out

    dd = pd.to_numeric(passed["worst_fold_max_drawdown_pct"], errors="coerce").fillna(1e9)
    exp = pd.to_numeric(passed["worst_fold_expectancy_r"], errors="coerce").fillna(-1e9)
    passed = passed.assign(_dd=dd, _exp=exp)
    passed = passed.sort_values(by=["_dd", "_exp"], ascending=[True, False]).drop(columns=["_dd", "_exp"])
    passed = passed.head(top_n)

    out = passed[["exit_c1_name"]].copy()
    out["reason_selected"] = REASON_SELECTED
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6.3: Select top exit-C1 finalists from leaderboard for Mode X vs Y showdown."
    )
    parser.add_argument(
        "--leaderboard",
        default=str(LEADERBOARD_CSV),
        help="Path to leaderboard_exit_c1.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(FINALISTS_CSV),
        help="Path to write finalists.csv.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_N,
        help="Number of finalists to select.",
    )
    args = parser.parse_args()
    df = select_finalists(
        leaderboard_path=Path(args.leaderboard),
        output_path=Path(args.output),
        top_n=args.top,
    )
    print(f"Wrote {args.output} ({len(df)} finalists)")


if __name__ == "__main__":
    main()

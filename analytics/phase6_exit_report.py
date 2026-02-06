# Phase 6 — Exit variant comparison report. Reads WFO outputs, writes CSV + notes.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS_ROOT = ROOT / "results" / "phase6_exit"

VARIANT_SLUGS = [
    "baseline_A_coral_disagree_exit",
    "variant_B_tmf_exit",
    "variant_C_coral_flip_only_exit",
    "variant_D1_tmf_OR_coral_flip_exit",
]

CHURN_TRADE_RATIO_CEILING = 1.25
CHURN_HOLD_RATIO_FLOOR = 0.70


def _latest_run_dir(variant_dir: Path) -> Path | None:
    run_dirs = sorted(
        (p for p in variant_dir.iterdir() if p.is_dir() and (p / "wfo_run_meta.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _median_hold_days(trades_path: Path) -> float:
    if not trades_path.exists():
        return float("nan")
    try:
        df = pd.read_csv(trades_path)
        if df.empty or "entry_date" not in df.columns or "exit_date" not in df.columns:
            return float("nan")
        ed = pd.to_datetime(df["entry_date"], errors="coerce")
        xd = pd.to_datetime(df["exit_date"], errors="coerce")
        days = (xd - ed).dt.days.dropna()
        return float(days.median()) if not days.empty else float("nan")
    except Exception:
        return float("nan")


def _fold_metrics(oos_dir: Path) -> dict:
    from scripts.batch_sweeper import parse_summary_or_trades  # noqa: E402

    m = parse_summary_or_trades(oos_dir)
    trades_path = oos_dir / "trades.csv"
    total_trades = int(m.get("total_trades") or 0)
    max_dd_pct = float(m.get("max_dd_pct") or 0.0)
    expectancy = float(m.get("expectancy") or 0.0)
    median_hold = _median_hold_days(trades_path)
    return {
        "total_trades": total_trades,
        "max_dd_pct": max_dd_pct,
        "expectancy": expectancy,
        "median_hold_days": median_hold,
    }


def _aggregate_variant(run_dir: Path) -> dict:
    fold_dirs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")),
        key=lambda p: p.name,
    )
    rows = []
    for fold_dir in fold_dirs:
        oos = fold_dir / "out_of_sample"
        if oos.exists():
            rows.append(_fold_metrics(oos))
    if not rows:
        return {
            "worst_fold_max_drawdown_pct": float("nan"),
            "worst_fold_expectancy_r": float("nan"),
            "trade_count_total": 0,
            "median_bars_held": float("nan"),
            "fold_level_available": False,
            "worst_fold_trade_count": 0,
            "fold_median_holds": [],
            "fold_trade_counts": [],
        }
    total_trades = sum(r["total_trades"] for r in rows)
    worst_dd = min((r["max_dd_pct"] for r in rows), default=0.0)
    worst_expectancy_fold = min(rows, key=lambda r: r["expectancy"])
    worst_fold_expectancy = worst_expectancy_fold["expectancy"]
    median_holds = [r["median_hold_days"] for r in rows if pd.notna(r["median_hold_days"])]
    median_bars_held = float(pd.Series(median_holds).median()) if median_holds else float("nan")
    return {
        "worst_fold_max_drawdown_pct": worst_dd,
        "worst_fold_expectancy_r": worst_fold_expectancy,
        "trade_count_total": total_trades,
        "median_bars_held": median_bars_held,
        "fold_level_available": True,
        "worst_fold_trade_count": min(r["total_trades"] for r in rows),
        "fold_median_holds": median_holds,
        "fold_trade_counts": [r["total_trades"] for r in rows],
    }


def _decision_and_reason(
    row: dict, ref_trades: int, ref_hold: float, variant_slug: str
) -> tuple[str, str]:
    trade_ratio = (row["trade_count_total"] / ref_trades) if ref_trades else float("nan")
    hold_ratio = (row["median_bars_held"] / ref_hold) if ref_hold and pd.notna(ref_hold) else float("nan")

    reasons = []
    if not row.get("fold_level_available"):
        reasons.append("fold-level unavailable")
    if pd.notna(trade_ratio) and trade_ratio > CHURN_TRADE_RATIO_CEILING:
        reasons.append(f"trade_count_ratio_vs_A={trade_ratio:.2f}>1.25")
    if pd.notna(hold_ratio) and hold_ratio < CHURN_HOLD_RATIO_FLOOR:
        reasons.append(f"median_hold_ratio_vs_A={hold_ratio:.2f}<0.70")

    if reasons:
        return "DIAGNOSTIC", "; ".join(reasons)
    if row.get("worst_fold_expectancy_r", 0) >= 0 and row.get("worst_fold_max_drawdown_pct", 0) >= -20:
        return "KEEP", "within churn limits"
    return "DISCARD", "DD or expectancy fail"


def build_report(phase6_root: Path) -> pd.DataFrame:
    phase6_root = phase6_root.resolve()
    if not phase6_root.exists():
        raise FileNotFoundError(f"Phase 6 results root not found: {phase6_root}")

    aggs_by_slug = {}
    for slug in VARIANT_SLUGS:
        variant_dir = phase6_root / slug
        run_dir = _latest_run_dir(variant_dir) if variant_dir.exists() else None
        if run_dir is None:
            aggs_by_slug[slug] = None
            continue
        aggs_by_slug[slug] = _aggregate_variant(run_dir)

    ref_trades = 0
    ref_hold = float("nan")
    ref_agg = aggs_by_slug.get("baseline_A_coral_disagree_exit")
    if ref_agg:
        ref_trades = ref_agg["trade_count_total"]
        ref_hold = ref_agg["median_bars_held"]

    records = []
    for slug in VARIANT_SLUGS:
        agg = aggs_by_slug.get(slug)
        if agg is None:
            records.append(
                {
                    "variant": slug,
                    "worst_fold_max_drawdown_pct": float("nan"),
                    "worst_fold_expectancy_r": float("nan"),
                    "trade_count_total": 0,
                    "median_bars_held": float("nan"),
                    "trade_count_ratio_vs_A": float("nan"),
                    "median_hold_ratio_vs_A": float("nan"),
                    "decision": "DIAGNOSTIC",
                    "reason": "no WFO run found",
                }
            )
            continue
        trade_ratio = (
            (agg["trade_count_total"] / ref_trades) if ref_trades else float("nan")
        )
        hold_ratio = (
            (agg["median_bars_held"] / ref_hold)
            if ref_hold and pd.notna(ref_hold) else float("nan")
        )
        decision, reason = _decision_and_reason(agg, ref_trades, ref_hold, slug)
        records.append(
            {
                "variant": slug,
                "worst_fold_max_drawdown_pct": agg["worst_fold_max_drawdown_pct"],
                "worst_fold_expectancy_r": agg["worst_fold_expectancy_r"],
                "trade_count_total": agg["trade_count_total"],
                "median_bars_held": agg["median_bars_held"],
                "trade_count_ratio_vs_A": trade_ratio,
                "median_hold_ratio_vs_A": hold_ratio,
                "decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(records)


def write_notes(phase6_root: Path, out_path: Path, df: pd.DataFrame) -> None:
    lines = []
    for _, row in df.iterrows():
        v = row["variant"]
        dd = row["worst_fold_max_drawdown_pct"]
        exp = row["worst_fold_expectancy_r"]
        lines.append(
            f"**{v}**: Worst-fold DD%={dd:.2f}, expectancy_r={exp:.2f}. "
            "DD/expectancy on worst fold and churn flags (trade_count_ratio_vs_A>1.25 or "
            "median_hold_ratio_vs_A<0.70) drive decision. "
            f"Decision={row['decision']}; {row['reason']}."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Build exit comparison report from WFO outputs.")
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT),
        help="Root directory (e.g. results/phase6_exit).",
    )
    args = parser.parse_args()
    root = Path(args.results_root).resolve()
    df = build_report(root)
    csv_path = root / "exit_comparison_report.csv"
    notes_path = root / "exit_comparison_notes.md"
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_notes(root, notes_path, df)
    print(f"Wrote {csv_path}")
    print(f"Wrote {notes_path}")


if __name__ == "__main__":
    main()

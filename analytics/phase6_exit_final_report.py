# Phase 6.3 — Final comparison: Mode Y vs Mode X vs Baseline A; KEEP/DISCARD per exit.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE6_ROOT = ROOT / "results" / "phase6_exit"
C1_AS_EXIT_FINAL_ROOT = PHASE6_ROOT / "c1_as_exit_final"
BASELINE_A_SLUG = "baseline_A_coral_disagree_exit"
CHURN_TRADE_RATIO_CEILING = 1.25
CHURN_HOLD_RATIO_FLOOR = 0.70
DECISIONS_CSV = PHASE6_ROOT / "exit_final_decisions.csv"
DECISIONS_MD = PHASE6_ROOT / "exit_final_decisions.md"


def _latest_run_dir_by_mtime(variant_dir: Path) -> Path | None:
    run_dirs = [
        p
        for p in variant_dir.iterdir()
        if p.is_dir() and (p / "wfo_run_meta.json").exists()
    ]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


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


def _aggregate_run(run_dir: Path) -> dict:
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
        }
    worst_dd = min((r["max_dd_pct"] for r in rows), default=0.0)
    worst_expectancy = min(r["expectancy"] for r in rows)
    total_trades = sum(r["total_trades"] for r in rows)
    median_holds = [r["median_hold_days"] for r in rows if pd.notna(r["median_hold_days"])]
    median_bars_held = float(pd.Series(median_holds).median()) if median_holds else float("nan")
    return {
        "worst_fold_max_drawdown_pct": worst_dd,
        "worst_fold_expectancy_r": worst_expectancy,
        "trade_count_total": total_trades,
        "median_bars_held": median_bars_held,
    }


def _dd_improvement_rel(candidate_dd: float, baseline_dd: float) -> float:
    """Relative DD improvement (>=0.10 = 10% better). Uses absolute values; DD is typically negative."""
    try:
        if pd.isna(candidate_dd) or pd.isna(baseline_dd):
            return 0.0
        base = abs(float(baseline_dd))
        cand = abs(float(candidate_dd))
        if base <= 0.0:
            return 0.0
        return (base - cand) / base
    except Exception:
        return 0.0


def _expectancy_delta(candidate_exp: float, baseline_exp: float) -> float:
    """Expectancy delta (candidate - baseline)."""
    try:
        if pd.isna(candidate_exp) or pd.isna(baseline_exp):
            return 0.0
        return float(candidate_exp) - float(baseline_exp)
    except Exception:
        return 0.0


def _mode_qualifies(
    dd: float,
    exp: float,
    ref_dd: float,
    ref_exp: float,
    dd_threshold: float = 0.10,
    exp_threshold: float = 0.05,
) -> tuple[bool, str, float, float]:
    """Return (qualifies, qualifies_by, dd_improvement_rel, expectancy_delta)."""
    dd_impr = _dd_improvement_rel(dd, ref_dd)
    exp_delta = _expectancy_delta(exp, ref_exp)
    qualifies_dd = dd_impr >= dd_threshold
    qualifies_exp = exp_delta >= exp_threshold
    if qualifies_dd and qualifies_exp:
        qualifies_by = "both"
    elif qualifies_dd:
        qualifies_by = "dd"
    elif qualifies_exp:
        qualifies_by = "expectancy"
    else:
        qualifies_by = "none"
    return qualifies_by != "none", qualifies_by, dd_impr, exp_delta


def _churn_breach(trade_ratio: float, hold_ratio: float) -> bool:
    if pd.notna(trade_ratio) and trade_ratio > CHURN_TRADE_RATIO_CEILING:
        return True
    if pd.notna(hold_ratio) and hold_ratio < CHURN_HOLD_RATIO_FLOOR:
        return True
    return False


def build_final_decisions(
    phase6_root: Path,
    c1_as_exit_final_root: Path,
) -> tuple[pd.DataFrame, str]:
    """Compare Mode Y / Mode X vs Baseline A; KEEP if either improves (DD/expectancy) without churn else DISCARD."""
    phase6_root = Path(phase6_root).resolve()
    c1_as_exit_final_root = Path(c1_as_exit_final_root).resolve()
    baseline_dir = phase6_root / BASELINE_A_SLUG
    baseline_run = _latest_run_dir_by_mtime(baseline_dir) if baseline_dir.exists() else None
    baseline_run_id = baseline_run.name if baseline_run else None
    ref_agg = _aggregate_run(baseline_run) if baseline_run else {}
    ref_dd = ref_agg.get("worst_fold_max_drawdown_pct", float("nan"))
    ref_exp = ref_agg.get("worst_fold_expectancy_r", float("nan"))
    ref_trades = ref_agg.get("trade_count_total", 0) or 0
    ref_hold = ref_agg.get("median_bars_held", float("nan"))

    rows: list[dict] = []
    if not c1_as_exit_final_root.exists():
        df = pd.DataFrame(columns=[
            "exit_c1_name", "decision", "decision_reason",
            "baseline_dd", "baseline_expectancy", "baseline_trades", "baseline_hold",
            "mode_Y_dd", "mode_Y_expectancy", "mode_Y_trades", "mode_Y_hold",
            "mode_X_dd", "mode_X_expectancy", "mode_X_trades", "mode_X_hold",
        ])
        return df, f"Baseline A run_id: {baseline_run_id or 'not found'}."

    for exit_dir in sorted(c1_as_exit_final_root.iterdir()):
        if not exit_dir.is_dir() or exit_dir.name.startswith("_"):
            continue
        exit_c1_name = exit_dir.name
        mode_y_dir = exit_dir / "mode_Y"
        mode_x_dir = exit_dir / "mode_X"
        run_y = _latest_run_dir_by_mtime(mode_y_dir) if mode_y_dir.exists() else None
        run_x = _latest_run_dir_by_mtime(mode_x_dir) if mode_x_dir.exists() else None
        agg_y = _aggregate_run(run_y) if run_y else {}
        agg_x = _aggregate_run(run_x) if run_x else {}

        dd_y = agg_y.get("worst_fold_max_drawdown_pct", float("nan"))
        exp_y = agg_y.get("worst_fold_expectancy_r", float("nan"))
        trades_y = agg_y.get("trade_count_total", 0) or 0
        hold_y = agg_y.get("median_bars_held", float("nan"))
        dd_x = agg_x.get("worst_fold_max_drawdown_pct", float("nan"))
        exp_x = agg_x.get("worst_fold_expectancy_r", float("nan"))
        trades_x = agg_x.get("trade_count_total", 0) or 0
        hold_x = agg_x.get("median_bars_held", float("nan"))

        trade_ratio_y = (trades_y / ref_trades) if ref_trades else float("nan")
        hold_ratio_y = (hold_y / ref_hold) if ref_hold and pd.notna(ref_hold) else float("nan")
        trade_ratio_x = (trades_x / ref_trades) if ref_trades else float("nan")
        hold_ratio_x = (hold_x / ref_hold) if ref_hold and pd.notna(ref_hold) else float("nan")

        qualifies_y, qualifies_by_y, dd_impr_y, exp_delta_y = _mode_qualifies(dd_y, exp_y, ref_dd, ref_exp)
        qualifies_x, qualifies_by_x, dd_impr_x, exp_delta_x = _mode_qualifies(dd_x, exp_x, ref_dd, ref_exp)
        churn_y = _churn_breach(trade_ratio_y, hold_ratio_y)
        churn_x = _churn_breach(trade_ratio_x, hold_ratio_x)
        ok_y = qualifies_y and not churn_y
        ok_x = qualifies_x and not churn_x

        chosen_mode = ""
        chosen_by = "none"
        chosen_dd_impr = 0.0
        chosen_exp_delta = 0.0
        chosen_trade_ratio = float("nan")
        chosen_hold_ratio = float("nan")

        if ok_y and not ok_x:
            chosen_mode = "mode_Y"
            chosen_by = qualifies_by_y
            chosen_dd_impr = dd_impr_y
            chosen_exp_delta = exp_delta_y
            chosen_trade_ratio = trade_ratio_y
            chosen_hold_ratio = hold_ratio_y
        elif ok_x and not ok_y:
            chosen_mode = "mode_X"
            chosen_by = qualifies_by_x
            chosen_dd_impr = dd_impr_x
            chosen_exp_delta = exp_delta_x
            chosen_trade_ratio = trade_ratio_x
            chosen_hold_ratio = hold_ratio_x
        elif ok_y and ok_x:
            # Prefer mode with larger DD improvement; tie-break on expectancy delta.
            if dd_impr_y > dd_impr_x or (dd_impr_y == dd_impr_x and exp_delta_y >= exp_delta_x):
                chosen_mode = "mode_Y"
                chosen_by = qualifies_by_y
                chosen_dd_impr = dd_impr_y
                chosen_exp_delta = exp_delta_y
                chosen_trade_ratio = trade_ratio_y
                chosen_hold_ratio = hold_ratio_y
            else:
                chosen_mode = "mode_X"
                chosen_by = qualifies_by_x
                chosen_dd_impr = dd_impr_x
                chosen_exp_delta = exp_delta_x
                chosen_trade_ratio = trade_ratio_x
                chosen_hold_ratio = hold_ratio_x

        if chosen_mode:
            decision = "KEEP"
            if chosen_by == "both":
                decision_reason = "dd_and_expectancy_improved"
            elif chosen_by == "dd":
                decision_reason = "dd_improved"
            elif chosen_by == "expectancy":
                decision_reason = "expectancy_improved"
            else:
                decision_reason = "improves_vs_baseline"
        else:
            decision = "DISCARD"
            if (qualifies_y and churn_y) or (qualifies_x and churn_x):
                decision_reason = "churn"
            elif qualifies_y or qualifies_x:
                decision_reason = "churn_or_other_filter"
            else:
                decision_reason = "no_improvement"

        rows.append({
            "exit_c1_name": exit_c1_name,
            "decision": decision,
            "decision_reason": decision_reason,
            "baseline_dd": ref_dd,
            "baseline_expectancy": ref_exp,
            "baseline_trades": ref_trades,
            "baseline_hold": ref_hold,
            "mode_Y_dd": dd_y,
            "mode_Y_expectancy": exp_y,
            "mode_Y_trades": trades_y,
            "mode_Y_hold": hold_y,
            "mode_X_dd": dd_x,
            "mode_X_expectancy": exp_x,
            "mode_X_trades": trades_x,
            "mode_X_hold": hold_x,
            "chosen_mode": chosen_mode,
            "qualifies_by": chosen_by,
            "dd_improvement_rel": chosen_dd_impr,
            "expectancy_delta": chosen_exp_delta,
            "churn_trade_ratio": chosen_trade_ratio,
            "churn_hold_ratio": chosen_hold_ratio,
        })

    df = pd.DataFrame(rows)
    note = f"Baseline A run_id: {baseline_run_id or 'none'} (latest by mtime)."
    return df, note


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6.3: Final report Mode Y vs Mode X vs Baseline A; KEEP/DISCARD."
    )
    parser.add_argument("--phase6-root", default=str(PHASE6_ROOT), help="Phase 6 results root.")
    parser.add_argument("--c1-as-exit-final-root", default=str(C1_AS_EXIT_FINAL_ROOT), help="c1_as_exit_final root.")
    parser.add_argument("--output-csv", default=str(DECISIONS_CSV), help="Output CSV path.")
    parser.add_argument("--output-md", default=str(DECISIONS_MD), help="Output MD path.")
    args = parser.parse_args()
    phase6_root = Path(args.phase6_root).resolve()
    c1_as_exit_final_root = Path(args.c1_as_exit_final_root).resolve()
    df, note = build_final_decisions(phase6_root, c1_as_exit_final_root)
    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    md_lines = [note, "", "| exit_c1_name | decision | decision_reason |", "| --- | --- | --- |"]
    for _, row in df.iterrows():
        md_lines.append(f"| {row['exit_c1_name']} | {row['decision']} | {row['decision_reason']} |")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(note)


if __name__ == "__main__":
    main()

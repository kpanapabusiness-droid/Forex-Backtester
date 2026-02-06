# Phase 6.2 — C1-as-exit leaderboard. Baseline A = reference; per exit_c1_name metrics + churn flags.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE6_ROOT = ROOT / "results" / "phase6_exit"
C1_AS_EXIT_ROOT = PHASE6_ROOT / "c1_as_exit"
BASELINE_A_SLUG = "baseline_A_coral_disagree_exit"
CHURN_TRADE_RATIO_CEILING = 1.25
CHURN_HOLD_RATIO_FLOOR = 0.70
MIN_TRADES_TOTAL = 300
MIN_TRADES_PER_FOLD = 50


def _latest_run_dir_by_mtime(variant_dir: Path) -> Path | None:
    """Return the run_id dir with latest mtime (not by name)."""
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
            "trade_count_min_fold": None,
            "median_bars_held": float("nan"),
        }
    total_trades = sum(r["total_trades"] for r in rows)
    min_fold = min(r["total_trades"] for r in rows) if rows else None
    worst_dd = min((r["max_dd_pct"] for r in rows), default=0.0)
    worst_expectancy = min(r["expectancy"] for r in rows)
    median_holds = [r["median_hold_days"] for r in rows if pd.notna(r["median_hold_days"])]
    median_bars_held = float(pd.Series(median_holds).median()) if median_holds else float("nan")
    return {
        "worst_fold_max_drawdown_pct": worst_dd,
        "worst_fold_expectancy_r": worst_expectancy,
        "trade_count_total": total_trades,
        "trade_count_min_fold": min_fold,
        "median_bars_held": median_bars_held,
    }


def _compute_decision(
    total_trades: int,
    min_fold: int | None,
    churn_trade: bool,
    churn_hold: bool,
) -> tuple[str, str]:
    """Return (decision, decision_reason). Uses MIN_TRADES_* and CHURN_* constants."""
    if total_trades < MIN_TRADES_TOTAL:
        return "REJECT", "insufficient_trades_total"
    if min_fold is not None and min_fold < MIN_TRADES_PER_FOLD:
        return "REJECT", "insufficient_trades_fold"
    if churn_trade or churn_hold:
        return "REJECT", "churn"
    return "PASS", ""


def _load_run_status(variant_dir: Path) -> dict | None:
    path = variant_dir / "run_status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_leaderboard(phase6_root: Path, c1_as_exit_root: Path) -> tuple[pd.DataFrame, str]:
    """Build leaderboard DataFrame and baseline_run_id note. Uses baseline A latest by mtime."""
    phase6_root = Path(phase6_root).resolve()
    c1_as_exit_root = Path(c1_as_exit_root).resolve()
    baseline_dir = phase6_root / BASELINE_A_SLUG
    baseline_run_dir = _latest_run_dir_by_mtime(baseline_dir) if baseline_dir.exists() else None
    baseline_run_id = baseline_run_dir.name if baseline_run_dir else None
    ref_trades = 0
    ref_hold = float("nan")
    if baseline_run_dir:
        ref_agg = _aggregate_run(baseline_run_dir)
        ref_trades = ref_agg["trade_count_total"]
        ref_hold = ref_agg["median_bars_held"]

    def _blank_row(name: str, status: str, reason: str) -> dict:
        return {
            "exit_c1_name": name,
            "status": status,
            "reason": reason,
            "worst_fold_max_drawdown_pct": float("nan"),
            "worst_fold_expectancy_r": float("nan"),
            "trade_count_total": 0,
            "trade_count_min_fold": "",
            "insufficient_trades_flag": False,
            "decision": status if status in ("REJECT", "ERROR") else "REJECT",
            "decision_reason": reason,
            "median_bars_held": float("nan"),
            "trade_count_ratio_vs_A": float("nan"),
            "median_hold_ratio_vs_A": float("nan"),
            "churn_trade_ratio_flag": False,
            "churn_hold_ratio_flag": False,
        }

    rows: list[dict] = []
    if not c1_as_exit_root.exists():
        df = pd.DataFrame(columns=list(_blank_row("", "", "").keys()))
        return df, f"Baseline A run_id: {baseline_run_id or 'not found'}."

    for exit_c1_dir in sorted(c1_as_exit_root.iterdir()):
        if not exit_c1_dir.is_dir() or exit_c1_dir.name.startswith("_"):
            continue
        exit_c1_name = exit_c1_dir.name
        status_obj = _load_run_status(exit_c1_dir)
        if status_obj and status_obj.get("status") in ("REJECT", "ERROR"):
            st = status_obj.get("status", "")
            reason = status_obj.get("reason", "")
            row = _blank_row(exit_c1_name, st, reason)
            row["decision"] = st
            row["decision_reason"] = reason
            rows.append(row)
            continue

        run_dir = _latest_run_dir_by_mtime(exit_c1_dir)
        if run_dir is None:
            row = _blank_row(exit_c1_name, "NO_RUN", "no run_id")
            row["decision"] = "REJECT"
            row["decision_reason"] = "no run_id"
            rows.append(row)
            continue

        agg = _aggregate_run(run_dir)
        total_trades = agg["trade_count_total"]
        min_fold = agg.get("trade_count_min_fold")
        insufficient = total_trades < MIN_TRADES_TOTAL or (
            min_fold is not None and min_fold < MIN_TRADES_PER_FOLD
        )
        trade_ratio = (total_trades / ref_trades) if ref_trades else float("nan")
        hold_ratio = (
            (agg["median_bars_held"] / ref_hold)
            if ref_hold and pd.notna(ref_hold) else float("nan")
        )
        churn_trade = pd.notna(trade_ratio) and trade_ratio > CHURN_TRADE_RATIO_CEILING
        churn_hold = pd.notna(hold_ratio) and hold_ratio < CHURN_HOLD_RATIO_FLOOR

        decision, decision_reason = _compute_decision(
            total_trades, min_fold, churn_trade, churn_hold
        )

        rows.append(
            {
                "exit_c1_name": exit_c1_name,
                "status": "OK",
                "reason": "",
                "worst_fold_max_drawdown_pct": agg["worst_fold_max_drawdown_pct"],
                "worst_fold_expectancy_r": agg["worst_fold_expectancy_r"],
                "trade_count_total": total_trades,
                "trade_count_min_fold": min_fold if min_fold is not None else "",
                "insufficient_trades_flag": insufficient,
                "decision": decision,
                "decision_reason": decision_reason,
                "median_bars_held": agg["median_bars_held"],
                "trade_count_ratio_vs_A": trade_ratio,
                "median_hold_ratio_vs_A": hold_ratio,
                "churn_trade_ratio_flag": churn_trade,
                "churn_hold_ratio_flag": churn_hold,
            }
        )

    df = pd.DataFrame(rows)
    note = f"Baseline A run_id used: {baseline_run_id or 'none'} (latest by mtime under {BASELINE_A_SLUG}/)."
    return df, note


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6.2: Build C1-as-exit leaderboard from WFO outputs."
    )
    parser.add_argument(
        "--phase6-root",
        default=str(PHASE6_ROOT),
        help="Phase 6 results root (e.g. results/phase6_exit).",
    )
    parser.add_argument(
        "--c1-as-exit-root",
        default=str(C1_AS_EXIT_ROOT),
        help="C1-as-exit results root (e.g. results/phase6_exit/c1_as_exit).",
    )
    args = parser.parse_args()
    phase6_root = Path(args.phase6_root).resolve()
    c1_as_exit_root = Path(args.c1_as_exit_root).resolve()
    df, baseline_note = build_leaderboard(phase6_root, c1_as_exit_root)
    c1_as_exit_root.mkdir(parents=True, exist_ok=True)
    csv_path = c1_as_exit_root / "leaderboard_exit_c1.csv"
    notes_path = c1_as_exit_root / "leaderboard_exit_c1_notes.md"
    df.to_csv(csv_path, index=False)
    notes_lines = [
        baseline_note,
        "",
        f"Sample-size thresholds: trade_count_total >= {MIN_TRADES_TOTAL}; "
        f"min trades per fold >= {MIN_TRADES_PER_FOLD}.",
        f"Churn limits: trade_count_ratio_vs_A > {CHURN_TRADE_RATIO_CEILING} "
        f"or median_hold_ratio_vs_A < {CHURN_HOLD_RATIO_FLOOR} -> decision=REJECT reason=churn.",
    ]
    notes_path.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {notes_path}")
    print(baseline_note)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_ROOT_DEFAULT = ROOT / "results" / "phase7"
WFO_ROOT_NAME = "wfo"
DISCOVERY_NAME = "phase7_discovery.json"


def _load_discovery(results_root: Path) -> Dict[str, Any]:
    """Load discovery manifest written by phase7_run_volume_veto_wfo."""
    path = results_root / DISCOVERY_NAME
    if not path.exists():
        return {"all": [], "runnable": [], "skipped": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run_dir(variant_dir: Path) -> Optional[Path]:
    """Return latest WFO v2 run dir under variant_dir, or None."""
    if not variant_dir.exists():
        return None
    run_dirs = sorted(
        (
            p
            for p in variant_dir.iterdir()
            if p.is_dir() and (p / "wfo_run_meta.json").exists()
        ),
        key=lambda p: p.name,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _first_metric(metrics: Dict[str, Any], candidates: List[str], default: float = 0.0) -> float:
    """Extract first present metric value (case-insensitive key match)."""
    lowered = {k.lower(): v for k, v in metrics.items()}
    for key in candidates:
        v = lowered.get(key.lower())
        if v is not None:
            try:
                return float(v)
            except Exception:
                continue
    return float(default)


TRADES_KEYS = ["total_trades", "trades", "n_trades", "trade_count"]
DD_KEYS = ["max_dd_pct", "max_drawdown_pct", "max_drawdown", "dd_pct"]
EXPECTANCY_KEYS = ["expectancy_r", "expectancy_R", "expectancy", "exp_r"]


def _fold_rows_for_run(run_dir: Path) -> List[Dict[str, Any]]:
    """
    Parse per-fold WFO v2 metrics for a single run.

    We read fold_XX/out_of_sample using batch_sweeper.parse_summary_or_trades so that
    we stay aligned with the rest of the project.
    """
    from scripts.batch_sweeper import parse_summary_or_trades

    fold_dirs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")),
        key=lambda p: p.name,
    )
    rows: List[Dict[str, Any]] = []
    for fold_dir in fold_dirs:
        oos_dir = fold_dir / "out_of_sample"
        if not oos_dir.exists():
            continue
        try:
            metrics = parse_summary_or_trades(oos_dir)
        except Exception:
            continue
        try:
            idx_str = fold_dir.name.split("_", 1)[1]
            fold_id = int(idx_str)
        except Exception:
            fold_id = 0

        trades = int(_first_metric(metrics, TRADES_KEYS, default=0.0))
        dd = _first_metric(metrics, DD_KEYS, default=0.0)
        exp = _first_metric(metrics, EXPECTANCY_KEYS, default=0.0)
        rows.append(
            {
                "fold_id": fold_id,
                "trades": trades,
                "max_dd": dd,
                "expectancy": exp,
            }
        )
    rows.sort(key=lambda r: r["fold_id"])
    return rows


def _aggregate_run(run_dir: Path) -> Dict[str, Any]:
    """Aggregate per-fold metrics for baseline or a candidate volume run."""
    folds = _fold_rows_for_run(run_dir)
    if not folds:
        return {
            "folds": [],
            "total_trades": 0,
            "worst_dd": 0.0,
            "worst_expectancy": 0.0,
        }
    total_trades = int(sum(f["trades"] for f in folds))
    worst_dd = float(min((f["max_dd"] for f in folds), default=0.0))
    worst_expectancy = float(min((f["expectancy"] for f in folds), default=0.0))
    return {
        "folds": folds,
        "total_trades": total_trades,
        "worst_dd": worst_dd,
        "worst_expectancy": worst_expectancy,
    }


def _dd_improves_by_at_least_5pct(baseline_dd: float, candidate_dd: float) -> bool:
    """
    Return True if candidate worst DD improves by >=5% vs baseline.

    Works for both positive and negative DD conventions by using absolute values.
    """
    base_abs = abs(float(baseline_dd))
    cand_abs = abs(float(candidate_dd))
    if base_abs <= 0.0:
        return False
    return cand_abs <= base_abs * 0.95


def _fold_collapse(
    candidate_folds: List[Dict[str, Any]],
    baseline_folds: Dict[int, int],
) -> bool:
    """
    Collapse reject:
      any fold OOS trades < max(10, 10% of baseline fold trades) => True (collapse).
    """
    for row in candidate_folds:
        fid = int(row.get("fold_id", 0))
        trades_on = int(row.get("trades", 0))
        trades_off = int(baseline_folds.get(fid, 0))
        threshold = max(10, int(round(trades_off * 0.10)))
        if trades_on < threshold:
            return True
    return False


def _decision_for_candidate(
    volume_name: str,
    base_agg: Dict[str, Any],
    cand_agg: Optional[Dict[str, Any]],
    skip_reason: Optional[str],
) -> Dict[str, Any]:
    """Apply Phase 7 decision logic for a single volume candidate."""
    base_trades = int(base_agg.get("total_trades", 0))
    base_dd = float(base_agg.get("worst_dd", 0.0))
    base_exp = float(base_agg.get("worst_expectancy", 0.0))

    # Default candidate metrics
    trades_on = 0
    worst_dd_on = 0.0
    worst_exp_on = 0.0
    collapse_flag = False
    decision = "SKIP"
    reason = skip_reason or "stub/non-functional"

    if cand_agg is None:
        # No WFO folder at all for this indicator
        final_reason = skip_reason or "no_wfo_run"
        return {
            "volume_name": volume_name,
            "decision": "SKIP",
            "reason": final_reason,
            "trades_off": base_trades,
            "trades_on": 0,
            "worst_dd_off": base_dd,
            "worst_dd_on": 0.0,
            "worst_expectancy_off": base_exp,
            "worst_expectancy_on": 0.0,
            "collapse_flag": False,
            "trade_ratio": 0.0,
            "expectancy_delta": -base_exp,
        }

    trades_on = int(cand_agg.get("total_trades", 0))
    worst_dd_on = float(cand_agg.get("worst_dd", 0.0))
    worst_exp_on = float(cand_agg.get("worst_expectancy", 0.0))

    # Invariant: trades_on <= trades_off else DISCARD
    if trades_on > base_trades:
        decision = "DISCARD"
        reason = "trade_count_increases_vs_baseline"
    else:
        # Collapse reject per-fold
        base_folds_map = {int(f["fold_id"]): int(f["trades"]) for f in base_agg["folds"]}
        collapse_flag = _fold_collapse(
            candidate_folds=cand_agg["folds"],
            baseline_folds=base_folds_map,
        )
        if collapse_flag:
            decision = "DISCARD"
            reason = "fold_collapse_trade_count"
        else:
            dd_improved = _dd_improves_by_at_least_5pct(base_dd, worst_dd_on)
            exp_ok = worst_exp_on >= (base_exp - 0.02)
            if dd_improved and exp_ok:
                decision = "KEEP"
                reason = "worst_fold_dd_improves_and_expectancy_ok"
            else:
                decision = "DISCARD"
                if not dd_improved and not exp_ok:
                    reason = "dd_worse_and_expectancy_degrades"
                elif not dd_improved:
                    reason = "dd_not_improved_by_5pct"
                else:
                    reason = "expectancy_degrades_over_0.02R"

    trade_ratio = float(trades_on / base_trades) if base_trades else 0.0
    expectancy_delta = float(worst_exp_on - base_exp)

    return {
        "volume_name": volume_name,
        "decision": decision,
        "reason": reason,
        "trades_off": base_trades,
        "trades_on": trades_on,
        "worst_dd_off": base_dd,
        "worst_dd_on": worst_dd_on,
        "worst_expectancy_off": base_exp,
        "worst_expectancy_on": worst_exp_on,
        "collapse_flag": collapse_flag,
        "trade_ratio": trade_ratio,
        "expectancy_delta": expectancy_delta,
    }


def build_leaderboard(results_root: Path) -> pd.DataFrame:
    """
    Build Phase 7 volume leaderboard dataframe.

    Columns:
      - volume_name
      - decision (KEEP / DISCARD / SKIP / BASELINE)
      - reason
      - trades_off, trades_on
      - worst_dd_off, worst_dd_on
      - worst_expectancy_off, worst_expectancy_on
      - collapse_flag
      - trade_ratio
      - expectancy_delta
    """
    results_root = results_root.resolve()
    wfo_root = results_root / WFO_ROOT_NAME
    if not wfo_root.exists():
        raise FileNotFoundError(f"WFO root not found: {wfo_root}")

    # Baseline aggregation
    baseline_dir = wfo_root / "baseline_off"
    baseline_run = _latest_run_dir(baseline_dir)
    if baseline_run is None:
        raise FileNotFoundError(
            f"No WFO run found under {baseline_dir}. Run phase7_run_volume_veto_wfo first."
        )
    base_agg = _aggregate_run(baseline_run)

    discovery = _load_discovery(results_root)
    all_entries = discovery.get("all") or []
    skipped_entries = {e["name"]: e.get("reason") for e in discovery.get("skipped") or []}
    all_names = sorted({str(e.get("name")) for e in all_entries if e.get("name")})

    rows: List[Dict[str, Any]] = []

    # Baseline row for reference
    rows.append(
        {
            "volume_name": "baseline_off",
            "decision": "BASELINE",
            "reason": "baseline_volume_off",
            "trades_off": int(base_agg.get("total_trades", 0)),
            "trades_on": int(base_agg.get("total_trades", 0)),
            "worst_dd_off": float(base_agg.get("worst_dd", 0.0)),
            "worst_dd_on": float(base_agg.get("worst_dd", 0.0)),
            "worst_expectancy_off": float(base_agg.get("worst_expectancy", 0.0)),
            "worst_expectancy_on": float(base_agg.get("worst_expectancy", 0.0)),
            "collapse_flag": False,
            "trade_ratio": 1.0,
            "expectancy_delta": 0.0,
        }
    )

    for name in all_names:
        skip_reason = skipped_entries.get(name)

        variant_dir = wfo_root / name
        run_dir = _latest_run_dir(variant_dir)
        cand_agg = _aggregate_run(run_dir) if run_dir is not None else None

        row = _decision_for_candidate(
            volume_name=name,
            base_agg=base_agg,
            cand_agg=cand_agg,
            skip_reason=skip_reason,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["decision", "volume_name"],
        key=lambda col: col.map(
            {
                "BASELINE": 0,
                "KEEP": 1,
                "DISCARD": 2,
                "SKIP": 3,
            }
        )
        if col.name == "decision"
        else col,
    ).reset_index(drop=True)
    return df


def write_decisions_markdown(results_root: Path, df: pd.DataFrame) -> None:
    """Write human-readable decisions markdown file."""
    md_path = results_root / "volume_decisions.md"
    lines: List[str] = []
    lines.append("# Phase 7 — Volume veto decisions")
    lines.append("")

    baseline_row = df.loc[df["decision"] == "BASELINE"].iloc[0]
    lines.append("## Baseline (volume OFF)")
    lines.append("")
    lines.append(
        f"- baseline_off: trades={baseline_row['trades_off']}, "
        f"worst_dd={baseline_row['worst_dd_off']:.2f}, "
        f"worst_expectancy={baseline_row['worst_expectancy_off']:.4f}"
    )
    lines.append("")

    lines.append("## Volume candidates")
    lines.append("")
    for _, row in df.iterrows():
        if row["volume_name"] == "baseline_off":
            continue
        lines.append(
            f"- {row['volume_name']}: decision={row['decision']}, reason={row['reason']}; "
            f"trades_on={row['trades_on']} vs trades_off={row['trades_off']}, "
            f"worst_dd_on={row['worst_dd_on']:.2f} vs {row['worst_dd_off']:.2f}, "
            f"worst_exp_on={row['worst_expectancy_on']:.4f} vs {row['worst_expectancy_off']:.4f}"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7 — Build volume veto leaderboard and decision report from WFO outputs."
    )
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT_DEFAULT),
        help="Phase 7 results root (default: results/phase7).",
    )
    args = parser.parse_args(argv)
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    df = build_leaderboard(results_root)
    leaderboard_path = results_root / "volume_leaderboard.csv"
    df.to_csv(leaderboard_path, index=False)
    write_decisions_markdown(results_root, df)


if __name__ == "__main__":
    main()


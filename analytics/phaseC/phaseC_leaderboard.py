# Phase C — C1 identity leaderboard: aggregate WFO v2 fold outputs, apply PASS/REJECT gates.
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

WFO_RUNS_NAME = "wfo_runs"
AGGREGATE_DIR = "aggregate"
LEADERBOARD_CSV = "leaderboard_c1_identity.csv"
SURVIVORS_MD = "phaseC_survivors.md"
OVERLAP_CSV = "overlap_matrix.csv"

MIN_TRADES_STARVATION = 300
REGIME_COLLAPSE_RATIO = 0.20
MAX_DD_CATASTROPHIC = 0.25
WORST_ROI_THRESHOLD = -0.05
MAX_SCRATCH_RATE = 0.65
MAX_DD_VS_MEDIAN_MULT = 1.5
MIN_SURVIVORS = 5


def _latest_run_dir(c1_dir: Path) -> Optional[Path]:
    """Return latest WFO v2 run dir under c1_dir (has wfo_run_meta.json), or None."""
    if not c1_dir.exists():
        return None
    run_dirs = sorted(
        (
            p
            for p in c1_dir.iterdir()
            if p.is_dir() and (p / "wfo_run_meta.json").exists()
        ),
        key=lambda p: p.name,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _fold_rows_for_run(run_dir: Path) -> List[Dict[str, Any]]:
    """Parse each fold's out_of_sample with parse_summary_or_trades; return list of fold dicts."""
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
            fold_id = int(fold_dir.name.split("_", 1)[1])
        except Exception:
            fold_id = 0
        try:
            metrics = parse_summary_or_trades(oos_dir)
        except Exception:
            continue
        trades = int(metrics.get("total_trades") or 0)
        scratches = int(metrics.get("scratches") or 0)
        scratch_rate = (scratches / trades) if trades > 0 else 0.0
        roi_pct = float(metrics.get("roi_pct") or 0.0)
        max_dd_pct = float(metrics.get("max_dd_pct") or 0.0)
        rows.append({
            "fold_id": fold_id,
            "trades": trades,
            "scratches": scratches,
            "scratch_rate": scratch_rate,
            "roi_pct": roi_pct,
            "max_dd_pct": max_dd_pct,
        })
    rows.sort(key=lambda r: r["fold_id"])
    return rows


def _worst_fold_by_roi(fold_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Worst fold = minimum roi_pct (most negative)."""
    if not fold_rows:
        return {}
    return min(fold_rows, key=lambda r: r["roi_pct"])


def _apply_gates(
    c1_name: str,
    fold_rows: List[Dict[str, Any]],
    worst: Dict[str, Any],
    median_roi: float,
    median_max_dd: float,
    median_trades: float,
) -> tuple[str, str]:
    """
    Apply Phase C PASS/REJECT rules. Return (pass_reject, rejection_reason).
    rejection_reason blank if PASS.
    """
    if not fold_rows or not worst:
        return "REJECT", "no_fold_data"

    worst_trades = int(worst.get("trades") or 0)
    worst_roi = float(worst.get("roi_pct") or 0.0)
    worst_max_dd = float(worst.get("max_dd_pct") or 0.0)
    worst_scratch_rate = float(worst.get("scratch_rate") or 0.0)
    if worst_max_dd > 0:
        worst_max_dd = -abs(worst_max_dd)

    if worst_trades == 0:
        return "REJECT", "zero_trade_collapse"
    if worst_trades < MIN_TRADES_STARVATION:
        return "REJECT", "trade_starvation"
    if median_trades > 0 and worst_trades < REGIME_COLLAPSE_RATIO * median_trades:
        return "REJECT", "regime_collapse"
    if worst_max_dd <= -100 * MAX_DD_CATASTROPHIC:
        return "REJECT", "catastrophic_dd"
    if worst_roi <= 100 * WORST_ROI_THRESHOLD:
        return "REJECT", "worst_fold_roi_below_threshold"
    if worst_scratch_rate > MAX_SCRATCH_RATE:
        return "REJECT", "worst_fold_scratch_rate_high"
    if median_max_dd != 0 and worst_max_dd < MAX_DD_VS_MEDIAN_MULT * median_max_dd:
        return "REJECT", "worst_fold_dd_vs_median"

    return "PASS", ""


def _safe_median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(pd.Series(values).median())


def build_leaderboard(wfo_runs_root: Path) -> pd.DataFrame:
    """
    Build Phase C leaderboard from results/phaseC/wfo_runs/<c1_name>/<run_id>/fold_XX/out_of_sample.
    One row per C1 with pass_reject, worst_fold_*, median_fold_*, rejection_reason.
    """
    wfo_runs_root = wfo_runs_root.resolve()
    if not wfo_runs_root.exists():
        return pd.DataFrame(columns=[
            "c1_name", "pass_reject", "worst_fold_id", "worst_fold_roi", "worst_fold_max_dd",
            "worst_fold_trades", "worst_fold_scratch_rate", "median_fold_roi", "median_fold_max_dd",
            "median_fold_trades", "rejection_reason",
        ])

    rows: List[Dict[str, Any]] = []
    for c1_dir in sorted(wfo_runs_root.iterdir()):
        if not c1_dir.is_dir():
            continue
        c1_name = c1_dir.name
        run_dir = _latest_run_dir(c1_dir)
        if run_dir is None:
            rows.append({
                "c1_name": c1_name,
                "pass_reject": "REJECT",
                "worst_fold_id": 0,
                "worst_fold_roi": 0.0,
                "worst_fold_max_dd": 0.0,
                "worst_fold_trades": 0,
                "worst_fold_scratch_rate": 0.0,
                "median_fold_roi": 0.0,
                "median_fold_max_dd": 0.0,
                "median_fold_trades": 0,
                "rejection_reason": "no_wfo_run",
            })
            continue

        fold_rows = _fold_rows_for_run(run_dir)
        if not fold_rows:
            rows.append({
                "c1_name": c1_name,
                "pass_reject": "REJECT",
                "worst_fold_id": 0,
                "worst_fold_roi": 0.0,
                "worst_fold_max_dd": 0.0,
                "worst_fold_trades": 0,
                "worst_fold_scratch_rate": 0.0,
                "median_fold_roi": 0.0,
                "median_fold_max_dd": 0.0,
                "median_fold_trades": 0,
                "rejection_reason": "no_fold_data",
            })
            continue

        worst = _worst_fold_by_roi(fold_rows)
        median_roi = _safe_median([r["roi_pct"] for r in fold_rows])
        median_max_dd = _safe_median([r["max_dd_pct"] for r in fold_rows])
        median_trades = _safe_median([r["trades"] for r in fold_rows])

        pass_reject, rejection_reason = _apply_gates(
            c1_name, fold_rows, worst,
            median_roi, median_max_dd, median_trades,
        )

        worst_roi = float(worst.get("roi_pct") or 0.0)
        worst_max_dd = float(worst.get("max_dd_pct") or 0.0)
        rows.append({
            "c1_name": c1_name,
            "pass_reject": pass_reject,
            "worst_fold_id": int(worst.get("fold_id") or 0),
            "worst_fold_roi": worst_roi,
            "worst_fold_max_dd": worst_max_dd,
            "worst_fold_trades": int(worst.get("trades") or 0),
            "worst_fold_scratch_rate": float(worst.get("scratch_rate") or 0.0),
            "median_fold_roi": median_roi,
            "median_fold_max_dd": median_max_dd,
            "median_fold_trades": median_trades,
            "rejection_reason": rejection_reason,
        })

    return pd.DataFrame(rows)


def _write_survivors_md(
    leaderboard_df: pd.DataFrame,
    out_path: Path,
    results_root: Path,
) -> None:
    """Write phaseC_survivors.md: PASS list with bullets, REJECT list with one-line reason."""
    survivors = leaderboard_df[leaderboard_df["pass_reject"] == "PASS"]
    rejects = leaderboard_df[leaderboard_df["pass_reject"] == "REJECT"]

    lines: List[str] = [
        "# Phase C — C1 Identity Survivors",
        "",
        "## PASS",
        "",
    ]
    for _, row in survivors.iterrows():
        name = row["c1_name"]
        lines.append(f"### {name}")
        lines.append("- Worst-fold ROI and DD within gates; sufficient OOS trades.")
        lines.append("- No zero-trade or regime collapse; scratch rate acceptable.")
        lines.append("")
    if survivors.empty:
        lines.append("(none)")
        lines.append("")

    lines.append("## REJECT")
    lines.append("")
    for _, row in rejects.iterrows():
        reason = row.get("rejection_reason") or "unknown"
        lines.append(f"- **{row['c1_name']}**: {reason}")
    lines.append("")

    n_pass = len(survivors)
    if n_pass < MIN_SURVIVORS:
        lines.append("---")
        lines.append(
            f"**Phase C failed: {n_pass} survivors < {MIN_SURVIVORS}. Return to Phase B.1 Round 2.**"
        )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _try_overlap_matrix(wfo_runs_root: Path, out_path: Path) -> bool:
    """
    If trade logs exist with pair+entry_time+direction, compute approximate overlap matrix.
    Return True if written, False if skipped (missing keys or no data).
    """
    wfo_runs_root = wfo_runs_root.resolve()
    c1_dirs = [p for p in wfo_runs_root.iterdir() if p.is_dir()]
    if len(c1_dirs) < 2:
        return False

    run_dirs = []
    for c1_dir in c1_dirs:
        run_dir = _latest_run_dir(c1_dir)
        if run_dir is None:
            continue
        run_dirs.append((c1_dir.name, run_dir))

    def _load_oos_trade_keys(run_dir: Path) -> set:
        keys = set()
        for fold_dir in run_dir.iterdir():
            if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                continue
            trades_path = fold_dir / "out_of_sample" / "trades.csv"
            if not trades_path.exists():
                continue
            try:
                df = pd.read_csv(trades_path)
            except Exception:
                continue
            for col in ("pair", "entry_time", "direction"):
                if col not in df.columns:
                    return set()
            for _, r in df.iterrows():
                keys.add((str(r.get("pair", "")), str(r.get("entry_time", "")), str(r.get("direction", ""))))
        return keys

    sets = {}
    for c1_name, run_dir in run_dirs:
        sets[c1_name] = _load_oos_trade_keys(run_dir)
        if not sets[c1_name] and run_dir.exists():
            for fold_dir in run_dir.iterdir():
                if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
                    oos = fold_dir / "out_of_sample" / "trades.csv"
                    if oos.exists():
                        return False
            break
    if not sets:
        return False

    names = sorted(sets.keys())
    matrix = pd.DataFrame(0.0, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                matrix.iloc[i, j] = 1.0
            else:
                inter = len(sets[a] & sets[b])
                un = len(sets[a] | sets[b])
                matrix.iloc[i, j] = (inter / un) if un else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_path)
    return True


def run_phaseC_leaderboard(results_root: Path, write_overlap: bool = True) -> None:
    """
    Read wfo_runs under results_root, build leaderboard, write CSV + survivors MD.
    Optionally write overlap_matrix.csv if trade keys available.
    """
    results_root = results_root.resolve()
    wfo_runs_root = results_root / WFO_RUNS_NAME
    aggregate_dir = results_root / AGGREGATE_DIR
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    df = build_leaderboard(wfo_runs_root)
    csv_path = aggregate_dir / LEADERBOARD_CSV
    df.to_csv(csv_path, index=False)

    md_path = aggregate_dir / SURVIVORS_MD
    _write_survivors_md(df, md_path, results_root)

    overlap_path = aggregate_dir / OVERLAP_CSV
    if write_overlap:
        wrote = _try_overlap_matrix(wfo_runs_root, overlap_path)
        if not wrote and md_path.exists():
            content = md_path.read_text(encoding="utf-8")
            if "overlap" not in content.lower():
                md_path.write_text(
                    content.rstrip() + "\n\n(overlap_matrix.csv omitted: missing standard trade keys)\n",
                    encoding="utf-8",
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase C — Build leaderboard and survivors from WFO v2 outputs."
    )
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results" / "phaseC"),
        help="Results root (e.g. results/phaseC)",
    )
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Do not write overlap_matrix.csv",
    )
    args = parser.parse_args()
    run_phaseC_leaderboard(Path(args.results_root), write_overlap=not args.no_overlap)


if __name__ == "__main__":
    main()

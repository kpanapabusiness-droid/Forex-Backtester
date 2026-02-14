"""
Phase E-1: Coexistence geometry analysis between two signal sources (e.g. CEB vs LSR).
Computes overlap ratio, conflict rate, conditional lift, and event-level relations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_signal_events(path: Path) -> pd.DataFrame:
    """Load signal_events.csv; return (pair, date, signal) with date normalized."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).clip(-1, 1).astype(int)
    return df[["pair", "date", "signal"]]


def _extract_p3r_from_run(run_dir: Path) -> tuple[float | None, float | None]:
    """
    Read P_3R_before_2R from leaderboard_geometry_lock.csv or pooled_signal_lift_stability.
    Returns (P_3R_disc, P_3R_val) or (None, None) if not found.
    """
    lb_path = run_dir / "leaderboard_geometry_lock.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        if not lb.empty:
            row = lb.iloc[0]
            disc = row.get("P_3R_before_2R_disc")
            val = row.get("P_3R_before_2R_val")
            return (float(disc), float(val)) if pd.notna(disc) and pd.notna(val) else (None, None)

    stab_path = run_dir / "pooled_signal_lift_stability.csv"
    if stab_path.exists():
        stab = pd.read_csv(stab_path)
        sub = stab[(stab["x"] == 2) & (stab["y"] == 3.0)]
        if not sub.empty:
            row = sub.iloc[0]
            disc = row.get("signal_rate_discovery")
            val = row.get("signal_rate_validation")
            return (float(disc), float(val)) if pd.notna(disc) and pd.notna(val) else (None, None)
    return (None, None)


def run(
    ceb_dir: Path,
    lsr_dir: Path,
    outdir: Path,
) -> None:
    ceb_events_path = ceb_dir / "signal_events.csv"
    lsr_events_path = lsr_dir / "signal_events.csv"
    if not ceb_events_path.exists():
        raise FileNotFoundError(f"CEB signal_events.csv not found: {ceb_events_path}")
    if not lsr_events_path.exists():
        raise FileNotFoundError(f"LSR signal_events.csv not found: {lsr_events_path}")

    ceb_df = _load_signal_events(ceb_events_path)
    lsr_df = _load_signal_events(lsr_events_path)

    ceb_df = ceb_df.rename(columns={"signal": "ceb_signal"})
    lsr_df = lsr_df.rename(columns={"signal": "lsr_signal"})

    merged = ceb_df.merge(
        lsr_df, on=["pair", "date"], how="outer", suffixes=("_ceb", "_lsr")
    )
    merged["ceb_signal"] = merged["ceb_signal"].fillna(0).astype(int)
    merged["lsr_signal"] = merged["lsr_signal"].fillna(0).astype(int)

    ceb_only = (merged["ceb_signal"] != 0) & (merged["lsr_signal"] == 0)
    lsr_only = (merged["ceb_signal"] == 0) & (merged["lsr_signal"] != 0)
    both_any_mask = (merged["ceb_signal"] != 0) & (merged["lsr_signal"] != 0)
    both_same_dir = both_any_mask & (merged["ceb_signal"] == merged["lsr_signal"])
    both_conflict = both_any_mask & (merged["ceb_signal"] != merged["lsr_signal"])

    n_ceb = int((merged["ceb_signal"] != 0).sum())
    n_lsr = int((merged["lsr_signal"] != 0).sum())
    n_both_any = int(both_any_mask.sum())
    n_both_same_dir = int(both_same_dir.sum())
    n_both_conflict = int(both_conflict.sum())

    overlap_ratio_vs_ceb = n_both_any / n_ceb if n_ceb else 0.0
    overlap_ratio_vs_lsr = n_both_any / n_lsr if n_lsr else 0.0
    conflict_rate_vs_overlap = n_both_conflict / n_both_any if n_both_any else 0.0

    merged["relation"] = "none"
    merged.loc[ceb_only, "relation"] = "ceb_only"
    merged.loc[lsr_only, "relation"] = "lsr_only"
    merged.loc[both_same_dir, "relation"] = "both_same_dir"
    merged.loc[both_conflict, "relation"] = "both_conflict"

    events_out = merged[merged["relation"] != "none"].copy()
    events_out = events_out[["pair", "date", "ceb_signal", "lsr_signal", "relation"]]
    events_out = events_out.sort_values(["pair", "date"])

    outdir.mkdir(parents=True, exist_ok=True)
    events_out.to_csv(outdir / "coexistence_events.csv", index=False)

    ceb_p3r_disc, ceb_p3r_val = _extract_p3r_from_run(ceb_dir)
    lsr_p3r_disc, lsr_p3r_val = _extract_p3r_from_run(lsr_dir)

    lines = [
        "=== Phase E-1 Coexistence Geometry ===",
        "",
        "A) COUNTS",
        f"  n_ceb: {n_ceb}",
        f"  n_lsr: {n_lsr}",
        f"  n_both_any (intersection): {n_both_any}",
        f"  n_both_same_dir: {n_both_same_dir}",
        f"  n_both_conflict: {n_both_conflict}",
        "",
        "B) RATES",
        f"  overlap_ratio_vs_ceb: {overlap_ratio_vs_ceb:.4f}",
        f"  overlap_ratio_vs_lsr: {overlap_ratio_vs_lsr:.4f}",
        f"  conflict_rate_vs_overlap: {conflict_rate_vs_overlap:.4f}",
        "",
        "C) CONDITIONAL LIFT (from D-6G pooled outputs)",
        f"  CEB P_3R_before_2R_disc: {ceb_p3r_disc}",
        f"  CEB P_3R_before_2R_val: {ceb_p3r_val}",
        f"  LSR P_3R_before_2R_disc: {lsr_p3r_disc}",
        f"  LSR P_3R_before_2R_val: {lsr_p3r_val}",
    ]

    lines.append("")
    lines.append("D) PER-EVENT CONDITIONAL P3R")
    lines.append(
        "  Per-event outcome file not found; conditional P3R for "
        "ceb_only/lsr_only/both_same_dir not computed."
    )

    summary_text = "\n".join(lines)
    (outdir / "coexistence_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"\nWrote: {outdir / 'coexistence_summary.txt'}")
    print(f"Wrote: {outdir / 'coexistence_events.csv'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase E-1 — Coexistence geometry (CEB vs LSR overlap/conflict).",
    )
    parser.add_argument(
        "--ceb-dir",
        required=True,
        help="CEB run dir (must contain signal_events.csv)",
    )
    parser.add_argument(
        "--lsr-dir",
        required=True,
        help="LSR run dir (must contain signal_events.csv)",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for coexistence_summary.txt and coexistence_events.csv",
    )
    args = parser.parse_args(argv)

    ceb_dir = Path(args.ceb_dir).resolve()
    lsr_dir = Path(args.lsr_dir).resolve()
    outdir = Path(args.outdir).resolve()
    run(ceb_dir, lsr_dir, outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

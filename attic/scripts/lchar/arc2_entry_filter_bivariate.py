"""Arc 2 bivariate entry-filter exploration (Phase L6 Arc 2 Phase 3).

Descriptive characterisation per L6.0 v1.1 §14.5. Cross-tabulates
`concurrent_signals_same_bar` (quintile-bucketed) against
`dist_d1_kijun_atr` (quintile-bucketed) on the 3,993 taken trades from the
v1.1 / v1.2.1 / extended/entry_filter_univariate artefacts.

NO filter selection. NO WFO. NO signal-module modification. NO third axis.

Strict §14.5 disposition discipline: descriptive language only in the report
outside the explicitly tagged "Planning input" subsection.

Determinism convention (gate 6): two consecutive build passes within one
invocation must produce byte-identical CSV outputs; timestamps and wallclock
are emitted to stdout only and never written into any output file.

Output: results/l6/arc2/characterisation/extended/entry_filter_bivariate/
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Locked inputs (§1)
# ---------------------------------------------------------------------------

LOCKED_SHA256: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv": "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv": "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv": "4a61407f0f1fc1b74486f0614928e776201dc6469d874db8393e689d20cdb2ff",
    "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_L_all_features_summary.csv": "32a735c7bdccd135d6afa2b5e17d7504b2bc144c7f06e3098acce5c7766378a0",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py": "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml": "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "L6_0_METHODOLOGY_LOCK.md": "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}

# Exact baselines per prompt §3 — do NOT recompute.
BASELINE_SL_RATE: float = 0.7603
BASELINE_MEAN_R: float = -0.0193

# Reference quintile boundaries for dist_d1_kijun_atr per univariate report (§2).
EXPECTED_DIST_D1_BOUNDARIES: List[Tuple[float, float]] = [
    (-21.4963, -8.2322),
    (-8.2302, -5.3712),
    (-5.3711, -3.1497),
    (-3.1495, -1.4545),
    (-1.4540, -0.0047),
]
# Tolerance for matching boundaries against report-rounded values (4 d.p.).
BOUNDARY_TOL: float = 5e-5

# Univariate marginal cell sizes (per univariate report, gate 8).
EXPECTED_Q_SIZES_BY_Q: Dict[str, int] = {
    "Q1": 799,
    "Q2": 799,
    "Q3": 799,
    "Q4": 798,
    "Q5": 798,
}

OUT_DIR: Path = (
    REPO_ROOT
    / "results"
    / "l6"
    / "arc2"
    / "characterisation"
    / "extended"
    / "entry_filter_bivariate"
)


# ---------------------------------------------------------------------------
# sha256 helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_locked_inputs() -> None:
    fails: List[str] = []
    for rel, expected in LOCKED_SHA256.items():
        p = REPO_ROOT / rel
        if not p.exists():
            fails.append(f"MISSING: {rel}")
            continue
        got = _sha256_file(p)
        if got != expected:
            fails.append(f"MISMATCH: {rel}\n  expected: {expected}\n  got:      {got}")
    if fails:
        raise RuntimeError("HALT (gate 1): locked input integrity failed:\n  " + "\n  ".join(fails))


# ---------------------------------------------------------------------------
# Rank-based quintile bucketing (matches univariate run convention)
# ---------------------------------------------------------------------------


def _make_quintile_labels(
    values: pd.Series,
    tie_break: pd.Series,
) -> Tuple[pd.Series, List[Tuple[float, float]]]:
    """Rank-based quintile bucketing with deterministic tie-breaking.

    Ranks values ascending; ties broken by the ordering of `tie_break`
    (typically trade_id). Splits into 5 contiguous rank groups with sizes
    [base+1]*rem + [base]*(5-rem) where base = n // 5 and rem = n - base*5.
    For n = 3993: base = 798, rem = 3, so sizes = [799, 799, 799, 798, 798].

    Returns:
        labels:  Series of 'Q1'..'Q5' aligned to `values.index`.
        bounds:  [(min, max)] per quintile in Q1..Q5 order.
    """
    df = pd.DataFrame({"v": values.values, "t": tie_break.values}, index=values.index)
    df = df.sort_values(["v", "t"], kind="stable")
    n = len(df)
    base = n // 5
    rem = n - base * 5
    sizes = [base + (1 if i < rem else 0) for i in range(5)]
    labels: List[str] = []
    bounds: List[Tuple[float, float]] = []
    cursor = 0
    for qi, sz in enumerate(sizes):
        seg = df.iloc[cursor : cursor + sz]
        labels.extend([f"Q{qi + 1}"] * sz)
        bounds.append((float(seg["v"].min()), float(seg["v"].max())))
        cursor += sz
    df["q"] = labels
    return df["q"].reindex(values.index), bounds


# ---------------------------------------------------------------------------
# CSV writer — deterministic float format
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: Path, float_fmt: str = "%.10g") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", float_format=float_fmt)


# ---------------------------------------------------------------------------
# Markdown table renderer
# ---------------------------------------------------------------------------


def _df_to_md(df: pd.DataFrame, float_fmt: str = "{:.6f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            elif pd.isna(v):
                cells.append("")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Disposition discipline grep (§14.5)
# ---------------------------------------------------------------------------


FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "should filter",
    "best filter",
    "we should exclude",
    "this filter would pass",
    "recommend",
)
# "should filter on", "we should exclude" — these are case-insensitive substrings.
# We also block bare "recommend" anywhere outside the Planning input section.


def _check_disposition_discipline(report_text: str) -> List[Tuple[int, str, str]]:
    """Return list of (line_no, pattern, line_text) violations found outside the
    'Planning input' subsection. Case-insensitive substring match.

    The 'Planning input' subsection starts at the FIRST line matching
    `^## Planning input` and runs to end of file.
    """
    lines = report_text.splitlines()
    # Find planning-input boundary.
    planning_start: Optional[int] = None
    for i, ln in enumerate(lines):
        if re.match(r"^##\s+Planning input", ln, re.IGNORECASE):
            planning_start = i
            break
    violations: List[Tuple[int, str, str]] = []
    for i, ln in enumerate(lines):
        if planning_start is not None and i >= planning_start:
            continue
        ln_lc = ln.lower()
        for pat in FORBIDDEN_PATTERNS:
            if pat in ln_lc:
                violations.append((i + 1, pat, ln))
    return violations


# ---------------------------------------------------------------------------
# Build pass — performs all Block P computations and writes CSVs+manifest
# ---------------------------------------------------------------------------


def _build_pass(*, write_manifest: bool, run_label: str) -> Dict:
    """Single build pass. Returns receipt dict. If `write_manifest=False`,
    the manifest is not (re)written so it doesn't appear in the determinism
    comparison until the final pass."""
    t0 = _time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gate 1: input integrity at start of every pass.
    _verify_locked_inputs()

    # --- Load inputs ---
    sf_path = REPO_ROOT / "results/l6/arc2/characterisation/v1_1_full/signals_features.csv"
    ti_path = REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv"
    bm_path = (
        REPO_ROOT
        / "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv"
    )

    sf = pd.read_csv(sf_path)
    ti = pd.read_csv(ti_path)
    bm = pd.read_csv(bm_path)

    # --- Filter to taken; merge ---
    sf_taken = sf[sf["taken"] == True].copy()  # noqa: E712
    sf_taken = sf_taken.rename(columns={"time": "signal_bar_ts"})
    sf_taken["signal_bar_ts"] = pd.to_datetime(sf_taken["signal_bar_ts"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    taken = sf_taken.merge(
        ti[["trade_id", "pair", "signal_bar_ts", "atr_1h_wilder_at_signal"]],
        on=["pair", "signal_bar_ts"],
        how="left",
        validate="one_to_one",
    )
    taken = taken.merge(
        bm[["trade_id", "dist_d1_kijun_atr"]],
        on="trade_id",
        how="left",
        validate="one_to_one",
    )
    taken = taken.sort_values("trade_id").reset_index(drop=True)

    if len(taken) != 3993:
        raise RuntimeError(f"HALT: taken row count {len(taken)} != 3993")

    # --- Bucket features (§2) ---
    qa_labels, qa_bounds = _make_quintile_labels(
        taken["concurrent_signals_same_bar"],
        taken["trade_id"],
    )
    taken["Q_A_concurrent"] = qa_labels

    qb_labels, qb_bounds = _make_quintile_labels(
        taken["dist_d1_kijun_atr"],
        taken["trade_id"],
    )
    taken["Q_B_dist_d1"] = qb_labels

    # --- §2 validation: each Q marginal count in {798, 799} for both features ---
    qa_counts = taken["Q_A_concurrent"].value_counts().to_dict()
    qb_counts = taken["Q_B_dist_d1"].value_counts().to_dict()
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        exp_n = EXPECTED_Q_SIZES_BY_Q[q]
        a_n = int(qa_counts.get(q, 0))
        b_n = int(qb_counts.get(q, 0))
        if a_n != exp_n:
            raise RuntimeError(
                f"HALT (§2 gate): concurrent_signals_same_bar marginal {q} = {a_n}, expected {exp_n}.\n"
                f"  full QA counts: {qa_counts}"
            )
        if b_n != exp_n:
            raise RuntimeError(
                f"HALT (§2 gate): dist_d1_kijun_atr marginal {q} = {b_n}, expected {exp_n}.\n"
                f"  full QB counts: {qb_counts}"
            )

    # --- Gate 4 (§8): dist_d1_kijun_atr boundary parity against univariate report ---
    boundary_violations: List[str] = []
    for qi, (lo_got, hi_got) in enumerate(qb_bounds):
        lo_exp, hi_exp = EXPECTED_DIST_D1_BOUNDARIES[qi]
        if abs(round(lo_got, 4) - lo_exp) > BOUNDARY_TOL:
            boundary_violations.append(f"Q{qi + 1} min: got {lo_got:.6f}, expected ~{lo_exp:.4f}")
        if abs(round(hi_got, 4) - hi_exp) > BOUNDARY_TOL:
            boundary_violations.append(f"Q{qi + 1} max: got {hi_got:.6f}, expected ~{hi_exp:.4f}")
    if boundary_violations:
        raise RuntimeError(
            "HALT (gate 4): dist_d1_kijun_atr quintile boundaries diverge from "
            "univariate report:\n  " + "\n  ".join(boundary_violations)
        )

    # --- §3: Block P 5x5 cross-tab cell metrics ---
    exit_reason_map = {
        "stop_loss": "SL",
        "time_exit": "TE",
        "data_end": "DE",
    }
    taken["er_short"] = taken["exit_reason"].map(exit_reason_map)
    # The prompt's exit-reason codes are 'SL', 'TE', 'DE' (§3).

    Q_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    cell_rows: List[Dict] = []
    for qa in Q_ORDER:
        for qb in Q_ORDER:
            sub = taken[(taken["Q_A_concurrent"] == qa) & (taken["Q_B_dist_d1"] == qb)]
            n = int(len(sub))
            if n == 0:
                sl_rate = float("nan")
                te_rate = float("nan")
                de_rate = float("nan")
                mean_R = float("nan")
                median_R = float("nan")
                pct_reached_1R = float("nan")
            else:
                sl_rate = float((sub["er_short"] == "SL").mean())
                te_rate = float((sub["er_short"] == "TE").mean())
                de_rate = float((sub["er_short"] == "DE").mean())
                mean_R = float(sub["R"].mean())
                median_R = float(sub["R"].median())
                pct_reached_1R = float((sub["mfe_R"] >= 1.0).mean())
            sl_rate_lift = (sl_rate - BASELINE_SL_RATE) if n > 0 else float("nan")
            mean_R_lift = (mean_R - BASELINE_MEAN_R) if n > 0 else float("nan")
            if n < 30:
                thin = "very_thin"
            elif n < 100:
                thin = "thin"
            else:
                thin = ""
            cell_rows.append(
                {
                    "Q_A_concurrent": qa,
                    "Q_B_dist_d1": qb,
                    "n": n,
                    "sl_rate": sl_rate,
                    "te_rate": te_rate,
                    "de_rate": de_rate,
                    "mean_R": mean_R,
                    "median_R": median_R,
                    "pct_reached_1R_mfe": pct_reached_1R,
                    "sl_rate_lift_vs_baseline": sl_rate_lift,
                    "mean_R_lift_vs_baseline": mean_R_lift,
                    "thin_cells": thin,
                }
            )
    cells = pd.DataFrame(cell_rows)

    # --- Gate 3 (§8): bivariate cell-count sum = 3,993 ---
    sum_n = int(cells["n"].sum())
    if sum_n != 3993:
        raise RuntimeError(f"HALT (gate 3): sum of bivariate cell n = {sum_n} != 3993")

    # --- §5 marginal consistency check (gate 2/§8) ---
    # Recover each univariate marginal from bivariate sums; compare to
    # expected sizes AND recompute the univariate sl_rate from the cells
    # within each Q_A (resp Q_B) row, then assert exact match to the
    # univariate report value.
    marg_a = cells.groupby("Q_A_concurrent", as_index=False).agg(
        sum_n=("n", "sum"),
    )
    marg_b = cells.groupby("Q_B_dist_d1", as_index=False).agg(
        sum_n=("n", "sum"),
    )
    marg_mismatch: List[str] = []
    for _, row in marg_a.iterrows():
        q = row["Q_A_concurrent"]
        exp_n = EXPECTED_Q_SIZES_BY_Q[q]
        if int(row["sum_n"]) != exp_n:
            marg_mismatch.append(f"Q_A_concurrent {q}: sum_n={int(row['sum_n'])} != {exp_n}")
    for _, row in marg_b.iterrows():
        q = row["Q_B_dist_d1"]
        exp_n = EXPECTED_Q_SIZES_BY_Q[q]
        if int(row["sum_n"]) != exp_n:
            marg_mismatch.append(f"Q_B_dist_d1 {q}: sum_n={int(row['sum_n'])} != {exp_n}")

    # Now reconstruct univariate sl_rate from bivariate cells: sl_rate(Q) =
    # sum(n_cell * sl_rate_cell) / sum(n_cell) per Q.
    # Compare against the SL count computed directly from `taken`.
    direct_a = taken.groupby("Q_A_concurrent", as_index=False).agg(
        sl_rate_direct=("er_short", lambda s: float((s == "SL").mean())),
        n_direct=("er_short", "count"),
    )
    direct_b = taken.groupby("Q_B_dist_d1", as_index=False).agg(
        sl_rate_direct=("er_short", lambda s: float((s == "SL").mean())),
        n_direct=("er_short", "count"),
    )
    # Reconstructed from cells.
    cells_with_slcount = cells.assign(sl_count=cells["n"] * cells["sl_rate"])
    recon_a = cells_with_slcount.groupby("Q_A_concurrent", as_index=False).agg(
        sl_count_sum=("sl_count", "sum"),
        n_sum=("n", "sum"),
    )
    recon_a["sl_rate_recon"] = recon_a["sl_count_sum"] / recon_a["n_sum"]
    recon_b = cells_with_slcount.groupby("Q_B_dist_d1", as_index=False).agg(
        sl_count_sum=("sl_count", "sum"),
        n_sum=("n", "sum"),
    )
    recon_b["sl_rate_recon"] = recon_b["sl_count_sum"] / recon_b["n_sum"]

    margin_recon_disp: List[Dict] = []
    for _, row in direct_a.merge(recon_a, on="Q_A_concurrent").iterrows():
        diff = abs(float(row["sl_rate_direct"]) - float(row["sl_rate_recon"]))
        if diff > 1e-12:
            marg_mismatch.append(
                f"Q_A_concurrent {row['Q_A_concurrent']}: sl_rate_direct={row['sl_rate_direct']} "
                f"!= sl_rate_recon={row['sl_rate_recon']} (diff={diff:.3e})"
            )
        margin_recon_disp.append(
            {
                "axis": "Q_A_concurrent",
                "cell": row["Q_A_concurrent"],
                "n_direct": int(row["n_direct"]),
                "n_recon": int(row["n_sum"]),
                "sl_rate_direct": float(row["sl_rate_direct"]),
                "sl_rate_recon": float(row["sl_rate_recon"]),
                "abs_diff": diff,
            }
        )
    for _, row in direct_b.merge(recon_b, on="Q_B_dist_d1").iterrows():
        diff = abs(float(row["sl_rate_direct"]) - float(row["sl_rate_recon"]))
        if diff > 1e-12:
            marg_mismatch.append(
                f"Q_B_dist_d1 {row['Q_B_dist_d1']}: sl_rate_direct={row['sl_rate_direct']} "
                f"!= sl_rate_recon={row['sl_rate_recon']} (diff={diff:.3e})"
            )
        margin_recon_disp.append(
            {
                "axis": "Q_B_dist_d1",
                "cell": row["Q_B_dist_d1"],
                "n_direct": int(row["n_direct"]),
                "n_recon": int(row["n_sum"]),
                "sl_rate_direct": float(row["sl_rate_direct"]),
                "sl_rate_recon": float(row["sl_rate_recon"]),
                "abs_diff": diff,
            }
        )

    if marg_mismatch:
        raise RuntimeError(
            "HALT (gate 2 / §5): bivariate marginal consistency check failed:\n  "
            + "\n  ".join(marg_mismatch)
        )

    margin_recon_df = pd.DataFrame(margin_recon_disp)

    # --- §4 per-fold stability ---
    per_fold_rows: List[Dict] = []
    stability_rows: List[Dict] = []
    folds_in_data = sorted(taken["fold_id"].unique().tolist())
    for qa in Q_ORDER:
        for qb in Q_ORDER:
            sub_cell = taken[(taken["Q_A_concurrent"] == qa) & (taken["Q_B_dist_d1"] == qb)]
            sl_rate_per_fold: List[Tuple[int, int, float]] = []  # (fold, n, sl_rate)
            for fold in folds_in_data:
                sub_f = sub_cell[sub_cell["fold_id"] == fold]
                n_f = int(len(sub_f))
                sl_f = float((sub_f["er_short"] == "SL").mean()) if n_f > 0 else float("nan")
                sl_rate_per_fold.append((int(fold), n_f, sl_f))
                per_fold_rows.append(
                    {
                        "Q_A_concurrent": qa,
                        "Q_B_dist_d1": qb,
                        "fold_id": int(fold),
                        "n_fold": n_f,
                        "sl_rate_fold": sl_f,
                    }
                )
            qualifying = [(f, n, sl) for (f, n, sl) in sl_rate_per_fold if n >= 10]
            folds_n_ge_10 = len(qualifying)
            folds_below = sum(1 for (_, _, sl) in qualifying if sl < BASELINE_SL_RATE)
            folds_above = sum(1 for (_, _, sl) in qualifying if sl > BASELINE_SL_RATE)
            if folds_n_ge_10 >= 2:
                variance = float(np.var([sl for (_, _, sl) in qualifying], ddof=0))
            else:
                variance = float("nan")
            if folds_n_ge_10 < 5:
                stability_flag = "thin"
            elif folds_below >= 5 or folds_above >= 5:
                stability_flag = "stable"
            else:
                stability_flag = "variable"
            stability_rows.append(
                {
                    "Q_A_concurrent": qa,
                    "Q_B_dist_d1": qb,
                    "folds_with_n_ge_10": folds_n_ge_10,
                    "folds_sl_rate_below_baseline": folds_below,
                    "folds_sl_rate_above_baseline": folds_above,
                    "sl_rate_variance_across_folds": variance,
                    "stability_flag": stability_flag,
                }
            )
    per_fold_df = (
        pd.DataFrame(per_fold_rows)
        .sort_values(["Q_A_concurrent", "Q_B_dist_d1", "fold_id"])
        .reset_index(drop=True)
    )
    stability_df = (
        pd.DataFrame(stability_rows)
        .sort_values(["Q_A_concurrent", "Q_B_dist_d1"])
        .reset_index(drop=True)
    )

    # --- Write CSVs ---
    _write_csv(cells, OUT_DIR / "block_P_bivariate_cells.csv")
    _write_csv(per_fold_df, OUT_DIR / "block_P_bivariate_per_fold.csv")
    _write_csv(stability_df, OUT_DIR / "block_P_bivariate_stability.csv")

    # --- Build markdown report ---
    report = _build_report(
        cells=cells,
        stability_df=stability_df,
        margin_recon_df=margin_recon_df,
        qa_bounds=qa_bounds,
        qb_bounds=qb_bounds,
        per_fold_df=per_fold_df,
    )

    # --- Gate 8: disposition discipline grep ---
    violations = _check_disposition_discipline(report)
    if violations:
        msg = "\n  ".join(f"line {ln}: pattern '{pat}' in: {txt!r}" for ln, pat, txt in violations)
        raise RuntimeError(
            f"HALT (gate 8): disposition discipline violations in report outside Planning input:\n  {msg}"
        )

    (OUT_DIR / "entry_filter_bivariate.md").write_text(report, encoding="utf-8")

    # --- Write manifest (only on final pass to keep determinism semantics clean) ---
    out_sha: Dict[str, str] = {}
    for p in sorted(OUT_DIR.glob("*")):
        if p.name == "run_manifest.txt":
            continue
        out_sha[p.name] = _sha256_file(p)

    if write_manifest:
        manifest_lines: List[str] = []
        manifest_lines.append("Arc 2 bivariate entry-filter exploration — run_manifest.txt")
        manifest_lines.append("=" * 72)
        manifest_lines.append("")
        manifest_lines.append(
            "Wall-clock and generation timestamp are omitted from this file to keep "
            "it byte-identical across consecutive build passes (gate 6). They are "
            "printed to stdout by the script for the operator's records."
        )
        manifest_lines.append("")
        manifest_lines.append("Locked input artefacts (gate 1):")
        for rel, h in LOCKED_SHA256.items():
            manifest_lines.append(f"  {h}  {rel}")
        manifest_lines.append("")
        manifest_lines.append("Output artefacts (sha256):")
        for name in sorted(out_sha):
            manifest_lines.append(f"  {out_sha[name]}  {name}")
        manifest_lines.append("")
        (OUT_DIR / "run_manifest.txt").write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8"
        )

    wallclock_s = _time.time() - t0
    return {
        "wallclock_s": wallclock_s,
        "n_taken": int(len(taken)),
        "qa_bounds": qa_bounds,
        "qb_bounds": qb_bounds,
        "cells": cells,
        "stability_df": stability_df,
        "out_sha": out_sha,
        "margin_recon_df": margin_recon_df,
    }


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------


def _build_report(
    *,
    cells: pd.DataFrame,
    stability_df: pd.DataFrame,
    margin_recon_df: pd.DataFrame,
    qa_bounds: List[Tuple[float, float]],
    qb_bounds: List[Tuple[float, float]],
    per_fold_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# Arc 2 — Bivariate Entry-Filter Exploration")
    lines.append("")
    lines.append(
        "Descriptive characterisation per L6.0 v1.1 §14.5. Cross-tabulates "
        "`concurrent_signals_same_bar` (quintile Q_A) against `dist_d1_kijun_atr` "
        "(quintile Q_B) on the 3,993 taken trades."
    )
    lines.append("")
    lines.append(
        "Generation timestamp and wallclock are intentionally omitted from this "
        "report to preserve byte-level determinism across consecutive build "
        "passes (gate 6). Run metadata is emitted to stdout."
    )
    lines.append("")

    # Locked-input manifest.
    lines.append("## Locked input sha256 manifest")
    lines.append("")
    for rel, h in LOCKED_SHA256.items():
        lines.append(f"- `{rel}` = {h}")
    lines.append("")

    # Determinism receipt.
    lines.append("## Determinism receipt")
    lines.append("")
    lines.append(
        "The build is executed twice within one invocation. After the second "
        "pass, the sha256 of each output CSV and of this markdown file is "
        "compared against the first-pass values. Byte-identical match across "
        "all four output files is the gate-6 pass condition. See "
        "`run_manifest.txt` for the final output sha256s."
    )
    lines.append("")
    lines.append("Quintile boundaries (min, max) actually observed in this run:")
    lines.append("")
    qa_b = ", ".join(f"Q{i + 1}=[{lo:.4f},{hi:.4f}]" for i, (lo, hi) in enumerate(qa_bounds))
    qb_b = ", ".join(f"Q{i + 1}=[{lo:.4f},{hi:.4f}]" for i, (lo, hi) in enumerate(qb_bounds))
    lines.append(f"- `concurrent_signals_same_bar`: {qa_b}")
    lines.append(f"- `dist_d1_kijun_atr`: {qb_b}")
    lines.append("")

    # Marginal consistency check.
    lines.append("## Marginal consistency check (gate 5 disposition)")
    lines.append("")
    lines.append(
        "For each univariate quintile, the bivariate cell counts are summed "
        "across the other feature's quintiles and compared against the direct "
        "univariate SL rate computed from the same taken population. The "
        "expected per-quintile sizes from the univariate report are "
        "{Q1: 799, Q2: 799, Q3: 799, Q4: 798, Q5: 798}."
    )
    lines.append("")
    lines.append(
        "Disposition: **PASS** — every marginal n matches the expected size and "
        "every reconstructed sl_rate matches the direct sl_rate to 1e-12."
    )
    lines.append("")
    margin_disp = margin_recon_df.copy()
    margin_disp["sl_rate_direct"] = margin_disp["sl_rate_direct"].round(10)
    margin_disp["sl_rate_recon"] = margin_disp["sl_rate_recon"].round(10)
    margin_disp["abs_diff"] = margin_disp["abs_diff"].apply(lambda v: f"{v:.3e}")
    lines.append(_df_to_md(margin_disp, "{:.10f}"))
    lines.append("")

    # Block P cells.
    lines.append("## Block P — 5x5 cross-tab cell metrics")
    lines.append("")
    lines.append(
        "One row per (Q_A_concurrent, Q_B_dist_d1) intersection. Baselines per "
        f"prompt §3: SL rate baseline = {BASELINE_SL_RATE}, mean R baseline = "
        f"{BASELINE_MEAN_R}. Lift columns are cell value minus baseline. "
        "`thin_cells` = 'thin' when n < 100, 'very_thin' when n < 30, empty "
        "otherwise."
    )
    lines.append("")
    cell_display = cells.copy()
    lines.append(_df_to_md(cell_display, "{:.4f}"))
    lines.append("")

    # Per-fold stability.
    lines.append("## Block P — per-fold stability")
    lines.append("")
    lines.append(
        "One row per (Q_A, Q_B) cell. `folds_with_n_ge_10` is the count of "
        "folds (out of 7) where the cell has at least 10 trades. The two "
        "directional counts measure within those qualifying folds. "
        "`stability_flag` = 'thin' if fewer than 5 qualifying folds; "
        "'stable' if a clear directional majority (≥ 5 folds on one side of "
        "the population SL baseline); 'variable' otherwise."
    )
    lines.append("")
    lines.append(_df_to_md(stability_df, "{:.4f}"))
    lines.append("")

    # Thin-cell flag summary.
    lines.append("## Block P — thin-cell flag summary")
    lines.append("")
    thin_counts = cells["thin_cells"].value_counts().to_dict()
    n_very_thin = int(thin_counts.get("very_thin", 0))
    n_thin = int(thin_counts.get("thin", 0))
    n_ok = int(thin_counts.get("", 0))
    lines.append(f"- very_thin (n < 30): **{n_very_thin}** cells")
    lines.append(f"- thin (30 ≤ n < 100): **{n_thin}** cells")
    lines.append(f"- ≥ 100: **{n_ok}** cells")
    lines.append("")
    if n_very_thin > 0 or n_thin > 0:
        thin_disp = cells[cells["thin_cells"] != ""][
            [
                "Q_A_concurrent",
                "Q_B_dist_d1",
                "n",
                "thin_cells",
            ]
        ].reset_index(drop=True)
        lines.append("Thin / very-thin cells:")
        lines.append("")
        lines.append(_df_to_md(thin_disp, "{:.4f}"))
        lines.append("")

    # Distribution descriptives.
    lines.append("## Block P — distribution of mean_R and SL rate across cells")
    lines.append("")
    lines.append(
        "Descriptive summary of cell-level metric distributions across the "
        "25 cells (all included, including thin ones)."
    )
    lines.append("")
    desc_rows = []
    for metric in (
        "n",
        "sl_rate",
        "te_rate",
        "mean_R",
        "median_R",
        "pct_reached_1R_mfe",
        "sl_rate_lift_vs_baseline",
        "mean_R_lift_vs_baseline",
    ):
        s = cells[metric].astype(float)
        desc_rows.append(
            {
                "metric": metric,
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "mean": float(s.mean()),
            }
        )
    lines.append(_df_to_md(pd.DataFrame(desc_rows), "{:.4f}"))
    lines.append("")

    # Planning-input subsection (explicitly tagged — disposition discipline grep
    # is suppressed inside this section).
    lines.append("## Planning input")
    lines.append("")
    lines.append(
        "This subsection is explicitly tagged per prompt §6 / §14.5. Entries "
        "are descriptive rankings sorted by criterion; no imperative language "
        "is used. The chat layer uses this section to inform the Phase 3 spec "
        "lock; this script does not select or lock a filter."
    )
    lines.append("")

    merged = cells.merge(
        stability_df[
            [
                "Q_A_concurrent",
                "Q_B_dist_d1",
                "stability_flag",
                "folds_with_n_ge_10",
                "folds_sl_rate_below_baseline",
                "folds_sl_rate_above_baseline",
            ]
        ],
        on=["Q_A_concurrent", "Q_B_dist_d1"],
        how="left",
    )

    def _rank_table(df_sub: pd.DataFrame, by: str, ascending: bool, k: int = 5) -> pd.DataFrame:
        d = df_sub.sort_values(by, ascending=ascending).head(k).copy()
        return d[
            [
                "Q_A_concurrent",
                "Q_B_dist_d1",
                "n",
                "mean_R",
                "mean_R_lift_vs_baseline",
                "sl_rate",
                "sl_rate_lift_vs_baseline",
                "pct_reached_1R_mfe",
                "stability_flag",
                "folds_sl_rate_below_baseline",
                "folds_with_n_ge_10",
            ]
        ]

    # (a) Top 5 cells by mean_R lift (descending = highest lift first).
    lines.append("### (a) Top 5 cells by mean_R lift vs baseline")
    lines.append("")
    lines.append(
        _df_to_md(_rank_table(merged, "mean_R_lift_vs_baseline", ascending=False), "{:.4f}")
    )
    lines.append("")

    # (b) Top 5 cells by SL rate lift below baseline (= most negative lift first).
    lines.append("### (b) Top 5 cells by SL rate lift below baseline (most negative lift)")
    lines.append("")
    lines.append(
        _df_to_md(_rank_table(merged, "sl_rate_lift_vs_baseline", ascending=True), "{:.4f}")
    )
    lines.append("")

    # (c) Top 5 cells by pct_reached_1R_mfe.
    lines.append("### (c) Top 5 cells by pct_reached_1R_mfe")
    lines.append("")
    lines.append(_df_to_md(_rank_table(merged, "pct_reached_1R_mfe", ascending=False), "{:.4f}"))
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Determinism harness — run two consecutive build passes within one invocation
# ---------------------------------------------------------------------------


def _git_head() -> str:
    try:
        out = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return out
    except Exception:
        return "<unavailable>"


def main() -> Dict:
    head_at_start = _git_head()

    # Pass A: build, write CSVs and markdown; do NOT write manifest yet.
    print("=== Pass A (build) ===", flush=True)
    rcpt_a = _build_pass(write_manifest=False, run_label="A")
    pass_a_sha: Dict[str, str] = {}
    for p in sorted(OUT_DIR.glob("*")):
        if p.name == "run_manifest.txt":
            continue
        pass_a_sha[p.name] = _sha256_file(p)
    print(f"  pass A wallclock: {rcpt_a['wallclock_s']:.3f}s")
    for k, h in pass_a_sha.items():
        print(f"    {h}  {k}")

    # Pass B: rebuild same outputs and write manifest.
    print("=== Pass B (rebuild + manifest) ===", flush=True)
    rcpt_b = _build_pass(write_manifest=True, run_label="B")
    pass_b_sha: Dict[str, str] = {}
    for p in sorted(OUT_DIR.glob("*")):
        if p.name == "run_manifest.txt":
            continue
        pass_b_sha[p.name] = _sha256_file(p)
    print(f"  pass B wallclock: {rcpt_b['wallclock_s']:.3f}s")

    # Gate 6: determinism.
    diffs: List[Tuple[str, str, str]] = []
    if pass_a_sha.keys() != pass_b_sha.keys():
        only_a = sorted(set(pass_a_sha) - set(pass_b_sha))
        only_b = sorted(set(pass_b_sha) - set(pass_a_sha))
        raise RuntimeError(
            f"HALT (gate 6): output file set differs between passes.\n"
            f"  only in A: {only_a}\n  only in B: {only_b}"
        )
    for k in sorted(pass_a_sha):
        if pass_a_sha[k] != pass_b_sha[k]:
            diffs.append((k, pass_a_sha[k], pass_b_sha[k]))
    if diffs:
        msg = "\n  ".join(f"{k}\n    A: {a}\n    B: {b}" for k, a, b in diffs)
        raise RuntimeError(f"HALT (gate 6): non-deterministic outputs:\n  {msg}")

    # Gate 7: locked-artefact integrity post-run.
    _verify_locked_inputs()

    # Gate 9: no auto-commit (HEAD unchanged + nothing staged).
    head_at_end = _git_head()
    if head_at_end != head_at_start:
        raise RuntimeError(
            f"HALT (gate 9): HEAD changed during run: {head_at_start} -> {head_at_end}"
        )
    try:
        staged = (
            subprocess.check_output(
                ["git", "diff", "--cached", "--name-only"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        staged = ""
    # We're not allowed to STAGE anything from this run. The repo may have
    # pre-existing staged files (e.g. from the user's working tree at session
    # start); we report status but only HALT if our outputs/script appear staged.
    if staged:
        offending: List[str] = []
        for ln in staged.splitlines():
            ln = ln.strip()
            if ln.startswith("results/l6/arc2/characterisation/extended/entry_filter_bivariate/"):
                offending.append(ln)
            if ln == "scripts/lchar/arc2_entry_filter_bivariate.py":
                offending.append(ln)
        if offending:
            raise RuntimeError(
                f"HALT (gate 9): bivariate outputs/script staged for commit: {offending}"
            )

    receipt = {
        "wallclock_s_total": rcpt_a["wallclock_s"] + rcpt_b["wallclock_s"],
        "wallclock_s_a": rcpt_a["wallclock_s"],
        "wallclock_s_b": rcpt_b["wallclock_s"],
        "n_taken": rcpt_b["n_taken"],
        "out_sha": pass_b_sha,
        "head_at_start": head_at_start,
        "head_at_end": head_at_end,
        "qa_bounds": rcpt_b["qa_bounds"],
        "qb_bounds": rcpt_b["qb_bounds"],
        "cells": rcpt_b["cells"],
        "stability_df": rcpt_b["stability_df"],
    }
    return receipt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    receipt = main()
    print()
    print("=== Bivariate entry-filter exploration — receipt ===")
    print(f"n_taken: {receipt['n_taken']}")
    print(f"wallclock_s (pass A + B): {receipt['wallclock_s_total']:.3f}")
    print(f"git HEAD start: {receipt['head_at_start']}")
    print(f"git HEAD end:   {receipt['head_at_end']}")
    print()
    print("Output sha256:")
    for k, h in receipt["out_sha"].items():
        print(f"  {h}  {k}")
    print()
    print("Quintile boundaries (final pass):")
    print("  concurrent_signals_same_bar:", receipt["qa_bounds"])
    print("  dist_d1_kijun_atr:", receipt["qb_bounds"])

    # Headline cells: top stable cell by mean_R lift with n >= 50, and (Q5, Q3).
    cells = receipt["cells"]
    stab = receipt["stability_df"]
    merged = cells.merge(
        stab[
            [
                "Q_A_concurrent",
                "Q_B_dist_d1",
                "stability_flag",
                "folds_with_n_ge_10",
                "folds_sl_rate_below_baseline",
            ]
        ],
        on=["Q_A_concurrent", "Q_B_dist_d1"],
        how="left",
    )
    stable_50 = merged[(merged["stability_flag"] == "stable") & (merged["n"] >= 50)]
    if len(stable_50) > 0:
        top = stable_50.sort_values("mean_R_lift_vs_baseline", ascending=False).iloc[0]
        print()
        print("Top stable cell (n>=50) by mean_R lift:")
        print(
            f"  (Q_A={top['Q_A_concurrent']}, Q_B={top['Q_B_dist_d1']})  "
            f"n={int(top['n'])}, mean_R={top['mean_R']:.4f}, "
            f"mean_R_lift={top['mean_R_lift_vs_baseline']:.4f}, "
            f"sl_rate={top['sl_rate']:.4f}, "
            f"sl_rate_lift={top['sl_rate_lift_vs_baseline']:.4f}, "
            f"folds_below_baseline/folds_n_ge_10 = "
            f"{int(top['folds_sl_rate_below_baseline'])}/{int(top['folds_with_n_ge_10'])}"
        )
    q5q3 = merged[(merged["Q_A_concurrent"] == "Q5") & (merged["Q_B_dist_d1"] == "Q3")]
    if len(q5q3) > 0:
        q = q5q3.iloc[0]
        print()
        print("(Q5, Q3) cell:")
        print(
            f"  n={int(q['n'])}, mean_R={q['mean_R']:.4f}, "
            f"mean_R_lift={q['mean_R_lift_vs_baseline']:.4f}, "
            f"sl_rate={q['sl_rate']:.4f}, "
            f"sl_rate_lift={q['sl_rate_lift_vs_baseline']:.4f}, "
            f"stability_flag={q['stability_flag']}"
        )

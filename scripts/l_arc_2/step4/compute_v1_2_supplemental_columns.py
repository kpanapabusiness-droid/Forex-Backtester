"""Arc 2 step 4 — v1.2 A3 §2 supplemental columns.

Joins existing per-candidate trades_post_mechanism.csv outputs with
cluster_assignments.csv (K=2 kmeans) and signals_features.csv to compute
five disposition columns that the v1.1 component table did not produce.

This script does NOT recompute any v1.1 step 4 mechanism. It is a join +
aggregation pass over outputs that already exist.

Output:
  results/l_arc_2/step4/v1_2_supplemental_columns.csv  (11 rows, schema in dispatch)
  results/l_arc_2/step4/v1_2_supplemental_columns_receipts.txt

Pool definition: "mirror" = all trades with K2_kmeans != 1 (target). The
1137-row mirror pool = 1025 (K2_kmeans == 0) + 112 (K2_kmeans == -2 sentinel,
preserved from NaN-drop in clustering per step3 §4.1).

Determinism: byte-identical CSV on rerun. No randomness; deterministic
pandas reductions over sorted inputs.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
STEP2_DIR = REPO / "results" / "l_arc_2" / "step2_descriptive"
STEP3_DIR = REPO / "results" / "l_arc_2" / "step3_extractability"
STEP4_DIR = REPO / "results" / "l_arc_2" / "step4"

SIGNALS_CSV = STEP2_DIR / "signals_features.csv"
CLUSTER_CSV = STEP3_DIR / "cluster_assignments.csv"

OUT_CSV = STEP4_DIR / "v1_2_supplemental_columns.csv"
OUT_RECEIPTS = STEP4_DIR / "v1_2_supplemental_columns_receipts.txt"

# Cluster column in cluster_assignments.csv
CLUSTER_COL = "K2_kmeans"
TARGET_CLUSTER_ID = 1  # mirror = all non-target trades (including sentinel -2)

# Hard pool assertions (dispatch §"Receipts")
EXPECTED_POOL_TOTAL = 3993
EXPECTED_MIRROR_POOL = 1137
EXPECTED_MIRROR_FRAC = 0.2848  # rounded; assert within 1e-4

# Forward-geometry columns in signals_features.csv
FWD_MFE_COL = "fwd_mfe_h24_atr"
FWD_MAE_COL = "fwd_mae_h24_atr"
BARS_PLUS_COL = "bars_to_plus_1.0_atr_capped_480"
BARS_MINUS_COL = "bars_to_minus_1.0_atr_capped_480"
FOLD_COL = "fold_id"

# Candidate slugs — hard-coded order per dispatch (NOT a glob)
SLUGS = (
    "delayed_entry_t_gb",
    "exit_cluster_cond_gb",
    "exit_cluster_cond_gb_h240",
    "exit_only_unfiltered_h240",
    "filter_atr_at_signal_above_p50",
    "filter_basket_eur_above_p50",
    "filter_basket_gbp_above_p50",
    "filter_basket_jpy_above_p50",
    "filter_basket_usd_above_p50",
    "filter_concurrent_signals_above_p75",
    "filter_jpy_pairs",
)

# Slugs whose mechanism does not drop trades — hardcoded per dispatch
EXIT_CLASS_NO_DROP = frozenset(
    {
        "exit_only_unfiltered_h240",
        "exit_cluster_cond_gb",
        "exit_cluster_cond_gb_h240",
    }
)

OUT_SCHEMA = (
    "slug",
    "winner_retention_rate",
    "race_preservation",
    "ratio_preservation",
    "concentration_lift_sign_consistency_per_fold",
    "concentration_lift_stability_cv",
    "n_mirror_pool",
    "n_mirror_retained",
    "n_post_total",
    "notes",
)

FOLDS = (1, 2, 3, 4, 5, 6, 7)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_float(x: float) -> str:
    """Deterministic float formatting for CSV output (6 sig figs)."""
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "inf" if x > 0 else "-inf" if x < 0 else "nan"
    return f"{x:.6g}"


def candidate_trade_ids(slug: str) -> set[int]:
    f = STEP4_DIR / slug / "trades_post_mechanism.csv"
    df = pd.read_csv(f, usecols=["trade_id"])
    return set(df["trade_id"].astype(int).tolist())


def compute_filter_row(
    slug: str,
    pool: pd.DataFrame,
    retained_ids: set[int],
    notes_acc: list[str],
) -> dict:
    """Compute 5 supplemental columns for a filter-class candidate."""
    notes: list[str] = []
    is_mirror = pool["is_mirror"].to_numpy()
    in_post = pool["trade_id"].isin(retained_ids).to_numpy()

    n_mirror_pool = int(is_mirror.sum())
    n_mirror_retained = int((is_mirror & in_post).sum())
    n_post_total = int(in_post.sum())

    # Column 1: winner_retention_rate
    winner_retention_rate = n_mirror_retained / n_mirror_pool

    # Column 2: race_preservation — median(bars_to_+1atr - bars_to_-1atr) over mirror
    race_diff = pool[BARS_PLUS_COL].to_numpy() - pool[BARS_MINUS_COL].to_numpy()
    race_pool_vals = race_diff[is_mirror]
    race_post_vals = race_diff[is_mirror & in_post]
    race_pool = float(np.median(race_pool_vals)) if race_pool_vals.size else float("nan")
    race_post = (
        float(np.median(race_post_vals)) if race_post_vals.size else float("nan")
    )
    if race_pool == 0 or (race_pool > 0) != (race_post > 0):
        race_preservation: float | str = race_post - race_pool
        notes.append(
            f"race_pool={race_pool:.4f} race_post={race_post:.4f} "
            "ratio undefined; reporting raw diff"
        )
    else:
        race_preservation = race_post / race_pool

    # Column 3: ratio_preservation — median(fwd_mfe/fwd_mae) over mirror, mae!=0
    mfe = pool[FWD_MFE_COL].to_numpy()
    mae = pool[FWD_MAE_COL].to_numpy()
    valid_ratio = (mae != 0) & np.isfinite(mae) & np.isfinite(mfe)
    n_dropped_pool = int((is_mirror & ~valid_ratio).sum())
    n_dropped_post = int((is_mirror & in_post & ~valid_ratio).sum())
    pool_mask = is_mirror & valid_ratio
    post_mask = is_mirror & in_post & valid_ratio
    ratios = np.where(valid_ratio, mfe / np.where(mae == 0, 1, mae), np.nan)
    ratio_pool_vals = ratios[pool_mask]
    ratio_post_vals = ratios[post_mask]
    ratio_pool = (
        float(np.median(ratio_pool_vals)) if ratio_pool_vals.size else float("nan")
    )
    ratio_post = (
        float(np.median(ratio_post_vals)) if ratio_post_vals.size else float("nan")
    )
    if ratio_pool == 0:
        ratio_preservation: float | str = ratio_post - ratio_pool
        notes.append("ratio_pool==0; reporting raw diff")
    else:
        ratio_preservation = ratio_post / ratio_pool
    if n_dropped_pool > 5 or n_dropped_post > 5:
        notes.append(
            f"dropped mae==0 from ratio: pool={n_dropped_pool} post={n_dropped_post}"
        )

    # Columns 4 & 5: per-fold concentration lift
    lifts: list[float] = []
    fold_vals = pool[FOLD_COL].to_numpy()
    for f in FOLDS:
        f_mask = fold_vals == f
        n_post_f = int((f_mask & in_post).sum())
        n_pool_f = int(f_mask.sum())
        if n_post_f == 0 or n_pool_f == 0:
            notes.append(f"fold {f}: empty (n_post={n_post_f} n_pool={n_pool_f})")
            lifts.append(float("nan"))
            continue
        p_mirror_retains_f = (f_mask & in_post & is_mirror).sum() / n_post_f
        p_mirror_pool_f = (f_mask & is_mirror).sum() / n_pool_f
        if p_mirror_pool_f == 0:
            lifts.append(float("inf") if p_mirror_retains_f > 0 else float("nan"))
            notes.append(f"fold {f}: p_mirror_pool_f==0")
        else:
            lifts.append(p_mirror_retains_f / p_mirror_pool_f)

    lifts_arr = np.array(lifts, dtype=float)
    finite_lifts = lifts_arr[np.isfinite(lifts_arr)]
    sign_consistent = int((lifts_arr > 1.0).sum())
    sign_consistency_str = f"{sign_consistent}/7"

    if finite_lifts.size == 0 or np.mean(finite_lifts) == 0:
        stability_cv: float | str = float("inf")
        notes.append("stability_cv: zero or non-finite mean of lifts")
    else:
        stability_cv = float(np.std(finite_lifts, ddof=0) / np.mean(finite_lifts))

    notes_str = "; ".join(notes) if notes else ""
    notes_acc.append(f"{slug}: lifts_per_fold={[fmt_float(x) for x in lifts]}")

    return {
        "slug": slug,
        "winner_retention_rate": fmt_float(winner_retention_rate),
        "race_preservation": fmt_float(race_preservation),
        "ratio_preservation": fmt_float(ratio_preservation),
        "concentration_lift_sign_consistency_per_fold": sign_consistency_str,
        "concentration_lift_stability_cv": fmt_float(stability_cv)
        if not isinstance(stability_cv, str)
        else stability_cv,
        "n_mirror_pool": str(n_mirror_pool),
        "n_mirror_retained": str(n_mirror_retained),
        "n_post_total": str(n_post_total),
        "notes": notes_str,
        # Internals (not in output schema; used for GBP check)
        "_lifts": lifts_arr,
        "_winner_retention_rate_f": winner_retention_rate,
        "_n_sign_consistent": sign_consistent,
    }


def compute_exit_class_row(slug: str, pool: pd.DataFrame) -> dict:
    """Hardcoded values for exit-class mechanisms that don't drop trades."""
    n_mirror_pool = int(pool["is_mirror"].sum())
    n_pool_total = int(len(pool))
    return {
        "slug": slug,
        "winner_retention_rate": "1.0",
        "race_preservation": "1.0",
        "ratio_preservation": "1.0",
        "concentration_lift_sign_consistency_per_fold": "N/A",
        "concentration_lift_stability_cv": "N/A",
        "n_mirror_pool": str(n_mirror_pool),
        "n_mirror_retained": str(n_mirror_pool),
        "n_post_total": str(n_pool_total),
        "notes": "mechanism_does_not_filter",
    }


def main() -> int:
    t0 = time.time()
    receipts: list[str] = []
    receipts.append("# v1.2 A3 §2 supplemental columns — Arc 2 step 4")
    receipts.append("")

    # Input sha256s
    sig_sha = sha256_file(SIGNALS_CSV)
    clu_sha = sha256_file(CLUSTER_CSV)
    receipts.append(f"signals_features.csv sha256:    {sig_sha}")
    receipts.append(f"cluster_assignments.csv sha256: {clu_sha}")

    # Load signals_features (only needed columns)
    sig_cols = [
        "trade_id",
        FOLD_COL,
        FWD_MFE_COL,
        FWD_MAE_COL,
        BARS_PLUS_COL,
        BARS_MINUS_COL,
    ]
    sig = pd.read_csv(SIGNALS_CSV, usecols=sig_cols)
    clu = pd.read_csv(CLUSTER_CSV, usecols=["trade_id", CLUSTER_COL])

    # Deterministic sort BEFORE join
    sig = sig.sort_values("trade_id", kind="mergesort").reset_index(drop=True)
    clu = clu.sort_values("trade_id", kind="mergesort").reset_index(drop=True)

    pool = sig.merge(clu, on="trade_id", how="inner", validate="one_to_one")
    pool["is_mirror"] = pool[CLUSTER_COL] != TARGET_CLUSTER_ID

    # --- Hard pool assertions (dispatch §Receipts) ---
    n_pool_total = int(len(pool))
    n_mirror_pool = int(pool["is_mirror"].sum())
    mirror_frac = n_mirror_pool / n_pool_total

    receipts.append("")
    receipts.append("## Pool-level assertions")
    receipts.append(f"n_pool_total     = {n_pool_total} (expected {EXPECTED_POOL_TOTAL})")
    receipts.append(f"n_mirror_pool    = {n_mirror_pool} (expected {EXPECTED_MIRROR_POOL})")
    receipts.append(
        f"mirror_fraction  = {mirror_frac:.4f} (expected {EXPECTED_MIRROR_FRAC:.4f})"
    )
    receipts.append(
        f"mirror = trades with K2_kmeans != {TARGET_CLUSTER_ID} "
        "(includes 112 sentinel-(-2) rows preserved from NaN-drop in clustering)"
    )

    if n_pool_total != EXPECTED_POOL_TOTAL:
        msg = f"POOL DRIFT: n_pool_total={n_pool_total} != {EXPECTED_POOL_TOTAL}"
        receipts.append(f"FATAL: {msg}")
        OUT_RECEIPTS.write_text("\n".join(receipts) + "\n", encoding="utf-8")
        raise SystemExit(msg)
    if n_mirror_pool != EXPECTED_MIRROR_POOL:
        msg = (
            f"POOL DRIFT: n_mirror_pool={n_mirror_pool} != {EXPECTED_MIRROR_POOL}"
        )
        receipts.append(f"FATAL: {msg}")
        OUT_RECEIPTS.write_text("\n".join(receipts) + "\n", encoding="utf-8")
        raise SystemExit(msg)
    if abs(mirror_frac - EXPECTED_MIRROR_POOL / EXPECTED_POOL_TOTAL) > 1e-4:
        msg = f"POOL DRIFT: mirror_fraction={mirror_frac:.6f}"
        receipts.append(f"FATAL: {msg}")
        OUT_RECEIPTS.write_text("\n".join(receipts) + "\n", encoding="utf-8")
        raise SystemExit(msg)

    receipts.append("ALL POOL ASSERTIONS PASSED.")
    receipts.append("")

    # --- Per-candidate row counts + sha256s ---
    receipts.append("## Per-candidate inputs")
    cand_meta: dict[str, dict] = {}
    for slug in SLUGS:
        f = STEP4_DIR / slug / "trades_post_mechanism.csv"
        sha = sha256_file(f)
        rows = sum(1 for _ in f.open("r", encoding="utf-8")) - 1
        receipts.append(f"{slug:42s} rows={rows:5d}  sha256={sha}")
        cand_meta[slug] = {"sha": sha, "rows": rows}
    receipts.append("")

    # --- Compute supplemental columns ---
    notes_acc: list[str] = []
    rows: list[dict] = []
    for slug in SLUGS:
        if slug in EXIT_CLASS_NO_DROP:
            row = compute_exit_class_row(slug, pool)
        else:
            retained_ids = candidate_trade_ids(slug)
            row = compute_filter_row(slug, pool, retained_ids, notes_acc)
        rows.append(row)

    # --- GBP contradiction check (dispatch §Validation checklist) ---
    # v1.2 dispositions GBP as "concentration in wrong direction" — i.e. the
    # filter retains mirror at a rate BELOW its overall retention rate, and
    # per-fold lift > 1 in ≤ 3 of 7 folds. Contradiction = either of those
    # signals reverses (lift > 1 in 5+ folds, OR mirror retained at a
    # meaningfully higher rate than the filter's overall retention).
    receipts.append("## GBP contradiction check")
    for r in rows:
        if r["slug"] != "filter_basket_gbp_above_p50":
            continue
        n_sign = r.get("_n_sign_consistent", 0)
        wrr = r.get("_winner_retention_rate_f", float("nan"))
        n_post = int(r["n_post_total"])
        overall_retention = n_post / EXPECTED_POOL_TOTAL
        receipts.append(
            f"filter_basket_gbp_above_p50: "
            f"sign_consistency={n_sign}/7, "
            f"winner_retention_rate={wrr:.4f}, "
            f"overall_retention_rate={overall_retention:.4f}"
        )
        contradiction = (n_sign >= 5) or (wrr > overall_retention * 1.05)
        if contradiction:
            receipts.append("CONTRADICTION: GBP supplemental columns disagree with v1.2")
            receipts.append("disposition ('concentration in wrong direction').")
            receipts.append("Halting WITHOUT writing v1_2_supplemental_columns.csv.")
            receipts.append("Chat must re-read before regenerating.")
            OUT_RECEIPTS.write_text("\n".join(receipts) + "\n", encoding="utf-8")
            print("\n".join(receipts))
            raise SystemExit(2)
    receipts.append("GBP contradiction check passed (no halt).")
    receipts.append("")

    # --- Sort alphabetically by slug, drop internals, write CSV ---
    rows_sorted = sorted(rows, key=lambda r: r["slug"])
    assert len(rows_sorted) == 11, f"expected 11 rows, got {len(rows_sorted)}"

    csv_lines = [",".join(OUT_SCHEMA)]
    for r in rows_sorted:
        cells = [str(r.get(col, "")) for col in OUT_SCHEMA]
        # CSV safety: quote if a value contains a comma
        cells = [f'"{c}"' if "," in c else c for c in cells]
        csv_lines.append(",".join(cells))
    csv_body = "\n".join(csv_lines)  # NO trailing newline per dispatch
    OUT_CSV.write_bytes(csv_body.encode("utf-8"))

    out_sha = sha256_file(OUT_CSV)
    wall = time.time() - t0
    receipts.append(f"## Output")
    receipts.append(f"path: {OUT_CSV.relative_to(REPO).as_posix()}")
    receipts.append(f"sha256: {out_sha}")
    receipts.append(f"rows: 11 + header")
    receipts.append(f"wall_time_seconds: {wall:.3f}")
    receipts.append("")
    if notes_acc:
        receipts.append("## Per-candidate lift traces (debug)")
        for ln in notes_acc:
            receipts.append(ln)
        receipts.append("")

    OUT_RECEIPTS.write_text("\n".join(receipts) + "\n", encoding="utf-8")

    print("\n".join(receipts))
    return 0


if __name__ == "__main__":
    sys.exit(main())

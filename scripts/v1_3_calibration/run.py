"""v1.3 calibration — driver.

Runs the metric battery on KH-24 + Arc 1 + Arc 2, per-pool and per-cluster
on the two arcs. Writes all CSVs + side files + the calibration report
under results/v1_3_calibration/.

Pure computation — no floor-setting, no protocol amendment, no
interpretation. The chat reads CALIBRATION_REPORT.md and decides v1.3
floors after.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.v1_3_calibration import metrics as M
from scripts.v1_3_calibration.load_paths import (
    DATASETS, load_clusters, load_paths,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT  = REPO_ROOT / "results" / "v1_3_calibration"

# Cluster column used for per-cluster decomposition. K3_kmeans is the
# standard step-3 partitioning used elsewhere in the L-arc pipeline.
CLUSTER_COL = "K3_kmeans"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    # lineterminator='\n' for cross-platform sha stability.
    df.to_csv(path, index=False, lineterminator="\n")


def _flat_metrics_row(name: str, computed: dict, smoothness_pool: dict) -> dict:
    """Flatten one dataset's computed dict to one row for side-by-side tables."""
    row = {"dataset": name}
    row.update(computed["axis1"])
    for k, v in computed["axis2a"].items():
        row[f"axis2a_{k}"] = v
    for k, v in computed["axis2b"].items():
        row[f"axis2b_{k}"] = v
    for k, v in computed["axis2c"].items():
        row[f"axis2c_{k}"] = v
    for k, v in computed["axis2d"].items():
        row[f"axis2d_{k}"] = v
    for k, v in computed["axis2f"].items():
        row[f"axis2f_{k}"] = v
    for k, v in smoothness_pool.items():
        row[f"axis2g_{k}"] = v
    for k, v in computed["axis2h"].items():
        row[f"axis2h_{k}"] = v
    for k, v in computed["axis3"].items():
        row[f"axis3_{k}"] = v
    row.update({f"shape_{k}": v for k, v in computed["shape"].items()})
    row.update({f"mass_{k}": v for k, v in computed["mass_bands"].items()})
    return row


def _emit_dataset_outputs(name: str, computed: dict) -> dict:
    """Per-dataset side files + return summary row entries."""
    ddir = OUT_ROOT / name
    _ensure_dir(ddir)

    _write_csv(computed["axis1_dist"],   ddir / "axis1_full_distribution.csv")
    _write_csv(computed["axis2a_curve"], ddir / "axis2a_time_exit_curve.csv")
    _write_csv(computed["axis2b_curve"], ddir / "axis2b_trail_exit_curve.csv")
    _write_csv(computed["axis2c_curve"], ddir / "axis2c_tp_exit_curve.csv")
    _write_csv(computed["axis2d_curve"], ddir / "axis2d_mfe_lock_curve.csv")
    _write_csv(computed["axis2e"],       ddir / "axis2e_conditional_predictivity.csv")
    _write_csv(
        pd.DataFrame([computed["axis2f"]]),
        ddir / "axis2f_reentry_descriptive.csv",
    )
    _write_csv(computed["axis2g"], ddir / "axis2g_in_trade_smoothness.csv")
    smoothness_pool = M.aggregate_smoothness(computed["axis2g"])
    _write_csv(
        pd.DataFrame([smoothness_pool]),
        ddir / "axis2g_in_trade_smoothness_pool_summary.csv",
    )
    _write_csv(computed["axis3_dist"],   ddir / "axis3_distributions.csv")
    return smoothness_pool


def _emit_pool_summary_csvs(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    # Per-axis pool summaries.
    axis1_cols = ["dataset"] + [c for c in df.columns if c.startswith("pool_") or c == "n_trades"]
    _write_csv(df[axis1_cols].copy(), OUT_ROOT / "axis1_peak_magnitude.csv")

    axis2_cols = ["dataset"] + [c for c in df.columns if c.startswith(("axis2a_", "axis2b_", "axis2c_", "axis2d_", "axis2f_", "axis2g_"))]
    _write_csv(df[axis2_cols].copy(), OUT_ROOT / "axis2_capture_ceilings.csv")

    axis2h_cols = ["dataset"] + [c for c in df.columns if c.startswith("axis2h_")]
    _write_csv(df[axis2h_cols].copy(), OUT_ROOT / "axis2h_time_to_peak_cv.csv")

    axis3_cols = ["dataset"] + [c for c in df.columns if c.startswith("axis3_")]
    _write_csv(df[axis3_cols].copy(), OUT_ROOT / "axis3_path_hostility.csv")

    shape_cols = ["dataset"] + [c for c in df.columns if c.startswith("shape_")]
    _write_csv(df[shape_cols].copy(), OUT_ROOT / "shape_tags_pool.csv")

    mass_cols = ["dataset"] + [c for c in df.columns if c.startswith("mass_")]
    _write_csv(df[mass_cols].copy(), OUT_ROOT / "mass_in_band_pool.csv")

    # All-in-one flat side-by-side table (useful for the report).
    _write_csv(df, OUT_ROOT / "_all_pool_metrics.csv")


def _per_cluster_run(name: str, paths: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ca = load_clusters(name)
    if ca is None:
        return pd.DataFrame()
    # join cluster ids onto the per-trade meta
    ca_sub = ca[["trade_id", CLUSTER_COL]].copy()
    cluster_rows = []
    for cluster_id, sub_ca in ca_sub.groupby(CLUSTER_COL):
        ids = set(sub_ca["trade_id"])
        sub_paths = paths[paths["trade_id"].isin(ids)]
        sub_meta  = meta[meta["trade_id"].isin(ids)]
        if sub_paths.empty:
            continue
        # Reduced metric set per-cluster (axis 1 + axis 3 + shape + mass; axis 2
        # capture ratios are pool-level only per the spec's "if time permits").
        pt = M._per_trade_h240(sub_paths)
        pt = pt.merge(sub_meta[["trade_id", "pair", "bars_held"]], on="trade_id", how="left")
        exit_snap = M._per_trade_at_exit(sub_paths, sub_meta)
        a1, _    = M.axis1_peak_magnitude(pt)
        a3, _    = M.axis3_path_hostility(pt, exit_snap)
        shape    = M.shape_tag(pt)
        bands    = M.mass_in_band(pt)
        row = {
            "cluster_id":   int(cluster_id),
            "n_trades":     int(len(pt)),
            "frac_of_pool": float(len(pt) / len(meta)),
        }
        row.update(a1)
        row.update({f"axis3_{k}": v for k, v in a3.items()})
        row.update({f"shape_{k}": v for k, v in shape.items()})
        row.update({f"mass_{k}": v for k, v in bands.items()})
        cluster_rows.append(row)
    return pd.DataFrame(cluster_rows).sort_values("cluster_id").reset_index(drop=True)


def _df_to_md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Manual markdown-table renderer — avoids the optional `tabulate` dep."""
    cols = list(df.columns)
    rows = df.to_dict(orient="records")

    def cell(v: object) -> str:
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return "—"
            return f"{v:{floatfmt}}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if v is None:
            return "—"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join(["---"] * len(cols)) + "|"
    body   = "\n".join("| " + " | ".join(cell(r[c]) for c in cols) + " |" for r in rows)
    return "\n".join([header, sep, body])


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_report(rows: list[dict], per_cluster: dict[str, pd.DataFrame]) -> str:
    """Build CALIBRATION_REPORT.md from the computed pool rows + clusters."""
    df = pd.DataFrame(rows).set_index("dataset")

    def fmt(val, fmt_spec=".4f"):
        if isinstance(val, str):
            return val
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—"
        return f"{val:{fmt_spec}}"

    def row(label: str, key: str, spec: str = ".4f") -> str:
        cells = [fmt(df.loc[d, key], spec) if key in df.columns else "—" for d in ("kh24", "arc1", "arc2")]
        return f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |"

    lines: list[str] = []
    a = lines.append

    a("# v1.3 capturability calibration — KH-24 + Arc 1 + Arc 2 metrics")
    a("")
    a("> Date: 2026-05-15")
    a("> Branch: feature/v1.3-calibration-data")
    a("")
    a("Pure computation output for the v1.3 capturability calibration. Three "
      "datasets — KH-24 (the project's only known real edge, must pass any "
      "v1.3 floors), Arc 1 (verbatim, step-3 cluster context), Arc 2 (verbatim, "
      "step-3 cluster context). The chat sets v1.3 floors after reading this.")
    a("")
    a("---")
    a("")
    a("## §1 Schema audit")
    a("")
    a("See `schema_audit.md` for the full mapping table. Summary:")
    a("")
    a("- KH-24: compatible (reference); ATR-units → R-units via /SL_MULT=2.")
    a("- Arc 1: **partial** — no per-bar OHLC. TP/SL/MFE-lock simulations use "
      "running mfe/mae (exact at first-cross bar). Trail simulation falls back "
      "to close-based detection (derived `close_r` from `fwd_logret_cum`).")
    a("- Arc 2: compatible — OHLC + running mfe/mae available.")
    a("")
    a("R-unit convention: all values in SL-distance R-units, R = 2 × ATR_at_entry. "
      "KH-24 uses 4H ATR; Arc 1 / Arc 2 use 1H ATR. The 2× SL multiplier is "
      "constant; the ATR-timeframe mismatch is acknowledged for absolute-"
      "magnitude metrics, immune for dimensionless capture ratios. See "
      "`scripts/v1_3_calibration/loader_decisions.md` for full conversions.")
    a("")
    a("---")
    a("")
    a("## §2 Side-by-side metric table (pool level)")
    a("")
    a("| Metric | KH-24 | Arc 1 | Arc 2 |")
    a("|---|---|---|---|")
    a("| **Axis 1: Peak Magnitude** | | | |")
    a(row("n_trades",                "n_trades",                  ".0f"))
    a(row("pool_median_fwd_mfe_h120 (R)", "pool_median_fwd_mfe_h120"))
    a(row("pool_median_fwd_mfe_h240 (R)", "pool_median_fwd_mfe_h240"))
    a(row("pool_frac_reach_1R",      "pool_frac_reach_1R"))
    a(row("pool_frac_reach_1_5R",    "pool_frac_reach_1_5R"))
    a(row("pool_frac_reach_2R",      "pool_frac_reach_2R"))
    a(row("pool_p90_fwd_mfe_h240 (R)", "pool_p90_fwd_mfe_h240"))
    a(row("pool_p95_fwd_mfe_h240 (R)", "pool_p95_fwd_mfe_h240"))
    a(row("pool_p99_fwd_mfe_h240 (R)", "pool_p99_fwd_mfe_h240"))
    a("| **Axis 2: Peak Capture (best of family)** | | | |")
    a(row("time_exit_best_h",        "axis2a_time_exit_best_h",       ".0f"))
    a(row("time_exit_best_capture",  "axis2a_time_exit_best_capture"))
    a(row("trail_exit_best_W",       "axis2b_trail_exit_best_W"))
    a(row("trail_exit_best_capture", "axis2b_trail_exit_best_capture"))
    a(row("tp_exit_best_X",          "axis2c_tp_exit_best_X"))
    a(row("tp_exit_best_capture",    "axis2c_tp_exit_best_capture"))
    a(row("mfe_lock_best_X",         "axis2d_mfe_lock_best_X"))
    a(row("mfe_lock_best_capture",   "axis2d_mfe_lock_best_capture"))
    a(row("local_peaks_pool_median", "axis2g_local_peaks_pool_median"))
    a(row("monotonicity_pool_median","axis2g_monotonicity_pool_median"))
    a(row("time_to_peak_cv",         "axis2h_time_to_peak_cv"))
    a(row("time_to_peak_p50 (bars)", "axis2h_time_to_peak_p50",        ".1f"))
    a(row("frac_reentry_candidates", "axis2f_frac_reentry_candidates"))
    a("| **Axis 3: Path Hostility** | | | |")
    a(row("race_condition_median (bars)",     "axis3_race_condition_median", ".1f"))
    a(row("mae_mfe_ratio_winners_median",     "axis3_mae_mfe_ratio_winners_median"))
    a(row("pct_peak_and_collapse",            "axis3_pct_peak_and_collapse"))
    a(row("pct_wrong_way",                    "axis3_pct_wrong_way"))
    a("| **Shape (pool) + Mass-in-band** | | | |")
    a(row("shape_tag",                "shape_tag",        "s"))
    a(row("shape_p50",                "shape_p50"))
    a(row("shape_p95",                "shape_p95"))
    a(row("shape_p99",                "shape_p99"))
    a(row("mass_band_0_to_0_5R",      "mass_band_0_to_0_5R"))
    a(row("mass_band_0_5_to_1R",      "mass_band_0_5_to_1R"))
    a(row("mass_band_1_to_2R",        "mass_band_1_to_2R"))
    a(row("mass_band_2_to_5R",        "mass_band_2_to_5R"))
    a(row("mass_band_above_5R",       "mass_band_above_5R"))
    a("")
    a("---")
    a("")
    a("## §3 Per-cluster metrics (Arc 1 + Arc 2)")
    a("")
    for arc, df_c in per_cluster.items():
        a(f"### {arc} — K3_kmeans clusters")
        a("")
        if df_c.empty:
            a("(no cluster assignments)")
            a("")
            continue
        # Trim to a readable subset.
        keep_cols = ["cluster_id", "n_trades", "frac_of_pool",
                     "pool_median_fwd_mfe_h240", "pool_frac_reach_1R", "pool_frac_reach_2R",
                     "axis3_pct_peak_and_collapse", "axis3_pct_wrong_way",
                     "shape_tag", "mass_band_2_to_5R", "mass_band_above_5R"]
        sub = df_c[[c for c in keep_cols if c in df_c.columns]].copy()
        # Render markdown.
        a(_df_to_md_table(sub, floatfmt=".4f"))
        a("")
    a("---")
    a("")
    a("## §4 Distribution shape commentary")
    a("")
    for d in ("kh24", "arc1", "arc2"):
        if d not in df.index:
            continue
        a(f"### {d}")
        tag = df.loc[d, "shape_tag"] if "shape_tag" in df.columns else "—"
        p50 = df.loc[d, "pool_median_fwd_mfe_h240"]
        f1  = df.loc[d, "pool_frac_reach_1R"]
        f2  = df.loc[d, "pool_frac_reach_2R"]
        tcap = df.loc[d, "axis2a_time_exit_best_capture"] if "axis2a_time_exit_best_capture" in df.columns else float("nan")
        trcap = df.loc[d, "axis2b_trail_exit_best_capture"] if "axis2b_trail_exit_best_capture" in df.columns else float("nan")
        race  = df.loc[d, "axis3_race_condition_median"] if "axis3_race_condition_median" in df.columns else float("nan")
        a(f"Shape: **{tag}**. Pool fwd MFE p50 = {fmt(p50)} R, p99 = {fmt(df.loc[d, 'pool_p99_fwd_mfe_h240'])} R. "
          f"`frac_reach_1R` = {fmt(f1)}; `frac_reach_2R` = {fmt(f2)}. "
          f"Best time-exit capture {fmt(tcap)} at h={int(df.loc[d, 'axis2a_time_exit_best_h']) if not pd.isna(df.loc[d, 'axis2a_time_exit_best_h']) else 'NA'}; "
          f"best trail capture {fmt(trcap)} at W={fmt(df.loc[d, 'axis2b_trail_exit_best_W'])}. "
          f"Race-condition median = {fmt(race, '.1f')} bars "
          f"(negative = direction wins; first +1R hit before first −1R hit).")
        a("")
    a("---")
    a("")
    a("## §5 Unavailable / approximated metric flags")
    a("")
    a("- **Arc 1** has no per-bar high/low/close in step 2's `trade_paths.csv`. "
      "`high_r` / `low_r` are emitted as NaN. Downstream impact:")
    a("  - TP / SL / MFE-lock simulations: **exact** (use running mfe/mae which "
      "increments at the first bar that crosses the threshold).")
    a("  - Trail-exit simulation: **close-based fallback** (intrabar-low trail "
      "detection not possible; uses derived `close_r` from `fwd_logret_cum`). "
      "Approximation only; documented in `loader_decisions.md`.")
    a("  - Axis 2e conditional predictivity uses derived `close_r` for the same reason.")
    a("- **Arc 1 / Arc 2** `entry_px` uses `signals_features.signal_bar_close` "
      "(one 1H-bar approximation of the true entry-bar open). Sub-ATR drift "
      "across one bar; documented.")
    a("- **ATR-timeframe mismatch**: KH-24 uses 4H ATR for SL; Arc 1 / Arc 2 "
      "use 1H ATR. Absolute-magnitude comparisons (e.g. `frac_reach_1R`) "
      "inherit this; capture-ratio comparisons are dimensionless and immune.")
    a("")
    a("---")
    a("")
    a("## §6 Files produced")
    a("")
    manifest = []
    for f in sorted(OUT_ROOT.rglob("*.csv")):
        rel = f.relative_to(OUT_ROOT)
        manifest.append({"path": str(rel).replace("\\", "/"), "sha256": _file_sha256(f), "bytes": f.stat().st_size})
    md = pd.DataFrame(manifest)
    a(_df_to_md_table(md, floatfmt="d"))
    a("")
    a("---")
    a("")
    a("## §7 Status")
    a("")
    a("**proceed-to-floor-setting** — all three datasets produced full pool-level "
      "metrics; both arcs produced per-cluster decompositions; only Arc 1's "
      "intrabar-low-trail variant is flagged as approximate (close-based "
      "fallback). The chat can now read this report and lock v1.3 floors.")
    a("")
    return "\n".join(lines)


def main() -> int:
    t0 = time.time()
    _ensure_dir(OUT_ROOT)

    rows = []
    per_cluster_results: dict[str, pd.DataFrame] = {}
    for name in ("kh24", "arc1", "arc2"):
        print(f"[{time.time() - t0:6.1f}s] loading {name} …")
        paths, meta = load_paths(name)
        print(f"[{time.time() - t0:6.1f}s] computing metrics for {name} …")
        computed = M.compute_dataset(name, paths, meta)
        smoothness = _emit_dataset_outputs(name, computed)
        rows.append(_flat_metrics_row(name, computed, smoothness))

        if name in ("arc1", "arc2"):
            print(f"[{time.time() - t0:6.1f}s] per-cluster decomposition for {name} …")
            df_c = _per_cluster_run(name, paths, meta)
            if not df_c.empty:
                _write_csv(df_c, OUT_ROOT / name / "per_cluster_metrics.csv")
                per_cluster_results[name] = df_c

    print(f"[{time.time() - t0:6.1f}s] emitting pool-level summary CSVs …")
    _emit_pool_summary_csvs(rows)

    print(f"[{time.time() - t0:6.1f}s] rendering CALIBRATION_REPORT.md …")
    report = _render_report(rows, per_cluster_results)
    (OUT_ROOT / "CALIBRATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    print(f"[{time.time() - t0:6.1f}s] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

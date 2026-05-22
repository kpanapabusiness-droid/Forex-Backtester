"""Arc 8 Step 3 — capturability.

Per L_PROTOCOL §2 Step 3 + dispatch §"Step 3":

  Per cluster:
    - Reach rates P(MFE >= 1R, 2R, 3R)
    - MFE distribution p25/p50/p75/p90
    - Wrong-way path prevalence ww_pp = P(MAE hits -1R before MFE hits +1R)
    - Time-to-peak distribution
    - Mean R, p25 R, p50 R
    - Capturability composite (ranking aid only)
  SL sweep per cluster: SL in {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} x ATR (dispatch chat F3)
  Candidate flag: reach_1R >= 0.50 AND ww_pp <= 0.30 AND mfe_p50 >= 1.5R
  Explicit archetype label per cluster (drives Step 5 architecture selection).

Per-cluster R denomination at swept SL:
  Step 1 SL was 2.0 x ATR. final_r at SL=2.0 is recorded; at swept SL=s the
  equivalent R is the same dollar PnL divided by (s/2.0) x original_sl_distance.
  We approximate by re-deriving SL fire / no-fire from MFE/MAE in ORIGINAL R
  units, then rescaling: at SL multiplier s the SL distance = s/2.0 in original
  R units. A trade SL-fires under candidate SL iff its MAE in original R is
  <= -(s/2.0). If it doesn't SL-fire, final_r at the new SL is identical (we
  still hold to time-exit at the same close) but scaled by 2.0/s. If it does
  SL-fire under the new SL, final_r becomes -1 in the new R units.

Reads ``step_1/pool.parquet`` + ``step_2/cluster_assignments.parquet``.
Writes ``step_3/capturability.csv``, ``capturability_summary.md``,
``sl_sweep.csv``, ``manifest.json``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.determinism import seed_everything, write_text_deterministic
from core.manifest import write_manifest
from scripts.l_arc_8.shared import RESULTS_ROOT, SL_ATR_MULT_STEP1, SL_SWEEP

STEP_DIR: Path = RESULTS_ROOT / "step_3"


def _rescale_outcome_at_sl(
    final_r_orig: float,
    mfe_r_orig: float,
    mae_r_orig: float,
    sl_mult_new: float,
    sl_mult_orig: float = SL_ATR_MULT_STEP1,
) -> tuple[float, float, float, str]:
    """Re-evaluate (final_r, mfe_r, mae_r) under a different SL multiplier.

    All inputs are in ORIGINAL R units (Step 1 SL = sl_mult_orig x ATR).
    Returns (final_r_new, mfe_r_new, mae_r_new, exit_reason_new) where
    new-R units are denominated against sl_mult_new x ATR.

    Logic:
      Scale factor f = sl_mult_orig / sl_mult_new. New R = old R * f.
      SL line in new R: -1. In old R: -sl_mult_new / sl_mult_orig = -1/f.
      If mae_r_orig <= -1/f: SL fires under new SL. final_r_new = -1.
      Else: final_r_new = final_r_orig * f (same exit price, rescaled).
      mfe/mae rescale by f when SL doesn't fire. When SL fires under new SL,
      we lose forward-path detail — conservatively mfe_new=min(mfe*f, ...)
      and we use the SL line as mae_new=-1. (Not exact but a faithful
      approximation given we lack per-bar timing of MFE vs MAE.)
    """
    f = sl_mult_orig / sl_mult_new
    sl_orig_line = -1.0 / f  # in original R
    if mae_r_orig <= sl_orig_line:
        return -1.0, mfe_r_orig * f, -1.0, "sl"
    return final_r_orig * f, mfe_r_orig * f, mae_r_orig * f, "time_exit"


def _wrong_way_prevalence(mfe_r: pd.Series, mae_r: pd.Series) -> float:
    """ww_pp = P(MAE hits -1R before MFE hits +1R).

    With only summary MFE/MAE (no per-bar timing), approximate:
      - if mfe < 1 and mae <= -1 -> wrong-way for sure (1)
      - if mfe >= 1 and mae > -1 -> right-way for sure (0)
      - if both -> need timing; treat as wrong-way if mae timing precedes
        mfe timing. Without explicit timing, use a magnitude proxy: classify
        as wrong-way when |mae| >= mfe (the larger excursion happened first
        on average).

    This is conservative — over-estimates ww_pp when MFE and MAE both exceed
    1R; in pure trend-resume the bias may understate edge slightly.
    """
    n = len(mfe_r)
    if n == 0:
        return 0.0
    a = np.asarray(mfe_r, dtype=float)
    b = np.asarray(mae_r, dtype=float)
    wrong = (
        ((a < 1.0) & (b <= -1.0))                 # never reached +1R, hit -1R
        | ((a >= 1.0) & (b <= -1.0) & (-b >= a))  # both, but MAE magnitude >= MFE
    )
    return float(wrong.mean())


def _capturability_metrics(
    cluster_pool: pd.DataFrame,
    sl_mult_new: float,
) -> dict:
    """Per-cluster metrics under a candidate SL multiplier."""
    rescaled = cluster_pool.apply(
        lambda r: _rescale_outcome_at_sl(
            float(r["final_r"]), float(r["mfe_r"]), float(r["mae_r"]), sl_mult_new
        ),
        axis=1, result_type="expand",
    )
    rescaled.columns = ["final_r", "mfe_r", "mae_r", "exit_reason"]
    fr = rescaled["final_r"]
    mfe = rescaled["mfe_r"]
    mae = rescaled["mae_r"]
    n = len(rescaled)
    ww_pp = _wrong_way_prevalence(mfe, mae)
    reach_1r = float((mfe >= 1.0).mean())
    reach_2r = float((mfe >= 2.0).mean())
    reach_3r = float((mfe >= 3.0).mean())
    mfe_p50 = float(mfe.quantile(0.50))
    # capturability composite: ranking aid only — weighted aggregate
    # 0.4 * reach_1R + 0.4 * (1 - ww_pp) + 0.2 * min(mfe_p50/3, 1.0)
    composite = (
        0.4 * reach_1r
        + 0.4 * (1.0 - ww_pp)
        + 0.2 * min(mfe_p50 / 3.0, 1.0)
    )
    return {
        "n": n,
        "sl_mult": sl_mult_new,
        "reach_1R": reach_1r,
        "reach_2R": reach_2r,
        "reach_3R": reach_3r,
        "ww_pp": ww_pp,
        "mfe_p25": float(mfe.quantile(0.25)),
        "mfe_p50": mfe_p50,
        "mfe_p75": float(mfe.quantile(0.75)),
        "mfe_p90": float(mfe.quantile(0.90)),
        "mean_r": float(fr.mean()),
        "p25_r": float(fr.quantile(0.25)),
        "p50_r": float(fr.quantile(0.50)),
        "p75_r": float(fr.quantile(0.75)),
        "sl_fire_rate": float((rescaled["exit_reason"] == "sl").mean()),
        "capturability_composite": composite,
    }


def _cluster_archetype(metrics_df_per_cluster: pd.DataFrame, modal_tag: str) -> str:
    """Pick the dispatch's archetype taxonomy from cluster characteristics.

    Maps the Step 2 modal shape_tag and per-cluster Step 3 metrics into
    one of {Stepwise climber, V-shape recovery, Bimodal, Monotonic up,
    Monotonic down, Choppy}. Drives Step 5 architecture selection per
    dispatch §"Step 5" archetype map.
    """
    # Modal_tag is already one of these labels (or 'Mixed' fallback).
    if modal_tag in {"V-shape recovery", "Stepwise climber", "Monotonic up",
                     "Monotonic down", "Bimodal", "Choppy"}:
        return modal_tag
    # 'Mixed' fallback — derive from metrics
    if metrics_df_per_cluster["mean_r"].mean() < -0.3 and metrics_df_per_cluster["sl_fire_rate"].mean() > 0.5:
        return "Monotonic down"
    if metrics_df_per_cluster["mfe_p50"].mean() >= 1.5 and metrics_df_per_cluster["reach_1R"].mean() >= 0.5:
        return "V-shape recovery"
    return "Choppy"


def main() -> Path:
    seed_everything(42)
    t0 = time.perf_counter()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    step1 = RESULTS_ROOT / "step_1"
    step2 = RESULTS_ROOT / "step_2"
    pool = pd.read_parquet(step1 / "pool.parquet")
    feats = pd.read_parquet(step2 / "path_features.parquet")
    assignments = pd.read_parquet(step2 / "cluster_assignments.parquet")
    metrics_step2 = pd.read_csv(step2 / "cluster_metrics.csv")
    # Join cluster_primary to pool by trade_id
    pool = pool.merge(
        assignments[["trade_id", "cluster_primary", "shape_tag"]],
        on="trade_id", how="left", validate="one_to_one",
    )
    n_clusters = int(pool["cluster_primary"].nunique())
    print(f"[step3] pool merged: {len(pool)} trades, {n_clusters} clusters")

    # Sweep SL per cluster
    sweep_rows: list[dict] = []
    for cid in sorted(pool["cluster_primary"].unique()):
        cpool = pool[pool["cluster_primary"] == cid]
        for sl in SL_SWEEP:
            m = _capturability_metrics(cpool, sl)
            m["cluster_id"] = int(cid)
            sweep_rows.append(m)
    sweep_df = pd.DataFrame(sweep_rows).sort_values(["cluster_id", "sl_mult"]).reset_index(drop=True)
    sweep_path = STEP_DIR / "sl_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False, lineterminator="\n")

    # Best SL per cluster (by capturability composite, tie-break: smaller SL)
    best_per_cluster: list[dict] = []
    for cid in sorted(pool["cluster_primary"].unique()):
        cdf = sweep_df[sweep_df["cluster_id"] == cid].sort_values(
            ["capturability_composite", "sl_mult"], ascending=[False, True]
        )
        best = cdf.iloc[0].to_dict()
        # Archetype label
        modal_tag = metrics_step2[metrics_step2["cluster_id"] == cid]["modal_shape_tag"].iloc[0]
        best["modal_shape_tag"] = modal_tag
        best["archetype"] = _cluster_archetype(cdf, modal_tag)
        # Candidate flag
        best["candidate"] = bool(
            (best["reach_1R"] >= 0.50)
            and (best["ww_pp"] <= 0.30)
            and (best["mfe_p50"] >= 1.5)
        )
        best_per_cluster.append(best)
    cap_df = pd.DataFrame(best_per_cluster)
    cap_path = STEP_DIR / "capturability.csv"
    cap_df.to_csv(cap_path, index=False, lineterminator="\n")
    print(f"[step3] best SL per cluster: {cap_df[['cluster_id','sl_mult','archetype','candidate']].to_dict('records')}")

    # Capturability summary md
    summary = _build_summary_md(cap_df=cap_df, sweep_df=sweep_df, metrics_step2=metrics_step2)
    summary_path = STEP_DIR / "capturability_summary.md"
    write_text_deterministic(summary_path, summary)

    write_manifest(STEP_DIR / "manifest.json", artefacts=[sweep_path, cap_path, summary_path])
    elapsed = time.perf_counter() - t0
    print(f"[step3] DONE in {elapsed:.1f}s — {int(cap_df['candidate'].sum())} candidate cluster(s)")
    return STEP_DIR


def _build_summary_md(cap_df: pd.DataFrame, sweep_df: pd.DataFrame, metrics_step2: pd.DataFrame) -> str:
    lines = [
        "# Arc 8 — Step 3 Capturability Summary",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
        f"SL sweep: {list(SL_SWEEP)} x ATR (dispatch §\"Step 3\", chat F3).",
        f"Original Step 1 SL: {SL_ATR_MULT_STEP1} x ATR.",
        "",
        "## Per-cluster best-SL metrics",
        "",
        "| Cluster | Archetype (modal tag) | n | Best SL | Reach 1R | Reach 2R | Reach 3R | ww_pp | MFE p50 | Mean R | p50 R | SL-fire rate | Composite | Candidate? |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, r in cap_df.iterrows():
        cand = "YES" if r["candidate"] else "no"
        lines.append(
            f"| {int(r['cluster_id'])} "
            f"| {r['archetype']} ({r['modal_shape_tag']}) "
            f"| {int(r['n']):,} "
            f"| {r['sl_mult']:.1f} "
            f"| {r['reach_1R']:.2%} "
            f"| {r['reach_2R']:.2%} "
            f"| {r['reach_3R']:.2%} "
            f"| {r['ww_pp']:.2%} "
            f"| {r['mfe_p50']:+.3f} "
            f"| {r['mean_r']:+.3f} "
            f"| {r['p50_r']:+.3f} "
            f"| {r['sl_fire_rate']:.2%} "
            f"| {r['capturability_composite']:.3f} "
            f"| **{cand}** |"
        )

    n_cand = int(cap_df["candidate"].sum())
    lines += [
        "",
        f"**Candidate clusters:** {n_cand} (pass: reach_1R >= 0.50 AND ww_pp <= 0.30 AND mfe_p50 >= 1.5R)",
        "",
        "## Full SL sweep per cluster",
        "",
        "| Cluster | SL | Reach 1R | Reach 2R | ww_pp | MFE p50 | Mean R | SL-fire rate | Composite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in sweep_df.iterrows():
        lines.append(
            f"| {int(r['cluster_id'])} | {r['sl_mult']:.1f} "
            f"| {r['reach_1R']:.2%} | {r['reach_2R']:.2%} "
            f"| {r['ww_pp']:.2%} | {r['mfe_p50']:+.3f} "
            f"| {r['mean_r']:+.3f} | {r['sl_fire_rate']:.2%} "
            f"| {r['capturability_composite']:.3f} |"
        )

    lines += [
        "",
        "## Archetype taxonomy → Step 5 architecture selection",
        "",
        "Per dispatch §\"Step 5\" archetype map:",
        "- **Stepwise climber** → A1, A2, A4",
        "- **V-shape recovery** → A1, A3, A6",
        "- **Bimodal** → A1, A4",
        "- **Monotonic up** → A1, A2, A6",
        "- **Choppy** → no architectures; document and skip",
        "- **A5 (portfolio composition)** → only if >= 2 candidate clusters survive",
        "",
        "## Methodology notes",
        "",
        "- Per-cluster MFE / MAE / final_R were rescaled from the Step-1 sim (SL=2.0xATR) "
        "by the closed-form rule in `_rescale_outcome_at_sl`: at swept SL multiplier s, "
        "trade SL-fires iff `mae_r_orig <= -1/(s/2.0)`; otherwise the original exit "
        "carries through with R units multiplied by `2.0/s`.",
        "- `ww_pp` is approximated from summary MFE/MAE without per-bar timing — when "
        "both MFE and MAE exceed 1R, classified as wrong-way iff `|MAE| >= MFE` "
        "(conservative — over-estimates wrong-way on bimodal paths).",
        "- Capturability composite = `0.4 * reach_1R + 0.4 * (1 - ww_pp) + 0.2 * min(mfe_p50/3, 1)`. "
        "Ranking aid only, never a gate.",
        "",
    ]
    if n_cand == 0:
        lines.append("**FLAG:** Zero candidate clusters. Step 4 proceeds on the highest-composite cluster per L_PROTOCOL §2 Step 3 failure-diagnostics path.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

"""Arc 11 — Step 3 capturability + SL sweep (L_PROTOCOL v3.0 §2 Step 3).

Inputs:
  - results/l_arc_11/step_1/pool.parquet
  - results/l_arc_11/step_1/trades_paths.parquet
  - results/l_arc_11/step_2/cluster_assignments.parquet
  - results/l_arc_11/step_2/path_features.parquet

Per cluster:
  - Reach rates (P(MFE >= 1R / 2R / 3R)), MFE p25/p50/p75/p90,
    wrong-way path prevalence (ww_pp), time-to-peak, mean/p25/p50 R.
  - Capturability composite = weighted aggregate of reach_1R, mfe_p50,
    (1 - ww_pp). Ranking aid only.
  - Candidate cluster flag: reach_1R >= 0.50 AND ww_pp <= 0.30 AND mfe_p50 >= 1.5R.
  - SL sweep over {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} x ATR. Recompute
    final_r per trade at each SL by rescaling the paths' high_r/low_r
    (the path series were generated at SL=2.0 x ATR).
  - Archetype label per cluster derived from path-shape distribution
    (V-shape recovery, Stepwise climber, Bimodal, Monotonic up, Choppy).

Outputs:
  - results/l_arc_11/step_3/capturability.csv         (per-cluster, per-SL)
  - results/l_arc_11/step_3/capturability_summary.md
  - results/l_arc_11/step_3/per_cluster_archetype.csv
  - results/l_arc_11/step_3/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.determinism import RANDOM_STATE, seed_everything
from scripts.l_arc_11.common import (
    REPO_ROOT,
    load_config,
    results_root,
    sha256_file,
    write_manifest,
)


SL_BASE_MULTIPLIER = 2.0   # Step 1 pool was simulated at 2.0xATR


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[arc_11 step_3 {ts}] {msg}", flush=True)


# ─── SL-sweep R recompute per trade ────────────────────────────────


def recompute_R_at_sl(
    paths_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    new_sl_multiplier: float,
) -> pd.DataFrame:
    """Recompute final_r, mfe_r, mae_r per trade if the SL were ``new_sl_multiplier``
    instead of the base 2.0xATR.

    The path R values are normalised by sl_distance_price (at base SL=2.0xATR).
    Recovered raw-price ratio = R_old * (base / new); equivalently price diffs
    scale by (base / new). Then check if a new SL = -1.0 in NEW R-space is hit
    (i.e. low_r_new <= -1.0). If hit, final_r at the hit bar = -1.0; else, the
    trade closes at time_exit (open_bid normalised same way).

    Returns DataFrame: trade_id, pair, sl_multiplier, final_r, mfe_r, mae_r,
    exit_reason, bars_held, year
    """
    scale = SL_BASE_MULTIPLIER / float(new_sl_multiplier)
    rows = []
    # Pool gives us original exit_reason, calendar_year, original bars_held, sl_distance_price.
    pool_by_id = pool_df.set_index("trade_id")
    held_paths = paths_df[paths_df["is_held"] == 1]
    for trade_id, g in held_paths.groupby("trade_id"):
        if trade_id not in pool_by_id.index:
            continue
        prow = pool_by_id.loc[trade_id]
        g = g.sort_values("bar_offset")
        # Rescale R series
        high_r_new = g["high_r"].to_numpy(dtype=float) * scale
        low_r_new = g["low_r"].to_numpy(dtype=float) * scale
        close_r_new = g["close_r"].to_numpy(dtype=float) * scale
        bars = g["bar_offset"].to_numpy(dtype=int)

        new_exit_reason = "time_exit"
        new_final_r = float(close_r_new[-1])
        new_bars_held = int(bars[-1] + 1)
        # SL hit at first bar where low_r_new <= -1.0 (NEW SL space)
        sl_hit_mask = low_r_new <= -1.0
        if sl_hit_mask.any():
            first_hit = int(np.argmax(sl_hit_mask))
            new_final_r = -1.0
            new_exit_reason = "stoploss"
            new_bars_held = int(bars[first_hit] + 1)
            # Truncate path to first hit for MFE/MAE
            high_r_new = high_r_new[: first_hit + 1]
            low_r_new = low_r_new[: first_hit + 1]
            close_r_new = close_r_new[: first_hit + 1]
        new_mfe_r = float(np.maximum.accumulate(high_r_new).max()) if len(high_r_new) > 0 else 0.0
        new_mae_r = float(np.minimum.accumulate(low_r_new).min()) if len(low_r_new) > 0 else 0.0

        # If original trade was time_exit, the actual exit price scaling differs from SL-distance scaling
        # but the per-bar close_r is correct.
        rows.append(
            {
                "trade_id": int(trade_id),
                "pair": prow["pair"],
                "sl_multiplier": float(new_sl_multiplier),
                "final_r": float(new_final_r),
                "mfe_r": float(new_mfe_r),
                "mae_r": float(new_mae_r),
                "exit_reason": new_exit_reason,
                "bars_held": int(new_bars_held),
                "year": int(prow.get("calendar_year", pd.Timestamp(prow["entry_time"]).year)),
            }
        )
    return pd.DataFrame(rows)


# ─── Per-cluster metrics ──────────────────────────────────────────


def cluster_capturability(
    rsweep_df: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    sl_multiplier: float,
) -> pd.DataFrame:
    """For a given SL multiplier, compute per-cluster capturability metrics."""
    sub = rsweep_df[rsweep_df["sl_multiplier"] == sl_multiplier].merge(
        cluster_assignments[["trade_id", "cluster_best"]],
        on="trade_id",
        how="inner",
    )
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for cid, g in sub.groupby("cluster_best"):
        if len(g) < 1:
            continue
        reach_1r = float((g["mfe_r"] >= 1.0).mean())
        reach_2r = float((g["mfe_r"] >= 2.0).mean())
        reach_3r = float((g["mfe_r"] >= 3.0).mean())
        # wrong-way path prevalence: hit -1R MAE before +1R MFE.
        # Approximation: trades where final_r == -1.0 (SL hit) AND mfe_r < 1.0.
        ww_pp_proxy = float(((g["final_r"] == -1.0) & (g["mfe_r"] < 1.0)).mean())
        mfe_p25 = float(g["mfe_r"].quantile(0.25))
        mfe_p50 = float(g["mfe_r"].quantile(0.50))
        mfe_p75 = float(g["mfe_r"].quantile(0.75))
        mfe_p90 = float(g["mfe_r"].quantile(0.90))
        mean_r = float(g["final_r"].mean())
        p25_r = float(g["final_r"].quantile(0.25))
        p50_r = float(g["final_r"].quantile(0.50))
        # Capturability composite: 0.4*reach_1R + 0.4*min(mfe_p50,3)/3 + 0.2*(1-ww_pp)
        comp = 0.4 * reach_1r + 0.4 * min(mfe_p50, 3.0) / 3.0 + 0.2 * (1.0 - ww_pp_proxy)
        rows.append(
            {
                "cluster": int(cid),
                "sl_multiplier": float(sl_multiplier),
                "n": int(len(g)),
                "reach_1R": reach_1r,
                "reach_2R": reach_2r,
                "reach_3R": reach_3r,
                "mfe_p25": mfe_p25,
                "mfe_p50": mfe_p50,
                "mfe_p75": mfe_p75,
                "mfe_p90": mfe_p90,
                "ww_pp": ww_pp_proxy,
                "mean_R": mean_r,
                "p25_R": p25_r,
                "p50_R": p50_r,
                "capturability_composite": float(comp),
                "candidate_cluster": bool(
                    reach_1r >= 0.50 and ww_pp_proxy <= 0.30 and mfe_p50 >= 1.5
                ),
            }
        )
    return pd.DataFrame(rows)


# ─── Archetype labelling ──────────────────────────────────────────


def cluster_archetype(path_feats: pd.DataFrame, cluster_id: int) -> str:
    """Per-cluster dominant archetype from path-shape distribution.

    Voting via the per-trade shape_tag (computed in Step 2):
      - if dominant tag is v_shape_recovery → 'V-shape recovery'
      - if dominant tag is stepwise_climber → 'Stepwise climber'
      - if dominant tag is bimodal → 'Bimodal'
      - if dominant tag is monotonic_up → 'Monotonic up'
      - if dominant tag is choppy → 'Choppy'
      - if dominant tag is monotonic_down → 'Monotonic down'
      - else 'mixed' → fallback to mean-driven heuristic
    """
    sub = path_feats[path_feats["cluster_best"] == cluster_id]
    if sub.empty or "shape_tag" not in sub.columns:
        return "unknown"
    counts = sub["shape_tag"].value_counts()
    if counts.empty:
        return "unknown"
    top = counts.idxmax()
    mapping = {
        "v_shape_recovery": "V-shape recovery",
        "stepwise_climber": "Stepwise climber",
        "bimodal": "Bimodal",
        "monotonic_up": "Monotonic up",
        "monotonic_down": "Monotonic down",
        "choppy": "Choppy",
        "mixed": "Mixed",
    }
    arche = mapping.get(top, "Mixed")
    if arche == "Mixed":
        # Fallback heuristic on cluster means
        mono = float(sub["close_r_mono"].mean())
        peaks = float(sub["peaks"].mean())
        dd_depth = float(sub["drawdown_depth"].mean())
        recov = float(sub["recovery_shape"].mean())
        if dd_depth > 0.7 and recov > 0.6:
            return "V-shape recovery"
        if peaks >= 3.0 and mono > 0.5:
            return "Stepwise climber"
        if mono > 0.6:
            return "Monotonic up"
        if peaks >= 5.0:
            return "Choppy"
        if peaks == 2.0:
            return "Bimodal"
        return "Mixed"
    return arche


# ─── Step 3 driver ─────────────────────────────────────────────────


def run(cfg: dict) -> dict:
    seed_everything(RANDOM_STATE)

    results_dir = results_root(cfg)
    step1_dir = results_dir / "step_1"
    step2_dir = results_dir / "step_2"
    step3_dir = results_dir / "step_3"
    step3_dir.mkdir(parents=True, exist_ok=True)

    pool_df = pd.read_parquet(step1_dir / "pool.parquet")
    paths_df = pd.read_parquet(step1_dir / "trades_paths.parquet")
    ca_df = pd.read_parquet(step2_dir / "cluster_assignments.parquet")
    pf_df = pd.read_parquet(step2_dir / "path_features.parquet")
    if "shape_tag" not in pf_df.columns:
        pf_df["shape_tag"] = "mixed"
    # cluster_best from ca_df
    if "cluster_best" not in pf_df.columns:
        pf_df = pf_df.merge(ca_df[["trade_id", "cluster_best"]], on="trade_id", how="left")

    sl_multipliers = list(cfg["step_3"]["sl_sweep_multipliers"])

    _log(f"Computing SL-sweep R per trade across {len(sl_multipliers)} multipliers")
    all_sweep = []
    for sl in sl_multipliers:
        rs = recompute_R_at_sl(paths_df, pool_df, sl)
        all_sweep.append(rs)
        _log(f"  SL={sl}: {len(rs)} rows, hit_rate={(rs['exit_reason']=='stoploss').mean():.3f}")
    sweep_df = pd.concat(all_sweep, ignore_index=True)

    _log("Computing per-cluster capturability per SL")
    cap_rows = []
    for sl in sl_multipliers:
        ccs = cluster_capturability(sweep_df, ca_df, sl)
        cap_rows.append(ccs)
    cap_df = pd.concat(cap_rows, ignore_index=True) if cap_rows else pd.DataFrame()

    # Per-cluster best SL (max capturability_composite)
    best_sl_per_cluster: dict[int, dict] = {}
    if not cap_df.empty:
        for cid, g in cap_df.groupby("cluster"):
            best = g.sort_values("capturability_composite", ascending=False).iloc[0]
            best_sl_per_cluster[int(cid)] = {
                "best_sl_multiplier": float(best["sl_multiplier"]),
                "composite": float(best["capturability_composite"]),
                "candidate_cluster": bool(best["candidate_cluster"]),
                "reach_1R": float(best["reach_1R"]),
                "mfe_p50": float(best["mfe_p50"]),
                "ww_pp": float(best["ww_pp"]),
                "n": int(best["n"]),
            }

    # Archetypes per cluster
    arches = {}
    for cid in best_sl_per_cluster:
        arches[cid] = cluster_archetype(pf_df, cid)

    arche_df = pd.DataFrame(
        [
            {
                "cluster": cid,
                "archetype": arches[cid],
                "best_sl": best_sl_per_cluster[cid]["best_sl_multiplier"],
                "composite": best_sl_per_cluster[cid]["composite"],
                "candidate_cluster": best_sl_per_cluster[cid]["candidate_cluster"],
                "n": best_sl_per_cluster[cid]["n"],
            }
            for cid in sorted(best_sl_per_cluster.keys())
        ]
    )

    cap_path = step3_dir / "capturability.csv"
    cap_df.to_csv(cap_path, index=False, lineterminator="\n")
    arche_path = step3_dir / "per_cluster_archetype.csv"
    arche_df.to_csv(arche_path, index=False, lineterminator="\n")

    # Summary md
    lines = ["# Arc 11 v3.0 — Step 3 Capturability Summary", "", "Per L_PROTOCOL §2 Step 3.", ""]
    lines.append("## Per-cluster best SL + archetype + candidate flag")
    lines.append("")
    lines.append("| cluster | archetype | best SL | composite | reach_1R | mfe_p50 | ww_pp | n | candidate |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for _, r in arche_df.iterrows():
        cid = int(r["cluster"])
        bs = best_sl_per_cluster[cid]
        lines.append(
            f"| {cid} | {r['archetype']} | {bs['best_sl_multiplier']:.1f} | {bs['composite']:.4f} | "
            f"{bs['reach_1R']:.3f} | {bs['mfe_p50']:.3f} | {bs['ww_pp']:.3f} | {bs['n']} | "
            f"{'YES' if bs['candidate_cluster'] else 'no'} |"
        )
    lines.append("")

    lines.append("## Full per-cluster x per-SL grid")
    lines.append("")
    if not cap_df.empty:
        for cid in sorted(cap_df["cluster"].unique()):
            sub = cap_df[cap_df["cluster"] == cid].sort_values("sl_multiplier")
            lines.append(f"### Cluster {int(cid)} ({arches.get(int(cid), 'unknown')})")
            lines.append("")
            lines.append("| SL | n | reach_1R | reach_2R | reach_3R | mfe_p50 | ww_pp | mean_R | composite | candidate |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r['sl_multiplier']:.1f} | {int(r['n'])} | {r['reach_1R']:.3f} | "
                    f"{r['reach_2R']:.3f} | {r['reach_3R']:.3f} | {r['mfe_p50']:.3f} | "
                    f"{r['ww_pp']:.3f} | {r['mean_R']:.3f} | {r['capturability_composite']:.4f} | "
                    f"{'YES' if r['candidate_cluster'] else 'no'} |"
                )
            lines.append("")

    summary_path = step3_dir / "capturability_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    candidate_clusters = [cid for cid, v in best_sl_per_cluster.items() if v["candidate_cluster"]]
    manifest = {
        "step": 3,
        "sl_multipliers_tested": sl_multipliers,
        "n_clusters": len(best_sl_per_cluster),
        "candidate_clusters": sorted(candidate_clusters),
        "per_cluster_best_sl": best_sl_per_cluster,
        "per_cluster_archetype": arches,
        "sha256": {
            "capturability_csv": sha256_file(cap_path),
            "per_cluster_archetype_csv": sha256_file(arche_path),
        },
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "env": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_manifest(step3_dir / "manifest.json", manifest)
    _log(f"Step 3 complete: candidate_clusters={sorted(candidate_clusters)}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/wfo_l_arc_11.yaml")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Arc 10 v3.0 — Step 3 capturability.

Per L_PROTOCOL v3.0 §2 Step 3. Per cluster (from Step 2):
    - reach rates: P(MFE ≥ 1R), P(MFE ≥ 2R), P(MFE ≥ 3R)
    - MFE distribution: p25, p50, p75, p90
    - wrong_way_pp
    - time-to-peak distribution
    - mean R, p25 R, p50 R
    - SL sweep over {1.5..4.0} × ATR
    - capturability composite + candidate flag (reach_1R ≥ 0.50 AND ww_pp ≤ 0.30 AND mfe_p50 ≥ 1.5R)
    - explicit archetype label per cluster (carried from Step 2)

Outputs:
    results/l_arc_10/step_3/capturability.csv
    results/l_arc_10/step_3/capturability_summary.md
    results/l_arc_10/step_3/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.determinism import RANDOM_STATE, seed_everything  # noqa: E402
from scripts.l_arc_10_v3._common import load_config, sha256_file, write_manifest  # noqa: E402

SL_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def _per_cluster_metrics(grp: pd.DataFrame, paths: pd.DataFrame) -> dict:
    """Compute capturability metrics for one cluster (at Step 1 SL=2.0×ATR)."""
    n = int(len(grp))
    if n == 0:
        return dict(n=0)
    mfe = grp["mfe_r"].to_numpy()
    mae = grp["mae_r"].to_numpy()
    final_r = grp["final_r"].to_numpy()
    ttp = grp["time_to_peak_mfe"].to_numpy()
    bars_held = grp["bars_held"].to_numpy()
    ww = grp["path_wrong_way_first"].to_numpy()
    return dict(
        n=n,
        reach_1r=float(np.nanmean(mfe >= 1.0)),
        reach_2r=float(np.nanmean(mfe >= 2.0)),
        reach_3r=float(np.nanmean(mfe >= 3.0)),
        mfe_p25=float(np.nanpercentile(mfe, 25)),
        mfe_p50=float(np.nanpercentile(mfe, 50)),
        mfe_p75=float(np.nanpercentile(mfe, 75)),
        mfe_p90=float(np.nanpercentile(mfe, 90)),
        mae_p50=float(np.nanpercentile(mae, 50)),
        ww_pp=float(np.nanmean(ww)),
        ttp_p50=float(np.nanpercentile(ttp, 50)),
        ttp_p75=float(np.nanpercentile(ttp, 75)),
        bars_held_p50=float(np.nanpercentile(bars_held, 50)),
        mean_r=float(np.nanmean(final_r)),
        p25_r=float(np.nanpercentile(final_r, 25)),
        p50_r=float(np.nanpercentile(final_r, 50)),
    )


def _sl_sweep_per_cluster(grp: pd.DataFrame, paths: pd.DataFrame) -> dict[float, dict]:
    """For each SL ∈ SL_MULTIPLIERS × ATR, recompute realised R using path data.

    The Step 1 simulation used SL=2.0×ATR. To re-simulate under a different SL
    multiplier M, we reread the per-bar path to find when low_price * sl_M is
    breached. Path features were stored in R-multiples relative to SL=2.0×ATR
    distance (sl_distance_2 = 2*ATR), so:

        bar low_r (R relative to old SL): mae_so_far_r at peak adverse
        new SL distance: M * ATR = (M/2) * sl_distance_2
        breach condition at bar k: mae_so_far_r[k] <= -(M/2)
        new R if SL hit at bar k: -(M/2) * (sl_distance_2 / new_sl_dist) = -1.0
            but rescaled to NEW R units: just -1.0 (by definition of SL)
        new R if exit otherwise: final_price_change / new_sl_dist
            = (final_r_old * sl_distance_2) / (M * ATR)
            = final_r_old * (2 / M)
        new MFE in new units: mfe_r_old * (2 / M)
        new MAE in new units: mae_r_old * (2 / M)
    """
    out = {}
    trade_ids = grp["trade_id"].to_numpy()
    grp_paths = paths[paths["trade_id"].isin(trade_ids)].sort_values(["trade_id", "bar_offset"])

    # Per-trade SL=2 outcomes
    final_r_2 = dict(zip(grp["trade_id"].to_numpy(), grp["final_r"].to_numpy()))
    mfe_r_2 = dict(zip(grp["trade_id"].to_numpy(), grp["mfe_r"].to_numpy()))

    for M in SL_MULTIPLIERS:
        scale = 2.0 / M
        thresh = -(M / 2.0)  # breach threshold in OLD R units

        new_finals = []
        new_mfes = []
        reach_1r_count = 0
        reach_2r_count = 0
        reach_3r_count = 0
        ww_count = 0
        n_eval = 0
        for tid, tg in grp_paths.groupby("trade_id"):
            mae_arr = tg["mae_so_far_r"].to_numpy()
            mfe_arr = tg["mfe_so_far_r"].to_numpy()
            # Index where new SL is breached (mae_so_far_r in OLD units crosses thresh)
            breach_idx = -1
            for i, m in enumerate(mae_arr):
                if not np.isfinite(m):
                    continue
                if m <= thresh:
                    breach_idx = i
                    break
            if breach_idx >= 0:
                # New final R = -1.0 (SL hit), and MFE is the peak mid-price advance prior to breach
                mfe_before = float(np.nanmax(mfe_arr[: breach_idx + 1])) * scale if breach_idx >= 0 else 0.0
                new_finals.append(-1.0)
                new_mfes.append(mfe_before)
            else:
                # No SL breach under new multiplier — final = scaled close_r, MFE = scaled max
                final_old = final_r_2.get(tid, np.nan)
                mfe_old = mfe_r_2.get(tid, np.nan)
                new_finals.append(float(final_old) * scale if np.isfinite(final_old) else np.nan)
                new_mfes.append(float(mfe_old) * scale if np.isfinite(mfe_old) else np.nan)

            # Reach checks on new MFE
            if new_mfes[-1] >= 1.0:
                reach_1r_count += 1
            if new_mfes[-1] >= 2.0:
                reach_2r_count += 1
            if new_mfes[-1] >= 3.0:
                reach_3r_count += 1
            # Wrong-way: did MAE hit -(M/2) before MFE hit 1R (in old units)? Approximate from path:
            mae_below_1 = np.where(np.isfinite(mae_arr) & (mae_arr <= -(M / 2.0)))[0]
            mfe_above_1 = np.where(np.isfinite(mfe_arr) & (mfe_arr * scale >= 1.0))[0]
            first_mae = int(mae_below_1[0]) if mae_below_1.size > 0 else 10**9
            first_mfe = int(mfe_above_1[0]) if mfe_above_1.size > 0 else 10**9
            if first_mae < first_mfe and first_mae < 10**9:
                ww_count += 1
            n_eval += 1

        if n_eval == 0:
            out[M] = dict(n=0)
            continue

        new_finals_arr = np.array(new_finals, dtype=float)
        new_mfes_arr = np.array(new_mfes, dtype=float)
        out[M] = dict(
            n=n_eval,
            mean_r=float(np.nanmean(new_finals_arr)),
            p50_r=float(np.nanpercentile(new_finals_arr, 50)),
            reach_1r=reach_1r_count / n_eval,
            reach_2r=reach_2r_count / n_eval,
            reach_3r=reach_3r_count / n_eval,
            mfe_p50=float(np.nanpercentile(new_mfes_arr, 50)),
            mfe_p75=float(np.nanpercentile(new_mfes_arr, 75)),
            ww_pp=ww_count / n_eval,
            composite=_composite(
                reach_1r=reach_1r_count / n_eval,
                mfe_p50=float(np.nanpercentile(new_mfes_arr, 50)),
                ww_pp=ww_count / n_eval,
            ),
        )

    return out


def _composite(reach_1r: float, mfe_p50: float, ww_pp: float) -> float:
    """Capturability composite — weighted aggregate (ranking aid only)."""
    if not np.isfinite(reach_1r) or not np.isfinite(mfe_p50) or not np.isfinite(ww_pp):
        return np.nan
    return 0.4 * reach_1r + 0.35 * min(mfe_p50 / 3.0, 1.0) + 0.25 * (1.0 - ww_pp)


def _candidate_flag(reach_1r: float, ww_pp: float, mfe_p50: float) -> bool:
    return bool(
        np.isfinite(reach_1r)
        and np.isfinite(ww_pp)
        and np.isfinite(mfe_p50)
        and reach_1r >= 0.50
        and ww_pp <= 0.30
        and mfe_p50 >= 1.5
    )


def run(cfg_path: Path, *, write_manifest_flag: bool = True) -> dict:
    seed_everything(RANDOM_STATE)
    cfg = load_config(cfg_path)

    pool_path = REPO_ROOT / cfg["output"]["results_dir"] / cfg["output"]["pool_parquet"]
    pool = pd.read_parquet(pool_path)
    paths_path = REPO_ROOT / cfg["output"]["results_dir"] / "trade_paths.parquet"
    paths = pd.read_parquet(paths_path)

    # Arc root + step dirs derived from Step 1 results_dir. Byte-identical
    # resolution for Arc 10 v3.0; correct routing for Arc 10 v3.0.2.
    arc_root = REPO_ROOT / Path(cfg["output"]["results_dir"]).parent
    assignments = pd.read_parquet(arc_root / "step_2" / "cluster_assignments.parquet")
    pool = pool.merge(
        assignments[["trade_id", "cluster_primary", "archetype_primary", "primary_K"]], on="trade_id", how="left"
    )

    out_dir = arc_root / "step_3"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_k = int(pool["primary_K"].iloc[0])
    per_cluster_rows = []
    sl_sweep_records = []

    for cid in sorted(pool["cluster_primary"].dropna().unique()):
        if cid == -1:
            continue
        cid = int(cid)
        grp = pool[pool["cluster_primary"] == cid]
        arch = grp["archetype_primary"].iloc[0]
        m = _per_cluster_metrics(grp, paths)
        m["cluster_id"] = cid
        m["archetype"] = arch
        m["composite"] = _composite(m["reach_1r"], m["mfe_p50"], m["ww_pp"])
        m["candidate_at_sl_2x"] = _candidate_flag(m["reach_1r"], m["ww_pp"], m["mfe_p50"])

        # SL sweep
        sl_sweep = _sl_sweep_per_cluster(grp, paths)
        # Pick best SL by composite
        best_sl = None
        best_composite = -np.inf
        for M, vals in sl_sweep.items():
            c = vals.get("composite", np.nan)
            if np.isfinite(c) and c > best_composite:
                best_composite = c
                best_sl = M
        m["best_sl_multiplier"] = float(best_sl) if best_sl is not None else np.nan
        m["best_sl_composite"] = float(best_composite) if best_composite != -np.inf else np.nan
        m["candidate_at_best_sl"] = (
            _candidate_flag(
                sl_sweep[best_sl]["reach_1r"],
                sl_sweep[best_sl]["ww_pp"],
                sl_sweep[best_sl]["mfe_p50"],
            )
            if best_sl is not None
            else False
        )
        per_cluster_rows.append(m)

        for M, vals in sl_sweep.items():
            row = dict(cluster_id=cid, archetype=arch, sl_multiplier=M, **vals)
            sl_sweep_records.append(row)

    df_metrics = pd.DataFrame(per_cluster_rows)
    df_sweep = pd.DataFrame(sl_sweep_records)
    df_metrics.to_csv(out_dir / "capturability.csv", index=False, lineterminator="\n")
    df_sweep.to_csv(out_dir / "sl_sweep.csv", index=False, lineterminator="\n")

    # Markdown summary
    lines = []
    lines.append("# Arc 10 v3.0 — Step 3 Capturability Summary\n\n")
    lines.append(f"- Primary K = {best_k}\n")
    lines.append(f"- Pool size: {len(pool)}\n")
    lines.append("- Candidate flag rule: reach_1R ≥ 0.50 AND ww_pp ≤ 0.30 AND mfe_p50 ≥ 1.5R\n\n")
    lines.append("## Per-cluster metrics (at Step 1 SL=2.0×ATR)\n\n")
    lines.append(
        "| Cluster | Archetype | n | reach_1R | reach_2R | mfe_p50 | mfe_p75 | ww_pp | mean_R | composite | candidate@2x | best_SL | composite@best | candidate@best |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|\n"
    )
    for r in sorted(per_cluster_rows, key=lambda x: -x["composite"] if np.isfinite(x.get("composite", np.nan)) else 1):
        lines.append(
            f"| c{r['cluster_id']} | {r['archetype']} | {r['n']} | "
            f"{r['reach_1r']:.3f} | {r['reach_2r']:.3f} | {r['mfe_p50']:.3f} | {r['mfe_p75']:.3f} | "
            f"{r['ww_pp']:.3f} | {r['mean_r']:.3f} | {r['composite']:.4f} | "
            f"{'✓' if r['candidate_at_sl_2x'] else '✗'} | "
            f"{r['best_sl_multiplier']:.1f} | {r['best_sl_composite']:.4f} | "
            f"{'✓' if r['candidate_at_best_sl'] else '✗'} |\n"
        )

    lines.append("\n## SL sweep per cluster\n\n")
    for cid in sorted(df_sweep["cluster_id"].unique()):
        cluster_sweep = df_sweep[df_sweep["cluster_id"] == cid].sort_values("sl_multiplier")
        arch = cluster_sweep["archetype"].iloc[0]
        lines.append(f"\n### Cluster c{cid} ({arch})\n\n")
        lines.append("| SL × ATR | n | reach_1R | mfe_p50 | ww_pp | mean_R | composite |\n|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in cluster_sweep.iterrows():
            lines.append(
                f"| {row['sl_multiplier']:.1f} | {row['n']} | {row['reach_1r']:.3f} | "
                f"{row['mfe_p50']:.3f} | {row['ww_pp']:.3f} | {row['mean_r']:.3f} | "
                f"{row['composite']:.4f} |\n"
            )

    candidates = [r for r in per_cluster_rows if r["candidate_at_best_sl"]]
    lines.append(f"\n## Candidate clusters: **{len(candidates)}**\n")
    if candidates:
        for c in candidates:
            lines.append(f"- c{c['cluster_id']} ({c['archetype']}) at SL={c['best_sl_multiplier']:.1f}×ATR, composite={c['best_sl_composite']:.4f}\n")
    else:
        lines.append("- None — no cluster clears reach_1R ≥ 0.50 AND ww_pp ≤ 0.30 AND mfe_p50 ≥ 1.5R at any SL.\n")

    summary_path = out_dir / "capturability_summary.md"
    summary_path.write_text("".join(lines), encoding="utf-8", newline="\n")

    manifest = dict(
        arc_name="l_arc_10",
        step="step_3",
        protocol_version="v3.0",
        primary_K=best_k,
        per_cluster_metrics=per_cluster_rows,
        candidates=[r["cluster_id"] for r in candidates],
        sha256=dict(
            capturability_csv=sha256_file(out_dir / "capturability.csv"),
            sl_sweep_csv=sha256_file(out_dir / "sl_sweep.csv"),
            capturability_summary_md=sha256_file(summary_path),
        ),
        env=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        determinism=dict(random_state=RANDOM_STATE, n_jobs=1, line_terminator="\\n"),
        run_timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    if write_manifest_flag:
        write_manifest(out_dir / "manifest.json", manifest)

    print(
        f"[step_3] candidate clusters: {len(candidates)} / {len(per_cluster_rows)}",
        flush=True,
    )
    for c in candidates:
        print(
            f"  c{c['cluster_id']} ({c['archetype']}) "
            f"SL={c['best_sl_multiplier']:.1f}× composite={c['best_sl_composite']:.4f}",
            flush=True,
        )
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0 — Step 3 capturability")
    p.add_argument("-c", "--config", required=True, type=Path)
    args = p.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

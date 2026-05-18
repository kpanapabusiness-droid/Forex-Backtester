"""Arc 8 — Path 2 Diagnostic: Signal-tightening filter sweep.

Tests whether mechanical filters on the existing Step 1 trigger pool can
preferentially exclude c2 (dead Early-peak-hold bucket) while preserving c1.
Bypasses classifier prediction entirely — would change when the signal
fires, not how trades are filtered post-fire.

Single-rule sweeps:
  F1: trigger_close_pos ≥ θ for θ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
  F2: pullback_depth_atr ≥ θ for θ ∈ {0.5, 0.75, 1.0, 1.25, 1.5}
  F3: trigger_body_atr ≥ θ for θ ∈ {0.3, 0.5, 0.7, 1.0}
  F4: trigger_break_size_atr ≥ θ for θ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
  F5: ret_5bar_atr ≥ θ for θ ∈ {0.0, 0.5, 1.0, 1.5}
  F6: pos_in_20bar_range ≥ θ for θ ∈ {0.4, 0.5, 0.6, 0.7}

Two-rule combinations: top-3 single rules (by c1_retention × (1 − c2_retention))
× pairwise AND at each rule's best individual threshold.

Per filter:
  - pool_size, per-cluster retention, c1_fraction_post, c1_lift, c2_kill_rate
  - aggregate_mean_r, aggregate_hit_rate (at SL=4.0×ATR, default exit — c1's
    unit R-frame for consistency with Steps 4-5)
  - §5 floor check (pool_size ≥ 500)

Verdict:
  PATH_2_VIABLE if c1_retention ≥ 0.80 AND c2_retention ≤ 0.30 AND pool_size ≥ 500
  PATH_2_MARGINAL if partial satisfaction
  PATH_2_DEAD if none of the above

random_state=42, n_jobs=1. Deterministic.

Usage:
    py scripts/l_arc_8/diagnostics/signal_tightening.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_8.step3_capturability import _eval_trade_at_sl  # noqa: E402

OUT_DIR = _REPO_ROOT / "results" / "l_arc_8" / "diagnostics" / "signal_tightening"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIT_SL: float = 4.0
ORIGINAL_SL: float = 2.0

# Filter spec: (rule_id, feature_name, threshold_sweep).
SINGLE_RULES: List[Tuple[str, str, List[float]]] = [
    ("F1", "trigger_close_pos", [0.5, 0.6, 0.7, 0.8, 0.9]),
    ("F2", "pullback_depth_atr", [0.5, 0.75, 1.0, 1.25, 1.5]),
    ("F3", "trigger_body_atr", [0.3, 0.5, 0.7, 1.0]),
    ("F4", "trigger_break_size_atr", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("F5", "ret_5bar_atr", [0.0, 0.5, 1.0, 1.5]),
    ("F6", "pos_in_20bar_range", [0.4, 0.5, 0.6, 0.7]),
]

CLUSTER_IDS = [0, 1, 2, 3]


def _build_paths_index(paths_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    paths_sorted = paths_df.sort_values(["trade_id", "bar_offset"], kind="mergesort")
    for tid, g in paths_sorted.groupby("trade_id", sort=True):
        out[int(tid)] = g.reset_index(drop=True)
    return out


def _compute_base_e_one_feature(trades_df: pd.DataFrame, feature: str) -> pd.Series:
    """For features not in trades_all.csv (8 base entry), compute via step4."""
    from scripts.l_arc_8.step4_extractability import _build_pair_cache, compute_base_e_features
    pairs = sorted(trades_df["pair"].unique())
    pair_caches = {p: _build_pair_cache(p, "data/4hr") for p in pairs}
    base_e = compute_base_e_features(trades_df, pair_caches)
    return base_e.set_index("trade_id")[feature]


@dataclass
class _FilterResult:
    rule_id: str
    feature: str
    threshold: float
    rule_str: str
    pool_size: int
    pool_size_pre: int
    retention_c0: float
    retention_c1: float
    retention_c2: float
    retention_c3: float
    c1_fraction_pre: float
    c1_fraction_post: float
    c1_lift: float
    c2_kill_rate: float
    aggregate_mean_r_unit: float
    aggregate_hit_rate: float
    floor_500_pass: bool


def _apply_filter_single(
    trades_with: pd.DataFrame, feature_series: pd.Series,
    threshold: float, gte: bool = True,
) -> pd.Series:
    """Returns boolean mask aligned to trades_with index."""
    s = feature_series.reindex(trades_with["trade_id"].to_numpy())
    if gte:
        return (s >= threshold).fillna(False).to_numpy()
    return (s <= threshold).fillna(False).to_numpy()


def _evaluate_filter(
    trades_with: pd.DataFrame, mask: np.ndarray,
    paths_index: Dict[int, pd.DataFrame], unit_final_r_map: Dict[int, float],
    rule_id: str, feature: str, threshold: float, rule_str: str,
) -> _FilterResult:
    n_pre = len(trades_with)
    pool = trades_with[mask].copy()
    pool_size = len(pool)
    counts_pre = {
        cid: int((trades_with["cluster_id"] == cid).sum()) for cid in CLUSTER_IDS
    }
    counts_post = {
        cid: int((pool["cluster_id"] == cid).sum()) for cid in CLUSTER_IDS
    }
    retentions = {
        cid: counts_post[cid] / counts_pre[cid] if counts_pre[cid] else 0.0
        for cid in CLUSTER_IDS
    }
    c1_frac_pre = counts_pre[1] / n_pre if n_pre else 0.0
    c1_frac_post = counts_post[1] / pool_size if pool_size else 0.0
    c1_lift = c1_frac_post / c1_frac_pre if c1_frac_pre > 0 else float("nan")
    c2_kill = 1.0 - retentions[2]

    # Aggregate economics on filtered pool under SL=4.0×ATR unit frame.
    if pool_size > 0:
        unit_rs = np.array([unit_final_r_map[int(t)] for t in pool["trade_id"].to_numpy()],
                           dtype=float)
        agg_mean_r = float(unit_rs.mean())
        agg_hit = float((unit_rs > 0).mean())
    else:
        agg_mean_r = float("nan")
        agg_hit = float("nan")

    return _FilterResult(
        rule_id=rule_id, feature=feature, threshold=threshold, rule_str=rule_str,
        pool_size=pool_size, pool_size_pre=n_pre,
        retention_c0=retentions[0], retention_c1=retentions[1],
        retention_c2=retentions[2], retention_c3=retentions[3],
        c1_fraction_pre=c1_frac_pre, c1_fraction_post=c1_frac_post,
        c1_lift=c1_lift, c2_kill_rate=c2_kill,
        aggregate_mean_r_unit=agg_mean_r, aggregate_hit_rate=agg_hit,
        floor_500_pass=pool_size >= 500,
    )


def _filter_to_row(r: _FilterResult) -> Dict[str, Any]:
    return {
        "rule_id": r.rule_id, "feature": r.feature, "threshold": r.threshold,
        "rule_str": r.rule_str,
        "pool_size_pre": r.pool_size_pre, "pool_size_post": r.pool_size,
        "pool_kept_pct": r.pool_size / r.pool_size_pre if r.pool_size_pre else 0.0,
        "retention_c0": r.retention_c0, "retention_c1": r.retention_c1,
        "retention_c2": r.retention_c2, "retention_c3": r.retention_c3,
        "c1_fraction_pre": r.c1_fraction_pre, "c1_fraction_post": r.c1_fraction_post,
        "c1_lift": r.c1_lift, "c2_kill_rate": r.c2_kill_rate,
        "aggregate_mean_r_unit": r.aggregate_mean_r_unit,
        "aggregate_hit_rate": r.aggregate_hit_rate,
        "floor_500_pass": int(r.floor_500_pass),
    }


def _verdict(r: _FilterResult) -> str:
    if r.retention_c1 >= 0.80 and r.retention_c2 <= 0.30 and r.floor_500_pass:
        return "VIABLE"
    if (r.retention_c1 >= 0.65 and r.retention_c2 <= 0.40) or (
        r.retention_c1 >= 0.80 and r.retention_c2 <= 0.30 and not r.floor_500_pass
    ):
        return "MARGINAL"
    return "DEAD"


def pareto_frontier(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return non-dominated rows on (c2_kill_rate, retention_c1) maximisation,
    subject to pool_size ≥ 500.
    """
    candidates = [r for r in rows if r.get("pool_size_post", 0) >= 500
                  and not math.isnan(r.get("retention_c1", float("nan")))
                  and not math.isnan(r.get("c2_kill_rate", float("nan")))]
    pareto: List[Dict[str, Any]] = []
    for r in candidates:
        dominated = False
        for s in candidates:
            if r is s:
                continue
            if (s["retention_c1"] >= r["retention_c1"]
                    and s["c2_kill_rate"] >= r["c2_kill_rate"]
                    and (s["retention_c1"] > r["retention_c1"]
                         or s["c2_kill_rate"] > r["c2_kill_rate"])):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda x: x["c2_kill_rate"])
    return pareto


def write_pareto_plot(rows: List[Dict[str, Any]], pareto: List[Dict[str, Any]],
                      out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pool_sizes = np.array([r["pool_size_post"] for r in rows], dtype=float)
    c1_ret = np.array([r["retention_c1"] for r in rows], dtype=float)
    c2_kill = np.array([r["c2_kill_rate"] for r in rows], dtype=float)
    agg_mean = np.array([r.get("aggregate_mean_r_unit", float("nan")) for r in rows], dtype=float)

    # Mask NaN.
    valid = ~np.isnan(c1_ret) & ~np.isnan(c2_kill)
    pool_sizes = pool_sizes[valid]
    c1_ret = c1_ret[valid]
    c2_kill = c2_kill[valid]
    agg_mean = agg_mean[valid]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    # Size scale.
    sizes = np.clip(pool_sizes / 8.0, 8, 300)
    sc = ax.scatter(c2_kill, c1_ret, s=sizes, c=agg_mean, cmap="RdYlGn",
                    alpha=0.65, edgecolors="black", linewidths=0.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("aggregate mean_r (unit SL=4×ATR R-frame)", fontsize=9)

    # Pareto frontier.
    if pareto:
        pareto_x = [p["c2_kill_rate"] for p in pareto]
        pareto_y = [p["retention_c1"] for p in pareto]
        ax.plot(pareto_x, pareto_y, "k--", linewidth=1.0, alpha=0.6, label="Pareto frontier (pool ≥ 500)")
        ax.scatter(pareto_x, pareto_y, s=80, facecolors="none", edgecolors="black",
                   linewidths=1.5, label="_nolegend_")

    ax.axhline(0.80, color="green", linewidth=0.8, linestyle=":", alpha=0.6,
               label="c1_retention 0.80 target")
    ax.axvline(0.70, color="red", linewidth=0.8, linestyle=":", alpha=0.6,
               label="c2_kill 0.70 target (= retention 0.30)")
    ax.set_xlabel("c2_kill_rate (= 1 − retention_c2)")
    ax.set_ylabel("c1_retention")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Arc 8 — signal-tightening filter scan\n"
                 "point size = pool_size; goal = upper-right of viability box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, metadata={"Software": ""})
    plt.close(fig)


def write_top5_retention_bars(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """4-panel bar chart of per-cluster retention for top-5 filters by
    (c1_retention − c2_retention) with pool_size ≥ 500.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qualifying = [r for r in rows if r.get("pool_size_post", 0) >= 500]
    if not qualifying:
        qualifying = rows[:5]
    qualifying.sort(
        key=lambda r: (r.get("retention_c1", 0.0) - r.get("retention_c2", 1.0)),
        reverse=True,
    )
    top5 = qualifying[:5]
    if not top5:
        return

    fig, axes = plt.subplots(1, len(top5), figsize=(4 * len(top5), 4.5), sharey=True)
    if len(top5) == 1:
        axes = [axes]
    cluster_colors = {0: "#888888", 1: "#2ca02c", 2: "#d62728", 3: "#1f77b4"}
    for ax, r in zip(axes, top5):
        vals = [r["retention_c0"], r["retention_c1"], r["retention_c2"], r["retention_c3"]]
        ax.bar(["c0", "c1", "c2", "c3"], vals,
               color=[cluster_colors[c] for c in CLUSTER_IDS], alpha=0.75)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{r['rule_id']}: {r['rule_str']}\npool={r['pool_size_post']}, "
                     f"mean_r={r['aggregate_mean_r_unit']:.3f}",
                     fontsize=9)
        ax.axhline(0.80, color="green", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.axhline(0.30, color="red", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")
    axes[0].set_ylabel("retention fraction")
    fig.suptitle("Top-5 filter configurations (by c1_retention − c2_retention, pool ≥ 500)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=110, metadata={"Software": ""})
    plt.close(fig)


def main() -> int:
    print("[path2] Loading inputs...", file=sys.stderr)
    trades = pd.read_csv(_REPO_ROOT / "results/l_arc_8/step1_verbatim/trades_all.csv")
    paths_df = pd.read_csv(_REPO_ROOT / "results/l_arc_8/step1_verbatim/trades_paths.csv")
    clusters = pd.read_csv(_REPO_ROOT / "results/l_arc_8/step2/clusters_K4.csv")
    trades_with = trades.merge(clusters, on="trade_id", how="left")

    paths_index = _build_paths_index(paths_df)

    # Precompute final_r under unit SL=4.0×ATR for each trade — needed for
    # aggregate economics on filtered pools.
    print("[path2] Precomputing final_r under SL=4.0×ATR for full pool...", file=sys.stderr)
    unit_final_r_map: Dict[int, float] = {}
    for _, tr in trades_with.iterrows():
        path = paths_index[int(tr["trade_id"])]
        ev = _eval_trade_at_sl(path, UNIT_SL, ORIGINAL_SL)
        unit_final_r_map[int(tr["trade_id"])] = float(ev.final_r_new)

    # Build feature_series for each rule. PR-HHHL features in trades_all.csv;
    # ret_5bar_atr / pos_in_20bar_range need on-the-fly compute via Step 4 helper.
    feature_series: Dict[str, pd.Series] = {}
    in_trades = set(trades.columns)
    for rule_id, feat, _thresholds in SINGLE_RULES:
        if feat in in_trades:
            feature_series[feat] = trades_with.set_index("trade_id")[feat]
    needs_compute = [feat for _, feat, _ in SINGLE_RULES if feat not in feature_series]
    if needs_compute:
        from scripts.l_arc_8.step4_extractability import _build_pair_cache, compute_base_e_features
        print(f"[path2] Computing base entry features for: {needs_compute}", file=sys.stderr)
        pairs = sorted(trades_with["pair"].unique())
        pair_caches = {p: _build_pair_cache(p, "data/4hr") for p in pairs}
        base_e = compute_base_e_features(trades_with, pair_caches)
        for feat in needs_compute:
            feature_series[feat] = base_e.set_index("trade_id")[feat]

    # ===== Single-rule sweep =====
    print("[path2] === Single-rule sweep ===", file=sys.stderr)
    single_rows: List[Dict[str, Any]] = []
    single_results: List[_FilterResult] = []
    for rule_id, feat, thresholds in SINGLE_RULES:
        for thr in thresholds:
            rule_str = f"{feat} >= {thr}"
            mask = _apply_filter_single(trades_with, feature_series[feat], thr)
            res = _evaluate_filter(
                trades_with, mask, paths_index, unit_final_r_map,
                rule_id, feat, thr, rule_str,
            )
            single_rows.append(_filter_to_row(res))
            single_results.append(res)
            print(f"  {rule_id}({feat}>={thr}): pool={res.pool_size}, "
                  f"c1_ret={res.retention_c1:.3f}, c2_ret={res.retention_c2:.3f}, "
                  f"c2_kill={res.c2_kill_rate:.3f}, mean_r={res.aggregate_mean_r_unit:+.4f}",
                  file=sys.stderr)

    with (OUT_DIR / "single_rule_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(single_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for row in single_rows:
            w.writerow(row)

    # ===== Two-rule combinations =====
    # Rank single rules by score = c1_retention × c2_kill_rate, keeping only
    # rule_id-level best threshold (one threshold per rule).
    print("[path2] === Two-rule combinations ===", file=sys.stderr)
    per_rule_best: Dict[str, _FilterResult] = {}
    for res in single_results:
        score = res.retention_c1 * res.c2_kill_rate
        cur_best = per_rule_best.get(res.rule_id)
        cur_score = (cur_best.retention_c1 * cur_best.c2_kill_rate) if cur_best else -1.0
        if score > cur_score:
            per_rule_best[res.rule_id] = res
    # Pick top-3 rules by best-threshold score.
    ranked = sorted(per_rule_best.values(),
                    key=lambda r: r.retention_c1 * r.c2_kill_rate, reverse=True)
    top3 = ranked[:3]
    print(f"[path2] Top-3 single rules for combos: "
          f"{[(r.rule_id, r.feature, r.threshold) for r in top3]}", file=sys.stderr)

    combo_rows: List[Dict[str, Any]] = []
    for i, ra in enumerate(top3):
        for rb in top3[i + 1:]:
            mask_a = _apply_filter_single(trades_with, feature_series[ra.feature], ra.threshold)
            mask_b = _apply_filter_single(trades_with, feature_series[rb.feature], rb.threshold)
            mask = mask_a & mask_b
            rule_id = f"{ra.rule_id}+{rb.rule_id}"
            rule_str = f"({ra.feature} >= {ra.threshold}) AND ({rb.feature} >= {rb.threshold})"
            res = _evaluate_filter(
                trades_with, mask, paths_index, unit_final_r_map,
                rule_id, f"{ra.feature}__AND__{rb.feature}",
                float("nan"), rule_str,
            )
            row = _filter_to_row(res)
            row["component_a"] = f"{ra.rule_id}: {ra.feature} >= {ra.threshold}"
            row["component_b"] = f"{rb.rule_id}: {rb.feature} >= {rb.threshold}"
            combo_rows.append(row)
            print(f"  {rule_id}: pool={res.pool_size}, c1_ret={res.retention_c1:.3f}, "
                  f"c2_ret={res.retention_c2:.3f}, mean_r={res.aggregate_mean_r_unit:+.4f}",
                  file=sys.stderr)

    if combo_rows:
        with (OUT_DIR / "two_rule_combinations.csv").open("w", encoding="utf-8", newline="") as f:
            cols = list(combo_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            for row in combo_rows:
                w.writerow(row)

    # ===== Pareto frontier across all configurations (single + combo) =====
    all_rows = single_rows + combo_rows
    pareto = pareto_frontier(all_rows)
    if pareto:
        with (OUT_DIR / "pareto_frontier.csv").open("w", encoding="utf-8", newline="") as f:
            cols = list(pareto[0].keys())
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            for row in pareto:
                w.writerow(row)
    print(f"[path2] Pareto frontier (pool ≥ 500): {len(pareto)} points", file=sys.stderr)

    # ===== Plots =====
    write_pareto_plot(all_rows, pareto, OUT_DIR / "signal_tightening_pareto_plot.png")
    write_top5_retention_bars(all_rows, OUT_DIR / "cluster_retention_per_filter.png")

    # ===== Verdict =====
    viable = [r for r in all_rows
              if r["retention_c1"] >= 0.80 and r["retention_c2"] <= 0.30
              and r["pool_size_post"] >= 500]
    marginal_pool_short = [r for r in all_rows
                           if r["retention_c1"] >= 0.80 and r["retention_c2"] <= 0.30
                           and r["pool_size_post"] < 500]
    marginal_partial = [r for r in all_rows
                        if (r["retention_c1"] >= 0.65 and r["retention_c2"] <= 0.40)
                        and r["pool_size_post"] >= 500]

    if viable:
        verdict = "PATH_2_VIABLE"
        best = max(viable, key=lambda r: r["aggregate_mean_r_unit"])
    elif marginal_pool_short or marginal_partial:
        verdict = "PATH_2_MARGINAL"
        marg_pool = marginal_pool_short + marginal_partial
        best = max(marg_pool, key=lambda r: r["aggregate_mean_r_unit"])
    else:
        verdict = "PATH_2_DEAD"
        # Just pick the best-c1-retention-with-pool-500 as the "least bad" for reference.
        candidates_500 = [r for r in all_rows if r["pool_size_post"] >= 500]
        if candidates_500:
            best = max(candidates_500, key=lambda r: r["retention_c1"] - r["retention_c2"])
        else:
            best = max(all_rows, key=lambda r: r["retention_c1"] - r["retention_c2"])

    verdict_payload = {
        "verdict": verdict,
        "best_config": best,
        "n_viable_configs": len(viable),
        "n_marginal_configs": len(marginal_pool_short) + len(marginal_partial),
        "n_total_configs": len(all_rows),
        "n_pareto_points": len(pareto),
    }
    (OUT_DIR / "PATH2_VERDICT.json").write_text(
        json.dumps(verdict_payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\n[path2] Verdict: {verdict}", file=sys.stderr)
    print(f"[path2] Best config: {best['rule_str']}", file=sys.stderr)
    print(f"  pool={best['pool_size_post']}, c1_ret={best['retention_c1']:.3f}, "
          f"c2_ret={best['retention_c2']:.3f}, mean_r={best['aggregate_mean_r_unit']:+.4f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

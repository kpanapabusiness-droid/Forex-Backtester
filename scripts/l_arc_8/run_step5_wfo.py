"""Arc 8 Step 5 — WFO architecture search (pool-level approximation).

Per L_PROTOCOL §2 Step 5 + Amendments 1 + 2 + dispatch §"Step 5":

  - Archetype-driven architecture selection.
  - 11-fold WFO 2010-2020 (search) + 2021-present one-shot holdout per top-3.
  - Oracle WFO (true cluster labels) reported alongside.

Implementation pragmatic note: this runner operates at POOL LEVEL —
trades from Step 1 carry their per-trade (final_r, mfe_r, mae_r, exit_time)
through to fold-level metrics. Different exit policies are applied by
rule (sl_plus_tp_2r caps any trade with mfe_r >= 2 at +2R; etc.). SL
rescaling uses the closed-form rule from Step 3. Exposure caps applied
chronologically per pair / per currency.

This is NOT a bar-by-bar multipair backtester run. The full sim is
out-of-scope for this arc and tracked in the closure §6 improvements.
The pool-level approximation captures architecture-axis effects (filter
yes/no, SL choice, exit cap, exposure cap) with reasonable fidelity for
ranking purposes, and is calibration-honest: the cluster-2 oracle WFO
shows the upper bound under the same approximation.

Reads ``step_1/pool.parquet``, ``step_2/cluster_assignments.parquet``,
``step_3/capturability.csv``, ``step_4/classifiers/*.pkl``,
``step_1/features.parquet``. Writes ``step_5/wfo_results.csv``,
``wfo_oracle.csv``, ``architectures_ranked.md``, ``best_candidate.md``,
``holdout_results.csv``, ``manifest.json``.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.determinism import seed_everything, write_text_deterministic
from core.manifest import write_manifest
from core.wfo.folds import build_v3_folds
from scripts.l_arc_8.shared import RESULTS_ROOT, SL_ATR_MULT_STEP1

STEP_DIR: Path = RESULTS_ROOT / "step_5"
RISK_PCT: float = 0.005   # dispatch §header
STARTING_BALANCE: float = 100_000.0

# Per dispatch §"Step 5": SL search = Step 3 selected ± 1 step (3 values).
# Step 3 selected SL=1.5 for cluster 2. Steps in sweep are 0.5 → {1.5, 2.0, 2.5}
# (1.0 not in dispatch sweep set; clamp to range start).
SL_SEARCH: tuple[float, ...] = (1.5, 2.0, 2.5)

# Per dispatch §"Step 5": V-shape archetype-matched exits.
# Always include sl_only baseline.
EXIT_POLICIES_VSHAPE: tuple[str, ...] = ("sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r")

# Exposure axis per dispatch.
EXPOSURE_PER_PAIR: tuple[str, ...] = ("unlimited", "max_concurrent_per_pair_1")

# A3 bar count axis (deferred entry).
A3_N_BARS: tuple[int, ...] = (3, 5)

# A6 threshold pairs (lower, upper) → (0.5x, 1.0x) sizing.
A6_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (0.3, 0.5), (0.4, 0.6), (0.5, 0.7),
)


# ── exit-policy + SL rescaling ─────────────────────────────────────


def _apply_exit_policy(final_r: float, mfe_r: float, mae_r: float, policy: str) -> float:
    """Approximate exit-policy effect on per-trade R outcome.

    sl_only: keep final_r as-is (the pool's exit reason already enforces SL).
    sl_plus_tp_2r: if MFE >= 2R, cap at +2R; else final_r.
        (Approximation: assumes TP fires before any pull-back to SL when MFE >= 2.)
    sl_plus_tp_3r: same logic at +3R.
    """
    if policy == "sl_only":
        return final_r
    if policy == "sl_plus_tp_2r":
        return 2.0 if mfe_r >= 2.0 else final_r
    if policy == "sl_plus_tp_3r":
        return 3.0 if mfe_r >= 3.0 else final_r
    return final_r


def _rescale_outcome_at_sl(
    final_r_orig: float, mfe_r_orig: float, mae_r_orig: float,
    sl_mult_new: float, sl_mult_orig: float = SL_ATR_MULT_STEP1,
) -> tuple[float, float, float]:
    """Per Step 3 — rescale outcome under a new SL multiplier."""
    f = sl_mult_orig / sl_mult_new
    sl_orig_line = -1.0 / f
    if mae_r_orig <= sl_orig_line:
        return -1.0, mfe_r_orig * f, -1.0
    return final_r_orig * f, mfe_r_orig * f, mae_r_orig * f


# ── exposure cap ───────────────────────────────────────────────────


def _apply_exposure_cap(pool: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Chronologically drop trades that violate the exposure rule.

    Supported rules:
      - 'unlimited': pass-through
      - 'max_concurrent_per_pair_1': only one open position per pair at a time.
        Drop any trade whose entry_time precedes the previous trade-on-this-pair's
        exit_time.
    """
    if rule == "unlimited":
        return pool
    if rule == "max_concurrent_per_pair_1":
        kept = []
        last_exit: dict[str, pd.Timestamp] = {}
        for _, row in pool.sort_values("entry_time").iterrows():
            pair = row["pair"]
            le = last_exit.get(pair)
            if le is None or row["entry_time"] >= le:
                kept.append(row)
                last_exit[pair] = row["exit_time"]
        return pd.DataFrame(kept) if kept else pool.iloc[0:0]
    raise ValueError(f"unknown exposure rule {rule}")


# ── fold metrics ───────────────────────────────────────────────────


def _slice_pool_to_fold(pool: pd.DataFrame, oos_start: date, oos_end: date) -> pd.DataFrame:
    """Slice trades whose entry_time falls in the OOS window."""
    ts_start = pd.Timestamp(oos_start, tz="UTC")
    ts_end = pd.Timestamp(oos_end, tz="UTC") + pd.Timedelta(days=1)
    return pool[(pool["entry_time"] >= ts_start) & (pool["entry_time"] < ts_end)]


def _fold_metrics(trades: pd.DataFrame, r_col: str, size_col: str | None = None) -> dict:
    """Compute per-fold equity metrics from the trade R outcomes.

    Equity = balance * prod((1 + r_i * size_i * risk_pct))
    DD     = max(1 - equity / running_max)
    ROI    = (final / starting) - 1
    """
    if len(trades) == 0:
        return {
            "n_trades": 0, "mean_r": 0.0, "roi_pct": 0.0, "max_dd_pct": 0.0,
            "daily_5pct_breaches": 0, "roi_dd_ratio": 0.0,
        }
    rs = trades[r_col].to_numpy()
    sizes = trades[size_col].to_numpy() if size_col else np.ones(len(trades))
    trade_pnl_pct = rs * sizes * RISK_PCT
    # Apply chronologically (entry_time order)
    order = trades["entry_time"].argsort().to_numpy()
    trade_pnl_pct = trade_pnl_pct[order]
    eq = STARTING_BALANCE * np.cumprod(1.0 + trade_pnl_pct)
    if len(eq) == 0 or not np.isfinite(eq).all():
        return {"n_trades": int(len(trades)), "mean_r": float(np.mean(rs)), "roi_pct": 0.0,
                "max_dd_pct": 0.0, "daily_5pct_breaches": 0, "roi_dd_ratio": 0.0}
    running_max = np.maximum.accumulate(eq)
    dd = (running_max - eq) / running_max
    max_dd = float(np.max(dd))
    roi = float(eq[-1] / STARTING_BALANCE - 1.0)
    # Daily 5% breaches — approximate as count of bars where consecutive equity drop > 5%
    sorted_trades = trades.iloc[order]
    daily_groups = sorted_trades.groupby(sorted_trades["entry_time"].dt.normalize())
    daily_pnl = daily_groups[r_col].sum() * RISK_PCT  # ignores intra-day sizing variance
    daily_breaches = int((daily_pnl < -0.05).sum())
    ratio = roi / max_dd if max_dd > 0 else (999.0 if roi > 0 else 0.0)
    return {
        "n_trades": int(len(trades)),
        "mean_r": float(np.mean(rs)),
        "roi_pct": roi,
        "max_dd_pct": max_dd,
        "daily_5pct_breaches": daily_breaches,
        "roi_dd_ratio": ratio,
    }


# ── architecture builders ──────────────────────────────────────────


def _build_pool_for_config(
    full_pool: pd.DataFrame,
    sl_mult: float,
    exit_policy: str,
    exposure_rule: str,
    filter_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Take the Step 1 pool, apply (SL, exit, exposure, filter), return the
    realised-R pool ready for fold-metrics.

    filter_mask: boolean array same length as full_pool selecting admitted trades.
    """
    pool = full_pool.copy()
    if filter_mask is not None:
        pool = pool.loc[filter_mask].reset_index(drop=True)
    # Rescale outcome at new SL
    rescaled = pool.apply(
        lambda r: _rescale_outcome_at_sl(
            float(r["final_r"]), float(r["mfe_r"]), float(r["mae_r"]), sl_mult,
        ),
        axis=1, result_type="expand",
    )
    rescaled.columns = ["fr_sl", "mfe_sl", "mae_sl"]
    pool["final_r_sl"] = rescaled["fr_sl"]
    pool["mfe_r_sl"] = rescaled["mfe_sl"]
    pool["mae_r_sl"] = rescaled["mae_sl"]
    # Apply exit policy
    pool["final_r_exit"] = pool.apply(
        lambda r: _apply_exit_policy(
            float(r["final_r_sl"]), float(r["mfe_r_sl"]), float(r["mae_r_sl"]), exit_policy,
        ),
        axis=1,
    )
    # Apply exposure cap
    pool = _apply_exposure_cap(pool, exposure_rule)
    return pool


def _run_wfo(
    pool: pd.DataFrame, folds: list, r_col: str, size_col: str | None = None,
) -> pd.DataFrame:
    rows = []
    for fold in folds:
        if fold.is_empty_is and fold.fold_id == 1:
            # Per build_v3_folds: fold 1 has empty IS; OOS still meaningful for arc data starting 2010-01.
            pass
        ftrades = _slice_pool_to_fold(pool, fold.oos_start, fold.oos_end)
        m = _fold_metrics(ftrades, r_col=r_col, size_col=size_col)
        m["fold_id"] = int(fold.fold_id)
        m["oos_start"] = str(fold.oos_start)
        m["oos_end"] = str(fold.oos_end)
        rows.append(m)
    return pd.DataFrame(rows)


def _aggregate_wfo(per_fold: pd.DataFrame) -> dict:
    """Per L_PROTOCOL §3: worst-fold ROI/DD, mean-fold, sign-consistency, etc."""
    if len(per_fold) == 0:
        return {}
    return {
        "worst_fold_roi_pct": float(per_fold["roi_pct"].min()),
        "worst_fold_dd_pct": float(per_fold["max_dd_pct"].max()),
        "worst_fold_ratio": float(per_fold["roi_dd_ratio"].min()),
        "mean_fold_roi_pct": float(per_fold["roi_pct"].mean()),
        "mean_fold_dd_pct": float(per_fold["max_dd_pct"].mean()),
        "mean_fold_ratio": float(per_fold["roi_dd_ratio"].mean()),
        "sign_pos_folds": int((per_fold["roi_pct"] > 0).sum()),
        "sign_total_folds": int(len(per_fold)),
        "n_trades_total": int(per_fold["n_trades"].sum()),
        "any_daily_breach": int(per_fold["daily_5pct_breaches"].sum() > 0),
        "max_dd_across_folds": float(per_fold["max_dd_pct"].max()),
    }


def _assign_verdict(agg: dict) -> str:
    """Per L_PROTOCOL §3 gates."""
    if not agg:
        return "FAIL"
    pass_dep = (
        agg.get("worst_fold_ratio", 0) >= 2.0
        and agg.get("worst_fold_roi_pct", -1) > 0
        and agg.get("sign_pos_folds", 0) == agg.get("sign_total_folds", 1)
        and agg.get("max_dd_across_folds", 1) <= 0.08
        and agg.get("any_daily_breach", 1) == 0
        and agg.get("n_trades_total", 0) >= 25 * agg.get("sign_total_folds", 1)
    )
    if pass_dep:
        return "PASS-DEPLOYABLE"
    pass_via = (
        agg.get("worst_fold_ratio", 0) >= 2.0
        and agg.get("mean_fold_ratio", 0) >= 2.5
        and agg.get("max_dd_across_folds", 1) <= 0.10
        and agg.get("any_daily_breach", 1) == 0
        and agg.get("n_trades_total", 0) >= 25 * agg.get("sign_total_folds", 1)
    )
    if pass_via:
        return "PASS-VIABLE"
    return "FAIL"


# ── orchestrator ───────────────────────────────────────────────────


def main() -> Path:
    seed_everything(42)
    t0 = time.perf_counter()
    STEP_DIR.mkdir(parents=True, exist_ok=True)

    step1 = RESULTS_ROOT / "step_1"
    step2 = RESULTS_ROOT / "step_2"
    step3 = RESULTS_ROOT / "step_3"
    step4 = RESULTS_ROOT / "step_4"

    full_pool = pd.read_parquet(step1 / "pool.parquet")
    full_pool["entry_time"] = pd.to_datetime(full_pool["entry_time"], utc=True)
    full_pool["exit_time"] = pd.to_datetime(full_pool["exit_time"], utc=True)
    full_pool["signal_time"] = pd.to_datetime(full_pool["signal_time"], utc=True)
    assignments = pd.read_parquet(step2 / "cluster_assignments.parquet")
    full_pool = full_pool.merge(
        assignments[["trade_id", "cluster_primary"]], on="trade_id", how="left",
    ).sort_values("entry_time").reset_index(drop=True)
    cap = pd.read_csv(step3 / "capturability.csv")
    feat = pd.read_parquet(step1 / "features.parquet")

    candidate_cluster_ids = cap.loc[cap["candidate"], "cluster_id"].astype(int).tolist()
    if not candidate_cluster_ids:
        candidate_cluster_ids = [int(cap.sort_values("capturability_composite", ascending=False).iloc[0]["cluster_id"])]
    target_cluster = candidate_cluster_ids[0]  # V-shape (cluster 2)
    archetype = cap.loc[cap["cluster_id"] == target_cluster, "archetype"].iloc[0]
    print(f"[step5] target cluster {target_cluster} (archetype: {archetype})")

    structure = build_v3_folds()
    search_folds = list(structure.folds)
    holdout_fold = structure.holdout
    print(f"[step5] WFO structure: {len(search_folds)} search folds + 1 holdout "
          f"({holdout_fold.oos_start} -> {holdout_fold.oos_end})")

    # Load Step 4 classifier (for A6 sizing).
    clf_path = step4 / "classifiers" / f"cluster_{target_cluster}_rf.pkl"
    if clf_path.exists():
        with clf_path.open("rb") as f:
            clf_bundle = pickle.load(f)
        clf_feature_names = clf_bundle["feature_names"]
        # Score all trades' features → cluster-2-membership proba
        feat_aligned = feat.sort_values("trade_id").reset_index(drop=True)
        feat_indexed = feat_aligned.set_index("trade_id")[clf_feature_names].copy()
        # Drop columns that may be NaN (already done at Step 4); replace residual NaN with median
        for c in feat_indexed.columns:
            if feat_indexed[c].isna().any():
                feat_indexed[c] = feat_indexed[c].fillna(feat_indexed[c].median())
        X_all = feat_indexed.to_numpy(dtype=float)
        if clf_bundle.get("scaler") is not None:
            X_all = clf_bundle["scaler"].transform(X_all)
        proba_all = clf_bundle["classifier"].predict_proba(X_all)[:, 1]
        proba_df = pd.DataFrame({"trade_id": feat_aligned["trade_id"], "proba_c2": proba_all})
        full_pool = full_pool.merge(proba_df, on="trade_id", how="left")
    else:
        full_pool["proba_c2"] = 0.5
        clf_bundle = None

    # ── Architecture configs ──────────────────────────────────────
    configs: list[dict] = []
    # A1 — system_level_filter (no classifier, full pool subjected to system rules)
    for sl in SL_SEARCH:
        for exit_pol in EXIT_POLICIES_VSHAPE:
            for exp in EXPOSURE_PER_PAIR:
                configs.append({
                    "arch": "A1",
                    "sl_atr": sl,
                    "exit_policy": exit_pol,
                    "exposure": exp,
                    "extra": {},
                })
    # A6 — meta-labeling (Step 4 RF reused; classifier confidence → size mapping)
    # Apply size = 0 below lower threshold, 0.5 below upper, 1.0 above upper.
    if clf_bundle is not None:
        for sl in SL_SEARCH:
            for exit_pol in EXIT_POLICIES_VSHAPE:
                for exp in EXPOSURE_PER_PAIR:
                    for (lo, hi) in A6_THRESHOLDS:
                        configs.append({
                            "arch": "A6",
                            "sl_atr": sl,
                            "exit_policy": exit_pol,
                            "exposure": exp,
                            "extra": {"lo": lo, "hi": hi},
                        })
    # A3 — pipeline_de: not implemented (closure §6). Documented intent.
    n_a3_intended = len(SL_SEARCH) * len(EXIT_POLICIES_VSHAPE) * len(EXPOSURE_PER_PAIR) * len(A3_N_BARS)

    n_configs = len(configs)
    print(f"[step5] {n_configs} configs to evaluate (A1: {len(SL_SEARCH)*len(EXIT_POLICIES_VSHAPE)*len(EXPOSURE_PER_PAIR)}; "
          f"A6: {n_configs - len(SL_SEARCH)*len(EXIT_POLICIES_VSHAPE)*len(EXPOSURE_PER_PAIR)}; "
          f"A3 not implemented; intended count: {n_a3_intended})")
    search_scope = "thin" if n_configs < 50 else ("normal" if n_configs < 100 else "broad")

    # ── Evaluate each config across WFO ───────────────────────────
    wfo_rows: list[dict] = []
    per_fold_rows: list[dict] = []
    for idx, cfg in enumerate(configs):
        # Build admitted pool. For A1: full pool. For A6: full pool with size.
        if cfg["arch"] == "A1":
            cfg_pool = _build_pool_for_config(
                full_pool, cfg["sl_atr"], cfg["exit_policy"], cfg["exposure"],
            )
            cfg_pool["size_mult"] = 1.0
            r_col = "final_r_exit"
        elif cfg["arch"] == "A6":
            cfg_pool = _build_pool_for_config(
                full_pool, cfg["sl_atr"], cfg["exit_policy"], cfg["exposure"],
            )
            lo, hi = cfg["extra"]["lo"], cfg["extra"]["hi"]
            cfg_pool["size_mult"] = cfg_pool["proba_c2"].apply(
                lambda p: 0.0 if p < lo else (0.5 if p < hi else 1.0)
            )
            r_col = "final_r_exit"
        per_fold = _run_wfo(cfg_pool, search_folds, r_col=r_col, size_col="size_mult")
        agg = _aggregate_wfo(per_fold)
        verdict = _assign_verdict(agg)
        row = {
            "config_id": idx,
            "arch": cfg["arch"],
            "sl_atr": cfg["sl_atr"],
            "exit_policy": cfg["exit_policy"],
            "exposure": cfg["exposure"],
            "extra": json.dumps(cfg["extra"], sort_keys=True),
            "verdict": verdict,
            **agg,
        }
        wfo_rows.append(row)
        for _, fr in per_fold.iterrows():
            per_fold_rows.append({"config_id": idx, **fr.to_dict()})

    wfo_df = pd.DataFrame(wfo_rows).sort_values("worst_fold_ratio", ascending=False).reset_index(drop=True)
    per_fold_df = pd.DataFrame(per_fold_rows)
    wfo_path = STEP_DIR / "wfo_results.csv"
    perfold_path = STEP_DIR / "per_fold_metrics.csv"
    wfo_df.to_csv(wfo_path, index=False, lineterminator="\n")
    per_fold_df.to_csv(perfold_path, index=False, lineterminator="\n")
    print(f"[step5] best worst-fold ratio: {wfo_df.iloc[0]['worst_fold_ratio']:.4f} "
          f"(arch={wfo_df.iloc[0]['arch']}, verdict={wfo_df.iloc[0]['verdict']})")

    # ── Oracle WFO (cluster 2 trades only) ────────────────────────
    print("[step5] Oracle WFO (true cluster-2 membership)...")
    oracle_pool = full_pool[full_pool["cluster_primary"] == target_cluster]
    oracle_rows: list[dict] = []
    for sl in SL_SEARCH:
        for exit_pol in EXIT_POLICIES_VSHAPE:
            for exp in EXPOSURE_PER_PAIR:
                cfg_pool = _build_pool_for_config(
                    oracle_pool, sl, exit_pol, exp,
                )
                cfg_pool["size_mult"] = 1.0
                per_fold = _run_wfo(cfg_pool, search_folds, r_col="final_r_exit", size_col="size_mult")
                agg = _aggregate_wfo(per_fold)
                oracle_rows.append({
                    "sl_atr": sl, "exit_policy": exit_pol, "exposure": exp,
                    "verdict": _assign_verdict(agg), **agg,
                })
    oracle_df = pd.DataFrame(oracle_rows).sort_values("worst_fold_ratio", ascending=False).reset_index(drop=True)
    oracle_path = STEP_DIR / "wfo_oracle.csv"
    oracle_df.to_csv(oracle_path, index=False, lineterminator="\n")
    print(f"[step5] oracle best worst-fold ratio: {oracle_df.iloc[0]['worst_fold_ratio']:.4f}")

    # ── Top-3 holdout ─────────────────────────────────────────────
    top_k = 3
    top_configs = wfo_df.head(top_k).to_dict("records")
    holdout_rows: list[dict] = []
    for top_row in top_configs:
        cfg_id = int(top_row["config_id"])
        cfg = configs[cfg_id]
        if cfg["arch"] == "A1":
            cfg_pool = _build_pool_for_config(full_pool, cfg["sl_atr"], cfg["exit_policy"], cfg["exposure"])
            cfg_pool["size_mult"] = 1.0
        else:  # A6
            cfg_pool = _build_pool_for_config(full_pool, cfg["sl_atr"], cfg["exit_policy"], cfg["exposure"])
            lo, hi = cfg["extra"]["lo"], cfg["extra"]["hi"]
            cfg_pool["size_mult"] = cfg_pool["proba_c2"].apply(
                lambda p: 0.0 if p < lo else (0.5 if p < hi else 1.0)
            )
        ftrades = _slice_pool_to_fold(cfg_pool, holdout_fold.oos_start, holdout_fold.oos_end)
        m = _fold_metrics(ftrades, r_col="final_r_exit", size_col="size_mult")
        holdout_verdict = _assign_verdict({
            "worst_fold_ratio": m["roi_dd_ratio"],
            "worst_fold_roi_pct": m["roi_pct"],
            "mean_fold_ratio": m["roi_dd_ratio"],
            "sign_pos_folds": 1 if m["roi_pct"] > 0 else 0,
            "sign_total_folds": 1,
            "max_dd_across_folds": m["max_dd_pct"],
            "any_daily_breach": 1 if m["daily_5pct_breaches"] > 0 else 0,
            "n_trades_total": m["n_trades"],
        })
        holdout_rows.append({
            "rank": top_configs.index(top_row) + 1,
            "config_id": cfg_id, "arch": cfg["arch"],
            "sl_atr": cfg["sl_atr"], "exit_policy": cfg["exit_policy"], "exposure": cfg["exposure"],
            "extra": json.dumps(cfg["extra"], sort_keys=True),
            **m,
            "verdict": holdout_verdict,
        })
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_path = STEP_DIR / "holdout_results.csv"
    holdout_df.to_csv(holdout_path, index=False, lineterminator="\n")

    # ── Ranked architectures markdown ─────────────────────────────
    summary = _build_summary_md(
        wfo_df=wfo_df, oracle_df=oracle_df, holdout_df=holdout_df,
        target_cluster=target_cluster, archetype=archetype,
        n_configs=n_configs, search_scope=search_scope,
        n_a3_intended=n_a3_intended,
    )
    summary_path = STEP_DIR / "architectures_ranked.md"
    write_text_deterministic(summary_path, summary)

    write_manifest(
        STEP_DIR / "manifest.json",
        artefacts=[wfo_path, perfold_path, oracle_path, holdout_path, summary_path],
    )
    elapsed = time.perf_counter() - t0
    print(f"[step5] DONE in {elapsed:.1f}s -- top verdict: {wfo_df.iloc[0]['verdict']}; holdout verdicts: {[r['verdict'] for r in holdout_rows]}")
    return STEP_DIR


def _build_summary_md(
    wfo_df: pd.DataFrame, oracle_df: pd.DataFrame, holdout_df: pd.DataFrame,
    target_cluster: int, archetype: str, n_configs: int, search_scope: str,
    n_a3_intended: int,
) -> str:
    lines = [
        "# Arc 8 — Step 5 WFO Architecture Search",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
        f"- Target cluster: {target_cluster} (archetype: {archetype})",
        f"- Total configs evaluated: {n_configs}",
        f"- Selection-bias flag: **{search_scope}**",
        f"- A3 (Pipeline DE) not implemented this arc — intended config count "
        f"{n_a3_intended}; tracked in closure §6.",
        "",
        "## Top-10 by worst-fold ratio",
        "",
        "| Rank | Config | Arch | SL | Exit | Exposure | Extra | Worst ROI | Worst DD | Worst Ratio | Pos folds | Trades | Verdict |",
        "|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for idx, r in wfo_df.head(10).iterrows():
        lines.append(
            f"| {idx+1} | {int(r['config_id'])} | {r['arch']} | "
            f"{r['sl_atr']:.1f} | {r['exit_policy']} | {r['exposure']} | "
            f"{r['extra']} | {r['worst_fold_roi_pct']:+.4%} | "
            f"{r['worst_fold_dd_pct']:.4%} | {r['worst_fold_ratio']:.3f} | "
            f"{int(r['sign_pos_folds'])}/{int(r['sign_total_folds'])} | "
            f"{int(r['n_trades_total']):,} | **{r['verdict']}** |"
        )

    lines += [
        "",
        "## Oracle WFO (true cluster-2 membership at entry)",
        "",
        "Upper bound under the same approximation — if cluster-2 ID were known at entry, "
        "we'd see these numbers. The gap to real WFO above is the Step 4 classifier ceiling.",
        "",
        "| Rank | SL | Exit | Exposure | Worst ROI | Worst DD | Worst Ratio | Mean Ratio | Pos folds | Trades | Verdict |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, r in oracle_df.head(10).iterrows():
        lines.append(
            f"| {idx+1} | {r['sl_atr']:.1f} | {r['exit_policy']} | {r['exposure']} | "
            f"{r['worst_fold_roi_pct']:+.4%} | {r['worst_fold_dd_pct']:.4%} | "
            f"{r['worst_fold_ratio']:.3f} | {r['mean_fold_ratio']:.3f} | "
            f"{int(r['sign_pos_folds'])}/{int(r['sign_total_folds'])} | "
            f"{int(r['n_trades_total']):,} | **{r['verdict']}** |"
        )

    lines += [
        "",
        "## Holdout results (top-3 from search WFO, one-shot 2021 → present)",
        "",
        "| Rank | Config | Arch | SL | Exit | Exposure | Extra | ROI | DD | Ratio | Trades | Verdict |",
        "|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for _, r in holdout_df.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {int(r['config_id'])} | {r['arch']} | "
            f"{r['sl_atr']:.1f} | {r['exit_policy']} | {r['exposure']} | "
            f"{r['extra']} | {r['roi_pct']:+.4%} | {r['max_dd_pct']:.4%} | "
            f"{r['roi_dd_ratio']:.3f} | {int(r['n_trades']):,} | **{r['verdict']}** |"
        )

    lines += [
        "",
        "## Methodology notes",
        "",
        "- Pool-level WFO: per-trade R-outcomes from Step 1 carried through with SL "
        "rescaling + exit-policy approximation + exposure cap. Equity = "
        "`balance * prod(1 + r * size * risk_pct)`. Risk = 0.5% per trade.",
        "- Bar-by-bar multipair sim NOT used; this is an approximation. See closure §6 "
        "for the full-sim PR scope.",
        "- A1 (no filter) baseline: full pool of all 6,757 trades through system rules.",
        "- A6 (meta-labeling): Step 4 RF classifier (mean AUC 0.5300) → size mapping "
        "via (lo, hi) threshold pair. lo=0.4, hi=0.6 maps to 0x/0.5x/1.0x risk.",
        "- A3 (Pipeline DE) requires new per-fold classifier training on path-so-far "
        "features; not built within this arc's scope (closure §6).",
        "- Oracle WFO uses TRUE cluster-2 membership labels (only the 1,540 cluster-2 "
        "trades enter the per-config pool). Establishes the upper bound IF entry-time "
        "classification were perfect.",
        "- Holdout: one-shot per candidate, no re-tuning permitted (L_PROTOCOL §2 Step 5).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

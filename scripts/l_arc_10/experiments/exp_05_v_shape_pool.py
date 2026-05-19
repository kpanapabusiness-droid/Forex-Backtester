"""EXP-05 — Cross-arc V-shape pool (Arc 7 c3 + Arc 10 c1).

Question: does pooling V-shape cohorts from multiple arcs and training a
single E pipeline produce an AUC that clears 0.65?

In-scope candidates with step1 artefacts available in this branch:
  - Arc 10 c1 — V-shape recovery, n=228, SL=3.0xATR (this arc)
  - Arc 7 c3  — V-shape recovery, n=365, SL=2.0xATR (results/l_arc_7/step1)

OUT OF SCOPE / BLOCKED:
  - Arc 6: only `docs/arc_results/ARC_6_RESULT.md` present in-branch; no
    step1 trades_all.csv / trades_paths.csv. Also Arc 6 was Stepwise climber
    (not V-shape) per its closure doc — wrong archetype for the pool anyway.

Pooling constraints:
  - Different signal classes (DLR vs liquidity-sweep+reclaim) → DLR-specific
    HTF features (L1_value, L0_value, L1_age_d1_bars, etc.) are not defined
    on Arc 7 trades. Use the generic 17-feature subset only — same as
    EXP-02's `base_no_HTF` variant on Arc 10.
  - Different selected SLs (Arc 7 c3 = 2.0, Arc 10 c1 = 3.0) → re-impose a
    single common SL = 3.0×ATR on both pools so success labels are comparable.
    Note: this means Arc 7's success rate may differ from its Step 4 baseline.

Procedure:
  1. Recompute Arc 7 c3 path success labels at SL=3.0×ATR using the same
     _eval_trade_at_sl primitive Step 4 uses.
  2. Compute Pipeline E features for Arc 7 c3 trades using the Arc 10 Step 4
     compute_pipeline_e_features (DLR fields fall through to NaN — those
     columns will not be used for training in this experiment).
  3. Pool with Arc 10 c1. Train RF on the generic 17-feature subset only,
     5-fold TimeSeriesSplit ordered by entry_time across the combined frame.
  4. Compare pooled AUC to per-arc baselines (Arc 10 c1 base_no_HTF, Arc 7
     c3 with the same feature set).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10.experiments._common import (  # noqa: E402
    C1_SL_ATR_MULT,
    ORIGINAL_SL_ATR_MULT,
    PIPELINE_E_FEATURES,
    load_arc10_c1_bundle,
    wf_oof_preds,
    mean_auc_safe,
    sha256_file,
    _build_pair_cache,
    DATA_DIR_4H,
    DATA_DIR_D1,
)
from scripts.l_arc_10.step4_extractability import (  # noqa: E402
    compute_pipeline_e_features,
    compute_success_labels,
    _build_paths_index,
)

OUT_DIR = _REPO_ROOT / "results" / "l_arc_10" / "experiments"
RAW_DIR = OUT_DIR / "raw"

GENERIC_FEATURES = [
    f for f in PIPELINE_E_FEATURES
    if f not in {
        "L1_to_atr_proximity", "reject_buffer_atr", "upper_fraction",
        "L1_age_d1_bars", "L0_age_d1_bars",
        "L1_minus_L0_atr", "L1_minus_L0_d1_bars",
    }
]


def load_arc7_c3_bundle(pair_caches):
    """Load Arc 7 cluster 3 (V-shape, n=365) trades + paths + features."""
    s1 = _REPO_ROOT / "results" / "l_arc_7" / "step1"
    s2 = _REPO_ROOT / "results" / "l_arc_7" / "step2"
    trades = pd.read_csv(s1 / "trades_all.csv")
    # Arc 7 trades_all uses different column names — find entry_time / signal_bar_time.
    # Inspect:
    for col in trades.columns:
        pass  # quiet; just to keep linter happy
    if "entry_time" not in trades.columns:
        # Try alternative names common across arcs.
        for cand in ["entry_date", "entry_timestamp"]:
            if cand in trades.columns:
                trades = trades.rename(columns={cand: "entry_time"})
                break
    if "signal_bar_time" not in trades.columns:
        for cand in ["signal_time", "signal_date", "signal_bar_date"]:
            if cand in trades.columns:
                trades = trades.rename(columns={cand: "signal_bar_time"})
                break
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["signal_bar_time"] = pd.to_datetime(trades["signal_bar_time"])
    paths = pd.read_csv(s1 / "trades_paths.csv")
    clusters = pd.read_csv(s2 / "clusters_K4.csv")
    c3_tids = sorted(clusters[clusters["cluster_id"] == 3]["trade_id"].astype(int).tolist())
    trades_c3 = trades[trades["trade_id"].isin(c3_tids)].reset_index(drop=True)
    paths_c3 = paths[paths["trade_id"].isin(c3_tids)].reset_index(drop=True)
    return trades, trades_c3, paths_c3, c3_tids


def run_pool(label, e_features, y, feature_set):
    op, oy, paf, fs, fm = wf_oof_preds(e_features, y, feature_set)
    m, s = mean_auc_safe(paf)
    return {"label": label, "n": len(y), "base": float(y.mean()),
            "mean_auc": m, "std_auc": s, "per_fold": [float(a) for a in paf],
            "n_features": len(feature_set)}


def main() -> int:
    print("[EXP-05] loading Arc 10 c1 bundle...", file=sys.stderr)
    b10 = load_arc10_c1_bundle()

    print("[EXP-05] loading Arc 7 c3 bundle...", file=sys.stderr)
    trades_a7_full, trades_a7_c3, paths_a7_c3, c3_tids = load_arc7_c3_bundle(b10.pair_caches)

    # Build pair caches for any Arc 7 pairs not in Arc 10's cache.
    extra_pairs = sorted(set(trades_a7_full["pair"].astype(str).unique()) - set(b10.pair_caches.keys()))
    pair_caches = dict(b10.pair_caches)
    for p in extra_pairs:
        print(f"  caching extra pair {p}", file=sys.stderr)
        pair_caches[p] = _build_pair_cache(p, DATA_DIR_4H, DATA_DIR_D1)

    print("[EXP-05] computing Arc 7 c3 Pipeline E features (DLR fields NaN)...",
          file=sys.stderr)
    # Arc 7 trades lack DLR-specific columns (L1_value etc.). The Arc 10
    # compute_pipeline_e_features uses t.get(col, NaN) so missing columns
    # become NaN. We then restrict training to GENERIC_FEATURES.
    # Compute on the FULL Arc 7 trades_full so pair_id_int encoding is
    # alphabetical over all Arc 7 pairs (matches Arc 7's Step 4 convention
    # for its own classifier, irrelevant here since pair_id_int IS in
    # GENERIC_FEATURES but we will re-encode below to ensure cross-arc consistency).
    e_full_a7 = compute_pipeline_e_features(trades_a7_full, pair_caches)
    e_a7_c3 = (
        e_full_a7[e_full_a7["trade_id"].isin(c3_tids)]
        .sort_values("entry_time", kind="mergesort")
        .reset_index(drop=True)
    )

    # Recompute Arc 7 c3 success labels at the common SL = 3.0×ATR.
    paths_index_a7 = _build_paths_index(paths_a7_c3)
    y_a7_c3_dict = compute_success_labels(
        c3_tids, paths_index_a7, C1_SL_ATR_MULT, ORIGINAL_SL_ATR_MULT
    )
    y_a7_c3 = np.array([y_a7_c3_dict[int(tid)] for tid in e_a7_c3["trade_id"]], dtype=int)

    # Per-arc baselines (generic features only).
    print("[EXP-05] baseline: Arc 10 c1 (generic features only)...", file=sys.stderr)
    r_a10 = run_pool("Arc 10 c1 alone (generic features, SL=3.0)",
                     b10.e_features, b10.y, GENERIC_FEATURES)

    print("[EXP-05] baseline: Arc 7 c3 alone (generic features, SL=3.0 re-imposed)...",
          file=sys.stderr)
    r_a7 = run_pool("Arc 7 c3 alone (generic features, SL=3.0 re-imposed)",
                    e_a7_c3, y_a7_c3, GENERIC_FEATURES)

    # Pool: combine, re-encode pair_id_int alphabetically across the union.
    print("[EXP-05] pooling Arc 7 c3 + Arc 10 c1...", file=sys.stderr)
    # Tag arc_source for cross-arc pair_id_int re-encoding and inspection.
    a10_pool = b10.e_features.copy()
    a10_pool["arc_source"] = "Arc 10"
    a7_pool = e_a7_c3.copy()
    a7_pool["arc_source"] = "Arc 7"
    # Ensure both frames have the same columns (DLR fields will be NaN on Arc 7).
    union_cols = sorted(set(a10_pool.columns) | set(a7_pool.columns))
    for c in union_cols:
        if c not in a10_pool.columns:
            a10_pool[c] = float("nan")
        if c not in a7_pool.columns:
            a7_pool[c] = float("nan")
    pooled = pd.concat([a10_pool, a7_pool], ignore_index=True, sort=False)
    # Re-encode pair_id_int over the pooled pair set (alphabetical).
    pairs_pool = sorted(pooled["pair"].astype(str).unique())
    pid_map = {p: i for i, p in enumerate(pairs_pool)}
    pooled["pair_id_int"] = pooled["pair"].map(pid_map).astype(int)
    pooled = pooled.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    # Stitch labels.
    a10_pairs = list(zip(b10.e_features["trade_id"].astype(int).tolist(),
                          ["Arc 10"] * len(b10.e_features)))
    a10_y = {int(tid): int(b10.y[i]) for i, tid in enumerate(b10.e_features["trade_id"])}
    a7_y = {int(tid): int(y_a7_c3[i]) for i, tid in enumerate(e_a7_c3["trade_id"])}
    # Compose pooled y in pooled-order.
    y_pool = np.array(
        [
            a10_y[int(r["trade_id"])] if r["arc_source"] == "Arc 10" else a7_y[int(r["trade_id"])]
            for _, r in pooled.iterrows()
        ],
        dtype=int,
    )

    r_pool = run_pool("Pool (Arc 7 c3 + Arc 10 c1) (generic features, SL=3.0)",
                       pooled, y_pool, GENERIC_FEATURES)

    rows = [r_a10, r_a7, r_pool]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW_DIR / "exp_05_pool_results.csv",
                              index=False, lineterminator="\n")

    md = []
    md.append("# EXP-05 — Cross-arc V-shape pool")
    md.append("")
    md.append("**Status:** experimental (not a Step 5 gate).")
    md.append("")
    md.append("## Question")
    md.append("Does pooling V-shape cohorts from multiple arcs and training a single E")
    md.append("pipeline produce an AUC that clears 0.65?")
    md.append("")
    md.append("## Scope and BLOCKED items")
    md.append("- **Arc 7 c3** (V-shape recovery, n=365, original SL=2.0×ATR): step1 artefacts in-branch.")
    md.append("- **Arc 10 c1** (V-shape recovery, n=228, SL=3.0×ATR): this arc.")
    md.append("- **BLOCKED — Arc 6**: only `ARC_6_RESULT.md` doc in-branch; no step1 trades_all.csv. ")
    md.append("  Also Arc 6 was Stepwise climber per its closure doc — wrong archetype for a V-shape pool.")
    md.append("")
    md.append("## Constraints handled")
    md.append("- **Mixed signal classes** (DLR vs liquidity-sweep+reclaim): the Arc 10 Pipeline E feature")
    md.append("  set includes 8 DLR-specific HTF features that are undefined on Arc 7 trades. Training uses")
    md.append(f"  only the **{len(GENERIC_FEATURES)} generic features** (volatility / EMA / range / bar shape / cyclic time / pair).")
    md.append("- **Different selected SLs** (Arc 7 c3 = 2.0×ATR, Arc 10 c1 = 3.0×ATR): re-imposed Arc 10's")
    md.append("  SL = 3.0×ATR on Arc 7 c3 so success labels are comparable. Arc 7 c3's base success rate")
    md.append("  under SL=3.0×ATR may differ from its original Step 4 baseline at SL=2.0×ATR.")
    md.append("- **Pair encoding**: pair_id_int re-encoded alphabetically across the union of arc pair sets.")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append("| Pool | n | base success | mean AUC | std | per-fold AUC |")
    md.append("|---|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(
            f"| {r['label']} | {r['n']} | {r['base']:.4f} | "
            f"{r['mean_auc']:.4f} | {r['std_auc']:.4f} | "
            f"{[round(a,4) for a in r['per_fold']]} |"
        )
    md.append("")
    md.append("## Interpretation")
    pool_auc = r_pool["mean_auc"]
    if pool_auc >= 0.65:
        md.append(
            f"- **Pooled AUC {pool_auc:.4f} clears 0.65 with generic features alone.** Strong evidence "
            f"for treating V-shape as a cross-arc archetype with deployable structure. Flag for v2.4 calibration."
        )
    else:
        gap = 0.65 - pool_auc
        md.append(
            f"- Pooled AUC **{pool_auc:.4f}** does not clear 0.65 (gap {gap:.4f}). "
            f"Pooling does not rescue extractability — V-shape geometry is not a self-sufficient "
            f"feature class on the generic 17-feature set, even at the larger pooled n={r_pool['n']}."
        )
    a10_to_pool = pool_auc - r_a10["mean_auc"]
    a7_to_pool = pool_auc - r_a7["mean_auc"]
    md.append(
        f"- Pooling delta vs Arc 10 c1 baseline: **{a10_to_pool:+.4f}** "
        f"({r_a10['mean_auc']:.4f} → {pool_auc:.4f})."
    )
    md.append(
        f"- Pooling delta vs Arc 7 c3 baseline (re-imposed SL=3.0): **{a7_to_pool:+.4f}** "
        f"({r_a7['mean_auc']:.4f} → {pool_auc:.4f})."
    )
    md.append("- Caveat: pool n=593 is still small for a 5-fold TimeSeriesSplit; per-fold AUC variance dominates.")
    md.append("")
    md.append("## Artefacts")
    md.append(f"- `raw/exp_05_pool_results.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_05_pool_results.csv')[:16]}…`)")
    md.append("")

    (OUT_DIR / "EXP_05_v_shape_pool.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[EXP-05] wrote {OUT_DIR / 'EXP_05_v_shape_pool.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

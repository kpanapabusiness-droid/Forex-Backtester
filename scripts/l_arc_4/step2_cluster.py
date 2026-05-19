"""Arc 4 — Step 2 path-shape clustering.

L_ARC_PROTOCOL v2.1.1 §6. Outcome-blind clustering of Step 1's trade pool on
four path-shape features computed over held bars only (is_held=1). Forward
observation bars (is_held=0) are reserved for §7 SL sweep at Step 3 and MUST
NOT enter feature computation.

Path-shape features (§6 / §17):
  1. monotonicity_ratio_in_profit
  2. local_peaks_count
  3. pullback_magnitude_median  (operational def: min close_r between peaks)
  4. time_to_peak_mfe_relative

Clustering: KMeans(random_state=42, n_init=10, max_iter=300) on
StandardScaler-transformed features. K ∈ {3, 4, 5, 6, 7}.

Per-K gate (conjunctive):
  silhouette ≥ 0.30
  no cluster > 90% of trades
  all clusters ≥ 30 trades

K selection: highest silhouette among passing; smaller K wins within an
absolute silhouette tolerance of 0.01 (§6 tolerance / Open-12).

Archetype labelling (chosen K only, centroid-only at Step 2 — forward-outcome
metrics like pct_peak_and_collapse, fwd_mfe_p50, bimodal_separated are
DEFERRED to Step 3 per the Arc 4 prompt and §11):
  - Centroid rules from §11 rows 1-6 evaluated on the four-feature centroid.
  - Rule parts that reference non-centroid features (fwd_mfe_p50,
    pct_peak_and_collapse, MAE-before-peak, peak-position-in-trade) are
    treated as "deferred to Step 3"; a cluster that satisfies the
    centroid-feature parts of multiple §11 rows is labelled "boundary"
    with all candidate matches reported.
  - Single clean match → certainty = "clean".
  - Multiple candidate rows (including deferred-disambiguation) → "boundary".
  - No row's centroid-features match → certainty = "unclassified".
  - Same-archetype clusters (≥ 2 clusters share the clean label) flagged for
    §7 per-aggregate evaluation at Step 3.

Outputs (results/l_arc_4/step2/):
  - path_features.csv
  - clusters_K{3..7}.csv
  - centroids_K{3..7}.csv  (raw + standardised columns)
  - silhouette_K{3..7}.txt
  - archetype_assignments.csv  (chosen K only)
  - step2_diagnostics.md

Determinism: byte-identical on re-run for ALL written files. Both runs'
sha256s logged in step2_diagnostics.md.

Usage:
  py scripts/l_arc_4/step2_clustering.py -c configs/l_arc_4.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLS: Tuple[str, ...] = (
    "monotonicity_ratio_in_profit",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
)
SHORT_FEATURE_NAMES: Tuple[str, ...] = (
    "mono",
    "local_peaks",
    "pullback",
    "time_to_peak",
)

K_SWEEP: Tuple[int, ...] = (3, 4, 5, 6, 7)

DEGENERACY_THRESHOLD: float = 0.80
MIN_CLUSTER_SIZE: int = 30
MAX_CLUSTER_FRACTION: float = 0.90
SILHOUETTE_FLOOR: float = 0.30
K_SILHOUETTE_TOLERANCE: float = 0.01

KMEANS_RANDOM_STATE: int = 42
KMEANS_N_INIT: int = 10
KMEANS_MAX_ITER: int = 300


# ---------------------------------------------------------------------------
# Path-shape feature computation per trade (held bars only).
# ---------------------------------------------------------------------------


def _features_for_trade(paths_g: pd.DataFrame, bars_held_authoritative: int) -> Dict[str, float]:
    """Compute the four v2.1.1 §6 path-shape features for one trade.

    `paths_g` has rows sorted by bar_offset for one trade_id, RESTRICTED TO
    is_held=1 rows (held + exit bar). Forward observation bars must already
    be filtered out by the caller.

    bars_held_authoritative is the trade's bars_held from trades_all.csv —
    used directly for time_to_peak_mfe_relative's denominator.
    """
    if len(paths_g) == 0:
        return {
            "monotonicity_ratio_in_profit": 0.0,
            "local_peaks_count": 0,
            "pullback_magnitude_median": 0.0,
            "time_to_peak_mfe_relative": 0.0,
        }

    close_r = paths_g["close_r"].to_numpy(dtype=float)
    mfe = paths_g["mfe_so_far_r"].to_numpy(dtype=float)
    bar_offset = paths_g["bar_offset"].to_numpy(dtype=int)

    # Feature 1: monotonicity_ratio_in_profit
    # "among bars where close_r > 0, fraction where close_r ≥ previous in-profit bar"
    # Edge case (§6): zero in-profit bars → 0. We extend: zero comparable pairs
    # (i.e., exactly 1 in-profit bar) → 0 (matches §6 spirit — no signal of
    # monotonicity from a single in-profit observation).
    in_profit = close_r > 0.0
    if int(in_profit.sum()) == 0:
        monotonicity = 0.0
    else:
        in_profit_closes = close_r[in_profit]
        if in_profit_closes.size <= 1:
            monotonicity = 0.0
        else:
            gte = in_profit_closes[1:] >= in_profit_closes[:-1]
            monotonicity = float(gte.sum() / gte.size)

    # Feature 2: local_peaks_count — count of bars where mfe_so_far_r > previous bar.
    # §6 edge case: bars_held = 0 → 0 (handled by mfe.size <= 1).
    if mfe.size <= 1:
        local_peaks = 0
    else:
        local_peaks = int(np.sum(mfe[1:] > mfe[:-1]))

    # Feature 3: pullback_magnitude_median — for consecutive peak pairs,
    # earlier_peak's mfe_so_far_r − min(close_r between peaks); median.
    # §6 edge case: < 2 peaks → 0.
    peak_positions = np.where(mfe[1:] > mfe[:-1])[0] + 1
    if peak_positions.size < 2:
        pullback_median = 0.0
    else:
        diffs: List[float] = []
        for i in range(len(peak_positions) - 1):
            p1 = int(peak_positions[i])
            p2 = int(peak_positions[i + 1])
            if p2 - p1 < 2:
                continue
            min_close_r_between = float(np.min(close_r[p1 + 1 : p2]))
            earlier_peak_mfe = float(mfe[p1])
            diffs.append(earlier_peak_mfe - min_close_r_between)
        pullback_median = float(np.median(diffs)) if diffs else 0.0

    # Feature 4: time_to_peak_mfe_relative — time_to_peak_mfe / max(bars_held, 1).
    # §6 edge case: trade never in profit → 0.
    if mfe.size == 0 or float(mfe.max()) <= 0.0:
        time_to_peak_rel = 0.0
    else:
        peak_value = float(mfe.max())
        first_peak_idx = int(np.argmax(mfe >= peak_value - 1e-12))
        ttp = int(bar_offset[first_peak_idx])
        bars_held_denom = max(int(bars_held_authoritative), 1)
        time_to_peak_rel = min(float(ttp) / float(bars_held_denom), 1.0)

    return {
        "monotonicity_ratio_in_profit": float(monotonicity),
        "local_peaks_count": int(local_peaks),
        "pullback_magnitude_median": float(pullback_median),
        "time_to_peak_mfe_relative": float(time_to_peak_rel),
    }


def compute_path_features(paths_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trade features on held bars only.

    Caller passes the full paths_df (with both is_held=0 and is_held=1 rows);
    this function filters to is_held=1 before feature computation. The result
    DataFrame carries the authoritative bars_held + final_r columns from
    trades_df for downstream inspection (final_r informational only).
    """
    if "is_held" not in paths_df.columns:
        raise ValueError(
            "trades_paths.csv missing required column 'is_held' (v2.1.1 §5 schema)"
        )
    held = paths_df[paths_df["is_held"] == 1].copy()
    held = held.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)

    # Build a per-trade lookup of bars_held from trades_all.csv.
    bh_lookup = dict(
        zip(
            trades_df["trade_id"].to_numpy(dtype=int),
            trades_df["bars_held"].to_numpy(dtype=int),
        )
    )
    final_r_lookup = dict(
        zip(
            trades_df["trade_id"].to_numpy(dtype=int),
            trades_df["final_r"].to_numpy(dtype=float),
        )
    )

    feature_rows: List[Dict[str, Any]] = []
    paths_by_trade = {tid: g for tid, g in held.groupby("trade_id", sort=False)}

    for tid_raw in trades_df["trade_id"].to_numpy():
        tid = int(tid_raw)
        g = paths_by_trade.get(tid)
        bh = int(bh_lookup.get(tid, 0))
        if g is None or len(g) == 0:
            feats = _features_for_trade(pd.DataFrame(columns=held.columns), bh)
        else:
            feats = _features_for_trade(g, bh)
        feature_rows.append(
            {
                "trade_id": tid,
                **feats,
                "bars_held": bh,
                "final_r": float(final_r_lookup.get(tid, float("nan"))),
            }
        )

    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Degeneracy audit (§6).
# ---------------------------------------------------------------------------


def _degeneracy_audit(features: pd.DataFrame) -> Dict[str, Any]:
    """Per-feature modal-value fraction. Any feature > 80% at single value
    flags; ≥ 2 flagged features → halt arc.
    """
    audit: Dict[str, Any] = {}
    for col in FEATURE_COLS:
        s = features[col].to_numpy()
        n = int(len(s))
        if col == "local_peaks_count":
            counts = pd.Series(s).value_counts()
            modal_value = float(counts.index[0])
            modal_count = int(counts.iloc[0])
        else:
            bucket = np.round(s.astype(float), 9)
            counts = pd.Series(bucket).value_counts()
            modal_value = float(counts.index[0])
            modal_count = int(counts.iloc[0])
        frac = modal_count / n if n > 0 else 0.0
        audit[col] = {
            "modal_value": modal_value,
            "modal_count": modal_count,
            "modal_fraction": float(frac),
            "is_degenerate_gt_80pct": bool(frac > DEGENERACY_THRESHOLD),
        }
    degenerate_cols = [c for c in FEATURE_COLS if audit[c]["is_degenerate_gt_80pct"]]
    audit["_summary"] = {
        "degenerate_features": degenerate_cols,
        "count": len(degenerate_cols),
        "arc_halt": len(degenerate_cols) >= 2,
    }
    return audit


# ---------------------------------------------------------------------------
# K-sweep.
# ---------------------------------------------------------------------------


@dataclass
class _KResult:
    K: int
    labels: np.ndarray
    centroids_orig: np.ndarray          # K × 4 (raw feature space)
    centroids_standardised: np.ndarray  # K × 4 (StandardScaler space)
    sizes: np.ndarray
    silhouette: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray

    @property
    def size_fractions(self) -> np.ndarray:
        n = int(self.sizes.sum())
        return self.sizes / n if n > 0 else np.zeros_like(self.sizes, dtype=float)


def _fit_kmeans(features: pd.DataFrame, K: int) -> _KResult:
    X = features[list(FEATURE_COLS)].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(
        n_clusters=K,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    )
    labels = km.fit_predict(X_scaled)
    sil = float(silhouette_score(X_scaled, labels))

    centroids_orig = np.zeros((K, len(FEATURE_COLS)), dtype=float)
    centroids_std = np.zeros((K, len(FEATURE_COLS)), dtype=float)
    sizes = np.zeros(K, dtype=int)
    for k in range(K):
        mask = labels == k
        sizes[k] = int(mask.sum())
        if sizes[k] > 0:
            centroids_orig[k] = X[mask].mean(axis=0)
            centroids_std[k] = X_scaled[mask].mean(axis=0)
    return _KResult(
        K=K,
        labels=labels,
        centroids_orig=centroids_orig,
        centroids_standardised=centroids_std,
        sizes=sizes,
        silhouette=sil,
        scaler_mean=np.asarray(scaler.mean_, dtype=float),
        scaler_scale=np.asarray(scaler.scale_, dtype=float),
    )


def _gate_check(res: _KResult) -> Tuple[bool, Dict[str, Any]]:
    n_total = int(res.sizes.sum())
    min_size = int(res.sizes.min())
    max_size = int(res.sizes.max())
    min_frac = float(res.sizes.min() / n_total) if n_total > 0 else 0.0
    max_frac = float(res.sizes.max() / n_total) if n_total > 0 else 0.0
    cond_sil = res.silhouette >= SILHOUETTE_FLOOR
    cond_min = min_size >= MIN_CLUSTER_SIZE
    cond_max = max_frac <= MAX_CLUSTER_FRACTION
    passes = bool(cond_sil and cond_min and cond_max)
    reasons: List[str] = []
    if not cond_sil:
        reasons.append(f"silhouette {res.silhouette:.4f} < {SILHOUETTE_FLOOR}")
    if not cond_min:
        reasons.append(f"min cluster size {min_size} < {MIN_CLUSTER_SIZE}")
    if not cond_max:
        reasons.append(f"max cluster fraction {max_frac:.4f} > {MAX_CLUSTER_FRACTION}")
    return passes, {
        "silhouette": float(res.silhouette),
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "min_size_fraction": min_frac,
        "max_size_fraction": max_frac,
        "passes_silhouette": bool(cond_sil),
        "passes_min_size": bool(cond_min),
        "passes_max_fraction": bool(cond_max),
        "passes_all": passes,
        "fail_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# §11 centroid-only archetype matching (Step 2 — forward outcomes deferred).
# ---------------------------------------------------------------------------


@dataclass
class CentroidMatch:
    archetype_label: str
    row: int                              # 1..6
    centroid_rule_text: str               # what the centroid-features part says
    centroid_satisfied: bool
    deferred_to_step3: List[str]          # rule parts that need forward outcomes
    unmet: List[str]                      # centroid-feature parts not satisfied


def _match_centroid_to_archetypes(
    mono: float, peaks: float, pullback: float, ttp_rel: float
) -> List[CentroidMatch]:
    """Evaluate the four-feature centroid against §11 rows 1-6 patterns.

    Rules with forward-outcome parts (rows 3, 4, 5) record those parts under
    `deferred_to_step3`; centroid_satisfied reflects only the centroid-feature
    parts (and is False if the row has no centroid-feature parts at all, as
    with row 5).
    """
    out: List[CentroidMatch] = []

    # Row 1 — Monotone ascent: mono ≥ 0.55 AND local_peaks ≤ 4 AND time_to_peak_rel ≥ 0.50
    unmet: List[str] = []
    if not (mono >= 0.55):
        unmet.append(f"mono {mono:.3f} < 0.55")
    if not (peaks <= 4):
        unmet.append(f"local_peaks {peaks:.2f} > 4")
    if not (ttp_rel >= 0.50):
        unmet.append(f"time_to_peak_rel {ttp_rel:.3f} < 0.50")
    out.append(
        CentroidMatch(
            archetype_label="Monotone ascent",
            row=1,
            centroid_rule_text="mono ≥ 0.55 AND local_peaks ≤ 4 AND time_to_peak_rel ≥ 0.50",
            centroid_satisfied=(len(unmet) == 0),
            deferred_to_step3=[],
            unmet=unmet,
        )
    )

    # Row 2 — Stepwise climber: mono ≥ 0.50 AND local_peaks 5-30 AND pullback ≤ 0.5R AND time_to_peak_rel ≥ 0.50
    unmet = []
    if not (mono >= 0.50):
        unmet.append(f"mono {mono:.3f} < 0.50")
    if not (5 <= peaks <= 30):
        unmet.append(f"local_peaks {peaks:.2f} not in [5, 30]")
    if not (pullback <= 0.5):
        unmet.append(f"pullback {pullback:.3f} > 0.5R")
    if not (ttp_rel >= 0.50):
        unmet.append(f"time_to_peak_rel {ttp_rel:.3f} < 0.50")
    out.append(
        CentroidMatch(
            archetype_label="Stepwise climber",
            row=2,
            centroid_rule_text="mono ≥ 0.50 AND local_peaks ∈ [5, 30] AND pullback ≤ 0.5R AND time_to_peak_rel ≥ 0.50",
            centroid_satisfied=(len(unmet) == 0),
            deferred_to_step3=[],
            unmet=unmet,
        )
    )

    # Row 3 — Early-peak hold: time_to_peak_rel ≤ 0.30 AND fwd_mfe_p50 ≥ 1.5R AND pct_peak_and_collapse < 0.30
    unmet = []
    if not (ttp_rel <= 0.30):
        unmet.append(f"time_to_peak_rel {ttp_rel:.3f} > 0.30")
    out.append(
        CentroidMatch(
            archetype_label="Early-peak hold",
            row=3,
            centroid_rule_text="time_to_peak_rel ≤ 0.30  (+ fwd_mfe_p50 ≥ 1.5R + pct_peak_and_collapse < 0.30 — deferred)",
            centroid_satisfied=(len(unmet) == 0),
            deferred_to_step3=["fwd_mfe_p50 ≥ 1.5R", "pct_peak_and_collapse < 0.30"],
            unmet=unmet,
        )
    )

    # Row 4 — Peak-and-collapse: time_to_peak_rel ≤ 0.30 AND pct_peak_and_collapse ≥ 0.50
    unmet = []
    if not (ttp_rel <= 0.30):
        unmet.append(f"time_to_peak_rel {ttp_rel:.3f} > 0.30")
    out.append(
        CentroidMatch(
            archetype_label="Peak-and-collapse",
            row=4,
            centroid_rule_text="time_to_peak_rel ≤ 0.30  (+ pct_peak_and_collapse ≥ 0.50 — deferred)",
            centroid_satisfied=(len(unmet) == 0),
            deferred_to_step3=["pct_peak_and_collapse ≥ 0.50"],
            unmet=unmet,
        )
    )

    # Row 5 — V-shape recovery: MAE early AND peak position in [0.4, 0.8] of trade.
    # NOT centroid-matchable from the 4-feature centroid alone — MAE timing
    # and peak position are non-centroid path properties (could be computed
    # at Step 3 from path data + per-trade MAE timing if needed).
    out.append(
        CentroidMatch(
            archetype_label="V-shape recovery",
            row=5,
            centroid_rule_text="MAE-before-peak ≥ 5 bars AND peak in [0.4, 0.8] of trade  (not expressible from 4-feature centroid)",
            centroid_satisfied=False,
            deferred_to_step3=["MAE-before-peak ≥ 5 bars", "peak position ∈ [0.4, 0.8]"],
            unmet=["row 5 is not centroid-matchable from the 4-feature centroid"],
        )
    )

    # Row 6 — Random walk: local_peaks ≥ 8 AND mono ≤ 0.30 AND pullback ≥ 1R
    unmet = []
    if not (peaks >= 8):
        unmet.append(f"local_peaks {peaks:.2f} < 8")
    if not (mono <= 0.30):
        unmet.append(f"mono {mono:.3f} > 0.30")
    if not (pullback >= 1.0):
        unmet.append(f"pullback {pullback:.3f} < 1.0R")
    out.append(
        CentroidMatch(
            archetype_label="Random walk",
            row=6,
            centroid_rule_text="local_peaks ≥ 8 AND mono ≤ 0.30 AND pullback ≥ 1R",
            centroid_satisfied=(len(unmet) == 0),
            deferred_to_step3=[],
            unmet=unmet,
        )
    )

    return out


@dataclass
class ClusterAssignment:
    cluster_id: int
    archetype_label: str        # joined "row3+row4" syntax when multiple candidates
    archetype_row: str          # joined "3+4" syntax when multiple candidates
    assignment_certainty: str   # "clean" | "boundary" | "unclassified"
    candidate_matches: List[CentroidMatch]
    reason: str
    aggregation_partners: List[int]


def _assign_cluster_label(
    cluster_id: int, matches: List[CentroidMatch]
) -> ClusterAssignment:
    """Apply §6/§11 centroid-only labelling.

    Note: row 5 (V-shape recovery) is structurally not-centroid-matchable; it
    is excluded from the candidate set for clean/boundary disposition (it can
    only be examined at Step 3 with additional path-feature analysis).
    """
    centroid_satisfied = [m for m in matches if m.centroid_satisfied]
    # Row 5 is never centroid_satisfied=True so it never appears here.

    # Disjoint structural overlaps (sanity):
    # rows 1, 2, 6 are mutually disjoint on local_peaks.
    # rows 1/2 and rows 3/4 are mutually disjoint on time_to_peak_rel.
    # rows 3 and 4 share `time_to_peak_rel ≤ 0.30` as their centroid-feature
    # part — both will satisfy together when ttp_rel ≤ 0.30. The forward-outcome
    # disambiguator (pct_peak_and_collapse) is deferred to Step 3.

    if len(centroid_satisfied) == 1:
        m = centroid_satisfied[0]
        return ClusterAssignment(
            cluster_id=cluster_id,
            archetype_label=m.archetype_label,
            archetype_row=str(m.row),
            assignment_certainty="clean"
            if not m.deferred_to_step3
            else "boundary",
            candidate_matches=centroid_satisfied,
            reason=(
                "clean centroid match"
                if not m.deferred_to_step3
                else f"single-candidate match; deferred to Step 3: {', '.join(m.deferred_to_step3)}"
            ),
            aggregation_partners=[],
        )

    if len(centroid_satisfied) >= 2:
        labels = [m.archetype_label for m in centroid_satisfied]
        rows = [m.row for m in centroid_satisfied]
        deferred_parts = sorted({d for m in centroid_satisfied for d in m.deferred_to_step3})
        reason = (
            f"multiple centroid-feature matches: {', '.join(labels)}; "
            f"Step 3 to disambiguate via: {', '.join(deferred_parts) if deferred_parts else 'empirical capture-ratio test'}"
        )
        return ClusterAssignment(
            cluster_id=cluster_id,
            archetype_label=" + ".join(labels),
            archetype_row="+".join(str(r) for r in rows),
            assignment_certainty="boundary",
            candidate_matches=centroid_satisfied,
            reason=reason,
            aggregation_partners=[],
        )

    return ClusterAssignment(
        cluster_id=cluster_id,
        archetype_label="unclassified",
        archetype_row="",
        assignment_certainty="unclassified",
        candidate_matches=[],
        reason="no §11 row's centroid-feature part is satisfied — route to §6 boundary test at Step 3",
        aggregation_partners=[],
    )


def _attach_aggregation_partners(assignments: List[ClusterAssignment]) -> None:
    """If two clusters share the same CLEAN archetype label, flag aggregation
    partners. (Boundary / unclassified clusters are not aggregated at this
    step — Step 3 disambiguates.)
    """
    by_label: Dict[str, List[int]] = {}
    for a in assignments:
        if a.assignment_certainty != "clean":
            continue
        by_label.setdefault(a.archetype_label, []).append(a.cluster_id)
    for a in assignments:
        if a.assignment_certainty != "clean":
            continue
        partners = [cid for cid in by_label.get(a.archetype_label, []) if cid != a.cluster_id]
        a.aggregation_partners = partners


# ---------------------------------------------------------------------------
# Output writers (deterministic).
# ---------------------------------------------------------------------------


def _write_path_features(features: pd.DataFrame, path: Path, float_fmt: str) -> None:
    cols = ["trade_id", *FEATURE_COLS, "bars_held", "final_r"]
    features[cols].to_csv(
        path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n"
    )


def _write_clusters(features: pd.DataFrame, res: _KResult, path: Path) -> None:
    df = pd.DataFrame(
        {
            "trade_id": features["trade_id"].to_numpy(dtype=int),
            "cluster_id": res.labels.astype(int),
        }
    )
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _write_centroids(res: _KResult, path: Path, float_fmt: str) -> None:
    n_total = int(res.sizes.sum())
    rows: List[Dict[str, Any]] = []
    for k in range(res.K):
        row: Dict[str, Any] = {
            "cluster_id": k,
            "size": int(res.sizes[k]),
            "size_fraction": float(res.sizes[k] / n_total) if n_total > 0 else 0.0,
        }
        for i, short in enumerate(SHORT_FEATURE_NAMES):
            row[f"{short}_raw"] = float(res.centroids_orig[k, i])
        for i, short in enumerate(SHORT_FEATURE_NAMES):
            row[f"{short}_standardised"] = float(res.centroids_standardised[k, i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n"
    )


def _write_silhouette(s: float, path: Path) -> None:
    path.write_text(f"{s:.10g}\n", encoding="utf-8")


def _write_archetype_assignments(
    chosen_K: int,
    res: _KResult,
    assignments: List[ClusterAssignment],
    path: Path,
    float_fmt: str,
) -> None:
    n_total = int(res.sizes.sum())
    rows: List[Dict[str, Any]] = []
    for a in assignments:
        k = a.cluster_id
        rows.append(
            {
                "chosen_K": int(chosen_K),
                "cluster_id": int(k),
                "size": int(res.sizes[k]),
                "size_fraction": float(res.sizes[k] / n_total) if n_total > 0 else 0.0,
                "centroid_mono": float(res.centroids_orig[k, 0]),
                "centroid_local_peaks": float(res.centroids_orig[k, 1]),
                "centroid_pullback": float(res.centroids_orig[k, 2]),
                "centroid_time_to_peak": float(res.centroids_orig[k, 3]),
                "archetype_label": a.archetype_label,
                "archetype_row": a.archetype_row,
                "aggregation_partners": ";".join(str(p) for p in a.aggregation_partners),
                "assignment_certainty": a.assignment_certainty,
                "reason": a.reason,
            }
        )
    pd.DataFrame(rows).to_csv(
        path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n"
    )


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Diagnostics markdown writer.
# ---------------------------------------------------------------------------


def _percentile(arr: np.ndarray, p: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def _feature_distribution_table(features: pd.DataFrame) -> List[str]:
    rows: List[str] = []
    rows.append("| Feature | p5 | p25 | p50 | p75 | p95 | mode | mode_pct |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for col in FEATURE_COLS:
        s = features[col].to_numpy()
        p5 = _percentile(s, 5)
        p25 = _percentile(s, 25)
        p50 = _percentile(s, 50)
        p75 = _percentile(s, 75)
        p95 = _percentile(s, 95)
        if col == "local_peaks_count":
            counts = pd.Series(s).value_counts()
            modal_value = float(counts.index[0])
            modal_count = int(counts.iloc[0])
        else:
            bucket = np.round(s.astype(float), 9)
            counts = pd.Series(bucket).value_counts()
            modal_value = float(counts.index[0])
            modal_count = int(counts.iloc[0])
        mode_pct = modal_count / len(s) if len(s) > 0 else 0.0
        rows.append(
            f"| {col} | {p5:.4g} | {p25:.4g} | {p50:.4g} | {p75:.4g} | "
            f"{p95:.4g} | {modal_value:.4g} | {mode_pct:.2%} |"
        )
    return rows


def write_diagnostics(
    out_path: Path,
    features: pd.DataFrame,
    degeneracy: Dict[str, Any],
    k_results: Dict[int, _KResult],
    per_k_summary: Dict[int, Dict[str, Any]],
    passing_ks: List[int],
    chosen_k: Optional[int],
    assignments: Optional[List[ClusterAssignment]],
    sha_run1: Dict[str, str],
    sha_run2: Optional[Dict[str, str]],
    determinism_gate: str,
    config_paths: Dict[str, str],
    config_shas: Dict[str, str],
    arc_status: str,
    halt_reason: Optional[str],
) -> None:
    pool_size = int(len(features))

    lines: List[str] = []
    lines.append("# Arc 4 — Step 2 path-shape clustering diagnostics")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §6")
    lines.append(
        "Signal:   `TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001` "
        "(LCHAR_TOPN_REGISTRY.md Entry 4)"
    )
    lines.append("")

    # Headline.
    if arc_status == "PASS":
        chosen_res = k_results[chosen_k]
        chosen_sil = chosen_res.silhouette
        labels_summary: Dict[str, int] = {}
        for a in assignments or []:
            labels_summary[a.archetype_label] = labels_summary.get(a.archetype_label, 0) + 1
        sorted_labels = sorted(labels_summary.items(), key=lambda kv: -kv[1])
        labels_str = ", ".join(f"{lbl} × {cnt}" for lbl, cnt in sorted_labels)
        cleans = sum(1 for a in assignments or [] if a.assignment_certainty == "clean")
        boundaries = sum(1 for a in assignments or [] if a.assignment_certainty == "boundary")
        unclassifieds = sum(
            1 for a in assignments or [] if a.assignment_certainty == "unclassified"
        )
        lines.append("## Summary")
        lines.append("")
        lines.append(
            f"Pool **{pool_size}** trades clustered on 4 path-shape features (held bars only). "
            f"K-sweep gate disposition: **PASS** ({len(passing_ks)} of {len(K_SWEEP)} K passed). "
            f"Chosen K = **{chosen_k}** (silhouette {chosen_sil:.4f}). "
            f"Archetype assignments: {cleans} clean / {boundaries} boundary / "
            f"{unclassifieds} unclassified — {labels_str}. "
            f"Determinism: **{determinism_gate}**. Step 2 disposition: **PASS**."
        )
    else:
        lines.append("## Summary")
        lines.append("")
        lines.append(
            f"Pool **{pool_size}** trades. Step 2 disposition: **FAIL** "
            f"({halt_reason})."
        )
    lines.append("")

    # Feature distributions.
    lines.append("## Feature distributions (held bars only, is_held=1)")
    lines.append("")
    lines.extend(_feature_distribution_table(features))
    lines.append("")

    # Degeneracy audit.
    lines.append("## Degenerate-feature audit (§6)")
    lines.append("")
    lines.append("Per-feature modal-value mass; > 80% at a single value flags. "
                 "2+ flagged features halt the arc.")
    lines.append("")
    lines.append("| Feature | Modal value | Modal count | Modal fraction | Result |")
    lines.append("|---|---:|---:|---:|---|")
    for col in FEATURE_COLS:
        d = degeneracy[col]
        verdict = "FLAG" if d["is_degenerate_gt_80pct"] else "PASS"
        lines.append(
            f"| {col} | {d['modal_value']:.4g} | {d['modal_count']:,} | "
            f"{d['modal_fraction']:.2%} | {verdict} |"
        )
    lines.append("")
    summary = degeneracy["_summary"]
    if summary["count"] == 0:
        lines.append("No degenerate features. Proceeded to clustering.")
    elif summary["count"] == 1:
        lines.append(
            f"One degenerate feature flagged: `{summary['degenerate_features'][0]}`. "
            "Below the 2-flag halt threshold — clustering proceeds; cluster centroids "
            "may underweight the flagged feature."
        )
    else:
        lines.append(
            f"{summary['count']} degenerate features flagged "
            f"({', '.join(summary['degenerate_features'])}) — **arc halted per §6**."
        )
    lines.append("")

    if arc_status != "PASS":
        # Even when halted, write the diagnostic skeleton; bail before
        # printing the K-sweep table if K-sweep wasn't run.
        if not per_k_summary:
            lines.append("**Step 2 halted before K-sweep. No clustering performed.**")
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return

    # K-sweep table.
    lines.append("## K-sweep")
    lines.append("")
    lines.append(
        "| K | silhouette | min cluster size | max cluster size | "
        "min size fraction | max size fraction | Gate | Reason if FAIL |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for K in K_SWEEP:
        info = per_k_summary[K]
        gate = "PASS" if info["passes_all"] else "FAIL"
        reason = "; ".join(info["fail_reasons"]) if info["fail_reasons"] else ""
        lines.append(
            f"| {K} | {info['silhouette']:.4f} | {info['min_cluster_size']} | "
            f"{info['max_cluster_size']} | {info['min_size_fraction']:.4f} | "
            f"{info['max_size_fraction']:.4f} | {gate} | {reason} |"
        )
    lines.append("")
    lines.append(
        f"Per-K gate: silhouette ≥ {SILHOUETTE_FLOOR}; min cluster size ≥ "
        f"{MIN_CLUSTER_SIZE}; max cluster fraction ≤ {MAX_CLUSTER_FRACTION}."
    )
    lines.append("")

    if arc_status != "PASS":
        if halt_reason and "no K" in (halt_reason or "").lower():
            lines.append("**No K satisfies the per-K gate — arc halted.**")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # K selection rationale.
    if not chosen_k:
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    sils = {K: per_k_summary[K]["silhouette"] for K in passing_ks}
    best_sil = max(sils.values())
    within_tolerance = [K for K, s in sils.items() if (best_sil - s) <= K_SILHOUETTE_TOLERANCE]
    smallest_within = min(within_tolerance)
    tolerance_invoked = smallest_within != max(sils, key=lambda K: (sils[K], -K))
    lines.append("## K selection")
    lines.append("")
    lines.append(
        f"- Passing K: {{{', '.join(str(K) for K in passing_ks)}}}."
    )
    lines.append(
        f"- Highest silhouette: K={max(sils, key=lambda K: (sils[K], -K))} "
        f"(silhouette {best_sil:.4f})."
    )
    if tolerance_invoked:
        lines.append(
            f"- K∈{{{', '.join(str(K) for K in within_tolerance)}}} within "
            f"{K_SILHOUETTE_TOLERANCE} absolute silhouette tolerance of best — "
            f"smaller K wins (parsimony / Open-12)."
        )
        lines.append(f"- **Chosen K = {chosen_k}** (parsimony tie-break applied).")
    else:
        lines.append(
            f"- No K within {K_SILHOUETTE_TOLERANCE} absolute tolerance of best — "
            f"no parsimony tie-break needed."
        )
        lines.append(f"- **Chosen K = {chosen_k}**.")
    lines.append("")

    # Centroids at chosen K (raw + standardised).
    chosen_res = k_results[chosen_k]
    n_total = int(chosen_res.sizes.sum())
    lines.append(f"## Centroids at chosen K={chosen_k}")
    lines.append("")
    lines.append(
        "| Cluster | Size | Size fraction | mono (raw) | local_peaks (raw) | "
        "pullback (raw) | time_to_peak (raw) | mono (std) | local_peaks (std) | "
        "pullback (std) | time_to_peak (std) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for k in range(chosen_res.K):
        size = int(chosen_res.sizes[k])
        sf = size / n_total if n_total > 0 else 0.0
        c_orig = chosen_res.centroids_orig[k]
        c_std = chosen_res.centroids_standardised[k]
        lines.append(
            f"| {k} | {size} | {sf:.4f} | {c_orig[0]:.4f} | {c_orig[1]:.2f} | "
            f"{c_orig[2]:.4f} | {c_orig[3]:.4f} | {c_std[0]:+.3f} | "
            f"{c_std[1]:+.3f} | {c_std[2]:+.3f} | {c_std[3]:+.3f} |"
        )
    lines.append("")

    # Archetype assignment table.
    lines.append(f"## Archetype assignments at chosen K={chosen_k}")
    lines.append("")
    lines.append(
        "| Cluster | Size | Size fraction | mono | local_peaks | pullback | "
        "time_to_peak | Archetype | Row | Certainty | Aggregation partners | Reason |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"
    )
    for a in assignments or []:
        k = a.cluster_id
        size = int(chosen_res.sizes[k])
        sf = size / n_total if n_total > 0 else 0.0
        c = chosen_res.centroids_orig[k]
        partners = ", ".join(str(p) for p in a.aggregation_partners) if a.aggregation_partners else "—"
        lines.append(
            f"| {k} | {size} | {sf:.4f} | {c[0]:.4f} | {c[1]:.2f} | {c[2]:.4f} | "
            f"{c[3]:.4f} | {a.archetype_label} | {a.archetype_row or '—'} | "
            f"{a.assignment_certainty} | {partners} | {a.reason} |"
        )
    lines.append("")

    # Aggregation flag report.
    agg_groups: Dict[str, List[int]] = {}
    for a in assignments or []:
        if a.assignment_certainty == "clean":
            agg_groups.setdefault(a.archetype_label, []).append(a.cluster_id)
    multi = {lbl: ids for lbl, ids in agg_groups.items() if len(ids) >= 2}
    lines.append("## Same-archetype aggregation flags (§7 per-aggregate eval at Step 3)")
    lines.append("")
    if multi:
        for lbl, ids in multi.items():
            lines.append(
                f"- **{lbl}**: clusters {ids} share this label — Step 3 evaluates "
                f"per-cluster AND per-aggregate."
            )
    else:
        lines.append("- No same-archetype clusters at the chosen K. No aggregation needed.")
    lines.append("")

    # Determinism.
    lines.append("## Determinism")
    lines.append("")
    lines.append("Two-run byte-identical sha256 hashes:")
    lines.append("")
    lines.append("| File | Run 1 sha256 | Run 2 sha256 | Match |")
    lines.append("|---|---|---|---|")
    file_order = sorted(sha_run1.keys())
    for fname in file_order:
        s1 = sha_run1[fname]
        s2 = sha_run2.get(fname) if sha_run2 else None
        match = "—"
        if s2 is not None:
            match = "PASS" if s1 == s2 else "FAIL"
        lines.append(
            f"| `{fname}` | `{s1}` | `{s2 or '—'}` | {match} |"
        )
    lines.append("")
    lines.append(f"**Determinism: {determinism_gate}**")
    lines.append("")

    # Configs / artefact sha256s.
    lines.append("## Config sha256s")
    lines.append("")
    for label, path in config_paths.items():
        lines.append(f"- `{label}` (`{path}`) — `{config_shas[label]}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def _env_dict() -> Dict[str, str]:
    try:
        import sklearn  # type: ignore

        sk = sklearn.__version__
    except Exception:
        sk = "not_installed"
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sk,
    }


def run_once(
    trades_csv: Path,
    paths_csv: Path,
    out_dir: Path,
    float_fmt: str,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
    Dict[int, _KResult],
    Dict[int, Dict[str, Any]],
    List[int],
    Optional[int],
    Optional[List[ClusterAssignment]],
    Dict[str, str],
    str,
    Optional[str],
]:
    """Single run: produces all CSV/TXT outputs and returns the artefact hashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(trades_csv)
    paths = pd.read_csv(paths_csv)

    features = compute_path_features(paths, trades)
    _write_path_features(features, out_dir / "path_features.csv", float_fmt)
    sha_files: Dict[str, str] = {"path_features.csv": _file_sha256(out_dir / "path_features.csv")}

    degeneracy = _degeneracy_audit(features)

    if degeneracy["_summary"]["arc_halt"]:
        return (
            features,
            degeneracy,
            {},
            {},
            [],
            None,
            None,
            sha_files,
            "FAIL",
            f"degenerate features: {', '.join(degeneracy['_summary']['degenerate_features'])}",
        )

    k_results: Dict[int, _KResult] = {}
    per_k_summary: Dict[int, Dict[str, Any]] = {}
    for K in K_SWEEP:
        res = _fit_kmeans(features, K)
        k_results[K] = res
        clusters_path = out_dir / f"clusters_K{K}.csv"
        centroids_path = out_dir / f"centroids_K{K}.csv"
        sil_path = out_dir / f"silhouette_K{K}.txt"
        _write_clusters(features, res, clusters_path)
        _write_centroids(res, centroids_path, float_fmt)
        _write_silhouette(res.silhouette, sil_path)
        sha_files[f"clusters_K{K}.csv"] = _file_sha256(clusters_path)
        sha_files[f"centroids_K{K}.csv"] = _file_sha256(centroids_path)
        sha_files[f"silhouette_K{K}.txt"] = _file_sha256(sil_path)
        passes, info = _gate_check(res)
        per_k_summary[K] = info

    passing_ks = [K for K in K_SWEEP if per_k_summary[K]["passes_all"]]

    if not passing_ks:
        return (
            features,
            degeneracy,
            k_results,
            per_k_summary,
            [],
            None,
            None,
            sha_files,
            "FAIL",
            "no K satisfies the per-K gate",
        )

    # Chosen K: highest silhouette; smaller K wins within absolute tolerance.
    sils = {K: per_k_summary[K]["silhouette"] for K in passing_ks}
    best_sil = max(sils.values())
    within_tolerance = sorted(
        [K for K, s in sils.items() if (best_sil - s) <= K_SILHOUETTE_TOLERANCE]
    )
    chosen_k = within_tolerance[0]
    chosen_res = k_results[chosen_k]

    # Centroid-only archetype labelling.
    assignments: List[ClusterAssignment] = []
    for k in range(chosen_res.K):
        c = chosen_res.centroids_orig[k]
        matches = _match_centroid_to_archetypes(
            float(c[0]), float(c[1]), float(c[2]), float(c[3])
        )
        assignments.append(_assign_cluster_label(k, matches))
    _attach_aggregation_partners(assignments)

    assignments_path = out_dir / "archetype_assignments.csv"
    _write_archetype_assignments(chosen_k, chosen_res, assignments, assignments_path, float_fmt)
    sha_files["archetype_assignments.csv"] = _file_sha256(assignments_path)

    return (
        features,
        degeneracy,
        k_results,
        per_k_summary,
        passing_ks,
        chosen_k,
        assignments,
        sha_files,
        "PASS",
        None,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arc 4 Step 2 path-shape clustering (L_ARC_PROTOCOL v2.1.1 §6)."
    )
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "l_arc_4.yaml",
        help="Arc 4 YAML config (default: configs/l_arc_4.yaml).",
    )
    p.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
        help="Override trades_all.csv path (default: from config output.results_dir).",
    )
    p.add_argument(
        "--paths-csv",
        type=Path,
        default=None,
        help="Override trades_paths.csv path (default: from config output.results_dir).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step2",
        help="Output directory (default: results/l_arc_4/step2/).",
    )
    p.add_argument(
        "--no-determinism-check",
        action="store_true",
        help="Skip the run-2 byte-identical re-verification.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    step1_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    trades_csv = args.trades_csv or (step1_dir / cfg["output"]["trades_csv"])
    paths_csv = args.paths_csv or (step1_dir / cfg["output"]["paths_csv"])
    out_dir = args.out_dir.resolve()
    float_fmt = str(cfg["output"].get("float_format", "%.10g"))

    # Confirm Step 1 PASS by reading its diagnostics file's headline disposition.
    step1_diag = step1_dir / cfg["output"]["summary_md"]
    if step1_diag.exists():
        diag_text = step1_diag.read_text(encoding="utf-8")
        if "Step 1 disposition" not in diag_text and "Headline disposition" not in diag_text:
            print(
                f"[l_arc_4 step2] WARNING: could not locate Step 1 disposition line "
                f"in {step1_diag} — proceeding anyway.",
                file=sys.stderr,
            )

    # === RUN 1 ===
    print("[l_arc_4 step2] === RUN 1 ===", file=sys.stderr)
    (
        features,
        degeneracy,
        k_results,
        per_k_summary,
        passing_ks,
        chosen_k,
        assignments,
        sha_run1,
        arc_status,
        halt_reason,
    ) = run_once(trades_csv, paths_csv, out_dir, float_fmt)

    sha_run2: Optional[Dict[str, str]] = None
    determinism_gate = "N/A"
    if not args.no_determinism_check:
        print("[l_arc_4 step2] === RUN 2 (determinism) ===", file=sys.stderr)
        (_, _, _, _, _, _, _, sha_run2_inner, _, _) = run_once(
            trades_csv, paths_csv, out_dir, float_fmt
        )
        sha_run2 = sha_run2_inner
        matched = all(
            sha_run1.get(k) == sha_run2.get(k) for k in sorted(set(sha_run1) | set(sha_run2))
        )
        determinism_gate = "PASS" if matched else "FAIL"

    # Config sha256s.
    config_paths = {
        "configs/l_arc_4.yaml": str(args.config.relative_to(_REPO_ROOT))
        if _REPO_ROOT in args.config.parents or _REPO_ROOT == args.config.parent.parent
        else str(args.config),
        f"{cfg['output']['results_dir']}/trades_all.csv": str(
            trades_csv.relative_to(_REPO_ROOT)
        ),
        f"{cfg['output']['results_dir']}/trades_paths.csv": str(
            paths_csv.relative_to(_REPO_ROOT)
        ),
    }
    config_shas = {
        label: _file_sha256(_REPO_ROOT / Path(path)) for label, path in config_paths.items()
    }

    diag_path = out_dir / "step2_diagnostics.md"
    write_diagnostics(
        diag_path,
        features,
        degeneracy,
        k_results,
        per_k_summary,
        passing_ks,
        chosen_k,
        assignments,
        sha_run1,
        sha_run2,
        determinism_gate,
        config_paths,
        config_shas,
        arc_status,
        halt_reason,
    )

    chosen_info = (
        f"chosen_K={chosen_k} silhouette={per_k_summary[chosen_k]['silhouette']:.4f}"
        if chosen_k is not None
        else f"chosen_K=None halt_reason={halt_reason}"
    )
    print(
        f"[l_arc_4 step2] DONE pool={len(features)} {chosen_info} "
        f"determinism={determinism_gate} disposition={arc_status}",
        file=sys.stderr,
    )
    print(f"[l_arc_4 step2] diagnostics → {diag_path}", file=sys.stderr)

    # Final disposition: PASS only if Step 2 itself passed AND determinism PASS (or N/A).
    final_pass = (
        arc_status == "PASS" and determinism_gate in ("PASS", "N/A")
    )

    # Also emit a tiny env footer JSON for debugging if the user ever needs it.
    env_info = {
        "env": _env_dict(),
        "args": {
            "config": str(args.config),
            "trades_csv": str(trades_csv),
            "paths_csv": str(paths_csv),
            "out_dir": str(out_dir),
            "no_determinism_check": bool(args.no_determinism_check),
        },
        "chosen_k": chosen_k,
        "determinism_gate": determinism_gate,
        "arc_status": arc_status,
    }
    (out_dir / "step2_env.json").write_text(
        json.dumps(env_info, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    return 0 if final_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())


# Helpers that are not in the live execution path but are referenced indirectly
# above (e.g. `math` import). Keeping them around to make the module
# self-contained for future analysis scripts.
_ = math

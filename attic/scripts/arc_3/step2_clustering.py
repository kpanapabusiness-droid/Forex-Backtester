"""Arc 3 — Step 2 path-shape clustering.

L_ARC_PROTOCOL v2.0 §6. Outcome-blind clustering of Step 1's trade pool on
four path-shape features. Produces:

  results/l_arc_3/step2_clustering/
    path_features.csv                   — per-trade four features
    clusters_K{3..7}.csv                — trade_id, cluster_label per K
    centroids_K{3..7}.csv               — cluster, size, four centroids (original)
    silhouette_K{3..7}.txt              — scalar silhouette score
    archetype_assignments.csv           — chosen K only: cluster -> §11 archetype
    step2_diagnostics.json              — gate audit, Open-07 §11 mismatch notes,
                                          Open-08 degeneracy evidence, determinism

Features (v2.0 §6, outcome-blind — no final_r, exit_reason, or future-only
information beyond bar-local mfe_so_far_r / close_r):
  monotonicity_ratio_in_profit
  local_peaks_count
  pullback_magnitude_median       (operational def: min close_r between peaks)
  time_to_peak_mfe_relative

Clustering: KMeans(random_state=42, n_init=10, max_iter=300) on
StandardScaler-transformed features. K ∈ {3, 4, 5, 6, 7}.

Per-K gate (conjunctive):
  silhouette ≥ 0.30
  no cluster > 90% of trades
  all clusters ≥ 30 trades

K selection: highest silhouette among passing; ties → smaller K.

Archetype labelling: match each cluster's centroid against the seven §11
patterns in document order. pct_peak_and_collapse (forward-outcome) is
allowed in the labelling step only — never in clustering.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FEATURE_COLS: Tuple[str, ...] = (
    "monotonicity_ratio_in_profit",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
)

K_SWEEP: Tuple[int, ...] = (3, 4, 5, 6, 7)

DEGENERACY_THRESHOLD: float = 0.80
MIN_CLUSTER_SIZE: int = 30
MAX_CLUSTER_FRACTION: float = 0.90
SILHOUETTE_FLOOR: float = 0.30


# ---------------------------------------------------------------------------
# Path-shape feature computation per trade.
# ---------------------------------------------------------------------------


def _features_for_trade(paths_g: pd.DataFrame) -> Dict[str, float]:
    """Compute the four v2.0 §6 path-shape features for one trade.

    `paths_g` has rows sorted by bar_offset for one trade_id; columns include
    bar_offset, close_r, mfe_so_far_r, mae_so_far_r, is_in_profit.
    """
    # bar_offset 0 is the entry bar; the trade's "held window" is rows
    # 0..len(paths_g)-1. bars_held proxy from path length minus 1 (excl entry).
    # NOTE: trades_all.csv carries the authoritative bars_held; we don't need
    # it here for the feature math (only for time_to_peak_mfe_relative, which
    # uses the path-internal time-to-peak vs path length).
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
    in_profit = close_r > 0.0
    n_in_profit = int(in_profit.sum())
    if n_in_profit == 0:
        monotonicity = 0.0
    else:
        # Walk through bars where close_r > 0 in order; count how many have
        # close_r ≥ previous in-profit bar's close_r. The first in-profit bar
        # has no previous to compare to — exclude it from numerator AND
        # denominator (counting fraction over comparable pairs).
        # §6 says "among bars where close_r > 0, fraction where close_r ≥
        # previous in-profit bar". Interpreting "fraction" as out of the
        # comparable pairs (i.e., all in-profit bars except the first), which
        # yields a value in [0, 1] meaningful for clustering.
        in_profit_closes = close_r[in_profit]
        if in_profit_closes.size <= 1:
            # One in-profit bar — no comparable pair. Convention: zero
            # comparable pairs ⇒ feature = 0 (mirrors the "zero in-profit
            # bars" edge case). This is a tie-breaker for trades that never
            # gained traction.
            monotonicity = 0.0
        else:
            gte = in_profit_closes[1:] >= in_profit_closes[:-1]
            monotonicity = float(gte.sum() / gte.size)

    # Feature 2: local_peaks_count — count of bars where mfe_so_far_r >
    # previous bar's mfe_so_far_r. bars_held = 0 → 0 (handled by path being
    # length 1 — only entry bar — which gives no comparable pair → 0).
    if mfe.size <= 1:
        local_peaks = 0
    else:
        # A "peak" is any bar that strictly increases mfe_so_far_r vs prior.
        local_peaks = int(np.sum(mfe[1:] > mfe[:-1]))

    # Feature 3: pullback_magnitude_median — for consecutive peak pairs,
    # earlier_peak.mfe_so_far_r − min(close_r between peaks). Median.
    # "Peak" same definition as feature 2 (strictly increasing mfe).
    peak_positions = np.where(mfe[1:] > mfe[:-1])[0] + 1  # indices in path
    if peak_positions.size < 2:
        pullback_median = 0.0
    else:
        diffs = []
        for i in range(len(peak_positions) - 1):
            p1 = int(peak_positions[i])
            p2 = int(peak_positions[i + 1])
            # bars STRICTLY between p1 and p2 — exclusive.
            if p2 - p1 < 2:
                # No bars strictly between consecutive peaks → no pullback to
                # measure. Skip this pair (don't pollute the median with 0s).
                continue
            min_close_r_between = float(np.min(close_r[p1 + 1 : p2]))
            earlier_peak_mfe = float(mfe[p1])
            diffs.append(earlier_peak_mfe - min_close_r_between)
        pullback_median = float(np.median(diffs)) if diffs else 0.0

    # Feature 4: time_to_peak_mfe_relative — time_to_peak_mfe / max(bars_held, 1),
    # capped at 1.0. Trade never in profit → 0.
    if mfe.size == 0 or float(mfe.max()) <= 0.0:
        time_to_peak_rel = 0.0
    else:
        # time_to_peak_mfe = bar_offset of the first bar that achieves the
        # global peak mfe_so_far_r. bars_held proxy = max(bar_offset) — the
        # last bar in the path. mfe_so_far_r is monotone non-decreasing, so
        # the peak first achieved when mfe == its max value.
        peak_value = float(mfe.max())
        first_peak_idx = int(np.argmax(mfe >= peak_value - 1e-12))
        ttp = int(bar_offset[first_peak_idx])
        bars_held_proxy = max(int(bar_offset.max()), 1)
        time_to_peak_rel = min(float(ttp) / float(bars_held_proxy), 1.0)

    return {
        "monotonicity_ratio_in_profit": float(monotonicity),
        "local_peaks_count": int(local_peaks),
        "pullback_magnitude_median": float(pullback_median),
        "time_to_peak_mfe_relative": float(time_to_peak_rel),
    }


def compute_path_features(paths_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trade features. Iterates over trade_id in trades_df order
    (which is chronological by signal_time, pair — see Step 1 sort order).
    """
    paths_df = paths_df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    feature_rows: List[Dict[str, Any]] = []
    groups = paths_df.groupby("trade_id", sort=False)
    paths_by_trade = {tid: g for tid, g in groups}

    for tid in trades_df["trade_id"].to_numpy():
        g = paths_by_trade.get(int(tid))
        if g is None or len(g) == 0:
            feats = _features_for_trade(pd.DataFrame(columns=paths_df.columns))
        else:
            feats = _features_for_trade(g)
        feature_rows.append({"trade_id": int(tid), **feats})

    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Degeneracy audit.
# ---------------------------------------------------------------------------


def _degeneracy_audit(features: pd.DataFrame) -> Dict[str, Any]:
    """Return per-feature modal-value fraction and a degeneracy flag list.

    For integer-typed features, modal value is exact. For float-typed, modal
    value uses ±1e-9 tolerance via np.isclose.
    """
    audit: Dict[str, Any] = {}
    for col in FEATURE_COLS:
        s = features[col].to_numpy()
        n = len(s)
        if col == "local_peaks_count":
            counts = pd.Series(s).value_counts()
            modal_value = float(counts.index[0])
            modal_count = int(counts.iloc[0])
        else:
            # Float comparison via rounding-to-9 — equivalent to ±1e-9 buckets
            # for the typical [0, ~5] feature range and exact 0/cap clusters.
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
    audit["_summary"] = {
        "degenerate_features": [
            col for col in FEATURE_COLS if audit[col]["is_degenerate_gt_80pct"]
        ],
    }
    audit["_summary"]["count"] = len(audit["_summary"]["degenerate_features"])
    audit["_summary"]["arc_kill"] = audit["_summary"]["count"] >= 2
    return audit


# ---------------------------------------------------------------------------
# K-sweep.
# ---------------------------------------------------------------------------


@dataclass
class _KResult:
    K: int
    labels: np.ndarray
    centroids_orig: np.ndarray
    sizes: np.ndarray
    silhouette: float

    @property
    def size_fractions(self) -> np.ndarray:
        n = int(self.sizes.sum())
        return self.sizes / n if n > 0 else np.zeros_like(self.sizes, dtype=float)


def _fit_kmeans(features: pd.DataFrame, K: int) -> _KResult:
    X = features[list(FEATURE_COLS)].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    sil = float(silhouette_score(X_scaled, labels))

    # Centroids in ORIGINAL feature space — mean of original features per
    # cluster (equivalent to inverse_transform of km.cluster_centers_ given
    # StandardScaler is linear, but we compute means directly for clarity).
    centroids_orig = np.zeros((K, len(FEATURE_COLS)), dtype=float)
    sizes = np.zeros(K, dtype=int)
    for k in range(K):
        mask = labels == k
        sizes[k] = int(mask.sum())
        if sizes[k] > 0:
            centroids_orig[k] = X[mask].mean(axis=0)
    return _KResult(K=K, labels=labels, centroids_orig=centroids_orig, sizes=sizes, silhouette=sil)


def _gate_check(res: _KResult) -> Tuple[bool, Dict[str, Any]]:
    n_total = int(res.sizes.sum())
    min_size = int(res.sizes.min())
    max_frac = float(res.sizes.max() / n_total) if n_total > 0 else 0.0
    cond_sil = res.silhouette >= SILHOUETTE_FLOOR
    cond_min = min_size >= MIN_CLUSTER_SIZE
    cond_max = max_frac <= MAX_CLUSTER_FRACTION
    passes = bool(cond_sil and cond_min and cond_max)
    return passes, {
        "silhouette": float(res.silhouette),
        "min_cluster_size": min_size,
        "max_cluster_fraction": max_frac,
        "passes_silhouette_gate": bool(cond_sil),
        "passes_min_size_gate": bool(cond_min),
        "passes_max_fraction_gate": bool(cond_max),
        "passes_all_gates": passes,
    }


# ---------------------------------------------------------------------------
# §11 archetype matching.
# ---------------------------------------------------------------------------


@dataclass
class ArchetypeMatch:
    archetype_label: str
    matched: bool
    rule_text: str
    unmet: List[str]


def _match_centroid_to_archetypes(
    mono: float, peaks: float, pullback: float, ttp_rel: float, pct_pc: float
) -> List[ArchetypeMatch]:
    """Return per-row §11 match status, in document order.

    Last row ("Bimodal split exit") is NOT centroid-matchable — it requires a
    cluster-level distribution check (two distinct fwd_mfe modes ≥ 1R apart).
    Marked accordingly. Same for V-shape recovery (needs "MAE early, peak in
    [0.4, 0.8]") which isn't expressible purely from centroid means of the
    four core features. Both are flagged for Open-07 review.
    """
    results: List[ArchetypeMatch] = []

    # Row 1: Monotone ascent
    unmet = []
    if not (mono >= 0.55):
        unmet.append(f"monotonicity {mono:.3f} < 0.55")
    if not (peaks <= 4):
        unmet.append(f"local_peaks {peaks:.2f} > 4")
    if not (ttp_rel >= 0.50):
        unmet.append(f"time_to_peak_rel {ttp_rel:.3f} < 0.50")
    results.append(
        ArchetypeMatch(
            archetype_label="Monotone ascent",
            matched=(len(unmet) == 0),
            rule_text="monotonicity >= 0.55 AND local_peaks <= 4 AND time_to_peak_rel >= 0.50",
            unmet=unmet,
        )
    )

    # Row 2: Stepwise climber
    unmet = []
    if not (mono >= 0.50):
        unmet.append(f"monotonicity {mono:.3f} < 0.50")
    if not (5 <= peaks <= 30):
        unmet.append(f"local_peaks {peaks:.2f} not in [5, 30]")
    if not (pullback <= 0.5):
        unmet.append(f"pullback {pullback:.3f} > 0.5R")
    if not (ttp_rel >= 0.50):
        unmet.append(f"time_to_peak_rel {ttp_rel:.3f} < 0.50")
    results.append(
        ArchetypeMatch(
            archetype_label="Stepwise climber",
            matched=(len(unmet) == 0),
            rule_text="monotonicity >= 0.50 AND local_peaks in [5,30] AND pullback <= 0.5R AND time_to_peak_rel >= 0.50",
            unmet=unmet,
        )
    )

    # Row 3: Early-peak hold (needs fwd_mfe_p50 >= 1.5R — not in 4-feature
    # centroid; flagged as Open-07 ambiguity. Apply the time_to_peak +
    # pct_peak_and_collapse part only.)
    unmet = []
    if not (ttp_rel <= 0.30):
        unmet.append(f"time_to_peak_rel {ttp_rel:.3f} > 0.30")
    if not (pct_pc < 0.30):
        unmet.append(f"pct_peak_and_collapse {pct_pc:.3f} >= 0.30")
    results.append(
        ArchetypeMatch(
            archetype_label="Early-peak hold",
            matched=(len(unmet) == 0),
            rule_text="time_to_peak_rel <= 0.30 AND pct_peak_and_collapse < 0.30 (fwd_mfe_p50 >= 1.5R deferred to Step 3)",
            unmet=unmet,
        )
    )

    # Row 4: Peak-and-collapse
    unmet = []
    if not (ttp_rel <= 0.30):
        unmet.append(f"time_to_peak_rel {ttp_rel:.3f} > 0.30")
    if not (pct_pc >= 0.50):
        unmet.append(f"pct_peak_and_collapse {pct_pc:.3f} < 0.50")
    results.append(
        ArchetypeMatch(
            archetype_label="Peak-and-collapse",
            matched=(len(unmet) == 0),
            rule_text="time_to_peak_rel <= 0.30 AND pct_peak_and_collapse >= 0.50",
            unmet=unmet,
        )
    )

    # Row 5: V-shape recovery — requires "MAE early (mae before peak >= 5
    # bars), peak in [0.4, 0.8] of trade" — not expressible from these
    # centroids alone. Mark unmatchable.
    results.append(
        ArchetypeMatch(
            archetype_label="V-shape recovery",
            matched=False,
            rule_text="MAE-before-peak >= 5 bars AND peak position in [0.4, 0.8] (not centroid-matchable from 4 features alone)",
            unmet=["pattern not expressible from 4 centroid features"],
        )
    )

    # Row 6: Random walk
    unmet = []
    if not (peaks >= 8):
        unmet.append(f"local_peaks {peaks:.2f} < 8")
    if not (mono <= 0.30):
        unmet.append(f"monotonicity {mono:.3f} > 0.30")
    if not (pullback >= 1.0):
        unmet.append(f"pullback {pullback:.3f} < 1.0R")
    results.append(
        ArchetypeMatch(
            archetype_label="Random walk",
            matched=(len(unmet) == 0),
            rule_text="local_peaks >= 8 AND monotonicity <= 0.30 AND pullback >= 1R",
            unmet=unmet,
        )
    )

    # Row 7: Bimodal / Split-exit — needs cluster-level fwd_mfe distribution
    # check, not centroid-matchable.
    results.append(
        ArchetypeMatch(
            archetype_label="Split exit variant (bimodal)",
            matched=False,
            rule_text="bimodal fwd_mfe distribution with two modes >= 1R apart (cluster-level, not centroid-matchable)",
            unmet=["distributional pattern; defer to Step 3"],
        )
    )

    return results


def _disambiguate_or_unassigned(
    matches: List[ArchetypeMatch], ttp_rel: float, pct_pc: float
) -> Tuple[str, str]:
    """Apply §11 disambiguation rules. Return (archetype_label, boundary_reason).

    Disambiguation rules:
      - Early-peak hold vs Peak-and-collapse share time_to_peak_rel <= 0.30.
        Differentiator is pct_peak_and_collapse:
          < 0.30 → Early-peak hold
          >= 0.50 → Peak-and-collapse
          [0.30, 0.50) → boundary, assign by empirical test (deferred)
    """
    matched_labels = [m.archetype_label for m in matches if m.matched]

    if len(matched_labels) == 1:
        return matched_labels[0], ""

    if len(matched_labels) > 1:
        # Multiple clean matches — figure out which (if any) clean rule fires.
        # Most common conflict: Monotone ascent ⊂ Stepwise climber's superset
        # because monotonicity >= 0.55 implies monotonicity >= 0.50 AND
        # time_to_peak_rel >= 0.50 also implies the same. Differentiator is
        # local_peaks_count (Monotone ≤ 4; Stepwise 5-30).
        if set(matched_labels) == {"Monotone ascent", "Stepwise climber"}:
            # Local_peaks rule fully disjoint (≤4 vs 5-30) — can't both match
            # cleanly. If both reported as matched, that's a coding bug.
            return ",".join(sorted(matched_labels)), "multi-match logic inconsistency"
        return ",".join(sorted(matched_labels)), "multiple clean matches"

    # Zero matches: check Early-peak / Peak-and-collapse ambiguity zone.
    if ttp_rel <= 0.30 and 0.30 <= pct_pc < 0.50:
        return "unassigned", "early_peak_pc_ambiguity (0.30 <= pct_peak_and_collapse < 0.50)"

    return "unassigned", "no §11 pattern matched"


# ---------------------------------------------------------------------------
# pct_peak_and_collapse — forward-outcome, used only at labelling step.
# ---------------------------------------------------------------------------


def _pct_peak_and_collapse_per_trade(trades: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    """Per-trade boolean flag: 1 if (peak_mfe_r >= 1R) AND (final_r <= 0.4 * peak_mfe_r).

    `peak_mfe_r` is the trade's mfe_r column (already the peak — Step 1's
    mfe_so_far_r is monotone non-decreasing; mfe_r in trades_all.csv is the
    final value, i.e., the peak).
    """
    merged = features.merge(trades[["trade_id", "final_r", "mfe_r"]], on="trade_id", how="left")
    peak = merged["mfe_r"].to_numpy(dtype=float)
    final = merged["final_r"].to_numpy(dtype=float)
    cond_peak = peak >= 1.0
    cond_collapse = final <= 0.4 * peak
    flag = (cond_peak & cond_collapse).astype(int)
    return flag


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------


def _write_path_features(features: pd.DataFrame, path: Path, float_fmt: str) -> None:
    features = features[["trade_id", *FEATURE_COLS]].copy()
    features.to_csv(path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n")


def _write_clusters(features: pd.DataFrame, res: _KResult, path: Path) -> None:
    df = pd.DataFrame(
        {
            "trade_id": features["trade_id"].to_numpy(),
            "cluster_label": res.labels.astype(int),
        }
    )
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _write_centroids(res: _KResult, path: Path, float_fmt: str) -> None:
    rows = []
    n_total = int(res.sizes.sum())
    for k in range(res.K):
        row = {
            "cluster_label": k,
            "size": int(res.sizes[k]),
            "size_fraction": float(res.sizes[k] / n_total) if n_total > 0 else 0.0,
        }
        for i, col in enumerate(FEATURE_COLS):
            row[f"centroid_{col}"] = float(res.centroids_orig[k, i])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n")


def _write_silhouette(s: float, path: Path) -> None:
    path.write_text(f"{s:.10g}\n", encoding="utf-8")


def _write_archetype_assignments(
    res: _KResult,
    labels: List[str],
    boundary_reasons: List[str],
    pct_pc_per_cluster: List[float],
    path: Path,
    float_fmt: str,
) -> None:
    n_total = int(res.sizes.sum())
    rows = []
    for k in range(res.K):
        rows.append(
            {
                "cluster_label": k,
                "archetype_label": labels[k],
                "centroid_monotonicity": float(res.centroids_orig[k, 0]),
                "centroid_local_peaks": float(res.centroids_orig[k, 1]),
                "centroid_pullback": float(res.centroids_orig[k, 2]),
                "centroid_time_to_peak_rel": float(res.centroids_orig[k, 3]),
                "pct_peak_and_collapse": float(pct_pc_per_cluster[k]),
                "size": int(res.sizes[k]),
                "size_fraction": float(res.sizes[k] / n_total) if n_total > 0 else 0.0,
                "boundary_reason": boundary_reasons[k],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n")


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def run(
    trades_csv: Path,
    paths_csv: Path,
    out_dir: Path,
    *,
    float_fmt: str = "%.10g",
    write_manifest: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(trades_csv)
    paths = pd.read_csv(paths_csv)

    # 2.1 — Path-shape features.
    features = compute_path_features(paths, trades)
    _write_path_features(features, out_dir / "path_features.csv", float_fmt)

    # 2.2 — Degeneracy audit.
    degeneracy = _degeneracy_audit(features)

    if degeneracy["_summary"]["arc_kill"]:
        diag = {
            "input_trades_csv": str(trades_csv.relative_to(_REPO_ROOT)),
            "input_paths_csv": str(paths_csv.relative_to(_REPO_ROOT)),
            "n_trades": int(len(features)),
            "degeneracy_audit": degeneracy,
            "k_sweep": {},
            "passing_ks": [],
            "chosen_k": None,
            "arc_status": "FAIL_DEGENERACY",
            "determinism": None,
            "env": _env_dict(),
        }
        if write_manifest:
            (out_dir / "step2_diagnostics.json").write_text(
                json.dumps(diag, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        return diag

    # 2.3 — K-sweep.
    k_results: Dict[int, _KResult] = {}
    per_k_summary: Dict[int, Dict[str, Any]] = {}
    for K in K_SWEEP:
        res = _fit_kmeans(features, K)
        k_results[K] = res
        _write_clusters(features, res, out_dir / f"clusters_K{K}.csv")
        _write_centroids(res, out_dir / f"centroids_K{K}.csv", float_fmt)
        _write_silhouette(res.silhouette, out_dir / f"silhouette_K{K}.txt")
        passes, info = _gate_check(res)
        per_k_summary[K] = {
            **info,
            "cluster_sizes": [int(s) for s in res.sizes.tolist()],
            "size_fractions": [float(f) for f in res.size_fractions.tolist()],
        }

    # 2.5 — K selection. Highest silhouette among passing; ties → smaller K.
    passing_ks = [K for K in K_SWEEP if per_k_summary[K]["passes_all_gates"]]

    if not passing_ks:
        diag = {
            "input_trades_csv": str(trades_csv.relative_to(_REPO_ROOT)),
            "input_paths_csv": str(paths_csv.relative_to(_REPO_ROOT)),
            "n_trades": int(len(features)),
            "degeneracy_audit": degeneracy,
            "k_sweep": per_k_summary,
            "passing_ks": [],
            "chosen_k": None,
            "arc_status": "FAIL_GATE",
            "determinism": None,
            "env": _env_dict(),
        }
        if write_manifest:
            (out_dir / "step2_diagnostics.json").write_text(
                json.dumps(diag, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        return diag

    # Tie-breaker: highest silhouette; smaller K wins ties.
    best_K = max(passing_ks, key=lambda K: (per_k_summary[K]["silhouette"], -K))
    chosen = k_results[best_K]

    # 2.6 — Cluster → archetype labelling. pct_peak_and_collapse is allowed
    # at this step only (forward-outcome).
    pc_flag_per_trade = _pct_peak_and_collapse_per_trade(trades, features)
    pct_pc_per_cluster: List[float] = []
    for k in range(chosen.K):
        mask = chosen.labels == k
        n_k = int(mask.sum())
        pct_pc_per_cluster.append(float(pc_flag_per_trade[mask].mean()) if n_k > 0 else 0.0)

    cluster_archetypes: List[str] = []
    boundary_reasons: List[str] = []
    cluster_match_diag: List[Dict[str, Any]] = []
    for k in range(chosen.K):
        c = chosen.centroids_orig[k]
        mono, peaks, pullback, ttp_rel = float(c[0]), float(c[1]), float(c[2]), float(c[3])
        pct_pc = pct_pc_per_cluster[k]
        matches = _match_centroid_to_archetypes(mono, peaks, pullback, ttp_rel, pct_pc)
        label, reason = _disambiguate_or_unassigned(matches, ttp_rel, pct_pc)
        cluster_archetypes.append(label)
        boundary_reasons.append(reason)
        cluster_match_diag.append(
            {
                "cluster_label": k,
                "centroid": {
                    "monotonicity_ratio_in_profit": mono,
                    "local_peaks_count": peaks,
                    "pullback_magnitude_median": pullback,
                    "time_to_peak_mfe_relative": ttp_rel,
                },
                "pct_peak_and_collapse": pct_pc,
                "matches": [dataclasses.asdict(m) for m in matches],
                "assignment": label,
                "boundary_reason": reason,
            }
        )

    _write_archetype_assignments(
        chosen,
        cluster_archetypes,
        boundary_reasons,
        pct_pc_per_cluster,
        out_dir / "archetype_assignments.csv",
        float_fmt,
    )

    # Determinism: hash the chosen-K clusters file. Second-run verification
    # happens at the CLI level (--verify-determinism).
    sha_clusters_chosen = _sha256_file(out_dir / f"clusters_K{best_K}.csv")
    sha_path_features = _sha256_file(out_dir / "path_features.csv")
    sha_centroids_chosen = _sha256_file(out_dir / f"centroids_K{best_K}.csv")
    sha_archetypes = _sha256_file(out_dir / "archetype_assignments.csv")

    diag: Dict[str, Any] = {
        "input_trades_csv": str(trades_csv.relative_to(_REPO_ROOT)),
        "input_paths_csv": str(paths_csv.relative_to(_REPO_ROOT)),
        "n_trades": int(len(features)),
        "degeneracy_audit": degeneracy,
        "k_sweep": per_k_summary,
        "passing_ks": list(passing_ks),
        "chosen_k": int(best_K),
        "chosen_k_archetype_assignments": [
            {
                "cluster_label": k,
                "archetype_label": cluster_archetypes[k],
                "boundary_reason": boundary_reasons[k],
                "size": int(chosen.sizes[k]),
                "size_fraction": float(chosen.sizes[k] / chosen.sizes.sum()),
                "centroid_monotonicity": float(chosen.centroids_orig[k, 0]),
                "centroid_local_peaks": float(chosen.centroids_orig[k, 1]),
                "centroid_pullback": float(chosen.centroids_orig[k, 2]),
                "centroid_time_to_peak_rel": float(chosen.centroids_orig[k, 3]),
                "pct_peak_and_collapse": float(pct_pc_per_cluster[k]),
            }
            for k in range(chosen.K)
        ],
        "open_07_match_audit": cluster_match_diag,
        "sha256": {
            "path_features_csv": sha_path_features,
            "clusters_chosen_csv": sha_clusters_chosen,
            "centroids_chosen_csv": sha_centroids_chosen,
            "archetype_assignments_csv": sha_archetypes,
        },
        "arc_status": "PASS",
        "env": _env_dict(),
    }
    if write_manifest:
        (out_dir / "step2_diagnostics.json").write_text(
            json.dumps(diag, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return diag


def _env_dict() -> Dict[str, str]:
    import sklearn  # type: ignore

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc 3 Step 2 path-shape clustering (L_ARC_PROTOCOL v2.0).")
    p.add_argument(
        "--trades-csv",
        type=Path,
        default=_REPO_ROOT / "results/l_arc_3/step1_plumbing/trades_all.csv",
    )
    p.add_argument(
        "--paths-csv",
        type=Path,
        default=_REPO_ROOT / "results/l_arc_3/step1_plumbing/trades_paths.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results/l_arc_3/step2_clustering",
    )
    p.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Run twice and confirm clusters_K<chosen>.csv is byte-identical.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.verify_determinism:
        info1 = run(args.trades_csv, args.paths_csv, args.out_dir, write_manifest=False)
        if info1.get("arc_status") != "PASS":
            (args.out_dir / "step2_diagnostics.json").write_text(
                json.dumps(info1, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            print(f"[arc_3 step 2] arc_status={info1.get('arc_status')} — gate failure, halting.")
            return 1
        chosen_k = info1["chosen_k"]
        sha1 = info1["sha256"]["clusters_chosen_csv"]

        info2 = run(args.trades_csv, args.paths_csv, args.out_dir, write_manifest=False)
        sha2 = info2["sha256"]["clusters_chosen_csv"]
        determinism_pass = sha1 == sha2
        info2["determinism"] = {
            "chosen_k": chosen_k,
            "run_1_clusters_chosen_sha256": sha1,
            "run_2_clusters_chosen_sha256": sha2,
            "byte_identical": determinism_pass,
        }
        (args.out_dir / "step2_diagnostics.json").write_text(
            json.dumps(info2, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            f"[arc_3 step 2] chosen_K={chosen_k} silhouette={info2['k_sweep'][chosen_k]['silhouette']:.4f} "
            f"determinism={'PASS' if determinism_pass else 'FAIL'}"
        )
        return 0 if determinism_pass else 1

    info = run(args.trades_csv, args.paths_csv, args.out_dir)
    print(f"[arc_3 step 2] arc_status={info['arc_status']} chosen_k={info.get('chosen_k')}")
    return 0 if info.get("arc_status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

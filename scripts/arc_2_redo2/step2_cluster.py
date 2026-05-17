"""Arc 2 redo2 — Step 2 path-shape clustering.

Per L_ARC_PROTOCOL.md v2.1.1 §§1, 6, 11:
  1. Compute four outcome-blind path-shape features per trade (is_held=1 only).
  2. K-means sweep K in {3..7} on StandardScaler-applied features.
  3. Apply §6 gate (silhouette >= 0.30, no cluster > 90%, all clusters >= 30).
  4. Select K (highest silhouette satisfying gate; within 0.01 absolute
     tolerance, smaller K preferred per v2.1.1 §6 / Open-12 closure).
  5. Label clusters per §11 centroid patterns; flag boundary clusters.
  6. Compute Jaccard overlap of trade_ids vs prior arc_2_redo K=4 clusters.
  7. Write outputs + STEP2_SUMMARY.md.

Forked from scripts/arc_2_redo/step2_cluster.py with surgical edits:
  - Filter column rename: still_open == 1 -> is_held == 1 (matches the v2.1.1
    fork's path schema; semantic is "trade is held / PnL-bearing"; includes
    the exit bar inclusive of exit_offset, vs prior which excluded exit bar).
  - K-selection rule updated to v2.1.1 0.01-absolute tie tolerance.
  - Added Jaccard overlap vs prior arc_2_redo K=4 clustering output.
  - Protocol refs v2.0 -> v2.1.1 throughout.

Determinism: random_state=42 everywhere. CSV outputs are byte-identical
across re-runs. PNG histogram is best-effort deterministic but not gated.

Usage:
  py scripts/arc_2_redo2/step2_cluster.py -c configs/arc_2_redo2/step2.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ============================================================
# Feature computation
# ============================================================

FEATURE_COLS = [
    "monotonicity_ratio_in_profit",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
]


def _features_for_trade(
    close_r: np.ndarray, mfe: np.ndarray
) -> Tuple[float, int, float, float, int]:
    """Compute the four path-shape features for one trade's held window.

    Inputs are arrays restricted to is_held=1, ordered by bar_offset
    ascending (so position == bar_offset since held window starts at offset 0).
    is_held=1 covers [0, exit_offset] inclusive per the v2.1.1 fork schema —
    one bar wider than the prior arc_2_redo's still_open=1 range.

    Returns (monotonicity, local_peaks, pullback_median, ttp_relative, bars_held).
    """
    n = int(close_r.size)
    if n == 0:
        return 0.0, 0, 0.0, 0.0, 0

    # Feature 1: monotonicity_ratio_in_profit
    in_profit = close_r[close_r > 0]
    if in_profit.size >= 2:
        non_dec = int(np.sum(in_profit[1:] >= in_profit[:-1]))
        monotonicity = non_dec / (in_profit.size - 1)
    else:
        monotonicity = 0.0

    # Feature 2: local_peaks_count
    diffs = np.diff(mfe)
    peak_pos = np.where(diffs > 0)[0] + 1  # index in held window
    local_peaks = int(peak_pos.size)

    # Feature 3: pullback_magnitude_median (operational: min(close_r) between peaks)
    pullbacks: List[float] = []
    for i in range(peak_pos.size - 1):
        a = int(peak_pos[i])
        b = int(peak_pos[i + 1])
        if b - a >= 2:
            between_min = float(close_r[a + 1 : b].min())
            pullbacks.append(float(mfe[a]) - between_min)
    pullback_median = float(np.median(pullbacks)) if pullbacks else 0.0

    # Feature 4: time_to_peak_mfe_relative
    max_close_r = float(close_r.max())
    if max_close_r <= 0.0:
        ttp_rel = 0.0
    else:
        max_mfe = float(mfe.max())
        # First index where mfe equals its max value (mfe is non-decreasing
        # by construction, so this is the index where the last increment landed).
        ttp = int(np.argmax(mfe == max_mfe))
        ttp_rel = min(ttp / max(n, 1), 1.0)

    return monotonicity, local_peaks, pullback_median, ttp_rel, n


def compute_path_features(paths_df: pd.DataFrame) -> pd.DataFrame:
    """Group trades_paths.csv by trade_id, compute features on is_held=1 subset."""
    held = paths_df[paths_df["is_held"] == 1].copy()
    held = held.sort_values(["trade_id", "bar_offset"], kind="mergesort")

    # Vectorise the per-group work via to_numpy + groupby split points.
    trade_ids = held["trade_id"].to_numpy()
    close_r = held["close_r"].to_numpy(dtype=float)
    mfe = held["mfe_so_far_r"].to_numpy(dtype=float)

    # Identify group boundaries.
    if trade_ids.size == 0:
        unique_ids = np.array([], dtype=np.int64)
        starts = np.array([], dtype=np.int64)
        ends = np.array([], dtype=np.int64)
    else:
        change = np.concatenate(([True], trade_ids[1:] != trade_ids[:-1]))
        starts = np.where(change)[0]
        ends = np.concatenate((starts[1:], [trade_ids.size]))
        unique_ids = trade_ids[starts]

    # Also need to include trades whose held window is empty (offset-0 SL trades
    # where still_open=0 throughout). These don't appear in `held`; pick them up
    # from the original paths_df trade_id list.
    all_trade_ids = np.sort(paths_df["trade_id"].unique())

    feat_map: Dict[int, Tuple[float, int, float, float, int]] = {}
    for s, e, tid in zip(starts, ends, unique_ids):
        feat_map[int(tid)] = _features_for_trade(close_r[s:e], mfe[s:e])

    rows = []
    for tid in all_trade_ids:
        f = feat_map.get(int(tid), (0.0, 0, 0.0, 0.0, 0))
        rows.append((int(tid), f[0], f[1], f[2], f[3], f[4]))
    out = pd.DataFrame(
        rows,
        columns=[
            "trade_id",
            "monotonicity_ratio_in_profit",
            "local_peaks_count",
            "pullback_magnitude_median",
            "time_to_peak_mfe_relative",
            "bars_held_feature",
        ],
    )
    return out


# ============================================================
# Feature diagnostics
# ============================================================


def feature_diagnostics(
    features: pd.DataFrame, bins: int, modal_threshold: float
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """Per-feature percentiles + degeneracy flag (modal histogram bin > threshold)."""
    rows = []
    flags: Dict[str, bool] = {}
    for col in FEATURE_COLS:
        v = features[col].to_numpy(dtype=float)
        n = v.size
        pctiles = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
        frac_zero = float(np.sum(v == 0.0)) / max(n, 1)
        frac_one = float(np.sum(v == 1.0)) / max(n, 1)
        # Modal-bin mass via 50-bin histogram.
        if v.max() > v.min():
            hist, _ = np.histogram(v, bins=bins)
        else:
            hist = np.array([n])
        modal_mass = float(hist.max()) / max(n, 1)
        degenerate = modal_mass > modal_threshold
        flags[col] = degenerate
        rows.append(
            (
                col,
                float(pctiles[0]),
                float(pctiles[1]),
                float(pctiles[2]),
                float(pctiles[3]),
                float(pctiles[4]),
                float(pctiles[5]),
                float(pctiles[6]),
                frac_zero,
                frac_one,
                modal_mass,
                degenerate,
            )
        )
    diag = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "p1",
            "p5",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "frac_at_zero",
            "frac_at_one",
            "modal_bin_mass",
            "degenerate_flag",
        ],
    )
    return diag, flags


def write_histograms(features: pd.DataFrame, bins: int, out_path: Path) -> None:
    """Write a 2x2 histogram panel. Best-effort deterministic; not gated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), FEATURE_COLS):
        v = features[col].to_numpy(dtype=float)
        ax.hist(v, bins=bins, color="#4c72b0", edgecolor="black", linewidth=0.3)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("trade count", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    fig.suptitle("Arc 2 redo2 Step 2 — path-shape feature histograms (n=12,262)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, metadata={"Software": ""})
    plt.close(fig)


# ============================================================
# K-means sweep
# ============================================================


@dataclass
class KFitResult:
    k: int
    labels: np.ndarray
    centroids_unscaled: np.ndarray  # shape (k, 4)
    cluster_counts: np.ndarray      # shape (k,)
    cluster_fractions: np.ndarray   # shape (k,)
    silhouette: float
    max_cluster_fraction: float
    min_cluster_count: int


def kmeans_sweep(features: pd.DataFrame, cfg: dict) -> Dict[int, KFitResult]:
    """Fit KMeans for each K in sweep; return results keyed by K."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    cl = cfg["clustering"]
    X_raw = features[FEATURE_COLS].to_numpy(dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    results: Dict[int, KFitResult] = {}
    for k in cl["k_sweep"]:
        km = KMeans(
            n_clusters=int(k),
            random_state=int(cl["random_state"]),
            n_init=int(cl["n_init"]),
            max_iter=int(cl["max_iter"]),
        )
        labels = km.fit_predict(X)
        # Unscaled centroids (back-transform).
        centroids_unscaled = scaler.inverse_transform(km.cluster_centers_)
        counts = np.bincount(labels, minlength=int(k)).astype(np.int64)
        fractions = counts / counts.sum()
        sil = float(
            silhouette_score(
                X,
                labels,
                sample_size=int(cl["silhouette_sample_size"]),
                random_state=int(cl["random_state"]),
            )
        )
        results[int(k)] = KFitResult(
            k=int(k),
            labels=labels,
            centroids_unscaled=centroids_unscaled,
            cluster_counts=counts,
            cluster_fractions=fractions,
            silhouette=sil,
            max_cluster_fraction=float(fractions.max()),
            min_cluster_count=int(counts.min()),
        )
    return results


# ============================================================
# Gate and K selection
# ============================================================


def apply_gate(results: Dict[int, KFitResult], cfg: dict) -> Dict[int, Tuple[bool, str]]:
    g = cfg["gates"]
    sil_min = float(g["silhouette_min"])
    max_frac = float(g["max_cluster_fraction"])
    min_count = int(g["min_cluster_count"])
    out: Dict[int, Tuple[bool, str]] = {}
    for k, r in results.items():
        c1 = r.silhouette >= sil_min
        c2 = r.max_cluster_fraction <= max_frac
        c3 = r.min_cluster_count >= min_count
        passes = c1 and c2 and c3
        why = []
        if not c1:
            why.append(f"silhouette {r.silhouette:.4f} < {sil_min}")
        if not c2:
            why.append(f"max_frac {r.max_cluster_fraction:.4f} > {max_frac}")
        if not c3:
            why.append(f"min_count {r.min_cluster_count} < {min_count}")
        out[k] = (passes, "PASS" if passes else "FAIL: " + "; ".join(why))
    return out


def select_k(
    results: Dict[int, KFitResult],
    gate: Dict[int, Tuple[bool, str]],
    tie_tolerance: float = 0.01,
) -> Tuple[Optional[int], Optional[int], List[int]]:
    """v2.1.1 §6 / §1.21 K-selection.

    Procedure:
      1. Among K satisfying the §6 gate, find the highest-silhouette K (`k_best`).
      2. Any K whose silhouette is within `tie_tolerance` (absolute) of `k_best`'s
         silhouette is considered tied for selection.
      3. Among the tied set, prefer the smallest K (parsimony per Open-12).

    Returns (chosen_k, k_best, tied_ks). `chosen_k` is the smallest within
    tolerance; `k_best` is the highest-silhouette; `tied_ks` is the sorted
    list of K values within tolerance of k_best (always includes k_best).
    All three None / empty if no K passes the gate.
    """
    passing = [(k, results[k].silhouette) for k, (ok, _) in gate.items() if ok]
    if not passing:
        return None, None, []
    max_sil = max(s for _, s in passing)
    k_best = min(k for k, s in passing if s == max_sil)
    tied_ks = sorted([k for k, s in passing if (max_sil - s) <= tie_tolerance])
    chosen_k = tied_ks[0]  # smallest K within tolerance
    return chosen_k, k_best, tied_ks


# ============================================================
# Archetype matching (v2.0 §11)
# ============================================================


@dataclass
class ArchetypeMatch:
    label: str
    rule: str
    boundary: bool


def _within(value: float, threshold: float, frac: float) -> bool:
    """True if value is within `frac` (relative) of threshold."""
    if threshold == 0:
        return abs(value - threshold) <= frac
    return abs(value - threshold) <= frac * abs(threshold)


def match_archetype(centroid: Dict[str, float], boundary_frac: float) -> ArchetypeMatch:
    """Match a cluster centroid against §11 patterns using the four path-shape
    features only. Step-3 quantities (fwd_mfe_p50, pct_peak_and_collapse) and
    Step-3 distribution shape (bimodal) are deferred — clusters whose §11 row
    requires them get flagged for empirical test at Step 3.
    """
    mono = float(centroid["monotonicity_ratio_in_profit"])
    peaks = float(centroid["local_peaks_count"])
    pull = float(centroid["pullback_magnitude_median"])
    ttp = float(centroid["time_to_peak_mfe_relative"])

    matches: List[Tuple[str, str]] = []
    near: List[str] = []

    # 1. Monotone ascent
    mono_a = mono >= 0.55
    peaks_a = peaks <= 4
    ttp_a = ttp >= 0.50
    if mono_a and peaks_a and ttp_a:
        matches.append(
            (
                "Monotone ascent",
                "monotonicity>=0.55 AND local_peaks<=4 AND time_to_peak_rel>=0.50",
            )
        )
    else:
        # Near-match if 2 of 3 satisfied AND the missing is within boundary_frac.
        sat = [mono_a, peaks_a, ttp_a]
        if sum(sat) == 2:
            if (not mono_a and _within(mono, 0.55, boundary_frac)) \
               or (not peaks_a and _within(peaks, 4, boundary_frac)) \
               or (not ttp_a and _within(ttp, 0.50, boundary_frac)):
                near.append("Monotone ascent")

    # 2. Stepwise climber
    mono_b = mono >= 0.50
    peaks_b = 5 <= peaks <= 30
    pull_b = pull <= 0.5
    ttp_b = ttp >= 0.50
    if mono_b and peaks_b and pull_b and ttp_b:
        matches.append(
            (
                "Stepwise climber",
                "monotonicity>=0.50 AND local_peaks in [5,30] AND pullback<=0.5 AND time_to_peak_rel>=0.50",
            )
        )
    else:
        sat = [mono_b, peaks_b, pull_b, ttp_b]
        if sum(sat) == 3:
            checks = [
                (not mono_b, _within(mono, 0.50, boundary_frac)),
                (not peaks_b, _within(peaks, 5, boundary_frac) or _within(peaks, 30, boundary_frac)),
                (not pull_b, _within(pull, 0.5, boundary_frac)),
                (not ttp_b, _within(ttp, 0.50, boundary_frac)),
            ]
            if any(missing and close for missing, close in checks):
                near.append("Stepwise climber")

    # 3+4. Early-peak hold / Peak-and-collapse — both share time_to_peak_rel<=0.30
    # and disambiguate via Step-3 pct_peak_and_collapse. The cluster needs a Step-3
    # empirical test, so flag as boundary by construction.
    pair_match = ttp <= 0.30
    pair_near = (not pair_match) and _within(ttp, 0.30, boundary_frac)
    if pair_match:
        matches.append(
            (
                "Early-peak hold OR Peak-and-collapse",
                "time_to_peak_rel<=0.30 (disambiguation needs Step-3 pct_peak_and_collapse)",
            )
        )
    elif pair_near:
        near.append("Early-peak hold OR Peak-and-collapse")

    # 5. V-shape recovery — requires MAE-timing data not in the 4 features.
    # Skipped at Step 2; if the cluster fits no other pattern it gets flagged for
    # Step 3 inspection.

    # 6. Random walk
    peaks_r = peaks >= 8
    mono_r = mono <= 0.30
    pull_r = pull >= 1.0
    if peaks_r and mono_r and pull_r:
        matches.append(
            (
                "Random walk",
                "local_peaks>=8 AND monotonicity<=0.30 AND pullback>=1.0",
            )
        )
    else:
        sat = [peaks_r, mono_r, pull_r]
        if sum(sat) == 2:
            checks = [
                (not peaks_r, _within(peaks, 8, boundary_frac)),
                (not mono_r, _within(mono, 0.30, boundary_frac)),
                (not pull_r, _within(pull, 1.0, boundary_frac)),
            ]
            if any(missing and close for missing, close in checks):
                near.append("Random walk")

    # Resolution
    if not matches and not near:
        return ArchetypeMatch(label="unclassified", rule="no §11 pattern matched", boundary=False)
    if not matches and near:
        return ArchetypeMatch(
            label="unclassified (near: " + ", ".join(near) + ")",
            rule="no §11 rule fully matched; near " + ", ".join(near),
            boundary=True,
        )
    if len(matches) == 1 and not near:
        # Early-peak hold OR Peak-and-collapse is a §11 boundary by construction.
        intrinsic_boundary = pair_match
        return ArchetypeMatch(
            label=matches[0][0], rule=matches[0][1], boundary=intrinsic_boundary
        )
    if len(matches) == 1 and near:
        return ArchetypeMatch(
            label=matches[0][0] + " (boundary)",
            rule=matches[0][1] + "; near " + ", ".join(near),
            boundary=True,
        )
    # Multiple matches.
    labels = [m[0] for m in matches]
    rules = "; ".join(m[1] for m in matches)
    return ArchetypeMatch(
        label="boundary (" + " OR ".join(labels) + ")",
        rule=rules + ("; near " + ", ".join(near) if near else ""),
        boundary=True,
    )


# ============================================================
# CSV writers (deterministic)
# ============================================================


def _fmt_g(x: float) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return ""
    except Exception:
        return ""
    return f"{xf:.10g}"


def write_features_csv(out_path: Path, features: pd.DataFrame, trades_csv_path: Path) -> None:
    """path_features.csv: one row per trade with features + bars_held + exit_reason."""
    # Pull exit_reason from trades_all.csv for diagnostics (not used in clustering).
    trades = pd.read_csv(trades_csv_path, usecols=["trade_id", "exit_reason"])
    merged = features.merge(trades, on="trade_id", how="left").sort_values("trade_id")
    cols = [
        "trade_id",
        "monotonicity_ratio_in_profit",
        "local_peaks_count",
        "pullback_magnitude_median",
        "time_to_peak_mfe_relative",
        "bars_held_feature",
        "exit_reason",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for _, row in merged.iterrows():
            f.write(
                ",".join(
                    [
                        str(int(row["trade_id"])),
                        _fmt_g(row["monotonicity_ratio_in_profit"]),
                        str(int(row["local_peaks_count"])),
                        _fmt_g(row["pullback_magnitude_median"]),
                        _fmt_g(row["time_to_peak_mfe_relative"]),
                        str(int(row["bars_held_feature"])),
                        str(row["exit_reason"]),
                    ]
                )
                + "\n"
            )


def write_diagnostics_csv(out_path: Path, diag: pd.DataFrame) -> None:
    cols = [
        "feature",
        "p1",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "frac_at_zero",
        "frac_at_one",
        "modal_bin_mass",
        "degenerate_flag",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for _, row in diag.iterrows():
            f.write(
                ",".join(
                    [
                        str(row["feature"]),
                        _fmt_g(row["p1"]),
                        _fmt_g(row["p5"]),
                        _fmt_g(row["p25"]),
                        _fmt_g(row["p50"]),
                        _fmt_g(row["p75"]),
                        _fmt_g(row["p95"]),
                        _fmt_g(row["p99"]),
                        _fmt_g(row["frac_at_zero"]),
                        _fmt_g(row["frac_at_one"]),
                        _fmt_g(row["modal_bin_mass"]),
                        str(bool(row["degenerate_flag"])),
                    ]
                )
                + "\n"
            )


def write_clusters_csv(
    out_path: Path, features_sorted: pd.DataFrame, labels: np.ndarray
) -> None:
    """features_sorted indexed 0..n-1 in the order labels were produced."""
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("trade_id,cluster_id\n")
        # Resort by trade_id ascending for stable output.
        idx_order = np.argsort(features_sorted["trade_id"].to_numpy(), kind="mergesort")
        tids = features_sorted["trade_id"].to_numpy()[idx_order]
        labs = labels[idx_order]
        for tid, lab in zip(tids, labs):
            f.write(f"{int(tid)},{int(lab)}\n")


def write_centroids_csv(out_path: Path, result: KFitResult) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("cluster_id," + ",".join(FEATURE_COLS) + ",count,fraction\n")
        for cid in range(result.k):
            cent = result.centroids_unscaled[cid]
            f.write(
                ",".join(
                    [
                        str(int(cid)),
                        _fmt_g(cent[0]),
                        _fmt_g(cent[1]),
                        _fmt_g(cent[2]),
                        _fmt_g(cent[3]),
                        str(int(result.cluster_counts[cid])),
                        _fmt_g(float(result.cluster_fractions[cid])),
                    ]
                )
                + "\n"
            )


def write_silhouette_txt(out_path: Path, silhouette: float) -> None:
    out_path.write_text(f"{silhouette:.10g}\n", encoding="utf-8")


def write_sweep_csv(
    out_path: Path,
    results: Dict[int, KFitResult],
    gate: Dict[int, Tuple[bool, str]],
) -> None:
    cols = [
        "k",
        "silhouette",
        "max_cluster_fraction",
        "min_cluster_count",
        "gate_pass",
        "gate_detail",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for k in sorted(results.keys()):
            r = results[k]
            ok, why = gate[k]
            f.write(
                ",".join(
                    [
                        str(int(k)),
                        _fmt_g(r.silhouette),
                        _fmt_g(r.max_cluster_fraction),
                        str(int(r.min_cluster_count)),
                        "PASS" if ok else "FAIL",
                        why.replace(",", ";"),
                    ]
                )
                + "\n"
            )


def write_archetypes_csv(
    out_path: Path, result: KFitResult, matches: List[ArchetypeMatch]
) -> None:
    cols = (
        ["cluster_id"]
        + [f"centroid_{c}" for c in FEATURE_COLS]
        + ["cluster_size", "cluster_fraction", "archetype_label", "matching_rule", "boundary_flag"]
    )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for cid in range(result.k):
            cent = result.centroids_unscaled[cid]
            m = matches[cid]
            rule = m.rule.replace(",", ";")
            f.write(
                ",".join(
                    [
                        str(int(cid)),
                        _fmt_g(cent[0]),
                        _fmt_g(cent[1]),
                        _fmt_g(cent[2]),
                        _fmt_g(cent[3]),
                        str(int(result.cluster_counts[cid])),
                        _fmt_g(float(result.cluster_fractions[cid])),
                        m.label.replace(",", ";"),
                        rule,
                        str(bool(m.boundary)),
                    ]
                )
                + "\n"
            )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# PR #129 concordance check
# ============================================================


def pr129_concordance(results: Dict[int, KFitResult]) -> Dict[str, str]:
    """PR #129 reported K=5 archetype 1 with centroid ~ monotonicity 0.54,
    local_peaks 26.5, size_fraction ~0.155. Find the cluster in this redo's
    K=5 closest to (0.54, 26.5) in (monotonicity, local_peaks) space and report.
    """
    if 5 not in results:
        return {"status": "no K=5 result", "detail": ""}
    r = results[5]
    target_mono = 0.54
    target_peaks = 26.5
    target_size_frac = 0.155
    best_cid = -1
    best_dist = math.inf
    for cid in range(r.k):
        m = float(r.centroids_unscaled[cid][0])  # monotonicity
        p = float(r.centroids_unscaled[cid][1])  # local_peaks
        d = ((m - target_mono) / 0.10) ** 2 + ((p - target_peaks) / 10.0) ** 2
        if d < best_dist:
            best_dist = d
            best_cid = cid
    cent = r.centroids_unscaled[best_cid]
    size_frac = float(r.cluster_fractions[best_cid])
    mono_close = abs(float(cent[0]) - target_mono) <= 0.10
    peaks_close = abs(float(cent[1]) - target_peaks) <= 10.0
    size_close = abs(size_frac - target_size_frac) <= 0.05
    status = "MATCH" if (mono_close and peaks_close and size_close) else "DIVERGENT"
    detail = (
        f"closest cluster_id={best_cid}: "
        f"mono={float(cent[0]):.4f} (target 0.54), "
        f"local_peaks={float(cent[1]):.2f} (target 26.5), "
        f"size_frac={size_frac:.4f} (target 0.155); "
        f"checks: mono<=0.10 {mono_close}, peaks<=10 {peaks_close}, size<=0.05 {size_close}"
    )
    return {"status": status, "detail": detail}


# ============================================================
# Jaccard overlap vs prior arc_2_redo K=4 clustering
# ============================================================


def jaccard_vs_prior(
    fork_features: pd.DataFrame,
    fork_labels: np.ndarray,
    prior_clusters_path: Path,
) -> Optional[pd.DataFrame]:
    """Compute |A ∩ B| / |A ∪ B| matrix between fork chosen-K clusters and
    prior arc_2_redo K=4 clusters, indexed by trade_id.

    Returns a DataFrame with rows=fork cluster_id, cols=prior cluster_id, plus
    a `best_match_prior_cid` + `best_jaccard` per row. None if prior file
    missing.
    """
    if not prior_clusters_path.exists():
        return None
    prior = pd.read_csv(prior_clusters_path)
    if not {"trade_id", "cluster_id"}.issubset(prior.columns):
        return None

    # The fork's features are sorted by trade_id by the time labels are produced.
    # Reconstruct trade_id -> fork_cluster mapping aligned to fork_labels order:
    fork_sorted = fork_features.sort_values("trade_id").reset_index(drop=True)
    fork_map = pd.DataFrame(
        {
            "trade_id": fork_sorted["trade_id"].astype(int).to_numpy(),
            "fork_cid": fork_labels.astype(int),
        }
    )
    prior_map = prior[["trade_id", "cluster_id"]].rename(
        columns={"cluster_id": "prior_cid"}
    )
    prior_map["trade_id"] = prior_map["trade_id"].astype(int)
    prior_map["prior_cid"] = prior_map["prior_cid"].astype(int)

    merged = fork_map.merge(prior_map, on="trade_id", how="inner")
    if len(merged) == 0:
        return None

    fork_cids = sorted(merged["fork_cid"].unique())
    prior_cids = sorted(merged["prior_cid"].unique())

    rows = []
    for fcid in fork_cids:
        fset = set(merged.loc[merged["fork_cid"] == fcid, "trade_id"].to_numpy())
        row = {"fork_cluster_id": int(fcid), "fork_size": len(fset)}
        best_j, best_pcid = -1.0, -1
        for pcid in prior_cids:
            pset = set(merged.loc[merged["prior_cid"] == pcid, "trade_id"].to_numpy())
            inter = len(fset & pset)
            union = len(fset | pset)
            j = inter / union if union > 0 else 0.0
            row[f"prior_cid_{int(pcid)}_jaccard"] = j
            row[f"prior_cid_{int(pcid)}_overlap_count"] = inter
            if j > best_j:
                best_j = j
                best_pcid = int(pcid)
        row["best_match_prior_cid"] = best_pcid
        row["best_jaccard"] = best_j
        rows.append(row)
    return pd.DataFrame(rows)


def write_jaccard_csv(out_path: Path, jacc: Optional[pd.DataFrame]) -> None:
    if jacc is None:
        out_path.write_text(
            "# jaccard overlap unavailable (prior clusters file missing)\n",
            encoding="utf-8",
        )
        return
    # Stable column order: fork_cluster_id, fork_size, prior_cid_*_jaccard,
    # prior_cid_*_overlap_count, best_*.
    front = ["fork_cluster_id", "fork_size"]
    prior_cols = [
        c
        for c in jacc.columns
        if c.startswith("prior_cid_") and c.endswith("_jaccard")
    ]
    prior_cols = sorted(prior_cols, key=lambda s: int(s.split("_")[2]))
    overlap_cols = [
        c
        for c in jacc.columns
        if c.startswith("prior_cid_") and c.endswith("_overlap_count")
    ]
    overlap_cols = sorted(overlap_cols, key=lambda s: int(s.split("_")[2]))
    tail = ["best_match_prior_cid", "best_jaccard"]
    ordered = front + prior_cols + overlap_cols + tail
    jacc_sorted = jacc[ordered].sort_values("fork_cluster_id").reset_index(drop=True)
    lines = [",".join(ordered)]
    for _, row in jacc_sorted.iterrows():
        cells = []
        for c in ordered:
            v = row[c]
            if c in front or c in overlap_cols or c == "best_match_prior_cid":
                cells.append(str(int(v)))
            else:
                cells.append(_fmt_g(float(v)))
        lines.append(",".join(cells))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Summary writer
# ============================================================


def write_summary(
    out_path: Path,
    cfg: dict,
    features: pd.DataFrame,
    diag: pd.DataFrame,
    degen_flags: Dict[str, bool],
    results: Dict[int, KFitResult],
    gate: Dict[int, Tuple[bool, str]],
    chosen_k: Optional[int],
    k_best: Optional[int],
    tied_ks: List[int],
    matches: Optional[List[ArchetypeMatch]],
    pr129: Dict[str, str],
    jaccard: Optional[pd.DataFrame],
    determinism_gate: str,
    csv_hashes_run1: Dict[str, str],
    csv_hashes_run2: Optional[Dict[str, str]],
) -> None:
    lines: List[str] = []
    lines.append("# Arc 2 redo2 — Step 2 path-shape clustering summary")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §§1, 6, 11")
    lines.append("")
    halted = chosen_k is None
    disposition = "FAIL (halt arc)" if halted else "PASS"
    lines.append(f"**Step 2 disposition: {disposition}**")
    lines.append("")
    if not halted:
        m_pass = sum(1 for ok, _ in gate.values() if ok)
        lines.append(
            f"Gate (§6): silhouette >= {cfg['gates']['silhouette_min']}, "
            f"max cluster fraction <= {cfg['gates']['max_cluster_fraction']}, "
            f"all clusters >= {cfg['gates']['min_cluster_count']} trades. "
            f"{m_pass} of {len(gate)} sweep K values pass."
        )
        tie_tol = float(cfg.get("k_selection", {}).get("tie_tolerance_absolute", 0.01))
        chose_smaller = (chosen_k != k_best) if k_best is not None else False
        rule_note = (
            f"k_best={k_best} at silhouette {results[k_best].silhouette:.4f}; "
            f"tied set (within ±{tie_tol} absolute): {tied_ks}; "
            f"selected smallest K within tolerance"
            + (" (would have been k_best under v2.0 highest-only rule)" if chose_smaller else "")
        )
        lines.append(
            f"Chosen K: **{chosen_k}** "
            f"(silhouette {results[chosen_k].silhouette:.4f}, "
            f"max_frac {results[chosen_k].max_cluster_fraction:.4f}, "
            f"min_count {results[chosen_k].min_cluster_count}). "
            f"Selection rule (v2.1.1 §6): {rule_note}."
        )
    lines.append("")

    # Feature diagnostics
    lines.append("## Path-shape feature distributions")
    lines.append("")
    lines.append("| Feature | p1 | p5 | p25 | p50 | p75 | p95 | p99 | %=0 | %=1 | modal-bin mass | degenerate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for _, row in diag.iterrows():
        lines.append(
            f"| {row['feature']} "
            f"| {row['p1']:.4g} | {row['p5']:.4g} | {row['p25']:.4g} | {row['p50']:.4g} "
            f"| {row['p75']:.4g} | {row['p95']:.4g} | {row['p99']:.4g} "
            f"| {row['frac_at_zero']*100:.2f}% | {row['frac_at_one']*100:.2f}% "
            f"| {row['modal_bin_mass']*100:.2f}% "
            f"| {'YES' if row['degenerate_flag'] else 'no'} |"
        )
    n_degen = sum(1 for v in degen_flags.values() if v)
    lines.append("")
    lines.append(
        f"Degenerate features (modal bin > {cfg['degeneracy']['modal_bin_mass_threshold']*100:.0f}%): "
        f"{n_degen} of 4 — "
        + (", ".join([f for f, v in degen_flags.items() if v]) if n_degen else "none")
    )
    if n_degen >= 2:
        lines.append("")
        lines.append("**Two+ degenerate features → halt arc per v2.0 §6.**")
    elif n_degen == 1:
        lines.append("")
        lines.append("Single degenerate feature flagged; per v2.0 §6 this is a flag, not a halt.")
    lines.append("")

    # K-sweep table
    lines.append("## K-sweep results")
    lines.append("")
    lines.append("| K | silhouette | max cluster fraction | min cluster count | gate |")
    lines.append("|---:|---:|---:|---:|:---|")
    for k in sorted(results.keys()):
        r = results[k]
        ok, why = gate[k]
        gate_str = "PASS" if ok else f"FAIL ({why.split('FAIL: ')[-1]})"
        lines.append(
            f"| {k} | {r.silhouette:.4f} | {r.max_cluster_fraction:.4f} "
            f"| {r.min_cluster_count} | {gate_str} |"
        )
    lines.append("")

    if not halted:
        # Chosen-K centroid table
        r = results[chosen_k]
        lines.append(f"## Chosen K={chosen_k} — cluster centroids and assignments")
        lines.append("")
        lines.append(
            "| cluster_id | monotonicity | local_peaks | pullback | time_to_peak_rel "
            "| size | fraction | archetype | matching rule | boundary |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|:---:|")
        assert matches is not None
        for cid in range(r.k):
            cent = r.centroids_unscaled[cid]
            m = matches[cid]
            lines.append(
                f"| {cid} | {cent[0]:.4f} | {cent[1]:.2f} | {cent[2]:.4f} | {cent[3]:.4f} "
                f"| {int(r.cluster_counts[cid])} | {float(r.cluster_fractions[cid]):.4f} "
                f"| {m.label} | {m.rule} | {'YES' if m.boundary else 'no'} |"
            )
        lines.append("")
        boundary_clusters = [
            cid for cid, m in enumerate(matches) if m.boundary
        ]
        lines.append(
            f"Boundary-flagged clusters: {len(boundary_clusters)}"
            + (f" ({', '.join(map(str, boundary_clusters))})" if boundary_clusters else "")
        )
        lines.append("")

    # Interpretation
    if not halted and matches is not None:
        r = results[chosen_k]
        notes: List[str] = []
        for cid in range(r.k):
            cent = r.centroids_unscaled[cid]
            m = matches[cid]
            mono = float(cent[0])
            peaks = float(cent[1])
            pull = float(cent[2])
            ttp = float(cent[3])
            frac = float(r.cluster_fractions[cid])
            if m.label == "unclassified":
                # Diagnostic context for clusters that fall outside §11.
                hint = ""
                if mono >= 0.50 and 5 <= peaks <= 30 and ttp >= 0.50 and pull > 0.5:
                    hint = (
                        " — looks like Stepwise climber with deeper pullbacks "
                        f"(pullback {pull:.2f} > 0.5 threshold)"
                    )
                elif mono >= 0.50 and peaks < 5 and ttp < 0.50:
                    hint = (
                        " — moderate climber, ttp in [0.30, 0.50] gap between "
                        "Early-peak hold and Monotone ascent §11 patterns"
                    )
                notes.append(
                    f"- Cluster {cid} (size {frac:.4f}): unclassified{hint}. "
                    f"Centroid mono={mono:.4f}, peaks={peaks:.2f}, pull={pull:.4f}, ttp={ttp:.4f}."
                )
            elif m.label.startswith("Early-peak hold OR Peak-and-collapse"):
                if mono < 0.10 and pull < 0.10:
                    notes.append(
                        f"- Cluster {cid} (size {frac:.4f}): centroid mono={mono:.4f}, "
                        f"pull={pull:.4f} are near zero → trades that mostly never went "
                        "in profit (degenerate path-shape). Step-3 fwd_mfe_p50 check will "
                        "likely separate these from genuine Early-peak hold trades."
                    )
        if notes:
            lines.append("## Cluster interpretation notes")
            lines.append("")
            lines.extend(notes)
            lines.append("")

    # PR #129 concordance
    lines.append("## PR #129 concordance (K=5)")
    lines.append("")
    lines.append(f"- Status: **{pr129['status']}**")
    lines.append(f"- Detail: {pr129['detail']}")
    lines.append(
        "- Reference: PR #129 reported K=5 archetype 1 with monotonicity ~0.54, "
        "local_peaks ~26.5, size_fraction ~0.155."
    )
    lines.append("")

    # Jaccard vs prior arc_2_redo K=4
    lines.append("## Jaccard overlap vs prior arc_2_redo K=4")
    lines.append("")
    if jaccard is None:
        lines.append(
            "Jaccard overlap unavailable — prior `results/l_arc_2_redo/step2/clusters_K4.csv` "
            "not found."
        )
    else:
        fork_cids = sorted(jaccard["fork_cluster_id"].astype(int).tolist())
        prior_cids = sorted(
            int(c.split("_")[2])
            for c in jaccard.columns
            if c.startswith("prior_cid_") and c.endswith("_jaccard")
        )
        header = "| fork \\ prior | " + " | ".join(str(p) for p in prior_cids) + " | best |"
        sep = "|---:|" + "|".join(["---:"] * (len(prior_cids) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        diag_vals: List[float] = []
        for fcid in fork_cids:
            row = jaccard[jaccard["fork_cluster_id"] == fcid].iloc[0]
            cells = [str(fcid)]
            for p in prior_cids:
                cells.append(f"{float(row[f'prior_cid_{p}_jaccard']):.4f}")
            best_p = int(row["best_match_prior_cid"])
            best_j = float(row["best_jaccard"])
            cells.append(f"prior {best_p} @ {best_j:.4f}")
            diag_vals.append(best_j)
            lines.append("| " + " | ".join(cells) + " |")
        avg_best = sum(diag_vals) / max(len(diag_vals), 1)
        lines.append("")
        lines.append(
            f"Average best-match Jaccard: **{avg_best:.4f}**. Interpretation: "
            + (
                "near-1.0 average → fork clustering decomposes the pool "
                "identically to the prior redo (minor reassignments at cluster boundaries "
                "are expected due to the 1-bar definition change in is_held vs still_open)."
                if avg_best >= 0.85
                else "moderate divergence — fork's clustering shifts cluster boundaries vs "
                "prior. Investigate whether the is_held=1 vs still_open=1 1-bar definition "
                "shift is the cause (likely) or whether a deeper change has crept in."
                if avg_best >= 0.50
                else "structural divergence — clusters are fundamentally different from prior."
            )
        )
    lines.append("")

    # Determinism
    lines.append("## Determinism (CSV byte-identical)")
    lines.append("")
    lines.append(f"**Gate: {determinism_gate}**")
    lines.append("")
    lines.append("Run-1 CSV hashes:")
    for name, h in sorted(csv_hashes_run1.items()):
        lines.append(f"- `{name}`: `{h}`")
    if csv_hashes_run2 is not None:
        lines.append("")
        lines.append("Run-2 CSV hashes:")
        for name, h in sorted(csv_hashes_run2.items()):
            match = "MATCH" if csv_hashes_run1.get(name) == h else "MISMATCH"
            lines.append(f"- `{name}`: `{h}` ({match})")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Driver
# ============================================================


def _run_once(cfg: dict) -> Tuple[Dict[str, str], dict]:
    """Single full run; returns ({csv_name: sha256}, run_context_dict)."""
    in_cfg = cfg["input"]
    out_cfg = cfg["output"]

    step1_dir = Path(in_cfg["step1_dir"])
    if not step1_dir.is_absolute():
        step1_dir = (_REPO_ROOT / step1_dir).resolve()
    paths_csv = step1_dir / in_cfg["paths_csv"]
    trades_csv = step1_dir / in_cfg["trades_csv"]

    out_dir = Path(out_cfg["results_dir"])
    if not out_dir.is_absolute():
        out_dir = (_REPO_ROOT / out_cfg["results_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[step2] loading paths", file=sys.stderr)
    paths_df = pd.read_csv(
        paths_csv,
        usecols=["trade_id", "bar_offset", "close_r", "mfe_so_far_r", "is_held"],
    )
    print(f"[step2] paths rows: {len(paths_df):,}", file=sys.stderr)

    print("[step2] computing path-shape features", file=sys.stderr)
    features = compute_path_features(paths_df)
    print(f"[step2] features computed for {len(features)} trades", file=sys.stderr)

    features_path = out_dir / out_cfg["features_csv"]
    write_features_csv(features_path, features, trades_csv)

    diag, degen_flags = feature_diagnostics(
        features,
        bins=int(cfg["degeneracy"]["histogram_bins"]),
        modal_threshold=float(cfg["degeneracy"]["modal_bin_mass_threshold"]),
    )
    diag_path = out_dir / out_cfg["diagnostics_csv"]
    write_diagnostics_csv(diag_path, diag)

    hist_path = out_dir / out_cfg["histograms_png"]
    try:
        write_histograms(features, int(cfg["degeneracy"]["histogram_bins"]), hist_path)
    except Exception as e:
        print(f"[step2] histogram write skipped: {e}", file=sys.stderr)

    print("[step2] k-means sweep", file=sys.stderr)
    results = kmeans_sweep(features, cfg)
    gate = apply_gate(results, cfg)
    tie_tol = float(cfg.get("k_selection", {}).get("tie_tolerance_absolute", 0.01))
    chosen_k, k_best, tied_ks = select_k(results, gate, tie_tolerance=tie_tol)

    # Per-K outputs (clusters, centroids, silhouette txt).
    csv_hashes: Dict[str, str] = {}
    features_sorted = features.sort_values("trade_id").reset_index(drop=True)
    for k, r in sorted(results.items()):
        clusters_path = out_dir / f"clusters_K{k}.csv"
        write_clusters_csv(clusters_path, features_sorted, r.labels)
        cent_path = out_dir / f"centroids_K{k}.csv"
        write_centroids_csv(cent_path, r)
        sil_txt = out_dir / f"silhouette_K{k}.txt"
        write_silhouette_txt(sil_txt, r.silhouette)
        csv_hashes[clusters_path.name] = _file_sha256(clusters_path)
        csv_hashes[cent_path.name] = _file_sha256(cent_path)

    sweep_path = out_dir / out_cfg["sweep_csv"]
    write_sweep_csv(sweep_path, results, gate)

    # Archetype assignments at chosen K.
    matches: Optional[List[ArchetypeMatch]] = None
    arche_path = out_dir / out_cfg["archetypes_csv"]
    if chosen_k is not None:
        r = results[chosen_k]
        boundary_frac = float(cfg["archetype"]["boundary_relative_distance"])
        matches = []
        for cid in range(r.k):
            cent = r.centroids_unscaled[cid]
            cd = {
                FEATURE_COLS[0]: float(cent[0]),
                FEATURE_COLS[1]: float(cent[1]),
                FEATURE_COLS[2]: float(cent[2]),
                FEATURE_COLS[3]: float(cent[3]),
            }
            matches.append(match_archetype(cd, boundary_frac))
        write_archetypes_csv(arche_path, r, matches)
    else:
        # Write empty header-only file so downstream finds the path.
        with arche_path.open("w", encoding="utf-8", newline="") as f:
            f.write(
                "cluster_id,"
                + ",".join([f"centroid_{c}" for c in FEATURE_COLS])
                + ",cluster_size,cluster_fraction,archetype_label,matching_rule,boundary_flag\n"
            )

    # Hash all CSVs we care about.
    for name in [
        out_cfg["features_csv"],
        out_cfg["diagnostics_csv"],
        out_cfg["sweep_csv"],
        out_cfg["archetypes_csv"],
    ]:
        p = out_dir / name
        csv_hashes[p.name] = _file_sha256(p)

    pr129 = pr129_concordance(results)

    # Jaccard vs prior arc_2_redo K=4 (or whichever prior file the config points to).
    prior_clusters_rel = in_cfg.get("prior_clusters_csv")
    jaccard_df: Optional[pd.DataFrame] = None
    if prior_clusters_rel and chosen_k is not None:
        prior_clusters_path = Path(prior_clusters_rel)
        if not prior_clusters_path.is_absolute():
            prior_clusters_path = (_REPO_ROOT / prior_clusters_path).resolve()
        jaccard_df = jaccard_vs_prior(
            features, results[chosen_k].labels, prior_clusters_path
        )
    jacc_path = out_dir / out_cfg.get("jaccard_csv", "jaccard_vs_prior_K4.csv")
    write_jaccard_csv(jacc_path, jaccard_df)
    csv_hashes[jacc_path.name] = _file_sha256(jacc_path)

    ctx = {
        "features": features,
        "diag": diag,
        "degen_flags": degen_flags,
        "results": results,
        "gate": gate,
        "chosen_k": chosen_k,
        "k_best": k_best,
        "tied_ks": tied_ks,
        "matches": matches,
        "pr129": pr129,
        "jaccard": jaccard_df,
        "out_dir": out_dir,
    }
    return csv_hashes, ctx


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 2 redo2 Step 2 — path-shape clustering.")
    ap.add_argument("-c", "--config", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    print("[step2] === RUN 1 ===", file=sys.stderr)
    hashes1, ctx = _run_once(cfg)

    hashes2: Optional[Dict[str, str]] = None
    if bool(cfg["output"].get("determinism_check", True)):
        print("[step2] === RUN 2 (determinism) ===", file=sys.stderr)
        hashes2, _ = _run_once(cfg)

    if hashes2 is not None:
        det_gate = "PASS" if all(hashes1[k] == hashes2.get(k) for k in hashes1) else "FAIL"
    else:
        det_gate = "N/A"

    summary_path = ctx["out_dir"] / cfg["output"]["summary_md"]
    write_summary(
        summary_path,
        cfg,
        ctx["features"],
        ctx["diag"],
        ctx["degen_flags"],
        ctx["results"],
        ctx["gate"],
        ctx["chosen_k"],
        ctx["k_best"],
        ctx["tied_ks"],
        ctx["matches"],
        ctx["pr129"],
        ctx["jaccard"],
        det_gate,
        hashes1,
        hashes2,
    )

    chosen = ctx["chosen_k"]
    print(
        f"[step2] DONE. chosen_K={chosen}, det={det_gate}, "
        f"degenerate={sum(1 for v in ctx['degen_flags'].values() if v)}/4",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

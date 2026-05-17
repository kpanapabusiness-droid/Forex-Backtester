"""Arc 7 — Step 2 path-shape clustering.

L_ARC_PROTOCOL.md v2.1.2 §§6, 11, 17. Operates on Arc 7's §15a-compliant
trades_paths.csv (is_held=1 rows only). Computes the four outcome-blind
path-shape features per trade, sweeps K ∈ {3..7} with KMeans (random_state=42,
n_init=10, max_iter=300) on StandardScaler-transformed features, applies §6
gate, selects K under the v2.1.1 tie tolerance (0.01 absolute, smaller K
preferred), and matches each cluster's centroid against v2.1.2 §11 archetype
patterns (Stepwise climber local_peaks ceiling 50, not 30).

Schema note carried from Step 1: Arc 7's mfe_so_far_r is running max of
high_r (intrabar) per the reference impl. Use as-is — do not recompute
against close_r. §15a text discrepancy logged as cross-arc item.

Forked from scripts/arc_5/step2_cluster.py with surgical edits:
  - Drops the parity-vs-Arc-2-redo2 check and Jaccard-vs-prior-arc block
    (Arc 7 has no equivalent prior).
  - Adds `tentative_<archetype>` labels for §11 rows that depend on
    forward-geometry metrics (Early-peak hold, Peak-and-collapse, V-shape
    recovery) — path-shape conditions checked at Step 2, forward-geometry
    confirmation deferred to Step 3.
  - Stepwise climber local_peaks ceiling 50 (v2.1.2), not 30 (v2.1.1).
  - Outputs match the Step 2 dispatch's exact filenames.

Determinism: random_state=42, deterministic CSV writers (".10g" floats,
"\\n" line terminator, sorted iteration). Two consecutive runs produce
byte-identical CSVs; sha256s logged in STEP2_SUMMARY.md.

Usage:
    py scripts/arc_7/step2_cluster.py -c configs/arc_7/step2.yaml
"""

from __future__ import annotations

import argparse
import csv
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
# Feature computation (§6 + §17)
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
    """Return (mono, local_peaks, pullback_median, ttp_rel, n_held).

    Inputs are arrays restricted to is_held=1, ordered by bar_offset ascending
    (so array position == bar_offset since held window starts at offset 0).
    """
    n = int(close_r.size)
    if n == 0:
        return 0.0, 0, 0.0, 0.0, 0

    # 1. monotonicity_ratio_in_profit — among close_r > 0 bars, fraction with
    #    close_r >= previous in-profit bar.
    in_profit = close_r[close_r > 0]
    if in_profit.size >= 2:
        non_dec = int(np.sum(in_profit[1:] >= in_profit[:-1]))
        monotonicity = non_dec / (in_profit.size - 1)
    else:
        monotonicity = 0.0

    # 2. local_peaks_count — count of bars where mfe_so_far_r > previous bar.
    diffs = np.diff(mfe)
    peak_pos = np.where(diffs > 0)[0] + 1
    local_peaks = int(peak_pos.size)

    # 3. pullback_magnitude_median — operational definition: for consecutive
    #    peak pairs, earlier peak's mfe_so_far_r − min(close_r) between peaks;
    #    median across pairs (per Open-08 closure).
    pullbacks: List[float] = []
    for i in range(peak_pos.size - 1):
        a = int(peak_pos[i])
        b = int(peak_pos[i + 1])
        if b - a >= 2:
            between_min = float(close_r[a + 1 : b].min())
            pullbacks.append(float(mfe[a]) - between_min)
    pullback_median = float(np.median(pullbacks)) if pullbacks else 0.0

    # 4. time_to_peak_mfe_relative — time_to_peak_mfe / max(bars_held, 1),
    #    capped at 1.0. Trade never in profit → 0.
    max_close_r = float(close_r.max())
    if max_close_r <= 0.0:
        ttp_rel = 0.0
    else:
        max_mfe = float(mfe.max())
        ttp = int(np.argmax(mfe == max_mfe))
        ttp_rel = min(ttp / max(n, 1), 1.0)

    return monotonicity, local_peaks, pullback_median, ttp_rel, n


def compute_path_features(paths_df: pd.DataFrame) -> pd.DataFrame:
    held = paths_df[paths_df["is_held"] == 1].copy()
    held = held.sort_values(["trade_id", "bar_offset"], kind="mergesort")

    trade_ids = held["trade_id"].to_numpy()
    close_r = held["close_r"].to_numpy(dtype=float)
    mfe = held["mfe_so_far_r"].to_numpy(dtype=float)

    if trade_ids.size == 0:
        unique_ids = np.array([], dtype=np.int64)
        starts = np.array([], dtype=np.int64)
        ends = np.array([], dtype=np.int64)
    else:
        change = np.concatenate(([True], trade_ids[1:] != trade_ids[:-1]))
        starts = np.where(change)[0]
        ends = np.concatenate((starts[1:], [trade_ids.size]))
        unique_ids = trade_ids[starts]

    all_trade_ids = np.sort(paths_df["trade_id"].unique())

    feat_map: Dict[int, Tuple[float, int, float, float, int]] = {}
    for s, e, tid in zip(starts, ends, unique_ids):
        feat_map[int(tid)] = _features_for_trade(close_r[s:e], mfe[s:e])

    rows = []
    for tid in all_trade_ids:
        f = feat_map.get(int(tid), (0.0, 0, 0.0, 0.0, 0))
        rows.append((int(tid), f[0], f[1], f[2], f[3], f[4]))
    return pd.DataFrame(
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


# ============================================================
# Feature diagnostics (degeneracy check)
# ============================================================


def feature_diagnostics(
    features: pd.DataFrame, bins: int, modal_threshold: float
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    rows = []
    flags: Dict[str, bool] = {}
    for col in FEATURE_COLS:
        v = features[col].to_numpy(dtype=float)
        n = v.size
        pctiles = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
        frac_zero = float(np.sum(v == 0.0)) / max(n, 1)
        frac_one = float(np.sum(v == 1.0)) / max(n, 1)
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
    fig.suptitle(f"Arc 7 Step 2 — path-shape feature histograms (n={len(features):,})")
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
    centroids_unscaled: np.ndarray
    cluster_counts: np.ndarray
    cluster_fractions: np.ndarray
    silhouette: float
    max_cluster_fraction: float
    min_cluster_count: int


def kmeans_sweep(features: pd.DataFrame, cfg: dict) -> Dict[int, KFitResult]:
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
        centroids_unscaled = scaler.inverse_transform(km.cluster_centers_)
        counts = np.bincount(labels, minlength=int(k)).astype(np.int64)
        fractions = counts / counts.sum()
        sample_size = cl.get("silhouette_sample_size")
        sil_kwargs = {"random_state": int(cl["random_state"])}
        if sample_size is not None:
            sil_kwargs["sample_size"] = int(sample_size)
        sil = float(silhouette_score(X, labels, **sil_kwargs))
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
    """§6 K-selection (v2.1.1 Open-12 closure)."""
    passing = [(k, results[k].silhouette) for k, (ok, _) in gate.items() if ok]
    if not passing:
        return None, None, []
    max_sil = max(s for _, s in passing)
    k_best = min(k for k, s in passing if s == max_sil)
    tied_ks = sorted([k for k, s in passing if (max_sil - s) <= tie_tolerance])
    return tied_ks[0], k_best, tied_ks


# ============================================================
# Archetype matching (§11 v2.1.2 patterns)
# ============================================================


@dataclass
class ArchetypeMatch:
    label: str             # archetype name OR "boundary (...)" OR "unassigned"
    status: str            # "assigned" | "tentative" | "boundary" | "unassigned"
    rule: str
    notes: str


def _within(value: float, threshold: float, frac: float) -> bool:
    if threshold == 0:
        return abs(value - threshold) <= frac
    return abs(value - threshold) <= frac * abs(threshold)


def match_archetype(centroid: Dict[str, float], boundary_frac: float) -> ArchetypeMatch:
    """v2.1.2 §11 centroid-pattern match using path-shape features only.

    Forward-geometry-dependent rows (Early-peak hold, Peak-and-collapse,
    V-shape recovery) are surfaced as `tentative_*` when their path-shape
    conditions match — Step 3 confirms with fwd_mfe_p50 / pct_peak_and_collapse
    / MAE-timing.
    """
    mono = float(centroid["monotonicity_ratio_in_profit"])
    peaks = float(centroid["local_peaks_count"])
    pull = float(centroid["pullback_magnitude_median"])
    ttp = float(centroid["time_to_peak_mfe_relative"])

    assigned: List[Tuple[str, str]] = []
    tentative: List[Tuple[str, str, str]] = []   # (label, rule, deferred_check)
    near: List[str] = []

    # Row 1: Monotone ascent — mono >= 0.55, peaks <= 4, ttp >= 0.50.
    mono_a = mono >= 0.55
    peaks_a = peaks <= 4
    ttp_a = ttp >= 0.50
    if mono_a and peaks_a and ttp_a:
        assigned.append(
            ("Monotone ascent",
             "mono>=0.55 AND local_peaks<=4 AND ttp_rel>=0.50")
        )
    else:
        sat = sum([mono_a, peaks_a, ttp_a])
        if sat == 2 and (
            (not mono_a and _within(mono, 0.55, boundary_frac))
            or (not peaks_a and _within(peaks, 4, boundary_frac))
            or (not ttp_a and _within(ttp, 0.50, boundary_frac))
        ):
            near.append("Monotone ascent")

    # Row 2: Stepwise climber — mono >= 0.50, peaks in [5, 50] (v2.1.2 ceiling),
    # pullback <= 0.5, ttp >= 0.50.
    mono_b = mono >= 0.50
    peaks_b = 5 <= peaks <= 50
    pull_b = pull <= 0.5
    ttp_b = ttp >= 0.50
    if mono_b and peaks_b and pull_b and ttp_b:
        assigned.append(
            ("Stepwise climber",
             "mono>=0.50 AND local_peaks in [5,50] AND pullback<=0.5 AND ttp_rel>=0.50")
        )
    else:
        sat = sum([mono_b, peaks_b, pull_b, ttp_b])
        if sat == 3:
            checks = [
                (not mono_b, _within(mono, 0.50, boundary_frac)),
                (not peaks_b, _within(peaks, 5, boundary_frac) or _within(peaks, 50, boundary_frac)),
                (not pull_b, _within(pull, 0.5, boundary_frac)),
                (not ttp_b, _within(ttp, 0.50, boundary_frac)),
            ]
            if any(missing and close for missing, close in checks):
                near.append("Stepwise climber")

    # Rows 3 & 4: ttp_rel <= 0.30 — disambiguated by pct_peak_and_collapse
    # (Step 3). Path-shape only condition is shared.
    pair_match = ttp <= 0.30
    pair_near = (not pair_match) and _within(ttp, 0.30, boundary_frac)
    if pair_match:
        tentative.append(
            ("Early-peak hold OR Peak-and-collapse",
             "ttp_rel<=0.30",
             "Step 3 pct_peak_and_collapse: <0.30 → Early-peak hold; >=0.50 → Peak-and-collapse")
        )
    elif pair_near:
        near.append("Early-peak hold OR Peak-and-collapse")

    # Row 5: V-shape recovery — MAE early + peak in [0.4, 0.8]. MAE-timing
    # is Step 3 territory. We can pre-screen on ttp_rel ∈ [0.4, 0.8].
    vshape_path_ok = 0.4 <= ttp <= 0.8
    if vshape_path_ok:
        tentative.append(
            ("V-shape recovery",
             "ttp_rel in [0.4, 0.8]",
             "Step 3 MAE-before-peak >= 5 bars confirmation")
        )

    # Row 6: Random walk — peaks >= 8, mono <= 0.30, pullback >= 1R.
    peaks_r = peaks >= 8
    mono_r = mono <= 0.30
    pull_r = pull >= 1.0
    if peaks_r and mono_r and pull_r:
        assigned.append(
            ("Random walk",
             "local_peaks>=8 AND mono<=0.30 AND pullback>=1.0")
        )
    else:
        sat = sum([peaks_r, mono_r, pull_r])
        if sat == 2:
            checks = [
                (not peaks_r, _within(peaks, 8, boundary_frac)),
                (not mono_r, _within(mono, 0.30, boundary_frac)),
                (not pull_r, _within(pull, 1.0, boundary_frac)),
            ]
            if any(missing and close for missing, close in checks):
                near.append("Random walk")

    # Compose final label.
    if assigned:
        if len(assigned) == 1 and not tentative and not near:
            lab, rul = assigned[0]
            return ArchetypeMatch(label=lab, status="assigned", rule=rul, notes="")
        if len(assigned) == 1 and tentative:
            lab, rul = assigned[0]
            tent_labels = " + ".join(t[0] for t in tentative)
            notes = "also matches path-shape conds for: " + tent_labels + " (Step 3 disambiguation)"
            return ArchetypeMatch(label=lab, status="assigned", rule=rul, notes=notes)
        if len(assigned) > 1:
            labs = " OR ".join(a[0] for a in assigned)
            rul = "; ".join(a[1] for a in assigned)
            return ArchetypeMatch(
                label="boundary (" + labs + ")",
                status="boundary",
                rule=rul,
                notes="multiple assigned-archetype rules satisfied — Step 3 empirical test",
            )
    if tentative:
        if len(tentative) == 1:
            lab, rul, defer = tentative[0]
            return ArchetypeMatch(
                label="tentative_" + lab,
                status="tentative",
                rule=rul,
                notes=defer,
            )
        labs = " | ".join("tentative_" + t[0] for t in tentative)
        rules = "; ".join(t[1] for t in tentative)
        defers = "; ".join(t[2] for t in tentative)
        return ArchetypeMatch(
            label="tentative_boundary (" + labs + ")",
            status="boundary",
            rule=rules,
            notes=defers,
        )
    if near:
        return ArchetypeMatch(
            label="unassigned (near: " + ", ".join(near) + ")",
            status="unassigned",
            rule="no §11 pattern fully satisfied",
            notes="near: " + ", ".join(near),
        )
    return ArchetypeMatch(
        label="unassigned",
        status="unassigned",
        rule="no §11 pattern satisfied",
        notes="",
    )


# ============================================================
# CSV writers (deterministic; ".10g" floats, "\n" line terminator)
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
    """trade_id, pair, four features. One row per trade, sorted by trade_id."""
    trades = pd.read_csv(trades_csv_path, usecols=["trade_id", "pair"])
    merged = features.merge(trades, on="trade_id", how="left").sort_values("trade_id")
    cols = ["trade_id", "pair"] + FEATURE_COLS
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for _, row in merged.iterrows():
            w.writerow(
                [
                    str(int(row["trade_id"])),
                    str(row["pair"]),
                    _fmt_g(row["monotonicity_ratio_in_profit"]),
                    str(int(row["local_peaks_count"])),
                    _fmt_g(row["pullback_magnitude_median"]),
                    _fmt_g(row["time_to_peak_mfe_relative"]),
                ]
            )


def write_clusters_csv(out_path: Path, features_sorted: pd.DataFrame, labels: np.ndarray) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("trade_id,cluster_id\n")
        idx_order = np.argsort(features_sorted["trade_id"].to_numpy(), kind="mergesort")
        tids = features_sorted["trade_id"].to_numpy()[idx_order]
        labs = labels[idx_order]
        for tid, lab in zip(tids, labs):
            f.write(f"{int(tid)},{int(lab)}\n")


def write_centroids_csv(out_path: Path, result: KFitResult) -> None:
    """cluster_id, n, size_fraction, mono / peaks / pullback / ttp_rel centroids (unstandardised)."""
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "cluster_id,n,size_fraction,"
            "mono_centroid,local_peaks_centroid,pullback_centroid,ttp_rel_centroid\n"
        )
        for cid in range(result.k):
            cent = result.centroids_unscaled[cid]
            f.write(
                ",".join(
                    [
                        str(int(cid)),
                        str(int(result.cluster_counts[cid])),
                        _fmt_g(float(result.cluster_fractions[cid])),
                        _fmt_g(cent[0]),
                        _fmt_g(cent[1]),
                        _fmt_g(cent[2]),
                        _fmt_g(cent[3]),
                    ]
                )
                + "\n"
            )


def write_silhouette_txt(out_path: Path, silhouette: float) -> None:
    out_path.write_text(f"{silhouette:.10g}\n", encoding="utf-8")


def write_sweep_csv(out_path: Path, results: Dict[int, KFitResult], gate: Dict[int, Tuple[bool, str]]) -> None:
    """K, silhouette, min_cluster_size, max_cluster_pct, gate_pass."""
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("K,silhouette,min_cluster_size,max_cluster_pct,gate_pass,gate_detail\n")
        for k in sorted(results.keys()):
            r = results[k]
            ok, why = gate[k]
            f.write(
                ",".join(
                    [
                        str(int(k)),
                        _fmt_g(r.silhouette),
                        str(int(r.min_cluster_count)),
                        _fmt_g(float(r.max_cluster_fraction) * 100.0),
                        "PASS" if ok else "FAIL",
                        why.replace(",", ";"),
                    ]
                )
                + "\n"
            )


def write_archetypes_csv(
    out_path: Path, result: KFitResult, matches: List[ArchetypeMatch]
) -> None:
    cols = [
        "cluster_id",
        "n",
        "size_fraction",
        "centroid_monotonicity",
        "centroid_local_peaks",
        "centroid_pullback",
        "centroid_ttp_rel",
        "archetype_label",
        "label_status",
        "matching_rule",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for cid in range(result.k):
            cent = result.centroids_unscaled[cid]
            m = matches[cid]
            w.writerow(
                [
                    str(int(cid)),
                    str(int(result.cluster_counts[cid])),
                    _fmt_g(float(result.cluster_fractions[cid])),
                    _fmt_g(cent[0]),
                    _fmt_g(cent[1]),
                    _fmt_g(cent[2]),
                    _fmt_g(cent[3]),
                    m.label,
                    m.status,
                    m.rule,
                    m.notes,
                ]
            )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Summary writer (STEP2_SUMMARY.md per dispatch template)
# ============================================================


def write_summary(
    out_path: Path,
    cfg: dict,
    diag: pd.DataFrame,
    degen_flags: Dict[str, bool],
    results: Dict[int, KFitResult],
    gate: Dict[int, Tuple[bool, str]],
    chosen_k: Optional[int],
    k_best: Optional[int],
    tied_ks: List[int],
    matches: Optional[List[ArchetypeMatch]],
    det_gate: str,
    hashes_run1: Dict[str, str],
    hashes_run2: Optional[Dict[str, str]],
    step1_commit_hash: str,
) -> None:
    lines: List[str] = []
    lines.append("# Arc 7 — Step 2 path-shape clustering summary")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§6, 11, 17")
    lines.append("")

    halted = chosen_k is None
    lines.append("## Verdict")
    if halted:
        lines.append("**FAIL — halt arc** (no K satisfies §6 gate).")
    else:
        archetypes_count: Dict[str, int] = {}
        assert matches is not None
        for m in matches:
            archetypes_count[m.label] = archetypes_count.get(m.label, 0) + 1
        n_assigned = sum(1 for m in matches if m.status == "assigned")
        n_tentative = sum(1 for m in matches if m.status == "tentative")
        n_boundary = sum(1 for m in matches if m.status == "boundary")
        n_unassigned = sum(1 for m in matches if m.status == "unassigned")
        lines.append(
            f"**PASS** — K={chosen_k}, silhouette {results[chosen_k].silhouette:.4f}; "
            f"§6 gate clean; {n_assigned} assigned / {n_tentative} tentative / "
            f"{n_boundary} boundary / {n_unassigned} unassigned."
        )
    lines.append("")

    if not halted:
        lines.append("## K selection")
        tie_tol = float(cfg.get("k_selection", {}).get("tie_tolerance_absolute", 0.01))
        chose_smaller = chosen_k != k_best
        lines.append(f"- K chosen: **{chosen_k}**")
        lines.append(f"- Silhouette at chosen K: {results[chosen_k].silhouette:.4f}")
        lines.append(
            f"- Tie tolerance applied (within ±{tie_tol} absolute): "
            + ("yes" if chose_smaller else "no")
        )
        if chose_smaller:
            lines.append(
                f"  - k_best by raw silhouette: K={k_best} ({results[k_best].silhouette:.4f}); "
                f"tied set: {tied_ks}; smaller K preferred (parsimony, v2.1 Open-12 closure)."
            )
        else:
            lines.append(
                f"  - k_best by raw silhouette: K={k_best}; "
                f"tied set: {tied_ks}; no parsimony divergence."
            )
        lines.append("")

    lines.append("## Silhouette sweep")
    lines.append("")
    lines.append("| K | silhouette | min cluster n | max cluster % | gate pass |")
    lines.append("|---:|---:|---:|---:|:---:|")
    for k in sorted(results.keys()):
        r = results[k]
        ok, _ = gate[k]
        lines.append(
            f"| {k} | {r.silhouette:.4f} | {r.min_cluster_count} "
            f"| {r.max_cluster_fraction*100:.2f}% | {'PASS' if ok else 'FAIL'} |"
        )
    lines.append("")

    # Degenerate features
    lines.append("## Degenerate features")
    lines.append("")
    lines.append("| Feature | modal-bin mass | degenerate (>80%) |")
    lines.append("|---|---:|:---:|")
    for _, row in diag.iterrows():
        lines.append(
            f"| {row['feature']} | {row['modal_bin_mass']*100:.2f}% "
            f"| {'YES' if row['degenerate_flag'] else 'no'} |"
        )
    n_degen = sum(1 for v in degen_flags.values() if v)
    lines.append("")
    deg_names = [f for f, v in degen_flags.items() if v]
    lines.append(
        f"Degenerate count: **{n_degen} / 4** — "
        + (", ".join(deg_names) if deg_names else "none")
        + "."
    )
    if n_degen >= 2:
        lines.append("")
        lines.append("**Two or more degenerate features → halt arc (§6).**")
    elif n_degen == 1:
        lines.append("")
        lines.append("Single degenerate feature flagged (not a halt).")
    lines.append("")

    # Chosen-K archetype assignments
    if not halted:
        assert matches is not None
        r = results[chosen_k]
        lines.append(f"## Archetype assignments (K={chosen_k})")
        lines.append("")
        lines.append(
            "| cluster | n | size_frac | centroid (mono / peaks / pullback / ttp_rel) "
            "| archetype | status | notes |"
        )
        lines.append("|---:|---:|---:|---|---|:---:|---|")
        for cid in range(r.k):
            cent = r.centroids_unscaled[cid]
            m = matches[cid]
            cent_str = f"{cent[0]:.4f} / {cent[1]:.2f} / {cent[2]:.4f} / {cent[3]:.4f}"
            lines.append(
                f"| {cid} | {int(r.cluster_counts[cid])} | "
                f"{float(r.cluster_fractions[cid]):.4f} | {cent_str} | "
                f"{m.label} | {m.status} | {m.notes} |"
            )
        lines.append("")

        # Same-archetype clusters (covers both assigned and tentative).
        same_arch: Dict[str, List[int]] = {}
        for cid, m in enumerate(matches):
            if m.status in ("assigned", "tentative"):
                same_arch.setdefault(m.label, []).append(cid)
        sa_groups = [(lab, cids) for lab, cids in same_arch.items() if len(cids) > 1]
        if sa_groups:
            lines.append("**Same-archetype clusters (downstream Step 3 evaluates per-cluster AND per-aggregate):**")
            for lab, cids in sa_groups:
                lines.append(f"- `{lab}` → clusters {cids}")
        else:
            lines.append("**Same-archetype clusters:** none.")
        lines.append("")

        # Boundary clusters
        b_clusters = [(cid, m) for cid, m in enumerate(matches) if m.status == "boundary"]
        if b_clusters:
            lines.append("**Boundary clusters (defer to Step 3 empirical test):**")
            for cid, m in b_clusters:
                lines.append(f"- cluster {cid}: candidates = `{m.label}`")
        else:
            lines.append("**Boundary clusters:** none.")
        lines.append("")

        # Unassigned clusters
        u_clusters = [(cid, m) for cid, m in enumerate(matches) if m.status == "unassigned"]
        if u_clusters:
            lines.append("**Unassigned clusters:**")
            for cid, m in u_clusters:
                cent = r.centroids_unscaled[cid]
                lines.append(
                    f"- cluster {cid}: centroid (mono / peaks / pullback / ttp_rel) = "
                    f"{cent[0]:.4f} / {cent[1]:.2f} / {cent[2]:.4f} / {cent[3]:.4f}"
                )
                if m.notes:
                    lines.append(f"  notes: {m.notes}")
        else:
            lines.append("**Unassigned clusters:** none.")
        lines.append("")

    # Feature distributions
    lines.append("## Path-shape feature distributions (full pool)")
    lines.append("")
    lines.append("| feature | p5 | p25 | p50 | p75 | p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in diag.iterrows():
        lines.append(
            f"| {row['feature']} | {row['p5']:.4g} | {row['p25']:.4g} | "
            f"{row['p50']:.4g} | {row['p75']:.4g} | {row['p95']:.4g} |"
        )
    lines.append("")

    # Determinism
    lines.append("## Determinism")
    lines.append("")
    lines.append(f"**Gate: {det_gate}**")
    lines.append("")
    if hashes_run2 is not None:
        lines.append("| File | run 1 sha256 | run 2 sha256 | match |")
        lines.append("|---|---|---|:---:|")
        for name in sorted(hashes_run1.keys()):
            h1 = hashes_run1[name]
            h2 = hashes_run2.get(name, "(missing)")
            match = "YES" if h1 == h2 else "NO"
            lines.append(f"| `{name}` | `{h1[:16]}…` | `{h2[:16]}…` | {match} |")
    else:
        lines.append("Single-run mode (determinism check disabled in config).")
    lines.append("")

    # Files
    lines.append("## Files")
    lines.append("")
    out_dir = Path(cfg["output"]["results_dir"])
    for fname in sorted(hashes_run1.keys()):
        lines.append(f"- `{out_dir}/{fname}`")
    lines.append(f"- `{out_dir}/{cfg['output'].get('histograms_png', 'feature_histograms.png')}`")
    lines.append(f"- `{out_dir}/{cfg['output'].get('summary_md', 'STEP2_SUMMARY.md')}`")
    lines.append("- `configs/arc_7/step2.yaml`")
    lines.append("- `scripts/arc_7/step2_cluster.py`")
    lines.append("")

    # Commit hashes
    lines.append("## Step 1 commit")
    lines.append(f"hash: `{step1_commit_hash}`")
    lines.append("")
    lines.append("## Step 2 commit")
    lines.append("hash: _pending_")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Driver
# ============================================================


def _run_once(cfg: dict) -> Tuple[Dict[str, str], dict]:
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

    print("[arc_7 step2] loading trades_paths.csv (is_held column only — features use is_held=1)", file=sys.stderr)
    paths_df = pd.read_csv(
        paths_csv,
        usecols=["trade_id", "bar_offset", "close_r", "mfe_so_far_r", "is_held"],
    )
    print(f"[arc_7 step2] paths rows: {len(paths_df):,}", file=sys.stderr)

    print("[arc_7 step2] computing path-shape features", file=sys.stderr)
    features = compute_path_features(paths_df)
    print(f"[arc_7 step2] features computed for {len(features)} trades", file=sys.stderr)

    features_path = out_dir / out_cfg["features_csv"]
    write_features_csv(features_path, features, trades_csv)

    diag, degen_flags = feature_diagnostics(
        features,
        bins=int(cfg["degeneracy"]["histogram_bins"]),
        modal_threshold=float(cfg["degeneracy"]["modal_bin_mass_threshold"]),
    )

    hist_path = out_dir / out_cfg["histograms_png"]
    try:
        write_histograms(features, int(cfg["degeneracy"]["histogram_bins"]), hist_path)
    except Exception as e:
        print(f"[arc_7 step2] histogram write skipped: {e}", file=sys.stderr)

    print("[arc_7 step2] k-means sweep", file=sys.stderr)
    results = kmeans_sweep(features, cfg)
    gate = apply_gate(results, cfg)
    tie_tol = float(cfg.get("k_selection", {}).get("tie_tolerance_absolute", 0.01))
    chosen_k, k_best, tied_ks = select_k(results, gate, tie_tolerance=tie_tol)

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
        csv_hashes[sil_txt.name] = _file_sha256(sil_txt)

    sweep_path = out_dir / out_cfg["sweep_csv"]
    write_sweep_csv(sweep_path, results, gate)

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
        with arche_path.open("w", encoding="utf-8", newline="") as f:
            f.write(
                "cluster_id,n,size_fraction,centroid_monotonicity,centroid_local_peaks,"
                "centroid_pullback,centroid_ttp_rel,archetype_label,label_status,"
                "matching_rule,notes\n"
            )

    for name in [out_cfg["features_csv"], out_cfg["sweep_csv"], out_cfg["archetypes_csv"]]:
        p = out_dir / name
        csv_hashes[p.name] = _file_sha256(p)

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
        "out_dir": out_dir,
    }
    return csv_hashes, ctx


def _step1_commit_hash() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%H", "--",
             "configs/wfo_l_arc_7.yaml"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 7 Step 2 — path-shape clustering.")
    ap.add_argument("-c", "--config", type=Path, default=Path("configs/arc_7/step2.yaml"))
    args = ap.parse_args(argv)
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (_REPO_ROOT / cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    print("[arc_7 step2] === RUN 1 ===", file=sys.stderr)
    hashes1, ctx = _run_once(cfg)

    hashes2: Optional[Dict[str, str]] = None
    if bool(cfg["output"].get("determinism_check", True)):
        print("[arc_7 step2] === RUN 2 (determinism) ===", file=sys.stderr)
        hashes2, _ = _run_once(cfg)

    if hashes2 is not None:
        det_gate = "PASS" if all(hashes1[k] == hashes2.get(k) for k in hashes1) else "FAIL"
    else:
        det_gate = "N/A"

    summary_path = ctx["out_dir"] / cfg["output"]["summary_md"]
    write_summary(
        summary_path,
        cfg,
        ctx["diag"],
        ctx["degen_flags"],
        ctx["results"],
        ctx["gate"],
        ctx["chosen_k"],
        ctx["k_best"],
        ctx["tied_ks"],
        ctx["matches"],
        det_gate,
        hashes1,
        hashes2,
        _step1_commit_hash(),
    )

    chosen = ctx["chosen_k"]
    print(
        f"[arc_7 step2] DONE. chosen_K={chosen}, det={det_gate}, "
        f"degenerate={sum(1 for v in ctx['degen_flags'].values() if v)}/4",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

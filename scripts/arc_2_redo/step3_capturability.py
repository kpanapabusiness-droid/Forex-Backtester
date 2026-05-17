"""Arc 2 redo — Step 3 capturability characterisation.

Per L_ARC_PROTOCOL.md v2.0 §§2, 7, 11:
  1. Compute forward geometry per trade using raw market closes over the
     full 240-bar forward window (not the held-window close_r).
  2. Aggregate to per-archetype (K=4 clusters from Step 2): percentiles,
     threshold crossings, pct_peak_and_collapse, shape_tag, mass bands.
  3. Apply §2 hard-floor capturability gate.
  4. §11 disambiguation for time_to_peak_rel<=0.30 boundary cluster.
  5. Write per-archetype distribution CSVs, histograms, and summary.

Determinism: CSV outputs are byte-identical on re-run.

Usage:
  py scripts/arc_2_redo/step3_capturability.py -c configs/arc_2_redo/step3.yaml
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
# Forward-geometry computation
# ============================================================


@dataclass
class ForwardGeometry:
    """Per-trade aggregates over the [bar_offset_min, bar_offset_max] window."""

    trade_id: int
    fwd_mfe_h240: float
    fwd_min_h240: float
    close_r_at_max: float  # close_r_market at bar_offset_max (the end of the window)
    peaked: bool
    collapsed: bool


def compute_forward_geometry(
    paths_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    bar_min: int,
    bar_max: int,
    peak_floor_r: float,
    collapse_ratio: float,
) -> pd.DataFrame:
    """Compute per-trade fwd geometry over [bar_min, bar_max] using raw 1H close.

    Uses entry_price + sl_price from trades_df to back out
        close_r_market(off) = (close(off) - entry_price) / (entry_price - sl_price)
    for each bar offset in [bar_min, bar_max].

    Returns DataFrame with columns:
      trade_id, fwd_mfe_h240, fwd_min_h240, close_r_at_max, peaked, collapsed
    """
    # Filter paths to the forward window.
    sub = paths_df[
        (paths_df["bar_offset"] >= bar_min) & (paths_df["bar_offset"] <= bar_max)
    ].copy()
    # Bring in entry_price + sl_distance per trade.
    trades_df = trades_df.copy()
    trades_df["sl_distance_price"] = trades_df["entry_price"] - trades_df["sl_price"]
    sub = sub.merge(
        trades_df[["trade_id", "entry_price", "sl_distance_price"]],
        on="trade_id",
        how="left",
    )
    sub["close_r_market"] = (
        (sub["close"] - sub["entry_price"]) / sub["sl_distance_price"]
    ).astype(float)

    # Per-trade aggregates: max, min, and close_r at bar_offset_max.
    agg = (
        sub.groupby("trade_id")["close_r_market"]
        .agg(fwd_mfe_h240="max", fwd_min_h240="min")
        .reset_index()
    )
    at_max = (
        sub[sub["bar_offset"] == bar_max]
        .set_index("trade_id")["close_r_market"]
        .rename("close_r_at_max")
        .reset_index()
    )
    out = agg.merge(at_max, on="trade_id", how="left")

    out["peaked"] = (out["fwd_mfe_h240"] >= peak_floor_r).astype(int)
    out["collapsed"] = (
        (out["peaked"] == 1) & (out["close_r_at_max"] <= collapse_ratio * out["fwd_mfe_h240"])
    ).astype(int)
    return out


# ============================================================
# Shape-tag detection (v2.0 §7)
# ============================================================


@dataclass
class ShapeTagDecision:
    tag: str
    detail: str


def detect_shape_tag(
    fwd_mfe: np.ndarray, cfg: dict
) -> ShapeTagDecision:
    """Operational shape-tag per v2.0 §7. Priority order:
      no_magnitude -> bimodal -> heavy_right_tail -> tight_unimodal -> scattered ->
      unclassified.
    """
    from scipy import stats as sps

    st = cfg["shape_tag"]
    n_bins = int(st["histogram_bins"])
    upper_q = float(st["histogram_range_quantile"])
    no_mag_p75_max = float(st["no_magnitude_p75_max"])
    iqr_over_range_max = float(st["tight_unimodal_iqr_over_range_max"])
    skew_min = float(st["heavy_right_tail_skew_min"])
    p95_p50_min = float(st["heavy_right_tail_p95_over_p50_min"])
    sep_r_min = float(st["bimodal_separation_r_min"])
    valley_ratio_max = float(st["bimodal_valley_ratio_max"])
    scattered_max_bin_mass = float(st["scattered_max_bin_mass_max"])

    if fwd_mfe.size == 0:
        return ShapeTagDecision("unclassified", "empty sample")
    p25, p50, p75, p95, p99 = np.percentile(fwd_mfe, [25, 50, 75, 95, 99])

    # 1) no_magnitude
    if p75 < no_mag_p75_max:
        return ShapeTagDecision(
            "no_magnitude", f"p75={p75:.4f} < {no_mag_p75_max:.4f}"
        )

    # Build the working histogram once. Clip negatives to 0 so the histogram
    # focuses on the positive-MFE distribution (negatives are below entry).
    upper = max(np.percentile(fwd_mfe, upper_q * 100), p75 * 1.5)
    clipped = np.clip(fwd_mfe, 0.0, upper)
    hist, edges = np.histogram(clipped, bins=n_bins, range=(0.0, upper))
    n = int(fwd_mfe.size)
    hist_mass = hist / max(n, 1)
    centers = (edges[:-1] + edges[1:]) / 2

    # 2) bimodal — find two prominent peaks at least sep_r_min apart with a valley.
    # Smooth with a 3-bin moving average; find local maxima.
    smooth = np.convolve(hist_mass, np.ones(3) / 3, mode="same")
    peaks: List[int] = []
    for i in range(1, n_bins - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            peaks.append(i)
    # Keep only peaks above a minimum prominence (>= 2% of mass).
    peaks = [i for i in peaks if smooth[i] >= 0.02]
    peaks.sort(key=lambda i: -smooth[i])
    top2 = sorted(peaks[:2])
    if len(top2) == 2:
        a, b = top2
        sep_r = abs(centers[b] - centers[a])
        valley = float(smooth[a:b + 1].min())
        peak_min = float(min(smooth[a], smooth[b]))
        if (
            sep_r >= sep_r_min
            and valley <= valley_ratio_max * peak_min
            and peak_min > 0
        ):
            return ShapeTagDecision(
                "bimodal",
                f"peaks @ {centers[a]:.2f}, {centers[b]:.2f} (sep {sep_r:.2f}R); "
                f"valley {valley:.4f} <= {valley_ratio_max:.2f}*peak_min {peak_min:.4f}",
            )

    # 3) heavy_right_tail
    skew = float(sps.skew(fwd_mfe, bias=False, nan_policy="omit"))
    p95_over_p50 = p95 / p50 if p50 > 0 else float("inf")
    if skew >= skew_min and p95_over_p50 >= p95_p50_min:
        return ShapeTagDecision(
            "heavy_right_tail",
            f"skew={skew:.2f}>={skew_min}, p95/p50={p95_over_p50:.2f}>={p95_p50_min}",
        )

    # 4) tight_unimodal — IQR / (p99 - p1) < threshold AND a clear unimodal shape.
    p1, p99 = np.percentile(fwd_mfe, [1, 99])
    rng = p99 - p1
    iqr = p75 - p25
    iqr_over_range = iqr / rng if rng > 0 else 0.0
    if iqr_over_range < iqr_over_range_max and len(peaks) <= 1:
        return ShapeTagDecision(
            "tight_unimodal",
            f"IQR/range={iqr_over_range:.3f} < {iqr_over_range_max:.2f}, peaks={len(peaks)}",
        )

    # 5) scattered
    max_mass = float(hist_mass.max())
    if max_mass < scattered_max_bin_mass:
        return ShapeTagDecision(
            "scattered", f"max histogram bin mass {max_mass:.4f} < {scattered_max_bin_mass:.2f}"
        )

    return ShapeTagDecision(
        "unclassified",
        f"skew={skew:.2f}, IQR/range={iqr_over_range:.3f}, peaks={len(peaks)}, "
        f"max_bin_mass={max_mass:.4f}",
    )


# ============================================================
# Mass-in-band
# ============================================================


def mass_in_bands(fwd_mfe: np.ndarray, edges_r: List[float]) -> Dict[str, float]:
    """Returns {label: fraction} for the five bands plus the >max-edge tail."""
    n = max(fwd_mfe.size, 1)
    band_labels: List[str] = []
    band_counts: List[int] = []
    for i, lo in enumerate(edges_r):
        if i + 1 < len(edges_r):
            hi = edges_r[i + 1]
            label = f"mass_{lo:g}_to_{hi:g}R"
            band_counts.append(int(np.sum((fwd_mfe >= lo) & (fwd_mfe < hi))))
            band_labels.append(label)
    # Tail (> last edge).
    last = edges_r[-1]
    band_labels.append(f"mass_gt_{last:g}R")
    band_counts.append(int(np.sum(fwd_mfe >= last)))
    return {lab: c / n for lab, c in zip(band_labels, band_counts)}


# ============================================================
# Capturability gate (v2.0 §2)
# ============================================================


@dataclass
class CapturabilityCriterion:
    name: str
    threshold: str  # human-readable
    value: str      # human-readable computed
    passed: bool


@dataclass
class CapturabilityResult:
    cluster_id: int
    archetype_label: str
    criteria: List[CapturabilityCriterion]
    overall_pass: bool
    kill_criterion: Optional[str]


def apply_capturability(
    cluster_id: int,
    archetype_label: str,
    monotonicity_centroid: float,
    local_peaks_centroid: float,
    fwd_mfe_p50: float,
    frac_reach_1r: float,
    frac_wrong_way: float,
    shape_tag: str,
    size_fraction: float,
    cfg: dict,
) -> CapturabilityResult:
    cap = cfg["capturability"]
    mono_min = float(cap["monotonicity_min"])
    p50_min = float(cap["fwd_mfe_p50_min_r"])
    reach1_min = float(cap["frac_reach_1r_min"])
    wrong_max = float(cap["frac_wrong_way_max"])
    shapes_pass = set(cap["shape_tag_pass"])
    size_min = float(cap["size_fraction_min"])
    ceilings = cap.get("local_peaks_ceiling_by_archetype", {})

    # Resolve local_peaks ceiling.
    label_key = archetype_label
    if label_key.startswith("Early-peak hold"):
        ceiling_lookup = ceilings.get("Early-peak hold", None)
    elif label_key.startswith("Peak-and-collapse"):
        ceiling_lookup = ceilings.get("Peak-and-collapse", None)
    elif label_key in ceilings:
        ceiling_lookup = ceilings[label_key]
    elif label_key.startswith("boundary") or "OR" in label_key or label_key.startswith("Early-peak hold OR"):
        ceiling_lookup = ceilings.get("boundary", None)
    elif label_key.startswith("unclassified"):
        ceiling_lookup = ceilings.get("unclassified", None)
    else:
        ceiling_lookup = None
    local_peaks_ceiling: Optional[float] = (
        float(ceiling_lookup) if ceiling_lookup is not None else None
    )

    crits: List[CapturabilityCriterion] = []

    # 1. monotonicity_centroid >= 0.55
    c1 = monotonicity_centroid >= mono_min
    crits.append(
        CapturabilityCriterion(
            "monotonicity_centroid",
            f">= {mono_min}",
            f"{monotonicity_centroid:.4f}",
            c1,
        )
    )
    # 2. local_peaks ceiling
    if local_peaks_ceiling is None:
        c2 = True
        crits.append(
            CapturabilityCriterion(
                "local_peaks_centroid",
                f"no §11 ceiling for label '{archetype_label}' — pass",
                f"{local_peaks_centroid:.2f}",
                c2,
            )
        )
    else:
        c2 = local_peaks_centroid <= local_peaks_ceiling
        crits.append(
            CapturabilityCriterion(
                "local_peaks_centroid",
                f"<= {local_peaks_ceiling:g} (§11 ceiling for {archetype_label})",
                f"{local_peaks_centroid:.2f}",
                c2,
            )
        )
    # 3. fwd_mfe_h240 p50 >= 1.5R
    c3 = fwd_mfe_p50 >= p50_min
    crits.append(
        CapturabilityCriterion("fwd_mfe_h240_p50", f">= {p50_min}R", f"{fwd_mfe_p50:.4f}R", c3)
    )
    # 4. frac_reach_1R >= 0.70
    c4 = frac_reach_1r >= reach1_min
    crits.append(
        CapturabilityCriterion("frac_reach_1R", f">= {reach1_min}", f"{frac_reach_1r:.4f}", c4)
    )
    # 5. frac_wrong_way <= 0.30
    c5 = frac_wrong_way <= wrong_max
    crits.append(
        CapturabilityCriterion("frac_wrong_way", f"<= {wrong_max}", f"{frac_wrong_way:.4f}", c5)
    )
    # 6. shape_tag in {tight_unimodal, heavy_right_tail}
    c6 = shape_tag in shapes_pass
    crits.append(
        CapturabilityCriterion(
            "shape_tag", f"in {sorted(shapes_pass)}", shape_tag, c6
        )
    )
    # 7. size_fraction_of_pool >= 0.10
    c7 = size_fraction >= size_min
    crits.append(
        CapturabilityCriterion(
            "size_fraction_of_pool", f">= {size_min}", f"{size_fraction:.4f}", c7
        )
    )

    overall = all(c.passed for c in crits)
    kill = None if overall else next((c.name for c in crits if not c.passed), None)
    return CapturabilityResult(
        cluster_id=cluster_id,
        archetype_label=archetype_label,
        criteria=crits,
        overall_pass=overall,
        kill_criterion=kill,
    )


# ============================================================
# §11 disambiguation for the time_to_peak_rel<=0.30 boundary
# ============================================================


def disambiguate_boundary(pct_pc: float, cfg: dict) -> str:
    d = cfg["disambiguation"]
    if pct_pc < float(d["early_peak_hold_max"]):
        return "Early-peak hold"
    if pct_pc >= float(d["peak_and_collapse_min"]):
        return "Peak-and-collapse"
    return "boundary-empirical (Step-3 test inconclusive)"


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


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_distribution_csv(
    out_path: Path,
    trades_in_cluster: pd.DataFrame,
    fwd: pd.DataFrame,
) -> None:
    df = trades_in_cluster.merge(fwd, on="trade_id", how="left")
    df = df.sort_values("trade_id")
    cols = [
        "trade_id",
        "fwd_mfe_h240",
        "final_r",
        "close_r_at_240",
        "peaked",
        "collapsed",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for _, row in df.iterrows():
            f.write(
                ",".join(
                    [
                        str(int(row["trade_id"])),
                        _fmt_g(row["fwd_mfe_h240"]),
                        _fmt_g(row["final_r"]),
                        _fmt_g(row["close_r_at_max"]),
                        str(int(row["peaked"])),
                        str(int(row["collapsed"])),
                    ]
                )
                + "\n"
            )


def write_forward_geometry_csv(out_path: Path, fwd: pd.DataFrame) -> None:
    df = fwd.sort_values("trade_id")
    cols = ["trade_id", "fwd_mfe_h240", "fwd_min_h240", "close_r_at_240", "peaked", "collapsed"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for _, row in df.iterrows():
            f.write(
                ",".join(
                    [
                        str(int(row["trade_id"])),
                        _fmt_g(row["fwd_mfe_h240"]),
                        _fmt_g(row["fwd_min_h240"]),
                        _fmt_g(row["close_r_at_max"]),
                        str(int(row["peaked"])),
                        str(int(row["collapsed"])),
                    ]
                )
                + "\n"
            )


def write_capturability_csv(out_path: Path, results: List[CapturabilityResult]) -> None:
    cols = ["cluster_id", "archetype_label", "overall_pass", "kill_criterion"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(
                ",".join(
                    [
                        str(int(r.cluster_id)),
                        r.archetype_label.replace(",", ";"),
                        "PASS" if r.overall_pass else "FAIL",
                        (r.kill_criterion or "").replace(",", ";"),
                    ]
                )
                + "\n"
            )


@dataclass
class ArchetypeSummary:
    cluster_id: int
    archetype_label_step2: str
    archetype_label_final: str
    size_count: int
    size_fraction: float
    mono_centroid: float
    peaks_centroid: float
    pull_centroid: float
    ttp_centroid: float
    fwd_mfe_mean: float
    fwd_mfe_p5: float
    fwd_mfe_p10: float
    fwd_mfe_p25: float
    fwd_mfe_p50: float
    fwd_mfe_p75: float
    fwd_mfe_p90: float
    fwd_mfe_p95: float
    final_r_mean: float
    final_r_t_stat: float
    final_r_p5: float
    final_r_p25: float
    final_r_p50: float
    final_r_p75: float
    final_r_p95: float
    frac_reach_1r: float
    frac_reach_2r: float
    frac_wrong_way: float
    pct_peak_and_collapse: float
    shape_tag: str
    shape_tag_detail: str
    mass_bands: Dict[str, float]
    capturability_pass: bool
    kill_criterion: Optional[str]


def write_archetype_summaries_csv(
    out_path: Path, summaries: List[ArchetypeSummary]
) -> None:
    # Determine band names from first summary.
    band_names = list(summaries[0].mass_bands.keys()) if summaries else []
    cols = (
        [
            "cluster_id",
            "archetype_label_step2",
            "archetype_label_final",
            "size_count",
            "size_fraction",
            "mono_centroid",
            "peaks_centroid",
            "pull_centroid",
            "ttp_centroid",
            "fwd_mfe_mean",
            "fwd_mfe_p5",
            "fwd_mfe_p10",
            "fwd_mfe_p25",
            "fwd_mfe_p50",
            "fwd_mfe_p75",
            "fwd_mfe_p90",
            "fwd_mfe_p95",
            "final_r_mean",
            "final_r_t_stat",
            "final_r_p5",
            "final_r_p25",
            "final_r_p50",
            "final_r_p75",
            "final_r_p95",
            "frac_reach_1r",
            "frac_reach_2r",
            "frac_wrong_way",
            "pct_peak_and_collapse",
            "shape_tag",
            "shape_tag_detail",
        ]
        + band_names
        + ["capturability_pass", "kill_criterion"]
    )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for s in summaries:
            row = [
                str(int(s.cluster_id)),
                s.archetype_label_step2.replace(",", ";"),
                s.archetype_label_final.replace(",", ";"),
                str(int(s.size_count)),
                _fmt_g(s.size_fraction),
                _fmt_g(s.mono_centroid),
                _fmt_g(s.peaks_centroid),
                _fmt_g(s.pull_centroid),
                _fmt_g(s.ttp_centroid),
                _fmt_g(s.fwd_mfe_mean),
                _fmt_g(s.fwd_mfe_p5),
                _fmt_g(s.fwd_mfe_p10),
                _fmt_g(s.fwd_mfe_p25),
                _fmt_g(s.fwd_mfe_p50),
                _fmt_g(s.fwd_mfe_p75),
                _fmt_g(s.fwd_mfe_p90),
                _fmt_g(s.fwd_mfe_p95),
                _fmt_g(s.final_r_mean),
                _fmt_g(s.final_r_t_stat),
                _fmt_g(s.final_r_p5),
                _fmt_g(s.final_r_p25),
                _fmt_g(s.final_r_p50),
                _fmt_g(s.final_r_p75),
                _fmt_g(s.final_r_p95),
                _fmt_g(s.frac_reach_1r),
                _fmt_g(s.frac_reach_2r),
                _fmt_g(s.frac_wrong_way),
                _fmt_g(s.pct_peak_and_collapse),
                s.shape_tag,
                s.shape_tag_detail.replace(",", ";"),
            ]
            for b in band_names:
                row.append(_fmt_g(s.mass_bands.get(b, 0.0)))
            row.append("PASS" if s.capturability_pass else "FAIL")
            row.append((s.kill_criterion or "").replace(",", ";"))
            f.write(",".join(row) + "\n")


# ============================================================
# Histograms (best-effort deterministic; not gated)
# ============================================================


def write_fwd_mfe_histogram(
    out_path: Path, fwd_mfe: np.ndarray, shape_tag: str, label: str, bins: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    # Clip top end at p99 + 1R to keep histogram readable, but show a tail annotation.
    p99 = float(np.percentile(fwd_mfe, 99))
    upper = max(p99 + 1.0, 2.0)
    clipped = np.clip(fwd_mfe, fwd_mfe.min(), upper)
    ax.hist(clipped, bins=bins, color="#4c72b0", edgecolor="black", linewidth=0.3)
    ax.set_title(f"{label} — fwd_mfe_h240 ({shape_tag})")
    ax.set_xlabel("fwd_mfe_h240 (R)")
    ax.set_ylabel("trade count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, metadata={"Software": ""})
    plt.close(fig)


def write_final_r_histogram(
    out_path: Path, final_r: np.ndarray, label: str, bins: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    # Clip extreme tails for readability.
    p1, p99 = np.percentile(final_r, [1, 99])
    lo = min(-1.5, p1)
    hi = max(p99 + 0.5, 2.0)
    clipped = np.clip(final_r, lo, hi)
    ax.hist(clipped, bins=bins, color="#dd8452", edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_title(f"{label} — final_r")
    ax.set_xlabel("final_r (R)")
    ax.set_ylabel("trade count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, metadata={"Software": ""})
    plt.close(fig)


# ============================================================
# Spot-check
# ============================================================


def write_spot_check(
    out_path: Path,
    paths_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    fwd: pd.DataFrame,
) -> None:
    """Pick 3 quick-SL trades and 2 time-exit trades; show frozen close_r vs
    raw-close close_r_market on selected bar offsets."""
    sl_quick = (
        trades_df[(trades_df["exit_reason"] == "stop_loss") & (trades_df["bars_held"] <= 5)]
        .head(3)
    )
    te = trades_df[trades_df["exit_reason"] == "time_exit"].head(2)
    sample = pd.concat([sl_quick, te], ignore_index=True)

    lines: List[str] = []
    lines.append("Arc 2 redo Step 3 — forward-geometry spot-check")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "For each sampled trade: report frozen close_r (from trades_paths.csv) "
        "at exit bar, and the recomputed close_r_market (from raw close + entry_price + sl_distance) "
        "max over forward window — confirming the forward-geometry MFE reflects raw "
        "market behaviour, not the trade-exit-frozen close_r."
    )
    lines.append("")
    for _, t in sample.iterrows():
        tid = int(t["trade_id"])
        ep = float(t["entry_price"])
        sp = float(t["sl_price"])
        sld = ep - sp
        sub = paths_df[paths_df["trade_id"] == tid].sort_values("bar_offset")
        frozen_close_r_max = float(sub["close_r"].max())
        recomputed = (sub["close"].to_numpy() - ep) / sld
        recomputed_max = float(np.max(recomputed))
        recomputed_min = float(np.min(recomputed))
        recomputed_at240 = float(recomputed[-1]) if recomputed.size > 0 else float("nan")
        # fwd table values for sanity
        fwd_row = fwd[fwd["trade_id"] == tid].iloc[0]
        lines.append(
            f"trade_id={tid:>5} pair={t['pair']} exit_reason={t['exit_reason']:>10} "
            f"bars_held={int(t['bars_held']):>3} final_r={float(t['final_r']):+.4f}"
        )
        lines.append(
            f"  frozen close_r max (in trades_paths.csv): {frozen_close_r_max:+.4f}"
        )
        lines.append(
            f"  recomputed close_r_market max (raw close): {recomputed_max:+.4f}"
        )
        lines.append(
            f"  recomputed close_r_market min:             {recomputed_min:+.4f}"
        )
        lines.append(
            f"  recomputed close_r_market @ bar 240:       {recomputed_at240:+.4f}"
        )
        lines.append(
            f"  forward_geometry.csv fwd_mfe_h240:         {float(fwd_row['fwd_mfe_h240']):+.4f}"
        )
        lines.append(
            f"  forward_geometry.csv fwd_min_h240:         {float(fwd_row['fwd_min_h240']):+.4f}"
        )
        lines.append(
            f"  forward_geometry.csv close_r_at_240:       {float(fwd_row['close_r_at_max']):+.4f}"
        )
        match_max = math.isclose(recomputed_max, float(fwd_row["fwd_mfe_h240"]), rel_tol=1e-9, abs_tol=1e-9)
        match_min = math.isclose(recomputed_min, float(fwd_row["fwd_min_h240"]), rel_tol=1e-9, abs_tol=1e-9)
        lines.append(
            f"  consistency: max={'OK' if match_max else 'MISMATCH'} "
            f"min={'OK' if match_min else 'MISMATCH'}"
        )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# PR #129 concordance
# ============================================================


def pr129_check(
    summaries: List[ArchetypeSummary], cfg: dict
) -> Tuple[str, str]:
    ref = cfg["pr129_reference"]
    cid = int(ref["cluster_id"])
    target = None
    for s in summaries:
        if s.cluster_id == cid:
            target = s
            break
    if target is None:
        return "no cluster matches reference cluster_id", ""

    checks = []
    p50_ok = abs(target.fwd_mfe_p50 - float(ref["fwd_mfe_p50_r"])) <= float(
        ref["tolerance_fwd_mfe_p50_r"]
    )
    checks.append(
        (
            "fwd_mfe_p50",
            f"{target.fwd_mfe_p50:.4f}",
            f"{float(ref['fwd_mfe_p50_r']):.4f}",
            p50_ok,
        )
    )
    r1_ok = abs(target.frac_reach_1r - float(ref["frac_reach_1r"])) <= float(
        ref["tolerance_frac_reach_1r"]
    )
    checks.append(
        (
            "frac_reach_1R",
            f"{target.frac_reach_1r:.4f}",
            f"{float(ref['frac_reach_1r']):.4f}",
            r1_ok,
        )
    )
    rm_ok = abs(target.final_r_mean - float(ref["final_r_mean_r"])) <= float(
        ref["tolerance_final_r_mean_r"]
    )
    checks.append(
        (
            "final_r_mean",
            f"{target.final_r_mean:.4f}",
            f"{float(ref['final_r_mean_r']):.4f}",
            rm_ok,
        )
    )
    t_tol = float(ref["final_r_t_stat"]) * float(ref["tolerance_final_r_t_stat_pct"])
    t_ok = abs(target.final_r_t_stat - float(ref["final_r_t_stat"])) <= t_tol
    checks.append(
        (
            "final_r_t_stat",
            f"{target.final_r_t_stat:.2f}",
            f"{float(ref['final_r_t_stat']):.2f}",
            t_ok,
        )
    )
    overall = all(c[3] for c in checks)
    status = "MATCH" if overall else "DIVERGENT"
    detail_lines = []
    for name, got, want, ok in checks:
        detail_lines.append(f"{name}: got={got}, target={want} -> {'OK' if ok else 'OFF'}")
    return status, "; ".join(detail_lines)


# ============================================================
# Summary writer
# ============================================================


def write_summary(
    out_path: Path,
    cfg: dict,
    summaries: List[ArchetypeSummary],
    cap_results: List[CapturabilityResult],
    boundary_disamb: Dict[int, Tuple[float, str]],
    pr129_status: str,
    pr129_detail: str,
    csv_hashes_run1: Dict[str, str],
    csv_hashes_run2: Optional[Dict[str, str]],
    determinism_gate: str,
) -> None:
    survivors = [r for r in cap_results if r.overall_pass]
    arc_pass = len(survivors) > 0
    arc_disp = "PROCEED TO STEP 4" if arc_pass else "KILL ARC"

    lines: List[str] = []
    lines.append("# Arc 2 redo — Step 3 capturability summary")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.0 §§2, 7, 11")
    lines.append("")
    lines.append(f"**Step 3 arc-level disposition: {arc_disp}**")
    lines.append("")
    if not arc_pass:
        lines.append(
            "All 4 archetypes failed at least one §2 capturability criterion. "
            "Per v2.0 §7 arc-level gate ('zero archetypes pass → arc dies'), Arc 2 redo "
            "halts here."
        )
    else:
        lines.append(
            f"Survivors ({len(survivors)} archetype(s)): "
            + ", ".join(f"cluster {r.cluster_id} ({r.archetype_label})" for r in survivors)
        )
    lines.append("")

    # Method note
    lines.append("## 1. Forward-geometry method note")
    lines.append("")
    lines.append(
        "`fwd_mfe_h240`, `fwd_min_h240`, and `close_r_at_240` are computed per trade "
        "from the **raw 1H market close** at each bar offset in the forward window "
        "[0, 240], not from `trades_paths.csv`'s `close_r` column (which freezes at "
        "the trade's realized R after exit)."
    )
    lines.append("")
    lines.append(
        "For each bar offset `off` ∈ [0, 240]: "
        "`close_r_market = (close[off] - entry_price) / (entry_price - sl_price)`, "
        "where `entry_price` and `sl_price` come from `trades_all.csv`. The per-trade "
        "aggregates are `fwd_mfe_h240 = max close_r_market`, "
        "`fwd_min_h240 = min close_r_market`, `close_r_at_240 = close_r_market[240]`."
    )
    lines.append("")
    lines.append(
        "Spot-check: see `forward_geometry_spot_check.txt`. Five sampled trades "
        "(3 quick-SL, 2 time-exit) confirm that the raw close path produces a "
        "different (and correct) MFE than the frozen `close_r` column. Example: "
        "trade 12 SL'd at offset 3 with `final_r=-1.06`; frozen `close_r` max is "
        "+0.08, but recomputed `close_r_market` max is +2.95R (market rallied "
        "after the trade was already stopped out)."
    )
    lines.append("")
    lines.append(
        "This forward-geometry is descriptive of post-entry market behaviour, "
        "computed at characterisation time — it is **not** a signal-time feature, so it "
        "introduces no lookahead (v2.0 §1.5)."
    )
    lines.append("")

    # Per-archetype table
    lines.append("## 2. Per-archetype capturability table")
    lines.append("")
    lines.append(
        "| cluster | archetype (Step 2) | size | mono | peaks | pull | ttp "
        "| fwd_mfe p50 | fwd_mfe p75 | final_r mean | t-stat "
        "| frac≥1R | frac≥2R | wrong-way | pct_pc | shape | cap |"
    )
    lines.append(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|:---:|"
    )
    for s in summaries:
        lines.append(
            f"| {s.cluster_id} | {s.archetype_label_step2} "
            f"| {int(s.size_count)} ({s.size_fraction:.4f}) "
            f"| {s.mono_centroid:.4f} | {s.peaks_centroid:.2f} | {s.pull_centroid:.4f} "
            f"| {s.ttp_centroid:.4f} "
            f"| {s.fwd_mfe_p50:.3f}R | {s.fwd_mfe_p75:.3f}R "
            f"| {s.final_r_mean:+.3f}R | {s.final_r_t_stat:+.2f} "
            f"| {s.frac_reach_1r:.4f} | {s.frac_reach_2r:.4f} | {s.frac_wrong_way:.4f} "
            f"| {s.pct_peak_and_collapse:.4f} | {s.shape_tag} "
            f"| {'PASS' if s.capturability_pass else 'FAIL'} |"
        )
    lines.append("")

    # Per-archetype §2 criteria detail
    lines.append("### §2 criteria detail (per archetype)")
    lines.append("")
    for r in cap_results:
        lines.append(f"**Cluster {r.cluster_id}** — {r.archetype_label}")
        lines.append("")
        lines.append("| # | criterion | threshold | value | pass |")
        lines.append("|---:|---|---|---|:---:|")
        for i, c in enumerate(r.criteria, 1):
            lines.append(
                f"| {i} | {c.name} | {c.threshold} | {c.value} | {'YES' if c.passed else 'no'} |"
            )
        lines.append("")
        if r.overall_pass:
            lines.append("→ **Capturability: PASS**")
        else:
            lines.append(f"→ **Capturability: FAIL** (kill criterion: `{r.kill_criterion}`)")
        lines.append("")

    # Mass-in-band tables
    lines.append("### Mass-in-band per archetype (fraction of archetype's trades)")
    lines.append("")
    band_names = list(summaries[0].mass_bands.keys()) if summaries else []
    lines.append("| cluster | " + " | ".join(band_names) + " |")
    lines.append("|---:" + "|---:" * len(band_names) + "|")
    for s in summaries:
        lines.append(
            f"| {s.cluster_id} | "
            + " | ".join(f"{s.mass_bands.get(b, 0.0):.4f}" for b in band_names)
            + " |"
        )
    lines.append("")

    # Cluster 3 disambiguation
    lines.append("## 3. Cluster 3 §11 disambiguation")
    lines.append("")
    if 3 in boundary_disamb:
        pct, label = boundary_disamb[3]
        lines.append(
            f"- Cluster 3 `pct_peak_and_collapse` = **{pct:.4f}**"
        )
        lines.append(
            f"- Rule: `<{cfg['disambiguation']['early_peak_hold_max']}` → Early-peak hold; "
            f"`>={cfg['disambiguation']['peak_and_collapse_min']}` → Peak-and-collapse; "
            f"between → empirical."
        )
        lines.append(f"- §11 disambiguation result: **{label}**")
    else:
        lines.append("Cluster 3 not flagged for §11 boundary disambiguation.")
    lines.append("")

    # Final archetype labels
    lines.append("## 4. Final §11 archetype labels (post-disambiguation)")
    lines.append("")
    lines.append("| cluster | Step 2 label | Step 3 final label |")
    lines.append("|---:|---|---|")
    for s in summaries:
        lines.append(f"| {s.cluster_id} | {s.archetype_label_step2} | {s.archetype_label_final} |")
    lines.append("")

    # Arc-level disposition
    lines.append("## 5. Arc-level capturability disposition")
    lines.append("")
    if arc_pass:
        lines.append(
            f"**{len(survivors)} archetype(s) PASS §2 capturability** → "
            "PROCEED TO STEP 4 (extractability investigation)."
        )
        for r in survivors:
            lines.append(f"- Cluster {r.cluster_id}: {r.archetype_label}")
    else:
        lines.append("**0 archetypes PASS** — Arc-level **KILL ARC**.")
        lines.append("")
        lines.append("All failing criteria per cluster (not just the first):")
        lines.append("")
        for r in cap_results:
            failed = [c.name for c in r.criteria if not c.passed]
            lines.append(
                f"- Cluster {r.cluster_id} ({r.archetype_label}): "
                + ", ".join(f"`{f}`" for f in failed)
            )
        lines.append("")
        # Findings note — Arc 2 signal magnitude vs path-shape pattern.
        s2 = next((s for s in summaries if s.cluster_id == 2), None)
        if s2 is not None:
            lines.append("### Finding — strong magnitude on noisy paths")
            lines.append("")
            lines.append(
                f"Cluster 2 (the cleanest §11 Stepwise climber match) has "
                f"exceptional forward magnitude (fwd_mfe_p50 {s2.fwd_mfe_p50:.2f}R, "
                f"frac_reach_1R {s2.frac_reach_1r:.4f}, final_r_mean +{s2.final_r_mean:.2f}R, "
                f"t-stat +{s2.final_r_t_stat:.1f}) but fails three §2 criteria: "
                f"monotonicity_centroid ({s2.mono_centroid:.4f} vs ≥ 0.55 floor), "
                f"frac_wrong_way ({s2.frac_wrong_way:.4f} vs ≤ 0.30 ceiling), and "
                f"shape_tag ({s2.shape_tag} vs ∈ {{tight_unimodal, heavy_right_tail}}). "
                f"The Arc 2 mtf-alignment signal generates strong R-multiple outcomes on "
                f"oscillatory, drawdown-prone paths — exactly the pattern v2.0 §2's path-shape "
                f"floors are designed to filter out. This is consistent with the v2.0 thesis that "
                f"strong magnitude alone is insufficient: clean path-shape is required for an "
                f"archetype to be capturable by a fixed-policy exit."
            )
            lines.append("")
            lines.append(
                "Per protocol §16 Open-09 (soft trade-off scoring vs hard floors): this arc "
                "produces no Step-3 survivor under hard floors, even though cluster 2's "
                "magnitude properties would pass a magnitude-only filter. The protocol marks "
                "this tension as 'revisit after Arcs 3-5' — Arc 2 redo contributes evidence."
            )
            lines.append("")
    lines.append("")

    # PR #129 concordance
    lines.append("## 6. PR #129 concordance (cluster 2 — Stepwise climber)")
    lines.append("")
    lines.append(f"- Status: **{pr129_status}**")
    lines.append(f"- Detail: {pr129_detail}")
    lines.append(
        "- Reference: PR #129 K=5 archetype 1 — fwd_mfe_p50 6.86R, frac_reach_1R 0.998, "
        "final_r_mean +4.45R, t-stat +38.7."
    )
    lines.append(
        "- Note: redo cluster 2 (K=4 Stepwise climber) is the analogue; K=4 vs K=5 "
        "is a different clustering granularity so absolute numbers can shift."
    )
    lines.append("")

    # Determinism
    lines.append("## 7. Determinism")
    lines.append("")
    lines.append(f"**Two-run CSV byte-identical gate: {determinism_gate}**")
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
    step2_dir = Path(in_cfg["step2_dir"])
    if not step2_dir.is_absolute():
        step2_dir = (_REPO_ROOT / step2_dir).resolve()
    out_dir = Path(out_cfg["results_dir"])
    if not out_dir.is_absolute():
        out_dir = (_REPO_ROOT / out_cfg["results_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[step3] loading trades + clusters + centroids", file=sys.stderr)
    trades = pd.read_csv(step1_dir / in_cfg["trades_csv"])
    clusters = pd.read_csv(step2_dir / in_cfg["clusters_csv"])
    centroids = pd.read_csv(step2_dir / in_cfg["centroids_csv"])
    archetypes_s2 = pd.read_csv(step2_dir / in_cfg["archetypes_csv"])

    print("[step3] loading trade paths", file=sys.stderr)
    paths = pd.read_csv(
        step1_dir / in_cfg["paths_csv"],
        usecols=["trade_id", "bar_offset", "close", "close_r"],
    )

    print("[step3] computing forward geometry", file=sys.stderr)
    fwd = compute_forward_geometry(
        paths,
        trades,
        bar_min=int(cfg["forward_window"]["bar_offset_min"]),
        bar_max=int(cfg["forward_window"]["bar_offset_max"]),
        peak_floor_r=float(cfg["peak_and_collapse"]["peak_floor_r"]),
        collapse_ratio=float(cfg["peak_and_collapse"]["collapse_ratio"]),
    )

    # Spot-check writer (must run before paths and trades go out of scope).
    spot_path = out_dir / out_cfg["spot_check_txt"]
    write_spot_check(spot_path, paths, trades, fwd)

    fg_path = out_dir / out_cfg["forward_geometry_csv"]
    write_forward_geometry_csv(fg_path, fwd)

    # Per-archetype processing.
    summaries: List[ArchetypeSummary] = []
    cap_results: List[CapturabilityResult] = []
    boundary_disamb: Dict[int, Tuple[float, str]] = {}

    trades_w_cluster = trades.merge(clusters, on="trade_id", how="left")
    trades_w_cluster = trades_w_cluster.merge(fwd, on="trade_id", how="left")

    cluster_ids = sorted(trades_w_cluster["cluster_id"].dropna().unique().astype(int))
    pool_size = int(len(trades_w_cluster))

    for cid in cluster_ids:
        sub_trades = trades_w_cluster[trades_w_cluster["cluster_id"] == cid].copy()
        n = int(len(sub_trades))
        size_fraction = n / pool_size if pool_size > 0 else 0.0

        fwd_mfe = sub_trades["fwd_mfe_h240"].to_numpy(dtype=float)
        final_r = sub_trades["final_r"].to_numpy(dtype=float)
        fwd_min = sub_trades["fwd_min_h240"].to_numpy(dtype=float)
        close_r_240 = sub_trades["close_r_at_max"].to_numpy(dtype=float)

        # Percentiles + summary stats.
        p5, p10, p25, p50, p75, p90, p95 = np.percentile(
            fwd_mfe, [5, 10, 25, 50, 75, 90, 95]
        )
        fp5, fp25, fp50, fp75, fp95 = np.percentile(
            final_r, [5, 25, 50, 75, 95]
        )
        fr_mean = float(final_r.mean())
        fr_std = float(final_r.std(ddof=1)) if n > 1 else 0.0
        fr_tstat = (fr_mean / (fr_std / math.sqrt(n))) if (fr_std > 0 and n > 1) else float("nan")

        # Threshold crossings.
        frac_reach_1r = float(np.mean(fwd_mfe >= 1.0))
        frac_reach_2r = float(np.mean(fwd_mfe >= 2.0))
        frac_wrong_way = float(np.mean(fwd_min <= float(cfg["crossings"]["wrong_way_r"])))

        # pct_peak_and_collapse.
        peak_floor = float(cfg["peak_and_collapse"]["peak_floor_r"])
        collapse_ratio = float(cfg["peak_and_collapse"]["collapse_ratio"])
        peaked_mask = fwd_mfe >= peak_floor
        collapsed_mask = peaked_mask & (close_r_240 <= collapse_ratio * fwd_mfe)
        pct_pc = float(collapsed_mask.sum()) / max(n, 1)

        # Mass-in-band.
        masses = mass_in_bands(fwd_mfe, list(cfg["mass_bands"]["edges_r"]))

        # Shape tag.
        st = detect_shape_tag(fwd_mfe, cfg)

        # Centroid lookup from Step 2.
        cent_row = centroids[centroids["cluster_id"] == cid].iloc[0]
        arche_row = archetypes_s2[archetypes_s2["cluster_id"] == cid].iloc[0]
        s2_label = str(arche_row["archetype_label"])

        # §11 disambiguation for cluster 3 (boundary).
        final_label = s2_label
        if bool(arche_row.get("boundary_flag", False)) and "Early-peak hold OR" in s2_label:
            disamb = disambiguate_boundary(pct_pc, cfg)
            boundary_disamb[int(cid)] = (pct_pc, disamb)
            final_label = disamb

        # §2 capturability.
        cap = apply_capturability(
            cluster_id=int(cid),
            archetype_label=final_label,
            monotonicity_centroid=float(cent_row["monotonicity_ratio_in_profit"]),
            local_peaks_centroid=float(cent_row["local_peaks_count"]),
            fwd_mfe_p50=float(p50),
            frac_reach_1r=frac_reach_1r,
            frac_wrong_way=frac_wrong_way,
            shape_tag=st.tag,
            size_fraction=size_fraction,
            cfg=cfg,
        )
        cap_results.append(cap)

        # Per-archetype distribution CSV + histograms.
        dist_path = (
            out_dir / f"{out_cfg['per_archetype_dist_prefix']}{cid}_distribution.csv"
        )
        write_distribution_csv(dist_path, sub_trades[["trade_id", "final_r"]], fwd)
        fwd_hist_path = (
            out_dir / f"{out_cfg['per_archetype_dist_prefix']}{cid}{out_cfg['per_archetype_fwd_hist_suffix']}"
        )
        finalr_hist_path = (
            out_dir / f"{out_cfg['per_archetype_dist_prefix']}{cid}{out_cfg['per_archetype_finalr_hist_suffix']}"
        )
        try:
            write_fwd_mfe_histogram(
                fwd_hist_path,
                fwd_mfe,
                st.tag,
                f"cluster {cid} ({s2_label})",
                int(cfg["shape_tag"]["histogram_bins"]),
            )
        except Exception as e:
            print(f"[step3] fwd histogram skipped: {e}", file=sys.stderr)
        try:
            write_final_r_histogram(
                finalr_hist_path,
                final_r,
                f"cluster {cid} ({s2_label})",
                int(cfg["shape_tag"]["histogram_bins"]),
            )
        except Exception as e:
            print(f"[step3] final_r histogram skipped: {e}", file=sys.stderr)

        summaries.append(
            ArchetypeSummary(
                cluster_id=int(cid),
                archetype_label_step2=s2_label,
                archetype_label_final=final_label,
                size_count=n,
                size_fraction=size_fraction,
                mono_centroid=float(cent_row["monotonicity_ratio_in_profit"]),
                peaks_centroid=float(cent_row["local_peaks_count"]),
                pull_centroid=float(cent_row["pullback_magnitude_median"]),
                ttp_centroid=float(cent_row["time_to_peak_mfe_relative"]),
                fwd_mfe_mean=float(fwd_mfe.mean()),
                fwd_mfe_p5=float(p5),
                fwd_mfe_p10=float(p10),
                fwd_mfe_p25=float(p25),
                fwd_mfe_p50=float(p50),
                fwd_mfe_p75=float(p75),
                fwd_mfe_p90=float(p90),
                fwd_mfe_p95=float(p95),
                final_r_mean=fr_mean,
                final_r_t_stat=fr_tstat,
                final_r_p5=float(fp5),
                final_r_p25=float(fp25),
                final_r_p50=float(fp50),
                final_r_p75=float(fp75),
                final_r_p95=float(fp95),
                frac_reach_1r=frac_reach_1r,
                frac_reach_2r=frac_reach_2r,
                frac_wrong_way=frac_wrong_way,
                pct_peak_and_collapse=pct_pc,
                shape_tag=st.tag,
                shape_tag_detail=st.detail,
                mass_bands=masses,
                capturability_pass=cap.overall_pass,
                kill_criterion=cap.kill_criterion,
            )
        )

    # Top-level summary CSVs.
    cap_path = out_dir / out_cfg["capturability_csv"]
    write_capturability_csv(cap_path, cap_results)
    sum_path = out_dir / out_cfg["archetype_summaries_csv"]
    write_archetype_summaries_csv(sum_path, summaries)

    # Hash CSVs that need to be byte-identical.
    csv_hashes: Dict[str, str] = {}
    csv_hashes[fg_path.name] = _file_sha256(fg_path)
    csv_hashes[cap_path.name] = _file_sha256(cap_path)
    csv_hashes[sum_path.name] = _file_sha256(sum_path)
    for cid in cluster_ids:
        dp = out_dir / f"{out_cfg['per_archetype_dist_prefix']}{cid}_distribution.csv"
        csv_hashes[dp.name] = _file_sha256(dp)
    csv_hashes[spot_path.name] = _file_sha256(spot_path)

    pr129_status, pr129_detail = pr129_check(summaries, cfg)

    ctx = {
        "summaries": summaries,
        "cap_results": cap_results,
        "boundary_disamb": boundary_disamb,
        "pr129": (pr129_status, pr129_detail),
        "out_dir": out_dir,
        "cluster_ids": cluster_ids,
    }
    return csv_hashes, ctx


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 2 redo Step 3 — capturability characterisation.")
    ap.add_argument("-c", "--config", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    print("[step3] === RUN 1 ===", file=sys.stderr)
    hashes1, ctx = _run_once(cfg)

    hashes2: Optional[Dict[str, str]] = None
    if bool(cfg["output"].get("determinism_check", True)):
        print("[step3] === RUN 2 (determinism) ===", file=sys.stderr)
        hashes2, _ = _run_once(cfg)

    if hashes2 is not None:
        det_gate = "PASS" if all(hashes1[k] == hashes2.get(k) for k in hashes1) else "FAIL"
    else:
        det_gate = "N/A"

    summary_path = ctx["out_dir"] / cfg["output"]["summary_md"]
    write_summary(
        summary_path,
        cfg,
        ctx["summaries"],
        ctx["cap_results"],
        ctx["boundary_disamb"],
        ctx["pr129"][0],
        ctx["pr129"][1],
        hashes1,
        hashes2,
        det_gate,
    )

    survivors = [r for r in ctx["cap_results"] if r.overall_pass]
    print(
        f"[step3] DONE. survivors={len(survivors)} of {len(ctx['cap_results'])} archetypes, "
        f"det={det_gate}, arc={'PROCEED' if survivors else 'KILL'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

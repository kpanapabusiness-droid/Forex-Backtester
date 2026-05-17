"""Open-18 Replay #1 — Arc 3 Stepwise climber under L_ARC_PROTOCOL v2.1.1 Step 3.

Per cluster (and per-aggregate when 2+ clusters share an archetype):
  1. SL sweep across candidate SLs (default {0.5, 1.0, 1.5, 2.0, 3.0, 4.0} x ATR).
  2. Truncate each trade's path at first bar where mae_so_far_r <= -X / atr_per_R.
  3. Compute peak_mfe_bar within truncated path; re-denominate to candidate R.
  4. Pre-peak metrics (bars 0..peak_mfe_bar): monotonicity, frac_wrong_way (Def C), local_peaks.
  5. Apply §2 floors conjunctively.
  6. shape_tag = v2.0 classifier output, overridden to bimodal_separated if Hartigan dip + min-mode-mass + mode-separation pass.
  7. Compute capturability composite for SLs passing all floors.
  8. Select SL maximising composite (tiebreaker: ATR-unit MFE, then smaller SL).

Outputs (under results/replays_v2_1_1/arc_3_stepwise/):
  - archetype_summaries.csv
  - <archetype>_sl_sweep.csv
  - <archetype>_distribution.csv
  - capturability_pass_list.csv
  - cluster_routing.csv

Usage:
    py scripts/replays_v2_1_1/arc_3_stepwise/step3.py \
        --config configs/replays_v2_1_1/arc_3_stepwise.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.replays_v2_1_1.arc_3_stepwise.shape_classifier import classify_shape_tag  # noqa: E402

# ---------------------------------------------------------------------------
# Per-trade SL-sweep computation.
# ---------------------------------------------------------------------------


@dataclass
class TradePathResult:
    trade_id: int
    truncate_bar: int  # last bar in truncated path
    sl_hit: bool
    peak_mfe_bar: int  # bar of max mfe_so_far_r within truncated path
    peak_mfe_r_candidate: float  # peak mfe in candidate R units
    final_r_candidate: float
    mae_at_peak_r_candidate: float
    pre_peak_monotonicity: float
    pre_peak_wrong_way: bool  # any bar in 0..peak_mfe_bar with mae_so_far_r <= threshold
    local_peaks_pre_peak: int


def truncate_path_at_sl(
    bar_offsets: np.ndarray,
    close_r: np.ndarray,
    mfe_so_far_r: np.ndarray,
    mae_so_far_r: np.ndarray,
    sl_threshold: float,
) -> int:
    """Return the truncate bar index (position within arrays).

    Returns first index where mae_so_far_r <= sl_threshold. If no such index,
    returns len-1 (path runs to end without SL hit).
    """
    hit_mask = mae_so_far_r <= sl_threshold
    if hit_mask.any():
        return int(np.argmax(hit_mask))  # first True
    return len(mae_so_far_r) - 1


def compute_trade_path_result(
    trade_id: int,
    bar_offsets: np.ndarray,  # sorted ascending starting from 0
    close_r: np.ndarray,
    mfe_so_far_r: np.ndarray,
    mae_so_far_r: np.ndarray,
    sl_atr: float,
    atr_per_R_simulation: float,
) -> TradePathResult:
    """Apply SL sweep at sl_atr ATR units and compute per-trade metrics."""
    sl_threshold = -sl_atr / atr_per_R_simulation  # in stored close_r units
    truncate_idx = truncate_path_at_sl(
        bar_offsets, close_r, mfe_so_far_r, mae_so_far_r, sl_threshold
    )
    sl_hit = bool(mae_so_far_r[truncate_idx] <= sl_threshold)

    # Truncate inclusive of truncate_idx.
    tr_close_r = close_r[: truncate_idx + 1]
    tr_mfe = mfe_so_far_r[: truncate_idx + 1]
    tr_mae = mae_so_far_r[: truncate_idx + 1]
    tr_offsets = bar_offsets[: truncate_idx + 1]

    # Peak MFE bar within truncated path. argmax returns first occurrence.
    peak_idx = int(np.argmax(tr_mfe))
    peak_mfe_bar = int(tr_offsets[peak_idx])
    peak_mfe_stored = float(tr_mfe[peak_idx])
    peak_mfe_r_candidate = peak_mfe_stored * (atr_per_R_simulation / sl_atr)

    # final_r: SL hit -> -1; else close at last bar re-denominated.
    if sl_hit:
        final_r_candidate = -1.0
    else:
        final_r_candidate = float(tr_close_r[-1]) * (atr_per_R_simulation / sl_atr)

    mae_at_peak_r_candidate = float(tr_mae[peak_idx]) * (atr_per_R_simulation / sl_atr)

    # Pre-peak metrics: bars 0..peak_mfe_bar inclusive.
    pp_close_r = tr_close_r[: peak_idx + 1]
    pp_mae = tr_mae[: peak_idx + 1]
    pp_mfe = tr_mfe[: peak_idx + 1]

    # Pre-peak monotonicity: among bars where close_r > 0, fraction where close_r >= previous in-profit bar.
    in_profit_mask = pp_close_r > 0
    in_profit_vals = pp_close_r[in_profit_mask]
    if in_profit_vals.size == 0:
        pre_peak_mono = 0.0
    elif in_profit_vals.size == 1:
        pre_peak_mono = 1.0  # only one in-profit bar trivially "monotone" (no prior to compare)
    else:
        prev = in_profit_vals[:-1]
        curr = in_profit_vals[1:]
        ge_count = int((curr >= prev).sum()) + 1  # include first bar (no prev, count as monotone)
        pre_peak_mono = float(ge_count) / float(in_profit_vals.size)

    # Pre-peak wrong_way: any bar in 0..peak with mae_so_far_r <= threshold.
    pre_peak_wrong_way = bool((pp_mae <= sl_threshold).any())

    # Pre-peak local_peaks: bars where mfe_so_far_r strictly increased.
    if pp_mfe.size <= 1:
        local_peaks_pre_peak = 0
    else:
        local_peaks_pre_peak = int((pp_mfe[1:] > pp_mfe[:-1]).sum())

    return TradePathResult(
        trade_id=trade_id,
        truncate_bar=int(tr_offsets[truncate_idx]),
        sl_hit=sl_hit,
        peak_mfe_bar=peak_mfe_bar,
        peak_mfe_r_candidate=float(peak_mfe_r_candidate),
        final_r_candidate=float(final_r_candidate),
        mae_at_peak_r_candidate=float(mae_at_peak_r_candidate),
        pre_peak_monotonicity=float(pre_peak_mono),
        pre_peak_wrong_way=pre_peak_wrong_way,
        local_peaks_pre_peak=int(local_peaks_pre_peak),
    )


# ---------------------------------------------------------------------------
# Bimodal-separated test.
# ---------------------------------------------------------------------------


def bimodal_separated_test(
    fwd_mfe_r: np.ndarray,
    hartigan_dip_p_max: float = 0.05,
    min_mode_mass: float = 0.20,
    min_mode_separation_r: float = 1.0,
) -> dict[str, Any]:
    """Apply v2.1.1 bimodal_separated test on fwd_mfe distribution at selected SL.

    Returns dict with dip_stat, dip_p, mode locations and masses, pass/fail flags.
    """
    import diptest
    import scipy.signal
    import scipy.stats as sps

    result: dict[str, Any] = {
        "n": int(fwd_mfe_r.size),
        "dip_stat": None,
        "dip_p": None,
        "kde_mode_1": None,
        "kde_mode_2": None,
        "mode_separation_r": None,
        "mass_mode_1": None,
        "mass_mode_2": None,
        "min_mode_mass": None,
        "pass_dip": False,
        "pass_mode_mass": False,
        "pass_separation": False,
        "bimodal_separated": False,
    }
    if fwd_mfe_r.size < 4:
        return result

    dip_stat, dip_p = diptest.diptest(fwd_mfe_r)
    result["dip_stat"] = float(dip_stat)
    result["dip_p"] = float(dip_p)
    result["pass_dip"] = bool(dip_p < hartigan_dip_p_max)

    # KDE peaks for mode locations.
    try:
        kde = sps.gaussian_kde(fwd_mfe_r)
        lo, hi = float(fwd_mfe_r.min()), float(fwd_mfe_r.max())
        if hi - lo < 1e-9:
            return result
        x_grid = np.linspace(lo, hi, 512)
        dens = kde(x_grid)
        peaks, _ = scipy.signal.find_peaks(dens)
        if peaks.size < 2:
            return result
        # Top two by density.
        top_two = peaks[np.argsort(-dens[peaks])[:2]]
        top_two_sorted = np.sort(top_two)
        m1, m2 = float(x_grid[top_two_sorted[0]]), float(x_grid[top_two_sorted[1]])
        result["kde_mode_1"] = m1
        result["kde_mode_2"] = m2
        separation = abs(m2 - m1)
        result["mode_separation_r"] = float(separation)
        result["pass_separation"] = bool(separation >= min_mode_separation_r)

        half = separation / 2.0
        mass_1 = float(((fwd_mfe_r >= m1 - half) & (fwd_mfe_r <= m1 + half)).sum()) / fwd_mfe_r.size
        mass_2 = float(((fwd_mfe_r >= m2 - half) & (fwd_mfe_r <= m2 + half)).sum()) / fwd_mfe_r.size
        result["mass_mode_1"] = mass_1
        result["mass_mode_2"] = mass_2
        min_mass = min(mass_1, mass_2)
        result["min_mode_mass"] = min_mass
        result["pass_mode_mass"] = bool(min_mass >= min_mode_mass)
    except Exception as e:
        result["kde_error"] = str(e)
        return result

    result["bimodal_separated"] = bool(
        result["pass_dip"] and result["pass_mode_mass"] and result["pass_separation"]
    )
    return result


# ---------------------------------------------------------------------------
# Composite + floor evaluation.
# ---------------------------------------------------------------------------


def evaluate_floors(
    metrics: dict[str, float],
    cfg_floors: dict[str, Any],
) -> dict[str, bool]:
    """Per-floor pass/fail."""
    return {
        "monotonicity_pre_peak": metrics["monotonicity_pre_peak_centroid"]
        >= cfg_floors["monotonicity_pre_peak_min"],
        "local_peaks": (
            cfg_floors["local_peaks_min"]
            <= metrics["local_peaks_centroid"]
            <= cfg_floors["local_peaks_max"]
        ),
        "fwd_mfe_p50": metrics["fwd_mfe_h240_p50"] >= cfg_floors["fwd_mfe_h240_p50_min"],
        "frac_reach_1R": metrics["frac_reach_1R"] >= cfg_floors["frac_reach_1R_min"],
        "frac_wrong_way_pre_peak": metrics["frac_wrong_way_pre_peak"]
        <= cfg_floors["frac_wrong_way_pre_peak_max"],
        "shape_tag": metrics["shape_tag"] in cfg_floors["shape_tag_admitted"],
        "size_fraction": metrics["size_fraction"] >= cfg_floors["size_fraction_min"],
    }


def capturability_composite(metrics: dict[str, float]) -> float:
    return (
        (metrics["monotonicity_pre_peak_centroid"] - 0.55)
        + (metrics["frac_reach_1R"] - 0.70)
        + (0.30 - metrics["frac_wrong_way_pre_peak"])
    )


# ---------------------------------------------------------------------------
# Archetype evaluation: full sweep + selection.
# ---------------------------------------------------------------------------


def percentiles(arr: np.ndarray, qs: list[int]) -> dict[str, float]:
    if arr.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def evaluate_sl_for_archetype(
    archetype_id: str,
    trade_ids: list[int],
    paths_grouped: dict[int, dict[str, np.ndarray]],
    sl_atr: float,
    atr_per_R_sim: float,
    cfg_floors: dict[str, Any],
    bimodal_cfg: dict[str, Any],
    pool_size: int,
    original_held_bars: int,
    features_by_trade: dict[int, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Run SL sweep at one candidate SL for one archetype. Returns metrics dict."""
    trs: list[TradePathResult] = []
    for tid in trade_ids:
        if tid not in paths_grouped:
            continue
        pg = paths_grouped[tid]
        tr = compute_trade_path_result(
            trade_id=tid,
            bar_offsets=pg["bar_offset"],
            close_r=pg["close_r"],
            mfe_so_far_r=pg["mfe_so_far_r"],
            mae_so_far_r=pg["mae_so_far_r"],
            sl_atr=sl_atr,
            atr_per_R_simulation=atr_per_R_sim,
        )
        trs.append(tr)

    n = len(trs)
    if n == 0:
        return {"n": 0, "skip": True}

    peak_mfe_arr = np.array([t.peak_mfe_r_candidate for t in trs], dtype=float)
    final_r_arr = np.array([t.final_r_candidate for t in trs], dtype=float)
    mae_at_peak_arr = np.array([t.mae_at_peak_r_candidate for t in trs], dtype=float)
    mono_arr = np.array([t.pre_peak_monotonicity for t in trs], dtype=float)
    local_peaks_arr = np.array([t.local_peaks_pre_peak for t in trs], dtype=float)
    peak_bar_arr = np.array([t.peak_mfe_bar for t in trs], dtype=int)

    frac_wrong_pre = float(sum(t.pre_peak_wrong_way for t in trs)) / n
    # frac_wrong_way_all_time: any bar (across the *original full path*, not truncated)
    # has mae_so_far_r <= threshold. Reference only, not gated.
    sl_threshold = -sl_atr / atr_per_R_sim
    frac_wrong_all = 0
    for tid in trade_ids:
        if tid not in paths_grouped:
            continue
        mae_full = paths_grouped[tid]["mae_so_far_r"]
        if (mae_full <= sl_threshold).any():
            frac_wrong_all += 1
    frac_wrong_all_time = float(frac_wrong_all) / n

    p_qs = [5, 10, 25, 50, 75, 90, 95]
    fwd_mfe_pcts = percentiles(peak_mfe_arr, p_qs)
    final_r_pcts = percentiles(final_r_arr, p_qs)
    mae_pcts = percentiles(mae_at_peak_arr, p_qs)

    # v2.0 shape_tag on fwd_mfe distribution (matches v2.1.1 §7 layering).
    v2_tag, v2_log = classify_shape_tag(peak_mfe_arr)
    bs_result = bimodal_separated_test(
        peak_mfe_arr,
        hartigan_dip_p_max=bimodal_cfg["hartigan_dip_p_max"],
        min_mode_mass=bimodal_cfg["min_mode_mass"],
        min_mode_separation_r=bimodal_cfg["min_mode_separation_r"],
    )
    if bs_result.get("bimodal_separated"):
        shape_tag = "bimodal_separated"
    else:
        shape_tag = v2_tag

    # Centroid (path-shape features from Step 2 if available; else compute simple proxies).
    if features_by_trade is not None:
        feats = [features_by_trade[tid] for tid in trade_ids if tid in features_by_trade]
        if feats:
            centroid_mono_v2 = float(np.mean([f["monotonicity_ratio_in_profit"] for f in feats]))
            centroid_local_peaks_v2 = float(np.mean([f["local_peaks_count"] for f in feats]))
            centroid_pullback = float(np.mean([f["pullback_magnitude_median"] for f in feats]))
            centroid_ttp_rel = float(np.mean([f["time_to_peak_mfe_relative"] for f in feats]))
        else:
            centroid_mono_v2 = centroid_local_peaks_v2 = centroid_pullback = centroid_ttp_rel = (
                float("nan")
            )
    else:
        centroid_mono_v2 = centroid_local_peaks_v2 = centroid_pullback = centroid_ttp_rel = float(
            "nan"
        )

    # Centroid for floor evaluation uses PRE-PEAK metrics under candidate R (v2.1).
    centroid_mono_pp = float(np.mean(mono_arr))
    centroid_local_peaks_pp = float(np.mean(local_peaks_arr))

    frac_reach_1R = float((peak_mfe_arr >= 1.0).sum()) / n
    frac_reach_2R = float((peak_mfe_arr >= 2.0).sum()) / n
    size_fraction = float(n) / float(pool_size)

    final_r_mean = float(final_r_arr.mean())
    final_r_std = float(final_r_arr.std(ddof=1)) if n > 1 else 0.0
    final_r_t = final_r_mean / (final_r_std / math.sqrt(n)) if final_r_std > 0 else 0.0

    frac_peak_at_bar_0 = float((peak_bar_arr == 0).sum()) / n
    frac_peak_after_bar_120 = float((peak_bar_arr > original_held_bars).sum()) / n

    metrics: dict[str, Any] = {
        "archetype_id": archetype_id,
        "selected_sl_atr": sl_atr,
        "n": n,
        "size_fraction": size_fraction,
        "centroid_monotonicity_v2": centroid_mono_v2,
        "centroid_local_peaks_v2": centroid_local_peaks_v2,
        "centroid_pullback": centroid_pullback,
        "centroid_time_to_peak_rel": centroid_ttp_rel,
        "monotonicity_pre_peak_centroid": centroid_mono_pp,
        "local_peaks_centroid": centroid_local_peaks_pp,
        "fwd_mfe_h240_p25": fwd_mfe_pcts["p25"],
        "fwd_mfe_h240_p50": fwd_mfe_pcts["p50"],
        "fwd_mfe_h240_p75": fwd_mfe_pcts["p75"],
        "fwd_mfe_h240_p95": fwd_mfe_pcts["p95"],
        "frac_reach_1R": frac_reach_1R,
        "frac_reach_2R": frac_reach_2R,
        "frac_wrong_way_pre_peak": frac_wrong_pre,
        "frac_wrong_way_all_time": frac_wrong_all_time,
        "frac_peak_at_bar_0": frac_peak_at_bar_0,
        "frac_peak_after_bar_120": frac_peak_after_bar_120,
        "final_r_mean": final_r_mean,
        "final_r_t_stat": float(final_r_t),
        "final_r_p25": final_r_pcts["p25"],
        "final_r_p50": final_r_pcts["p50"],
        "final_r_p75": final_r_pcts["p75"],
        "shape_tag": shape_tag,
        "shape_tag_v2_classifier": v2_tag,
        "hartigan_dip_stat": bs_result.get("dip_stat"),
        "hartigan_dip_p": bs_result.get("dip_p"),
        "mode_separation_r": bs_result.get("mode_separation_r"),
        "min_mode_mass": bs_result.get("min_mode_mass"),
        "kde_mode_1": bs_result.get("kde_mode_1"),
        "kde_mode_2": bs_result.get("kde_mode_2"),
        "pass_dip": bs_result.get("pass_dip"),
        "pass_mode_mass": bs_result.get("pass_mode_mass"),
        "pass_separation": bs_result.get("pass_separation"),
    }
    floor_results = evaluate_floors(metrics, cfg_floors)
    metrics.update({f"floor_{k}": v for k, v in floor_results.items()})
    metrics["all_floors_pass"] = all(floor_results.values())
    if metrics["all_floors_pass"]:
        metrics["composite"] = capturability_composite(metrics)
        # Tiebreaker stat: peak MFE in ATR units (use median across trades).
        metrics["peak_mfe_atr_median"] = float(np.median(peak_mfe_arr * sl_atr))
    else:
        metrics["composite"] = None
        metrics["peak_mfe_atr_median"] = float(np.median(peak_mfe_arr * sl_atr))

    # Distribution-side details for distribution.csv
    metrics["_extra_distribution"] = {
        "fwd_mfe_r_pcts": fwd_mfe_pcts,
        "final_r_pcts": final_r_pcts,
        "mae_at_peak_r_pcts": mae_pcts,
        "mass_in_band": {
            "0-0.5R": int(((peak_mfe_arr >= 0.0) & (peak_mfe_arr < 0.5)).sum()),
            "0.5-1R": int(((peak_mfe_arr >= 0.5) & (peak_mfe_arr < 1.0)).sum()),
            "1-2R": int(((peak_mfe_arr >= 1.0) & (peak_mfe_arr < 2.0)).sum()),
            "2-5R": int(((peak_mfe_arr >= 2.0) & (peak_mfe_arr < 5.0)).sum()),
            ">5R": int((peak_mfe_arr >= 5.0).sum()),
        },
    }
    return metrics


def select_sl(sweep_rows: list[dict[str, Any]], tolerance: float = 0.02) -> dict[str, Any] | None:
    passing = [r for r in sweep_rows if r.get("all_floors_pass")]
    if not passing:
        return None
    best_composite = max(r["composite"] for r in passing)
    # Tiebreaker 1: within tolerance, higher peak_mfe_atr_median wins.
    candidates = [r for r in passing if r["composite"] >= best_composite - tolerance]
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda r: (-r["peak_mfe_atr_median"], r["selected_sl_atr"]))
    return candidates[0]


# ---------------------------------------------------------------------------
# I/O and driver.
# ---------------------------------------------------------------------------


def load_paths_grouped(paths_csv: Path) -> dict[int, dict[str, np.ndarray]]:
    df = pd.read_csv(paths_csv)
    df = df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    out: dict[int, dict[str, np.ndarray]] = {}
    for tid, g in df.groupby("trade_id", sort=False):
        out[int(tid)] = {
            "bar_offset": g["bar_offset"].to_numpy(dtype=int),
            "close_r": g["close_r"].to_numpy(dtype=float),
            "mfe_so_far_r": g["mfe_so_far_r"].to_numpy(dtype=float),
            "mae_so_far_r": g["mae_so_far_r"].to_numpy(dtype=float),
        }
    return out


def load_features(features_csv: Path) -> dict[int, dict[str, float]]:
    df = pd.read_csv(features_csv)
    return {
        int(r["trade_id"]): {
            "monotonicity_ratio_in_profit": float(r["monotonicity_ratio_in_profit"]),
            "local_peaks_count": float(r["local_peaks_count"]),
            "pullback_magnitude_median": float(r["pullback_magnitude_median"]),
            "time_to_peak_mfe_relative": float(r["time_to_peak_mfe_relative"]),
        }
        for _, r in df.iterrows()
    }


def write_sweep_csv(sweep_rows: list[dict[str, Any]], out_path: Path) -> None:
    # Strip the _extra_distribution from each row for the sweep CSV.
    rows_clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in sweep_rows]
    df = pd.DataFrame(rows_clean)
    df.to_csv(out_path, index=False, float_format="%.6g")


def write_distribution_csv(selected_row: dict[str, Any], archetype_id: str, out_path: Path) -> None:
    extra = selected_row.get("_extra_distribution", {})
    fwd_p = extra.get("fwd_mfe_r_pcts", {})
    fin_p = extra.get("final_r_pcts", {})
    mae_p = extra.get("mae_at_peak_r_pcts", {})
    mass = extra.get("mass_in_band", {})
    rows = []
    for k in ["p5", "p10", "p25", "p50", "p75", "p90", "p95"]:
        rows.append(
            {
                "archetype_id": archetype_id,
                "metric": "fwd_mfe_r_candidate",
                "key": k,
                "value": fwd_p.get(k),
            }
        )
        rows.append(
            {
                "archetype_id": archetype_id,
                "metric": "final_r_candidate",
                "key": k,
                "value": fin_p.get(k),
            }
        )
        rows.append(
            {
                "archetype_id": archetype_id,
                "metric": "mae_at_peak_r_candidate",
                "key": k,
                "value": mae_p.get(k),
            }
        )
    for band, count in mass.items():
        rows.append(
            {
                "archetype_id": archetype_id,
                "metric": "mass_in_band",
                "key": band,
                "value": count,
            }
        )
    for k in [
        "hartigan_dip_stat",
        "hartigan_dip_p",
        "mode_separation_r",
        "min_mode_mass",
        "kde_mode_1",
        "kde_mode_2",
    ]:
        rows.append(
            {
                "archetype_id": archetype_id,
                "metric": "bimodal_test",
                "key": k,
                "value": selected_row.get(k),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.6g")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    out_dir = Path(args.output_dir) if args.output_dir else _REPO_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    paths_extended_path = _REPO_ROOT / cfg["inputs"]["trades_paths_extended"]
    if not paths_extended_path.exists():
        raise FileNotFoundError(
            f"Extended trades_paths missing: {paths_extended_path}. Run extend_paths.py first."
        )

    print(f"Loading extended paths: {paths_extended_path}")
    paths_grouped = load_paths_grouped(paths_extended_path)
    print(f"  {len(paths_grouped)} trades loaded")

    print(f"Loading clusters: {cfg['inputs']['clusters']}")
    clusters_df = pd.read_csv(_REPO_ROOT / cfg["inputs"]["clusters"])
    features_by_trade = load_features(_REPO_ROOT / cfg["inputs"]["path_features"])

    pool_size = int(cfg["pool_size"])
    atr_per_R = float(cfg["atr_units_per_R_at_simulation"])
    sl_candidates = cfg["sl_sweep_candidates_atr"]
    cfg_floors = cfg["floors"]
    bimodal_cfg = cfg["bimodal_separated_test"]
    tolerance = float(cfg["composite_tiebreaker_tolerance"])
    original_held_bars = int(cfg["original_held_window_bars"])

    cohort = cfg["cohort"]
    individual_cluster_ids = cohort["cluster_ids_individual"]
    aggregate_cluster_ids = cohort["cluster_ids_aggregate"]

    archetype_sweeps: dict[str, list[dict[str, Any]]] = {}
    archetype_selected: dict[str, dict[str, Any] | None] = {}

    # Per-cluster individual evaluations.
    for cluster_id in individual_cluster_ids:
        trade_ids = clusters_df.loc[clusters_df["cluster_label"] == cluster_id, "trade_id"].tolist()
        archetype_id = f"cluster_{cluster_id}"
        print(f"\n=== {archetype_id} (n={len(trade_ids)}) ===")
        sweep: list[dict[str, Any]] = []
        for sl in sl_candidates:
            print(f"  Sweep at SL={sl} ATR...")
            row = evaluate_sl_for_archetype(
                archetype_id=archetype_id,
                trade_ids=trade_ids,
                paths_grouped=paths_grouped,
                sl_atr=float(sl),
                atr_per_R_sim=atr_per_R,
                cfg_floors=cfg_floors,
                bimodal_cfg=bimodal_cfg,
                pool_size=pool_size,
                original_held_bars=original_held_bars,
                features_by_trade=features_by_trade,
            )
            sweep.append(row)
        archetype_sweeps[archetype_id] = sweep
        write_sweep_csv(sweep, out_dir / f"{archetype_id}_sl_sweep.csv")
        sel = select_sl(sweep, tolerance=tolerance)
        archetype_selected[archetype_id] = sel
        if sel:
            print(f"  SELECTED SL={sel['selected_sl_atr']} composite={sel['composite']:.4f}")
            write_distribution_csv(sel, archetype_id, out_dir / f"{archetype_id}_distribution.csv")
        else:
            print("  NO PASSING SL")

    # Aggregate evaluation.
    agg_trade_ids = clusters_df.loc[
        clusters_df["cluster_label"].isin(aggregate_cluster_ids), "trade_id"
    ].tolist()
    agg_id_str = "+".join(str(c) for c in sorted(aggregate_cluster_ids))
    agg_archetype = f"aggregate_stepwise_climber_{agg_id_str}"
    print(f"\n=== {agg_archetype} (n={len(agg_trade_ids)}) ===")
    sweep = []
    for sl in sl_candidates:
        print(f"  Sweep at SL={sl} ATR...")
        row = evaluate_sl_for_archetype(
            archetype_id=agg_archetype,
            trade_ids=agg_trade_ids,
            paths_grouped=paths_grouped,
            sl_atr=float(sl),
            atr_per_R_sim=atr_per_R,
            cfg_floors=cfg_floors,
            bimodal_cfg=bimodal_cfg,
            pool_size=pool_size,
            original_held_bars=original_held_bars,
            features_by_trade=features_by_trade,
        )
        sweep.append(row)
    archetype_sweeps[agg_archetype] = sweep
    write_sweep_csv(sweep, out_dir / f"{agg_archetype}_sl_sweep.csv")
    sel = select_sl(sweep, tolerance=tolerance)
    archetype_selected[agg_archetype] = sel
    if sel:
        print(f"  SELECTED SL={sel['selected_sl_atr']} composite={sel['composite']:.4f}")
        write_distribution_csv(sel, agg_archetype, out_dir / f"{agg_archetype}_distribution.csv")
    else:
        print("  NO PASSING SL")

    # archetype_summaries.csv — one row per archetype at selected SL (or fail row).
    summaries = []
    for aid, sel in archetype_selected.items():
        if sel is not None:
            row = {k: v for k, v in sel.items() if not k.startswith("_")}
            row["verdict"] = "PASS"
        else:
            # Use simulation SL row for the fail summary.
            sim_row = next(
                (r for r in archetype_sweeps[aid] if r.get("selected_sl_atr") == atr_per_R),
                archetype_sweeps[aid][0],
            )
            row = {k: v for k, v in sim_row.items() if not k.startswith("_")}
            row["verdict"] = "FAIL_NO_PASSING_SL"
        summaries.append(row)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "archetype_summaries.csv", index=False, float_format="%.6g")

    # capturability_pass_list.csv
    pass_rows = []
    for aid, sel in archetype_selected.items():
        if sel is not None:
            pass_rows.append(
                {
                    "archetype_id": aid,
                    "selected_sl_atr": sel["selected_sl_atr"],
                    "composite": sel["composite"],
                    "verdict": "PASS",
                }
            )
    pd.DataFrame(
        pass_rows, columns=["archetype_id", "selected_sl_atr", "composite", "verdict"]
    ).to_csv(out_dir / "capturability_pass_list.csv", index=False, float_format="%.6g")

    # cluster_routing.csv
    routing_rows = []
    agg_pass = archetype_selected[agg_archetype] is not None
    for cluster_id in individual_cluster_ids:
        aid = f"cluster_{cluster_id}"
        ind_pass = archetype_selected[aid] is not None
        if ind_pass and agg_pass:
            routing = "both"
        elif ind_pass and not agg_pass:
            routing = "individual"
        elif not ind_pass and agg_pass:
            routing = "aggregate"
        else:
            routing = "dies"
        routing_rows.append(
            {
                "cluster_id": cluster_id,
                "individual_verdict": "PASS" if ind_pass else "FAIL",
                "aggregate_verdict": "PASS" if agg_pass else "FAIL",
                "routing": routing,
            }
        )
    pd.DataFrame(routing_rows).to_csv(out_dir / "cluster_routing.csv", index=False)

    print(f"\nWrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

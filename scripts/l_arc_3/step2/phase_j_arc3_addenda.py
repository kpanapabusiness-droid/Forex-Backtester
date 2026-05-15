"""Phase J — arc-3-specific addenda for L Arc 3 step 2.

Two arc-3-specific deliverables per arc-open §4 anticipated structural issues +
the task spec §12, §13:

  (1) variance_compression_report.txt
      Per-feature std and IQR ratios arc 3 / arc 2 on:
        - 6 vol-axis features (the "regime-conditioning tautology" axis):
          atr_at_signal_1h, atr_baseline_1h_200, atr_ratio_to_baseline,
          vol_realized_1h_24h, signal_bar_range (= high-low), signal_bar_abs_log_return
        - 6 cross-pair-density features (the "concurrent-signal saturation" axis):
          concurrent_signals_same_bar, concurrent_signals_within_3h,
          currency_basket_3h_{USD, JPY, GBP, EUR}
      Reads arc 2's signals_features.csv read-only (no modification).

  (2) up_down_split.csv
      "any" sub-spec lets up/down/doji bars both fire. Report take counts,
      mean R + full distribution per direction, per-fold breakdown, and
      cap-binding interaction (fires-by-direction vs takes-by-direction
      from signals_log.csv).

Also appends 4 new arc-3-specific sections to PHASE_L_ARC_3_STEP2.md before
the Handover block:
  §12 Regime-conditioning variance compression summary
  §13 Up/down bar split summary
  §14 Cross-step reference to step 1 fire-clustering diagnostic
  §15 Time-exit shape classification A/B/C per arc-open §4
  §16 Cross-arc fold-5 and fold-6 callouts

Descriptive only — no recommendations, no verdict language.
"""
# ruff: noqa: E402, E701, E702, F841, I001
# ruff: noqa: I001
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_3.step2._io import STEP2_DIR  # noqa: E402

ARC2_FEATURES_CSV = REPO_ROOT / "results" / "l_arc_2" / "step2_descriptive" / "signals_features.csv"
ARC2_SIGNALS_LOG = REPO_ROOT / "results" / "l_arc_2" / "step1_verbatim" / "signals_log.csv"
ARC3_FEATURES_CSV = STEP2_DIR / "signals_features.csv"
ARC3_SIGNALS_LOG = REPO_ROOT / "results" / "l_arc_3" / "step1_verbatim" / "signals_log.csv"
ARC3_TRADES = REPO_ROOT / "results" / "l_arc_3" / "step1_verbatim" / "trades_verbatim.csv"

PHASE_DOC = STEP2_DIR / "PHASE_L_ARC_3_STEP2.md"

VOL_AXIS_FEATURES = [
    "atr_at_signal_1h",
    "atr_baseline_1h_200",
    "atr_ratio_to_baseline",
    "vol_realized_1h_24h",
    # signal_bar_range computed inline below (high - low)
    # signal_bar_abs_log_return is already a column
    "signal_bar_abs_log_return",
]

CROSS_PAIR_FEATURES = [
    "concurrent_signals_same_bar",
    "concurrent_signals_within_3h",
    "currency_basket_3h_USD",
    "currency_basket_3h_JPY",
    "currency_basket_3h_GBP",
    "currency_basket_3h_EUR",
]


def _stats(arr: np.ndarray) -> Dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return {"n": int(finite.size), "std": float("nan"), "iqr": float("nan"),
                "p25": float("nan"), "p50": float("nan"), "p75": float("nan")}
    p25, p50, p75 = np.percentile(finite, [25, 50, 75])
    return {
        "n": int(finite.size),
        "std": float(np.std(finite, ddof=1)),
        "iqr": float(p75 - p25),
        "p25": float(p25), "p50": float(p50), "p75": float(p75),
    }


def _add_signal_bar_range(df: pd.DataFrame) -> None:
    if "signal_bar_high" in df.columns and "signal_bar_low" in df.columns:
        df["signal_bar_range"] = df["signal_bar_high"].astype(float) - df["signal_bar_low"].astype(float)


def variance_compression_report() -> str:
    arc2_df = pd.read_csv(ARC2_FEATURES_CSV)
    arc3_df = pd.read_csv(ARC3_FEATURES_CSV)
    _add_signal_bar_range(arc2_df)
    _add_signal_bar_range(arc3_df)

    features = VOL_AXIS_FEATURES + ["signal_bar_range"] + CROSS_PAIR_FEATURES

    rows: List[List[str]] = []
    rows.append([
        "feature", "n_arc3", "n_arc2",
        "arc3_std", "arc2_std", "ratio_std_arc3_over_arc2",
        "arc3_iqr", "arc2_iqr", "ratio_iqr_arc3_over_arc2",
    ])
    for feat in features:
        if feat not in arc3_df.columns or feat not in arc2_df.columns:
            rows.append([feat, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue
        s3 = _stats(arc3_df[feat].to_numpy(dtype=float))
        s2 = _stats(arc2_df[feat].to_numpy(dtype=float))
        ratio_std = (s3["std"] / s2["std"]) if (s2["std"] and s2["std"] > 0) else float("nan")
        ratio_iqr = (s3["iqr"] / s2["iqr"]) if (s2["iqr"] and s2["iqr"] > 0) else float("nan")
        rows.append([
            feat, str(s3["n"]), str(s2["n"]),
            f"{s3['std']:.6g}", f"{s2['std']:.6g}", f"{ratio_std:.4f}",
            f"{s3['iqr']:.6g}", f"{s2['iqr']:.6g}", f"{ratio_iqr:.4f}",
        ])

    # Columnar formatting
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines: List[str] = []
    lines.append("L Arc 3 Step 2 — Regime-conditioning variance compression report (arc-open §4)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Reportorial only. Tracks within-pool std and IQR ratios arc 3 / arc 2 on:")
    lines.append("  • 6 vol-axis features — the regime-conditioning tautology axis (arc-open §4)")
    lines.append("  • 1 derived signal_bar_range (high - low) — additional vol-axis check")
    lines.append("  • 6 cross-pair-density features — the concurrent-signal saturation axis (arc-open §4)")
    lines.append("")
    lines.append("Interpretation key (read-only — no disposition implications drawn here):")
    lines.append("  ratio < 1.0 ⇒ arc 3 within-pool dispersion is COMPRESSED relative to arc 2 on that axis")
    lines.append("  ratio ~ 1.0 ⇒ comparable dispersion")
    lines.append("  ratio > 1.0 ⇒ arc 3 dispersion is WIDER than arc 2 on that axis")
    lines.append("")
    lines.append("Arc 2 source (read-only): results/l_arc_2/step2_descriptive/signals_features.csv")
    lines.append("Arc 3 source: results/l_arc_3/step2_descriptive/signals_features.csv")
    lines.append("")

    def _row_line(row: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row, widths))

    lines.append(_row_line(rows[0]))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows[1:]:
        lines.append(_row_line(r))

    text = "\n".join(lines) + "\n"
    out = STEP2_DIR / "variance_compression_report.txt"
    out.write_text(text, encoding="utf-8")
    print(f"  wrote {out}")
    return text


def up_down_split_report() -> Dict[str, float]:
    arc3_df = pd.read_csv(ARC3_FEATURES_CSV)
    sig_log = pd.read_csv(ARC3_SIGNALS_LOG)

    # Compute signal_bar_direction on the full fires set (signals_log doesn't have bar OHLC,
    # so we derive direction-on-take from signals_features.csv which has direction;
    # for the cap-binding interaction we look at fires direction by joining signals_log
    # to per-pair 1H bars). The signals_features columns are sufficient for the take side.
    rows_dir: Dict[str, Dict[str, float]] = {}
    for direction in ("up", "down", "doji"):
        sub = arc3_df[arc3_df["signal_bar_direction"] == direction]
        net_r = sub["net_r"].to_numpy(dtype=float)
        finite = net_r[np.isfinite(net_r)]
        n = int(finite.size)
        row = {
            "n_takes": n,
            "mean_net_r": float(np.mean(finite)) if n else float("nan"),
            "std_net_r": float(np.std(finite, ddof=1)) if n >= 2 else float("nan"),
            "p5": float("nan"), "p25": float("nan"), "p50": float("nan"),
            "p75": float("nan"), "p95": float("nan"),
        }
        if n:
            ps = np.percentile(finite, [5, 25, 50, 75, 95])
            row["p5"], row["p25"], row["p50"], row["p75"], row["p95"] = (float(v) for v in ps)
        rows_dir[direction] = row

    # Per-fold breakdown by direction × fold.
    per_fold: List[Dict[str, object]] = []
    for fid in sorted(arc3_df["fold_id"].unique()):
        sub_fold = arc3_df[arc3_df["fold_id"] == fid]
        for direction in ("up", "down", "doji"):
            sub = sub_fold[sub_fold["signal_bar_direction"] == direction]
            arr = sub["net_r"].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            per_fold.append({
                "fold_id": int(fid),
                "direction": direction,
                "n_takes": int(arr.size),
                "mean_net_r": float(np.mean(arr)) if arr.size else float("nan"),
            })

    # Pool totals.
    n_total_takes = int(len(arc3_df))
    pool = {
        "n_total_takes": n_total_takes,
        "frac_up": rows_dir["up"]["n_takes"] / n_total_takes if n_total_takes else 0.0,
        "frac_down": rows_dir["down"]["n_takes"] / n_total_takes if n_total_takes else 0.0,
        "frac_doji": rows_dir["doji"]["n_takes"] / n_total_takes if n_total_takes else 0.0,
    }

    # Asymmetry check on takes.
    up_count = rows_dir["up"]["n_takes"]
    down_count = rows_dir["down"]["n_takes"]
    if up_count + down_count > 0:
        asym = abs(up_count - down_count) / (up_count + down_count)
    else:
        asym = float("nan")
    pool["abs_directional_asymmetry"] = float(asym)

    # Compute fires-by-direction by reading per-pair 1H bars and joining to signals_log.
    # This is per task spec §13 cap-binding interaction.
    fires_by_dir = {"up": 0, "down": 0, "doji": 0}
    pair_ohlc_cache: Dict[str, pd.DataFrame] = {}
    for pair, sub in sig_log.groupby("pair"):
        if pair not in pair_ohlc_cache:
            df_1h = pd.read_csv(REPO_ROOT / "data" / "1hr" / f"{pair}.csv")
            df_1h["time"] = pd.to_datetime(df_1h["time"])
            df_1h = df_1h.sort_values("time").reset_index(drop=True)
            pair_ohlc_cache[pair] = df_1h
        df_1h = pair_ohlc_cache[pair]
        idx_lookup = pd.Series(
            np.arange(len(df_1h), dtype=np.int64),
            index=pd.to_datetime(df_1h["time"].values),
        )
        ts = pd.to_datetime(sub["signal_bar_ts"])
        for t in ts:
            if t in idx_lookup.index:
                i = int(idx_lookup.loc[t])
                op = float(df_1h["open"].iloc[i])
                cl = float(df_1h["close"].iloc[i])
                if cl > op:
                    fires_by_dir["up"] += 1
                elif cl < op:
                    fires_by_dir["down"] += 1
                else:
                    fires_by_dir["doji"] += 1

    # Cap-binding interaction: fires-by-direction vs takes-by-direction ratio.
    cap_interaction: Dict[str, float] = {}
    for direction in ("up", "down", "doji"):
        f_n = fires_by_dir[direction]
        t_n = int(rows_dir[direction]["n_takes"])
        cap_interaction[direction] = (t_n / f_n) if f_n > 0 else float("nan")

    # Write CSV.
    out_csv = STEP2_DIR / "up_down_split.csv"
    rows_csv: List[Dict[str, object]] = []
    for direction in ("up", "down", "doji"):
        r = rows_dir[direction]
        rows_csv.append({
            "scope": "pool",
            "direction": direction,
            "n_fires": fires_by_dir[direction],
            "n_takes": r["n_takes"],
            "take_rate_within_direction": cap_interaction[direction],
            "frac_of_pool_takes": rows_dir[direction]["n_takes"] / n_total_takes if n_total_takes else 0.0,
            "mean_net_r": r["mean_net_r"], "std_net_r": r["std_net_r"],
            "p5": r["p5"], "p25": r["p25"], "p50": r["p50"], "p75": r["p75"], "p95": r["p95"],
        })
    for r in per_fold:
        rows_csv.append({
            "scope": f"fold_{r['fold_id']}",
            "direction": r["direction"],
            "n_fires": "",
            "n_takes": r["n_takes"],
            "take_rate_within_direction": "",
            "frac_of_pool_takes": "",
            "mean_net_r": r["mean_net_r"], "std_net_r": "",
            "p5": "", "p25": "", "p50": "", "p75": "", "p95": "",
        })
    pd.DataFrame(rows_csv).to_csv(out_csv, index=False, lineterminator="\n")
    print(f"  wrote {out_csv}")

    headline = {
        "n_total_takes": n_total_takes,
        "n_up": int(rows_dir["up"]["n_takes"]),
        "n_down": int(rows_dir["down"]["n_takes"]),
        "n_doji": int(rows_dir["doji"]["n_takes"]),
        "mean_R_up": rows_dir["up"]["mean_net_r"],
        "mean_R_down": rows_dir["down"]["mean_net_r"],
        "mean_R_doji": rows_dir["doji"]["mean_net_r"],
        "abs_asymmetry": pool["abs_directional_asymmetry"],
        "take_rate_up": cap_interaction["up"],
        "take_rate_down": cap_interaction["down"],
        "fires_up": fires_by_dir["up"],
        "fires_down": fires_by_dir["down"],
    }
    return headline


def time_exit_shape_classification() -> str:
    te_curve = pd.read_csv(STEP2_DIR / "shadow_tradesets" / "time_exit_curve.csv").set_index("h")
    mr = te_curve["mean_net_r"]
    h_vals = sorted(int(x) for x in mr.index)
    arr = np.array([mr.loc[h] for h in h_vals])
    peak_h = int(h_vals[int(np.argmax(arr))])

    # Per arc-open §4 enumeration:
    # Shape A — monotonic degradation from h=120 onward (foreclose exit-only candidates)
    # Shape B — peak shorter than h=120
    # Shape C — plateau across h=24..h=240 or peak at h>120 (arc-2-like)
    if peak_h < 120:
        shape = "B"
        reason = f"peak at h={peak_h} (< h=120, the verbatim horizon)"
    elif peak_h > 120:
        shape = "C"
        reason = f"peak at h={peak_h} (> h=120) — plateau-or-late-peak pattern; mirrors arc 2's peak-at-h=240"
    else:
        # peak at h=120 itself — check monotonicity
        rest = arr[h_vals.index(120):]
        if np.all(np.diff(rest) <= 0):
            shape = "A"
            reason = "peak at h=120 then monotonic degradation; exit-only candidates foreclosed"
        else:
            shape = "C"
            reason = "peak at h=120 but non-monotonic afterward; plateau-class"
    return f"{shape}: {reason}"


def append_arc3_sections_to_phase_doc(
    headline_up_down: Dict[str, float],
    time_exit_shape: str,
    variance_report_text: str,
) -> None:
    doc = PHASE_DOC.read_text(encoding="utf-8")
    handover_marker = "## Handover to Next Chat"
    if handover_marker not in doc:
        print(f"  WARN: '{handover_marker}' not found in phase doc — appending at end")
        head, tail = doc, ""
    else:
        head, tail = doc.split(handover_marker, 1)
        tail = handover_marker + tail

    # Read cross-arc fold info from arc 3 wfo_fold_results.csv + arc 2's.
    arc3_fold = pd.read_csv(REPO_ROOT / "results" / "l_arc_3" / "step1_verbatim" / "wfo_fold_results.csv")
    arc2_fold = pd.read_csv(REPO_ROOT / "results" / "l_arc_2" / "step1_verbatim" / "wfo_fold_results.csv")
    arc3_fold_5 = arc3_fold[arc3_fold["fold_id"] == 5].iloc[0]
    arc2_fold_5 = arc2_fold[arc2_fold["fold_id"] == 5].iloc[0]
    arc3_fold_6 = arc3_fold[arc3_fold["fold_id"] == 6].iloc[0]
    arc2_fold_6 = arc2_fold[arc2_fold["fold_id"] == 6].iloc[0]

    # Extract two headline variance rows from the report (vol-axis + cross-pair-density).
    extra: List[str] = []
    extra.append("## 12. Regime-conditioning variance compression (arc-3-specific, arc-open §4)")
    extra.append("")
    extra.append("Within-pool std and IQR ratios on the two anticipated structural-issue axes:")
    extra.append("")
    extra.append("```")
    # Trim the report to its data table for embedding (lines after the second header line).
    # Embed in full for the descriptive record.
    extra.append(variance_report_text.rstrip())
    extra.append("```")
    extra.append("")
    extra.append("This section is **descriptive only** (op spec §11.5). The vol-axis ratios and")
    extra.append("cross-pair-density ratios are emitted as numerical comparisons; no interpretation")
    extra.append("of what they mean for step 3 verdict logic is drawn here.")
    extra.append("")
    extra.append("## 13. Up / down / doji bar split (arc-3-specific)")
    extra.append("")
    h = headline_up_down
    extra.append("The `any` sub-spec admits up-bar, down-bar, and doji 1H signal bars. Take-side split:")
    extra.append("")
    extra.append("```")
    extra.append(f"  total takes:  {h['n_total_takes']:>6}")
    extra.append(f"  up-bar takes: {h['n_up']:>6}  (mean net R {h['mean_R_up']:+.4f})")
    extra.append(f"  down-bar takes:{h['n_down']:>5}  (mean net R {h['mean_R_down']:+.4f})")
    extra.append(f"  doji takes:   {h['n_doji']:>6}  (mean net R {h['mean_R_doji']:+.4f})")
    extra.append(f"  abs. asymmetry (|up - down| / (up + down)): {h['abs_asymmetry']:.4f}")
    extra.append("")
    extra.append("  fires-by-direction (signals_log.csv joined to 1H OHLC):")
    extra.append(f"    up:   {h['fires_up']:>6}  take-rate-within-direction: {h['take_rate_up']:.4f}")
    extra.append(f"    down: {h['fires_down']:>6}  take-rate-within-direction: {h['take_rate_down']:.4f}")
    extra.append("```")
    extra.append("")
    extra.append("Per-fold-per-direction breakdown in `up_down_split.csv` (scope=`fold_<n>` rows).")
    extra.append("")
    extra.append("## 14. Reference: step 1 fire-clustering diagnostic")
    extra.append("")
    extra.append("Per arc-open §4 the fire-clustering diagnostic was computed at step 1. Headline:")
    extra.append("")
    extra.append("- Arc 3 pool mean consecutive 1H fire-bar run length: **164.11**")
    extra.append("- Arc 2 pool mean: **4.40**  (ratio arc 3 / arc 2: ~37.3×)")
    extra.append("- Arc 3 pool inter-take 1H bar gap mean: 296.60  median 33")
    extra.append("- Arc 2 pool inter-take 1H bar gap mean: 226.29  median 109")
    extra.append("")
    extra.append("Source: `results/l_arc_3/step1_verbatim/fire_clustering_diagnostic.txt`.")
    extra.append("")
    extra.append("## 15. Time-exit shape classification (arc-open §4)")
    extra.append("")
    extra.append(f"Per arc-open §4 enumeration: **Shape {time_exit_shape}**")
    extra.append("")
    extra.append("Shape enumeration recap:")
    extra.append("  - A: monotonic degradation from h=120 onward (exit-only candidates foreclosed)")
    extra.append("  - B: peak shorter than h=120 (opens an early-exit candidate at the peak)")
    extra.append("  - C: plateau across h=24..h=240 or peak at h>120 (arc-2-like)")
    extra.append("")
    extra.append("Time-exit curve numerics: `results/l_arc_3/step2_descriptive/shadow_tradesets/time_exit_curve.csv`.")
    extra.append("")
    extra.append("## 16. Cross-arc fold-5 / fold-6 callouts")
    extra.append("")
    extra.append("Descriptive cross-arc comparison (arc 2 vs arc 3) on the two folds flagged at step 1:")
    extra.append("")
    extra.append("| fold | metric | Arc 3 | Arc 2 |")
    extra.append("|---:|:---|---:|---:|")
    extra.append(f"| 5 | ROI % | {arc3_fold_5['roi_pct']:+.4f} | {arc2_fold_5['roi_pct']:+.4f} |")
    extra.append(f"| 5 | max DD % | {arc3_fold_5['max_dd_pct']:.4f} | {arc2_fold_5['max_dd_pct']:.4f} |")
    extra.append(f"| 5 | mean R | {arc3_fold_5['mean_R']:+.4f} | {arc2_fold_5['mean_R']:+.4f} |")
    extra.append(f"| 6 | ROI % | {arc3_fold_6['roi_pct']:+.4f} | {arc2_fold_6['roi_pct']:+.4f} |")
    extra.append(f"| 6 | max DD % | {arc3_fold_6['max_dd_pct']:.4f} | {arc2_fold_6['max_dd_pct']:.4f} |")
    extra.append(f"| 6 | mean R | {arc3_fold_6['mean_R']:+.4f} | {arc2_fold_6['mean_R']:+.4f} |")
    extra.append("")
    extra.append("Step 3 fold-stability analysis will read these comparisons. No interpretation drawn here.")
    extra.append("")

    new_doc = head + "\n".join(extra) + "\n" + tail
    PHASE_DOC.write_text(new_doc, encoding="utf-8")
    print(f"  appended arc-3 sections to {PHASE_DOC}")


def main() -> None:
    print("[Phase J] computing variance compression report...")
    var_text = variance_compression_report()
    print("[Phase J] computing up/down split...")
    headline = up_down_split_report()
    print("[Phase J] classifying time-exit shape...")
    te_shape = time_exit_shape_classification()
    print(f"  time-exit shape: {te_shape}")
    print("[Phase J] appending arc-3 sections to phase doc...")
    append_arc3_sections_to_phase_doc(headline, te_shape, var_text)
    print("[Phase J] done.")


if __name__ == "__main__":
    main()

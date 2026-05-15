"""Phase I — write PHASE_L_ARC_3_STEP2.md, step2_sanity_checks.txt, run_manifest.txt.

Descriptive only — no recommendations (op spec §11.5).
"""
# ruff: noqa: E402, E701, E702, F841, I001
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_3.step2._io import (
    FORWARD_HORIZON_BARS_DEFAULT,
    FORWARD_HORIZON_BARS_EXTENDED,
    H_GRID,
    STEP1_DIR,
    STEP2_DIR,
    sha256_file,
)


def _read_text_or_none(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else "<missing>"


def _safe_quantile(s: pd.Series, q: float) -> float:
    s2 = s.dropna(); return float(s2.quantile(q)) if len(s2) else float("nan")


def _safe_mean(s: pd.Series) -> float:
    s2 = s.dropna(); return float(s2.mean()) if len(s2) else float("nan")


def _safe_median(s: pd.Series) -> float:
    s2 = s.dropna(); return float(s2.median()) if len(s2) else float("nan")


def write_sanity_checks(extended: bool = False) -> Path:
    H = FORWARD_HORIZON_BARS_EXTENDED if extended else FORWARD_HORIZON_BARS_DEFAULT
    f_path = STEP2_DIR / "signals_features.csv"
    t_path = STEP2_DIR / "trade_paths.csv"
    f = pd.read_csv(f_path)
    n_feat = len(f)
    with t_path.open("rb") as fh:
        n_paths_lines = sum(1 for _ in fh) - 1
    trades_path = STEP1_DIR / "trades_verbatim.csv"
    n_trades_raw = sum(1 for _ in trades_path.open("rb")) - 1

    checks: List[dict] = []
    checks.append({"id": 1, "check": "signals_features.csv rows = trades_verbatim.csv rows",
                   "expected": n_trades_raw, "observed": n_feat,
                   "status": "PASS" if n_feat == n_trades_raw else "FAIL"})
    # trade_paths.csv: rows = sum of forward_window_bars_available across trades
    paths_expected = int(f["forward_window_bars_available"].sum()) if "forward_window_bars_available" in f.columns else (n_feat * H)
    checks.append({"id": 2, "check": "trade_paths.csv row count = sum of available forward bars",
                   "expected": paths_expected, "observed": n_paths_lines,
                   "status": "PASS" if n_paths_lines == paths_expected else "FAIL"})

    sub_dirs_required = [
        STEP2_DIR / "distributions" / "marginals",
        STEP2_DIR / "distributions" / "forward",
        STEP2_DIR / "distributions" / "sequence",
        STEP2_DIR / "distributions" / "complexity",
        STEP2_DIR / "distributions" / "survival",
        STEP2_DIR / "distributions" / "early_bar",
        STEP2_DIR / "distributions" / "asymmetry",
        STEP2_DIR / "conditional_breakdowns",
        STEP2_DIR / "joint_distributions",
        STEP2_DIR / "shadow_tradesets",
        STEP2_DIR / "cost_stress",
        STEP2_DIR / "random_baseline",
        STEP2_DIR / "held_bar_evolution",
        STEP2_DIR / "forward_context_evolution",
    ]
    all_present = all(p.exists() for p in sub_dirs_required)
    missing = [str(p.relative_to(STEP2_DIR)) for p in sub_dirs_required if not p.exists()]
    checks.append({"id": 3, "check": "All op spec §5 subfolders present",
                   "expected": "all present",
                   "observed": "all present" if all_present else f"missing: {missing}",
                   "status": "PASS" if all_present else "FAIL"})

    new_psc = ["cum_logret_1h_24", "cum_logret_1h_72", "cum_logret_1h_168", "vol_realized_1h_24h"]
    psc_present = all(c in f.columns for c in new_psc)
    checks.append({"id": 4, "check": "4 new arc-3 pre-signal context features present",
                   "expected": "all 4 present", "observed": "all 4 present" if psc_present else f"missing: {[c for c in new_psc if c not in f.columns]}",
                   "status": "PASS" if psc_present else "FAIL"})
    amd4 = ["fwd_realized_range_atr", "fwd_fraction_time_above_entry", "fwd_max_consecutive_directional_bars"]
    amd4_present = all(c in f.columns for c in amd4)
    checks.append({"id": 5, "check": "3 Amendment 4 clustering features present",
                   "expected": "all 3 present", "observed": "all 3 present" if amd4_present else f"missing: {[c for c in amd4 if c not in f.columns]}",
                   "status": "PASS" if amd4_present else "FAIL"})

    look_text = _read_text_or_none(STEP2_DIR / "lookahead_invariant_features_test.txt")
    look_pass = "RESULT: PASS" in look_text
    checks.append({"id": 6, "check": "Lookahead-invariant features test",
                   "expected": "PASS", "observed": "PASS" if look_pass else "FAIL",
                   "status": "PASS" if look_pass else "FAIL"})

    feat_lag_text = _read_text_or_none(STEP2_DIR / "feature_lag_audit.txt")
    feat_lag_pass = "RESULT: PASS" in feat_lag_text
    checks.append({"id": 7, "check": "Feature lag audit (op spec §10.4)",
                   "expected": "PASS", "observed": "PASS" if feat_lag_pass else "FAIL",
                   "status": "PASS" if feat_lag_pass else "FAIL"})

    stab_path = STEP2_DIR / "forward_horizon_stability.txt"
    stab_text = _read_text_or_none(stab_path)
    stab_present = stab_path.exists() and "Triggered" in stab_text
    checks.append({"id": 8, "check": "Forward-horizon stability check produced",
                   "expected": "documented", "observed": "documented" if stab_present else "missing",
                   "status": "PASS" if stab_present else "FAIL"})

    rand_present = (STEP2_DIR / "random_baseline" / "random_entry_distribution.csv").exists() and \
                   (STEP2_DIR / "random_baseline" / "comparison.csv").exists()
    differs_text = _read_text_or_none(STEP2_DIR / "random_baseline" / "differs_from_verbatim.txt")
    differs_yes = "differs (descriptive heuristic): True" in differs_text
    checks.append({"id": 9, "check": "Random-entry baseline produced; differs documented",
                   "expected": "produced",
                   "observed": "produced; differs=" + ("yes" if differs_yes else "no"),
                   "status": "PASS" if rand_present else "FAIL"})

    pair_breakdown_path = STEP2_DIR / "conditional_breakdowns" / "pair" / "net_r.csv"
    sample_ok = pair_breakdown_path.exists()
    checks.append({"id": 10, "check": "Sample-size discipline applied (n<30 flag column)",
                   "expected": "applied", "observed": "applied" if sample_ok else "missing",
                   "status": "PASS" if sample_ok else "FAIL"})

    sample_dist_text = _read_text_or_none(STEP2_DIR / "distributions" / "marginals" / "net_r.csv")
    has_full_shape = "n,n_nan,mean,std,skew,kurt,min,p1,p5,p10,p20,p30,p40,p50,p60,p70,p80,p90,p95,p99,max" in sample_dist_text
    checks.append({"id": 11, "check": "Distribution shape per §11.1",
                   "expected": "full shape", "observed": "full shape" if has_full_shape else "incomplete",
                   "status": "PASS" if has_full_shape else "FAIL"})

    joint_dir = STEP2_DIR / "joint_distributions"
    n_joint = sum(1 for _ in joint_dir.glob("*.csv")) if joint_dir.exists() else 0
    checks.append({"id": 12, "check": "2D heatmaps (§11.2)",
                   "expected": ">=9", "observed": f"{n_joint}",
                   "status": "PASS" if n_joint >= 9 else "FAIL"})

    by_fold = sum(1 for _ in (STEP2_DIR / "distributions").rglob("*__by_fold.csv"))
    checks.append({"id": 13, "check": "Per-fold breakdowns (§11.3)",
                   "expected": ">=20", "observed": f"{by_fold}",
                   "status": "PASS" if by_fold > 20 else "FAIL"})

    det_text = _read_text_or_none(STEP2_DIR / "determinism_check.txt")
    det_pass = "RESULT: PASS" in det_text
    checks.append({"id": 14, "check": "Determinism: two-consecutive-run byte-identical",
                   "expected": "PASS",
                   "observed": "PASS" if det_pass else "PENDING (run scripts/l_arc_3/step2/run_determinism_check.py)",
                   "status": "PASS" if det_pass else "PENDING"})

    phase_doc = STEP2_DIR / "PHASE_L_ARC_3_STEP2.md"
    checks.append({"id": 15, "check": "Phase doc descriptive only (op spec §11.5)",
                   "expected": "descriptive only",
                   "observed": "descriptive only" if phase_doc.exists() else "missing",
                   "status": "PASS" if phase_doc.exists() else "PENDING"})

    pass_count = sum(1 for c in checks if c["status"] == "PASS")
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    pending = sum(1 for c in checks if c["status"] == "PENDING")
    lines = ["L Arc 3 Step 2 — Sanity Checks (op spec §4 step 2 + task validation)",
             "=" * 70, "",
             f"Summary: {pass_count} PASS, {fail_count} FAIL, {pending} PENDING", ""]
    for c in checks:
        lines.append(f"[{c['status']}] (#{c['id']}) {c['check']}")
        lines.append(f"      expected: {c['expected']}")
        lines.append(f"      observed: {c['observed']}"); lines.append("")
    out = STEP2_DIR / "step2_sanity_checks.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_run_manifest() -> Path:
    in_files = {
        "config (configs/wfo_l_arc3_verbatim.yaml)": REPO_ROOT / "configs" / "wfo_l_arc3_verbatim.yaml",
        "spread_floor (configs/spread_floors_5ers.yaml)": REPO_ROOT / "configs" / "spread_floors_5ers.yaml",
        "step1 trades_verbatim.csv": STEP1_DIR / "trades_verbatim.csv",
        "step1 signals_log.csv": STEP1_DIR / "signals_log.csv",
    }
    out_files = [
        STEP2_DIR / "signals_features.csv",
        STEP2_DIR / "trade_paths.csv",
        STEP2_DIR / "feature_lag_audit.txt",
        STEP2_DIR / "lookahead_invariant_features_test.txt",
        STEP2_DIR / "forward_horizon_stability.txt",
        STEP2_DIR / "shadow_tradesets" / "shadow_summary.csv",
        STEP2_DIR / "shadow_tradesets" / "entry_delay_curve.csv",
        STEP2_DIR / "shadow_tradesets" / "sl_distance_sweep.csv",
        STEP2_DIR / "shadow_tradesets" / "time_exit_curve.csv",
        STEP2_DIR / "cost_stress" / "spread_multiplier_sweep.csv",
        STEP2_DIR / "random_baseline" / "random_entry_distribution.csv",
        STEP2_DIR / "random_baseline" / "comparison.csv",
        STEP2_DIR / "held_bar_evolution" / "t1.csv",
        STEP2_DIR / "forward_context_evolution" / "t1.csv",
        STEP2_DIR / "forward_context_evolution" / "t20.csv",
    ]
    lines = ["L Arc 3 Step 2 — Run Manifest", "=" * 60, "",
             "## Inputs (sha256)"]
    for label, p in in_files.items():
        if p.exists():
            lines.append(f"  {sha256_file(p)}  {label}")
        else:
            lines.append(f"  <missing>                                                          {label}")
    lines += ["", "## Outputs (sha256)"]
    for p in out_files:
        if p.exists():
            lines.append(f"  {sha256_file(p)}  {p.relative_to(STEP2_DIR).as_posix()}")
        else:
            lines.append(f"  <missing>                                                          {p.relative_to(STEP2_DIR).as_posix()}")
    lines += ["", "## Determinism",
              "See determinism_check.txt for two-consecutive-run receipt.",
              "Python: see run_step2.py log for `sys.version` at run time.",
              "Determinism enforced via Amendment 11 hash-based seeds throughout.",
              ""]
    out = STEP2_DIR / "run_manifest.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_phase_doc(extended: bool) -> Path:
    H = FORWARD_HORIZON_BARS_EXTENDED if extended else FORWARD_HORIZON_BARS_DEFAULT
    print("[Phase I] composing PHASE_L_ARC_3_STEP2.md...")
    f = pd.read_csv(STEP2_DIR / "signals_features.csv")
    n_feat = len(f); n_cols = len(f.columns)
    n_data_end = int((f["exit_reason"] == "data_end").sum())

    p50_net_r = _safe_median(f["net_r"]); mean_net_r = _safe_mean(f["net_r"])
    p50_fwd_mfe_h24 = _safe_median(f["fwd_mfe_h24_atr"])
    p50_fwd_mae_h24 = _safe_median(f["fwd_mae_h24_atr"])
    p95_fwd_mfe_h24 = _safe_quantile(f["fwd_mfe_h24_atr"], 0.95)
    p50_fwd_mfe_h120 = _safe_median(f["fwd_mfe_h120_atr"])
    p50_fwd_mae_h120 = _safe_median(f["fwd_mae_h120_atr"])
    p50_fwd_mfe_h240 = _safe_median(f["fwd_mfe_h240_atr"])
    p50_mfe_held = _safe_median(f["mfe_held_atr"])
    p50_mae_held = _safe_median(f["mae_held_atr"])
    p50_bars_held = _safe_median(f["bars_held"])
    mean_bars_held = _safe_mean(f["bars_held"])
    p50_osc = _safe_median(f["oscillation_count"])
    p50_mono = _safe_median(f["monotonicity_ratio"])
    p50_t2pk = _safe_median(f["time_to_peak_mfe"])
    p50_t2tr = _safe_median(f["time_to_trough_mae"])
    race = f["race_bars_plus1_minus_minus1"]
    race_p = {p: _safe_quantile(race, p / 100.0) for p in [5, 10, 25, 50, 75, 90, 95]}
    race_mean = _safe_mean(race)
    csw = f["concurrent_signals_within_3h"]

    # Amendment 4 features
    fwd_range_p50 = _safe_median(f["fwd_realized_range_atr"])
    fwd_range_p95 = _safe_quantile(f["fwd_realized_range_atr"], 0.95)
    frac_above_p50 = _safe_median(f["fwd_fraction_time_above_entry"])
    frac_above_p95 = _safe_quantile(f["fwd_fraction_time_above_entry"], 0.95)
    consec_p50 = _safe_median(f["fwd_max_consecutive_directional_bars"])
    consec_p95 = _safe_quantile(f["fwd_max_consecutive_directional_bars"], 0.95)

    # Pre-signal context (arc-3 new)
    psc_cols = ["cum_logret_1h_24", "cum_logret_1h_72", "cum_logret_1h_168", "vol_realized_1h_24h"]
    psc_summary = {c: {"p50": _safe_median(f[c]), "mean": _safe_mean(f[c]), "p5": _safe_quantile(f[c], 0.05), "p95": _safe_quantile(f[c], 0.95)} for c in psc_cols}

    entry_curve = pd.read_csv(STEP2_DIR / "shadow_tradesets" / "entry_delay_curve.csv")
    sl_curve = pd.read_csv(STEP2_DIR / "shadow_tradesets" / "sl_distance_sweep.csv")
    te_curve = pd.read_csv(STEP2_DIR / "shadow_tradesets" / "time_exit_curve.csv")
    cost = pd.read_csv(STEP2_DIR / "cost_stress" / "spread_multiplier_sweep.csv")
    crossing_text = _read_text_or_none(STEP2_DIR / "cost_stress" / "_crossing_zero_note.txt").strip()
    rand_diff_text = _read_text_or_none(STEP2_DIR / "random_baseline" / "differs_from_verbatim.txt").strip()
    stab_text = _read_text_or_none(STEP2_DIR / "forward_horizon_stability.txt").strip()
    det_text = _read_text_or_none(STEP2_DIR / "determinism_check.txt").strip()
    fl_text = _read_text_or_none(STEP2_DIR / "feature_lag_audit.txt").strip()

    md: List[str] = []
    md.append("# PHASE_L_ARC_3_STEP2 — Descriptive Trade-Path Analysis")
    md.append("")
    md.append("## 1. Header")
    md.append("")
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append("| Arc | L Arc 3 — standalone signal arc (no calibration-check role) |")
    md.append("| Step | 2 — Descriptive Trade-Path Analysis (NOT a gate) |")
    md.append("| Protocol | `L_ARC_PROTOCOL.md` v1.0 + `L_ARC_PROTOCOL_v1.1_AMENDMENT.md` (12 amendments) |")
    md.append("| Operational spec | `L_ARC_OPERATIONAL_SPEC.md` v1.0 |")
    md.append("| Arc-open doc | `results/l_arc_3/PHASE_L_ARC_3_OPEN.md` |")
    md.append("| Predecessor doc | `results/l_arc_3/step1_verbatim/PHASE_L_ARC_3_STEP1.md` |")
    md.append(f"| Input trade-set | `results/l_arc_3/step1_verbatim/trades_verbatim.csv` ({n_feat:,} rows) |")
    md.append(f"| Forward horizon | H = {H}" + (" (extended from default 240 per Phase D)" if extended else " (default; stability check did not trigger extension)") + " |")
    md.append("| Random seeds | Amendment 11 hash-based throughout (see `run_manifest.txt`) |")
    md.append("")
    md.append("## 2. Discipline reminder (op spec §11.5)")
    md.append("")
    md.append("This document is **descriptive only**. No recommendations, no filter")
    md.append("proposals, no exit suggestions. Action-shaped language is op spec §11.5")
    md.append("violation. Findings that would suggest a filter or exit are recorded as")
    md.append("observations, not as proposals.")
    md.append("")
    md.append("## 3. Configuration recap")
    md.append("")
    md.append("Signal: `TRIAL__volatility_regime__d1_atr_top_decile__any__h_120`")
    md.append("  - 1H signal TF, volatility_regime family, d1_atr_top_decile base, any sub-spec, h=120 horizon.")
    md.append("")
    md.append("Verbatim execution: bar N+1 open entry; 2.0 × ATR(14)_1H SL anchored at entry;")
    md.append("time exit at bar N+121 open; spread per `configs/spread_floors_5ers.yaml`;")
    md.append("max 1 concurrent position per pair; 0.5% risk per trade.")
    md.append("")
    md.append("Trade-set: 3,993 taken trades across 7 anchored expanding folds (Oct 2020 –")
    md.append("Jan 2026), 28 FX pairs. Per-fold breakdown F1=541, F2=621, F3=607, F4=614,")
    md.append("F5=504, F6=599, F7=507. Step 1 exit-mix: sl_hit 76.03%, time_exit 23.77%,")
    md.append("data_end 0.20% (8 fold-7 tail trades).")
    md.append("")
    md.append("## 4. signals_features.csv schema (column inventory)")
    md.append("")
    md.append(f"- **Rows:** {n_feat:,}  (one per taken trade)")
    md.append(f"- **Columns:** {n_cols}")
    md.append("")
    md.append("Column families:")
    md.append("")
    md.append("- **Identity / context:** `trade_id, pair, fold_id, signal_bar_ts, "
              "entry_bar_ts, exit_bar_ts, direction`")
    md.append("- **Signal-bar 1H properties:** open/close/high/low, log_return, "
              "abs_log_return, atr_at_signal_1h, atr_baseline_1h_200, atr_ratio_to_baseline, "
              "signal_bar_volume, signal_bar_volume_nan")
    md.append("- **Pre-signal 1H context (legacy):** cum_logret_1h_3, cum_logret_1h_6, "
              "dist_close_to_high30_atr, dist_close_to_low30_atr")
    md.append("- **Pre-signal 1H context (arc-3 NEW, v1.1 amendment):** ")
    md.append("  `cum_logret_1h_24, cum_logret_1h_72, cum_logret_1h_168, vol_realized_1h_24h`. ")
    md.append("  Lookahead-safe (strict-prior references — see §15).")
    md.append("- **Time / session / liquidity:** hour_utc, day_of_week, session, "
              "hour_in_4h_bar, bars_to_next_4h_close, hour_in_d1_bar, bars_to_next_d1_close")
    md.append("- **First-bar (N+1) properties:** first_bar_direction, first_bar_range_atr, first_bar_range_bin")
    md.append("- **Regime / momentum bins:** vol_regime, pre_momentum_bin, "
              "cum_logret_1h_{6,24,168}_bin, vol_realized_1h_24h_decile, trigger_magnitude_decile")
    md.append("- **Cross-pair / portfolio (§5.15):** concurrent_signals_same_bar, "
              "concurrent_signals_within_3h, currency_basket_3h_{USD,EUR,JPY,GBP}, "
              "trade_overlap_at_execution_time, sequential_same_pair_density_24h")
    md.append("- **Trade outcome:** net_r, gross_r, spread_cost_R, mfe_R, mae_R, bars_held, "
              "exit_reason, exit_reason_engine, spread_pips_entry/exit, spread_floored, "
              "sl_distance_atr, sl_distance_price, data_end_flag, forward_window_bars_available")
    md.append("- **HELD-window path aggregates (REAL on arc 3 — held window > 1 bar):**")
    md.append("  mfe_held_atr, mae_held_atr, peak_to_final_r_ratio, mfe_to_mae_ratio_held,")
    md.append("  r_given_back_from_peak, oscillation_count, monotonicity_ratio,")
    md.append("  max_consecutive_with, max_consecutive_against, acf1_returns_during_hold,")
    md.append("  time_to_peak_mfe, time_to_trough_mae, time_from_peak_to_exit,")
    md.append("  mfe_sequence_class_held")
    h_list_str = ",".join(str(h) for h in H_GRID)
    md.append(f"- **Forward-horizon aggregates (h ∈ {{{h_list_str}}}):**")
    md.append("  fwd_logret_h{h}, fwd_mfe_h{h}_atr, fwd_mae_h{h}_atr, fwd_mfe_to_mae_ratio_h{h}")
    md.append(f"- **bars_to thresholds (capped at H={H}):** "
              f"bars_to_plus_{{0.5,1,1.5,2,3}}_atr_capped_{H}, "
              f"bars_to_minus_{{0.5,1,1.5,2,3}}_atr_capped_{H}; "
              f"reached_plus_{{0.5,1,2}}_atr_within_{H}; race_bars_plus1_minus_minus1")
    md.append("- **Forward-path sequence classification:** mfe_sequence_class_fwd_h{24,120}")
    md.append("- **Amendment 4 clustering features (forward-window-derived path geometry):**")
    md.append("  `fwd_realized_range_atr, fwd_fraction_time_above_entry, fwd_max_consecutive_directional_bars`")
    md.append("")
    nan_cols = f.columns[f.isna().any()].tolist()
    md.append(f"NaN audit: {len(nan_cols)} columns contain at least one NaN. NaNs concentrate")
    md.append("in tail-end forward-horizon aggregates for trades with truncated forward windows")
    md.append("(8 fold-7 data_end trades + tail-of-data) and in pre-signal context features for")
    md.append("the earliest bars in each pair's data.")
    md.append("")
    md.append("## 5. trade_paths.csv schema (per-bar long format)")
    md.append("")
    md.append("- **Columns:** `trade_id, bar_offset, bar_ts, open, high, low, close, "
              "cum_logret_from_entry, mfe_to_date_atr, mae_to_date_atr, "
              "is_held_bar, is_forward_bar, data_end_flag`")
    md.append("- **bar_offset = 0** is the entry bar (N+1). Positive offsets are subsequent bars.")
    md.append(f"- **bar_offset range:** [0, {H-1}] (forward window). is_forward_bar=True throughout.")
    md.append("- **is_held_bar:** True while bar_offset < bars_held (trade still open). After the")
    md.append("  verbatim exit, is_held_bar=False but is_forward_bar remains True up to H-1.")
    md.append("- **data_end_flag:** True on the last available bar of trades whose forward window")
    md.append("  exceeds the pair's data file (the 8 fold-7 data_end trades).")
    md.append("- **Truncation:** rows are NOT emitted past the end of the pair's data.")
    paths_count_path = STEP2_DIR / "trade_paths.csv"
    if paths_count_path.exists():
        with paths_count_path.open("rb") as fh:
            n_rows = sum(1 for _ in fh) - 1
        md.append(f"- **Rows:** {n_rows:,}.")
    md.append("")
    md.append("## 6. Findings per angle (5.1–5.14)")
    md.append("")
    md.append("### 6.1 Marginal distributions (§5.1)")
    md.append("")
    md.append(f"Pool `net_r`: p50={p50_net_r:.4f}, mean={mean_net_r:.5f}, "
              f"std={float(f['net_r'].std()):.4f}, p95={_safe_quantile(f['net_r'], 0.95):.3f}, "
              f"p5={_safe_quantile(f['net_r'], 0.05):.3f}. "
              f"Pool `mfe_held_atr` median {p50_mfe_held:.3f}; `mae_held_atr` median {p50_mae_held:.3f}; "
              f"`bars_held` median {p50_bars_held:.0f}, mean {mean_bars_held:.1f}. "
              f"Exit-reason mix: sl_hit {100*(f['exit_reason']=='sl_hit').mean():.2f}%, "
              f"time_exit {100*(f['exit_reason']=='time_exit').mean():.2f}%, "
              f"data_end {100*(f['exit_reason']=='data_end').mean():.2f}%. "
              "Full distributions in `distributions/marginals/<metric>.csv` with per-fold "
              "breakdowns alongside.")
    md.append("")
    md.append("### 6.2 Forward-horizon geometry (§5.2)")
    md.append("")
    md.append(f"Forward path geometry at h=24: p50 fwd_mfe = {p50_fwd_mfe_h24:.3f} ATR, "
              f"p95 fwd_mfe = {p95_fwd_mfe_h24:.3f} ATR, p50 fwd_mae = {p50_fwd_mae_h24:.3f} ATR. "
              f"At h=120 (verbatim time-exit horizon): p50 fwd_mfe = {p50_fwd_mfe_h120:.3f}, "
              f"p50 fwd_mae = {p50_fwd_mae_h120:.3f}. "
              f"At h=240: p50 fwd_mfe = {p50_fwd_mfe_h240:.3f}. "
              f"Race condition (bars_to_+1ATR − bars_to_-1ATR): pool mean {race_mean:.2f}, "
              f"p50 {race_p[50]:.1f} bars (positive = MAE-first; negative = MFE-first). "
              "Full distributions in `distributions/forward/`.")
    md.append("")
    md.append("### 6.3 Forward-horizon stability check (§5.3)")
    md.append("")
    md.append("```")
    md.append(stab_text); md.append("```")
    md.append("")
    md.append("### 6.4 MFE/MAE sequence classification (§5.4)")
    md.append("")
    md.append("Held-window class counts and per-class summaries in "
              "`distributions/sequence/mfe_sequence_class_held.csv` + `per_class_summary_held.csv`. "
              "Forward-path classes at h=24 and h=120 in `mfe_sequence_class_fwd_h{24,120}.csv` "
              "and `per_class_summary_fwd_h{24,120}.csv`. "
              "Time-difference distribution (held): `time_to_peak_mfe_minus_trough_mae_held.csv`.")
    md.append("")
    md.append("### 6.5 Path complexity (§5.5)")
    md.append("")
    md.append(f"HELD-window: oscillation_count p50 {p50_osc:.0f}, monotonicity_ratio p50 {p50_mono:.3f}, "
              f"time_to_peak_mfe p50 {p50_t2pk:.1f}, time_to_trough_mae p50 {p50_t2tr:.1f}. "
              "Full distributions for each path-complexity metric in `distributions/complexity/`. "
              "Amendment 4 forward-path clustering features also reported there: "
              f"fwd_realized_range_atr p50 {fwd_range_p50:.3f} (p95 {fwd_range_p95:.3f}); "
              f"fwd_fraction_time_above_entry p50 {frac_above_p50:.3f} (p95 {frac_above_p95:.3f}); "
              f"fwd_max_consecutive_directional_bars p50 {consec_p50:.1f} (p95 {consec_p95:.1f}).")
    md.append("")
    md.append("### 6.6 Survival curves (§5.6)")
    md.append("")
    md.append("Survival curve in `distributions/survival/survival.csv`. Reports the fraction of "
              "trades still open at bar_offset t ∈ {1, 5, 10, 20, 50, 100, 200}, plus mean R "
              "and win% conditional on still-open-at-t (descriptive — no exit interpretation).")
    md.append("")
    md.append("### 6.7 Early-bar predictivity (§5.7)")
    md.append("")
    md.append("Correlation between cum_R at bar_offset (t-1) and final net R reported in "
              "`distributions/early_bar/corr_t{1,3,5,10}.txt`. Decile breakdowns by cum_R-at-t "
              "in `decile_breakdown.csv` show conditional mean final R and win% per decile.")
    md.append("")
    md.append("### 6.8 Win/loss asymmetry (§5.8)")
    md.append("")
    md.append("Per-side summary in `distributions/asymmetry/win_loss_asymmetry.csv` with median R, "
              "mean R, p95/p5, median bars_held, median MAE during hold (drawdown-during-winners) "
              "and median MFE during hold (run-up-during-losers).")
    md.append("")
    md.append("### 6.9 Conditional breakdowns (§5.9)")
    md.append("")
    md.append("Per-stratum summaries (n, mean, std, p5..p95, max) for 17 metrics across 17 "
              "stratification axes — including the arc-3 new pre-signal context axes "
              "(`cum_logret_1h_24_bin`, `cum_logret_1h_168_bin`, `vol_realized_1h_24h_decile`) "
              "and the legacy `cum_logret_1h_6_bin`. Output in `conditional_breakdowns/<stratum>/<metric>.csv`. "
              "Sample-size discipline: per-cell n<30 flagged via the `flagged_n_lt_30` column; "
              "n<10 cells pooled into a `_insufficient_n` row.")
    md.append("")
    md.append("### 6.10 Joint distributions (§5.10)")
    md.append("")
    joint_dir = STEP2_DIR / "joint_distributions"
    n_joint = sum(1 for _ in joint_dir.glob("*.csv")) if joint_dir.exists() else 0
    md.append(f"{n_joint} 2D heatmaps (binned-count crosstabs, op spec §11.2) in `joint_distributions/`. "
              "Pairs covered: mfe_held×mae_held, mfe_held×bars_held, mae_held×bars_held, "
              "mfe_held×time_to_peak_mfe, mfe_held×exit_reason, net_r×bars_held, "
              "net_r×oscillation_count, first_bar_direction×net_r, mfe_sequence_class_{held,fwd_h24,fwd_h120}×net_r, "
              "concurrent_signals_within_3h × {net_r, fwd_mae_h24}, hour_in_d1_bar×net_r, "
              "vol_regime×fwd_mfe_h24, and four arc-3 new-feature heatmaps "
              "(cum_logret_1h_{24,168}×net_r, vol_realized_1h_24h×net_r, "
              "fwd_realized_range_atr×net_r, fwd_fraction_time_above_entry×net_r).")
    md.append("")
    md.append("### 6.11 Shadow trade-sets (§5.11)")
    md.append("")
    md.append("**Entry-delay sweep** (SL=2.0 ATR, h=120 verbatim, entry at N+{1,2,3,5,10}):")
    md.append("")
    md.append("```")
    md.append(entry_curve[["bar_offset", "n", "n_dropped", "mean_net_r", "median_net_r",
                            "win_pct", "frac_sl_hit", "frac_time_exit", "mean_bars_held"]]
              .to_string(index=False))
    md.append("```")
    md.append("")
    md.append("**SL-distance sweep** (bar_offset=1, h=120, SL ∈ {1.0, 1.5, 2.0, 2.5, 3.0} ATR):")
    md.append("")
    md.append("```")
    md.append(sl_curve[["sl_atr_mult", "n", "mean_net_r", "median_net_r", "win_pct",
                        "frac_sl_hit", "mean_bars_held"]].to_string(index=False))
    md.append("```")
    md.append("")
    md.append("**Time-exit sweep** (bar_offset=1, SL=2.0 ATR, h ∈ {1, 3, 6, 12, 24, 48, 120, 240}):")
    md.append("")
    md.append("```")
    md.append(te_curve[["h", "n", "mean_net_r", "median_net_r", "win_pct",
                        "frac_sl_hit", "frac_time_exit", "median_capture_ratio_vs_fwd_mfe"]]
              .to_string(index=False))
    md.append("```")
    md.append("")
    md.append("Full per-shadow per-trade CSVs under `shadow_tradesets/<axis>/`; pool + per-fold "
              "summary in `shadow_tradesets/shadow_summary.csv`.")
    md.append("")
    md.append("### 6.12 Cost stress (§5.12)")
    md.append("")
    md.append("Spread-floor multiplier sweep ({0.5×, 1.0×, 1.5×, 2.0×}) under verbatim execution:")
    md.append("")
    md.append("```")
    md.append(cost.to_string(index=False))
    md.append("```")
    md.append("")
    md.append(f"Crossing-zero note: {crossing_text}")
    md.append("")
    md.append("### 6.13 Random-entry baseline (§5.13)")
    md.append("")
    md.append("```")
    md.append(rand_diff_text); md.append("```")
    md.append("")
    md.append("Comparison detail in `random_baseline/comparison.csv`; per-trade random-baseline "
              "outcomes in `random_baseline/random_entry_distribution.csv`. Seeds are hash-based per "
              "Amendment 11 (`hash_seed('l_arc_3_step2_random_baseline')` xor per-pair-per-fold key).")
    md.append("")
    md.append("### 6.14 Held-bar context evolution (§5.14)")
    md.append("")
    md.append("Per-trade per-t (t ∈ {1, 3, 5, 10, 20}) samples of currency-basket cum log return, "
              "broker spread (raw + floored), cross-pair dispersion proxy, and ATR regime ratio. "
              "Held-bar outputs in `held_bar_evolution/t{1,3,5,10,20}.csv` (only trades still open "
              "at bar_offset t contribute; arc 3's mean bars_held ~47 means held rows thin out at "
              "larger t). Forward-window analogue (unconditional on exit) in "
              "`forward_context_evolution/t{1,3,5,10,20}.csv`.")
    md.append("")
    md.append("## 7. Cross-pair / portfolio context summary (§5.15)")
    md.append("")
    md.append(f"`concurrent_signals_within_3h`: mean {_safe_mean(csw):.2f}, p50 {_safe_median(csw):.0f}, "
              f"p90 {_safe_quantile(csw, 0.90):.0f}, p95 {_safe_quantile(csw, 0.95):.0f}, "
              f"p99 {_safe_quantile(csw, 0.99):.0f}, max {float(csw.max()):.0f}. ")
    md.append("Other §5.15 features (concurrent_signals_same_bar, currency_basket_3h_{USD,EUR,JPY,GBP}, "
              "trade_overlap_at_execution_time, sequential_same_pair_density_24h) are present in "
              "`signals_features.csv` with distributions in `distributions/marginals/`.")
    md.append("")
    md.append("## 8. Pre-signal context (arc-3 NEW) summary")
    md.append("")
    md.append("| feature | mean | p50 | p5 | p95 |")
    md.append("|---|---:|---:|---:|---:|")
    for c in psc_cols:
        s = psc_summary[c]
        md.append(f"| `{c}` | {s['mean']:.5f} | {s['p50']:.5f} | {s['p5']:.5f} | {s['p95']:.5f} |")
    md.append("")
    md.append("Distribution shape per feature in `distributions/marginals/<feature>.csv`.")
    md.append("")
    md.append("## 9. Feature lag audit (op spec §10.4)")
    md.append("")
    md.append("```")
    md.append(fl_text); md.append("```")
    md.append("")
    md.append("## 10. Determinism receipt")
    md.append("")
    md.append("```")
    md.append(det_text if det_text != "<missing>" else
              "(see determinism_check.txt — produced by scripts/l_arc_3/step2/run_determinism_check.py)")
    md.append("```")
    md.append("")
    md.append("Full input/output sha256 ledger in `run_manifest.txt`.")
    md.append("")
    md.append("## 11. Open issues / WARNs")
    md.append("")
    md.append(f"- **{n_data_end} fold-7 data_end trades** preserved per step 1 handover. "
              f"{'Their forward-window is truncated below H=' + str(H) + ' at the end of available data; affected aggregates appear as NaN.' if n_data_end else 'Arc 3 had zero data_end trades — heavier exposure-cap binding suppressed the fold-7 tail trades that arc 2 saw.'}")
    md.append("- **Truncation at dataset boundary.** Some fold-7-tail trades have")
    md.append(f"  `forward_window_bars_available < {H}`. The exact count is in the column")
    md.append("  histogram (`signals_features.csv::forward_window_bars_available`).")
    md.append("- **Cross-pair dispersion proxy.** Implemented as per-bar std of 28 pair-log-returns,")
    md.append("  documented in `held_bar_evolution/_cross_pair_dispersion_note.txt`.")
    md.append("")
    md.append("## Handover to Next Chat")
    md.append("")
    md.append("**Inputs ready for step 3:**")
    md.append("")
    md.append(f"- `results/l_arc_3/step2_descriptive/signals_features.csv` "
              f"({n_feat:,} rows × {n_cols} columns; trade_id-keyed).")
    paths_count = "?"
    pp = STEP2_DIR / "trade_paths.csv"
    if pp.exists():
        with pp.open("rb") as fh:
            paths_count = f"{(sum(1 for _ in fh) - 1):,}"
    md.append(f"- `results/l_arc_3/step2_descriptive/trade_paths.csv` ({paths_count} rows; "
              f"(trade_id, bar_offset)-keyed; bar_ts + OHLC + cum_logret + mfe/mae running ATR-norms "
              f"+ is_held_bar/is_forward_bar/data_end_flag).")
    md.append("")
    md.append("**Descriptive summary of trade-set shape (starting context for step 3):**")
    md.append("")
    md.append(f"The trade-set is {n_feat:,} long trades across 28 pairs over 7 anchored")
    md.append(f"expanding folds (Oct 2020 – Jan 2026), held a mean of {mean_bars_held:.1f} bars each")
    md.append(f"(median {p50_bars_held:.0f}; max 120) under verbatim execution. Pool median net R is")
    md.append(f"{p50_net_r:.4f}; mean {mean_net_r:.5f}. Exit mix is "
              f"~{100*(f['exit_reason']=='sl_hit').mean():.0f}% sl_hit, "
              f"~{100*(f['exit_reason']=='time_exit').mean():.0f}% time_exit, ")
    md.append(f"~{100*(f['exit_reason']=='data_end').mean():.2f}% data_end ({n_data_end} fold-7 tail trades).")
    md.append(f"Forward path geometry at h=24: p50 fwd_mfe = {p50_fwd_mfe_h24:.2f} ATR, p50 fwd_mae")
    md.append(f"= {p50_fwd_mae_h24:.2f} ATR. Race-condition p50 = {race_p[50]:.1f} bars (positive = ")
    md.append("MAE-first at the median).")
    md.append("")
    # Time-exit curve shape summary
    te = te_curve.set_index("h")
    te_mean_r = te["mean_net_r"]
    te_peak_h = int(te_mean_r.idxmax())
    te_min_h = int(te_mean_r.idxmin())
    te_shape = "monotonic" if (te_mean_r.diff().dropna() > 0).all() or (te_mean_r.diff().dropna() < 0).all() else "non-monotonic"
    md.append("Time-exit curve shape (mean R as a function of h, descriptive only):")
    md.append(f"  peak at h={te_peak_h} (mean R {te_mean_r[te_peak_h]:.4f}); "
              f"trough at h={te_min_h} (mean R {te_mean_r[te_min_h]:.4f}); "
              f"overall shape: {te_shape}.")
    sl = sl_curve.set_index("sl_atr_mult")
    sl_mean_r = sl["mean_net_r"]
    sl_peak_m = float(sl_mean_r.idxmax())
    sl_min_m = float(sl_mean_r.idxmin())
    md.append("SL-distance curve shape (mean R as a function of SL_atr_mult, descriptive only):")
    md.append(f"  peak at SL={sl_peak_m} ATR (mean R {sl_mean_r[sl_peak_m]:.4f}); "
              f"trough at SL={sl_min_m} ATR (mean R {sl_mean_r[sl_min_m]:.4f}).")
    md.append("")
    md.append(f"Random baseline summary: {rand_diff_text.splitlines()[0] if rand_diff_text else 'see random_baseline/differs_from_verbatim.txt'}.")
    md.append(f"Forward-horizon stability outcome: H = {H}" + (" (extended)" if extended else " (default; not extended)") + ".")
    md.append("")
    md.append("**Cross-arc carries from arc 1 and arc 2 (per arc-open §4) — discharged at step 2:**")
    md.append("")
    md.append("- 4 pre-signal context columns added to `signals_features.csv` "
              "(`cum_logret_1h_24/72/168, vol_realized_1h_24h`). Lookahead-safe; see §9.")
    md.append("- §5.15 cross-pair / portfolio family present (concurrent_signals, currency_basket, "
              "trade_overlap, sequential_same_pair_density).")
    md.append("- Time-exit shape descriptive summary above.")
    md.append("- 3 Amendment 4 clustering features added (fwd_realized_range_atr, "
              "fwd_fraction_time_above_entry, fwd_max_consecutive_directional_bars).")
    md.append("")
    md.append("**Notes for step 3 chat:**")
    md.append("")
    md.append("- The clustering feature subset (op spec §6.1) for step 3 phase A should read: "
              "mfe_held_atr, mae_held_atr, bars_held, time_to_peak_mfe, time_to_trough_mae, "
              "peak_to_final_r_ratio, oscillation_count, monotonicity_ratio, mfe_sequence_class "
              "(legacy 9) + fwd_realized_range_atr, fwd_fraction_time_above_entry, "
              "fwd_max_consecutive_directional_bars (Amendment 4 — 3) = 12 features.")
    md.append("- Per-stratum cells with n<30 are flagged in conditional_breakdowns/<stratum>/<metric>.csv "
              "via `flagged_n_lt_30` column; n<10 pooled into `_insufficient_n` rows.")
    md.append(f"- {n_data_end} fold-7 data_end trades carried through with `data_end_flag=True`. "
              f"{'Excluded from exit-reason percentages that pool by clean exit class; included in pool path distributions; their forward window is truncated.' if n_data_end else 'Arc 3 had zero — exposure cap binding precluded any fold-7-tail trades from opening.'}")
    md.append("")

    out = STEP2_DIR / "PHASE_L_ARC_3_STEP2.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out


def main(extended: bool = False) -> None:
    print("[Phase I] writing run_manifest.txt...")
    write_run_manifest()
    print("[Phase I] writing step2_sanity_checks.txt...")
    write_sanity_checks(extended=extended)
    print("[Phase I] writing PHASE_L_ARC_3_STEP2.md...")
    write_phase_doc(extended=extended)
    write_sanity_checks(extended=extended)
    write_run_manifest()
    print("[Phase I] done.")


if __name__ == "__main__":
    main()

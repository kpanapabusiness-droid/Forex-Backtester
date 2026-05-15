"""CC-1 winner-path-after-peak diagnostic for L Arc 2 step 5 delayed_entry_t_gb.

Lives under scripts/l_arc_2/step4/ alongside the CC-0 script for the same
reason: it re-uses step 4's data loaders and conventions from
_simulator.py / _data.py / _common.py.

Produces under results/l_arc_2/step5_recharacterisation/delayed_entry_t_gb/:
  - winner_paths_bar_by_bar.csv   (per-trade bar-by-bar held R sequence,
                                    delayed-entry-indexed)
  - first_cross_detail.csv         (per-trade × per-threshold first-cross
                                    + fall-back records)
  - winner_path_after_peak_f2_f5.csv  (F2-F5 summary table)
  - winner_path_after_peak_f6_f7.csv  (F6+F7 summary table)
  - comparison_f2_f5_vs_f6_f7.csv     (side-by-side comparison table)
  - run_manifest.json              (predictor_source_pin_verification +
                                    input/output sha256s + audit-trail)

Conventions (read from source, NOT trusted from dispatch hints):
  - Entry price: trade_paths[trade_id, bar_offset=t_star].open
    (per _simulator.py:241-242 — verbatim "entry-at-OPEN" convention;
    the config.yaml action label `enter_at_bar_N_plus_1_plus_t_close`
    is a misnomer relative to the actual code).
  - ATR_at_entry: atr_at_signal_1h × (atr_regime_ratio_t1 /
    atr_ratio_to_baseline) — per _simulator.py:263-270.
  - SL distance: 2.0 × ATR_at_entry.
  - Time exit: bar_offset=120 OPEN if not SL-out first.
  - Winner: net_r > 0 in trades_post_mechanism CSV (always = time_exit
    trades; SL trades return gross_r=-1, always loss).
  - Held R indexed FROM delayed entry price (not baseline signal-bar
    price). Step 4 tier A used baseline-indexed mfe_to_date_atr /
    mae_to_date_atr — re-derived numbers here will differ; audit-trail
    table compares both.

Two top-level entries:
  - main() → produces all five CSVs + manifest; reports sha256 map
  - lookahead_invariant_test() → execution-side lookahead test
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C
from . import _data as D

REPO = C.REPO
STEP4_OUT = REPO / "results" / "l_arc_2" / "step4" / "delayed_entry_t_gb"
STEP5_OUT = REPO / "results" / "l_arc_2" / "step5_recharacterisation" / "delayed_entry_t_gb"
T_STAR = 1
THRESHOLDS = (0.5, 1.0, 1.5, 2.0)

# Reproducibility pins (same pattern as CC-0 script).
PREDICTOR_PIN_COMMIT = "fe345be58a8517189e898d0618266f083bda900c"
CC0_SCRIPT_PIN_COMMIT = "86215fc"  # short hash; we resolve to full at runtime
PINNED_SCRIPT_FILES = (
    "scripts/l_arc_2/step4/_predictor.py",
    "scripts/l_arc_2/step4/_common.py",
    "scripts/l_arc_2/step4/_data.py",
    "scripts/l_arc_2/step4/_actions.py",
    "scripts/l_arc_2/step4/_simulator.py",
    "scripts/l_arc_2/step4/run_step4.py",
    "scripts/l_arc_2/step4/_cc0_materialise_f2_f5_delayed_entry.py",
)
CC0_PINNED_FILE = "scripts/l_arc_2/step4/_cc0_materialise_f2_f5_delayed_entry.py"


# ============================================================
# Hashing / pin verification
# ============================================================

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_head() -> str:
    out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO))
    return out.decode().strip()


def _git_blob_at_commit(commit: str, path: str) -> str:
    out = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"], cwd=str(REPO),
    )
    return out.decode().strip()


def _verify_predictor_source_pin() -> dict:
    """Same pattern as CC-0's pin verification: confirm every pinned
    step 4 script file has byte-identical blob at HEAD vs the pinned
    predictor commit. For the CC-0 script file, compare against the
    CC-0 pin commit instead.
    """
    head = _git_commit_head()
    per_file = {}
    all_match = True
    for path in PINNED_SCRIPT_FILES:
        head_blob = _git_blob_at_commit(head, path)
        pin_commit = CC0_SCRIPT_PIN_COMMIT if path == CC0_PINNED_FILE else PREDICTOR_PIN_COMMIT
        pin_blob = _git_blob_at_commit(pin_commit, path)
        match = head_blob == pin_blob
        per_file[path] = {
            "pin_commit": pin_commit,
            "blob_at_pin_commit": pin_blob,
            "blob_at_head": head_blob,
            "match": match,
        }
        if not match:
            all_match = False
    return {
        "predictor_pin_commit": PREDICTOR_PIN_COMMIT,
        "cc0_pin_commit": CC0_SCRIPT_PIN_COMMIT,
        "head_commit": head,
        "all_pinned_files_blob_match": all_match,
        "per_file": per_file,
    }


def _versions() -> dict:
    import sklearn
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


# ============================================================
# Per-trade entry / ATR / sequence computation
# ============================================================

def _compute_entry_and_atr(trade_ids: np.ndarray,
                           signals: pd.DataFrame,
                           paths_long: pd.DataFrame,
                           held_ctx_t: pd.DataFrame,
                           t: int = T_STAR) -> pd.DataFrame:
    """Mirror _simulator.simulate_delayed_entry's entry-price + ATR-at-t
    computation. Returns DataFrame indexed by trade_id with columns:
    entry_price, atr_at_entry.

    Per _simulator.py:241-270.
    """
    # Entry price = bar_offset=t open
    entry_row = paths_long[
        (paths_long["trade_id"].isin(trade_ids)) &
        (paths_long["bar_offset"] == t)
    ].set_index("trade_id")
    entry_price = entry_row["open"].reindex(trade_ids).values

    # ATR_at_entry = atr_at_signal × (atr_regime_ratio_t / atr_ratio_to_baseline)
    static = signals.set_index("trade_id").loc[trade_ids,
                                                ["atr_at_signal_1h",
                                                 "atr_ratio_to_baseline"]]
    atr_sig = static["atr_at_signal_1h"].values
    atr_ratio_signal = static["atr_ratio_to_baseline"].values

    held_t = held_ctx_t.set_index("trade_id").reindex(trade_ids)
    regime_col = f"atr_regime_ratio_t{t}"
    regime_ratio_at_t = held_t[regime_col].values

    scale = np.where(
        (atr_ratio_signal > 0) & ~np.isnan(regime_ratio_at_t),
        regime_ratio_at_t / atr_ratio_signal,
        1.0,
    )
    atr_at_entry = atr_sig * scale

    return pd.DataFrame({
        "trade_id": trade_ids,
        "entry_price": entry_price,
        "atr_at_entry": atr_at_entry,
    })


def _compute_bar_by_bar_sequences(winners: pd.DataFrame,
                                    paths_long: pd.DataFrame,
                                    entry_atr_df: pd.DataFrame,
                                    source_label: str) -> pd.DataFrame:
    """For each winner, build the bar-by-bar held R sequence from
    delayed-entry bar (bar_offset=T_STAR=1) through exit bar (action_bar
    in trades_post_mechanism CSV — either time_exit_delayed at 120 or
    sl_hit_post_delayed_entry at earlier bar).

    Returns long DataFrame with columns:
        trade_id, fold, source_set, bar_offset, bar_high, bar_low,
        bar_close, max_held_R, min_held_R
    """
    ea = entry_atr_df.set_index("trade_id")
    trade_meta = winners.set_index("trade_id")[["fold", "action_bar"]]

    pieces = []
    for tid in winners["trade_id"].values:
        entry_price = float(ea.at[tid, "entry_price"])
        atr_at_entry = float(ea.at[tid, "atr_at_entry"])
        if not np.isfinite(entry_price) or not np.isfinite(atr_at_entry) or atr_at_entry <= 0:
            continue
        denom = 2.0 * atr_at_entry  # SL distance in price units

        exit_bar = int(trade_meta.at[tid, "action_bar"])
        # Bars from delayed-entry (bar_offset = T_STAR = 1) through exit
        # inclusive. Trade is "open" during the entry bar.
        trade_rows = paths_long[
            (paths_long["trade_id"] == tid) &
            (paths_long["bar_offset"] >= T_STAR) &
            (paths_long["bar_offset"] <= exit_bar)
        ].copy()
        trade_rows = trade_rows.sort_values("bar_offset")

        bar_high = trade_rows["high"].values
        bar_low = trade_rows["low"].values
        bar_close = trade_rows["close"].values
        max_held_R = (bar_high - entry_price) / denom
        min_held_R = (bar_low - entry_price) / denom

        piece = pd.DataFrame({
            "trade_id": tid,
            "fold": int(trade_meta.at[tid, "fold"]),
            "source_set": source_label,
            "bar_offset": trade_rows["bar_offset"].values.astype(int),
            "bar_high": bar_high,
            "bar_low": bar_low,
            "bar_close": bar_close,
            "max_held_R": max_held_R,
            "min_held_R": min_held_R,
        })
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame(columns=[
            "trade_id", "fold", "source_set", "bar_offset",
            "bar_high", "bar_low", "bar_close", "max_held_R", "min_held_R",
        ])
    return pd.concat(pieces, ignore_index=True)


def _first_cross_detail(bar_by_bar: pd.DataFrame) -> pd.DataFrame:
    """For each (trade × threshold), find first-cross bar and min held R
    between first-cross (inclusive) and the trade's last bar (inclusive).
    """
    rows = []
    for (tid, fold, src), grp in bar_by_bar.groupby(
        ["trade_id", "fold", "source_set"], sort=True
    ):
        grp = grp.sort_values("bar_offset").reset_index(drop=True)
        for X in THRESHOLDS:
            cross_mask = grp["max_held_R"].values >= X
            if not cross_mask.any():
                rows.append({
                    "trade_id": int(tid),
                    "fold": int(fold),
                    "source_set": src,
                    "threshold": float(X),
                    "first_cross_bar": pd.NA,
                    "min_held_R_after_cross": pd.NA,
                    "fell_back_to_BE": pd.NA,
                    "excluded_no_cross": True,
                })
                continue
            first_idx = int(np.argmax(cross_mask))
            first_cross_bar = int(grp.loc[first_idx, "bar_offset"])
            tail_min = float(grp.loc[first_idx:, "min_held_R"].min())
            rows.append({
                "trade_id": int(tid),
                "fold": int(fold),
                "source_set": src,
                "threshold": float(X),
                "first_cross_bar": first_cross_bar,
                "min_held_R_after_cross": tail_min,
                "fell_back_to_BE": bool(tail_min < 0),
                "excluded_no_cross": False,
            })
    return pd.DataFrame(rows)


def _summary_table(first_cross: pd.DataFrame, source_label: str,
                   total_winners: int) -> pd.DataFrame:
    """Per-threshold summary for one source set."""
    sub = first_cross[first_cross["source_set"] == source_label]
    rows = []
    for X in THRESHOLDS:
        s = sub[sub["threshold"] == X]
        crossed = s[~s["excluded_no_cross"]]
        n_crossed = len(crossed)
        n_fb = int(crossed["fell_back_to_BE"].sum()) if n_crossed > 0 else 0
        if n_crossed > 0:
            med_first = float(crossed["first_cross_bar"].astype(float).median())
            med_min_R = float(crossed["min_held_R_after_cross"].astype(float).median())
            fb_frac = n_fb / n_crossed
        else:
            med_first = float("nan")
            med_min_R = float("nan")
            fb_frac = float("nan")
        rows.append({
            "threshold_atr": float(X),
            "n_winners": int(total_winners),
            "n_crossed": int(n_crossed),
            "n_fell_back_to_BE": int(n_fb),
            "fall_back_fraction_of_crossed": fb_frac,
            "median_first_cross_bar": med_first,
            "median_min_held_R_after_cross": med_min_R,
        })
    return pd.DataFrame(rows)


def _audit_trail(winners_f6_f7: pd.DataFrame,
                  bar_by_bar_f6_f7: pd.DataFrame) -> dict:
    """Compare step 4 tier A (baseline-indexed) numbers against CC-1
    delayed-entry-indexed numbers for F6+F7 winners.

    Step 4 tier A reference: winner median MFE 8.95 ATR, MAE 0.30 ATR,
    bars_held 119. CC-1 recomputes from bar_by_bar.
    """
    # bars_held: from action_bar in CSV (last bar of trade)
    bars_held_series = (winners_f6_f7["action_bar"] - T_STAR + 1).astype(float)
    median_bars_held_cc1 = float(bars_held_series.median())

    # Per-trade max(max_held_R) over its sequence = MFE in CC-1 frame
    per_trade_mfe = bar_by_bar_f6_f7.groupby("trade_id")["max_held_R"].max()
    per_trade_mae_min = bar_by_bar_f6_f7.groupby("trade_id")["min_held_R"].min()
    # MAE in ATR units (mfe_to_date_atr in step 2 is already in 1× ATR units;
    # held_R here is in SL-multiples = 2× ATR. Convert: held_R × 2 = ATR units.)
    median_mfe_atr_cc1 = float((per_trade_mfe * 2.0).median())
    # MAE convention in step 4 tier A is positive magnitude. CC-1 min_held_R
    # is negative for downside; take abs and convert to ATR units.
    median_mae_atr_cc1 = float((per_trade_mae_min.abs() * 2.0).median())

    # Also: median net_r as winner-economics anchor
    median_net_r = float(winners_f6_f7["net_r"].median())

    return {
        "n_winners_f6_f7": int(len(winners_f6_f7)),
        "step4_tier_a_baseline_indexed": {
            "median_winner_mfe_atr": 8.95,
            "median_winner_mae_atr": 0.30,
            "median_bars_held": 119,
        },
        "cc1_delayed_entry_indexed": {
            "median_winner_mfe_atr": median_mfe_atr_cc1,
            "median_winner_mae_atr": median_mae_atr_cc1,
            "median_bars_held": median_bars_held_cc1,
            "median_net_r": median_net_r,
        },
        "deltas": {
            "median_winner_mfe_atr_cc1_minus_tier_a": median_mfe_atr_cc1 - 8.95,
            "median_winner_mae_atr_cc1_minus_tier_a": median_mae_atr_cc1 - 0.30,
            "median_bars_held_cc1_minus_tier_a": median_bars_held_cc1 - 119,
        },
    }


# ============================================================
# Main pipeline
# ============================================================

def _load_inputs():
    """Load F2-F5 + F6+F7 trades, signals, paths, held_ctx."""
    f2_f5 = pd.read_csv(STEP5_OUT / "trades_post_mechanism_f2_f5.csv")
    f6_f7 = pd.read_csv(STEP4_OUT / "trades_post_mechanism.csv")
    signals = D.load_signals().sort_values("trade_id").reset_index(drop=True)
    D._load_paths_full.cache_clear()
    paths_long = D.load_paths_long(max_offset=120)
    held_ctx = D.load_held_ctx(T_STAR)
    return f2_f5, f6_f7, signals, paths_long, held_ctx


def main() -> dict:
    """Run all of Tasks A/B/C/D/E/F, write outputs, return manifest dict
    suitable for run_manifest.json.
    """
    STEP5_OUT.mkdir(parents=True, exist_ok=True)

    # Pin verification before compute
    source_pin = _verify_predictor_source_pin()
    if not source_pin["all_pinned_files_blob_match"]:
        raise RuntimeError(
            f"Predictor source pin drift detected:\n"
            f"{json.dumps(source_pin['per_file'], indent=2)}"
        )

    f2_f5, f6_f7, signals, paths_long, held_ctx = _load_inputs()

    # Winners: net_r > 0 (SL trades have gross_r=-1, never positive)
    win_f2_f5 = f2_f5[f2_f5["net_r"] > 0].copy().sort_values("trade_id").reset_index(drop=True)
    win_f6_f7 = f6_f7[f6_f7["net_r"] > 0].copy().sort_values("trade_id").reset_index(drop=True)

    # Per-trade entry/ATR
    ea_f2_f5 = _compute_entry_and_atr(
        win_f2_f5["trade_id"].values, signals, paths_long, held_ctx
    )
    ea_f6_f7 = _compute_entry_and_atr(
        win_f6_f7["trade_id"].values, signals, paths_long, held_ctx
    )

    # Bar-by-bar held R sequences
    bbb_f2_f5 = _compute_bar_by_bar_sequences(
        win_f2_f5, paths_long, ea_f2_f5, "f2_f5"
    )
    bbb_f6_f7 = _compute_bar_by_bar_sequences(
        win_f6_f7, paths_long, ea_f6_f7, "f6_f7"
    )

    bar_by_bar = pd.concat([bbb_f2_f5, bbb_f6_f7], ignore_index=True)
    bar_by_bar = bar_by_bar.sort_values(["source_set", "trade_id", "bar_offset"]).reset_index(drop=True)

    # First-cross detail
    fc_detail = _first_cross_detail(bar_by_bar)
    fc_detail = fc_detail.sort_values(
        ["source_set", "trade_id", "threshold"]
    ).reset_index(drop=True)

    # Summary tables
    sum_f2_f5 = _summary_table(fc_detail, "f2_f5", total_winners=len(win_f2_f5))
    sum_f6_f7 = _summary_table(fc_detail, "f6_f7", total_winners=len(win_f6_f7))

    # Audit trail on F6+F7 (step 4 tier A vs CC-1 delayed-entry-indexed)
    audit = _audit_trail(win_f6_f7, bbb_f6_f7)

    # Side-by-side comparison
    cmp_rows = []
    # Winner counts + path geometry
    per_trade_mfe_f2_f5 = bbb_f2_f5.groupby("trade_id")["max_held_R"].max() * 2.0
    per_trade_mae_f2_f5 = bbb_f2_f5.groupby("trade_id")["min_held_R"].min().abs() * 2.0
    bars_held_f2_f5 = (win_f2_f5["action_bar"] - T_STAR + 1).astype(float)
    per_trade_mfe_f6_f7 = bbb_f6_f7.groupby("trade_id")["max_held_R"].max() * 2.0
    per_trade_mae_f6_f7 = bbb_f6_f7.groupby("trade_id")["min_held_R"].min().abs() * 2.0
    bars_held_f6_f7 = (win_f6_f7["action_bar"] - T_STAR + 1).astype(float)

    def _diff_str(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return float(a - b)
        return a - b

    cmp_rows.append({
        "metric": "n_winners_total",
        "f2_f5": len(win_f2_f5),
        "f6_f7": len(win_f6_f7),
        "delta": len(win_f2_f5) - len(win_f6_f7),
    })
    cmp_rows.append({
        "metric": "median_winner_mfe_atr",
        "f2_f5": float(per_trade_mfe_f2_f5.median()),
        "f6_f7": float(per_trade_mfe_f6_f7.median()),
        "delta": float(per_trade_mfe_f2_f5.median() - per_trade_mfe_f6_f7.median()),
    })
    cmp_rows.append({
        "metric": "median_winner_mae_atr",
        "f2_f5": float(per_trade_mae_f2_f5.median()),
        "f6_f7": float(per_trade_mae_f6_f7.median()),
        "delta": float(per_trade_mae_f2_f5.median() - per_trade_mae_f6_f7.median()),
    })
    cmp_rows.append({
        "metric": "median_bars_held_winners",
        "f2_f5": float(bars_held_f2_f5.median()),
        "f6_f7": float(bars_held_f6_f7.median()),
        "delta": float(bars_held_f2_f5.median() - bars_held_f6_f7.median()),
    })
    # Fall-back fractions per threshold
    for X in THRESHOLDS:
        a = sum_f2_f5[sum_f2_f5["threshold_atr"] == X].iloc[0]
        b = sum_f6_f7[sum_f6_f7["threshold_atr"] == X].iloc[0]
        cmp_rows.append({
            "metric": f"fall_back_fraction_at_+{X:.1f}_ATR",
            "f2_f5": float(a["fall_back_fraction_of_crossed"]),
            "f6_f7": float(b["fall_back_fraction_of_crossed"]),
            "delta": float(a["fall_back_fraction_of_crossed"] - b["fall_back_fraction_of_crossed"]),
        })
    comparison = pd.DataFrame(cmp_rows)

    # Write all output CSVs
    paths_out = {
        "winner_paths_bar_by_bar.csv": STEP5_OUT / "winner_paths_bar_by_bar.csv",
        "first_cross_detail.csv": STEP5_OUT / "first_cross_detail.csv",
        "winner_path_after_peak_f2_f5.csv": STEP5_OUT / "winner_path_after_peak_f2_f5.csv",
        "winner_path_after_peak_f6_f7.csv": STEP5_OUT / "winner_path_after_peak_f6_f7.csv",
        "comparison_f2_f5_vs_f6_f7.csv": STEP5_OUT / "comparison_f2_f5_vs_f6_f7.csv",
    }
    bar_by_bar.to_csv(paths_out["winner_paths_bar_by_bar.csv"],
                       index=False, float_format="%.10g", lineterminator="\n")
    fc_detail.to_csv(paths_out["first_cross_detail.csv"],
                      index=False, float_format="%.10g", lineterminator="\n")
    sum_f2_f5.to_csv(paths_out["winner_path_after_peak_f2_f5.csv"],
                      index=False, float_format="%.10g", lineterminator="\n")
    sum_f6_f7.to_csv(paths_out["winner_path_after_peak_f6_f7.csv"],
                      index=False, float_format="%.10g", lineterminator="\n")
    comparison.to_csv(paths_out["comparison_f2_f5_vs_f6_f7.csv"],
                       index=False, float_format="%.10g", lineterminator="\n")

    output_shas = {fname: _sha256_file(p) for fname, p in paths_out.items()}

    # Manifest
    manifest = {
        "step": "l_arc_2/step5_recharacterisation/delayed_entry_t_gb/CC-1",
        "git_commit_head_at_run": _git_commit_head(),
        "predictor_source_pin_verification": source_pin,
        "versions": _versions(),
        "t_star": T_STAR,
        "thresholds_atr": list(THRESHOLDS),
        "n_winners": {
            "f2_f5": int(len(win_f2_f5)),
            "f6_f7": int(len(win_f6_f7)),
        },
        "audit_trail_f6_f7": audit,
        "inputs": {
            "trades_post_mechanism_f2_f5.csv": {
                "path": "results/l_arc_2/step5_recharacterisation/delayed_entry_t_gb/trades_post_mechanism_f2_f5.csv",
                "sha256": _sha256_file(STEP5_OUT / "trades_post_mechanism_f2_f5.csv"),
            },
            "trades_post_mechanism_f6_f7.csv": {
                "path": "results/l_arc_2/step4/delayed_entry_t_gb/trades_post_mechanism.csv",
                "sha256": _sha256_file(STEP4_OUT / "trades_post_mechanism.csv"),
            },
            "signals_features.csv": {
                "path": "results/l_arc_2/step2_descriptive/signals_features.csv",
                "sha256": _sha256_file(C.SIGNALS_CSV),
            },
            "trade_paths.csv": {
                "path": "results/l_arc_2/step2_descriptive/trade_paths.csv",
                "sha256": _sha256_file(C.PATHS_CSV),
            },
            "held_bar_evolution_t1.csv": {
                "path": "results/l_arc_2/step2_descriptive/held_bar_evolution/t1.csv",
                "sha256": _sha256_file(C.HELD_CTX / "t1.csv"),
            },
        },
        "outputs_sha256": output_shas,
        # No timestamp — deterministic across runs.
    }
    # NOTE: CC-0 owns STEP5_OUT / "run_manifest.json".
    # CC-1 writes to a CC-1-specific file to avoid clobber.
    cc1_manifest_path = STEP5_OUT / "cc1_run_manifest.json"
    with open(cc1_manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    manifest["outputs_sha256"]["cc1_run_manifest.json"] = _sha256_file(cc1_manifest_path)

    return manifest


# ============================================================
# Execution-side lookahead invariant test (Task H)
# ============================================================

def lookahead_invariant_test(n_sample: int = 50, seed: int = 23) -> dict:
    """For sampled winner trades, perturb OHLC at bars > k for a random
    k within each trade's held-R sequence. Re-compute held R at bars <= k.
    Assert byte-identical to unperturbed values.

    Held R formula: max_R = (high - entry) / (2 * ATR_at_entry); the
    formula has no dependency on bars > k by construction. The test
    exercises this empirically.
    """
    STEP5_OUT.mkdir(parents=True, exist_ok=True)
    f2_f5, f6_f7, signals, paths_long, held_ctx = _load_inputs()

    # Combine F2-F5 + F6+F7 winners
    win_f2_f5 = f2_f5[f2_f5["net_r"] > 0].copy()
    win_f6_f7 = f6_f7[f6_f7["net_r"] > 0].copy()
    all_winners = pd.concat([
        win_f2_f5.assign(source_set="f2_f5"),
        win_f6_f7.assign(source_set="f6_f7"),
    ], ignore_index=True).sort_values("trade_id").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    sample_n = min(n_sample, len(all_winners))
    sample_indices = rng.choice(len(all_winners), size=sample_n, replace=False)
    sample = all_winners.iloc[sample_indices].reset_index(drop=True)

    # Compute baseline held R sequences for the sample
    ea_baseline = _compute_entry_and_atr(
        sample["trade_id"].values, signals, paths_long, held_ctx
    ).set_index("trade_id")

    byte_identical = 0
    divergent = 0
    diffs = []

    for _, row in sample.iterrows():
        tid = int(row["trade_id"])
        exit_bar = int(row["action_bar"])

        entry_price = float(ea_baseline.at[tid, "entry_price"])
        atr_at_entry = float(ea_baseline.at[tid, "atr_at_entry"])
        if not np.isfinite(entry_price) or not np.isfinite(atr_at_entry) or atr_at_entry <= 0:
            continue
        denom = 2.0 * atr_at_entry

        # Trade's bars (delayed entry through exit)
        trade_rows = paths_long[
            (paths_long["trade_id"] == tid) &
            (paths_long["bar_offset"] >= T_STAR) &
            (paths_long["bar_offset"] <= exit_bar)
        ].sort_values("bar_offset").reset_index(drop=True)
        n_bars = len(trade_rows)
        if n_bars < 2:
            continue  # need at least 2 bars to pick k that has a "bars > k" tail

        # Random k within trade's bar range; choose k so that there's a non-empty
        # tail (bars > k) AND a non-empty prefix (bars <= k).
        k_idx = int(rng.integers(low=0, high=n_bars - 1))
        k_bar_offset = int(trade_rows.loc[k_idx, "bar_offset"])

        # Baseline held R values at bars <= k
        prefix = trade_rows.iloc[:k_idx + 1]
        baseline_max_R = (prefix["high"].values - entry_price) / denom
        baseline_min_R = (prefix["low"].values - entry_price) / denom

        # Perturb high/low/close at bars > k for this trade in a copy of paths_long
        paths_perturbed = paths_long.copy()
        perturb_mask = (
            (paths_perturbed["trade_id"] == tid) &
            (paths_perturbed["bar_offset"] > k_bar_offset)
        )
        n_perturb = int(perturb_mask.sum())
        if n_perturb == 0:
            continue
        noise_h = rng.normal(loc=0.0, scale=10.0, size=n_perturb)
        noise_l = rng.normal(loc=0.0, scale=10.0, size=n_perturb)
        noise_c = rng.normal(loc=0.0, scale=10.0, size=n_perturb)
        paths_perturbed.loc[perturb_mask, "high"] = paths_perturbed.loc[perturb_mask, "high"].values + noise_h
        paths_perturbed.loc[perturb_mask, "low"] = paths_perturbed.loc[perturb_mask, "low"].values + noise_l
        paths_perturbed.loc[perturb_mask, "close"] = paths_perturbed.loc[perturb_mask, "close"].values + noise_c

        # Re-extract trade's bars (prefix part should be unchanged)
        trade_rows_perturbed = paths_perturbed[
            (paths_perturbed["trade_id"] == tid) &
            (paths_perturbed["bar_offset"] >= T_STAR) &
            (paths_perturbed["bar_offset"] <= exit_bar)
        ].sort_values("bar_offset").reset_index(drop=True)
        prefix_perturbed = trade_rows_perturbed.iloc[:k_idx + 1]
        perturbed_max_R = (prefix_perturbed["high"].values - entry_price) / denom
        perturbed_min_R = (prefix_perturbed["low"].values - entry_price) / denom

        # Byte-identity check on the prefix
        if (np.array_equal(baseline_max_R, perturbed_max_R, equal_nan=True) and
            np.array_equal(baseline_min_R, perturbed_min_R, equal_nan=True)):
            byte_identical += 1
        else:
            divergent += 1
            diffs.append({
                "trade_id": tid,
                "k_bar_offset": k_bar_offset,
                "n_prefix_bars": int(k_idx + 1),
                "n_perturbed_bars": n_perturb,
            })

    return {
        "sampled": int(byte_identical + divergent),
        "byte_identical": byte_identical,
        "divergent": divergent,
        "diffs": diffs[:10],
        "perturbation_floor_bar_offset_gt_k": True,
    }


if __name__ == "__main__":
    m = main()
    print("CC-1 manifest summary:")
    print(json.dumps({
        "n_winners": m["n_winners"],
        "audit_trail_f6_f7": m["audit_trail_f6_f7"],
        "outputs_sha256": m["outputs_sha256"],
        "all_pinned_files_blob_match":
            m["predictor_source_pin_verification"]["all_pinned_files_blob_match"],
    }, indent=2, default=str))

# ruff: noqa: E402  (sys.path.insert needed before project imports)
"""Phase B — marginal distributions (op spec §5.1–§5.8).

For every metric in the §5.1–§5.8 catalogue we emit:
  - <metric>.csv               (full distribution per §11.1)
  - <metric>__by_fold.csv      (per-fold breakdown per §11.3)
  - <metric>__hist.csv         (histogram per §5)

Subfolders under step2_descriptive/distributions/:
  marginals/, forward/, sequence/, complexity/, survival/, early_bar/, asymmetry/

Degenerate angles (held-window items at h=1 plus survival/early_bar) are
emitted with a `# degenerate_by_construction: ...` header comment.

Descriptive only — emits artefacts, no recommendations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_1.step2._distributions import (
    write_distribution,
    write_per_fold_distribution,
)
from scripts.l_arc_1.step2._io import H_GRID, STEP2_DIR

DIST_DIR = STEP2_DIR / "distributions"


def _emit(
    values: pd.Series,
    subdir: str,
    metric: str,
    *,
    degenerate: bool = False,
    reason: str = "",
    features: pd.DataFrame | None = None,
    fold_col: str = "fold_id",
) -> None:
    out_dir = DIST_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_distribution(
        values.tolist(),
        out_dir / f"{metric}.csv",
        metric_name=metric,
        degenerate=degenerate,
        degenerate_reason=reason,
        hist_path=out_dir / f"{metric}__hist.csv",
    )
    if features is not None and fold_col in features.columns:
        per_fold_df = features[[fold_col]].copy()
        per_fold_df["__value"] = values.values
        write_per_fold_distribution(
            per_fold_df,
            "__value",
            fold_col,
            out_dir / f"{metric}__by_fold.csv",
            metric,
        )


def _emit_categorical_counts(
    values: pd.Series, subdir: str, metric: str, features: pd.DataFrame | None = None
) -> None:
    out_dir = DIST_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = values.value_counts(dropna=False).sort_index()
    fraction = counts / counts.sum() if counts.sum() else counts
    df = pd.DataFrame(
        {"category": counts.index.astype(str), "count": counts.values, "fraction": fraction.values}
    )
    df.to_csv(out_dir / f"{metric}.csv", index=False, lineterminator="\n")
    if features is not None and "fold_id" in features.columns:
        wide = pd.crosstab(features["fold_id"], values).reset_index()
        wide.to_csv(out_dir / f"{metric}__by_fold.csv", index=False, lineterminator="\n")


def run_phase_b() -> None:
    print("[Phase B] reading signals_features.csv...")
    f = pd.read_csv(STEP2_DIR / "signals_features.csv")
    print(f"  rows: {len(f):,}")

    # §5.1 marginals
    print("[Phase B] §5.1 marginals...")
    _emit(f["net_r"], "marginals", "net_r", features=f)
    _emit(f["gross_r"], "marginals", "gross_r", features=f)
    _emit(f["spread_cost_R"], "marginals", "spread_cost_R", features=f)
    _emit(
        f["mfe_held_atr"],
        "marginals",
        "mfe_held_atr",
        features=f,
        degenerate=True,
        reason="h=1 held window is 1 bar; identical to fwd_mfe_h1_atr",
    )
    _emit(
        f["mae_held_atr"],
        "marginals",
        "mae_held_atr",
        features=f,
        degenerate=True,
        reason="h=1 held window is 1 bar; identical to fwd_mae_h1_atr",
    )
    _emit(
        f["bars_held"],
        "marginals",
        "bars_held",
        features=f,
        degenerate=True,
        reason="time-exit h=1; all trades held 1 bar (except sl_hit which is intrabar)",
    )
    _emit_categorical_counts(f["exit_reason"], "marginals", "exit_reason", features=f)
    _emit(
        f["peak_to_final_r_ratio"],
        "marginals",
        "peak_to_final_r_ratio",
        features=f,
        degenerate=True,
        reason="peak/final ratio over a 1-bar held window",
    )
    _emit(
        f["mfe_to_mae_ratio_held"],
        "marginals",
        "mfe_to_mae_ratio_held",
        features=f,
        degenerate=True,
        reason="ratio over a 1-bar held window",
    )

    # §5.2 forward-horizon geometry
    print("[Phase B] §5.2 forward-horizon geometry...")
    for h in H_GRID:
        for stem in (
            f"fwd_logret_h{h}",
            f"fwd_mfe_h{h}_atr",
            f"fwd_mae_h{h}_atr",
            f"fwd_mfe_to_mae_ratio_h{h}",
        ):
            if stem in f.columns:
                _emit(f[stem], "forward", stem, features=f)
    # bars_to_+x_atr / -x_atr capped @ H (H = max h available in the dataset)
    cap = max(H_GRID)
    for x in (0.5, 1.0, 1.5, 2.0, 3.0):
        for prefix in ("plus", "minus"):
            col = f"bars_to_{prefix}_{x}_atr_capped_{cap}"
            if col in f.columns:
                _emit(f[col], "forward", col, features=f)
    # fraction reached markers
    for x in (0.5, 1.0, 2.0):
        col = f"reached_plus_{x}_atr_within_{cap}"
        if col in f.columns:
            _emit(f[col], "forward", col, features=f)
    # race condition (primary path-direction effect-size axis)
    _emit(f["race_bars_plus1_minus_minus1"], "forward", "race_bars_plus1_minus_minus1", features=f)

    # §5.4 MFE/MAE sequence classification
    print("[Phase B] §5.4 sequence classification...")
    _emit_categorical_counts(
        f["mfe_sequence_class_held"], "sequence", "mfe_sequence_class_held", features=f
    )
    # held-window time difference: degenerate (all 0 by construction)
    # since both peak and trough are within bar N+1.
    diff_held = pd.Series(np.zeros(len(f)))
    _emit(
        diff_held,
        "sequence",
        "time_to_peak_mfe_minus_trough_mae_held",
        features=f,
        degenerate=True,
        reason="h=1; peak and trough are both at t=1 by construction",
    )
    # Forward-path sequence classification at h=24 and h=120
    for h in (24, 120):
        col = f"mfe_sequence_class_fwd_h{h}"
        if col in f.columns:
            _emit_categorical_counts(f[col], "sequence", col, features=f)
    # Forward-path time differences and per-class breakdown
    diff_fwd = f["fwd_time_to_peak_mfe"] - f["fwd_time_to_trough_mae"]
    _emit(diff_fwd, "sequence", "time_to_peak_mfe_minus_trough_mae_fwd", features=f)
    # Per-class breakdown of net_r and forward stats at h=24 and h=120
    for h in (24, 120):
        seq_col = f"mfe_sequence_class_fwd_h{h}"
        if seq_col not in f.columns:
            continue
        out_dir = DIST_DIR / "sequence"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for cls in sorted(f[seq_col].unique()):
            sub = f[f[seq_col] == cls]
            rows.append(
                {
                    "class": cls,
                    "count": len(sub),
                    "fraction": len(sub) / len(f) if len(f) else 0.0,
                    "mean_net_r": float(sub["net_r"].mean()),
                    "median_net_r": float(sub["net_r"].median()),
                    "median_fwd_mfe_h24_atr": float(
                        sub.get("fwd_mfe_h24_atr", pd.Series([np.nan])).median()
                    ),
                    "median_fwd_mae_h24_atr": float(
                        sub.get("fwd_mae_h24_atr", pd.Series([np.nan])).median()
                    ),
                    "median_fwd_mfe_h120_atr": float(
                        sub.get("fwd_mfe_h120_atr", pd.Series([np.nan])).median()
                    ),
                    "median_fwd_mae_h120_atr": float(
                        sub.get("fwd_mae_h120_atr", pd.Series([np.nan])).median()
                    ),
                    "exit_reason_sl_hit_pct": float(
                        (sub["exit_reason"] == "stop_loss").mean() * 100.0
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(
            out_dir / f"per_class_summary_fwd_h{h}.csv",
            index=False,
            lineterminator="\n",
        )

    # §5.5 path complexity (re-cast on fwd path)
    print("[Phase B] §5.5 path complexity (fwd path re-cast)...")
    out_complexity = DIST_DIR / "complexity"
    out_complexity.mkdir(parents=True, exist_ok=True)
    note_lines = [
        "# Per prompt Phase A.6 schema_notes: §5.5 complexity metrics are recast on the",
        "# unconditional forward path (H=480 bars), since the verbatim h=1 held window",
        "# is degenerate (1 bar) and these metrics are trivial there.",
    ]
    (out_complexity / "_schema_note.txt").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    for col in (
        "fwd_oscillation_count",
        "fwd_monotonicity_ratio",
        "fwd_max_consecutive_with",
        "fwd_max_consecutive_against",
        "fwd_acf1_returns",
        "fwd_time_to_peak_mfe",
        "fwd_time_to_trough_mae",
    ):
        if col in f.columns:
            _emit(f[col], "complexity", col, features=f)
    # time_from_peak_to_exit / r_given_back_from_peak — degenerate at h=1; recast on forward path
    # On the forward path with H=480, define "exit" as t=H (end of forward window).
    t_peak = f["fwd_time_to_peak_mfe"].astype(float)
    H_max = max(H_GRID)
    time_from_peak_to_exit_fwd = H_max - t_peak
    _emit(
        time_from_peak_to_exit_fwd, "complexity", "time_from_peak_mfe_to_window_end_fwd", features=f
    )
    # r_given_back_from_peak (fwd): peak fwd_mfe (= max over window) minus log return at window end
    r_given_back = f["fwd_mfe_h{}_atr".format(H_max)] - f["fwd_logret_h{}".format(H_max)]
    _emit(r_given_back, "complexity", "r_given_back_from_peak_fwd_atr_units", features=f)
    # held-window degenerate placeholders
    out_dir = DIST_DIR / "complexity"
    pd.DataFrame(
        [
            {
                "metric": "held_window_complexity",
                "note": "degenerate; held window is 1 bar — see fwd_* columns above",
            }
        ]
    ).to_csv(out_dir / "held_window_complexity_degenerate.csv", index=False, lineterminator="\n")

    # §5.6 survival curves (degenerate at h=1; only t=1 non-vacuous)
    print("[Phase B] §5.6 survival curves (degenerate)...")
    out_dir = DIST_DIR / "survival"
    out_dir.mkdir(parents=True, exist_ok=True)
    surv_rows = []
    len(f)
    surv_ts = [1, 5, 10, 20, 50, 100, 200]
    # Only t=1 is non-vacuous: 100% of trades are still open at t=1 (entry bar).
    # Beyond t=1 the held trade is closed (time exit at N+2 = t=2 in held terms... but
    # the held window is 1 bar so t=1 already is the only held bar).
    # We report: fraction still open at t, mean R | still open, win% | still open
    for t in surv_ts:
        if t == 1:
            sub = f
            frac = 1.0
            mean_r_open = float(sub["net_r"].mean()) if len(sub) else np.nan
            win_open = float((sub["net_r"] > 0).mean() * 100.0) if len(sub) else np.nan
        else:
            frac = 0.0
            mean_r_open = np.nan
            win_open = np.nan
        surv_rows.append(
            {
                "t": t,
                "fraction_still_open": frac,
                "mean_r_given_open": mean_r_open,
                "win_pct_given_open": win_open,
            }
        )
    out_path = out_dir / "survival.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# degenerate_by_construction: true | reason: time exit h=1; trades exit at t=2\n")
        pd.DataFrame(surv_rows).to_csv(fh, index=False, lineterminator="\n")

    # §5.7 early-bar predictivity (degenerate at h=1 — cum R at t=1 ≡ final R)
    print("[Phase B] §5.7 early-bar predictivity (degenerate)...")
    out_dir = DIST_DIR / "early_bar"
    out_dir.mkdir(parents=True, exist_ok=True)
    eb_path = out_dir / "early_bar_predictivity.csv"
    eb_rows = [
        {
            "t": 1,
            "corr_cumR_t_to_finalR": 1.0,
            "note": "cum R at t=1 ≡ final R for h=1 trades; correlation is 1.0 by construction",
        }
    ]
    with eb_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "# degenerate_by_construction: true | reason: h=1 trades have final R = cum R at t=1\n"
        )
        pd.DataFrame(eb_rows).to_csv(fh, index=False, lineterminator="\n")

    # §5.8 win/loss asymmetry
    print("[Phase B] §5.8 win/loss asymmetry...")
    out_dir = DIST_DIR / "asymmetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    winners = f[f["net_r"] > 0]
    losers = f[f["net_r"] < 0]
    flats = f[f["net_r"] == 0]
    asym_rows = [
        {
            "side": "winners",
            "n": len(winners),
            "median_r": float(winners["net_r"].median()) if len(winners) else np.nan,
            "mean_r": float(winners["net_r"].mean()) if len(winners) else np.nan,
            "p95_r": float(winners["net_r"].quantile(0.95)) if len(winners) else np.nan,
            "p5_r": float(winners["net_r"].quantile(0.05)) if len(winners) else np.nan,
            "median_bars_held": float(winners["bars_held"].median()) if len(winners) else np.nan,
            "median_mae_R_during": float(winners["mae_R"].median()) if len(winners) else np.nan,
            "p95_mae_R_during": float(winners["mae_R"].quantile(0.95)) if len(winners) else np.nan,
            "median_mfe_R_during": float(winners["mfe_R"].median()) if len(winners) else np.nan,
        },
        {
            "side": "losers",
            "n": len(losers),
            "median_r": float(losers["net_r"].median()) if len(losers) else np.nan,
            "mean_r": float(losers["net_r"].mean()) if len(losers) else np.nan,
            "p95_r": float(losers["net_r"].quantile(0.95)) if len(losers) else np.nan,
            "p5_r": float(losers["net_r"].quantile(0.05)) if len(losers) else np.nan,
            "median_bars_held": float(losers["bars_held"].median()) if len(losers) else np.nan,
            "median_mae_R_during": float(losers["mae_R"].median()) if len(losers) else np.nan,
            "p95_mae_R_during": float(losers["mae_R"].quantile(0.95)) if len(losers) else np.nan,
            "median_mfe_R_during": float(losers["mfe_R"].median()) if len(losers) else np.nan,
        },
        {
            "side": "flats",
            "n": len(flats),
            "median_r": float(flats["net_r"].median()) if len(flats) else np.nan,
            "mean_r": float(flats["net_r"].mean()) if len(flats) else np.nan,
            "p95_r": float(flats["net_r"].quantile(0.95)) if len(flats) else np.nan,
            "p5_r": float(flats["net_r"].quantile(0.05)) if len(flats) else np.nan,
            "median_bars_held": float(flats["bars_held"].median()) if len(flats) else np.nan,
            "median_mae_R_during": float(flats["mae_R"].median()) if len(flats) else np.nan,
            "p95_mae_R_during": float(flats["mae_R"].quantile(0.95)) if len(flats) else np.nan,
            "median_mfe_R_during": float(flats["mfe_R"].median()) if len(flats) else np.nan,
        },
    ]
    pd.DataFrame(asym_rows).to_csv(
        out_dir / "win_loss_asymmetry.csv", index=False, lineterminator="\n"
    )

    # §5.15 portfolio context as marginal distributions (not just signal-time)
    print("[Phase B] §5.15 cross-pair / portfolio context distributions...")
    for col in (
        "concurrent_signals_same_bar",
        "concurrent_signals_within_3h",
        "currency_basket_3h_USD",
        "currency_basket_3h_EUR",
        "currency_basket_3h_JPY",
        "currency_basket_3h_GBP",
        "trade_overlap_at_execution_time",
        "sequential_same_pair_density_24h",
    ):
        if col in f.columns:
            _emit(f[col], "marginals", col, features=f)

    print("[Phase B] done.")


if __name__ == "__main__":
    run_phase_b()

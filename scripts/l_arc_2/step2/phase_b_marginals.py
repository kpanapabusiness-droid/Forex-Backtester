# ruff: noqa: E402  (sys.path.insert needed before project imports)
"""Phase B — marginal distributions (op spec §5.1–§5.8).

For arc 2 the held window is REAL (mean ~47 bars, max 120); none of arc 1's
"degenerate at h=1" labelling applies. Survival and held-bar evolution curves
are non-trivial.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_2.step2._io import (
    H_GRID,
    STEP2_DIR,
    write_distribution,
    write_per_fold_distribution,
)

DIST_DIR = STEP2_DIR / "distributions"


def _emit(
    values: pd.Series,
    subdir: str,
    metric: str,
    features: pd.DataFrame | None = None,
    fold_col: str = "fold_id",
) -> None:
    out_dir = DIST_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_distribution(
        values.tolist(),
        out_dir / f"{metric}.csv",
        metric_name=metric,
        hist_path=out_dir / f"{metric}__hist.csv",
    )
    if features is not None and fold_col in features.columns:
        per_fold_df = features[[fold_col]].copy()
        per_fold_df["__value"] = values.values
        write_per_fold_distribution(
            per_fold_df, "__value", fold_col, out_dir / f"{metric}__by_fold.csv", metric
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

    # §5.1 marginals — REAL on arc 2 (held window > 1 bar)
    print("[Phase B] §5.1 marginals...")
    _emit(f["net_r"], "marginals", "net_r", features=f)
    _emit(f["gross_r"], "marginals", "gross_r", features=f)
    _emit(f["spread_cost_R"], "marginals", "spread_cost_R", features=f)
    _emit(f["mfe_held_atr"], "marginals", "mfe_held_atr", features=f)
    _emit(f["mae_held_atr"], "marginals", "mae_held_atr", features=f)
    _emit(f["bars_held"], "marginals", "bars_held", features=f)
    _emit_categorical_counts(f["exit_reason"], "marginals", "exit_reason", features=f)
    _emit(f["peak_to_final_r_ratio"], "marginals", "peak_to_final_r_ratio", features=f)
    _emit(f["mfe_to_mae_ratio_held"], "marginals", "mfe_to_mae_ratio_held", features=f)
    _emit(f["r_given_back_from_peak"], "marginals", "r_given_back_from_peak", features=f)

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
    cap = max(H_GRID)
    for x in (0.5, 1.0, 1.5, 2.0, 3.0):
        for prefix in ("plus", "minus"):
            col = f"bars_to_{prefix}_{x}_atr_capped_{cap}"
            if col in f.columns:
                _emit(f[col], "forward", col, features=f)
    for x in (0.5, 1.0, 2.0):
        col = f"reached_plus_{x}_atr_within_{cap}"
        if col in f.columns:
            _emit(f[col], "forward", col, features=f)
    _emit(f["race_bars_plus1_minus_minus1"], "forward", "race_bars_plus1_minus_minus1", features=f)

    # §5.4 MFE/MAE sequence classification (held + forward h24, h120)
    print("[Phase B] §5.4 sequence classification...")
    _emit_categorical_counts(
        f["mfe_sequence_class_held"], "sequence", "mfe_sequence_class_held", features=f
    )
    diff_held = f["time_to_peak_mfe"] - f["time_to_trough_mae"]
    _emit(diff_held, "sequence", "time_to_peak_mfe_minus_trough_mae_held", features=f)
    for h in (24, 120):
        col = f"mfe_sequence_class_fwd_h{h}"
        if col in f.columns:
            _emit_categorical_counts(f[col], "sequence", col, features=f)
    # Per-class summary on h=24 and h=120 (held + forward)
    out_dir = DIST_DIR / "sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Held-window
    rows = []
    for cls in sorted(f["mfe_sequence_class_held"].dropna().unique()):
        sub = f[f["mfe_sequence_class_held"] == cls]
        rows.append(
            {
                "class": cls,
                "count": len(sub),
                "fraction": len(sub) / len(f) if len(f) else 0.0,
                "mean_net_r": float(sub["net_r"].mean()),
                "median_net_r": float(sub["net_r"].median()),
                "median_mfe_held_atr": float(sub["mfe_held_atr"].median()),
                "median_mae_held_atr": float(sub["mae_held_atr"].median()),
                "median_bars_held": float(sub["bars_held"].median()),
                "frac_sl_hit": float((sub["exit_reason"] == "sl_hit").mean()),
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / "per_class_summary_held.csv", index=False, lineterminator="\n"
    )
    for h in (24, 120):
        seq_col = f"mfe_sequence_class_fwd_h{h}"
        if seq_col not in f.columns:
            continue
        rows = []
        for cls in sorted(f[seq_col].dropna().unique()):
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
                    "frac_sl_hit": float((sub["exit_reason"] == "sl_hit").mean()),
                }
            )
        pd.DataFrame(rows).to_csv(
            out_dir / f"per_class_summary_fwd_h{h}.csv", index=False, lineterminator="\n"
        )

    # §5.5 path complexity (HELD — real on arc 2)
    print("[Phase B] §5.5 path complexity (HELD window)...")
    for col in (
        "oscillation_count",
        "monotonicity_ratio",
        "max_consecutive_with",
        "max_consecutive_against",
        "acf1_returns_during_hold",
        "time_to_peak_mfe",
        "time_to_trough_mae",
        "time_from_peak_to_exit",
    ):
        if col in f.columns:
            _emit(f[col], "complexity", col, features=f)
    # Amendment 4 clustering features (forward-window-derived path geometry)
    for col in (
        "fwd_realized_range_atr",
        "fwd_fraction_time_above_entry",
        "fwd_max_consecutive_directional_bars",
    ):
        if col in f.columns:
            _emit(f[col], "complexity", col, features=f)

    # §5.6 survival curves — REAL on arc 2 (time-exit h=120, SL hit possible at any held bar)
    print("[Phase B] §5.6 survival curves...")
    out_dir = DIST_DIR / "survival"
    out_dir.mkdir(parents=True, exist_ok=True)
    bars_held = f["bars_held"].to_numpy()
    n_total = len(f)
    surv_rows = []
    for t in [1, 5, 10, 20, 50, 100, 200]:
        # fraction still open at t = trades with bars_held > t (still being held past bar t)
        # Note: bar_offset 0 = entry bar; "still open at t" means trade has not exited by bar offset t
        # bars_held = N means trade exited at bar_offset N-1 (intrabar SL) or bar_offset N (time exit at next-bar open).
        # Conservatively interpret "still open at bar t" as bars_held > t (open through bar t).
        if t >= 121:
            frac = 0.0
            mean_r_open = np.nan
            win_open = np.nan
            n_open = 0
        else:
            still_open_mask = bars_held > t
            n_open = int(still_open_mask.sum())
            frac = n_open / n_total if n_total else 0.0
            sub = f[still_open_mask]
            mean_r_open = float(sub["net_r"].mean()) if n_open else np.nan
            win_open = float((sub["net_r"] > 0).mean() * 100.0) if n_open else np.nan
        surv_rows.append(
            {
                "t": t,
                "fraction_still_open": frac,
                "n_open": n_open,
                "mean_r_given_open": mean_r_open,
                "win_pct_given_open": win_open,
            }
        )
    pd.DataFrame(surv_rows).to_csv(out_dir / "survival.csv", index=False, lineterminator="\n")

    # §5.7 early-bar predictivity — using trade_paths.csv at t ∈ {1,3,5,10}
    print("[Phase B] §5.7 early-bar predictivity...")
    out_dir = DIST_DIR / "early_bar"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stream-read trade_paths.csv and pivot cum_logret_from_entry at the requested offsets
    paths = pd.read_csv(
        STEP2_DIR / "trade_paths.csv", usecols=["trade_id", "bar_offset", "cum_logret_from_entry"]
    )
    # Map t -> offset = t-1
    rows_eb = []
    for t in [1, 3, 5, 10]:
        off = t - 1
        sub = paths[paths["bar_offset"] == off][["trade_id", "cum_logret_from_entry"]]
        m = f.merge(sub, on="trade_id", how="left")
        cum_t = m["cum_logret_from_entry"].astype(float)
        final = m["net_r"].astype(float)
        corr = float(cum_t.corr(final)) if cum_t.notna().sum() >= 2 else np.nan
        # Conditional mean final R given cum_R at t deciles
        m["__dec"] = pd.qcut(cum_t.rank(method="first"), q=10, labels=False, duplicates="drop")
        per_dec = (
            m.groupby("__dec")
            .agg(
                n=("trade_id", "count"),
                mean_cum_t=("cum_logret_from_entry", "mean"),
                mean_final_r=("net_r", "mean"),
                win_pct=("net_r", lambda s: float((s > 0).mean() * 100)),
            )
            .reset_index()
        )
        per_dec["t"] = t
        rows_eb.extend(per_dec.to_dict("records"))
        with (out_dir / f"corr_t{t}.txt").open("w", encoding="utf-8") as fh:
            fh.write(f"corr(cum_R at bar_offset={off} (t={t}), final net R) = {corr:.6f}\n")
    pd.DataFrame(rows_eb).to_csv(out_dir / "decile_breakdown.csv", index=False, lineterminator="\n")

    # §5.8 win/loss asymmetry
    print("[Phase B] §5.8 win/loss asymmetry...")
    out_dir = DIST_DIR / "asymmetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    winners = f[f["net_r"] > 0]
    losers = f[f["net_r"] < 0]
    flats = f[f["net_r"] == 0]
    asym_rows = []
    for label, sub in (("winners", winners), ("losers", losers), ("flats", flats)):
        n = len(sub)
        asym_rows.append(
            {
                "side": label,
                "n": n,
                "median_r": float(sub["net_r"].median()) if n else np.nan,
                "mean_r": float(sub["net_r"].mean()) if n else np.nan,
                "p95_r": float(sub["net_r"].quantile(0.95)) if n else np.nan,
                "p5_r": float(sub["net_r"].quantile(0.05)) if n else np.nan,
                "median_bars_held": float(sub["bars_held"].median()) if n else np.nan,
                "median_mae_R_during": float(sub["mae_R"].median()) if n else np.nan,
                "p95_mae_R_during": float(sub["mae_R"].quantile(0.95)) if n else np.nan,
                "median_mfe_R_during": float(sub["mfe_R"].median()) if n else np.nan,
            }
        )
    pd.DataFrame(asym_rows).to_csv(
        out_dir / "win_loss_asymmetry.csv", index=False, lineterminator="\n"
    )

    # §5.15 portfolio context marginals
    print("[Phase B] §5.15 cross-pair / portfolio context...")
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
    # Pre-signal context (new) marginals
    for col in ("cum_logret_1h_24", "cum_logret_1h_72", "cum_logret_1h_168", "vol_realized_1h_24h"):
        if col in f.columns:
            _emit(f[col], "marginals", col, features=f)

    print("[Phase B] done.")


if __name__ == "__main__":
    run_phase_b()

"""§2.8 R-distribution statistical consistency.

Two complementary analyses (the bespoke Step-5 driver does not persist
per-trade R under the winning config, so we have:

  A. Step 1 pool R (3,152 trades, 2010-2026) — under the Step 1
     SL=2.0×ATR sl_only path. Diagnoses the underlying signal edge
     across the full window; insensitive to the Step 5 winning exit
     policy.
  B. Amendment 3 holdout-rerun R (1,093 trades, 2021-2026) — under the
     winning sl_partial_close_1r_runner_trail exit policy. The actual
     deployable-config R-distribution, but holdout-only.

Tests:
  1. Year-by-year stats (mean / median / win rate / +1R hit rate / sd / n).
  2. Two-sample tests:
     - 2010-2020 vs 2021-2026 (full-pool Step-1 R): tests whether the
       holdout era is statistically different from the IS era. KS +
       Welch's t.
     - 2021-2023 vs 2024-2025 (full-pool Step-1 R AND holdout-rerun R):
       tests whether the recent two years differ from the earlier
       holdout years — important because Phase 1 engine work happened
       against 2024-2025 data.
  3. Structural-break smell: rolling 12-month mean R on the full pool;
     flag any 12-month window whose mean differs from the prior 12 by
     more than 2σ of the per-12-month mean SE.

Output: results/l_arc_10_v3.0.2/exhaustive_audit/section_2_8_r_distribution.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "exhaustive_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "section_2_8_r_distribution.json"

POOL_PATH = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "step_1" / "pool.parquet"
HOLDOUT_R_SAFE = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "step_5" / "amendment_3" / "holdout_rerun_r_safe.csv"


def _year_stats(df: pd.DataFrame, label: str = "r") -> list[dict]:
    df = df.copy()
    df["year"] = df["signal_year"].astype(int)
    rows: list[dict] = []
    for year, sub in df.groupby("year"):
        r = sub[label].astype(float).to_numpy()
        rows.append(
            {
                "year": int(year),
                "n": int(len(r)),
                "mean_r": float(np.mean(r)) if len(r) else 0.0,
                "median_r": float(np.median(r)) if len(r) else 0.0,
                "win_rate": float(np.mean(r > 0)) if len(r) else 0.0,
                "hit_plus_1r_rate": float(np.mean(r >= 1.0)) if len(r) else 0.0,
                "std_r": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
                "sum_r": float(np.sum(r)) if len(r) else 0.0,
            }
        )
    return rows


def _ks_and_t(a: np.ndarray, b: np.ndarray) -> dict:
    """Return KS + Welch t test summary between two arrays."""
    if len(a) < 2 or len(b) < 2:
        return {"ks_stat": None, "ks_p": None, "t_stat": None, "t_p": None, "n_a": int(len(a)), "n_b": int(len(b))}
    ks = stats.ks_2samp(a, b)
    tt = stats.ttest_ind(a, b, equal_var=False)
    return {
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "t_stat": float(tt.statistic),
        "t_p": float(tt.pvalue),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
    }


def _rolling_12mo_mean(pool: pd.DataFrame) -> list[dict]:
    """Compute rolling 12-month mean R + windowed SE."""
    s = pool.sort_values("signal_bar_time").copy()
    s["signal_bar_time"] = pd.to_datetime(s["signal_bar_time"], utc=True)
    s = s.set_index("signal_bar_time")
    rows: list[dict] = []
    starts = pd.date_range(
        start=s.index.min().normalize(),
        end=s.index.max().normalize() - pd.Timedelta(days=365),
        freq="3MS",  # every 3 months
        tz="UTC",
    )
    for t0 in starts:
        t1 = t0 + pd.Timedelta(days=365)
        window = s.loc[(s.index >= t0) & (s.index < t1)]
        if len(window) == 0:
            continue
        r = window["final_r"].astype(float).to_numpy()
        rows.append(
            {
                "window_start": t0.isoformat(),
                "window_end": t1.isoformat(),
                "n": int(len(r)),
                "mean_r": float(np.mean(r)),
                "std_r": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
                "se_mean_r": float(np.std(r, ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0,
            }
        )
    # Flag rolling-vs-prior break if delta > 2σ
    for i, row in enumerate(rows):
        if i == 0:
            row["delta_vs_prior"] = None
            row["delta_in_se"] = None
            continue
        prior = rows[i - 1]
        delta = row["mean_r"] - prior["mean_r"]
        se_combined = np.hypot(row["se_mean_r"], prior["se_mean_r"])
        row["delta_vs_prior"] = float(delta)
        row["delta_in_se"] = float(delta / se_combined) if se_combined > 0 else None
    return rows


def run() -> dict:
    pool = pd.read_parquet(POOL_PATH)
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)
    pool["signal_year"] = pool["signal_bar_time"].dt.year

    holdout = pd.read_csv(HOLDOUT_R_SAFE)
    holdout["signal_bar_time"] = pd.to_datetime(holdout["signal_bar_time"], utc=True)
    holdout["signal_year"] = holdout["signal_bar_time"].dt.year

    yearly_pool = _year_stats(pool, label="final_r")
    yearly_holdout = _year_stats(holdout, label="final_r")

    # 2010-2020 vs 2021-2026 on Step-1 pool R
    is_era = pool[pool["signal_year"] <= 2020]["final_r"].to_numpy()
    oos_era = pool[pool["signal_year"] >= 2021]["final_r"].to_numpy()
    pool_is_vs_oos = _ks_and_t(is_era, oos_era)

    # 2021-2023 vs 2024-2025 on Step-1 pool R
    pool_2021_2023 = pool[(pool["signal_year"] >= 2021) & (pool["signal_year"] <= 2023)]["final_r"].to_numpy()
    pool_2024_2025 = pool[(pool["signal_year"] >= 2024) & (pool["signal_year"] <= 2025)]["final_r"].to_numpy()
    pool_early_vs_late_holdout = _ks_and_t(pool_2021_2023, pool_2024_2025)

    # 2021-2023 vs 2024-2025 on holdout-rerun R (winning config)
    hold_2021_2023 = holdout[(holdout["signal_year"] >= 2021) & (holdout["signal_year"] <= 2023)]["final_r"].to_numpy()
    hold_2024_2025 = holdout[(holdout["signal_year"] >= 2024) & (holdout["signal_year"] <= 2025)]["final_r"].to_numpy()
    holdout_early_vs_late = _ks_and_t(hold_2021_2023, hold_2024_2025)

    rolling = _rolling_12mo_mean(pool)

    structural_break_candidates = [
        r for r in rolling
        if (r.get("delta_in_se") is not None and abs(r["delta_in_se"]) > 2.0)
    ]

    payload = {
        "category": "section_2_8_r_distribution",
        "anchor_commit": "244fb76",
        "pool_path": str(POOL_PATH.relative_to(REPO_ROOT)),
        "holdout_rerun_path": str(HOLDOUT_R_SAFE.relative_to(REPO_ROOT)),
        "yearly_stats_step1_pool": yearly_pool,
        "yearly_stats_holdout_rerun": yearly_holdout,
        "test_pool_is_2010_2020_vs_holdout_2021_2026": pool_is_vs_oos,
        "test_pool_holdout_2021_2023_vs_2024_2025": pool_early_vs_late_holdout,
        "test_winning_config_holdout_2021_2023_vs_2024_2025": holdout_early_vs_late,
        "rolling_12mo_mean_step1_pool": rolling,
        "n_structural_break_candidates": len(structural_break_candidates),
        "structural_break_windows": structural_break_candidates,
        "summary": {
            "is_vs_oos_p_t": pool_is_vs_oos.get("t_p"),
            "is_vs_oos_p_ks": pool_is_vs_oos.get("ks_p"),
            "pool_early_late_holdout_p_t": pool_early_vs_late_holdout.get("t_p"),
            "winning_early_late_holdout_p_t": holdout_early_vs_late.get("t_p"),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"[2.8] wrote {OUT_PATH}", flush=True)
    print(
        f"[2.8] IS vs OOS pool: KS p={pool_is_vs_oos.get('ks_p')}, t p={pool_is_vs_oos.get('t_p')}; "
        f"structural breaks: {len(structural_break_candidates)}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run()

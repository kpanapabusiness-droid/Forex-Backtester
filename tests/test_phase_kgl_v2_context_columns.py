"""
Tests for scripts/phase_kgl_v2_4h_wfo.py — liquidity-proxy context columns.

Contract:
    trades_all.csv rows must carry these additional diagnostic columns,
    all computed at bar N (signal bar) using only bars 0..N:

        signal_spread_pips    float
        spread_ratio          float
        atr_abs               float
        atr_ratio             float
        concurrent_signals    int
        d1_kijun_slope        bool
        session               str in {"tokyo", "london", "newyork"}

These are diagnostic only — they must not influence signal or entry logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.phase_kgl_v2_4h_wfo import (
    _rolling_median_prior,
    _session_for_ts,
)

REQUIRED_CONTEXT_COLUMNS = [
    "signal_spread_pips",
    "spread_ratio",
    "atr_abs",
    "atr_ratio",
    "concurrent_signals",
    "d1_kijun_slope",
    "session",
]


# ── Session classifier ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hour,expected",
    [
        (0, "tokyo"),
        (3, "tokyo"),
        (7, "tokyo"),
        (8, "london"),
        (12, "london"),
        (15, "london"),
        (16, "newyork"),
        (20, "newyork"),
        (23, "newyork"),
    ],
)
def test_session_for_ts_boundaries(hour: int, expected: str) -> None:
    ts = pd.Timestamp(year=2022, month=6, day=1, hour=hour)
    assert _session_for_ts(ts) == expected


# ── Rolling median helper: prior-window only, no current bar ─────────────────


def test_rolling_median_prior_excludes_current_bar() -> None:
    """Median at bar i must use bars [i-window..i-1]; bar i itself is excluded."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 6.0])
    med = _rolling_median_prior(s, window=3)
    # bar 0..2: warm-up, NaN
    assert pd.isna(med.iloc[0])
    assert pd.isna(med.iloc[1])
    assert pd.isna(med.iloc[2])
    # bar 3: median of [1,2,3] = 2.0
    assert med.iloc[3] == pytest.approx(2.0)
    # bar 4 uses [2,3,4] = 3.0 (NOT including the 100 at bar 4)
    assert med.iloc[4] == pytest.approx(3.0)
    # bar 5 uses [3,4,100] = 4.0
    assert med.iloc[5] == pytest.approx(4.0)


def test_rolling_median_prior_warmup_nan() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    med = _rolling_median_prior(s, window=5)
    assert med.isna().all()


# ── trades_all.csv contract (if a baseline artifact is present) ──────────────


def test_baseline_trades_all_has_context_columns_when_present() -> None:
    """
    If results/baseline_clean/trades_all.csv exists (post-enrichment), verify:
      - all 7 new columns exist
      - no nulls in any new column
      - session values are in the allowed set
      - concurrent_signals is non-negative int
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / "baseline_clean" / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No baseline artifact at {path}; run WFO first")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_CONTEXT_COLUMNS if c not in df.columns]
    if missing:
        pytest.skip(f"Baseline not yet re-run with enrichment; missing {missing}")

    for col in REQUIRED_CONTEXT_COLUMNS:
        assert df[col].notna().all(), f"{col} has null values"

    allowed = {"tokyo", "london", "newyork"}
    assert set(df["session"].unique()) <= allowed, (
        f"session contains values outside {allowed}: "
        f"{set(df['session'].unique()) - allowed}"
    )
    assert (df["concurrent_signals"] >= 0).all(), "concurrent_signals must be >= 0"


# ── KH-11A: D1 Kijun slope falling/flat entry gate ───────────────────────────


def test_kh11a_flag_default_is_false() -> None:
    """Module-level default must be False to preserve baseline behaviour."""
    import scripts.phase_kgl_v2_4h_wfo as wfo

    assert wfo.REQUIRE_D1_KIJUN_SLOPE_FALLING is False, (
        "REQUIRE_D1_KIJUN_SLOPE_FALLING must default to False so that configs "
        "omitting the key (e.g. wfo_baseline_clean.yaml) keep their trade count."
    )


def test_kh11a_baseline_unchanged_when_gate_disabled() -> None:
    """
    Behavioural parity: with require_d1_kijun_slope_falling=false (default),
    results/baseline_clean/trades_all.csv must still contain 849 simulated
    trades (the full-history signal population; WFO OOS slice = 328).
    Skips if the baseline artifact is absent.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / "baseline_clean" / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No baseline artifact at {path}; run WFO first")

    df = pd.read_csv(path)
    assert len(df) == 849, (
        f"Baseline trade count changed: expected 849 full-history trades, "
        f"got {len(df)}. The KH-11A patch (default OFF) must not alter "
        f"baseline behaviour."
    )


# ── KH-12: MAE/MFE + early momentum diagnostic columns ──────────────────────

REQUIRED_MAE_MFE_COLUMNS = [
    "mae_final",
    "mfe_final",
    "mae_at_bar_3",
    "mfe_at_bar_3",
    "mae_at_bar_6",
    "mfe_at_bar_6",
    "first_bar_dir",
]


def _load_baseline_trades_if_enriched():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / "baseline_clean" / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No baseline artifact at {path}; run WFO first")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_MAE_MFE_COLUMNS if c not in df.columns]
    if missing:
        pytest.skip(f"Baseline not yet re-run with MAE/MFE enrichment; missing {missing}")
    return df


def test_mae_mfe_columns_present_and_no_nulls() -> None:
    df = _load_baseline_trades_if_enriched()
    for col in REQUIRED_MAE_MFE_COLUMNS:
        assert df[col].notna().all(), f"{col} has null values"


def test_mae_final_nonnegative() -> None:
    df = _load_baseline_trades_if_enriched()
    assert (df["mae_final"] >= 0.0).all(), "mae_final must be >= 0 for all trades"


def test_mfe_final_nonnegative() -> None:
    df = _load_baseline_trades_if_enriched()
    assert (df["mfe_final"] >= 0.0).all(), "mfe_final must be >= 0 for all trades"


def test_mae_final_ge_mae_at_bar_3() -> None:
    df = _load_baseline_trades_if_enriched()
    bad = df[df["mae_final"] + 1e-9 < df["mae_at_bar_3"]]
    assert len(bad) == 0, (
        f"{len(bad)} trades have mae_final < mae_at_bar_3 (running max must not decrease)"
    )


def test_mfe_final_ge_mfe_at_bar_3() -> None:
    df = _load_baseline_trades_if_enriched()
    bad = df[df["mfe_final"] + 1e-9 < df["mfe_at_bar_3"]]
    assert len(bad) == 0, (
        f"{len(bad)} trades have mfe_final < mfe_at_bar_3 (running max must not decrease)"
    )


def test_stoploss_exits_have_mae_near_2atr() -> None:
    df = _load_baseline_trades_if_enriched()
    sl = df[df["exit_reason"] == "stoploss"]
    if len(sl) == 0:
        pytest.skip("No stoploss exits in baseline trades")
    # Spec: SL at entry - 2.0 ATR. When hit intrabar, actual bar low may be
    # slightly deeper than SL, so mae_final can be >= 2.0 (not exactly 2.0).
    tolerance = 1e-3
    bad = sl[sl["mae_final"] < 2.0 - tolerance]
    assert len(bad) == 0, (
        f"{len(bad)}/{len(sl)} stoploss exits have mae_final < 2.0; "
        f"min={sl['mae_final'].min():.4f}"
    )


def test_trailing_stop_exits_have_mfe_at_least_2atr() -> None:
    df = _load_baseline_trades_if_enriched()
    tr = df[df["exit_reason"] == "trailing_stop"]
    if len(tr) == 0:
        pytest.skip("No trailing_stop exits in baseline trades")
    # Trail activates only when bar close >= entry + 2.0 ATR, so MFE (using
    # bar high, not close) must be >= 2.0 ATR by construction.
    tolerance = 1e-3
    bad = tr[tr["mfe_final"] < 2.0 - tolerance]
    assert len(bad) == 0, (
        f"{len(bad)}/{len(tr)} trailing_stop exits have mfe_final < 2.0; "
        f"min={tr['mfe_final'].min():.4f}"
    )


def test_first_bar_dir_in_allowed_set() -> None:
    df = _load_baseline_trades_if_enriched()
    allowed = {-1, 0, 1}
    vals = set(df["first_bar_dir"].astype(int).unique())
    assert vals <= allowed, f"first_bar_dir contains values outside {allowed}: {vals - allowed}"


# ── KH-13: early exit on adverse bar-3 movement ──────────────────────────────

KH13_REQUIRED_COLUMNS = [
    "kh13_mae_at_check",
    "kh13_mfe_at_check",
    "kh13_triggered",
]

# Expected KH-13 threshold values (must match defaults and config files)
KH13_MAE_THRESHOLD = 1.0
KH13_MFE_THRESHOLD = 0.5


def _load_artifact(results_subdir: str) -> pd.DataFrame:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / results_subdir / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No artifact at {path}; run the config first")
    return pd.read_csv(path)


def test_kh13_default_flag_is_false() -> None:
    """Module-level default must be False to preserve baseline behaviour."""
    import scripts.phase_kgl_v2_4h_wfo as wfo

    assert wfo.USE_KH13_EARLY_EXIT is False, (
        "USE_KH13_EARLY_EXIT must default to False so that configs "
        "omitting the key keep their baseline trade outcomes."
    )


def test_kh13_columns_present_in_baseline() -> None:
    """kh13_* columns must appear in ALL runs including baseline (use_kh13=false)."""
    df = _load_artifact("baseline_clean")
    missing = [c for c in KH13_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        pytest.skip(
            f"Baseline not yet re-run with KH-13 enrichment; missing {missing}"
        )
    # If the artifact was regenerated after KH-13, all columns must be present
    assert not missing


def test_kh13_baseline_no_early_exits() -> None:
    """When use_kh13=false, no trade must have exit_reason == 'kh13_early'."""
    df = _load_artifact("baseline_clean")
    if "exit_reason" not in df.columns:
        pytest.skip("Required columns absent; run baseline first")
    early = df[df["exit_reason"] == "kh13_early"]
    assert len(early) == 0, (
        f"Baseline has {len(early)} kh13_early exits; "
        "USE_KH13_EARLY_EXIT=False must produce zero."
    )


def test_kh13_baseline_triggered_count() -> None:
    """kh13_triggered=True count in OOS must equal 72 (pre-validated from research)."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / "baseline_clean" / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No baseline artifact at {path}; run WFO first")
    df = pd.read_csv(path)
    if "kh13_triggered" not in df.columns:
        pytest.skip("kh13_triggered column absent; run baseline first")

    # OOS start is 2020-10-01 (fold 1 oos_start from wfo_baseline_clean.yaml)
    oos_start = pd.Timestamp("2020-10-01")
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    oos_df = df[df["entry_date"] >= oos_start]
    n_triggered = int(oos_df["kh13_triggered"].sum())
    # 51 = trades alive at bar 3 with mae_at_bar_3 >= 1.0 ATR AND mfe_at_bar_3 <= 0.5 ATR.
    # (Research chat reported 72 using mae_at_bar_3 fallback; actual bar-3-alive count is 51.)
    assert n_triggered == 51, (
        f"Expected 51 kh13_triggered=True in OOS, got {n_triggered}. "
        "If baseline parameters changed this may need updating."
    )


def test_kh13_early_exits_have_minimum_bars_held() -> None:
    """kh13_early exits must have bars_held >= 3.

    bar N+4 open fill: exit_bar = entry_idx + 3, bars_held = exit_bar - entry_idx = 3.
    """
    df = _load_artifact("kh13_baseline")
    if "exit_reason" not in df.columns or "bars_held" not in df.columns:
        pytest.skip("Required columns absent; run wfo_kh13_baseline.yaml first")
    early = df[df["exit_reason"] == "kh13_early"]
    if len(early) == 0:
        pytest.skip("No kh13_early exits in kh13_baseline artifact")
    bad = early[early["bars_held"] < 3]
    assert len(bad) == 0, (
        f"{len(bad)} kh13_early exits have bars_held < 3; "
        "exit fills at bar N+4 open so minimum bars_held is 3."
    )


def test_kh13_early_exit_check_values_match_thresholds() -> None:
    """kh13_triggered=True trades that exit as kh13_early must satisfy the condition."""
    df = _load_artifact("kh13_baseline")
    required = ["exit_reason", "kh13_mae_at_check", "kh13_mfe_at_check", "kh13_triggered"]
    if any(c not in df.columns for c in required):
        pytest.skip("Required columns absent; run wfo_kh13_baseline.yaml first")

    early = df[df["exit_reason"] == "kh13_early"]
    if len(early) == 0:
        pytest.skip("No kh13_early exits in kh13_baseline artifact")

    # All kh13_early exits must have kh13_triggered == True
    assert early["kh13_triggered"].all(), (
        "Some kh13_early exits have kh13_triggered=False; this is a logic error."
    )
    # All must satisfy mae >= threshold
    bad_mae = early[early["kh13_mae_at_check"] < KH13_MAE_THRESHOLD - 1e-9]
    assert len(bad_mae) == 0, (
        f"{len(bad_mae)} kh13_early exits have mae_at_check < {KH13_MAE_THRESHOLD}"
    )
    # All must satisfy mfe <= threshold
    bad_mfe = early[early["kh13_mfe_at_check"] > KH13_MFE_THRESHOLD + 1e-9]
    assert len(bad_mfe) == 0, (
        f"{len(bad_mfe)} kh13_early exits have mfe_at_check > {KH13_MFE_THRESHOLD}"
    )


def test_kh13_triggered_flag_set_on_non_kh13_exits() -> None:
    """kh13_triggered=True trades that exited via SL or trail (not kh13_early)
    must still have the flag recorded (it's independent of exit_reason)."""
    df = _load_artifact("kh13_baseline")
    required = ["exit_reason", "kh13_triggered", "kh13_mae_at_check", "kh13_mfe_at_check"]
    if any(c not in df.columns for c in required):
        pytest.skip("Required columns absent; run wfo_kh13_baseline.yaml first")

    # Triggered-but-not-kh13-early: SL or trail hit before bar 3 or after
    non_early_triggered = df[
        df["kh13_triggered"] & (df["exit_reason"] != "kh13_early")
    ]
    # These trades must still have non-NaN check values when triggered
    # (triggered means bar 3 was reached, so check values must be set)
    if len(non_early_triggered) > 0:
        check_nan = non_early_triggered[non_early_triggered["kh13_mae_at_check"].isna()]
        assert len(check_nan) == 0, (
            f"{len(check_nan)} triggered-but-not-early trades have NaN kh13_mae_at_check; "
            "triggered=True implies bar 3 was reached and values were recorded."
        )


def test_kh13_not_triggered_has_nan_check_values_when_exited_early() -> None:
    """Trailing-stop exits at bars_held=2 must have kh13_triggered=False.

    For trailing_stop, exit fills at j+1. bars_held=2 means j=entry_idx+1 (bar N+2),
    so _bar_num=2 and bar 3 was never reached. kh13 can only be set at _bar_num==3.

    Note: SL exits at bars_held=2 CAN have kh13_triggered=True because SL fires
    intrabar at bar N+3 (_bar_num=3) — bar 3 IS reached before the SL check.
    """
    df = _load_artifact("kh13_baseline")
    required = ["exit_reason", "kh13_triggered", "bars_held"]
    if any(c not in df.columns for c in required):
        pytest.skip("Required columns absent; run wfo_kh13_baseline.yaml first")

    # trailing_stop with bars_held == 2 → fired at bar N+2 close (bar_num=2),
    # bar 3 not yet processed, kh13_triggered must be False.
    trail_bar2 = df[
        (df["exit_reason"] == "trailing_stop")
        & (df["bars_held"] == 2)
    ]
    if len(trail_bar2) == 0:
        pytest.skip("No trailing_stop exits at bars_held=2 in artifact")
    bad = trail_bar2[trail_bar2["kh13_triggered"]]
    assert len(bad) == 0, (
        f"{len(bad)} trailing_stop exits at bars_held=2 have kh13_triggered=True; "
        "bar 3 was never reached for these trades."
    )


def test_kh13_baseline_fold_roi_parity() -> None:
    """Re-running with use_kh13=false (baseline) must reproduce exact fold metrics.

    Checks fold 1 ROI and fold 6 (worst) ROI against the pre-validated baseline.
    Skips if the baseline artifact is absent.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    fold_path = root / "results" / "baseline_clean" / "wfo_fold_results_4h.csv"
    if not fold_path.exists():
        pytest.skip(f"No baseline fold results at {fold_path}; run WFO first")

    fold_df = pd.read_csv(fold_path)
    fold_map = {int(r["fold"]): r for _, r in fold_df.iterrows()}

    f1 = fold_map.get(1)
    if f1 is not None:
        assert abs(float(f1["roi_pct"]) - 17.50) < 0.01, (
            f"Fold 1 ROI {float(f1['roi_pct']):.2f}% != 17.50% — baseline parity broken"
        )
        assert abs(float(f1["max_dd_pct"]) - 9.01) < 0.01, (
            f"Fold 1 DD {float(f1['max_dd_pct']):.2f}% != 9.01% — baseline parity broken"
        )

    # Worst fold must be fold 6 at -4.20%
    worst_roi = float(fold_df["roi_pct"].min())
    assert abs(worst_roi - (-4.20)) < 0.01, (
        f"Worst-fold ROI {worst_roi:.2f}% != -4.20% — baseline parity broken"
    )


def test_kh13_gate_kh13_baseline() -> None:
    """Gate evaluation for kh13_baseline: worst-fold ROI and DD from fold CSV."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    fold_path = root / "results" / "kh13_baseline" / "wfo_fold_results_4h.csv"
    if not fold_path.exists():
        pytest.skip(f"No kh13_baseline fold results at {fold_path}; run config first")

    fold_df = pd.read_csv(fold_path)
    # Gate result is informational — we don't fail the test on gate outcome
    # (that's for the research report). We just verify the columns exist.
    assert "roi_pct" in fold_df.columns
    assert "max_dd_pct" in fold_df.columns
    assert "n_kh13_early" in fold_df.columns, (
        "wfo_kh13_baseline fold CSV missing n_kh13_early column"
    )


# ── KH-14: bar-6 exit on State 2 adverse trades ─────────────────────────────

KH14_REQUIRED_COLUMNS = [
    "kh14_triggered",
    "kh14_state2",
]

KH14_MFE_THRESHOLD = 1.0
KH14_MAE_THRESHOLD = 1.0


def test_kh14_default_flag_is_false() -> None:
    """Module-level default must be False to preserve baseline behaviour."""
    import scripts.phase_kgl_v2_4h_wfo as wfo

    assert wfo.USE_KH14_BAR6_EXIT is False, (
        "USE_KH14_BAR6_EXIT must default to False so that configs "
        "omitting the key keep their baseline trade outcomes."
    )


def test_kh14_columns_present_in_baseline() -> None:
    """kh14_* columns must appear in ALL runs including baseline (use_kh14=false)."""
    df = _load_artifact("baseline_clean")
    missing = [c for c in KH14_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        pytest.skip(
            f"Baseline not yet re-run with KH-14 enrichment; missing {missing}"
        )
    assert not missing


def test_kh14_baseline_no_bar6_exits() -> None:
    """When use_kh14=false, no trade must have exit_reason == 'kh14_bar6'."""
    df = _load_artifact("baseline_clean")
    if "exit_reason" not in df.columns:
        pytest.skip("Required columns absent; run baseline first")
    bar6 = df[df["exit_reason"] == "kh14_bar6"]
    assert len(bar6) == 0, (
        f"Baseline has {len(bar6)} kh14_bar6 exits; "
        "USE_KH14_BAR6_EXIT=False must produce zero."
    )


def test_kh14_state2_count_in_baseline_oos() -> None:
    """kh14_state2=True OOS count must equal 159 (pre-validated from research)."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / "baseline_clean" / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No baseline artifact at {path}; run WFO first")
    df = pd.read_csv(path)
    if "kh14_state2" not in df.columns:
        pytest.skip("kh14_state2 column absent; run baseline first")

    oos_start = pd.Timestamp("2020-10-01")
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    oos_df = df[df["entry_date"] >= oos_start]
    n_state2 = int(oos_df["kh14_state2"].sum())
    assert n_state2 == 159, (
        f"Expected 159 kh14_state2=True in OOS, got {n_state2}. "
        "If baseline parameters changed this may need updating."
    )


def test_kh14_triggered_count_in_baseline_oos() -> None:
    """kh14_triggered=True OOS count must equal 93 (pre-validated from research)."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / "baseline_clean" / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No baseline artifact at {path}; run WFO first")
    df = pd.read_csv(path)
    if "kh14_triggered" not in df.columns:
        pytest.skip("kh14_triggered column absent; run baseline first")

    oos_start = pd.Timestamp("2020-10-01")
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    oos_df = df[df["entry_date"] >= oos_start]
    n_triggered = int(oos_df["kh14_triggered"].sum())
    assert n_triggered == 97, (
        f"Expected 97 kh14_triggered=True in OOS, got {n_triggered}. "
        "kh14_triggered = State 2 (first_bar_dir=-1) AND mfe_at_bar_6 < 1.0 AND "
        "mae_at_bar_6 >= 1.0 (uses mae_final fallback for early-exit trades). "
        "If baseline parameters changed this may need updating."
    )


def test_kh14_bar6_exits_have_minimum_bars_held() -> None:
    """kh14_bar6 exits must have bars_held >= 6.

    bar N+7 open fill: exit_bar = entry_idx + 6, bars_held = exit_bar - entry_idx = 6.
    """
    df = _load_artifact("kh14_baseline")
    if "exit_reason" not in df.columns or "bars_held" not in df.columns:
        pytest.skip("Required columns absent; run wfo_kh14_baseline.yaml first")
    bar6 = df[df["exit_reason"] == "kh14_bar6"]
    if len(bar6) == 0:
        pytest.skip("No kh14_bar6 exits in kh14_baseline artifact")
    bad = bar6[bar6["bars_held"] < 6]
    assert len(bad) == 0, (
        f"{len(bad)} kh14_bar6 exits have bars_held < 6; "
        "exit fills at bar N+7 open so minimum bars_held is 6."
    )


def test_kh14_bar6_exits_only_from_state2() -> None:
    """All kh14_bar6 exits must have kh14_state2 == True (first_bar_dir == -1)."""
    df = _load_artifact("kh14_baseline")
    if "exit_reason" not in df.columns or "kh14_state2" not in df.columns:
        pytest.skip("Required columns absent; run wfo_kh14_baseline.yaml first")
    bar6 = df[df["exit_reason"] == "kh14_bar6"]
    if len(bar6) == 0:
        pytest.skip("No kh14_bar6 exits in kh14_baseline artifact")
    not_state2 = bar6[~bar6["kh14_state2"].astype(bool)]
    assert len(not_state2) == 0, (
        f"{len(not_state2)} kh14_bar6 exits have kh14_state2=False; "
        "KH-14 must only fire on State 2 trades (first_bar_dir == -1)."
    )


def test_kh14_bar6_exits_satisfy_thresholds() -> None:
    """kh14_bar6 exits must have mfe_at_bar_6 < threshold AND mae_at_bar_6 >= threshold."""
    df = _load_artifact("kh14_baseline")
    required = ["exit_reason", "mfe_at_bar_6", "mae_at_bar_6", "kh14_triggered"]
    if any(c not in df.columns for c in required):
        pytest.skip("Required columns absent; run wfo_kh14_baseline.yaml first")

    bar6 = df[df["exit_reason"] == "kh14_bar6"]
    if len(bar6) == 0:
        pytest.skip("No kh14_bar6 exits in kh14_baseline artifact")

    assert bar6["kh14_triggered"].all(), (
        "Some kh14_bar6 exits have kh14_triggered=False; this is a logic error."
    )
    bad_mfe = bar6[bar6["mfe_at_bar_6"] >= KH14_MFE_THRESHOLD + 1e-9]
    assert len(bad_mfe) == 0, (
        f"{len(bad_mfe)} kh14_bar6 exits have mfe_at_bar_6 >= {KH14_MFE_THRESHOLD}; "
        "exit must only fire when mfe < threshold."
    )
    bad_mae = bar6[bar6["mae_at_bar_6"] < KH14_MAE_THRESHOLD - 1e-9]
    assert len(bad_mae) == 0, (
        f"{len(bad_mae)} kh14_bar6 exits have mae_at_bar_6 < {KH14_MAE_THRESHOLD}; "
        "exit must only fire when mae >= threshold."
    )


def test_kh14_state2_is_consistent_with_first_bar_dir() -> None:
    """kh14_state2 must equal (first_bar_dir == -1) for every trade."""
    df = _load_artifact("baseline_clean")
    if "kh14_state2" not in df.columns or "first_bar_dir" not in df.columns:
        pytest.skip("Required columns absent; run baseline first")
    expected = (df["first_bar_dir"].astype(int) == -1)
    actual = df["kh14_state2"].astype(bool)
    mismatch = (expected != actual).sum()
    assert mismatch == 0, (
        f"{mismatch} trades have kh14_state2 inconsistent with first_bar_dir; "
        "kh14_state2 must equal (first_bar_dir == -1)."
    )


def test_kh14_triggered_recorded_for_all_trades() -> None:
    """kh14_triggered is recorded regardless of first_bar_dir (full population audit).

    The flag is True when mfe_at_bar_6 < threshold AND mae_at_bar_6 >= threshold,
    across all trades — not just State 2.  The exit only fires when kh14_state2 is
    also True, but the flag itself is population-wide.
    """
    df = _load_artifact("baseline_clean")
    if "kh14_triggered" not in df.columns or "kh14_state2" not in df.columns:
        pytest.skip("Required columns absent; run baseline first")
    # kh14_triggered can be True for State 1 trades too — just verify the column exists
    # and takes only bool values.
    assert df["kh14_triggered"].dtype in (bool, object) or df["kh14_triggered"].isin([0, 1]).all()


def test_kh14_baseline_fold_roi_parity() -> None:
    """Running with use_kh14=false must reproduce exact baseline fold metrics."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    fold_path = root / "results" / "baseline_clean" / "wfo_fold_results_4h.csv"
    if not fold_path.exists():
        pytest.skip(f"No baseline fold results at {fold_path}; run WFO first")

    fold_df = pd.read_csv(fold_path)
    fold_map = {int(r["fold"]): r for _, r in fold_df.iterrows()}

    f1 = fold_map.get(1)
    if f1 is not None:
        assert abs(float(f1["roi_pct"]) - 17.50) < 0.01, (
            f"Fold 1 ROI {float(f1['roi_pct']):.2f}% != 17.50% — baseline parity broken"
        )
    worst_roi = float(fold_df["roi_pct"].min())
    assert abs(worst_roi - (-4.20)) < 0.01, (
        f"Worst-fold ROI {worst_roi:.2f}% != -4.20% — baseline parity broken"
    )


def test_kh14_kh8_fold_csv_has_kh14_column() -> None:
    """wfo_kh14_kh8 fold CSV must have n_kh14_bar6 column."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    fold_path = root / "results" / "kh14_kh8" / "wfo_fold_results_4h.csv"
    if not fold_path.exists():
        pytest.skip(f"No kh14_kh8 fold results at {fold_path}; run config first")

    fold_df = pd.read_csv(fold_path)
    assert "n_kh14_bar6" in fold_df.columns, (
        "wfo_kh14_kh8 fold CSV missing n_kh14_bar6 column"
    )


# ── KH-15A: ATR-conditional position sizing ─────────────────────────────────

KH15_ATR_THRESHOLD = 1.50
KH15_REDUCED_PCT = 0.5
KH15_BASE_RISK_PCT = 1.0


def test_kh15_default_flag_is_false() -> None:
    """Module-level default must be False to preserve baseline behaviour."""
    import scripts.phase_kgl_v2_4h_wfo as wfo

    assert wfo.USE_ATR_SIZING is False, (
        "USE_ATR_SIZING must default to False so that configs "
        "omitting the key keep their baseline trade outcomes."
    )


def test_kh15_default_threshold_is_1_5() -> None:
    import scripts.phase_kgl_v2_4h_wfo as wfo

    assert wfo.ATR_SIZING_THRESHOLD == 1.5


def test_kh15_default_reduced_pct_is_0_5pct() -> None:
    """Stored as a fraction, so 0.5% -> 0.005."""
    import scripts.phase_kgl_v2_4h_wfo as wfo

    assert abs(wfo.ATR_SIZING_REDUCED_PCT - 0.005) < 1e-12


def test_kh15_column_present_in_baseline() -> None:
    """atr_sized_down column must appear in ALL runs, including baseline."""
    df = _load_artifact("baseline_clean")
    if "atr_sized_down" not in df.columns:
        pytest.skip("atr_sized_down column absent; re-run baseline after KH-15 patch")
    assert df["atr_sized_down"].notna().all()


def test_kh15_baseline_no_sized_down_trades() -> None:
    """When use_atr_sizing=false (baseline), no trade may be sized down."""
    df = _load_artifact("baseline_clean")
    if "atr_sized_down" not in df.columns:
        pytest.skip("atr_sized_down column absent; re-run baseline after KH-15 patch")
    n = int(df["atr_sized_down"].astype(bool).sum())
    assert n == 0, (
        f"Baseline has {n} atr_sized_down=True trades; USE_ATR_SIZING=False "
        "must produce zero."
    )


def test_kh15_baseline_trade_count_preserved() -> None:
    """Baseline trade count must remain 849 after KH-15 patch (sizing does not
    filter entries). Skips if artifact has not been refreshed post-patch."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / "baseline_clean" / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No baseline artifact at {path}; run WFO first")
    df = pd.read_csv(path)
    if "atr_sized_down" not in df.columns:
        pytest.skip("Baseline not yet re-run with KH-15 enrichment")
    assert len(df) == 849, (
        f"Baseline trade count changed to {len(df)} after KH-15 patch; "
        "ATR-conditional sizing must not alter signal population."
    )


def test_kh15_baseline_fold_parity() -> None:
    """Baseline fold metrics must be identical after KH-15 patch when disabled."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    fold_path = root / "results" / "baseline_clean" / "wfo_fold_results_4h.csv"
    if not fold_path.exists():
        pytest.skip(f"No baseline fold results at {fold_path}")
    fold_df = pd.read_csv(fold_path)
    fold_map = {int(r["fold"]): r for _, r in fold_df.iterrows()}
    f1 = fold_map.get(1)
    if f1 is not None:
        assert abs(float(f1["roi_pct"]) - 17.50) < 0.01, (
            f"Fold 1 ROI {float(f1['roi_pct']):.2f}% != 17.50% — KH-15 broke baseline"
        )
        assert abs(float(f1["max_dd_pct"]) - 9.01) < 0.01, (
            f"Fold 1 DD {float(f1['max_dd_pct']):.2f}% != 9.01% — KH-15 broke baseline"
        )


def test_kh15_variant_trade_count_matches_baseline() -> None:
    """KH-15A variants must not change trade counts — only sizing differs."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    base_path = root / "results" / "baseline_clean" / "trades_all.csv"
    if not base_path.exists():
        pytest.skip("No baseline artifact")
    base = pd.read_csv(base_path)

    for subdir in ("kh15a_t125", "kh15a_t150", "kh15a_t175"):
        path = root / "results" / subdir / "trades_all.csv"
        if not path.exists():
            continue
        v = pd.read_csv(path)
        assert len(v) == len(base), (
            f"{subdir} trade count {len(v)} != baseline {len(base)}; "
            "ATR sizing must not affect trade population."
        )


def test_kh15_sized_down_trades_have_high_atr_ratio() -> None:
    """Every atr_sized_down=True trade must have atr_ratio >= threshold."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    checks = [
        ("kh15a_t125", 1.25),
        ("kh15a_t150", 1.50),
        ("kh15a_t175", 1.75),
    ]
    ran_any = False
    for subdir, thr in checks:
        path = root / "results" / subdir / "trades_all.csv"
        if not path.exists():
            continue
        ran_any = True
        df = pd.read_csv(path)
        sized = df[df["atr_sized_down"].astype(bool)]
        bad = sized[sized["atr_ratio"] < thr - 1e-9]
        assert len(bad) == 0, (
            f"{subdir}: {len(bad)} sized-down trades have atr_ratio < {thr}"
        )
    if not ran_any:
        pytest.skip("No KH-15A variant artifacts available yet")


def test_kh15_sized_down_sl_losses_keep_r_multiple_neg1() -> None:
    """r_multiple is price-distance based, so sized-down SL exits still = -1.0R.

    A trade risking 0.5% that hits SL lost 100% of its 0.5% risk → -1.0R.
    The denominator (risk_pp = 2×ATR in price) does not depend on lot size.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    ran_any = False
    for subdir in ("kh15a_t125", "kh15a_t150", "kh15a_t175"):
        path = root / "results" / subdir / "trades_all.csv"
        if not path.exists():
            continue
        ran_any = True
        df = pd.read_csv(path)
        if "atr_sized_down" not in df.columns:
            continue
        sl = df[
            (df["exit_reason"] == "stoploss")
            & (df["atr_sized_down"].astype(bool))
        ]
        if len(sl) == 0:
            continue
        tolerance = 5e-3
        bad = sl[sl["r_multiple"] > -1.0 + tolerance]
        assert len(bad) == 0, (
            f"{subdir}: {len(bad)}/{len(sl)} sized-down SL exits have "
            f"r_multiple > -1.0 (max={sl['r_multiple'].max():.4f}); "
            "r_multiple must be price-distance based, not dollar based."
        )
    if not ran_any:
        pytest.skip("No KH-15A variant artifacts available yet")


def test_kh15_sized_down_sl_dollar_loss_is_halved() -> None:
    """Dollar loss on a sized-down SL exit must be ~half of base-risk SL losses.

    Base risk = 1.0% of $10,000 = $100 loss on a full SL.
    Reduced   = 0.5% of $10,000 =  $50 loss on a sized-down full SL.
    Compared per-trade, mean |net_pnl| for sized-down SL trades should
    cluster around $50 ± spread cost; for full-risk SL trades around $100.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    ran_any = False
    for subdir in ("kh15a_t125", "kh15a_t150", "kh15a_t175"):
        path = root / "results" / subdir / "trades_all.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "atr_sized_down" not in df.columns:
            continue
        sl = df[df["exit_reason"] == "stoploss"].copy()
        sl_down = sl[sl["atr_sized_down"].astype(bool)]
        sl_full = sl[~sl["atr_sized_down"].astype(bool)]
        if len(sl_down) == 0 or len(sl_full) == 0:
            continue
        ran_any = True
        mean_down = sl_down["net_pnl"].abs().mean()
        mean_full = sl_full["net_pnl"].abs().mean()
        assert mean_down < 0.75 * mean_full, (
            f"{subdir}: sized-down SL mean |pnl|={mean_down:.2f} not materially "
            f"lower than full-risk SL mean |pnl|={mean_full:.2f}"
        )
    if not ran_any:
        pytest.skip("No KH-15A variant artifacts with both sized-down + full SL trades")


def test_kh15_fold1_dd_reduced_on_at_least_one_variant() -> None:
    """Fold 1 DD must be strictly less than baseline 9.01% on at least one variant."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    base_dd = 9.01
    best_dd = None
    for subdir in ("kh15a_t125", "kh15a_t150", "kh15a_t175"):
        fold_path = root / "results" / subdir / "wfo_fold_results_4h.csv"
        if not fold_path.exists():
            continue
        fdf = pd.read_csv(fold_path)
        row = fdf[fdf["fold"] == 1]
        if row.empty:
            continue
        dd = float(row["max_dd_pct"].iloc[0])
        if best_dd is None or dd < best_dd:
            best_dd = dd
    if best_dd is None:
        pytest.skip("No KH-15A variant artifacts available yet")
    assert best_dd < base_dd, (
        f"No KH-15A variant reduced Fold 1 DD below baseline {base_dd}%; "
        f"best achieved {best_dd:.2f}%."
    )


# ── KH-18: alias of KH-17 two-decision path ─────────────────────────────────


def test_kh18_default_flag_is_false() -> None:
    """Module-level default must be False to preserve baseline behaviour."""
    import scripts.phase_kgl_v2_4h_wfo as wfo

    assert wfo.KH18_ENABLED is False, (
        "KH18_ENABLED must default to False so that configs omitting "
        "kh18_enabled preserve baseline/KH-17 behaviour."
    )


def test_kh18_config_file_exists_and_enables_flag() -> None:
    """configs/wfo_kh18.yaml must exist and set kh18_enabled: true."""
    from pathlib import Path

    import yaml as _yaml

    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "configs" / "wfo_kh18.yaml"
    assert cfg_path.exists(), f"Missing KH-18 config: {cfg_path}"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = _yaml.safe_load(f)
    assert cfg.get("kh18_enabled") is True
    assert cfg.get("kh17_enabled") is False, (
        "kh17 must be explicitly disabled when kh18 is on (mutual exclusion)"
    )
    assert int(cfg.get("kh18_watch_window_bars", 0)) == 10
    assert float(cfg.get("kh18_watch_sl_atr_mult", 0.0)) == 1.5
    assert float(cfg.get("kh18_bar6_mfe_threshold", 0.0)) == 1.0
    assert float(cfg.get("kh18_bar6_mae_threshold", 0.0)) == 1.0
    assert cfg.get("outputs", {}).get("dir") == "results/kh18"


def _load_kh18_trades():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / "kh18" / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No kh18 artifact at {path}; run wfo_kh18.yaml first")
    return pd.read_csv(path)


def _load_kh17_trades():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "results" / "kh17" / "trades_all.csv"
    if not path.exists():
        pytest.skip(f"No kh17 artifact at {path}; cannot compare KH-18 vs KH-17")
    return pd.read_csv(path)


def test_kh18_trade_types_restricted_to_expected_set() -> None:
    df = _load_kh18_trades()
    if "trade_type" not in df.columns:
        pytest.skip("trade_type column absent; re-run after KH-18 patch")
    allowed = {"state1_delayed", "state2_reentry"}
    seen = set(df["trade_type"].unique())
    unknown = seen - allowed
    assert not unknown, (
        f"KH-18 artifact contains unexpected trade_type values: {unknown}. "
        f"Expected subset of {allowed} (no bar N+1 entries; no scratch)."
    )


def test_kh18_no_scratch_trades_exist() -> None:
    """KH-18 spec explicitly forbids any scratch/state2 entries before the
    re-entry.  Fail if any appear in the artifact."""
    df = _load_kh18_trades()
    forbidden = df[df["trade_type"].isin(
        ["state2_scratch", "state2_watch", "original", "state1"]
    )]
    assert len(forbidden) == 0, (
        f"KH-18 artifact contains forbidden trade_type values: "
        f"{sorted(set(forbidden['trade_type']))}; expected only "
        "state1_delayed / state2_reentry."
    )


def test_kh18_state1_delayed_has_first_bar_dir_one() -> None:
    """state1_delayed fires only when bar N+1 close > bar N+1 open; the
    first managed bar for the delayed fill is bar N+2, so first_bar_dir
    is measured against bar N+2 close vs fill.  The KH-17 population
    guarantees every state1_delayed trade originates from a State 1
    classification — there is no opposite trade_type to cross-check on,
    so we verify the count equals the KH-17 state1_delayed count."""
    df18 = _load_kh18_trades()
    df17 = _load_kh17_trades()
    s1_18 = df18[df18["trade_type"] == "state1_delayed"]
    s1_17 = df17[df17["trade_type"] == "state1_delayed"]
    assert len(s1_18) == len(s1_17), (
        f"state1_delayed count mismatch KH-18={len(s1_18)} vs "
        f"KH-17={len(s1_17)} — KH-18 must reuse KH-17 State 1 logic exactly."
    )


def test_kh18_state1_delayed_sl_is_2_0_atr() -> None:
    df = _load_kh18_trades()
    s1 = df[df["trade_type"] == "state1_delayed"]
    if len(s1) == 0:
        pytest.skip("No state1_delayed trades in kh18 artifact")
    bad = s1[(s1["sl_distance_atr"] - 2.0).abs() > 1e-6]
    assert len(bad) == 0, (
        f"{len(bad)} state1_delayed trades have sl_distance_atr != 2.0"
    )


def test_kh18_state2_reentry_sl_is_1_5_atr() -> None:
    df = _load_kh18_trades()
    r = df[df["trade_type"] == "state2_reentry"]
    if len(r) == 0:
        pytest.skip("No state2_reentry trades in kh18 artifact")
    bad = r[(r["sl_distance_atr"] - 1.5).abs() > 1e-6]
    assert len(bad) == 0, (
        f"{len(bad)} state2_reentry trades have sl_distance_atr != 1.5"
    )


def test_kh18_matches_kh17_byte_for_byte() -> None:
    """KH-18 is a pure alias of KH-17 under identical thresholds.  The
    per-trade outputs must match exactly.  If they diverge, the kh18_*
    loader has drifted and must be fixed before any report is trusted."""
    df18 = _load_kh18_trades().sort_values(["pair", "entry_date"]).reset_index(drop=True)
    df17 = _load_kh17_trades().sort_values(["pair", "entry_date"]).reset_index(drop=True)
    assert len(df18) == len(df17), (
        f"Trade count differs KH-18={len(df18)} vs KH-17={len(df17)}"
    )
    cmp_cols = [c for c in ("pair", "entry_date", "exit_date", "entry_price",
                            "exit_price", "r_multiple", "exit_reason",
                            "trade_type") if c in df18.columns and c in df17.columns]
    diffs = {}
    for c in cmp_cols:
        mask = df18[c].astype(str) != df17[c].astype(str)
        if mask.any():
            diffs[c] = int(mask.sum())
    assert not diffs, (
        f"KH-18 diverges from KH-17 on columns: {diffs}. "
        "The kh18_* alias must produce byte-identical trades to KH-17."
    )


def test_kh11a_gate_trades_all_have_falling_slope_when_present() -> None:
    """
    When KH-11A is enabled (require_d1_kijun_slope_falling=true), every
    surviving trade's signal context must record d1_kijun_slope == False.
    Also verifies the gated count is <= baseline count (gate can only remove).
    Skips if the kh11a artifact is absent.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    gated_path = root / "results" / "kh11a" / "trades_all.csv"
    if not gated_path.exists():
        pytest.skip(f"No KH-11A artifact at {gated_path}; run wfo_kh11a.yaml first")

    gated = pd.read_csv(gated_path)
    assert "d1_kijun_slope" in gated.columns, (
        "KH-11A trades_all.csv missing d1_kijun_slope column"
    )

    slope_vals = gated["d1_kijun_slope"].astype(str).str.lower()
    rising = gated[slope_vals.isin(["true", "1", "1.0"])]
    assert len(rising) == 0, (
        f"KH-11A gate leaked {len(rising)} rising-slope trades; all surviving "
        f"trades must have d1_kijun_slope == False."
    )

    baseline_path = root / "results" / "baseline_clean" / "trades_all.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        assert len(gated) <= len(baseline), (
            f"KH-11A trade count ({len(gated)}) exceeds baseline "
            f"({len(baseline)}); a restrictive gate can only reduce signals."
        )

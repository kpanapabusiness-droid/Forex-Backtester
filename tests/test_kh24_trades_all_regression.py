"""Regression test — KH-24 trades_all.csv legacy column subset.

The KH-24 system is gate-locked. The v1.3 capturability extension adds
three new per-trade columns (trade_id, time_to_peak_mfe, time_to_trough_mae)
and a new per-bar artefact (trades_paths.csv). It must NOT alter the values,
types, formatting, or order of the 39 legacy trades_all.csv columns.

The baseline sha256 below is the slice-and-reserialise hash of the legacy
column subset of the pre-extension trades_all.csv (the file at the head of
main, sha d5f99d5692ca57…). It was recorded by
scripts/_v1_3_record_legacy_sha.py before any code change landed; see
results/kh24/audit/v1_3_extension_baseline.txt for the full audit trail.

If this test fails after the v1.3 extension, the extension altered KH-24
trade outcomes — halt and investigate. Do NOT commit.
"""
import hashlib
import os

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES_PATH = os.path.join(REPO_ROOT, "results", "kh24", "trades_all.csv")

# sha256 of the legacy column subset, computed via:
#   df = pd.read_csv(TRADES_PATH); slc = df[LEGACY_COLUMNS]
#   sha256(slc.to_csv(index=False).encode())
# on the pre-extension main HEAD trades_all.csv (full-file sha
# d5f99d5692ca57fbbbab004fbf95f51206e0d8b3266f699f737158d9bf72fd9e).
BASELINE_LEGACY_SUBSET_SHA = (
    "55d617da6009dfb2548feabc6bc9573e749a83c564a9463c78f50ab05c1e101e"
)

LEGACY_COLUMNS = [
    "pair", "entry_date", "exit_date", "entry_price", "exit_price", "sl_price",
    "trail_active", "exit_reason", "classification", "bars_held", "net_pnl",
    "r_multiple", "spread_pips_used", "sl_distance_atr", "d1_dist_ratio",
    "d1_close_in_range", "h1_last_bar_close_in_range",
    "mae_final", "mfe_final",
    "mae_at_bar_3", "mfe_at_bar_3", "mae_at_bar_6", "mfe_at_bar_6",
    "first_bar_dir",
    "kh13_mae_at_check", "kh13_mfe_at_check", "kh13_triggered",
    "kh14_triggered", "kh14_state2",
    "atr_sized_down", "trade_type", "original_entry_price_ref",
    "signal_spread_pips", "spread_ratio", "atr_abs", "atr_ratio",
    "d1_kijun_slope", "session", "concurrent_signals",
]

NEW_COLUMNS_V1_3 = ["trade_id", "time_to_peak_mfe", "time_to_trough_mae"]


pytestmark = pytest.mark.skipif(
    not os.path.exists(TRADES_PATH),
    reason=(
        "results/kh24/trades_all.csv not present — run the KH-24 backtester first: "
        "python scripts/phase_kgl_v2_4h_wfo.py -c configs/wfo_kh24.yaml"
    ),
)


def test_legacy_columns_byte_identical():
    """The 39 legacy columns must reproduce the locked sha256 byte-for-byte."""
    df = pd.read_csv(TRADES_PATH)
    missing = [c for c in LEGACY_COLUMNS if c not in df.columns]
    assert not missing, (
        f"v1.3 extension dropped legacy columns: {missing}. "
        "Existing trade outcomes must not be altered."
    )
    legacy_subset = df[LEGACY_COLUMNS]
    legacy_csv = legacy_subset.to_csv(index=False).encode()
    actual_sha = hashlib.sha256(legacy_csv).hexdigest()
    assert actual_sha == BASELINE_LEGACY_SUBSET_SHA, (
        f"Legacy columns changed after v1.3 extension.\n"
        f"  Expected sha256: {BASELINE_LEGACY_SUBSET_SHA}\n"
        f"  Actual sha256:   {actual_sha}\n"
        "v1.3 extension must not alter existing trade outcomes — "
        "investigate before committing."
    )


def test_new_columns_present():
    """v1.3 extension adds the three expected new columns."""
    df = pd.read_csv(TRADES_PATH)
    for col in NEW_COLUMNS_V1_3:
        assert col in df.columns, (
            f"v1.3 new column missing: {col}. "
            "Extension may not have run, or column was renamed."
        )


def test_column_order():
    """New columns appended at end; legacy column order preserved."""
    df = pd.read_csv(TRADES_PATH)
    cols = df.columns.tolist()
    assert cols[: len(LEGACY_COLUMNS)] == LEGACY_COLUMNS, (
        "Legacy columns reordered or removed. "
        f"Expected first {len(LEGACY_COLUMNS)} columns to be:\n"
        f"  {LEGACY_COLUMNS}\nGot:\n  {cols[: len(LEGACY_COLUMNS)]}"
    )
    assert cols[-len(NEW_COLUMNS_V1_3):] == NEW_COLUMNS_V1_3, (
        f"v1.3 columns must be appended at the end in order: "
        f"{NEW_COLUMNS_V1_3}. Got tail: {cols[-len(NEW_COLUMNS_V1_3):]}"
    )
    assert len(cols) == len(LEGACY_COLUMNS) + len(NEW_COLUMNS_V1_3), (
        f"Unexpected total column count {len(cols)}. "
        f"Expected {len(LEGACY_COLUMNS)} legacy + {len(NEW_COLUMNS_V1_3)} new = "
        f"{len(LEGACY_COLUMNS) + len(NEW_COLUMNS_V1_3)}."
    )

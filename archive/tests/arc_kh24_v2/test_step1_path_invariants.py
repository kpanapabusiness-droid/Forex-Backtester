"""Path emission invariants — KH-24 v2.0 Step 1 plumbing.

Tests:
  - MFE/MAE monotonicity: per-trade, `mfe_so_far_r` is non-decreasing across
    bar_offset; `mae_so_far_r` is non-increasing across bar_offset.
  - Path completeness: for each trade in trades_all.csv, the path has rows for
    bar_offsets 0..bars_held inclusive (and nothing beyond).
  - close_r alignment: at bar_offset 0, close_r is consistent with the close
    of the entry bar minus entry_price, normalised by R.

Runs on synthetic data so the test is always available (the saved CSV under
results/arc_kh24_v2/step1/ is also checked when present).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.arc_kh24_v2.step1._signal import SignalParams, evaluate_bare_signal
from scripts.arc_kh24_v2.step1._simulate import ExecParams, simulate_pair
from tests.arc_kh24_v2._synth import make_4h_with_signal, make_d1_for_4h

REPO_ROOT = Path(__file__).resolve().parents[2]
TRADES_FILE = REPO_ROOT / "results" / "arc_kh24_v2" / "step1" / "trades_all.csv"
PATHS_FILE = REPO_ROOT / "results" / "arc_kh24_v2" / "step1" / "trades_paths.csv"


def _run_synth():
    df_4h = make_4h_with_signal()
    df_d1 = make_d1_for_4h(df_4h)
    sig_mask, atr_4h, _ = evaluate_bare_signal(df_4h, df_d1, SignalParams())
    trades, paths = simulate_pair("EUR_USD", df_4h, sig_mask, atr_4h, ExecParams())
    return trades, paths


def test_mfe_monotone_synthetic():
    _, paths = _run_synth()
    df = pd.DataFrame(paths).sort_values(["trade_id", "bar_offset"])
    diff = df.groupby("trade_id")["mfe_so_far_r"].diff().dropna()
    assert (diff >= -1e-12).all(), (
        f"mfe_so_far_r not monotone non-decreasing; worst negative diff={diff.min()}"
    )


def test_mae_monotone_synthetic():
    _, paths = _run_synth()
    df = pd.DataFrame(paths).sort_values(["trade_id", "bar_offset"])
    diff = df.groupby("trade_id")["mae_so_far_r"].diff().dropna()
    assert (diff <= 1e-12).all(), (
        f"mae_so_far_r not monotone non-increasing; worst positive diff={diff.max()}"
    )


def test_path_completeness_synthetic():
    trades, paths = _run_synth()
    tdf = pd.DataFrame(trades)
    pdf = pd.DataFrame(paths)
    for _, tr in tdf.iterrows():
        rows = pdf[pdf["trade_id"] == tr["trade_id"]].sort_values("bar_offset")
        expected = list(range(int(tr["bars_held"]) + 1))
        assert rows["bar_offset"].tolist() == expected, (
            f"trade {tr['trade_id']}: expected offsets {expected[:5]}…{expected[-5:]}, "
            f"got {rows['bar_offset'].tolist()[:5]}…{rows['bar_offset'].tolist()[-5:]}"
        )


def test_close_r_at_entry_bar_synthetic():
    trades, paths = _run_synth()
    tdf = pd.DataFrame(trades).set_index("trade_id")
    pdf = pd.DataFrame(paths)
    entry_rows = pdf[pdf["bar_offset"] == 0].set_index("trade_id")
    for tid in tdf.index:
        ent = tdf.loc[tid]
        ent_path = entry_rows.loc[tid]
        R = float(ent["entry_price"]) - float(ent["sl_at_entry_price"])
        expected = (float(ent_path["close_mid"]) - float(ent["entry_price"])) / R
        assert abs(expected - float(ent_path["close_r"])) < 1e-12


# ---- Saved-CSV checks (run only when the real pipeline output exists) ----

_skip_csv = pytest.mark.skipif(
    not (TRADES_FILE.exists() and PATHS_FILE.exists()),
    reason=(
        "results/arc_kh24_v2/step1/{trades_all,trades_paths}.csv not present — "
        "run `python -m scripts.arc_kh24_v2.step1.run_step1` to populate."
    ),
)


@_skip_csv
def test_saved_mfe_monotone():
    df = pd.read_csv(PATHS_FILE).sort_values(["trade_id", "bar_offset"])
    diff = df.groupby("trade_id")["mfe_so_far_r"].diff().dropna()
    assert (diff >= -1e-12).all(), f"mfe diff min={diff.min()}"


@_skip_csv
def test_saved_mae_monotone():
    df = pd.read_csv(PATHS_FILE).sort_values(["trade_id", "bar_offset"])
    diff = df.groupby("trade_id")["mae_so_far_r"].diff().dropna()
    assert (diff <= 1e-12).all(), f"mae diff max={diff.max()}"


@_skip_csv
def test_saved_path_completeness():
    tdf = pd.read_csv(TRADES_FILE)[["trade_id", "bars_held"]]
    pdf = pd.read_csv(PATHS_FILE)[["trade_id", "bar_offset"]]
    max_off = pdf.groupby("trade_id")["bar_offset"].max().rename("max_offset")
    n_rows = pdf.groupby("trade_id")["bar_offset"].size().rename("n_rows")
    merged = tdf.merge(max_off, on="trade_id").merge(n_rows, on="trade_id")
    assert (merged["max_offset"] == merged["bars_held"]).all(), (
        "trades with max(bar_offset) != bars_held: "
        f"{merged[merged['max_offset'] != merged['bars_held']].head().to_dict(orient='records')}"
    )
    assert (merged["n_rows"] == merged["bars_held"] + 1).all(), (
        "trades with row count != bars_held + 1: "
        f"{merged[merged['n_rows'] != merged['bars_held'] + 1].head().to_dict(orient='records')}"
    )


@_skip_csv
def test_saved_trade_ids_match():
    t = set(pd.read_csv(TRADES_FILE)["trade_id"])
    p = set(pd.read_csv(PATHS_FILE)["trade_id"])
    assert t == p, f"trade_id mismatch: only_in_trades={t - p}, only_in_paths={p - t}"


@_skip_csv
def test_saved_pool_size_at_least_500():
    n = len(pd.read_csv(TRADES_FILE))
    assert n >= 500, f"Pool size {n} < 500"

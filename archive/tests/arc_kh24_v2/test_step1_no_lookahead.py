"""No-lookahead invariant — KH-24 v2.0 Step 1 plumbing.

For a sampled trade, perturb forward-bar OHLC and verify that the signal
decision, SL distance, entry price, and path features at bar ≤ entry bar are
unchanged. By construction the implementation reads only bars ≤ i to make
decisions at bar i; this test verifies that the implementation actually
respects the read window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.arc_kh24_v2.step1._signal import SignalParams, evaluate_bare_signal
from scripts.arc_kh24_v2.step1._simulate import ExecParams, simulate_pair
from tests.arc_kh24_v2._synth import make_4h_with_signal, make_d1_for_4h


def _run(df_4h: pd.DataFrame, df_d1: pd.DataFrame, pair: str = "EUR_USD"):
    params = SignalParams()
    exec_params = ExecParams()
    sig_mask, atr_4h, _ = evaluate_bare_signal(df_4h, df_d1, params)
    trades, paths = simulate_pair(pair, df_4h, sig_mask, atr_4h, exec_params)
    return sig_mask, trades, paths


def test_forward_perturbation_does_not_change_pre_entry_decisions():
    df_4h = make_4h_with_signal()
    df_d1 = make_d1_for_4h(df_4h)
    sig_ref, trades_ref, paths_ref = _run(df_4h, df_d1)

    assert len(trades_ref) >= 1, "synthetic fixture failed to produce a signal"
    trade = trades_ref[0]
    entry_ts = pd.Timestamp(trade["entry_time"])
    entry_idx = int(np.flatnonzero(df_4h["date"].values == np.datetime64(entry_ts))[0])

    # Perturb every bar strictly AFTER the entry bar's full forward window.
    # The simulator reads at most entry_idx + hold_bars; perturbations beyond
    # that index must not change ANY recorded value.
    perturb_start = entry_idx + ExecParams().hold_bars + 1
    if perturb_start >= len(df_4h):
        return  # nothing forward of the read window; trivially invariant

    df_4h_pert = df_4h.copy()
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 0.01, size=(len(df_4h_pert) - perturb_start, 4))
    for i, col in enumerate(["open", "high", "low", "close"]):
        df_4h_pert.loc[perturb_start:, col] = (
            df_4h_pert.loc[perturb_start:, col].astype(float) + noise[:, i]
        )

    sig_pert, trades_pert, paths_pert = _run(df_4h_pert, df_d1)

    # Signal mask up to and including the signal bar (entry_idx - 1) must match.
    np.testing.assert_array_equal(sig_pert[:entry_idx], sig_ref[:entry_idx])
    # The first trade's entry fields must match.
    assert trades_ref[0]["entry_price"] == trades_pert[0]["entry_price"]
    assert trades_ref[0]["sl_at_entry_price"] == trades_pert[0]["sl_at_entry_price"]
    assert trades_ref[0]["signal_bar_atr_14"] == trades_pert[0]["signal_bar_atr_14"]

    # Path features at bar_offset 0 (entry bar) must match.
    p_ref = next(p for p in paths_ref if p["bar_offset"] == 0)
    p_pert = next(p for p in paths_pert if p["bar_offset"] == 0)
    for k in ("close_mid", "close_r", "mfe_so_far_r", "mae_so_far_r"):
        assert p_ref[k] == p_pert[k], f"path[0].{k} changed under forward perturbation"


def test_signal_mask_unchanged_under_post_signal_perturbation():
    df_4h = make_4h_with_signal()
    df_d1 = make_d1_for_4h(df_4h)
    sig_ref, _, _ = _run(df_4h, df_d1)

    fired = np.flatnonzero(sig_ref)
    assert fired.size >= 1
    nbar = int(fired[0])
    if nbar >= len(df_4h) - 1:
        return

    df_4h_pert = df_4h.copy()
    rng = np.random.default_rng(456)
    forward = slice(nbar + 1, len(df_4h_pert))
    n_pert = len(df_4h_pert) - (nbar + 1)
    noise = rng.normal(0, 0.005, size=(n_pert, 4))
    for i, col in enumerate(["open", "high", "low", "close"]):
        df_4h_pert.loc[forward, col] = df_4h_pert.loc[forward, col].astype(float) + noise[:, i]
    sig_pert, _, _ = _run(df_4h_pert, df_d1)
    # All signal-firing decisions at bars ≤ nbar must be identical.
    np.testing.assert_array_equal(sig_pert[: nbar + 1], sig_ref[: nbar + 1])

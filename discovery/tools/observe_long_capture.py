"""Per-bar honest LONG capture / forward-drift OBSERVATION harness — EXPERIMENT tool (BUILT).

The step-(b) observation loop that arcs 1000–1006 (and chat 3001's "drift lens") each
re-rolled: for a hypothetical long at every bar, compute the honest +1R-before-SL
CAPTURE (in-tree take-the-loss label) and the forward N-bar DRIFT in ATR. Returns a
tidy per-bar DataFrame the arc then joins its own conditioning columns onto and groups
by — the fast PRE-pool screen that finds whether a conditioning axis separates.

CHARACTERIZATION / OBSERVATION ONLY — this is NOT a gate and NOT cost-aware:
  - It computes a descriptive label (does +1R come before the 2·ATR stop, take-the-loss
    ordering via `reached_1r_before_sl`) and a gross forward drift. NO spread/commission/
    slippage, NO daily-DD, NO portfolio. The ONLY gate is `MultiPairBacktester` via
    `ArcFoldRunner` (FundedNext costs, SL-first) — never trust a number from here as a
    verdict. Use it to screen conditionings; validate on the engine.
  - It matches `build_arc_pool`'s entry/SL convention (entry = next-bar `open_ask`; SL =
    signal-bar `close_ask` − sl_mult·ATR; ATR = canonical `wilder_atr(mid,14).shift(1)`),
    so the unconditional capture here predicts the pool's capture (sanity-checkable).

Usage:
    from discovery.tools.observe_long_capture import observe_long_capture
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24)
    obs["cond"] = ...                      # arc-specific conditioning, joined by (pair, signal_time)
    obs.groupby("cond")["capture"].mean()  # screen the axis (base ~0.49 on H4 majors)

Created by: post-arc-1007 tooling refactor (chat 1000-1999); generalizes the arc-1000..1006 +
chat-3001 observation loop. Verified faithful: unconditional H4-majors capture 0.4877 (EURUSD 0.4938,
GBPUSD 0.4817) reproduces the arc-1000 observation exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr
from core.sim.honest_label import reached_1r_before_sl


def observe_long_capture(
    panel,
    *,
    sl_mult: float = 2.0,
    hold: int = 120,
    drift_bars: int | None = None,
    warmup: int = 100,
    restrict: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per-bar hypothetical-long honest capture + forward drift across every pair.

    Parameters
    ----------
    panel : Panel — the primary-TF panel (uses each pair's bars).
    sl_mult : SL = signal-bar close_ask − sl_mult·ATR (default 2.0, matching the pool).
    hold : forward bars scanned for the +1R-before-SL label.
    drift_bars : if set, also compute forward `drift_bars`-bar mid-close drift in ATR
        (gross). If None, ``fwd_drift_atr`` is NaN.
    warmup : skip the first ``warmup`` bars per pair (ATR/feature stabilization).
    restrict : optional {pair -> bool ndarray aligned to that pair's bars} to compute
        only on selected bars (efficient for sparse conditions, e.g. week-opens). None =
        every bar.

    Returns
    -------
    DataFrame[pair, signal_time, capture (0/1), fwd_drift_atr, atr] — one row per
    evaluated bar with a finite ATR and a fillable next bar. Take-the-loss honest;
    gross; CHARACTERIZATION ONLY (see module docstring).
    """
    rows = []
    for pair in sorted(panel.pairs):
        df = panel.pair_dfs[pair]
        atr = wilder_atr(mid_high(df), mid_low(df), mid_close(df), 14).shift(1).values
        mc = mid_close(df).values
        open_ask = df["open_ask"].values
        close_ask = df["close_ask"].values
        high_bid = df["high_bid"].values
        low_bid = df["low_bid"].values
        idx = df.index
        n = len(df)
        sel = restrict.get(pair) if restrict is not None else None
        for t in range(max(warmup, 0), n - 1):
            if sel is not None and not sel[t]:
                continue
            a = atr[t]
            if not np.isfinite(a) or a <= 0:
                continue
            entry = float(open_ask[t + 1])
            sl = float(close_ask[t]) - sl_mult * a
            sl_dist = entry - sl
            if sl_dist <= 0 or not np.isfinite(sl_dist):
                continue
            exit_off = min(hold, n - 1 - (t + 1))
            res = reached_1r_before_sl(
                high_bid=high_bid, low_bid=low_bid, entry_idx=t + 1, exit_off=exit_off,
                entry_price=entry, sl_price=sl, sl_distance=sl_dist, exit_at_bar_open=False,
            )
            drift = float("nan")
            if drift_bars is not None:
                drift = (float(mc[min(t + drift_bars, n - 1)]) - entry) / a
            rows.append({
                "pair": pair, "signal_time": idx[t],
                "capture": 1 if np.isfinite(res) else 0,
                "fwd_drift_atr": drift, "atr": float(a),
            })
    return pd.DataFrame(rows, columns=["pair", "signal_time", "capture", "fwd_drift_atr", "atr"])


__all__ = ("observe_long_capture",)

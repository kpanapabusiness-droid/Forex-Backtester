"""Cross-rate triangulation residual + driver-shock conditioning — EXPERIMENT tool (BUILT).

Arc 3005 killed the cross-rate triangulation residual **unconditionally** at H4 (residual
median ≈ 0, |resid| > spread on only 1.3–6.1% of bars, forward-convergence corr ≈ 0.01).
This tool reproduces that residual and adds the angle arc 3005 never tested (the strategist
MENU item M1, `DISCOVERY_DIRECTION.md`): condition the residual on a large **D1 driver-leg
shock** and ask whether the dependent quoted cross re-prices with a lag at the next H4 bar.

Two pieces, both CHARACTERIZATION / OBSERVATION ONLY (never realize P&L; the gate is
`MultiPairBacktester` via `ArcFoldRunner`):

  - `triangle_log_residual_bp(cross_df, legA_df, legB_df, op)` — the triangular identity
    residual in basis points on the common H4 index. `op="mul"`: cross = legA·legB
    (e.g. EURJPY = EURUSD·USDJPY). `op="div"`: cross = legA/legB (e.g. EURGBP = EURUSD/GBPUSD).
    residual = 1e4·(log(cross_quoted_mid) − log(cross_synthetic_mid)). Pinned ≈ 0 by
    triangular arbitrage; its transient non-zero excursions are what M1 probes.

  - `d1_driver_shock(d1_df, atr_period)` — the ATR-normalized D1 close-to-close move
    z = (close_mid[d] − close_mid[d−1]) / Wilder-ATR(14).shift(1), the "driver shock"
    magnitude (|z| > 1.5 = a large directional shock). Indexed by the D1 bar label.

Alignment helper `shock_available_at_next_day(z_d1)` shifts the D1 shock series so it is
keyed to the **first H4 bar after the D1 close** (D1 and H4 share the 22:00-UTC EET
boundary, so the next-day D1 label IS that first H4 bar's label) — the no-lookahead point
at which a driver shock is fully known and the cross could lag.

Created by: arc 1027 (chat 1000-1999), M1 driver-shock-conditional triangulation residual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr


def _mid_close_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(mid_close(df).values, index=df.index)


def triangle_log_residual_bp(
    cross_df: pd.DataFrame,
    legA_df: pd.DataFrame,
    legB_df: pd.DataFrame,
    *,
    op: str = "mul",
) -> pd.Series:
    """Triangular-identity log residual (bp) on the common index of the three legs.

    cross = legA `op` legB. residual = 1e4·(log(cross_mid) − log(synth_mid)), where
    synth_mid = legA_mid·legB_mid (op="mul") or legA_mid/legB_mid (op="div").
    All MID closes (real bid/ask midpoint), contemporaneous bars. ≈ 0 under arb.
    """
    op = str(op).strip().lower()
    if op not in ("mul", "div"):
        raise ValueError(f"op must be 'mul' or 'div'; got {op!r}")
    c = _mid_close_series(cross_df)
    a = _mid_close_series(legA_df)
    b = _mid_close_series(legB_df)
    idx = c.index.intersection(a.index).intersection(b.index)
    c, a, b = c.reindex(idx), a.reindex(idx), b.reindex(idx)
    log_synth = np.log(a) + (np.log(b) if op == "mul" else -np.log(b))
    resid_bp = 1e4 * (np.log(c) - log_synth)
    return resid_bp.dropna()


def d1_driver_shock(d1_df: pd.DataFrame, *, atr_period: int = 14) -> pd.Series:
    """ATR-normalized D1 close-to-close move z (indexed by D1 bar label).

    z[d] = (close_mid[d] − close_mid[d−1]) / Wilder-ATR(atr_period).shift(1)[d].
    The ATR is shifted one bar (causal: known at the prior close). |z| > 1.5 = a large
    directional shock. Sign carries the shock direction.
    """
    atr = wilder_atr(mid_high(d1_df), mid_low(d1_df), mid_close(d1_df), atr_period).shift(1)
    mc = mid_close(d1_df)
    move = mc.diff()
    z = (move / atr)
    return pd.Series(z.values, index=d1_df.index).dropna()


def shock_available_at_next_day(z_d1: pd.Series) -> pd.Series:
    """Re-key the D1 shock series to the first H4 bar after the D1 close.

    A D1 bar labeled T covers [T, T+1day) and closes at T+1day; its shock is first
    actionable at the H4 bar labeled T+1day (the next D1 label, since D1 ⊂ H4 on the
    shared 22:00-UTC boundary). Returns z indexed by that next-day label.
    """
    if len(z_d1) < 2:
        return pd.Series(dtype=float)
    return pd.Series(z_d1.values[:-1], index=z_d1.index[1:])


__all__ = (
    "triangle_log_residual_bp",
    "d1_driver_shock",
    "shock_available_at_next_day",
)

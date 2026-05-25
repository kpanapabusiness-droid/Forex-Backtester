"""SignalModule implementation for the L4 mtf_alignment 2_down_mixed-kijun
trigger (LCHAR registry entry 5 with chat-locked h=120 horizon override).

Fires at the close of an H1 bar N when:

    kijun_sign_1H(N)    == -1   (1H mid-close below 1H Kijun-26)
    kijun_sign_4H_mr(N) == +1   (most-recent-completed 4H Kijun-26 above)
    kijun_sign_D1_mr(N) == -1   (most-recent-completed D1 Kijun-26 below)

"Most-recent-completed" means: at H1 bar N's close time T, the matched
higher-TF bar's left-edge timestamp must be strictly before T. Enforced
via ``core.signals.htf_alignment.get_htf_index_at`` with
``require_fully_closed=True`` — timezone-invariant under both UTC and
5ers EET storage conventions. The attic Arc 2 idiom of
``index.floor(freq) → map → idx - 1`` was Arc 5 v3.0.1's
ZERO-TRADE-POOL BUG under EET: ``.floor("4h")`` returns UTC-anchored
4h boundaries that don't match EET-anchored H4 bar labels under
``boundary_convention="5ers_eet"`` → ``.map()`` returns NaN for every
H1 bar → empty signal pool. This module is the canonical fix and is
restored to main alongside the canonical utility (per PR description).

Mid-OHLC for all kijun computations (mid = (bid + ask) / 2). ATR(14) on
H1 mid-OHLC. Lookahead invariant runtime-asserted on every firing bar
(raises ``RuntimeError`` if any firing bar's matched H4/D1 timestamp is
not strictly prior).

Conforms to :class:`core.arc.signal_protocol.SignalModule`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from core.arc.signal_protocol import (
    PerPairSignalState,
    SignalEvaluation,
)
from core.signals.htf_alignment import get_htf_index_at
from core.sim.panel import Panel

KIJUN_PERIOD: int = 26
ATR_PERIOD: int = 14


def _mid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": (df["open_bid"] + df["open_ask"]) / 2.0,
            "high": (df["high_bid"] + df["high_ask"]) / 2.0,
            "low": (df["low_bid"] + df["low_ask"]) / 2.0,
            "close": (df["close_bid"] + df["close_ask"]) / 2.0,
        },
        index=df.index,
    )


def _kijun_sign(mid: pd.DataFrame, period: int = KIJUN_PERIOD) -> pd.Series:
    hh = mid["high"].rolling(period, min_periods=period).max()
    ll = mid["low"].rolling(period, min_periods=period).min()
    kijun = (hh + ll) / 2.0
    return np.sign(mid["close"].astype(float) - kijun).astype(float)


def _wilder_atr(mid: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = mid["high"].to_numpy(dtype=float)
    low = mid["low"].to_numpy(dtype=float)
    close = mid["close"].to_numpy(dtype=float)
    n = len(mid)
    if n == 0:
        return pd.Series(dtype=float, index=mid.index)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return pd.Series(atr, index=mid.index)
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(atr, index=mid.index)


def _compute_pair_state(
    pair: str,
    h1_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    d1_df: pd.DataFrame,
) -> PerPairSignalState:
    """Compute signal_mask + atr for one pair, with lookahead invariant check."""
    mid_1h = _mid_ohlc(h1_df)
    mid_4h = _mid_ohlc(h4_df)
    mid_d1 = _mid_ohlc(d1_df)

    s1_full = _kijun_sign(mid_1h).to_numpy()
    s4_full = _kijun_sign(mid_4h).to_numpy()
    sd_full = _kijun_sign(mid_d1).to_numpy()

    # Most-recently-completed H4 / D1 index for each H1 bar — timezone-invariant.
    # Replaces the State-C `.floor("4h").map(...)` / `.normalize().map(...)` idiom
    # that hard-failed under 5ers EET storage (see module docstring).
    mr4 = get_htf_index_at(h1_df.index, h4_df, require_fully_closed=True, invalid_sentinel=-1)
    mrd = get_htf_index_at(h1_df.index, d1_df, require_fully_closed=True, invalid_sentinel=-1)
    val = (mr4 >= 0) & (mrd >= 0)

    n = len(h1_df)
    mask = np.zeros(n, dtype=bool)

    pos = np.where(val)[0]
    if pos.size > 0:
        s1 = s1_full[pos]
        s4 = s4_full[mr4[pos]]
        sd = sd_full[mrd[pos]]
        valid_signs = ~np.isnan(s1) & ~np.isnan(s4) & ~np.isnan(sd)
        pos_ok = pos[valid_signs]
        s1_ok = s1[valid_signs]
        s4_ok = s4[valid_signs]
        sd_ok = sd[valid_signs]

        # 2_down_mixed via decision-tree priority — matches attic invariant 7.
        has_zero = (s1_ok == 0) | (s4_ok == 0) | (sd_ok == 0)
        rest = ~has_zero
        all_up = rest & (s1_ok == 1) & (s4_ok == 1) & (sd_ok == 1)
        all_down = rest & (s1_ok == -1) & (s4_ok == -1) & (sd_ok == -1)
        rest = rest & ~all_up & ~all_down
        opposed = rest & (s1_ok != sd_ok)
        rest = rest & ~opposed
        up_mixed = rest & (s1_ok == 1) & (sd_ok == 1) & (s4_ok == -1)
        rest = rest & ~up_mixed
        down_mixed = rest & (s1_ok == -1) & (sd_ok == -1) & (s4_ok == 1)

        mask[pos_ok[down_mixed]] = True

        # Lookahead invariant — strict < on H4 + D1.
        # require_fully_closed=True guarantees mr4[i]+1 has started ≤ h1_ts[i],
        # so h4_df.index[mr4[i]] (one bar earlier than the now-active H4) is
        # strictly < h1_ts[i]. The assertion confirms this at runtime.
        sig_positions = np.where(mask)[0]
        if sig_positions.size > 0:
            ts_h1 = h1_df.index.to_numpy()
            ts_4h = h4_df.index.to_numpy()
            ts_d1 = d1_df.index.to_numpy()
            bad_4h = ts_4h[mr4[sig_positions]] >= ts_h1[sig_positions]
            bad_d1 = ts_d1[mrd[sig_positions]] >= ts_h1[sig_positions]
            if bad_4h.any():
                i = int(sig_positions[np.argmax(bad_4h)])
                raise RuntimeError(
                    f"4H lookahead at {pair} bar {i}: "
                    f"ts_4h_used={ts_4h[mr4[i]]} >= ts_h1={ts_h1[i]}"
                )
            if bad_d1.any():
                i = int(sig_positions[np.argmax(bad_d1)])
                raise RuntimeError(
                    f"D1 lookahead at {pair} bar {i}: "
                    f"ts_d1_used={ts_d1[mrd[i]]} >= ts_h1={ts_h1[i]}"
                )

    signal_mask = pd.Series(mask, index=h1_df.index, name="signal_mask")
    atr_series = _wilder_atr(mid_1h).rename("atr")
    return PerPairSignalState(
        signal_mask=signal_mask,
        atr=atr_series,
        additional_gates={},
        exit_predicate=None,
        path_feature_anchor=h1_df.index,
    )


@dataclass(frozen=True)
class MtfAlignment2DownMixedKijunSignal:
    """SignalModule for the mtf_alignment 2_down_mixed-kijun trigger."""

    signal_name: str = "mtf_alignment_2_down_mixed_kijun_h120"
    primary_tf: str = "H1"
    auxiliary_tfs: tuple[str, ...] = ("H4", "D1")
    causal_lineage: str = "clean"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        h4 = panels["H4"]
        d1 = panels["D1"]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            per_pair[pair] = _compute_pair_state(
                pair=pair,
                h1_df=primary.pair_dfs[pair],
                h4_df=h4.pair_dfs[pair],
                d1_df=d1.pair_dfs[pair],
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self.signal_name,
            causal_lineage=self.causal_lineage,
        )


__all__ = ("MtfAlignment2DownMixedKijunSignal", "KIJUN_PERIOD", "ATR_PERIOD")
